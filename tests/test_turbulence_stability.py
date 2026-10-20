"""Turbulence circuit-breaker stability (Fix #4).

The live bug class: tau_99 was a percentile over only ~55 freshly computed points, so it
interpolated the top two values and one spike could swing the active threshold between
~15 and ~146 within hours, and the tier flipped on a single bar. These tests pin the
persisted-history pool, the bootstrap floor, and the engage/disengage hysteresis.
"""
import os
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch

import pandas as pd

from regime_classifier import RegimeClassifier

ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT", "NEARUSDT"]
STEP_MS = 15 * 60 * 1000
N_BARS = 105


def _frame(pattern, n=N_BARS):
    now_ms = int(time.time() * 1000)
    # Last bar opens at now - step => fully closed, survives _normalize_bars.
    opens = [now_ms - (n - i) * STEP_MS for i in range(n)]
    closes = [p * (1.0 + 0.0001 * (i % 3)) for i, p in enumerate(pattern)]
    return pd.DataFrame({
        "open_time": opens,
        "open": closes,
        "high": [c * 1.0002 for c in closes],
        "low": [c * 0.9998 for c in closes],
        "close": closes,
        "volume": [1000.0] * n,
    })


def _calm_pattern():
    patterns = {}
    for k, asset in enumerate(ASSETS):
        base = [100.0 * (1 + k)]
        for i in range(1, N_BARS):
            wig = 0.001 if (i % 2) else -0.001
            base.append(base[-1] * (1.0 + wig))
        patterns[asset] = base
    return patterns


def _surge_pattern():
    patterns = _calm_pattern()
    # Shock: a 15% gap in the final bar of every asset => d_t explodes vs calm pool.
    for asset in patterns:
        patterns[asset][-1] *= 1.15
    return patterns


class _DummyResp:
    status_code = 200

    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


def _patched_get(patterns):
    def fake_get(url, timeout=None, **kw):
        sym = next((a for a in ASSETS if f"symbol={a}" in url), None)
        frame = _frame(patterns[sym])
        # Bitget returns newest-first rows with a quote_volume column; ordering is repaired
        # by _normalize_bars, so emit oldest-last to mirror the live shape.
        rows = [[str(r["open_time"]), f"{r['open']:.8f}", f"{r['high']:.8f}",
                 f"{r['low']:.8f}", f"{r['close']:.8f}", str(r["volume"]), str(r["volume"] * 100)]
                for r in frame.iloc[::-1].to_dict("records")]
        return _DummyResp({"code": "00000", "data": rows})
    return fake_get


class TestTurbulenceStability(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "turb.db")
        with sqlite3.connect(self.db) as conn:
            conn.execute("PRAGMA journal_mode=DELETE;")
            conn.execute("CREATE TABLE system_state (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
        self.calm = _calm_pattern()
        self.surge = _surge_pattern()

    def tearDown(self):
        for suffix in ("-wal", "-shm", "-journal"):
            try:
                os.remove(self.db + suffix)
            except OSError:
                pass
        try:
            self.tmp.cleanup()
        except PermissionError:
            pass

    def _compute(self, patterns, clf=None):
        clf = clf or RegimeClassifier(db_path=self.db)
        with patch("regime_classifier.requests.get", side_effect=_patched_get(patterns)):
            return clf, clf.calculate_mahalanobis_turbulence()

    def test_tier_engages_after_two_consecutive_breaches(self):
        clf = RegimeClassifier(db_path=self.db)
        states = [clf._advance_turbulence_tier(True) for _ in (1, 2)]
        self.assertEqual(states, ["CALM_NORMAL", "EXTREME_SHOCK"])
        self.assertEqual(clf._turb_breach_streak, 0)

    def test_tier_disengages_after_three_consecutive_calms(self):
        clf = RegimeClassifier(db_path=self.db)
        seq_breach = [clf._advance_turbulence_tier(True) for _ in (1, 2)]
        self.assertEqual(seq_breach, ["CALM_NORMAL", "EXTREME_SHOCK"])
        # Disengagement counts genuine calm passes only: the breaker holds through
        # three of them and releases on the third.
        seq = [clf._advance_turbulence_tier(False) for _ in (1, 2, 3, 4)]
        self.assertEqual(seq, ["EXTREME_SHOCK", "EXTREME_SHOCK", "CALM_NORMAL", "CALM_NORMAL"])

    def test_current_breach_flag_respects_engaged_tier(self):
        # Once EXTREME_SHOCK is engaged, breaching bars stay flagged until 3 calm passes clear it.
        clf = RegimeClassifier(db_path=self.db)
        for _ in (1, 2):
            clf._advance_turbulence_tier(True)
        self.assertEqual(clf.turbulence_tier, "EXTREME_SHOCK")
        flag = clf._advance_turbulence_tier(False)
        self.assertEqual(flag, "EXTREME_SHOCK")

    def test_shock_state_survives_new_instance(self):
        clf = RegimeClassifier(db_path=self.db)
        clf._advance_turbulence_tier(True)
        clf._advance_turbulence_tier(True)
        clf._persist_turbulence_tier()
        restored = RegimeClassifier(db_path=self.db)
        self.assertEqual(restored.turbulence_tier, "EXTREME_SHOCK")
        self.assertEqual(restored._turb_breach_streak, 0)

    def test_current_value_persisted_to_history_table(self):
        with sqlite3.connect(self.db) as conn:
            conn.execute("INSERT INTO system_state (key, value, updated_at) VALUES ('turbulence_tier_state', '{\"tier\": \"CALM_NORMAL\", \"breach_streak\": 0, \"calm_streak\": 0}', 'x')")
        clf, res = self._compute(self.calm)
        with sqlite3.connect(self.db) as conn:
            rows = conn.execute("SELECT ts, value FROM turbulence_history ORDER BY ts").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0][1], res["turbulence_index"], places=3)

    def test_bootstrap_below_100_persisted_points_uses_seeded_pool(self):
        # One spike is the max of the whole pool; with an untrusted (<100) persisted pool the
        # active threshold interpolates the seeded quantiles, so the breaker stays armed and
        # logs the bootstrap notice instead of trusting the top-two percentile.
        with self.assertLogs("regime_classifier", level="INFO") as captured:
            clf, res = self._compute(self.surge)
        self.assertTrue(any("[TURB-BOOTSTRAP]" in line for line in captured.output))
        self.assertTrue(res["bootstrap_active"])
        self.assertGreater(res["turbulence_index"], 1.0)
        self.assertFalse(res["turbulence_breach"])
        # Spike is above tau_90 but not a trusted-pool breach: elevated, not shock.
        self.assertEqual(res["turbulence_state"], "ELEVATED_STRESS")

    def test_persisted_pool_overrides_top_two_interpolation(self):
        # Seed 130 calm-range points plus one earlier breach pass (hysteresis needs two
        # consecutive breaches): the current spike is now inside a trusted distribution
        # and must engage EXTREME_SHOCK instead of defining its own percentile ceiling.
        now = time.time()
        with sqlite3.connect(self.db) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS turbulence_history (ts REAL NOT NULL, value REAL NOT NULL)")
            conn.executemany(
                "INSERT INTO turbulence_history (ts, value) VALUES (?, ?)",
                [(now - i * 300.0, 3.0 + (i % 5) * 0.1) for i in range(130)],
            )
        clf = RegimeClassifier(db_path=self.db)
        clf._turb_breach_streak = 1  # one breach already banked this hour
        with patch("regime_classifier.requests.get", side_effect=_patched_get(self.surge)):
            res = clf.calculate_mahalanobis_turbulence()
        self.assertGreaterEqual(res["history_pool_size"], 100)
        self.assertFalse(res["bootstrap_active"])
        self.assertTrue(res["turbulence_breach"])
        self.assertEqual(res["turbulence_state"], "EXTREME_SHOCK")

    def test_history_pruned_to_30_days(self):
        now = time.time()
        with sqlite3.connect(self.db) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS turbulence_history (ts REAL NOT NULL, value REAL NOT NULL)")
            conn.executemany(
                "INSERT INTO turbulence_history (ts, value) VALUES (?, ?)",
                [(now - 40 * 86400.0 + i, 2.0) for i in range(50)] + [(now - i * 60.0, 3.0) for i in range(120)],
            )
        self._compute(self.calm)
        with sqlite3.connect(self.db) as conn:
            old = conn.execute("SELECT COUNT(*) FROM turbulence_history WHERE ts < ?", (now - 30 * 86400.0,)).fetchone()[0]
            total = conn.execute("SELECT COUNT(*) FROM turbulence_history").fetchone()[0]
        self.assertEqual(old, 0)
        self.assertEqual(total, 121)


if __name__ == "__main__":
    unittest.main()

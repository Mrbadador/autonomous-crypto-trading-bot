"""
tests/test_regime_history_append.py

TASK 1 regression tests for the market_regime producer/consumer contract.

market_regime used to be a singleton (`INSERT OR REPLACE ... VALUES (1, ...)`) and its single
`timestamp` column held the publication time, so there was no history of regime transitions at
all and no way to tell "the classifier is current but analysed stale bars" from "the classifier
stopped writing". It is now append-only, with `source_close_ts` (close time of the newest 15m bar
used) recorded next to `timestamp` (publication time) and the pre-arbitration `math_regime`.

Invariant under test: every existing reader that wants "the current regime" — super_bot's
`ORDER BY id DESC LIMIT 1`, regime_monitor's `ORDER BY rowid DESC LIMIT 1`, and the
`WHERE id = 1` sites that had to be migrated — resolves to the newest row, and the additive
ALTER-TABLE path works against a snapshot of the real production schema.

Safety: temp DB only, synthetic bars, every network call stubbed (requests is neutered so an
accidental live fetch fails the test instead of hitting the exchange).
"""

import os
import re
import shutil
import sqlite3
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, UTC, timezone

import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

import regime_classifier  # noqa: E402
from regime_classifier import RegimeClassifier  # noqa: E402

# market_regime DDL exactly as captured from trading_data.db (mode=ro) on 2026-10-14:
# singleton-shaped, no source_close_ts, no math_regime.
PRODUCTION_DDL = """CREATE TABLE market_regime (
    id INTEGER PRIMARY KEY DEFAULT 1,
    timestamp TEXT,
    regime TEXT,
    hurst REAL,
    adx REAL
, plus_di REAL, minus_di REAL, macro_trend TEXT, ai_bias TEXT, trap_risk REAL,
   ai_thesis TEXT, turbulence_index REAL, turbulence_threshold REAL, turbulence_state TEXT)"""

SYSTEM_STATE_DDL = """CREATE TABLE system_state (
    key TEXT PRIMARY KEY, value TEXT, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)"""

SEED_TS = "2026-01-01T00:00:00+00:00"


def bars(n=120, step_ms=900_000, slope=0.01, end_near_now=True):
    """Rising bars with deterministic wicks. open_time is Bitget-style epoch ms.

    By default the newest bar closes about now, so publication time and source bar time are
    comparable; pass end_near_now=False to simulate the classifier analysing stale history.
    """
    if end_near_now:
        last_open = datetime.now(timezone.utc) - timedelta(milliseconds=step_ms)
        start_ms = int(last_open.timestamp() * 1000) - (n - 1) * step_ms
    else:
        start_ms = 1_760_000_000_000
    rows = []
    price = 100_000.0
    for i in range(n):
        step = slope * (1 + (i % 5))
        o = price
        c = price + step
        h = max(o, c) + abs(slope) * 2
        low = min(o, c) - abs(slope) * 2
        price = c
        rows.append({"open_time": start_ms + i * step_ms, "open": o, "high": h, "low": low,
                     "close": c, "volume": 10.0, "quote_volume": 100.0})
    return pd.DataFrame(rows)


class RegimeHistoryAppendTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="regime_history_")
        self.db_path = os.path.join(self.tmpdir, "trading_data.db")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(PRODUCTION_DDL)
            conn.execute(SYSTEM_STATE_DDL)
            # A pre-existing singleton row: append-only must not overwrite or reuse it.
            conn.execute(
                "INSERT INTO market_regime (id, timestamp, regime, hurst, adx, plus_di, minus_di) "
                "VALUES (1, ?, 'SEEDED_OLD_SINGLETON', 0.5, 10.0, 1.0, 1.0)",
                (SEED_TS,),
            )
            conn.commit()

        # Fail loudly rather than silently reaching the exchange.
        self._real_requests = regime_classifier.requests
        boom = types.ModuleType("requests")

        def _no_network(*a, **k):
            raise AssertionError("classify_and_save must not perform network I/O in tests")
        boom.get = _no_network
        boom.post = _no_network
        regime_classifier.requests = boom

        self.rc = RegimeClassifier(db_path=self.db_path)
        self._df15 = bars()
        self.rc.fetch_ohlcv = lambda symbol="BTCUSDT", interval="15m", limit=105: (
            self._df15 if interval == "15m" else bars(n=60, step_ms=14_400_000)
        )
        self.rc.fetch_funding_rate = lambda symbol: 0.0001
        self.rc.calculate_mahalanobis_turbulence = lambda: {
            "turbulence_index": 3.5, "active_threshold": 12.0,
            "turbulence_state": "CALM_NORMAL", "turbulence_breach": False,
        }
        self.rc._query_ai_cognitive_arbiter = lambda telemetry: None

    def tearDown(self):
        regime_classifier.requests = self._real_requests
        regime_classifier.REGIME_HISTORY_ROWS = 30000
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _rows(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute("SELECT * FROM market_regime ORDER BY id")]

    # ------------------------------------------------------------------ append-only core

    def test_each_classification_appends_a_new_row(self):
        self.assertTrue(self.rc.classify_and_save())
        self._df15 = bars(slope=-0.35)  # opposite direction -> the new row must reflect it
        self.assertTrue(self.rc.classify_and_save())

        rows = self._rows()
        self.assertEqual(len(rows), 3, "two appends on top of the seeded singleton")
        self.assertEqual(rows[0]["regime"], "SEEDED_OLD_SINGLETON", "the pre-existing row must not be replaced")
        self.assertEqual([r["id"] for r in rows], sorted(r["id"] for r in rows))
        self.assertGreater(rows[2]["id"], rows[1]["id"])
        self.assertGreater(rows[1]["plus_di"], rows[1]["minus_di"], "run 1 was an uptrend")
        self.assertLess(rows[2]["plus_di"], rows[2]["minus_di"], "run 2 was a downtrend; the row is not a copy")

    def test_current_gate_readers_see_the_newest_row(self):
        self.rc.classify_and_save()
        self._df15 = bars(slope=0.9)
        self.rc.classify_and_save()

        with sqlite3.connect(self.db_path) as conn:
            super_bot_row = conn.execute(
                "SELECT regime, adx, hurst, plus_di, minus_di, turbulence_index, turbulence_threshold, turbulence_state "
                "FROM market_regime ORDER BY id DESC LIMIT 1"
            ).fetchone()
            monitor_row = conn.execute(
                "SELECT hurst, adx, regime, timestamp, plus_di, minus_di, macro_trend "
                "FROM market_regime ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            brief_row = conn.execute(
                "SELECT adx, hurst, regime FROM market_regime ORDER BY id DESC LIMIT 1"
            ).fetchone()

        self.assertNotEqual(super_bot_row[0], "SEEDED_OLD_SINGLETON")
        self.assertAlmostEqual(super_bot_row[1], monitor_row[1])
        self.assertAlmostEqual(brief_row[0], super_bot_row[1])
        self.assertIn(monitor_row[6], ("BULL", "MILD_BULL", "NEUTRAL", "MILD_BEAR", "BEAR"))

    def test_stale_singleton_reader_pattern_is_gone(self):
        """`WHERE id = 1` silently freezes on the oldest row once the table appends.

        A grep guard, because the modules that used to contain it are heavy to import.
        """
        offenders = []
        for name in sorted(os.listdir(REPO_DIR)):
            if not name.endswith(".py") or name.startswith("test_"):
                continue
            path = os.path.join(REPO_DIR, name)
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            if re.search(r"FROM\s+market_regime\s+WHERE\s+id\s*=\s*1", text, re.I):
                offenders.append(name)
        self.assertEqual(offenders, [], f"stale singleton readers found: {offenders}")

    # ------------------------------------------------------------------ timestamp split

    def test_source_bar_close_time_is_separate_from_publication_time(self):
        last_open_ms = int(self._df15["open_time"].iloc[-1])
        expected_close = datetime.fromtimestamp((last_open_ms + 900_000) / 1000.0, timezone.utc)
        self.assertTrue(self.rc.classify_and_save())

        row = self._rows()[-1]
        self.assertEqual(row["source_close_ts"], expected_close.isoformat())
        published = datetime.fromisoformat(row["timestamp"])
        lag = published - expected_close
        self.assertGreaterEqual(lag, timedelta(0), "publication cannot precede the bar it describes")
        self.assertLess(lag, timedelta(minutes=5))
        self.assertNotEqual(row["timestamp"], row["source_close_ts"])

    def test_math_regime_is_persisted_alongside_the_arbitrated_regime(self):
        self.assertTrue(self.rc.classify_and_save())
        row = self._rows()[-1]
        self.assertIn(row["math_regime"], ("CHOP", "NEUTRAL", "BULL_TREND", "BEAR_TREND"))
        self.assertIn(row["regime"], ("CHOP", "NEUTRAL", "BULL_TREND", "BEAR_TREND", "DEFENSIVE_CHOP"))

    def test_unparsable_interval_records_null_instead_of_a_guess(self):
        self.assertIsNone(RegimeClassifier._bar_close_ts(bars(n=3), "15minutes"))
        self.assertIsNotNone(RegimeClassifier._bar_close_ts(bars(n=3), "15m"))
        self.assertIsNotNone(RegimeClassifier._bar_close_ts(bars(n=3), "4h"))

    # ------------------------------------------------------------------ retention

    def test_history_is_capped_but_always_keeps_the_newest_row(self):
        with sqlite3.connect(self.db_path) as conn:
            for i in range(2, 12):
                conn.execute(
                    "INSERT INTO market_regime (id, timestamp, regime, adx) VALUES (?, ?, ?, ?)",
                    (i, f"2026-01-0{i}T00:00:00+00:00", "CHOP", 10.0 + i),
                )
            conn.commit()

        regime_classifier.REGIME_HISTORY_ROWS = 3
        self.assertTrue(self.rc.classify_and_save())

        rows = self._rows()
        self.assertLessEqual(len(rows), 4, "cap plus the row just written")
        self.assertNotEqual(rows[-1]["regime"], "SEEDED_OLD_SINGLETON", "the live gate row must survive pruning")
        self.assertEqual(rows[-1]["id"], max(r["id"] for r in rows))

    def test_schema_migration_is_additive_only(self):
        """Existing columns and the seeded row are untouched; only new nullable columns appear."""
        self.rc.classify_and_save()
        with sqlite3.connect(self.db_path) as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(market_regime)")]
            seed = conn.execute("SELECT timestamp, regime, hurst, adx FROM market_regime WHERE id = 1").fetchone()
        for required in ("timestamp", "regime", "hurst", "adx", "plus_di", "minus_di", "macro_trend",
                         "ai_bias", "trap_risk", "ai_thesis", "turbulence_index",
                         "turbulence_threshold", "turbulence_state", "source_close_ts", "math_regime"):
            self.assertIn(required, cols)
        self.assertEqual(seed, (SEED_TS, "SEEDED_OLD_SINGLETON", 0.5, 10.0))


if __name__ == "__main__":
    unittest.main()

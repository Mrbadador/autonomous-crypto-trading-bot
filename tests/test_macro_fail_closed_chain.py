"""TASK 7: the macro sentinel chain must fail closed and may only ever tighten risk.

Covers
  * the killswitch latch: a partial or missing macro cycle may ARM the killswitch but can
    never DISCHARGE it (previously the computed killswitch was never persisted at all unless
    the whole Yahoo cycle was valid, so a SPY dump during a DXY outage was invisible to
    swing_trader's entry gate),
  * event playbooks and AI verdicts are tighten-only,
  * a provider failure holds the last protective posture/anchor instead of softening it,
  * explicit LLM bool parsing (bool('false') is True),
  * freshness-ranked regime precedence instead of "first non-empty key wins",
  * no write ever leaves the temp fixture directory.

Every fixture uses a throwaway directory DB. Nothing here places, cancels or closes an order.
"""

import asyncio
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import unittest
from datetime import datetime, UTC, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

import ai_engine  # noqa: E402
import config  # noqa: E402
import macro_sentinel  # noqa: E402

SYSTEM_STATE_DDL = """
CREATE TABLE IF NOT EXISTS "system_state" (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
)
"""

# A skew-halt verdict exactly as the deterministic gate leaves it with no killswitch armed, so a
# held copy must survive an AI outage unchanged.
PROTECTIVE_SYNTH = {
    "market_posture": "TACTICAL_DEFENSE",
    "macro_bias": "DEFENSIVE_CHOP",
    "derivatives_skew_halt": True,
    "derivatives_skew_regime": "DELEVERAGING_RISK",
    "derivatives_notes": "crowded longs, cascade risk",
    "executive_summary": "risk off",
}


def iso(seconds_ago: float) -> str:
    return (datetime.now(UTC) - timedelta(seconds=seconds_ago)).isoformat()


def macro_cycle(spy_change=0.10, dxy=104.2, killswitch=0, data_ok=True, regime="NEUTRAL"):
    """Shape of macro_sentinel._raw_fetch_macro()'s payload."""
    return {
        "spy_price": 661.4,
        "uup_price": 27.51,
        "dxy_price": dxy if dxy is not None else 0.0,
        "dxy_valid": dxy is not None,
        "spy_1h_change": spy_change,
        "uup_1h_change": -0.05,
        "global_killswitch": killswitch,
        "macro_regime": regime,
        "data_ok": data_ok,
    }


def synth(posture="TACTICAL_DEFENSE", bias="NEUTRAL_CHOP", halt=False,
          regime="BALANCED_CHOP", source="ai", market_posture=None, macro_bias=None,
          derivatives_skew_halt=None, derivatives_skew_regime=None, derivatives_notes=None,
          executive_summary=None):
    """Keyword-compatible with PROTECTIVE_SYNTH so a protective verdict can be spliced in."""
    posture = market_posture or posture
    bias = macro_bias or bias
    halt = derivatives_skew_halt if derivatives_skew_halt is not None else halt
    regime = derivatives_skew_regime or regime
    payload = {"market_posture": posture, "macro_bias": bias, "derivatives_skew_halt": halt,
               "derivatives_skew_regime": regime, "derivatives_notes": derivatives_notes or "funding normal",
               "executive_summary": executive_summary or "balanced", "key_risks": ["chop"],
               "tactical_guidance": "hold", "synthesis_source": source}
    return payload


def anchor(state="HEALTHY_EXPANSION", pause=False, source="ai"):
    return {"btc_anchor_state": state, "alt_long_pause": pause, "contagion_risk_score": 22,
            "key_btc_pivot": 64000.0, "anchor_directive": "normal ops", "confidence": 0.8,
            "anchor_source": source}


def deriv_skew():
    return {"symbols": {"BTCUSDT": {"funding_rate_8h": 0.0001, "open_interest_contracts": 90000.0}},
            "avg_funding_rate_8h": 0.0001, "max_funding_rate_8h": 0.0001, "valid_symbols_count": 1}


class CycleHarness:
    """Runs macro_sentinel.update_macro_context() for a fixed number of cycles against temp DBs."""

    def __init__(self, tmp_dir, cycles, macro_payloads, synths, anchors):
        self.tmp = tmp_dir
        self.cycles = cycles
        self.macro_payloads = list(macro_payloads)
        self.synths = list(synths)
        self.anchors = list(anchors)
        self.touched_paths = []
        self.threads = []

    class _Stop:
        def __init__(self, cycles):
            self.remaining = cycles
            self.waited = 0

        def is_set(self):
            return self.waited >= self.remaining

        def wait(self, timeout=None):
            self.waited += 1
            return True

        def set(self):
            self.waited = 10 ** 6

    def _db_path(self, name):
        path = os.path.join(self.tmp, name)
        self.touched_paths.append(path)
        return path

    def run(self):
        data_db = os.path.join(self.tmp, "trading_data.db")
        ledger_db = os.path.join(self.tmp, "trading_ledger.db")
        macro_sentinel.init_db(ledger_db)
        macro_sentinel.init_db(data_db)

        synth_mock = AsyncMock(side_effect=lambda *a, **k: self.synths.pop(0) if self.synths else synth())
        anchor_mock = AsyncMock(side_effect=lambda *a, **k: self.anchors.pop(0) if self.anchors else anchor())

        patches = [
            patch.object(macro_sentinel, "_get_db_path", side_effect=self._db_path),
            patch.object(macro_sentinel, "BASE_DIR", self.tmp),
            patch.object(macro_sentinel, "_stop_event", self._Stop(self.cycles)),
            patch.object(macro_sentinel, "fetch_macro_data", side_effect=lambda: self.macro_payloads.pop(0)),
            patch.object(macro_sentinel, "fetch_derivatives_skew", side_effect=deriv_skew),
            patch("economic_calendar.EconomicCalendarGate"),
            patch.object(ai_engine, "generate_macro_synthesis", synth_mock),
            patch.object(ai_engine, "evaluate_btc_anchor", anchor_mock),
            patch.object(ai_engine, "calculate_delta_neutral_hedge",
                         AsyncMock(return_value={"risk_shield_level": "GREEN_SHIELD"})),
            patch.object(ai_engine, "rank_asset_hierarchy",
                         AsyncMock(return_value={"priority_leaders": ["BTC/USDT"], "avoid_chop_list": []})),
            patch.object(ai_engine, "generate_daily_market_blueprint",
                         AsyncMock(return_value={"macro_bias": "NEUTRAL", "daily_narrative": "n"})),
        ]
        for p in patches:
            p.start()
        errors = []

        def _target():
            try:
                macro_sentinel.update_macro_context()
            except Exception as exc:  # pragma: no cover - surfaced to the test body
                errors.append(exc)

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout=45.0)
        for p in patches:
            p.stop()
        if thread.is_alive():
            raise AssertionError("update_macro_context() did not finish the requested cycles")
        if errors:
            raise AssertionError(f"update_macro_context() raised: {errors[0]!r}")
        self.synth_mock = synth_mock
        self.anchor_mock = anchor_mock
        return data_db, ledger_db

    @staticmethod
    def state(db_path):
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            return dict(conn.execute("SELECT key, value FROM system_state").fetchall())


class MacroChainFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="macro_chain_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.paths = []

    def seed(self, values, as_of=None):
        db = os.path.join(self.tmp, "trading_data.db")
        ledger = os.path.join(self.tmp, "trading_ledger.db")
        ts = as_of or datetime.now(UTC).isoformat()
        for path in (db, ledger):
            with sqlite3.connect(path) as conn:
                conn.execute(SYSTEM_STATE_DDL)
                for key, value in values.items():
                    stamp = value[1] if isinstance(value, tuple) else ts
                    payload = value[0] if isinstance(value, tuple) else value
                    conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                                 (key, payload, stamp))
        return db

    def run_cycles(self, macro_payloads, synths=None, anchors=None, cycles=None):
        harness = CycleHarness(self.tmp, cycles or len(macro_payloads), macro_payloads,
                               list(synths or []), list(anchors or []))
        data_db, ledger_db = harness.run()
        # Nothing may escape the fixture directory - the live ledgers are off limits.
        for path in harness.touched_paths:
            self.assertTrue(path.startswith(self.tmp), f"sentinel wrote outside the fixture dir: {path}")
        self.assertEqual(set(harness.touched_paths),
                         {os.path.join(self.tmp, "trading_data.db"), os.path.join(self.tmp, "trading_ledger.db")})
        self.harness = harness
        return harness.state(data_db), harness.state(ledger_db)


class TestKillswitchLatch(MacroChainFixture):
    def test_partial_cycle_with_spy_dump_persists_the_engaged_killswitch(self):
        """The headline defect: data_ok=False used to suppress the killswitch write entirely."""
        dumped = macro_cycle(spy_change=-1.42, dxy=None, killswitch=1, data_ok=False,
                             regime="RISK_OFF_KILLSWITCH")
        state, ledger_state = self.run_cycles([dumped])
        self.assertEqual(state["macro_global_killswitch"], "1")
        self.assertEqual(ledger_state["macro_global_killswitch"], "1")
        self.assertEqual(state["macro_data_ok"], "0")
        self.assertEqual(state["macro_killswitch_state"], "MEASURED_PARTIAL")
        # regime survives instead of being flattened to UNKNOWN
        self.assertEqual(state["macro_regime"], "RISK_OFF_KILLSWITCH")

    def test_unmeasured_cycle_holds_an_armed_killswitch(self):
        cycle_one = macro_cycle(spy_change=-1.42, dxy=None, killswitch=1, data_ok=False,
                                regime="RISK_OFF_KILLSWITCH")
        state, _ = self.run_cycles([cycle_one, None])
        self.assertEqual(state["macro_global_killswitch"], "1")
        self.assertEqual(state["macro_killswitch_state"], "HELD")

    def test_missing_measurement_cannot_discharge_an_armed_killswitch(self):
        first = macro_cycle(spy_change=-1.42, dxy=None, killswitch=1, data_ok=False,
                             regime="RISK_OFF_KILLSWITCH")
        # Second cycle: Yahoo returned nothing at all.
        state, _ = self.run_cycles([first, None])
        self.assertEqual(state["macro_global_killswitch"], "1")
        self.assertEqual(state["macro_killswitch_state"], "HELD")
        self.assertEqual(state["macro_data_ok"], "0")

    def test_partially_valid_cycle_cannot_discharge_either(self):
        """DXY outage + SPY flat: there is no authoritative 'clear', so the latch stays put."""
        first = macro_cycle(spy_change=-1.42, dxy=None, killswitch=1, data_ok=False,
                             regime="RISK_OFF_KILLSWITCH")
        second = macro_cycle(spy_change=-0.10, dxy=None, killswitch=0, data_ok=False, regime="UNKNOWN")
        state, _ = self.run_cycles([first, second])
        self.assertEqual(state["macro_global_killswitch"], "1")
        self.assertEqual(state["macro_killswitch_state"], "HELD")

    def test_fully_valid_cycle_discharges_the_killswitch(self):
        first = macro_cycle(spy_change=-1.42, dxy=None, killswitch=1, data_ok=False,
                             regime="RISK_OFF_KILLSWITCH")
        second = macro_cycle(spy_change=0.20, dxy=104.0, killswitch=0, data_ok=True, regime="NEUTRAL")
        state, _ = self.run_cycles([first, second])
        self.assertEqual(state["macro_global_killswitch"], "0")
        self.assertEqual(state["macro_killswitch_state"], "MEASURED")
        self.assertEqual(state["macro_data_ok"], "1")

    def test_killswitch_flag_is_persisted_with_the_computed_value_on_a_good_cycle(self):
        state, _ = self.run_cycles([macro_cycle(spy_change=-1.9, killswitch=1, data_ok=True,
                                                regime="RISK_OFF_KILLSWITCH")])
        self.assertEqual(state["macro_global_killswitch"], "1")
        self.assertEqual(state["macro_killswitch_state"], "MEASURED")

    def test_telemetry_history_records_the_armed_killswitch_and_spy_move(self):
        dumped = macro_cycle(spy_change=-1.42, dxy=None, killswitch=1, data_ok=False,
                             regime="RISK_OFF_KILLSWITCH")
        self.run_cycles([dumped])
        with sqlite3.connect(f"file:{os.path.join(self.tmp, 'trading_data.db')}?mode=ro", uri=True) as conn:
            row = conn.execute("SELECT global_killswitch, macro_regime, spy_1h_change, dxy_price "
                               "FROM macro_telemetry_history ORDER BY id DESC LIMIT 1").fetchone()
        self.assertEqual(row[0], 1)
        self.assertEqual(row[1], "RISK_OFF_KILLSWITCH")
        self.assertAlmostEqual(row[2], -1.42)
        self.assertIsNone(row[3])  # unmeasurable DXY is NULL, never a fabricated 0.0


class TestKillswitchHelpers(unittest.TestCase):
    def test_resolve_killswitch_matrix(self):
        armed_partial = macro_cycle(spy_change=-1.4, dxy=None, killswitch=1, data_ok=False,
                                    regime="RISK_OFF_KILLSWITCH")
        armed_full = macro_cycle(spy_change=-1.4, killswitch=1, data_ok=True, regime="RISK_OFF_KILLSWITCH")
        flat_partial = macro_cycle(spy_change=-0.1, dxy=None, killswitch=0, data_ok=False, regime="UNKNOWN")
        flat_full = macro_cycle(killswitch=0, data_ok=True)

        self.assertEqual(macro_sentinel.resolve_killswitch(armed_partial, 0, "", "T1"),
                         {"active": 1, "state": "MEASURED_PARTIAL", "asof": "T1"})
        self.assertEqual(macro_sentinel.resolve_killswitch(armed_full, 1, "T0", "T1"),
                         {"active": 1, "state": "MEASURED", "asof": "T1"})
        self.assertEqual(macro_sentinel.resolve_killswitch(flat_full, 1, "T0", "T1"),
                         {"active": 0, "state": "MEASURED", "asof": "T1"})
        self.assertEqual(macro_sentinel.resolve_killswitch(flat_partial, 1, "T0", "T1"),
                         {"active": 1, "state": "HELD", "asof": "T0"})
        self.assertEqual(macro_sentinel.resolve_killswitch(None, 0, "T0", "T1"),
                         {"active": 0, "state": "HELD", "asof": "T0"})
        self.assertEqual(macro_sentinel.resolve_killswitch(None, 0, "", "T1"),
                         {"active": 0, "state": "HELD", "asof": ""})

    def test_read_latch_uses_the_persisted_value_and_fails_closed_on_an_unreadable_db(self):
        tmp = tempfile.mkdtemp(prefix="latch_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        db = os.path.join(tmp, "trading_data.db")
        with sqlite3.connect(db) as conn:
            conn.execute(SYSTEM_STATE_DDL)
        self.assertEqual(macro_sentinel.read_killswitch_latch(db), (0, ""))
        with sqlite3.connect(db) as conn:
            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                         ("macro_global_killswitch", "1", "2026-10-14T00:00:00+00:00"))
            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                         ("macro_killswitch_asof", "2026-10-14T00:00:00+00:00", "2026-10-14T00:00:00+00:00"))
        flag, asof = macro_sentinel.read_killswitch_latch(db)
        self.assertEqual((flag, asof), (1, "2026-10-14T00:00:00+00:00"))
        with self.assertLogs("macro_sentinel", level="WARNING") as captured:
            self.assertEqual(macro_sentinel.read_killswitch_latch(os.path.join(tmp, "missing.db"))[0], 1)
        self.assertIn("fail-closed", captured.output[0])

    def test_row_age_and_configured_max_age(self):
        self.assertIsNone(macro_sentinel._row_age_sec(None))
        self.assertIsNone(macro_sentinel._row_age_sec(""))
        self.assertIsNone(macro_sentinel._row_age_sec("yesterday"))
        self.assertAlmostEqual(macro_sentinel._row_age_sec(iso(60)), 60.0, delta=2.0)
        future = (datetime.now(UTC) + timedelta(seconds=600)).isoformat()
        self.assertEqual(macro_sentinel._row_age_sec(future), 0.0)
        naive = datetime.now(UTC).replace(tzinfo=None).isoformat()
        self.assertIsNotNone(macro_sentinel._row_age_sec(naive))
        with patch.object(config, "REGIME_MAX_AGE_SEC", 1800.0):
            self.assertEqual(macro_sentinel._state_max_age_sec(), 1800.0)
        for broken in (0.0, -5.0, float("nan"), "not-a-number", None):
            with patch.object(config, "REGIME_MAX_AGE_SEC", broken):
                self.assertEqual(macro_sentinel._state_max_age_sec(), 900.0)


class TestRegimePrecedence(MacroChainFixture):
    def test_freshest_label_wins_regardless_of_key_order(self):
        rows = {
            "market_regime_label": ("CHOP", iso(420)),
            "super_bot_regime": (None, None),
            "warden_60m_regime": ("LONG_ONLY", iso(60)),
        }
        regime, key, age = macro_sentinel.select_fresh_regime(rows, max_age_sec=900.0)
        self.assertEqual((regime, key), ("LONG_ONLY", "warden_60m_regime"))
        self.assertAlmostEqual(age, 60.0, delta=2.0)

    def test_stale_candidates_are_excluded_and_the_next_fresh_one_is_used(self):
        rows = {
            "market_regime_label": ("CHOP", iso(60)),
            "warden_60m_regime": ("SHORT_ONLY", iso(7200)),
        }
        regime, key, _ = macro_sentinel.select_fresh_regime(rows, max_age_sec=900.0)
        self.assertEqual((regime, key), ("CHOP", "market_regime_label"))

    def test_all_stale_or_absent_yields_unknown(self):
        rows = {"market_regime_label": ("CHOP", iso(7200)), "warden_60m_regime": ("LONG_ONLY", None)}
        with self.assertLogs("macro_sentinel", level="WARNING") as captured:
            self.assertEqual(macro_sentinel.select_fresh_regime(rows, max_age_sec=900.0),
                             ("UNKNOWN", None, None))
        joined = " ".join(captured.output)
        self.assertIn("stale", joined)
        self.assertIn("unusable", joined)

    def test_blank_value_is_not_a_candidate(self):
        rows = {"market_regime_label": ("   ", iso(10)), "warden_60m_regime": ("CHOP_GRID_ACTIVE", iso(20))}
        self.assertEqual(macro_sentinel.select_fresh_regime(rows, max_age_sec=900.0)[:2],
                         ("CHOP_GRID_ACTIVE", "warden_60m_regime"))

    def test_default_budget_comes_from_config(self):
        rows = {"market_regime_label": ("CHOP", iso(300))}
        with patch.object(config, "REGIME_MAX_AGE_SEC", 120.0):
            self.assertEqual(macro_sentinel.select_fresh_regime(rows)[0], "UNKNOWN")
        with patch.object(config, "REGIME_MAX_AGE_SEC", 900.0):
            self.assertEqual(macro_sentinel.select_fresh_regime(rows)[0], "CHOP")

    def test_packet_carries_the_fresh_label_and_flags_an_absent_cache(self):
        self.seed({
            "market_regime_label": ("DEFENSIVE_CHOP", iso(1200)),   # stale: classifier was down
            "warden_60m_regime": ("LONG_ONLY", iso(90)),            # fresh: warden is alive
        })
        self.run_cycles([macro_cycle()])
        packet = self.harness.synth_mock.call_args[0][0]
        self.assertEqual(packet["warden_regime"], "LONG_ONLY")
        self.assertEqual(packet["warden_regime_source"], "warden_60m_regime")

    def test_absent_regime_cache_feeds_unknown_not_a_permissive_default(self):
        self.run_cycles([macro_cycle()])
        packet = self.harness.synth_mock.call_args[0][0]
        self.assertEqual(packet["warden_regime"], "UNKNOWN")
        self.assertEqual(packet["warden_regime_source"], "STALE_OR_ABSENT")


class TestProtectiveStateHolding(MacroChainFixture):
    def test_failed_synthesis_holds_the_last_protective_posture(self):
        self.run_cycles([macro_cycle(), macro_cycle()],
                        synths=[synth(**PROTECTIVE_SYNTH, source="ai"),
                                synth(posture="TACTICAL_DEFENSE", bias="NEUTRAL_CHOP", halt=False,
                                      regime="BALANCED_CHOP", source="fallback")])
        state = self.harness.state(os.path.join(self.tmp, "trading_data.db"))
        self.assertEqual(state["ai_macro_posture"], "TACTICAL_DEFENSE")
        self.assertEqual(state["ai_macro_bias"], "DEFENSIVE_CHOP")
        self.assertEqual(state["DERIVATIVES_SKEW_HALT"], "1")
        self.assertEqual(state["derivatives_skew_regime"], "DELEVERAGING_RISK")
        self.assertEqual(state["ai_macro_posture_held"], "1")
        # The held verdict is stamped with the cycle that produced it, not with this cycle's clock.
        self.assertNotEqual(state["ai_macro_posture_asof"], "")
        self.assertNotEqual(state["ai_macro_posture_asof"], state["macro_heartbeat"])

    def test_protective_posture_is_re_measured_once_a_valid_verdict_returns(self):
        self.run_cycles([macro_cycle(), macro_cycle()],
                        synths=[synth(**PROTECTIVE_SYNTH, source="ai"),
                                synth(posture="TACTICAL_DEFENSE", bias="NEUTRAL_CHOP", halt=False,
                                      regime="BALANCED_CHOP", source="ai")])
        state = self.harness.state(os.path.join(self.tmp, "trading_data.db"))
        self.assertEqual(state["DERIVATIVES_SKEW_HALT"], "0")
        self.assertEqual(state["ai_macro_posture_held"], "0")

    def test_failed_synthesis_without_any_prior_verdict_is_defensive_not_crashing(self):
        state, _ = self.run_cycles([None], synths=[{"synthesis_source": "fallback"}])
        self.assertEqual(state["ai_macro_posture"], "TACTICAL_DEFENSE")
        self.assertEqual(state["ai_macro_bias"], "NEUTRAL_CHOP")
        self.assertEqual(state["ai_macro_posture_held"], "1")
        self.assertEqual(state["ai_macro_posture_asof"], "")

    def test_killswitch_armed_verdict_survives_an_ai_outage_and_a_yahoo_outage(self):
        armed = macro_cycle(spy_change=-1.5, killswitch=1, data_ok=True, regime="RISK_OFF_KILLSWITCH")
        state, _ = self.run_cycles([armed, None],
                                   synths=[synth(posture="AGGRESSIVE_EXPANSION", bias="BULLISH"),
                                           {"synthesis_source": "fallback"}])
        self.assertEqual(state["ai_macro_posture"], "RISK_OFF_KILLSWITCH")
        self.assertEqual(state["ai_macro_bias"], "DEFENSIVE_BEAR")
        self.assertEqual(state["macro_global_killswitch"], "1")
        self.assertEqual(state["macro_killswitch_state"], "HELD")
        self.assertEqual(state["ai_macro_posture_held"], "1")

    def test_non_dict_synthesis_holds_rather_than_resetting(self):
        self.run_cycles([macro_cycle(), macro_cycle()],
                        synths=[synth(**PROTECTIVE_SYNTH, source="ai"), None])
        state = self.harness.state(os.path.join(self.tmp, "trading_data.db"))
        self.assertEqual(state["ai_macro_bias"], "DEFENSIVE_CHOP")
        self.assertEqual(state["DERIVATIVES_SKEW_HALT"], "1")

    def test_failed_anchor_holds_alt_long_pause(self):
        self.run_cycles([macro_cycle(), macro_cycle()],
                        anchors=[anchor(state="DISTRIBUTION_DUMP_RISK", pause=True, source="ai"),
                                 anchor(state="NEUTRAL_RANGE", pause=False, source="fallback")])
        state = self.harness.state(os.path.join(self.tmp, "trading_data.db"))
        self.assertEqual(state["ALT_LONG_PAUSE"], "1")
        self.assertEqual(state["alt_long_pause"], "1")
        self.assertEqual(state["btc_anchor_state"], "DISTRIBUTION_DUMP_RISK")
        self.assertEqual(state["btc_anchor_held"], "1")

    def test_anchor_disarmed_only_by_a_fresh_verdict(self):
        self.run_cycles([macro_cycle(), macro_cycle()],
                        anchors=[anchor(state="DISTRIBUTION_DUMP_RISK", pause=True, source="ai"),
                                 anchor(state="HEALTHY_ACCUMULATION", pause=False, source="ai")])
        state = self.harness.state(os.path.join(self.tmp, "trading_data.db"))
        self.assertEqual(state["ALT_LONG_PAUSE"], "0")
        self.assertEqual(state["btc_anchor_held"], "0")

    def test_normalize_macro_synth_helpers(self):
        held = dict(PROTECTIVE_SYNTH, _asof="2026-10-14T00:00:00+00:00")
        out, was_held, asof = macro_sentinel.normalize_macro_synth(synth(source="fallback"), held, "NOW")
        self.assertTrue(was_held)
        self.assertEqual(asof, "2026-10-14T00:00:00+00:00")
        self.assertEqual(out["market_posture"], "TACTICAL_DEFENSE")
        self.assertEqual(out["macro_bias"], "DEFENSIVE_CHOP")
        self.assertTrue(out["posture_held"])

        out, was_held, asof = macro_sentinel.normalize_macro_synth(synth(source="ai"), held, "NOW")
        self.assertFalse(was_held)
        self.assertEqual(asof, "NOW")
        self.assertNotIn("posture_held", out)

        unmarked = macro_sentinel.normalize_macro_synth({"market_posture": "AGGRESSIVE_EXPANSION"}, held, "NOW")
        self.assertTrue(unmarked[1])
        self.assertEqual(unmarked[0]["market_posture"], "TACTICAL_DEFENSE")

        out, was_held, asof = macro_sentinel.normalize_macro_synth(None, None, "NOW")
        self.assertTrue(was_held)
        self.assertEqual(asof, "")
        self.assertEqual(out["market_posture"], "TACTICAL_DEFENSE")
        self.assertEqual(out["derivatives_skew_halt"], False)

        anchor_held = {"btc_anchor_state": "DISTRIBUTION_DUMP_RISK", "alt_long_pause": True,
                       "_asof": "2026-10-14T00:00:00+00:00"}
        out, was_held, asof = macro_sentinel.normalize_btc_anchor(anchor(source="fallback"), anchor_held, "NOW")
        self.assertTrue(was_held)
        self.assertEqual(out["alt_long_pause"], True)
        self.assertEqual(asof, "2026-10-14T00:00:00+00:00")
        out, was_held, asof = macro_sentinel.normalize_btc_anchor({}, None, "NOW")
        self.assertTrue(was_held)
        self.assertEqual(asof, "")
        self.assertEqual(out["btc_anchor_state"], "NEUTRAL_RANGE")
        self.assertFalse(out["alt_long_pause"])

        marked = macro_sentinel.normalize_btc_anchor(anchor(state="HEALTHY_EXPANSION", source="ai"),
                                                     anchor_held, "NOW")
        self.assertFalse(marked[1])
        self.assertEqual(marked[0]["btc_anchor_state"], "HEALTHY_EXPANSION")


class TestTightenOnlyOverrides(unittest.TestCase):
    def test_bull_playbook_never_loosens_the_posture(self):
        payload = synth(posture="TACTICAL_DEFENSE", bias="NEUTRAL_CHOP")
        out, is_halt, _, _ = macro_sentinel.apply_macro_overrides(payload, "BULL_MOMENTUM_EXPANSION", 0, True)
        self.assertNotEqual(out["market_posture"], "AGGRESSIVE_EXPANSION")
        self.assertNotEqual(out["macro_bias"], "BULLISH")
        self.assertEqual(out["market_posture"], "TACTICAL_DEFENSE")
        self.assertFalse(is_halt)

    def test_bull_playbook_is_refused_while_the_killswitch_is_armed(self):
        out, _, _, _ = macro_sentinel.apply_macro_overrides(synth(), "BULL_MOMENTUM_EXPANSION", 1, True)
        self.assertEqual(out["market_posture"], "RISK_OFF_KILLSWITCH")
        self.assertEqual(out["macro_bias"], "DEFENSIVE_BEAR")

    def test_bull_playbook_is_refused_on_an_invalid_macro_cycle(self):
        out, _, _, _ = macro_sentinel.apply_macro_overrides(synth(posture="AGGRESSIVE_EXPANSION",
                                                                  bias="BULLISH"),
                                                           "BULL_MOMENTUM_EXPANSION", 0, False)
        self.assertEqual(out["market_posture"], "TACTICAL_DEFENSE")
        self.assertEqual(out["macro_bias"], "NEUTRAL")

    def test_bear_playbook_tightens(self):
        out, _, _, _ = macro_sentinel.apply_macro_overrides(synth(posture="AGGRESSIVE_EXPANSION", bias="BULLISH"),
                                                            "SECULAR_SHORT_BREAKDOWN", 0, True)
        self.assertEqual((out["market_posture"], out["macro_bias"]), ("TACTICAL_DEFENSE", "BEARISH"))

    def test_killswitch_beats_a_clean_synthesis_and_skew_halt_beats_a_bull_bias(self):
        out, _, _, _ = macro_sentinel.apply_macro_overrides(synth(posture="AGGRESSIVE_EXPANSION", bias="BULLISH"),
                                                            "NORMAL_OPERATIONS", 1, True)
        self.assertEqual((out["market_posture"], out["macro_bias"]), ("RISK_OFF_KILLSWITCH", "DEFENSIVE_BEAR"))
        out, is_halt, regime, _ = macro_sentinel.apply_macro_overrides(
            synth(posture="AGGRESSIVE_EXPANSION", bias="BULLISH", halt="true", regime="DELEVERAGING_RISK"),
            "NORMAL_OPERATIONS", 0, True)
        self.assertTrue(is_halt)
        self.assertEqual(regime, "DELEVERAGING_RISK")
        self.assertEqual((out["market_posture"], out["macro_bias"]), ("TACTICAL_DEFENSE", "DEFENSIVE_CHOP"))

    def test_string_false_does_not_arm_the_skew_circuit_breaker(self):
        out, is_halt, _, _ = macro_sentinel.apply_macro_overrides(synth(halt="false"), "NORMAL_OPERATIONS", 0, True)
        self.assertFalse(is_halt)
        self.assertEqual(out["market_posture"], "TACTICAL_DEFENSE")
        for truthy in ("true", "1", "YES", True, 1):
            _, is_halt, _, _ = macro_sentinel.apply_macro_overrides(synth(halt=truthy), "NORMAL_OPERATIONS", 0, True)
            self.assertTrue(is_halt, f"{truthy!r} should arm the halt")

    def test_synthesis_stays_tight_on_a_stale_regime_cache(self):
        """UNKNOWN warden_regime must reach the strategist instead of a permissive placeholder."""
        rows = {"market_regime_label": ("CHOP", None)}
        self.assertEqual(macro_sentinel.select_fresh_regime(rows, max_age_sec=900.0)[0], "UNKNOWN")


class TestProviderPayloadContract(unittest.TestCase):
    """ai_engine: explicit bool parsing and a source marker the sentinel can act on."""

    def test_strict_bool(self):
        self.assertFalse(ai_engine.strict_bool("false"))
        self.assertFalse(ai_engine.strict_bool("0"))
        self.assertFalse(ai_engine.strict_bool(""))
        self.assertFalse(ai_engine.strict_bool(None, False))
        self.assertFalse(ai_engine.strict_bool("  FAlse  "))
        self.assertTrue(ai_engine.strict_bool("true"))
        self.assertTrue(ai_engine.strict_bool(1))
        self.assertTrue(ai_engine.strict_bool(True))
        self.assertTrue(ai_engine.strict_bool(None, True))
        self.assertTrue(ai_engine.strict_bool("maybe", True))
        self.assertFalse(ai_engine.strict_bool("maybe", False))

    def _run_synth(self, gemini_payload):
        with patch.object(ai_engine, "get_llm", return_value=MagicMock()), \
                patch.object(ai_engine, "_ensure_active_session", return_value=MagicMock()), \
                patch.object(ai_engine, "_query_gemini_cloud",
                             AsyncMock(return_value=gemini_payload)), \
                patch.object(ai_engine, "_query_groq_cloud_with_rotation",
                             AsyncMock(return_value=None)):
            return asyncio.run(ai_engine.generate_macro_synthesis({"data_ok": True}, timeout_sec=1.0))

    def test_macro_synthesis_marks_an_ai_verdict_and_parses_the_halt_flag(self):
        body = {"market_posture": "TACTICAL_DEFENSE", "macro_bias": "BEARISH",
                "derivatives_skew_regime": "DELEVERAGING_RISK", "derivatives_skew_halt": "false"}
        response = {"choices": [{"message": {"content": json.dumps(body)}}]}
        out = self._run_synth(response)
        self.assertEqual(out["synthesis_source"], "ai")
        self.assertIs(out["derivatives_skew_halt"], False)

    def test_macro_synthesis_fallback_is_marked_so_it_can_be_held(self):
        out = self._run_synth(None)
        self.assertEqual(out["synthesis_source"], "fallback")
        self.assertIs(out["derivatives_skew_halt"], False)
        self.assertEqual(out["market_posture"], "TACTICAL_DEFENSE")

    def test_btc_anchor_payload_is_bool_normalised(self):
        body = {"btc_anchor_state": "DISTRIBUTION_DUMP_RISK", "alt_long_pause": "false"}
        response = {"choices": [{"message": {"content": json.dumps(body)}}]}
        with patch.object(ai_engine, "get_llm", return_value=MagicMock()), \
                patch.object(ai_engine, "_ensure_active_session", return_value=MagicMock()), \
                patch.object(ai_engine, "_query_gemini_cloud", AsyncMock(return_value=response)), \
                patch.object(ai_engine, "_query_groq_cloud_with_rotation", AsyncMock(return_value=None)):
            out = asyncio.run(ai_engine.evaluate_btc_anchor({"btc_price": 64000}, timeout_sec=1.0))
        self.assertEqual(out["anchor_source"], "ai")
        self.assertIs(out["alt_long_pause"], False)
        with patch.object(ai_engine, "get_llm", return_value=MagicMock()), \
                patch.object(ai_engine, "_ensure_active_session", return_value=MagicMock()), \
                patch.object(ai_engine, "_query_gemini_cloud", AsyncMock(return_value=None)), \
                patch.object(ai_engine, "_query_groq_cloud_with_rotation", AsyncMock(return_value=None)):
            fallback = asyncio.run(ai_engine.evaluate_btc_anchor({"btc_price": 64000}, timeout_sec=1.0))
        self.assertEqual(fallback["anchor_source"], "fallback")


class TestNoLiveWrites(unittest.TestCase):
    def test_module_targets_are_derived_from_the_patched_base_dir(self):
        """_get_db_path is the only path builder; the harness asserts every use stays in tmp."""
        with patch.object(macro_sentinel, "BASE_DIR", os.sep + "fixture_only"):
            self.assertEqual(macro_sentinel._get_db_path("trading_data.db"),
                             os.path.join(os.sep + "fixture_only", "trading_data.db"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

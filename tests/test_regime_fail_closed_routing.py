"""Regime P0 / TASK 2 — the regime consumer must fail closed for NEW entries only.

Proves four things about SuperBotTrader:
  * the engine starts blocked (regime_data_ok False) before any valid read;
  * _get_market_regime_sync rejects an absent row, a missing/unparseable/too-old
    publication timestamp, an UNKNOWN label and non-finite or out-of-range
    ADX/Hurst/+DI/-DI telemetry, with the ceiling read from
    config.REGIME_MAX_AGE_SEC (default 900s);
  * the block is logged once per health transition, not once per routing loop;
  * run() skips the NEW-entry scans while unhealthy while position management
    (trailing stops / breakeven ratchets / exits) keeps running, and resumes
    scanning as soon as a fresh valid row appears.

Isolation: the whole class chdirs into a temp directory holding its own
trading_data.db built from the production DDL, so the repository database is
never opened for writing. Nothing here places, cancels or closes an order.
"""

import asyncio
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, UTC, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

import config
from super_bot import SuperBotTrader

# super_bot.py uses logging.getLogger("super_bot"); the "[SUPER_BOT]" tag seen in the
# console output is a formatter prefix, not the logger name.
REGIME_LOGGER = "super_bot"

MARKET_REGIME_DDL = """
CREATE TABLE market_regime (
    id INTEGER PRIMARY KEY DEFAULT 1,
    timestamp TEXT,
    regime TEXT,
    hurst REAL,
    adx REAL,
    plus_di REAL,
    minus_di REAL,
    macro_trend TEXT,
    ai_bias TEXT,
    trap_risk REAL,
    ai_thesis TEXT,
    turbulence_index REAL,
    turbulence_threshold REAL,
    turbulence_state TEXT,
    source_close_ts TEXT,
    math_regime TEXT
)
"""

TELEMETRY_DDL = """
CREATE TABLE super_bot_telemetry (
    timestamp TEXT PRIMARY KEY,
    regime TEXT,
    adx REAL,
    hurst REAL,
    active_swing_count INTEGER,
    active_grid_count INTEGER,
    total_unrealized_pnl REAL,
    status TEXT
)
"""

SYSTEM_STATE_DDL = """
CREATE TABLE "system_state" (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
)
"""

OUT_OF_RANGE_ADX = 250.0
OUT_OF_RANGE_HURST = 4.0
# Distinguishes "caller wants the default fresh row" from "caller wants a literal NULL".
_DEFAULT = object()


class TestRegimeFailClosedConsumer(unittest.TestCase):
    bot: SuperBotTrader
    tmp_dir: str

    @classmethod
    def setUpClass(cls):
        cls.repo_cwd = os.getcwd()
        cls.tmp_dir = tempfile.mkdtemp(prefix="regime_gate_")
        os.chdir(cls.tmp_dir)
        with sqlite3.connect(os.path.join(cls.tmp_dir, "trading_data.db")) as conn:
            conn.execute(MARKET_REGIME_DDL)
            conn.execute(TELEMETRY_DDL)
            conn.execute(SYSTEM_STATE_DDL)
            conn.commit()
        cls.bot = SuperBotTrader()

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.repo_cwd)
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def setUp(self):
        with sqlite3.connect(os.path.join(self.tmp_dir, "trading_data.db")) as conn:
            conn.execute("DELETE FROM market_regime")
            conn.execute("DELETE FROM super_bot_telemetry")
            conn.commit()
        self.bot.regime_data_ok = False
        self.bot.regime_stale_reason = "no regime read yet"
        self.bot._regime_health_logged = None

    # -- helpers -------------------------------------------------------------
    def seed_row(self, *, regime="CHOP", adx=18.0, hurst=0.45, plus_di=10.0,
                 minus_di=12.0, age_sec=5.0, timestamp=_DEFAULT):
        if timestamp is _DEFAULT:
            timestamp = (datetime.now(UTC) - timedelta(seconds=age_sec)).isoformat()
        with sqlite3.connect(os.path.join(self.tmp_dir, "trading_data.db")) as conn:
            conn.execute(
                """INSERT INTO market_regime
                   (timestamp, regime, hurst, adx, plus_di, minus_di, macro_trend,
                    ai_bias, trap_risk, ai_thesis, turbulence_index,
                    turbulence_threshold, turbulence_state)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (timestamp, regime, hurst, adx, plus_di, minus_di, "NEUTRAL",
                 "NEUTRAL", 0.1, "test fixture", 5.0, 22.0, "CALM_NORMAL"),
            )
            conn.commit()

    def read(self):
        return self.bot._get_market_regime_sync()

    # -- golden path ---------------------------------------------------------
    def test_fresh_valid_row_enables_new_entries(self):
        self.seed_row(regime="MILD_BULL", adx=31.0, hurst=0.61, plus_di=24.0, minus_di=9.0)
        regime, adx, hurst = self.read()
        self.assertTrue(self.bot.regime_data_ok)
        self.assertEqual(regime, "MILD_BULL")
        self.assertEqual((adx, hurst), (31.0, 0.61))
        self.assertEqual(self.bot.market_regime_di_spread, 15.0)
        self.assertEqual(self.bot.regime_stale_reason, "regime telemetry valid")

    def test_engine_starts_blocked_before_first_read(self):
        """Fail-closed default: a freshly built engine has no entry permission."""
        fresh = SuperBotTrader()
        self.assertFalse(fresh.regime_data_ok)
        self.assertEqual(fresh.regime_stale_reason, "no regime read yet")

    # -- rejection cases -----------------------------------------------------
    def test_absent_row_blocks_entries(self):
        self.read()
        self.assertFalse(self.bot.regime_data_ok)
        self.assertIn("no rows", self.bot.regime_stale_reason)

    def test_unknown_label_blocks_entries_even_when_fresh(self):
        self.seed_row(regime="UNKNOWN", age_sec=1.0)
        self.read()
        self.assertFalse(self.bot.regime_data_ok)
        self.assertIn("UNKNOWN", self.bot.regime_stale_reason)

    def test_stale_row_blocks_entries_at_the_configured_ceiling(self):
        self.seed_row(age_sec=config.REGIME_MAX_AGE_SEC + 15.0)
        self.read()
        self.assertFalse(self.bot.regime_data_ok)
        self.assertIn("old (max", self.bot.regime_stale_reason)

    def test_max_age_is_config_driven(self):
        """The same ~20 minute old row is valid only when the ceiling is raised."""
        self.seed_row(age_sec=1200.0)
        self.read()
        self.assertFalse(self.bot.regime_data_ok, "1200s row must fail the 900s default")

        with patch.object(config, "REGIME_MAX_AGE_SEC", 1800.0):
            self.read()
            self.assertTrue(self.bot.regime_data_ok, "1200s row passes an 1800s ceiling")

        with patch.object(config, "REGIME_MAX_AGE_SEC", 0.0):
            # A nonsensical ceiling falls back to the documented 900s default.
            self.seed_row(age_sec=60.0)
            self.read()
            self.assertTrue(self.bot.regime_data_ok)

    def test_non_finite_and_out_of_range_telemetry_blocks_entries(self):
        cases = [
            ("ADX NULL", {"adx": None}),
            ("ADX text", {"adx": "not-a-number"}),
            ("ADX NaN", {"adx": "NaN"}),
            ("ADX out of range", {"adx": OUT_OF_RANGE_ADX}),
            ("Hurst NULL", {"hurst": None}),
            ("Hurst NaN", {"hurst": "NaN"}),
            ("Hurst out of range", {"hurst": OUT_OF_RANGE_HURST}),
            ("+DI NULL", {"plus_di": None}),
            ("-DI NULL", {"minus_di": None}),
        ]
        for label, overrides in cases:
            with self.subTest(label=label):
                with sqlite3.connect(os.path.join(self.tmp_dir, "trading_data.db")) as conn:
                    conn.execute("DELETE FROM market_regime")
                    timestamp = (datetime.now(UTC) - timedelta(seconds=5)).isoformat()
                    conn.execute(
                        """INSERT INTO market_regime
                           (timestamp, regime, hurst, adx, plus_di, minus_di,
                            turbulence_index, turbulence_threshold, turbulence_state)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (timestamp,
                         overrides.get("regime", "CHOP"),
                         overrides.get("hurst", 0.45),
                         overrides.get("adx", 18.0),
                         overrides.get("plus_di", 10.0),
                         overrides.get("minus_di", 12.0),
                         5.0, 22.0, "CALM_NORMAL"),
                    )
                    conn.commit()
                self.read()
                self.assertFalse(self.bot.regime_data_ok)

    def test_missing_or_unparseable_timestamp_blocks_entries(self):
        for bad_ts in (None, "", "not-a-timestamp", "2026-13-45T99:99:99"):
            with self.subTest(timestamp=bad_ts):
                self.seed_row(timestamp=bad_ts)
                self.read()
                self.assertFalse(self.bot.regime_data_ok)
                self.assertIn("timestamp", self.bot.regime_stale_reason)

    def test_future_row_is_treated_as_fresh_not_negative_age(self):
        self.seed_row(timestamp=(datetime.now(UTC) + timedelta(minutes=30)).isoformat())
        self.read()
        self.assertTrue(self.bot.regime_data_ok)

    def test_unreadable_database_blocks_entries(self):
        """A DB error must not leave the previous 'healthy' verdict in place."""
        self.seed_row(age_sec=1.0)
        self.read()
        self.assertTrue(self.bot.regime_data_ok)
        self.bot._regime_health_logged = None

        with patch("super_bot.sqlite3.connect", side_effect=sqlite3.OperationalError("locked")):
            regime, adx, hurst = self.read()
        self.assertFalse(self.bot.regime_data_ok)
        self.assertEqual(regime, "UNKNOWN")
        self.assertEqual(adx, 0.0)
        self.assertIn("read failed", self.bot.regime_stale_reason)

    # -- timestamp / finiteness helpers --------------------------------------
    def test_row_age_parser_accepts_the_shapes_the_producer_writes(self):
        now = datetime.now(UTC)
        for value in (
            now.isoformat(),
            now.isoformat().replace("+00:00", "Z"),
            now.strftime("%Y-%m-%d %H:%M:%S"),               # sqlite datetime('now')
            now.strftime("%Y-%m-%dT%H:%M:%S.%f"),            # naive, no offset
        ):
            with self.subTest(value=value):
                age = SuperBotTrader._regime_row_age_sec(value)
                self.assertIsNotNone(age)
                self.assertGreaterEqual(age, 0.0)
                self.assertLess(age, 120.0)
        for value in (None, "", "yesterday", "0"):
            with self.subTest(unparseable=value):
                self.assertIsNone(SuperBotTrader._regime_row_age_sec(value))

    def test_finite_parser_rejects_every_non_numeric_shape(self):
        for value in (None, "", "abc", float("nan"), float("inf")):
            with self.subTest(value=value):
                self.assertIsNone(SuperBotTrader._finite_or_none(value))
        self.assertEqual(SuperBotTrader._finite_or_none("27.5"), 27.5)
        self.assertEqual(SuperBotTrader._finite_or_none(0), 0.0)

    # -- one warning per transition ------------------------------------------
    def test_health_warning_is_logged_once_per_transition(self):
        self.seed_row(age_sec=99999.0)
        with self.assertLogs(REGIME_LOGGER, level="WARNING") as captured:
            for _ in range(20):
                self.read()
        gate_lines = [r for r in captured.output if "[REGIME-GATE]" in r]
        self.assertEqual(len(gate_lines), 1, f"expected a single warning, got {gate_lines}")

        self.seed_row(age_sec=2.0)
        with self.assertLogs(REGIME_LOGGER, level="INFO") as captured:
            for _ in range(20):
                self.read()
        gate_lines = [r for r in captured.output if "[REGIME-GATE]" in r]
        self.assertEqual(len(gate_lines), 1)
        self.assertIn("valid again", gate_lines[0])
        self.assertTrue(self.bot.regime_data_ok)

    # -- routing behaviour ---------------------------------------------------
    def _run_one_cycle(self):
        """Drive exactly one routing loop; returns the mocks that were called."""
        bot = self.bot
        scans_blocked = {"manage": AsyncMock(), "load": MagicMock(),
                         "fleet": AsyncMock(), "grid": AsyncMock(), "short": AsyncMock()}

        def _stop_after_management():
            bot._stop_event.set()
        scans_blocked["manage"].side_effect = _stop_after_management

        bot._stop_event = asyncio.Event()
        bot.symbols = ["BTC/USDT", "ETH/USDT"]
        bot.max_locks = 4
        bot.active_positions = {}
        bot.check_interval_fast = 1
        bot.check_interval_4h = 300
        bot.scalp_short_enabled = True
        bot.swing_engine = MagicMock()
        bot.swing_engine.manage_active_positions = scans_blocked["manage"]
        bot.swing_engine.scan_fleet = scans_blocked["fleet"]
        bot.scan_chop_grid_opportunities = scans_blocked["grid"]
        bot.scan_quick_scalp_short_opportunities = scans_blocked["short"]

        with patch.object(SuperBotTrader, "initialize_clients", AsyncMock()), \
             patch.object(SuperBotTrader, "_load_all_positions_sync", scans_blocked["load"]):
            asyncio.run(bot.run())
        return scans_blocked

    def test_stale_regime_skips_new_entries_but_keeps_management(self):
        self.seed_row(age_sec=99999.0, regime="CHOP", adx=18.0)
        calls = self._run_one_cycle()
        self.assertFalse(self.bot.regime_data_ok)
        self.assertEqual(calls["manage"].await_count, 1, "position management must keep running")
        self.assertEqual(calls["fleet"].await_count, 0)
        self.assertEqual(calls["grid"].await_count, 0)
        self.assertEqual(calls["short"].await_count, 0)

    def test_absent_regime_skips_new_entries_but_keeps_management(self):
        calls = self._run_one_cycle()
        self.assertFalse(self.bot.regime_data_ok)
        self.assertEqual(calls["manage"].await_count, 1)
        self.assertEqual(calls["grid"].await_count, 0)
        self.assertEqual(calls["short"].await_count, 0)

    def test_valid_chop_regime_routes_the_grid_scan(self):
        self.seed_row(age_sec=2.0, regime="CHOP", adx=18.0)
        calls = self._run_one_cycle()
        self.assertTrue(self.bot.regime_data_ok)
        self.assertEqual(calls["manage"].await_count, 1)
        self.assertEqual(calls["grid"].await_count, 1, "healthy CHOP must reach the grid scan")
        self.assertEqual(calls["short"].await_count, 1)
        self.assertEqual(calls["fleet"].await_count, 1)

    def test_routing_writes_land_in_the_fixture_database_not_the_repository(self):
        """The loop must only ever touch the temp DB the class chdir'd into."""
        self.seed_row(age_sec=2.0, regime="CHOP", adx=18.0)
        self._run_one_cycle()
        self.assertEqual(os.path.realpath(os.getcwd()), os.path.realpath(self.tmp_dir))
        with sqlite3.connect(os.path.join(self.tmp_dir, "trading_data.db")) as conn:
            telemetry_rows = conn.execute("SELECT COUNT(*) FROM super_bot_telemetry").fetchone()[0]
        self.assertGreaterEqual(telemetry_rows, 1, "routing loop did not write to the fixture DB")


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
tests/test_swing_event_gate.py

Event-driven macro gate tests for SwingArchitectTrader._check_macro_safety().

HISTORY / WHY THIS FILE WAS REWRITTEN (2026-10-13): the previous version connected to the
LIVE trading_data.db in the repository working directory and executed INSERT OR REPLACE on
macro_global_killswitch, ai_macro_bias, market_regime_label, ai_event_playbook and
ai_event_summary, then "cleaned up" only ai_event_playbook. Running it against a live bot left
the killswitch/macro bias/label/summary rows overwritten for the running engines.

This version never opens the live DB: it chdirs into a throwaway temp directory, seeds a fresh
SQLite file there, and imports swing_trader only after the directory switch (so any relative
path opened at import time also lands in the temp dir).

The two fail-open cases are CHARACTERIZATION assertions, not endorsements: they document that
`_read_macro()` defaults a MISSING row to the benign value instead of failing closed, and that
the macro table is read without any freshness/timestamp check.
"""

import asyncio
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)


def _set_state(db_path, mapping):
    with sqlite3.connect(db_path) as conn:
        for k, v in mapping.items():
            conn.execute(
                "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                (k, str(v)),
            )
        conn.commit()


class SwingEventGateTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="swing_event_gate_")
        self.db_path = os.path.join(self.tmpdir, "trading_data.db")
        self._prev_cwd = os.getcwd()
        os.chdir(self.tmpdir)

        # Hard guard: refuse to continue if the switch did not take effect.
        if os.path.abspath(self.db_path) != os.path.abspath(os.path.join(os.getcwd(), "trading_data.db")):
            os.chdir(self._prev_cwd)
            raise RuntimeError("temp-DB isolation failed; refusing to run against the live DB")

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE system_state (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)"
            )
            conn.commit()

        from swing_trader import SwingArchitectTrader  # noqa: E402  (imported after chdir)
        self.architect = SwingArchitectTrader()

    def tearDown(self):
        os.chdir(self._prev_cwd)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _check(self, direction):
        return asyncio.run(self.architect._check_macro_safety(direction=direction))

    # ---------- event playbook behaviour ----------
    def test_secular_short_breakdown_blocks_long_allows_short(self):
        _set_state(self.db_path, {
            "macro_global_killswitch": "0",
            "ai_macro_bias": "BEARISH",
            "market_regime_label": "NEUTRAL",
            "ai_event_playbook": "SECULAR_SHORT_BREAKDOWN",
            "ai_event_summary": "Sustained energy shock",
        })
        long_ok, _ = self._check("LONG")
        short_ok, _ = self._check("SHORT")
        self.assertFalse(long_ok, "longs must be halted during SECULAR_SHORT_BREAKDOWN")
        self.assertTrue(short_ok, "shorts must be authorized during SECULAR_SHORT_BREAKDOWN")

    def test_flash_rebound_trap_blocks_short(self):
        _set_state(self.db_path, {
            "macro_global_killswitch": "0",
            "ai_macro_bias": "BEARISH",
            "market_regime_label": "BEAR_TREND",
            "ai_event_playbook": "FLASH_REBOUND_TRAP",
            "ai_event_summary": "15m flash dump on exchange FUD",
        })
        _long_ok, _ = self._check("LONG")
        short_ok, _ = self._check("SHORT")
        self.assertFalse(short_ok, "shorts must be blocked during FLASH_REBOUND_TRAP to avoid a squeeze")

    def test_killswitch_blocks_both_directions(self):
        _set_state(self.db_path, {
            "macro_global_killswitch": "1",
            "ai_macro_bias": "BULLISH",
            "market_regime_label": "NEUTRAL",
            "ai_event_playbook": "BULL_MOMENTUM_EXPANSION",
            "ai_event_summary": "Killswitch must outrank the bullish event playbook",
        })
        long_ok, long_rsn = self._check("LONG")
        short_ok, _ = self._check("SHORT")
        self.assertFalse(long_ok, "killswitch must veto longs even under BULL_MOMENTUM_EXPANSION")
        self.assertFalse(short_ok, "killswitch must veto shorts too")
        self.assertIsNotNone(long_rsn)

    # ---------- characterization of the fail-open defaults ----------
    def test_missing_rows_default_to_benign_values(self):
        # No system_state rows at all: _read_macro() substitutes kill='0', bias='NEUTRAL',
        # label='NEUTRAL', playbook='NORMAL_OPERATIONS' instead of failing closed.
        long_ok, _ = self._check("LONG")
        self.assertTrue(
            long_ok,
            "CHARACTERIZATION ONLY: an empty macro table currently reads as all-clear (fail-open). "
            "If the gate is changed to fail closed, invert this assertion.",
        )

    def test_stale_rows_are_ignored_no_freshness_check(self):
        # A rows-stamped 2020 macro state is treated identically to a fresh one.
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('macro_global_killswitch', '0', '2020-01-01T00:00:00')"
            )
            conn.execute(
                "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_event_playbook', 'SECULAR_SHORT_BREAKDOWN', '2020-01-01T00:00:00')"
            )
            conn.commit()
        long_ok, _ = self._check("LONG")
        self.assertFalse(long_ok, "4-year-old playbook is still honoured - documents the missing age check")

    def test_live_db_never_opened(self):
        live_db = os.path.join(REPO_DIR, "trading_data.db")
        before = os.path.getmtime(live_db) if os.path.exists(live_db) else None
        _set_state(self.db_path, {"ai_event_playbook": "SECULAR_SHORT_BREAKDOWN", "macro_global_killswitch": "0"})
        self._check("LONG")
        after = os.path.getmtime(live_db) if os.path.exists(live_db) else None
        self.assertEqual(before, after, "the live trading_data.db must not be written by this test")


if __name__ == "__main__":
    unittest.main()

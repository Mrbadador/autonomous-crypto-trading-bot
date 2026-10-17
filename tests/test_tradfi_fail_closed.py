import unittest
import sqlite3
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone, timedelta
import asyncio

import tests.conftest  # noqa: F401 — enforce D-11 quarantine
from conviction_filter import ConvictionBoardroom
import config

class TestTradFiGateFailClosed(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test_tradfi_")
        self.db_path = os.path.join(self.tmp_dir, "trading_data.db")
        self.old_cwd = os.getcwd()
        os.chdir(self.tmp_dir)
        self.boardroom = ConvictionBoardroom()

    def tearDown(self):
        os.chdir(self.old_cwd)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _set_state(self, val, updated_at=None):
        if updated_at is None:
            updated_at = datetime.now(timezone.utc).isoformat()
        fresh_now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS system_state (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
            for db_name in ("trading_data.db", "trading_ledger.db", "live_paper_trades.db", "test_ledger.db"):
                conn.execute(
                    "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, 'OK', ?)",
                    (f"db_integrity_{db_name}", fresh_now),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, '0', ?)",
                    (f"db_integrity_{db_name}_consecutive_errors", fresh_now),
                )
            # also ensure allowed symbols doesn't block ADA/USDT
            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ALLOWED_SYMBOLS', ?, ?)",
                         (json.dumps(["ADA/USDT"]), fresh_now))
            if val is None:
                conn.execute("DELETE FROM system_state WHERE key = 'etf_macro_telemetry'")
            else:
                conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('etf_macro_telemetry', ?, ?)",
                             (val, updated_at))

    async def test_missing_telemetry_blocks_long(self):
        self._set_state(None)
        res = await self.boardroom.get_conviction("ADA/USDT", "long", {})
        self.assertFalse(res.get("execute", True), f"Missing telemetry must produce veto on LONG: {res}")
        self.assertEqual(res.get("stage"), "tradfi_macro_credit_shield")

    async def test_corrupt_json_blocks_long(self):
        self._set_state("{this_is_not_json}")
        res = await self.boardroom.get_conviction("ADA/USDT", "long", {})
        self.assertFalse(res.get("execute", True), f"Corrupt JSON must produce veto on LONG: {res}")
        self.assertEqual(res.get("stage"), "tradfi_macro_credit_shield")

    async def test_stale_telemetry_blocks_long(self):
        stale_ts = (datetime.now(timezone.utc) - timedelta(seconds=22000)).isoformat()
        payload = json.dumps({"veto_long_entries": False, "confidence_weight": 1.0, "rationale": "All good"})
        self._set_state(payload, updated_at=stale_ts)
        res = await self.boardroom.get_conviction("ADA/USDT", "long", {})
        self.assertFalse(res.get("execute", True), f"Stale telemetry (>21600s) must produce veto on LONG: {res}")
        self.assertEqual(res.get("stage"), "tradfi_macro_credit_shield")

    async def test_veto_with_low_confidence_weight_still_blocks_long(self):
        fresh_ts = datetime.now(timezone.utc).isoformat()
        payload = json.dumps({"veto_long_entries": True, "confidence_weight": 0.20, "rationale": "Credit blowout"})
        self._set_state(payload, updated_at=fresh_ts)
        res = await self.boardroom.get_conviction("ADA/USDT", "long", {})
        self.assertFalse(res.get("execute", True), f"veto=True with conf_weight=0.20 must still block LONG: {res}")
        self.assertEqual(res.get("stage"), "tradfi_macro_credit_shield")

if __name__ == "__main__":
    unittest.main()

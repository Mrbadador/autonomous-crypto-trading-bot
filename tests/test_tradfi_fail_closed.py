import unittest
import sqlite3
import json
import os
import shutil
from datetime import datetime, timezone, timedelta
import asyncio

from conviction_filter import ConvictionBoardroom
import config

class TestTradFiGateFailClosed(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.db_path = "trading_data.db"
        self.backup_path = "trading_data.db.test_backup"
        if os.path.exists(self.db_path):
            shutil.copyfile(self.db_path, self.backup_path)
        self.boardroom = ConvictionBoardroom()

    def tearDown(self):
        if os.path.exists(self.backup_path):
            shutil.copyfile(self.backup_path, self.db_path)
            os.remove(self.backup_path)

    def _set_state(self, val, updated_at=None):
        if updated_at is None:
            updated_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS system_state (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
            # also ensure allowed symbols doesn't block ADA/USDT
            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ALLOWED_SYMBOLS', ?, ?)",
                         (json.dumps(["ADA/USDT"]), updated_at))
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

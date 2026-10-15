import unittest
import tempfile
import sqlite3
import json
import os
from datetime import datetime, timezone, timedelta
import asyncio

from conviction_filter import ConvictionBoardroom
import config


class TestGridBoardroomPreFiltersReal(unittest.IsolatedAsyncioTestCase):
    """
    Q-3 FIX: Non-tautological pre-filter test exercising real pre-filter SQL logic
    against an isolated TEMP SQLite database (never live DB).
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_db = os.path.join(self.temp_dir.name, "test_state.db")
        conn = sqlite3.connect(self.temp_db)
        try:
            conn.execute("CREATE TABLE system_state (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
            # Seed ALLOWED_SYMBOLS with ADA/USDT
            now_iso = datetime.now(timezone.utc).isoformat()
            conn.execute("INSERT INTO system_state (key, value, updated_at) VALUES ('ALLOWED_SYMBOLS', ?, ?)",
                         (json.dumps(["ADA/USDT"]), now_iso))
            conn.commit()
        finally:
            conn.close()
        self.boardroom = ConvictionBoardroom(db_path=self.temp_db)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    async def test_bis_crash_veto_blocks_grid_entry(self):
        """Seeds funding_crash_risk = 'CRITICAL' and verifies real SQL reader produces veto."""
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.temp_db)
        try:
            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('funding_crash_risk', 'CRITICAL', ?)", (now_iso,))
            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('etf_macro_telemetry', ?, ?)",
                         (json.dumps({"veto_long_entries": False, "confidence_weight": 1.0}), now_iso))
            conn.commit()
        finally:
            conn.close()

        res = await self.boardroom.get_conviction("ADA/USDT", "long", {"strategy": "Mean Reversion Maker Dip-Buy", "timeframe": "5m", "rsi": 25.0})
        self.assertFalse(res.get("execute", True))
        self.assertEqual(res.get("stage"), "bis_crash_shield")

    async def test_tradfi_shield_veto_blocks_grid_entry(self):
        """Seeds etf_macro_telemetry with veto_long_entries = True and verifies real SQL reader produces veto."""
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.temp_db)
        try:
            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('bis_funding_regime', 'NORMAL', ?)", (now_iso,))
            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('etf_macro_telemetry', ?, ?)",
                         (json.dumps({"veto_long_entries": True, "confidence_weight": 1.0, "rationale": "Credit stress"}), now_iso))
            conn.commit()
        finally:
            conn.close()

        res = await self.boardroom.get_conviction("ADA/USDT", "long", {"strategy": "Mean Reversion Maker Dip-Buy", "timeframe": "5m", "rsi": 25.0})
        self.assertFalse(res.get("execute", True))
        self.assertEqual(res.get("stage"), "tradfi_macro_credit_shield")


if __name__ == "__main__":
    unittest.main()

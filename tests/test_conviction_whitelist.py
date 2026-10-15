import unittest
import json
import sqlite3
import tempfile
import shutil
import os
import asyncio
from unittest.mock import patch

from conviction_filter import ConvictionBoardroom
import config

class TestConvictionWhitelist(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "trading_data.db")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE system_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                )
            """)
        self.old_cwd = os.getcwd()
        os.chdir(self.tmp_dir)

    def tearDown(self):
        os.chdir(self.old_cwd)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_whitelist_intersects_with_config_symbols(self):
        # Insert a mix of legitimate futures and wrong-venue/rogue symbols
        raw_allowed = ["BTC/USDT", "ETH/USDT", "QQQU/USDT", "SOXL/USDT"]
        with sqlite3.connect("trading_data.db") as conn:
            conn.execute(
                "INSERT INTO system_state (key, value, updated_at) VALUES ('ALLOWED_SYMBOLS', ?, datetime('now'))",
                (json.dumps(raw_allowed),)
            )

        boardroom = ConvictionBoardroom()
        
        # BTC/USDT is in config.SYMBOLS -> should pass whitelist stage
        res_btc = asyncio.run(boardroom.get_conviction("BTC/USDT", "BUY", {"strategy": "Test"}))
        self.assertNotEqual(res_btc.get("stage"), "whitelist_gate")

        # QQQU/USDT is NOT in config.SYMBOLS -> should be vetoed by whitelist gate
        res_qqqu = asyncio.run(boardroom.get_conviction("QQQU/USDT", "BUY", {"strategy": "Test"}))
        self.assertEqual(res_qqqu.get("stage"), "whitelist_gate")
        self.assertFalse(res_qqqu.get("execute", True))

    def test_bis_crash_risk_ttl_fresh_blocks(self):
        # Fresh CRITICAL crash risk within 4h should block LONG
        boardroom = ConvictionBoardroom()
        with sqlite3.connect("trading_data.db") as conn:
            conn.execute(
                "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('funding_crash_risk', 'CRITICAL', datetime('now'))"
            )

        res = asyncio.run(boardroom.get_conviction("BTC/USDT", "BUY", {"strategy": "Test"}))
        self.assertEqual(res.get("stage"), "bis_crash_shield")
        self.assertFalse(res.get("execute", True))

    def test_bis_crash_risk_stale_fails_to_normal(self):
        # Stale CRITICAL crash risk (> 4h old) should fail-to-NORMAL and not block at bis_crash_shield
        boardroom = ConvictionBoardroom()
        with sqlite3.connect("trading_data.db") as conn:
            conn.execute(
                "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('funding_crash_risk', 'CRITICAL', '2020-01-01T00:00:00+00:00')"
            )

        res = asyncio.run(boardroom.get_conviction("BTC/USDT", "BUY", {"strategy": "Test"}))
        self.assertNotEqual(res.get("stage"), "bis_crash_shield")

if __name__ == "__main__":
    unittest.main()


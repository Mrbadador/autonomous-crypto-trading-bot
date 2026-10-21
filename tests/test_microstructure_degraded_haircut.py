import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
import tempfile
import os
import sqlite3
import json
import gc

from super_bot import SuperBotTrader
from swing_trader import SwingArchitectTrader
import config

_orig_sqlite_connect = sqlite3.connect

class TestMicrostructureDegradedHaircutReal(unittest.IsolatedAsyncioTestCase):
    """
    Q-3 FIX: Non-tautological test executing actual production methods
    in SuperBotTrader and SwingArchitectTrader to verify the 50% margin haircut
    and veto behavior on degraded microstructure telemetry.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_db = os.path.join(self.temp_dir.name, "test_trading.db")
        conn = _orig_sqlite_connect(self.temp_db)
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS system_state (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
            conn.execute("CREATE TABLE IF NOT EXISTS swing_open_positions (id INTEGER PRIMARY KEY, symbol TEXT, side TEXT)")
            conn.execute("CREATE TABLE IF NOT EXISTS open_positions (id INTEGER PRIMARY KEY, symbol TEXT, side TEXT)")
            conn.execute("CREATE TABLE IF NOT EXISTS asset_locks (symbol TEXT PRIMARY KEY, locked_by TEXT, heartbeat TEXT)")
            conn.commit()
        finally:
            conn.close()

    def tearDown(self):
        gc.collect()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    async def test_super_bot_margin_haircut_degraded_live_method(self):
        """
        Executes _execute_grid_maker_entry on SuperBotTrader with ms_is_degraded=True.
        Proves margin is halved and order placement receives halved quantity.
        """
        bot = SuperBotTrader()
        bot.uta_client = MagicMock()
        # $300 keeps the halved margin above the $6 exchange floor so this test exercises
        # the *halving*, not the veto; sub-$6 veto coverage lives in test_grid_sizing_parity.
        bot.uta_client.fetch_balance = AsyncMock(return_value=300.0)
        bot.uta_client.format_qty_precision = MagicMock(side_effect=lambda sym, q: round(q, 4))
        bot.uta_client.create_post_only_order = AsyncMock(return_value={"orderId": "grid_test_123"})
        bot.uta_client.fetch_order = AsyncMock(return_value={"status": "filled", "filled_qty": 50.0, "deal_avg_price": 1.0})
        bot.uta_client.place_plan_order = AsyncMock(return_value={"planId": "tp_123"})
        bot.uta_client.cancel_order = AsyncMock(return_value=True)

        bot.active_positions = {}
        bot.swing_engine = MagicMock()
        bot.swing_engine._count_fleet_long_positions_sync = MagicMock(return_value=0)
        bot.swing_engine._news_size_multipliers = {}
        bot.swing_engine._check_news_veto = AsyncMock(return_value=(True, "OK"))
        bot.boardroom = MagicMock()
        bot.boardroom.get_conviction = AsyncMock(return_value={"execute": True, "conviction_score": 1.0})

        # 1. Normal execution (ms_is_degraded = False)
        with patch.object(bot, '_get_finrl_strategy_weight', return_value=1.0), \
             patch('super_bot.sqlite3.connect', side_effect=lambda *a, **kw: _orig_sqlite_connect(self.temp_db)):
            await bot._execute_grid_maker_entry(
                symbol="ADA/USDT",
                current_price=1.0,
                rsi=25.0,
                conv_score=1.0,
                side="long",
                ms_is_degraded=False
            )
        
        self.assertTrue(bot.uta_client.create_post_only_order.called)
        normal_qty = bot.uta_client.create_post_only_order.call_args[1].get("qty")
        bot.uta_client.create_post_only_order.reset_mock()

        # 2. Degraded execution (ms_is_degraded = True) -> exactly halved margin and qty
        with patch.object(bot, '_get_finrl_strategy_weight', return_value=1.0), \
             patch('super_bot.sqlite3.connect', side_effect=lambda *a, **kw: _orig_sqlite_connect(self.temp_db)):
            await bot._execute_grid_maker_entry(
                symbol="ADA/USDT",
                current_price=1.0,
                rsi=25.0,
                conv_score=1.0,
                side="long",
                ms_is_degraded=True
            )

        self.assertTrue(bot.uta_client.create_post_only_order.called)
        degraded_qty = bot.uta_client.create_post_only_order.call_args[1].get("qty")

        # degraded_qty should be roughly half of normal_qty
        self.assertAlmostEqual(degraded_qty / normal_qty, 0.50, delta=0.05)

    async def test_swing_trader_margin_haircut_below_min_vetoed_live_method(self):
        """
        Executes open_swing_position on SwingArchitectTrader where margin halved falls below $6.00,
        verifying that the method vetoes and places zero exchange orders.
        """
        trader = SwingArchitectTrader()
        trader.uta_client = MagicMock()
        trader.uta_client.place_order = AsyncMock(return_value={"orderId": "swing_123"})
        # Mock exchange balance as $15.00 so available headroom is small
        trader.uta_client.fetch_balance = AsyncMock(return_value=15.0)
        trader.uta_client._normalize_symbol = MagicMock(return_value="ADAUSDT")
        trader.uta_client.exchange = MagicMock()
        trader.uta_client.exchange.fetch_ticker = AsyncMock(return_value={"last": 1.0, "close": 1.0})
        trader.active_positions = {}
        trader.active_locks = set()
        trader.db_path = self.temp_db

        ind_4h = {
            "close": 1.0,
            "atr": 0.05,
            "adx": 28.0,
            "rsi": 50.0,
            "cloud_top": 0.95,
            "cloud_bottom": 0.90,
            "kijun": 0.98
        }

        # Mock MicrostructureStateCache to return degraded state (empty dict)
        with patch('microstructure_guard.MicrostructureStateCache.get_state', return_value={}), \
             patch('swing_trader.sqlite3.connect', side_effect=lambda *a, **kw: _orig_sqlite_connect(self.temp_db)), \
             patch('ai_engine.tune_exit_geometry', new=AsyncMock(return_value={"suggested_sl": 0.98, "suggested_tp": 1.04, "confidence": 0.5})):
            await trader.open_swing_position("ADA/USDT", ind_4h, direction="LONG", conviction_score=0.75)

        # Confirm no orders were placed on exchange because halved margin was < $6.00 minimum
        self.assertFalse(trader.uta_client.place_order.called)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
import tempfile
import os
import sqlite3
import gc

from super_bot import SuperBotTrader
import config

_orig_sqlite_connect = sqlite3.connect

class TestGridRiskCapReal(unittest.IsolatedAsyncioTestCase):
    """
    Q-3 FIX: Non-tautological test executing actual _execute_grid_maker_entry
    in SuperBotTrader to verify 2.0% equity risk cap clamping and rounding veto.
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

    async def test_grid_risk_cap_clamping_live_method(self):
        """
        Tests that when desired margin would risk more than 2.0% equity,
        _execute_grid_maker_entry clamps margin so actual risk <= 2.0% equity.
        """
        bot = SuperBotTrader()
        bot.uta_client = MagicMock()
        bot.uta_client.fetch_balance = AsyncMock(return_value=900.0) # 2% cap = $18.00 risk
        bot.uta_client.format_qty_precision = MagicMock(side_effect=lambda sym, q: round(q, 4))
        bot.uta_client.create_post_only_order = AsyncMock(return_value={"orderId": "grid_risk_clamp_123"})
        bot.uta_client.fetch_order = AsyncMock(return_value={"status": "filled", "filled_qty": 225.0, "deal_avg_price": 1.0})
        bot.uta_client.place_plan_order = AsyncMock(return_value={"planId": "tp_123"})
        bot.uta_client.cancel_order = AsyncMock(return_value=True)

        bot.active_positions = {}
        bot.swing_engine = MagicMock()
        bot.swing_engine._count_fleet_long_positions_sync = MagicMock(return_value=0)
        bot.swing_engine._news_size_multipliers = {}
        bot.swing_engine._check_news_veto = AsyncMock(return_value=(True, "OK"))
        bot.boardroom = MagicMock()
        bot.boardroom.get_conviction = AsyncMock(return_value={"execute": True, "conviction_score": 1.0})

        # Set large SL (8.0% SL) so max_margin_by_risk = 18.00 / (3 * 0.08) = $75.00
        bot.grid_sl_pct = 0.08
        bot.grid_leverage = 3

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
        order_qty = bot.uta_client.create_post_only_order.call_args[1].get("qty")
        # With clamped margin $75.00 and 3x lev -> notional = $225 -> qty @ $1.00 = 225
        # Risk on 225 qty at 8% SL = 225 * 0.08 = $18.00 (exactly 2% of $900 balance)
        actual_risk = order_qty * 1.0 * 0.08
        self.assertAlmostEqual(actual_risk, 18.0, delta=0.5)

    async def test_post_rounding_step_risk_exceeded_vetoed(self):
        """
        Tests that if contract step rounding causes actual dollar risk to exceed
        max_allowed_post_round_risk (2% * 1.15), the order placement is aborted.
        """
        bot = SuperBotTrader()
        bot.uta_client = MagicMock()
        bot.uta_client.fetch_balance = AsyncMock(return_value=500.0) # 2% = $10.00, +15% tol = $11.50 max risk
        bot.uta_client.format_qty_precision = MagicMock(return_value=1300.0)
        bot.uta_client.create_post_only_order = AsyncMock(return_value={"orderId": "grid_should_not_reach"})

        bot.active_positions = {}
        bot.swing_engine = MagicMock()
        bot.swing_engine._count_fleet_long_positions_sync = MagicMock(return_value=0)
        bot.swing_engine._news_size_multipliers = {}
        bot.swing_engine._check_news_veto = AsyncMock(return_value=(True, "OK"))
        bot.boardroom = MagicMock()
        bot.boardroom.get_conviction = AsyncMock(return_value={"execute": True, "conviction_score": 1.0})

        bot.grid_sl_pct = 0.01 # 1.0% SL -> 1300 * 1.0 * 0.01 = $13.00 risk
        bot.grid_leverage = 3

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

        # Confirm order was vetoed due to post-rounding risk check
        self.assertFalse(bot.uta_client.create_post_only_order.called)


if __name__ == "__main__":
    unittest.main()

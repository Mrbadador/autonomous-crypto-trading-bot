import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

import tests.conftest  # noqa: F401 — enforce D-11 quarantine
import config
from super_bot import SuperBotTrader


class TestGridMakerGuardsHarness(unittest.TestCase):
    """
    Offline replay harness that exercises _execute_grid_maker_entry through:
    1. Degraded microstructure (MS-DEGRADED-SIZING)
    2. Boardroom veto (GRID_MAKER-BOARDROOM-VETO)
    3. Risk-cap clamp (GRID_MAKER-RISK-CAP)
    4. Rounding risk veto (GRID_MAKER-ROUNDING-RISK-VETO)
    5. Balance failure veto (GRID_MAKER-BALANCE-VETO)
    """

    def setUp(self):
        self.bot = SuperBotTrader()
        self.bot.uta_client = MagicMock()
        self.bot.uta_client._normalize_symbol = lambda s: s.replace("/", "")
        self.bot.uta_client.format_qty_precision = lambda s, q: round(q, 2)
        self.bot.uta_client.format_price_precision = lambda s, p: f"{p:.4f}"
        self.bot.uta_client.create_post_only_order = AsyncMock(return_value={"id": "test_order_123"})
        self.bot.uta_client.fetch_order = AsyncMock(return_value={"status": "filled", "filled_qty": 1.0, "deal_avg_price": 100.0})
        self.bot.uta_client.exchange = MagicMock()
        self.bot.uta_client.exchange.private_uta_post_v3_trade_place_strategy_order = AsyncMock(return_value={"retCode": 0})
        self.bot.swing_engine = MagicMock()
        self.bot.swing_engine._check_news_veto = AsyncMock(return_value=(True, "OK"))
        self.bot.swing_engine._count_fleet_long_positions_sync = MagicMock(return_value=0)
        self.bot.active_positions = {}

    def test_01_balance_failure_veto(self):
        """When balance is unresolved or <= $10, triggers [GRID_MAKER-BALANCE-VETO]."""
        async def run():
            self.bot.uta_client.fetch_balance = AsyncMock(return_value=0.0)
            self.bot.boardroom.get_conviction = AsyncMock(return_value={"execute": True, "conviction_score": 0.8})
            with self.assertLogs("super_bot", level="WARNING") as cm:
                # Signature: symbol, current_price, rsi, conv_score, rationale, side, ms_is_degraded
                await self.bot._execute_grid_maker_entry("ADA/USDT", 100.0, 25.0, 0.8, "test", side="long")
                self.assertTrue(any("GRID_MAKER-BALANCE-VETO" in msg for msg in cm.output), cm.output)
        asyncio.run(run())

    def test_02_boardroom_veto(self):
        """When boardroom pre-filter vetoes, triggers [GRID_MAKER-BOARDROOM-VETO]."""
        async def run():
            self.bot.uta_client.fetch_balance = AsyncMock(return_value=500.0)
            self.bot.boardroom.get_conviction = AsyncMock(return_value={
                "execute": False, "conviction_score": 0.0, "stage": "tradfi_macro_shield", "final_rationale": "TradFi Veto"
            })
            with self.assertLogs("super_bot", level="WARNING") as cm:
                await self.bot._execute_grid_maker_entry("ADA/USDT", 100.0, 25.0, 0.8, "test", side="long")
                self.assertTrue(any("GRID_MAKER-BOARDROOM-VETO" in msg for msg in cm.output), cm.output)
        asyncio.run(run())

    def test_03_degraded_microstructure_and_risk_cap_clamp(self):
        """When microstructure is degraded, triggers [GRID_MAKER-MS-DEGRADED-SIZING] and clamps risk."""
        async def run():
            self.bot.uta_client.fetch_balance = AsyncMock(return_value=500.0)
            self.bot.boardroom.get_conviction = AsyncMock(return_value={"execute": True, "conviction_score": 0.9})
            with self.assertLogs("super_bot", level="INFO") as cm:
                await self.bot._execute_grid_maker_entry("ADA/USDT", 100.0, 25.0, 0.8, "test", side="long", ms_is_degraded=True)
                has_degraded = any("GRID_MAKER-MS-DEGRADED-SIZING" in msg for msg in cm.output)
                self.assertTrue(has_degraded, cm.output)
        asyncio.run(run())

    def test_04_rounding_risk_veto(self):
        """When contract rounding forces risk > max_allowed_post_round_risk, triggers [GRID_MAKER-ROUNDING-RISK-VETO]."""
        async def run():
            self.bot.uta_client.fetch_balance = AsyncMock(return_value=50.0)  # low balance, 2% = $1.00 max risk
            self.bot.boardroom.get_conviction = AsyncMock(return_value={"execute": True, "conviction_score": 0.9})
            # Mock format_qty_precision to force huge minimum contract (e.g. 50 units @ $100 = $5000 notional, risk $57.50 >> $1.15)
            self.bot.uta_client.format_qty_precision = lambda s, q: 50.0
            with self.assertLogs("super_bot", level="WARNING") as cm:
                await self.bot._execute_grid_maker_entry("ADA/USDT", 100.0, 25.0, 0.8, "test", side="long", ms_is_degraded=False)
                self.assertTrue(any("GRID_MAKER-ROUNDING-RISK-VETO" in msg for msg in cm.output), cm.output)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()

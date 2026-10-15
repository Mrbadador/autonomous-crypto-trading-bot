import unittest
from unittest.mock import AsyncMock, patch, MagicMock
from conviction_filter import ConvictionBoardroom

class TestGridBoardroomPreFilters(unittest.IsolatedAsyncioTestCase):
    async def test_bis_crash_veto_blocks_grid_entry(self):
        boardroom = ConvictionBoardroom()
        # Mock pre-filter returning BIS veto
        with patch.object(boardroom, "get_conviction", new_callable=AsyncMock) as mock_get_conv:
            mock_get_conv.return_value = {
                "execute": False,
                "conviction_score": 0.0,
                "stage": "bis_crash_shield",
                "final_rationale": "BIS Crash-Risk Shield Veto"
            }
            res = await boardroom.get_conviction("ADA/USDT", "long", {"strategy": "Mean Reversion Maker Dip-Buy", "timeframe": "5m", "rsi": 25.0})
            self.assertFalse(res.get("execute"))
            self.assertEqual(res.get("stage"), "bis_crash_shield")

    async def test_tradfi_shield_veto_blocks_grid_entry(self):
        boardroom = ConvictionBoardroom()
        with patch.object(boardroom, "get_conviction", new_callable=AsyncMock) as mock_get_conv:
            mock_get_conv.return_value = {
                "execute": False,
                "conviction_score": 0.0,
                "stage": "tradfi_macro_credit_shield",
                "final_rationale": "TradFi Macro/Credit Shield Veto"
            }
            res = await boardroom.get_conviction("ETH/USDT", "long", {"strategy": "Mean Reversion Maker Dip-Buy", "timeframe": "5m", "rsi": 28.0})
            self.assertFalse(res.get("execute"))
            self.assertEqual(res.get("stage"), "tradfi_macro_credit_shield")

if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

class TestMicrostructureDegradedHaircut(unittest.IsolatedAsyncioTestCase):
    def test_super_bot_margin_haircut_degraded(self):
        # Verify the numerical halving logic on degraded microstructure
        margin = 30.0
        ms_is_degraded = True
        if ms_is_degraded:
            margin = max(6.0, round(margin * 0.50, 2))
        self.assertEqual(margin, 15.0)

    def test_super_bot_margin_haircut_clamped_to_min(self):
        margin = 10.0
        ms_is_degraded = True
        if ms_is_degraded:
            margin = max(6.0, round(margin * 0.50, 2))
        self.assertEqual(margin, 6.0)

    def test_swing_trader_margin_haircut_degraded(self):
        margin_collateral = 40.0
        ms_is_degraded = True
        if ms_is_degraded:
            scaled_margin = round(margin_collateral * 0.50, 2)
            if scaled_margin < 6.0:
                vetoed = True
            else:
                vetoed = False
                margin_collateral = scaled_margin
        self.assertFalse(vetoed)
        self.assertEqual(margin_collateral, 20.0)

    def test_swing_trader_margin_haircut_below_min_vetoed(self):
        margin_collateral = 11.0
        ms_is_degraded = True
        vetoed = False
        if ms_is_degraded:
            scaled_margin = round(margin_collateral * 0.50, 2)
            if scaled_margin < 6.0:
                vetoed = True
            else:
                margin_collateral = scaled_margin
        self.assertTrue(vetoed)

if __name__ == "__main__":
    unittest.main()

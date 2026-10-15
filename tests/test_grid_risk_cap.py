import unittest

class TestGridRiskCap(unittest.TestCase):
    def test_grid_risk_cap_clamping(self):
        # Account balance $900, 2.0% risk cap, 3x leverage, 1.2% SL
        live_bal = 900.0
        max_risk_pct = 0.02
        max_risk_usd = live_bal * max_risk_pct # $18.00
        grid_leverage = 3.0
        grid_sl_pct = 0.012 # 1.2%
        max_margin_by_risk = max_risk_usd / (grid_leverage * grid_sl_pct)
        # 18.00 / (3 * 0.012) = 18.00 / 0.036 = $500.00
        self.assertAlmostEqual(max_margin_by_risk, 500.0)

        # But if SL is larger, say 8% SL:
        large_sl_pct = 0.08
        max_margin_large_sl = max_risk_usd / (grid_leverage * large_sl_pct)
        # 18.00 / (3 * 0.08) = 18.00 / 0.24 = $75.00
        self.assertAlmostEqual(max_margin_large_sl, 75.0)

        # Desired margin $100 -> must clamp to $75.00
        desired_margin = 100.0
        clamped_margin = min(desired_margin, round(max_margin_large_sl, 2))
        self.assertEqual(clamped_margin, 75.0)

    def test_post_rounding_step_risk_exceeded(self):
        live_bal = 500.0
        max_risk_pct = 0.02
        max_allowed_risk = live_bal * max_risk_pct * 1.15 # $10 * 1.15 = $11.50

        # Suppose contract step forced qty higher so actual risk is $12.50
        actual_risk_usd = 12.50
        is_vetoed = actual_risk_usd > max_allowed_risk
        self.assertTrue(is_vetoed)

    def test_post_rounding_step_risk_tolerated(self):
        live_bal = 500.0
        max_risk_pct = 0.02
        max_allowed_risk = live_bal * max_risk_pct * 1.15 # $11.50

        # Actual risk is $11.00 (within 15% tolerance of $10 cap)
        actual_risk_usd = 11.00
        is_vetoed = actual_risk_usd > max_allowed_risk
        self.assertFalse(is_vetoed)

if __name__ == "__main__":
    unittest.main()

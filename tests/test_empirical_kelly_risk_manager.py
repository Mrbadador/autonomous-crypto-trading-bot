import unittest
from unittest.mock import patch
from risk_manager import RiskManager

class TestEmpiricalKellyRiskManager(unittest.TestCase):
    def test_calculate_empirical_kelly_fraction_positive(self):
        rm = RiskManager()
        # Mock historical return series with positive expectancy
        mock_returns = [0.03, -0.015, 0.04, -0.015, 0.03, 0.02, -0.01, 0.03, -0.02, 0.04]
        with patch.object(rm, "_get_historical_returns", return_value=mock_returns):
            kelly_f = rm.calculate_empirical_kelly_fraction()
            self.assertGreater(kelly_f, 0.0)
            self.assertLessEqual(kelly_f, 0.25)

    def test_calculate_empirical_kelly_fraction_negative(self):
        rm = RiskManager()
        # Mock historical return series with severe losses
        mock_returns = [-0.03, -0.02, -0.04, -0.01, -0.03, -0.02, -0.05]
        with patch.object(rm, "_get_historical_returns", return_value=mock_returns):
            kelly_f = rm.calculate_empirical_kelly_fraction()
            self.assertEqual(kelly_f, 0.0)

if __name__ == "__main__":
    unittest.main()

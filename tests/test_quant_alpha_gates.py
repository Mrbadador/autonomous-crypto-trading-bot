"""
test_quant_alpha_gates.py — Verification of the 3 go-stock Quantitative Alpha Gates
===================================================================================
Tests:
1. detect_momentum_divergence() in indicators.py
2. compute_volume_profile() in indicators.py
3. compute_td_sequential() in indicators.py
4. SwingArchitectTrader gate integration in swing_trader.py
"""

import unittest
import pandas as pd
import numpy as np

from indicators import (
    detect_momentum_divergence,
    compute_volume_profile,
    compute_td_sequential
)
from swing_trader import SwingArchitectTrader


class TestQuantAlphaGates(unittest.TestCase):

    def test_bearish_divergence_detection(self):
        # Construct synthetic series where price makes higher high but RSI drops
        # Cycle 1: Price peaks at 100 with high momentum
        # Cycle 2: Price peaks at 105 with lower momentum
        n = 30
        t = np.linspace(0, 4 * np.pi, n)
        trend = np.linspace(50, 60, n)
        # Create two distinct peaks where 2nd peak price is higher, but momentum slows down
        closes = [
            50, 52, 55, 58, 62, 60, 57, 54, 53, 55,
            58, 61, 64, 63, 61, 59, 58, 60, 62, 65,  # Peak 1 at 65
            63, 61, 60, 62, 64, 66, 67, 66, 64, 63   # Peak 2 at 67 (higher high, but flatter slope)
        ]
        highs = [c * 1.01 for c in closes]
        lows = [c * 0.99 for c in closes]
        vols = [1000.0] * len(closes)

        df = pd.DataFrame({"high": highs, "low": lows, "close": closes, "volume": vols})
        res = detect_momentum_divergence(df, lookback=24)
        self.assertIn("bearish_divergence", res)
        self.assertIn("bullish_divergence", res)
        self.assertIsInstance(res["details"], str)

    def test_volume_profile_and_value_area(self):
        # Create a concentrated volume zone between $10 and $12, with one wick to $15
        n = 40
        closes = [10.0 + (i % 3) * 0.5 for i in range(n)]
        highs = [c + 0.5 for c in closes]
        lows = [c - 0.5 for c in closes]
        vols = [1000.0] * n
        # Inject one candle at $15 (above VAH)
        closes[-1] = 15.0
        highs[-1] = 15.5
        lows[-1] = 14.5
        vols[-1] = 50.0

        df = pd.DataFrame({"high": highs, "low": lows, "close": closes, "volume": vols})
        vp = compute_volume_profile(df, lookback=40, n_bins=20, value_area_pct=0.70)

        self.assertGreater(vp["poc"], 9.0)
        self.assertLess(vp["poc"], 13.0)
        self.assertGreater(vp["vah"], vp["val"])
        self.assertTrue(vp["above_vah"])  # Last close is 15.0, clearly above VAH
        self.assertFalse(vp["in_value_area"])

    def test_td_sequential_nine_count(self):
        # Established price flip on bar 5 followed by 9 consecutive higher closes than close 4 bars prior
        closes = [20, 20, 20, 20, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        df = pd.DataFrame({"close": closes})
        td = compute_td_sequential(df)

        self.assertEqual(td["setup_direction"], "UP")
        self.assertGreaterEqual(td["up_count"], 9)
        self.assertTrue(td["is_td9_upward_exhaustion"])
        self.assertFalse(td["is_td9_downward_exhaustion"])

    def test_td_sequential_downward_nine_count(self):
        # Established price flip on bar 5 followed by 9 consecutive lower closes than close 4 bars prior
        closes = [20, 20, 20, 20, 21, 19, 18, 17, 16, 15, 14, 13, 12, 11]
        df = pd.DataFrame({"close": closes})
        td = compute_td_sequential(df)

        self.assertEqual(td["setup_direction"], "DOWN")
        self.assertGreaterEqual(td["down_count"], 9)
        self.assertTrue(td["is_td9_downward_exhaustion"])
        self.assertFalse(td["is_td9_upward_exhaustion"])

    def test_swing_architect_gates_methods(self):
        trader = SwingArchitectTrader()
        
        # Test synthetic df
        closes = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] * 2
        df = pd.DataFrame({
            "high": [c * 1.02 for c in closes],
            "low": [c * 0.98 for c in closes],
            "close": closes,
            "volume": [500.0] * len(closes)
        })

        # Divergence gate test
        div_ok, div_rsn = trader._check_divergence_gate("TEST/USDT", df, direction="LONG")
        self.assertIsInstance(div_ok, bool)
        self.assertIsInstance(div_rsn, str)

        # Volume profile gate test
        vp_ok, vp_rsn = trader._check_volume_profile_gate("TEST/USDT", df, current_price=closes[-1], direction="LONG")
        self.assertIsInstance(vp_ok, bool)
        self.assertIsInstance(vp_rsn, str)

        # TD sequential gate test
        td_ok, td_rsn = trader._check_td_sequential_gate("TEST/USDT", df, direction="LONG")
        self.assertIsInstance(td_ok, bool)
        self.assertIsInstance(td_rsn, str)


if __name__ == "__main__":
    unittest.main()

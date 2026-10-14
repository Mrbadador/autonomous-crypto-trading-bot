"""
tests/test_asset_scout_bitget_parity.py
Unit tests verifying Phase 1:
1. Bitget V2 endpoints integration (Top pairs & OHLCV formatting)
2. Lo-MacKinlay Heteroskedasticity-Robust Variance Ratio Test mathematical accuracy
3. Variance-Ratio to Hurst exponent mapping invariants
"""

import os
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from asset_scout import (
    get_top_futures_pairs,
    fetch_futures_ohlcv,
    calculate_variance_ratio,
    calculate_hurst
)


class TestAssetScoutBitgetParity(unittest.TestCase):

    def test_bitget_top_pairs_fetch(self):
        pairs = get_top_futures_pairs(limit=10)
        self.assertIsInstance(pairs, list)
        self.assertGreaterEqual(len(pairs), 5)
        # Verify symbol format (e.g. BTCUSDT)
        for p in pairs:
            self.assertTrue(p.endswith("USDT"), f"Symbol {p} should end with USDT")

    def test_bitget_ohlcv_fetch(self):
        df = fetch_futures_ohlcv("BTCUSDT", interval="15m", limit=30)
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreaterEqual(len(df), 20)

        expected_cols = ["open_time", "open", "high", "low", "close", "volume", "quote_asset_volume"]
        for c in expected_cols:
            self.assertIn(c, df.columns)

        # Verify ascending chronological order
        times = df["open_time"].values
        self.assertTrue(np.all(np.diff(times) > 0), "Timestamps must be sorted in ascending order")

        # Verify positive non-zero prices
        self.assertTrue((df["close"] > 0).all(), "All close prices must be positive")

    def test_variance_ratio_synthetic_series(self):
        # 1. Random Walk: VR should be close to 1.0, |z| small
        np.random.seed(42)
        rw = np.exp(np.cumsum(np.random.normal(0, 0.01, 300)))
        vr_rw, z_rw = calculate_variance_ratio(rw, q=10)
        self.assertAlmostEqual(vr_rw, 1.0, delta=0.25)
        self.assertLess(abs(z_rw), 1.96)

        # 2. Strong Mean-Reversion (Ornstein-Uhlenbeck / stationary process): VR < 1.0, z negative
        mr = [100.0]
        for _ in range(300):
            mr.append(mr[-1] + 0.35 * (100.0 - mr[-1]) + np.random.normal(0, 1.0))
        vr_mr, z_mr = calculate_variance_ratio(np.array(mr), q=10)
        self.assertLess(vr_mr, 0.6)
        self.assertLess(z_mr, -1.96, "Mean reverting process must have statistically significant negative z-score")

        # 3. Hurst conversion
        h_rw = calculate_hurst(rw)
        h_mr = calculate_hurst(np.array(mr))
        self.assertGreater(h_rw, h_mr, "Random walk Hurst must be higher than mean-reverting series")
        self.assertLess(h_mr, 0.50, "Mean-reverting series Hurst must be below 0.50")


if __name__ == "__main__":
    unittest.main()

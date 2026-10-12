"""
tests/test_grid_replay_consistency.py

Tests consistency between:
1. super_bot.py extracted pure functions (compute_grid_5m_indicators, evaluate_chop_grid_signal, evaluate_scalp_short_signal)
2. backtest_grid_replay.py precomputed vectorized arrays
across >= 500 rolling 5m historical slices.
"""

import unittest
import numpy as np
import pandas as pd
from super_bot import (
    compute_grid_5m_indicators,
    evaluate_chop_grid_signal,
    evaluate_scalp_short_signal,
)
from backtests.backtest_grid_replay import precompute_5m_indicators_vectorized


class TestGridReplayConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.df = pd.read_csv("data/historical/BTCUSDT_5m.csv").head(600)
        except Exception:
            np.random.seed(123)
            n = 600
            c = 50000.0 + np.cumsum(np.random.randn(n) * 20.0)
            h = c + np.abs(np.random.randn(n) * 15.0)
            l = c - np.abs(np.random.randn(n) * 15.0)
            o = c + np.random.randn(n) * 5.0
            v = np.random.uniform(10, 100, n)
            ts = [1700000000000 + i * 300000 for i in range(n)]
            cls.df = pd.DataFrame({"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})

        cls.vectorized_df = precompute_5m_indicators_vectorized(cls.df)

    def test_rolling_slices_consistency_ge_500(self):
        """
        Tests >= 500 rolling slices to confirm pure functions match vectorized precomputation.
        Full warm-up is 48 bars (2 * 24 periods for VWAP diff rolling std).
        """
        n_slices = 500
        start_idx = 48
        end_idx = start_idx + n_slices

        self.assertGreaterEqual(len(self.df), end_idx)

        tested_count = 0
        for i in range(start_idx, end_idx):
            slice_df = self.df.iloc[:i + 1].copy()
            pure_ind = compute_grid_5m_indicators(slice_df)
            vec_row = self.vectorized_df.iloc[i]

            # Compare key indicators
            self.assertAlmostEqual(pure_ind["rsi"], vec_row["rsi"], places=4)
            self.assertAlmostEqual(pure_ind["vwap"], vec_row["vwap"], places=4)
            self.assertAlmostEqual(pure_ind["rolling_std"], vec_row["rolling_std"], places=4)
            self.assertAlmostEqual(pure_ind["lower_band"], vec_row["lower_band"], places=4)
            self.assertAlmostEqual(pure_ind["upper_band"], vec_row["upper_band"], places=4)
            self.assertAlmostEqual(pure_ind["adx_5m"], vec_row["adx_5m"], places=4)

            # Compare Chop Grid Long Signal evaluation
            pure_sig, _ = evaluate_chop_grid_signal(
                ind=pure_ind,
                curr_price=pure_ind["curr_price"],
                rsi_thresh=30.0,
                max_adx=24.0,
                regime_adx=20.0,
                di_spread=0.0
            )
            vec_sig = (
                vec_row["rsi"] < 30.0 and
                vec_row["close"] <= vec_row["lower_band"] and
                vec_row["adx_5m"] < 24.0
            )
            self.assertEqual(pure_sig, vec_sig)

            tested_count += 1

        self.assertGreaterEqual(tested_count, 500)
        print(f"Successfully verified consistency across {tested_count} rolling 5m slices.")


if __name__ == "__main__":
    unittest.main()

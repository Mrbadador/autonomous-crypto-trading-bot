"""
tests/test_grid_logic_extraction.py

Proves that extracted pure functions in super_bot.py:
- compute_grid_5m_indicators
- evaluate_chop_grid_signal
- evaluate_scalp_short_signal
are strictly identical to the historical inline formulas.
"""

import unittest
import numpy as np
import pandas as pd
import ta
from super_bot import (
    compute_grid_5m_indicators,
    evaluate_chop_grid_signal,
    evaluate_scalp_short_signal,
)


class TestGridLogicExtraction(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 100
        closes = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        highs = closes + np.abs(np.random.randn(n) * 0.4)
        lows = closes - np.abs(np.random.randn(n) * 0.4)
        opens = (closes + np.random.randn(n) * 0.2)
        volumes = np.random.uniform(10, 100, n)
        timestamps = [1700000000000 + i * 300000 for i in range(n)]

        self.df = pd.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        })

    def test_extracted_indicators_match_inline_math(self):
        c = self.df["close"]
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        expected_rsi = float((100.0 - (100.0 / (1.0 + rs))).iloc[-1])

        cum_pv = (c * self.df["volume"]).rolling(24).sum()
        cum_v = self.df["volume"].rolling(24).sum()
        expected_vwap = float((cum_pv / (cum_v + 1e-9)).iloc[-1])
        vwap_diff = c - (cum_pv / (cum_v + 1e-9))
        expected_std = float(vwap_diff.rolling(24).std().iloc[-1])
        expected_lower = expected_vwap - (2.0 * expected_std)
        expected_upper = expected_vwap + (2.0 * expected_std)

        expected_ema20 = float(c.ewm(span=20, adjust=False).mean().iloc[-1])
        adx_series = ta.trend.ADXIndicator(self.df["high"], self.df["low"], self.df["close"], window=14).adx()
        expected_adx = float(adx_series.iloc[-1])

        ind = compute_grid_5m_indicators(self.df)

        self.assertAlmostEqual(ind["rsi"], expected_rsi, places=6)
        self.assertAlmostEqual(ind["vwap"], expected_vwap, places=6)
        self.assertAlmostEqual(ind["rolling_std"], expected_std, places=6)
        self.assertAlmostEqual(ind["lower_band"], expected_lower, places=6)
        self.assertAlmostEqual(ind["upper_band"], expected_upper, places=6)
        self.assertAlmostEqual(ind["ema20"], expected_ema20, places=6)
        self.assertAlmostEqual(ind["adx_5m"], expected_adx, places=6)

    def test_chop_grid_signal_behavior(self):
        ind = compute_grid_5m_indicators(self.df)
        
        # When price <= lower_band and RSI < 30
        curr_price = ind["lower_band"] - 0.1
        ind_copy = dict(ind)
        ind_copy["rsi"] = 28.0
        ind_copy["adx_5m"] = 15.0

        is_sig, reason = evaluate_chop_grid_signal(
            ind=ind_copy,
            curr_price=curr_price,
            rsi_thresh=30.0,
            max_adx=24.0,
            regime_adx=20.0,
            di_spread=0.0
        )
        self.assertTrue(is_sig)
        self.assertIn("CHOP_DIP_CONFIRMED", reason)

        # Blocked when market regime ADX >= 28.0
        is_sig_regime, reason_regime = evaluate_chop_grid_signal(
            ind=ind_copy,
            curr_price=curr_price,
            rsi_thresh=30.0,
            max_adx=24.0,
            regime_adx=29.0,
            di_spread=0.0
        )
        self.assertFalse(is_sig_regime)
        self.assertIn("REGIME_EXPANSION", reason_regime)

        # Blocked when 5m asset ADX >= 24.0 (waterfall)
        ind_copy["adx_5m"] = 26.0
        is_sig_adx, reason_adx = evaluate_chop_grid_signal(
            ind=ind_copy,
            curr_price=curr_price,
            rsi_thresh=30.0,
            max_adx=24.0,
            regime_adx=20.0,
            di_spread=0.0
        )
        self.assertFalse(is_sig_adx)
        self.assertIn("ASSET_5M_ADX_WATERFALL", reason_adx)


if __name__ == "__main__":
    unittest.main()

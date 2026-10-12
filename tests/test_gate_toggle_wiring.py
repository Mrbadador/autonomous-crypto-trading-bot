"""
tests/test_gate_toggle_wiring.py — Unit test verifying gate toggle wiring.
Proves that toggling each individual gate ON vs OFF directly alters the trade set,
confirming each gate is correctly wired, functional, and non-vacuous.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtests.backtest_ablation_harness import (
    run_ablation_simulation,
    compute_indicators,
    precompute_4h_indicators_vectorized,
    precompute_1d_trend_vectorized
)


class TestGateToggleWiring(unittest.TestCase):
    """
    Tests that toggling each gate ON vs OFF on real and targeted historical fixtures
    produces a concrete change in accepted trade indices, proving wiring integrity.
    """

    @classmethod
    def setUpClass(cls):
        # Prepare authentic resampled 4H and 1D data from the validated ledger
        csv_1h = "data/historical/BTCUSDT_1h_full.csv"
        if not os.path.exists(csv_1h):
            raise unittest.SkipTest(f"Historical source CSV not found: {csv_1h}")

        df_1h = pd.read_csv(csv_1h)
        ts_series = df_1h["timestamp"]
        unit = "ms" if ts_series.iloc[0] > 1e11 else "s"
        df_1h["dt"] = pd.to_datetime(ts_series, unit=unit)
        df_1h = df_1h.set_index("dt").sort_index()

        cls.df_4h = df_1h.resample("4h").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "timestamp": "last"
        }).dropna().reset_index(drop=True)

        cls.df_1d = df_1h.resample("1D").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "timestamp": "last"
        }).dropna().reset_index(drop=True)

        cls.pre_4h = compute_indicators(cls.df_4h, warmup_bars=min(len(cls.df_4h), 200), strict=False)
        cls.vec_4h = precompute_4h_indicators_vectorized(cls.df_4h)
        cls.vec_1d = precompute_1d_trend_vectorized(cls.df_1d)

        cls.all_off = {
            "gate_1d_trend": False,
            "gate_wick_exhaustion": False,
            "gate_divergence": False,
            "gate_td9": False,
            "gate_volume_profile": False
        }

    def test_gate_1d_trend_wiring(self):
        """Gate A (1D Trend): Toggling gate_1d_trend alters trade set significantly."""
        cfg_on = dict(self.all_off)
        cfg_on["gate_1d_trend"] = True

        r_off = run_ablation_simulation(self.df_4h, self.df_1d, self.all_off,
                                        df_4h_precomputed=self.pre_4h, df_4h_vec=self.vec_4h, df_1d_vec=self.vec_1d)
        r_on = run_ablation_simulation(self.df_4h, self.df_1d, cfg_on,
                                       df_4h_precomputed=self.pre_4h, df_4h_vec=self.vec_4h, df_1d_vec=self.vec_1d)

        trades_off = set(t["idx"] for t in r_off["trades"])
        trades_on = set(t["idx"] for t in r_on["trades"])
        diff = trades_off ^ trades_on

        self.assertGreater(len(diff), 0, "gate_1d_trend must alter the trade set when toggled")
        self.assertGreater(r_on["gate_fires"]["gate_1d_trend"], 0, "gate_1d_trend fire count must be > 0")

    def test_gate_wick_exhaustion_wiring(self):
        """Gate B (Wick Exhaustion): Toggling gate_wick_exhaustion filters long upper wicks."""
        cfg_on = dict(self.all_off)
        cfg_on["gate_wick_exhaustion"] = True

        r_off = run_ablation_simulation(self.df_4h, self.df_1d, self.all_off,
                                        df_4h_precomputed=self.pre_4h, df_4h_vec=self.vec_4h, df_1d_vec=self.vec_1d)
        r_on = run_ablation_simulation(self.df_4h, self.df_1d, cfg_on,
                                       df_4h_precomputed=self.pre_4h, df_4h_vec=self.vec_4h, df_1d_vec=self.vec_1d)

        trades_off = set(t["idx"] for t in r_off["trades"])
        trades_on = set(t["idx"] for t in r_on["trades"])
        diff = trades_off ^ trades_on

        self.assertGreater(len(diff), 0, "gate_wick_exhaustion must alter the trade set when toggled")
        self.assertGreater(r_on["gate_fires"]["gate_wick_exhaustion"], 0, "gate_wick_exhaustion fire count must be > 0")

    def test_gate_divergence_wiring(self):
        """Gate C (Momentum Divergence): Toggling gate_divergence alters trade set on divergence setups."""
        cfg_on = dict(self.all_off)
        cfg_on["gate_divergence"] = True

        r_off = run_ablation_simulation(self.df_4h, self.df_1d, self.all_off,
                                        df_4h_precomputed=self.pre_4h, df_4h_vec=self.vec_4h, df_1d_vec=self.vec_1d)
        r_on = run_ablation_simulation(self.df_4h, self.df_1d, cfg_on,
                                       df_4h_precomputed=self.pre_4h, df_4h_vec=self.vec_4h, df_1d_vec=self.vec_1d)

        trades_off = set(t["idx"] for t in r_off["trades"])
        trades_on = set(t["idx"] for t in r_on["trades"])
        diff = trades_off ^ trades_on

        self.assertGreater(len(diff), 0, "gate_divergence must alter the trade set when toggled")
        self.assertGreater(r_on["gate_fires"]["gate_divergence"], 0, "gate_divergence fire count must be > 0")

    def test_gate_td9_wiring(self):
        """Gate D (TD-9 Exhaustion): On window with TD9 exhaustion setup, toggling gate_td9 alters trade set."""
        # Use window [88:200] where bar index 90 has confirmed TD-9 upward exhaustion
        sub_4h = self.df_4h.iloc[88:200].reset_index(drop=True)
        sub_1d = self.df_1d.iloc[14:35].reset_index(drop=True)

        pre = compute_indicators(sub_4h, warmup_bars=min(len(sub_4h), 50), strict=False)
        vec4 = precompute_4h_indicators_vectorized(sub_4h)
        vec1 = precompute_1d_trend_vectorized(sub_1d)

        cfg_on = dict(self.all_off)
        cfg_on["gate_td9"] = True

        r_off = run_ablation_simulation(sub_4h, sub_1d, self.all_off,
                                        df_4h_precomputed=pre, df_4h_vec=vec4, df_1d_vec=vec1)
        r_on = run_ablation_simulation(sub_4h, sub_1d, cfg_on,
                                       df_4h_precomputed=pre, df_4h_vec=vec4, df_1d_vec=vec1)

        trades_off = set(t["idx"] for t in r_off["trades"])
        trades_on = set(t["idx"] for t in r_on["trades"])
        diff = trades_off ^ trades_on

        self.assertGreater(len(diff), 0, "gate_td9 must alter the trade set when toggled on TD9 exhaustion window")
        self.assertGreater(r_on["gate_fires"]["gate_td9"], 0, "gate_td9 fire count must be > 0")

    def test_gate_volume_profile_wiring(self):
        """Gate E (Volume Profile): Toggling gate_volume_profile alters trade set when overhead POC obstructs."""
        cfg_on = dict(self.all_off)
        cfg_on["gate_volume_profile"] = True

        r_off = run_ablation_simulation(self.df_4h, self.df_1d, self.all_off,
                                        df_4h_precomputed=self.pre_4h, df_4h_vec=self.vec_4h, df_1d_vec=self.vec_1d)
        r_on = run_ablation_simulation(self.df_4h, self.df_1d, cfg_on,
                                       df_4h_precomputed=self.pre_4h, df_4h_vec=self.vec_4h, df_1d_vec=self.vec_1d)

        trades_off = set(t["idx"] for t in r_off["trades"])
        trades_on = set(t["idx"] for t in r_on["trades"])
        diff = trades_off ^ trades_on

        self.assertGreater(len(diff), 0, "gate_volume_profile must alter the trade set when toggled")
        self.assertGreater(r_on["gate_fires"]["gate_volume_profile"], 0, "gate_volume_profile fire count must be > 0")


if __name__ == "__main__":
    unittest.main()

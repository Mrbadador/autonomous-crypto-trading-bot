import unittest
import numpy as np
from microstructure_metrics import (
    compute_mlofi_step,
    update_hawkes_state,
    HawkesSpoofGuard,
    calculate_bayesian_smoothed_kelly
)

class TestVectorizedMLOFIAndHawkes(unittest.TestCase):

    def test_compute_mlofi_step_buyer_depth(self):
        """Buyer queue expansion on top 10 levels should yield positive MLOFI."""
        prev_bids = np.zeros((10, 2), dtype=np.float64)
        curr_bids = np.zeros((10, 2), dtype=np.float64)
        prev_asks = np.zeros((10, 2), dtype=np.float64)
        curr_asks = np.zeros((10, 2), dtype=np.float64)

        for k in range(10):
            p_bid = 100.0 - k * 0.1
            p_ask = 100.1 + k * 0.1
            prev_bids[k] = [p_bid, 10.0]
            curr_bids[k] = [p_bid, 25.0]  # Increased bid depth (buyer arrival)
            prev_asks[k] = [p_ask, 10.0]
            curr_asks[k] = [p_ask, 10.0]  # Unchanged ask depth

        ofi_val = compute_mlofi_step(prev_bids, curr_bids, prev_asks, curr_asks, delta_kernel=0.5)
        self.assertGreater(ofi_val, 0.0)
        self.assertLessEqual(ofi_val, 1.0)
        self.assertGreaterEqual(ofi_val, -1.0)

    def test_compute_mlofi_step_seller_depth(self):
        """Seller queue expansion on top 10 levels should yield negative MLOFI."""
        prev_bids = np.zeros((10, 2), dtype=np.float64)
        curr_bids = np.zeros((10, 2), dtype=np.float64)
        prev_asks = np.zeros((10, 2), dtype=np.float64)
        curr_asks = np.zeros((10, 2), dtype=np.float64)

        for k in range(10):
            p_bid = 100.0 - k * 0.1
            p_ask = 100.1 + k * 0.1
            prev_bids[k] = [p_bid, 15.0]
            curr_bids[k] = [p_bid, 15.0]
            prev_asks[k] = [p_ask, 10.0]
            curr_asks[k] = [p_ask, 40.0]  # Increased ask depth (seller pressure)

        ofi_val = compute_mlofi_step(prev_bids, curr_bids, prev_asks, curr_asks, delta_kernel=0.5)
        self.assertLess(ofi_val, 0.0)
        self.assertGreaterEqual(ofi_val, -1.0)

    def test_hawkes_spoof_guard_benign_flow(self):
        """Balanced trading flow with reasonable order lifetimes should NOT veto."""
        guard = HawkesSpoofGuard()
        # 10 additions, 8 cancellations, 5 executions, 500ms residence time, dt=0.5s
        is_vetoed, spoof_idx, rho = guard.evaluate_tick(
            additions=10,
            cancellations=8,
            executions=5,
            avg_residence_sec=0.50,
            dt=0.5
        )
        self.assertFalse(is_vetoed)
        self.assertLess(spoof_idx, 0.72)
        self.assertLess(rho, 1.0)

    def test_hawkes_spoof_guard_predatory_spoofing(self):
        """Rapid additions accompanied by massive rapid cancellations with <100ms lifetime should trigger veto."""
        guard = HawkesSpoofGuard()
        # Simulate predatory flash liquidity: 100 additions, 250 cancellations, 20ms residence time
        is_vetoed = False
        spoof_idx = 0.0
        for _ in range(5):
            is_vetoed, spoof_idx, rho = guard.evaluate_tick(
                additions=100,
                cancellations=250,
                executions=1,
                avg_residence_sec=0.020,  # 20ms lifetime (flash spoofing)
                dt=0.1
            )
        self.assertTrue(is_vetoed)
        self.assertGreater(spoof_idx, 0.78)

    def test_bayesian_smoothed_kelly_low_sample(self):
        """With very few trades, Beta prior (13, 7) should smooth allocation safely."""
        few_returns = [0.02, -0.01, 0.015]
        alloc_f = calculate_bayesian_smoothed_kelly(
            returns=few_returns,
            fraction=0.25,
            max_cap=0.25,
            min_cap=0.03
        )
        # Should return a safe fractional allocation without blowing up
        self.assertGreaterEqual(alloc_f, 0.03)
        self.assertLessEqual(alloc_f, 0.25)

if __name__ == "__main__":
    unittest.main()

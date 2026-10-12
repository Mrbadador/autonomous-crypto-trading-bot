import unittest
import math
from microstructure_metrics import (
    compute_micro_price,
    compute_multi_level_ofi,
    VPINCalculator,
    calculate_empirical_growth_optimal_kelly,
    verify_bitget_crc32_checksum
)

class TestMicrostructureMetrics(unittest.TestCase):
    def test_compute_micro_price_symmetric(self):
        p_micro, bps = compute_micro_price(bid_price=100.0, bid_size=10.0, ask_price=101.0, ask_size=10.0)
        self.assertAlmostEqual(p_micro, 100.5, places=4)
        self.assertAlmostEqual(bps, 0.0, places=2)

    def test_compute_micro_price_asymmetric(self):
        # Heavy bid size pulls price towards ask
        p_micro, bps = compute_micro_price(bid_price=100.0, bid_size=30.0, ask_price=101.0, ask_size=10.0)
        # (100 * 10 + 101 * 30) / 40 = 100.75
        self.assertAlmostEqual(p_micro, 100.75, places=4)
        self.assertGreater(bps, 0.0)

    def test_compute_micro_price_zero_sizes(self):
        p_micro, bps = compute_micro_price(bid_price=100.0, bid_size=0.0, ask_price=101.0, ask_size=0.0)
        self.assertAlmostEqual(p_micro, 100.5, places=4)
        self.assertAlmostEqual(bps, 0.0, places=2)

    def test_compute_multi_level_ofi(self):
        prev_bids = [(100.0, 10.0), (99.0, 20.0), (98.0, 30.0)]
        curr_bids = [(100.0, 15.0), (99.0, 20.0), (98.0, 35.0)] # bid size +5 at lvl 0, +5 at lvl 2
        prev_asks = [(101.0, 10.0), (102.0, 20.0), (103.0, 30.0)]
        curr_asks = [(101.0, 10.0), (102.0, 20.0), (103.0, 30.0)] # no change
        
        ofi = compute_multi_level_ofi(prev_bids, curr_bids, prev_asks, curr_asks, levels=3, decay=0.5)
        # lvl 0: +5 (weight 1.0)
        # lvl 1: 0 (weight 0.5)
        # lvl 2: +5 (weight 0.25)
        # weighted: (5 * 1.0 + 0 * 0.5 + 5 * 0.25) / (1.0 + 0.5 + 0.25) = 6.25 / 1.75 = 3.5714
        self.assertGreater(ofi, 0.0)
        self.assertAlmostEqual(ofi, 6.25 / 1.75, places=3)

    def test_vpin_calculator(self):
        calc = VPINCalculator(bucket_volume=100.0, num_buckets=5)
        # Feed trades
        for _ in range(10):
            calc.update_trade(price=50000.0, size=50.0)
            calc.update_trade(price=50010.0, size=50.0)
        vpin = calc.get_vpin()
        self.assertTrue(0.0 <= vpin <= 1.0)

    def test_calculate_empirical_growth_optimal_kelly_positive(self):
        # 60% win rate with 1.5 R:R
        returns = [0.03, -0.02, 0.03, 0.03, -0.02, 0.03, -0.02, 0.03, 0.03, -0.02]
        stake = calculate_empirical_growth_optimal_kelly(returns, fraction=0.25, max_cap=0.25, min_cap=0.05)
        self.assertTrue(0.05 <= stake <= 0.25)

    def test_calculate_empirical_growth_optimal_kelly_negative(self):
        # Severe losing streak
        returns = [-0.03, -0.02, -0.01, -0.02, -0.03, -0.01]
        stake = calculate_empirical_growth_optimal_kelly(returns, fraction=0.25, max_cap=0.25, min_cap=0.05)
        self.assertEqual(stake, 0.0)

    def test_verify_bitget_crc32_checksum(self):
        bids = [[100.0, 1.0], [99.0, 2.0]]
        asks = [[101.0, 1.0], [102.0, 2.0]]
        import zlib
        # Bitget standard: alternating bid:ask pairs across levels
        expected = zlib.crc32(b"100.0:1.0:101.0:1.0:99.0:2.0:102.0:2.0") & 0xFFFFFFFF
        self.assertTrue(verify_bitget_crc32_checksum(bids, asks, expected))
        self.assertFalse(verify_bitget_crc32_checksum(bids, asks, expected + 1))

if __name__ == "__main__":
    unittest.main()

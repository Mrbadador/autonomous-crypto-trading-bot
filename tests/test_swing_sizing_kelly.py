"""Swing sizing (Fix #5): Kelly must actually size the position, not be theater.

The live bug class: floor (0.7 x fair share = 12.25%) sat above the per-trade cap
(.env MAX_MARGIN_PER_TRADE_PCT = 14% of balance), and margin_collateral used
max(6.0, min(target, buffer)) which could exceed remaining headroom. A losing Kelly
history could therefore never shrink a trade, and quantity rounding could round up
into over-margin.
"""
import unittest

from swing_trader import SwingArchitectTrader as SwingTrader


class TestKellyTarget(unittest.TestCase):
    def test_low_kelly_shrinks_below_fair_share(self):
        # fair=21 (70%/4 locks), kelly margin=10 -> Kelly must win, no floor rescues it.
        target = SwingTrader._compute_target_margin(fair_share=21.0, kelly_margin=10.0, per_trade_cap=30.0)
        self.assertAlmostEqual(target, 10.0)

    def test_high_kelly_capped_at_fair_share(self):
        target = SwingTrader._compute_target_margin(fair_share=21.0, kelly_margin=50.0, per_trade_cap=100.0)
        self.assertAlmostEqual(target, 21.0)

    def test_per_trade_cap_binds_last(self):
        target = SwingTrader._compute_target_margin(fair_share=21.0, kelly_margin=100.0, per_trade_cap=14.0)
        self.assertAlmostEqual(target, 14.0)

    def test_fail_closed_when_kelly_unavailable(self):
        target, note = SwingTrader._resolve_swing_target(
            live_bal=100.0, max_wallet_margin_pct=0.70, max_locks=4,
            per_trade_cap_pct=0.14, kelly_frac=None,
            conviction_score=0.7, finrl_mult=1.0, markov_mult=1.0,
        )
        # Minimum viable size = 30% of fair share, not the old fail-open full fair share.
        self.assertAlmostEqual(target, 100.0 * 0.70 / 4 * 0.30)
        self.assertIn("fail-closed", note)

    def test_normal_resolution_applies_multipliers(self):
        target, note = SwingTrader._resolve_swing_target(
            live_bal=100.0, max_wallet_margin_pct=0.70, max_locks=4,
            per_trade_cap_pct=0.14, kelly_frac=0.10,
            conviction_score=1.0, finrl_mult=1.0, markov_mult=1.0,
        )
        self.assertAlmostEqual(target, 10.0)
        self.assertIn("kelly", note)

    def test_collateral_never_exceeds_buffer(self):
        target = SwingTrader._compute_target_margin(fair_share=21.0, kelly_margin=18.0, per_trade_cap=30.0)
        collateral = SwingTrader._collateral_for(target, available_buffer=5.5)
        self.assertAlmostEqual(collateral, 5.5)

    def test_collateral_below_exchange_minimum_is_not_inflated(self):
        collateral = SwingTrader._collateral_for(18.0, available_buffer=2.0)
        self.assertLess(collateral, 6.0)


class TestFloorRounding(unittest.TestCase):
    def test_floor_never_rounds_up(self):
        self.assertEqual(SwingTrader._floor_to_step(1234.7, 1.0), 1234.0)
        self.assertEqual(SwingTrader._floor_to_step(0.00019, 0.0001), 0.0001)
        self.assertEqual(SwingTrader._floor_to_step(3.14159, 0.01), 3.14)
        self.assertEqual(SwingTrader._floor_to_step(2.55, 0.1), 2.5)

    def test_exact_step_values_survive_float_noise(self):
        self.assertEqual(SwingTrader._floor_to_step(0.3, 0.1), 0.3)
        self.assertEqual(SwingTrader._floor_to_step(12.0, 1.0), 12.0)
        self.assertEqual(SwingTrader._floor_to_step(3.0, 1.0), 3.0)


if __name__ == "__main__":
    unittest.main()

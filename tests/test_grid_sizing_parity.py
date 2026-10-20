"""Fix #5 ported to the live grid/scalp engine in super_bot.py.

The swing engine was repaired first, but _execute_grid_maker_entry still had the
identical bug class and it is what actually dispatched orders since restart:
a Kelly floor that collided with the per-trade cap (theater), a log.debug
fail-open on sizing errors, and max(6.0, ...) that could inflate margin above
remaining headroom - and re-inflate the very size-downs meant to shrink it.
"""
import inspect
import textwrap
import unittest

from swing_trader import SwingArchitectTrader as SwingTrader
from swing_trader import EXCHANGE_MIN_MARGIN_USD

OLD_FLOOR = "max(target_per_trade * 0.7"
INFLATING = "max(6.0,"


def _grid_entry_source():
    import super_bot
    return textwrap.dedent(inspect.getsource(super_bot.SuperBotTrader._execute_grid_maker_entry))


class TestGridSizingUsesSharedStatics(unittest.TestCase):
    def test_grid_entry_no_longer_inflates_margin(self):
        src = _grid_entry_source()
        self.assertNotIn(INFLATING, src, "max(6.0, ...) can exceed remaining headroom")
        self.assertIn("_collateral_for", src)

    def test_grid_entry_kelly_is_unfloored(self):
        src = _grid_entry_source()
        self.assertNotIn(OLD_FLOOR, src, "dead 0.7x floor is back; Kelly cannot shrink a trade")
        self.assertIn("_resolve_swing_target", src)

    def test_grid_entry_fails_closed_with_warning(self):
        src = _grid_entry_source()
        self.assertIn("KELLY-SIZING-FAILCLOSED", src)
        self.assertIn("MIN-MARGIN-VETO", src)


class TestHaircutCanOnlyShrink(unittest.TestCase):
    """The shared statics the grid path now leans on, under the multipliers it gets."""

    def test_resolve_target_shrinks_when_kelly_says_so(self):
        small, _ = SwingTrader._resolve_swing_target(
            live_bal=300.0, max_wallet_margin_pct=0.70, max_locks=4,
            per_trade_cap_pct=0.14, kelly_frac=0.01, conviction_score=0.7)
        self.assertLess(small, 300.0 * 0.70 / 4)

    def test_fail_closed_target_is_never_unplaceable(self):
        """Fail-closed must mean smallest viable trade, not a size the exchange rejects.
        At $90, 30% of fair share is $4.73 - under the $6 floor, which would silence
        every engine instead of de-risking it."""
        for bal in (90.0, 339.0, 1000.0):
            target, note = SwingTrader._resolve_swing_target(
                live_bal=bal, max_wallet_margin_pct=0.70, max_locks=4,
                per_trade_cap_pct=0.14, kelly_frac=None, conviction_score=0.9)
            self.assertIn("fail-closed", note)
            self.assertGreaterEqual(target, EXCHANGE_MIN_MARGIN_USD,
                                    f"kelly-less sizing at ${bal} balances is unplaceable (${target})")

    def test_collateral_never_exceeds_headroom_even_below_minimum(self):
        self.assertEqual(SwingTrader._collateral_for(20.0, available_buffer=4.50), 4.50)

    def test_half_haircut_floors_down_not_up(self):
        self.assertEqual(SwingTrader._floor_to_step(12.0999, 0.01), 12.09)
        self.assertLessEqual(SwingTrader._floor_to_step(11.5, 0.01), 11.5)


if __name__ == "__main__":
    unittest.main()

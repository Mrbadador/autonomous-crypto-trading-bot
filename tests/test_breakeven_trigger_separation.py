import inspect
import unittest

import config
import swing_trader


class TestBreakevenTriggerSeparation(unittest.TestCase):
    """Ratchet 0A (scalp short) and Ratchet 0B (grid short) must not share one
    trigger: the grid floor is 1.20%, the scalp-short fast-harvest is 0.75%."""

    @classmethod
    def setUpClass(cls):
        cls.src = inspect.getsource(swing_trader.SwingArchitectTrader.manage_active_positions)

    def test_scalp_short_trigger_exists_in_config(self):
        self.assertTrue(hasattr(config, "SCALP_SHORT_BREAKEVEN_TRIGGER_PCT"))
        self.assertAlmostEqual(config.SCALP_SHORT_BREAKEVEN_TRIGGER_PCT, 0.0075, places=4)

    def test_grid_trigger_is_unchanged(self):
        self.assertGreaterEqual(config.GRID_BREAKEVEN_TRIGGER_PCT, 0.0120)

    def test_ratchet_0b_uses_the_grid_trigger(self):
        grid_branch = self.src.split("Ratchet 0B")[1].split("# Ratchet 1:")[0]
        self.assertIn("peak_profit_pct >= grid_be_trigger", grid_branch)
        self.assertNotIn("short_be_trigger", grid_branch)

    def test_ratchet_0a_uses_the_scalp_short_trigger(self):
        scalp_branch = self.src.split("Ratchet 0A")[1].split("Ratchet 0B")[0]
        self.assertIn("peak_profit_pct >= short_be_trigger", scalp_branch)

    def test_both_triggers_are_derived_from_config(self):
        self.assertIn(
            'getattr(config, "SCALP_SHORT_BREAKEVEN_TRIGGER_PCT", 0.0075)', self.src
        )
        self.assertIn(
            'getattr(config, "GRID_BREAKEVEN_TRIGGER_PCT", 0.0050)', self.src
        )

"""
tests/test_config_hygiene.py

AST-based static validation suite to guarantee configuration hygiene in config.py:
1. Zero module-level duplicate variable assignments.
2. Verified single source of truth for all risk, sizing, and execution parameters.
"""

import ast
import os
import unittest


class TestConfigHygiene(unittest.TestCase):
    def test_zero_module_level_duplicate_assignments_in_config(self):
        """
        Parses config.py and verifies that no top-level configuration variable
        is assigned more than once, preventing silent 'last assignment wins' bugs.
        """
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.py")
        self.assertTrue(os.path.exists(config_path), f"config.py not found at {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source, filename="config.py")

        assignments = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assignments.setdefault(target.id, []).append(node.lineno)

        duplicates = {var: lines for var, lines in assignments.items() if len(lines) > 1}
        self.assertEqual(
            duplicates,
            {},
            f"Found duplicate variable assignments in config.py: {duplicates}. "
            f"Every configuration setting must have exactly one assignment."
        )

    def test_grid_exit_economics_calibrated_values(self):
        """
        Asserts that the critical exit economics parameters in config.py
        have their intended values to protect against premature winner choking.
        """
        import config
        # GRID_BREAKEVEN_TRIGGER_PCT must be at least 1.20% (relaxed from 0.75% to prevent micro-clipping)
        self.assertGreaterEqual(config.GRID_BREAKEVEN_TRIGGER_PCT, 0.0120, "GRID_BREAKEVEN_TRIGGER_PCT must be >= 1.20%")
        # Target TP must exceed SL
        self.assertGreater(config.GRID_TP_PCT, config.GRID_SL_PCT, "GRID_TP_PCT must exceed GRID_SL_PCT")


if __name__ == "__main__":
    unittest.main()

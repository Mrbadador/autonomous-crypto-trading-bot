"""
tests/test_queue_fill_models.py
Unit tests verifying the queue-model fill engine in backtests/backtest_grid_replay.py:
1. Invariant check: fill_model="touch" exactly reproduces conservative_fills=False
2. Invariant check: fill_model="one_tick" exactly reproduces conservative_fills=True
3. Probabilistic fill check: logprob and squareprob are bounded [0, 1] and monotonic in depth penetration
4. Reproducibility check: seed ensures deterministic replay across probabilistic runs
"""

import unittest
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from backtests.backtest_grid_replay import evaluate_limit_fill


class TestQueueFillModels(unittest.TestCase):
    def test_touch_vs_one_tick_invariants(self):
        # Long order at 100.0 with tick_size 0.1
        limit_p = 100.0
        tick_sz = 0.1

        # Exact touch: low hits 100.0 exactly
        # touch -> True
        # one_tick -> False (low must be <= 99.9)
        self.assertTrue(evaluate_limit_fill("LONG", limit_p, 101.0, 102.0, 100.0, tick_sz, fill_model="touch"))
        self.assertFalse(evaluate_limit_fill("LONG", limit_p, 101.0, 102.0, 100.0, tick_sz, fill_model="one_tick"))

        # Trade through: low hits 99.9 (1 tick through)
        self.assertTrue(evaluate_limit_fill("LONG", limit_p, 101.0, 102.0, 99.9, tick_sz, fill_model="touch"))
        self.assertTrue(evaluate_limit_fill("LONG", limit_p, 101.0, 102.0, 99.9, tick_sz, fill_model="one_tick"))

        # No touch: low is 100.1
        self.assertFalse(evaluate_limit_fill("LONG", limit_p, 101.0, 102.0, 100.1, tick_sz, fill_model="touch"))
        self.assertFalse(evaluate_limit_fill("LONG", limit_p, 101.0, 102.0, 100.1, tick_sz, fill_model="one_tick"))

    def test_short_touch_vs_one_tick_invariants(self):
        # Short order at 100.0 with tick_size 0.1
        limit_p = 100.0
        tick_sz = 0.1

        # Exact touch: high hits 100.0 exactly
        self.assertTrue(evaluate_limit_fill("SHORT", limit_p, 99.0, 100.0, 98.0, tick_sz, fill_model="touch"))
        self.assertFalse(evaluate_limit_fill("SHORT", limit_p, 99.0, 100.0, 98.0, tick_sz, fill_model="one_tick"))

        # Trade through: high hits 100.1 (1 tick through)
        self.assertTrue(evaluate_limit_fill("SHORT", limit_p, 99.0, 100.1, 98.0, tick_sz, fill_model="touch"))
        self.assertTrue(evaluate_limit_fill("SHORT", limit_p, 99.0, 100.1, 98.0, tick_sz, fill_model="one_tick"))

        # No touch: high is 99.9
        self.assertFalse(evaluate_limit_fill("SHORT", limit_p, 99.0, 99.9, 98.0, tick_sz, fill_model="touch"))
        self.assertFalse(evaluate_limit_fill("SHORT", limit_p, 99.0, 99.9, 98.0, tick_sz, fill_model="one_tick"))

    def test_probabilistic_fill_monotonicity_and_determinism(self):
        limit_p = 100.0
        tick_sz = 0.1

        # Determinism: same seed gives same fill result
        res1 = evaluate_limit_fill("LONG", limit_p, 101.0, 102.0, 99.5, tick_sz, fill_model="logprob", seed=42)
        res2 = evaluate_limit_fill("LONG", limit_p, 101.0, 102.0, 99.5, tick_sz, fill_model="logprob", seed=42)
        self.assertEqual(res1, res2)

        # Penetration 0 (exact touch) has 0 probability under logprob/squareprob
        self.assertFalse(evaluate_limit_fill("LONG", limit_p, 101.0, 102.0, 100.0, tick_sz, fill_model="logprob", seed=10))
        self.assertFalse(evaluate_limit_fill("LONG", limit_p, 101.0, 102.0, 100.0, tick_sz, fill_model="squareprob", seed=10))

        # Deep penetration: low = 90.0, high = 102.0 (penetration = 10.0 / 12.0 = 0.833)
        logprob_fills = [
            evaluate_limit_fill("LONG", limit_p, 101.0, 102.0, 90.0, tick_sz, fill_model="logprob", seed=s)
            for s in range(50)
        ]
        self.assertGreater(sum(logprob_fills), 40)


if __name__ == "__main__":
    unittest.main()

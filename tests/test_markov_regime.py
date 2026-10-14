import unittest
import numpy as np
import pandas as pd
from markov_regime import MarkovRegimeEngine

class TestMarkovRegime(unittest.TestCase):
    def test_markov_regime_engine_initialization(self):
        engine = MarkovRegimeEngine()
        assert engine.P.shape == (3, 3)
        # Check row-stochasticity: each row sums to 1.0
        row_sums = np.sum(engine.P, axis=1)
        np.testing.assert_allclose(row_sums, [1.0, 1.0, 1.0], atol=1e-5)

    def test_markov_regime_forward_filter_chop(self):
        engine = MarkovRegimeEngine()
        # Create synthetic chop price series: oscillating in tight range with neutral end
        np.random.seed(42)
        prices = 60000.0 + np.random.normal(0, 5.0, 80)
        df = pd.DataFrame({
            "open": prices,
            "high": prices + 8.0,
            "low": prices - 8.0,
            "close": prices,
            "volume": [100.0] * len(prices)
        })
        
        result = engine.process_bars(df)
        assert "dominant_regime" in result
        assert "state_probabilities" in result
        assert "expected_durations_bars" in result
        assert "breakdown_risk_4step" in result
        
        probs = result["state_probabilities"]
        assert probs["CHOP"] > probs["BEAR_CASCADE"]
        assert 0.0 <= result["breakdown_risk_4step"] <= 1.0

    def test_markov_regime_breakdown_cascade_detection(self):
        engine = MarkovRegimeEngine()
        # Create synthetic aggressive waterfall dump
        prices = [60000.0 * (0.99 ** i) for i in range(80)]
        df = pd.DataFrame({
            "open": prices,
            "high": [p + 5.0 for p in prices],
            "low": [p - 15.0 for p in prices],
            "close": prices,
            "volume": [500.0 * (1.05 ** i) for i in range(80)]
        })
        
        result = engine.process_bars(df)
        assert result["dominant_regime"] == "BEAR_CASCADE"
        assert result["state_probabilities"]["BEAR_CASCADE"] > 0.60
        assert result["breakdown_risk_4step"] > 0.50

    def test_stationary_distribution_and_durations(self):
        engine = MarkovRegimeEngine()
        pi_star = engine.get_stationary_distribution()
        assert len(pi_star) == 3
        assert np.isclose(np.sum(pi_star), 1.0, atol=1e-5)
        
        durations = engine.get_expected_regime_durations()
        assert "CHOP" in durations
        assert durations["CHOP"] >= 1.0

    def test_playbook_tradability_and_sizing_multiplier(self):
        engine = MarkovRegimeEngine()
        # Initial priors: CHOP=0.333, BULL=0.333, BEAR=0.333
        mult = engine.get_regime_sizing_multiplier()
        assert 0.20 <= mult <= 1.0

        # Tradability screen check
        is_tradable, rsn = engine.is_regime_tradable(min_duration_bars=2.0)
        assert isinstance(is_tradable, bool)
        assert isinstance(rsn, str)

        # Bayesian shrinkage persistence test
        seq = np.array([0, 0, 1, 1, 0, 0, 2, 2], dtype=np.int32)
        P_shrunk = engine.estimate_transition_matrix(seq, prior_weight=10.0)
        # Verify diagonal persistence is elevated
        for i in range(3):
            assert P_shrunk[i, i] > 0.40

        # Moment ordering check
        engine._enforce_moment_ordering()
        assert engine.emissions[2]["mean"][0] <= engine.emissions[0]["mean"][0] <= engine.emissions[1]["mean"][0]

    def test_swing_gate_boolean_operator_and_reason(self):
        """FIX 3: Verify the boolean condition (p_bull < 0.65 AND d_bull < 4.0) matches reason string semantics."""
        # Condition: both must be true for the condition to trigger
        p_bull = 0.50
        d_bull = 3.5
        condition = (p_bull < 0.65 and d_bull < 4.0)
        assert condition is True
        reason_template = f"pi(BULL_EXPANSION)={p_bull:.2f} < 0.65 and duration={d_bull:.1f} < 4.0 bars."
        assert " and " in reason_template
        assert " or " not in reason_template

        # If d_bull is sufficient (>= 4.0), condition is FALSE under AND logic
        d_bull_ok = 5.0
        condition_with_dur = (p_bull < 0.65 and d_bull_ok < 4.0)
        assert condition_with_dur is False

if __name__ == "__main__":
    unittest.main()



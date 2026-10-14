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
        # Create synthetic chop price series: oscillating in tight range
        t = np.linspace(0, 10, 80)
        prices = 60000.0 + 50.0 * np.sin(t)
        df = pd.DataFrame({
            "open": prices,
            "high": prices + 10.0,
            "low": prices - 10.0,
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

if __name__ == "__main__":
    unittest.main()

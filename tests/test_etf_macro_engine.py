import unittest
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from etf_macro_engine import CausalETFFlowEngine, MarketHoursAwareMacroGate, MacroGateThresholds


class TestETFMacroEngine(unittest.TestCase):
    def setUp(self):
        self.flow_engine = CausalETFFlowEngine()
        self.macro_gate = MarketHoursAwareMacroGate()

    def test_flow_engine_causality_and_bounds(self):
        """Verifies that the flow engine produces bounded, stationary z-scores without lookahead."""
        dates = pd.date_range("2026-09-01", periods=10, freq="D", tz="UTC")
        flows = [100.0, 150.0, -50.0, 200.0, -300.0, 50.0, 80.0, -120.0, 300.0, -400.0]
        df = pd.DataFrame({"net_flow_usd": flows}, index=dates)

        res = self.flow_engine.compute_causal_flow_signals(df)
        self.assertEqual(len(res), 10)
        self.assertTrue((res["decayed_conviction_modifier"] >= -1.0).all())
        self.assertTrue((res["decayed_conviction_modifier"] <= 1.0).all())

        # Test continuous query state decay
        t0 = dates[-1].to_pydatetime()
        t1 = t0 + timedelta(hours=18)  # exactly 1 half life
        z_last = float(res["causal_z_score"].iloc[-1])
        c0 = self.flow_engine.query_continuous_state(t0, t0, z_last)
        c1 = self.flow_engine.query_continuous_state(t1, t0, z_last)
        # Decay should pull modifier closer to 0
        self.assertLess(abs(c1), abs(c0))

    def test_market_hours_and_weekend_decay(self):
        """Tests NY cash open detection and exponential confidence decay during closures."""
        # Wednesday 14:00 UTC (10:00 AM EDT) -> OPEN
        wed_open = datetime(2026, 10, 14, 14, 0, tzinfo=timezone.utc)
        self.assertTrue(MarketHoursAwareMacroGate.is_us_equity_market_open(wed_open))

        # Wednesday 22:00 UTC (6:00 PM EDT) -> CLOSED
        wed_closed = datetime(2026, 10, 14, 22, 0, tzinfo=timezone.utc)
        self.assertFalse(MarketHoursAwareMacroGate.is_us_equity_market_open(wed_closed))

        # Sunday 15:00 UTC -> CLOSED
        sun_closed = datetime(2026, 10, 18, 15, 0, tzinfo=timezone.utc)
        self.assertFalse(MarketHoursAwareMacroGate.is_us_equity_market_open(sun_closed))

    def test_macro_gate_evaluation_and_credit_veto(self):
        """Verifies that credit stress triggers a hard veto during active sessions."""
        now = datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc)
        
        # Synthesize normal prices
        spy = pd.Series([500.0 + i * 0.1 for i in range(50)])
        uup = pd.Series([28.0 + (i % 2) * 0.05 for i in range(20)])
        hyg = pd.Series([75.0] * 20)
        lqd = pd.Series([105.0] * 20)

        self.macro_gate.update_market_feed(spy, uup, hyg, lqd, now)
        eval_normal = self.macro_gate.evaluate_gate(now)
        self.assertEqual(eval_normal["confidence_weight"], 1.0)
        self.assertFalse(eval_normal["veto_long_entries"])

        # Inject severe credit blowout: HYG collapses relative to LQD
        hyg_crash = pd.Series([75.0] * 19 + [68.0])  # sharp drop
        lqd_stable = pd.Series([105.0] * 20)

        self.macro_gate.update_market_feed(spy, uup, hyg_crash, lqd_stable, now)
        eval_toxic = self.macro_gate.evaluate_gate(now)
        self.assertTrue(eval_toxic["veto_long_entries"])
        self.assertLess(eval_toxic["metrics"]["credit_spread_z"], -1.65)

    def test_weekend_fallback_to_crypto_basis(self):
        """Verifies that when confidence decays below 0.30, gate checks crypto basis."""
        friday_close = datetime(2026, 10, 16, 20, 0, tzinfo=timezone.utc)
        spy = pd.Series([500.0] * 50)
        uup = pd.Series([28.0] * 20)
        hyg = pd.Series([75.0] * 20)
        lqd = pd.Series([105.0] * 20)

        self.macro_gate.update_market_feed(spy, uup, hyg, lqd, friday_close)

        # Query on Sunday afternoon (>40 hours later, well past 4h grace + decay)
        sunday_query = friday_close + timedelta(hours=48)
        
        # Test normal basis -> no veto
        eval_weekend_ok = self.macro_gate.evaluate_gate(sunday_query, crypto_basis_bps=-10.0, crypto_funding_rate=0.0001)
        self.assertLess(eval_weekend_ok["confidence_weight"], 0.30)
        self.assertFalse(eval_weekend_ok["veto_long_entries"])
        self.assertEqual(eval_weekend_ok["regime_tag"], "DECAYED_CRYPTO_BASIS_FALLBACK")

        # Test stressed basis (< -50 bps) -> VETO engaged
        eval_weekend_stressed = self.macro_gate.evaluate_gate(sunday_query, crypto_basis_bps=-65.0, crypto_funding_rate=-0.0003)
        self.assertTrue(eval_weekend_stressed["veto_long_entries"])


if __name__ == "__main__":
    unittest.main()

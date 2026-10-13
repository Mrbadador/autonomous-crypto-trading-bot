"""
tests/test_counterfactual_shadow.py — FIX 1 unit verification for the veto shadow replay

Covers:
  1. run_grid_replay(seeded_positions=...) — SL taker path, TP maker path, unresolved edge mark-out.
  2. Default (non-seeded) behavior unchanged: scan_signals=False yields zero trades.
  3. derive_setups geometry: SL = max(3.0*ATR14(4h), 1.5%), TP = 4.0*risk, no-lookahead signal candle.
  4. bootstrap CI helpers (N<2 guard, seed determinism).

No live DB, no network, no order paths.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import numpy as np
import pandas as pd

from backtests.backtest_grid_replay import run_grid_replay, TAKER_FEE, MAKER_FEE
from scripts.evaluate_boardroom_counterfactual import (
    derive_setups,
    replay_setup,
    wilder_atr,
    bootstrap_ci,
    bootstrap_diff_ci,
    SHADOW_ALLOCATION_USD,
    SHADOW_LEVERAGE,
)

NOTIONAL = SHADOW_ALLOCATION_USD * SHADOW_LEVERAGE


def make_df(rows):
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def seeded_run(df, side, entry, tp, sl, entry_ts, fill_model="touch"):
    return run_grid_replay(
        df,
        symbol="TESTUSDT",
        funding_df=None,
        btc_15m_df=None,
        fill_model=fill_model,
        enable_45m_tightening=False,
        trade_allocation_usd=SHADOW_ALLOCATION_USD,
        leverage=SHADOW_LEVERAGE,
        scan_signals=False,
        seeded_positions=[{"side": side, "entry_price": entry, "take_profit": tp,
                           "stop_loss": sl, "entry_ts": entry_ts, "symbol": "TESTUSDT", "label": "t"}],
    )


class TestSeededReplay(unittest.TestCase):
    def setUp(self):
        self.t0 = 1_790_822_400_000  # 2026-10-01 00:00 UTC
        self.step = 5 * 60 * 1000

    def _bars(self, n, price):
        return make_df([[self.t0 + i * self.step, price, price, price, price, 1.0] for i in range(n)])

    def test_stop_loss_taker_exit(self):
        entry = 100.0
        sl, tp = 98.0, 110.0
        bars = self._bars(50, entry)
        # Force SL on bar 40: low breaches SL
        bars.loc[40, "low"] = 97.0
        trades = seeded_run(bars, "LONG", entry, tp, sl, self.t0 + 30 * self.step)
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual(t["exit_reason"], "STOP_LOSS")
        self.assertFalse(t.get("unresolved", False))
        expected_net = (98.0 - 100.0) / 100.0 * NOTIONAL - NOTIONAL * TAKER_FEE * 2
        self.assertAlmostEqual(t["net_pnl"], expected_net, places=6)

    def test_take_profit_maker_exit(self):
        entry = 100.0
        sl, tp = 98.0, 102.0
        bars = self._bars(50, entry)
        bars.loc[40, "high"] = 102.5
        trades = seeded_run(bars, "LONG", entry, tp, sl, self.t0 + 30 * self.step)
        t = trades[0]
        self.assertEqual(t["exit_reason"], "TAKE_PROFIT")
        self.assertTrue(t["is_maker_exit"])
        expected_net = (102.0 - 100.0) / 100.0 * NOTIONAL - NOTIONAL * TAKER_FEE - NOTIONAL * MAKER_FEE
        self.assertAlmostEqual(t["net_pnl"], expected_net, places=6)

    def test_open_at_edge_markout_flagged_unresolved(self):
        bars = self._bars(50, 100.0)
        trades = seeded_run(bars, "LONG", 100.0, 110.0, 90.0, self.t0 + 30 * self.step)
        t = trades[0]
        self.assertEqual(t["exit_reason"], "OPEN_AT_EDGE_MARK_OUT")
        self.assertTrue(t["unresolved"])

    def test_no_bar_seed_reports_no_bar(self):
        bars = self._bars(50, 100.0)
        ghost_ts = self.t0 + 999 * self.step
        trades = seeded_run(bars, "LONG", 100.0, 110.0, 90.0, ghost_ts)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["exit_reason"], "NO_BAR")
        self.assertTrue(trades[0]["unresolved"])

    def test_defaults_unchanged_and_scan_off_yields_nothing(self):
        # scan_signals=False without seeds -> no trades regardless of price action
        bars = self._bars(200, 100.0)
        bars["low"] = np.where(bars.index > 100, 50.0, 100.0)
        trades = run_grid_replay(bars, symbol="TESTUSDT", scan_signals=False, fill_model="touch")
        self.assertEqual(trades, [])


class TestSetupReconstruction(unittest.TestCase):
    def _cache(self):
        t0 = 1_790_784_000_000  # 2026-09-30 16:00 UTC
        step4 = 4 * 3600 * 1000
        n = 60
        # 4h bars with a known constant range: high=close+1, low=close-1, close=100
        df4 = make_df([[t0 + i * step4, 100.0, 101.0, 99.0, 100.0, 1.0] for i in range(n)])
        # 5m bars: dense grid covering the 4h window plus 48h replay runway; close pinned to 100
        step5 = 5 * 60 * 1000
        rows5 = []
        ts = t0
        end = t0 + (n - 1) * step4 + step4 + 48 * 3600 * 1000
        while ts <= end:
            rows5.append([ts, 100.0, 100.0, 100.0, 100.0, 1.0])
            ts += step5
        df5 = make_df(rows5)
        return {"TEST": {"4h": df4, "5m": df5}}

    def test_geometry_and_signal_candle_selection(self):
        cache = self._cache()
        # debate after 4h bar 20 closed: signal candle = bar 20 (needs >= 15 bars for ATR14)
        t0 = 1_790_784_000_000
        step4 = 4 * 3600 * 1000
        debate_ts = t0 + 20 * step4 + 2 * 3600 * 1000
        debates = pd.DataFrame([{
            "id": 1, "symbol": "TEST/USDT", "side": "LONG", "execute": 0,
            "conviction_score": 0.0, "final_rationale": "", "timestamp": pd.Timestamp(debate_ts, unit="ms", tz="UTC").isoformat(),
        }])
        setups, skipped = derive_setups(debates, cache)
        self.assertEqual(len(setups), 1, msg=str(skipped))
        s = setups[0]
        # debate at bar20_open+2h: bar index 19 (open+4h <= debate) is the last CLOSED candle
        sig_idx = 19
        atr = wilder_atr(cache["TEST"]["4h"], sig_idx)
        self.assertAlmostEqual(atr, 2.0, places=6)  # TR constant = 2.0 for this synthetic series
        sl_dist = max(3.0 * atr, 100.0 * 0.015)
        self.assertAlmostEqual(s["stop_loss"], 100.0 - sl_dist, places=6)
        self.assertAlmostEqual(s["take_profit"], 100.0 + 4.0 * sl_dist, places=6)
        expected_entry_ts = t0 + sig_idx * step4 + step4 - 5 * 60_000  # last 5m bar OF the signal candle
        self.assertEqual(s["entry_ts"], expected_entry_ts)

    def test_end_to_end_replay_of_reconstructed_setup(self):
        cache = self._cache()
        t0 = 1_790_784_000_000
        step4 = 4 * 3600 * 1000
        # Inject an SL breach at 04:00 UTC on the day after bar 20 closes (4h bar 21)
        df5 = cache["TEST"]["5m"]
        breach_ts = t0 + 21 * step4 + 4 * 3600 * 1000
        df5.loc[df5["timestamp"] == breach_ts, ["low", "close"]] = [93.0, 93.0]
        debate_ts = t0 + 20 * step4 + 2 * 3600 * 1000
        debates = pd.DataFrame([{
            "id": 2, "symbol": "TEST/USDT", "side": "LONG", "execute": 0,
            "conviction_score": 0.0, "final_rationale": "",
            "timestamp": pd.Timestamp(debate_ts, unit="ms", tz="UTC").isoformat(),
        }])
        setups, skipped = derive_setups(debates, cache)
        self.assertEqual(len(setups), 1, msg=str(skipped))
        r = replay_setup(setups[0], cache, fill_model="touch")
        self.assertTrue(r["filled_and_closed"])
        self.assertEqual(r["exit_reason"], "STOP_LOSS")
        sl_dist = setups[0]["entry_price"] - setups[0]["stop_loss"]
        expected_net = -sl_dist / setups[0]["entry_price"] * NOTIONAL - NOTIONAL * TAKER_FEE * 2
        self.assertAlmostEqual(r["net_pnl_usd"], expected_net, places=4)


class TestBootstrapHelpers(unittest.TestCase):
    def test_ci_basic_and_guards(self):
        rng = np.random.default_rng(7)
        vals = rng.normal(-5, 2, 100)
        ci = bootstrap_ci(vals)
        self.assertEqual(ci["n"], 100)
        self.assertLess(ci["lower"], ci["upper"])
        self.assertIsNone(bootstrap_ci([]))
        single = bootstrap_ci([3.0])
        self.assertIsNone(single["lower"])

    def test_diff_ci_deterministic(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([-1.0, -2.0, -3.0, 5.0, 6.0])
        d1 = bootstrap_diff_ci(a, b)
        d2 = bootstrap_diff_ci(a, b)
        self.assertEqual(d1, d2)
        self.assertIsNone(bootstrap_diff_ci([1.0], b))


if __name__ == "__main__":
    unittest.main(verbosity=2)

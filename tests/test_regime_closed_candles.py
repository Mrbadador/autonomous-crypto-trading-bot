"""Regime P0 / TASK 3 — regime indicators must be computed on closed candles only,
and the Hurst R/S windowing must actually see the most recent bars.

Two defects are pinned down here:

1. Bitget's public candle endpoint appends the bar that is still forming. A read-only
   probe at 05:29:50Z returned a bar opening 05:15Z whose close mutated between two
   calls six seconds apart (83543.7 -> 83556.4). ADX, Hurst and turbulence therefore
   scored an incomplete bar, and the source_close_ts written by TASK 1 pointed at a bar
   that had not finished. _normalize_bars() now drops every bar whose close time has not
   passed, sorts and de-duplicates by open_time, and reports (never fabricates) gaps.

2. The R/S estimator sliced the return series into num_subsets = N // lag DISJOINT blocks
   starting at index 0, so it discarded the trailing N % lag returns for every lag. With
   the live 99-return window, lag 50 only ever looked at returns[0:50]: a violent move in
   the newest bars could not change the exponent at all. The estimator now averages over
   every overlapping window of each lag.

Isolation: temp-directory database, stubbed HTTP, no orders, no network.
"""

import os
import shutil
import sys
import tempfile
import time
import unittest
from datetime import UTC, datetime
from unittest.mock import patch

import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from regime_classifier import RegimeClassifier  # noqa: E402

STEP_15M = 15 * 60
CLASSIFIER_LOGGER = "regime_classifier"


def legacy_hurst(prices):
    """Verbatim copy of the pre-fix windowing, used only to prove the blind spot existed."""
    returns = np.diff(np.log(prices))
    N = len(returns)
    lags = [10, 20, 30, 40, 50]
    valid_pairs = []
    for lag in lags:
        num_subsets = N // lag
        if num_subsets == 0:
            continue
        rs_sub = []
        for i in range(num_subsets):
            sub = returns[i * lag:(i + 1) * lag]
            mean_adj = sub - np.mean(sub)
            cum_sum = np.cumsum(mean_adj)
            r = np.max(cum_sum) - np.min(cum_sum)
            s = np.std(sub)
            if s > 0:
                rs_sub.append(r / s)
        if rs_sub:
            avg_rs = float(np.mean(rs_sub))
            if avg_rs > 0:
                valid_pairs.append((lag, avg_rs))
    if len(valid_pairs) < 2:
        return 0.50
    poly = np.polyfit(np.log([p[0] for p in valid_pairs]), np.log([p[1] for p in valid_pairs]), 1)
    return max(0.0, min(1.0, float(poly[0])))


class StubResponse:
    """Minimal stand-in for requests.Response used by fetch_ohlcv."""

    def __init__(self, rows):
        self.status_code = 200
        self._rows = rows

    def json(self):
        return {"code": "00000", "data": self._rows}


def bitget_rows(open_ts_seconds, close=100.0):
    """Bitget candle rows, newest last: [open_time_ms, o, h, l, c, base_vol, quote_vol]."""
    return [[str(int(ot * 1000)), str(close), str(close * 1.001), str(close * 0.999),
             str(close), "1.0", "100.0"] for ot in open_ts_seconds]


class TestClosedCandleRegimeInput(unittest.TestCase):
    clf: RegimeClassifier

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="regime_bars_")
        cls.repo_cwd = os.getcwd()
        os.chdir(cls.tmp_dir)
        cls.db_path = os.path.join(cls.tmp_dir, "fixture_regime.db")
        cls.clf = RegimeClassifier(db_path=cls.db_path)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.repo_cwd)
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def frame(self, n=100, interval_sec=STEP_15M, newest_open_offset_sec=0.0, step_drift=0.0):
        """n bars opening every interval_sec; the newest open is shifted by the offset."""
        epoch = int(time.time())
        now_slot = epoch - epoch % interval_sec - int(newest_open_offset_sec)
        rows = []
        close = 100.0
        for i in range(n):
            open_ms = (now_slot - (n - 1 - i) * interval_sec) * 1000
            rows.append({"open_time": open_ms, "open": close, "high": close * 1.002,
                         "low": close * 0.998, "close": close, "volume": 10.0,
                         "quote_volume": close * 10.0})
            close *= (1.0 + step_drift)
        return pd.DataFrame(rows)

    # ---------------------------------------------------------------- 1. bars
    def test_still_forming_bar_is_dropped(self):
        bars = self.frame(n=100)                                     # newest covers the live slot
        closed = self.clf._normalize_bars(bars, "15m", symbol="BTCUSDT")
        self.assertEqual(len(closed), 99, "the bar whose close time has not passed must be dropped")
        self.assertEqual(closed["open_time"].iloc[-1], bars["open_time"].iloc[-2])

    def test_bar_that_finished_seconds_ago_is_kept(self):
        bars = self.frame(n=50, newest_open_offset_sec=STEP_15M + 20)  # closed 20s ago
        closed = self.clf._normalize_bars(bars, "15m", symbol="BTCUSDT")
        self.assertEqual(len(closed), 50)

    def test_forward_dated_bars_are_dropped(self):
        bars = self.frame(n=30)
        future = int(time.time()) + 3 * STEP_15M
        bars.loc[bars.index[-1], "open_time"] = future * 1000
        bars.loc[bars.index[-2], "open_time"] = (future - STEP_15M) * 1000
        closed = self.clf._normalize_bars(bars, "15m", symbol="BTCUSDT")
        self.assertEqual(len(closed), 28)

    def test_out_of_order_and_duplicate_timestamps_are_repaired(self):
        bars = self.frame(n=40)
        shuffled = bars.iloc[::-1].copy()                            # newest first, as returned
        dup = shuffled.iloc[[5]].copy()
        dup["close"] = 999.0                                         # keep='last' wins
        with_dup = pd.concat([shuffled, dup], ignore_index=True)
        fixed = self.clf._normalize_bars(with_dup, "15m", symbol="BTCUSDT")
        self.assertEqual(len(fixed), 39, "one forming bar dropped, one duplicate collapsed")
        self.assertTrue(fixed["open_time"].is_monotonic_increasing)
        self.assertEqual(fixed["open_time"].duplicated().sum(), 0)
        kept = fixed[fixed["open_time"] == shuffled["open_time"].iloc[5]]
        self.assertAlmostEqual(float(kept["close"].iloc[0]), 999.0, places=6)

    def test_discontinuity_is_warned_about_and_never_fabricated(self):
        bars = self.frame(n=40)
        shifted = bars["open_time"].astype("int64") + 2 * STEP_15M * 1000   # insert a 1-bar hole
        bars.loc[bars.index >= 20, "open_time"] = shifted[bars.index >= 20]
        bars = bars[bars["open_time"] + STEP_15M * 1000 <= int(time.time() * 1000)]
        with self.assertLogs(CLASSIFIER_LOGGER, level="WARNING") as captured:
            fixed = self.clf._normalize_bars(bars, "15m", symbol="BTCUSDT")
        self.assertEqual(len(fixed), len(bars), "gaps must be reported, not filled with invented bars")
        self.assertTrue(any("discontinuous step" in line for line in captured.output),
                        captured.output)

    def test_unparsable_interval_keeps_rows_and_reports_no_crash(self):
        bars = self.frame(n=25)
        fixed = self.clf._normalize_bars(bars, "15minutes", symbol="BTCUSDT")
        self.assertEqual(len(fixed), 25, "without a duration nothing can be judged forming")
        self.assertIsNone(self.clf._bar_close_ts(fixed, "15minutes"))
        self.assertIsNotNone(self.clf._bar_close_ts(fixed, "15m"))

    def test_empty_and_none_frames_pass_through(self):
        self.assertIsNone(self.clf._normalize_bars(None, "15m"))
        empty = self.clf._normalize_bars(pd.DataFrame(), "15m")
        self.assertEqual(len(empty), 0)

    def test_source_close_ts_is_never_in_the_future_after_normalisation(self):
        bars = self.frame(n=60)
        closed = self.clf._normalize_bars(bars, "15m", symbol="BTCUSDT")
        ts = self.clf._bar_close_ts(closed, "15m")
        parsed = datetime.fromisoformat(ts)
        self.assertLessEqual(parsed.astimezone(UTC), datetime.now(UTC),
                             "source_close_ts must be a market time the data already reflects")

    def test_fetch_ohlcv_returns_only_closed_bars(self):
        epoch = int(time.time())
        slot = epoch - epoch % STEP_15M
        opens = [slot - (9 - i) * STEP_15M for i in range(10)]       # last one is forming
        rows = bitget_rows(opens)
        with patch("regime_classifier.requests.get", return_value=StubResponse(rows)):
            out = self.clf.fetch_ohlcv(symbol="BTC/USDT", interval="15m", limit=10)
        self.assertEqual(len(out), 9)
        self.assertEqual(int(out["open_time"].iloc[-1]), (slot - STEP_15M) * 1000)

    def test_indicator_memory_shortens_by_exactly_the_forming_bar(self):
        """classify_and_save() requires >= 50 rows; a 100-limit pull now yields 99."""
        epoch = int(time.time())
        slot = epoch - epoch % STEP_15M
        opens = [slot - (99 - i) * STEP_15M for i in range(100)]
        with patch("regime_classifier.requests.get", return_value=StubResponse(bitget_rows(opens))):
            out = self.clf.fetch_ohlcv(symbol="BTCUSDT", interval="15m", limit=100)
        self.assertEqual(len(out), 99)
        self.assertGreaterEqual(len(out), 50)

    # --------------------------------------------------------------- 2. Hurst
    def test_legacy_windowing_was_blind_to_the_newest_return(self):
        """Before/after evidence pinned inside the suite."""
        rng = np.random.default_rng(11)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.0025, size=92)))
        shocked = prices.copy()
        shocked[-1] = shocked[-1] * 1.06                             # only returns[-1] changes
        self.assertEqual(legacy_hurst(prices), legacy_hurst(shocked),
                         "the shipped estimator must be provably insensitive to the last return")
        self.assertNotEqual(self.clf.calculate_hurst(prices),
                            self.clf.calculate_hurst(shocked))

    def test_large_recent_move_changes_the_hurst_exponent(self):
        rng = np.random.default_rng(7)
        base = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.0025, size=92)))
        trending = list(base)
        for _ in range(8):
            trending.append(trending[-1] * 1.015)
        quiet_h = self.clf.calculate_hurst(base)
        trend_h = self.clf.calculate_hurst(np.array(trending))
        self.assertGreater(trend_h, quiet_h + 0.01,
                           f"a sustained 8-bar advance must raise the exponent ({quiet_h:.4f} -> {trend_h:.4f})")
        legacy_quiet, legacy_trend = legacy_hurst(base), legacy_hurst(np.array(trending))
        self.assertLess(abs(legacy_trend - legacy_quiet), abs(trend_h - quiet_h),
                        "the fixed estimator must be more, not less, responsive to recent data")

    def test_spike_only_in_the_last_bars_moves_the_exponent(self):
        rng = np.random.default_rng(3)
        base = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.0025, size=99)))
        spiked = base.copy()
        spiked[-3:] = spiked[-4] * np.array([1.05, 1.05, 1.05])
        self.assertNotEqual(self.clf.calculate_hurst(base), self.clf.calculate_hurst(spiked))

    def test_hurst_still_separates_trend_from_mean_reversion(self):
        trend = 100.0 * np.exp(np.linspace(0.0, 0.35, 200))
        mean_reverting = 100.0 + np.array([1.0, -1.0, 0.7, -0.7] * 50, dtype=float)
        self.assertGreater(self.clf.calculate_hurst(trend),
                           self.clf.calculate_hurst(mean_reverting))

    def test_hurst_bounds_and_short_series_default(self):
        self.assertEqual(self.clf.calculate_hurst(np.array([1.0] * 49)), 0.50)
        rng = np.random.default_rng(5)
        for seed in range(6):
            series = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=120)))
            value = self.clf.calculate_hurst(series)
            self.assertTrue(0.0 <= value <= 1.0, f"{value} out of bounds")
        flat = np.full(120, 100.0)
        self.assertEqual(self.clf.calculate_hurst(flat), 0.50, "zero-variance series must default")

    def test_hurst_is_deterministic(self):
        series = 100.0 * np.exp(np.cumsum(np.random.default_rng(21).normal(0.0, 0.003, size=100)))
        self.assertEqual(self.clf.calculate_hurst(series), self.clf.calculate_hurst(series))


if __name__ == "__main__":
    unittest.main(verbosity=2)

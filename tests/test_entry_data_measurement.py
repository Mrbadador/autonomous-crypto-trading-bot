"""Entry measurement (item #2a): the 1D trend filter must run on real daily data.

Live bug class: Bitget swap OHLCV caps at 90 bars per request even for 1d, so
fetch_ohlcv(limit=210) returned ~90 rows, `_compute_1d_trend` hit its
`len < 200` stub and fail-OPENED with bullish_alignment=True for every symbol
(and fail-closed for shorts — asymmetric dead gate). Fix: page forward with
`since` until >=200 bars, and make the stub fail CLOSED when history is short.
"""
import asyncio
import os
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock

import pandas as pd

from swing_trader import SwingArchitectTrader as SwingTrader

DAY_MS = 86_400_000


def _rows(start_ts_ms: int, count: int):
    """ccxt OHLCV shape: [ts, o, h, l, c, vol], ascending."""
    return [[start_ts_ms + i * DAY_MS, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 10.0]
            for i in range(count)]


class _PagedExchange:
    """Fake ccxt exchange: total bars, but every call caps at 90 (Bitget behavior)."""
    def __init__(self, total=260):
        self.total = total
        self.calls = []

    async def fetch_ohlcv(self, symbol, timeframe="1d", since=None, limit=None, params=None):
        self.calls.append({"since": since, "limit": limit})
        if since is None:
            return self._series()[self.total - 90:]
        start_idx = max(0, int((since - self.base) // DAY_MS))
        return self._series()[start_idx:][:90]

    def _series(self):
        return _rows(self.base, self.total)

    @property
    def base(self):
        now = int(time.time() * 1000)
        day_floor = now - (now % DAY_MS)
        return day_floor - (self.total - 1) * DAY_MS


def _make_trader(exchange):
    trader = SwingTrader.__new__(SwingTrader)
    trader.uta_client = MagicMock()
    trader.uta_client.exchange = exchange
    return trader


class TestMergePages(unittest.TestCase):
    def test_dedupe_and_sort_ascending(self):
        a = _rows(1_000_000 * DAY_MS, 3)
        b = _rows(1_000_002 * DAY_MS, 3)  # overlaps a on last row
        merged = SwingTrader._merge_ohlcv_pages([b, a])
        self.assertEqual([r[0] for r in merged],
                         [1_000_000 * DAY_MS, 1_000_001 * DAY_MS, 1_000_002 * DAY_MS,
                          1_000_003 * DAY_MS, 1_000_004 * DAY_MS])

    def test_empty_input(self):
        self.assertEqual(SwingTrader._merge_ohlcv_pages([]), [])


class TestFetchDailyHistory(unittest.TestCase):
    def test_stitches_pages_until_200_bars(self):
        ex = _PagedExchange(total=260)
        trader = _make_trader(ex)
        rows = asyncio.run(trader._fetch_daily_history("BTCUSDT"))
        self.assertGreaterEqual(len(rows), 200)
        self.assertGreaterEqual(len(ex.calls), 2, "must page with since, not one capped call")
        # ascending, no duplicate timestamps
        ts = [r[0] for r in rows]
        self.assertEqual(ts, sorted(ts))
        self.assertEqual(len(ts), len(set(ts)))
        # freshest bar present
        self.assertEqual(max(ts), ex.base + 259 * DAY_MS)

    def test_short_history_returns_what_exists_and_paging_stops(self):
        ex = _PagedExchange(total=120)
        trader = _make_trader(ex)
        rows = asyncio.run(trader._fetch_daily_history("BTCUSDT"))
        self.assertLess(len(rows), 200)
        self.assertEqual(len(rows), 120)

    def test_fetch_error_returns_partial_not_raise(self):
        ex = AsyncMock()
        ex.fetch_ohlcv = AsyncMock(side_effect=RuntimeError("network down"))
        trader = _make_trader(ex)
        rows = asyncio.run(trader._fetch_daily_history("BTCUSDT"))
        self.assertEqual(rows, [])


def _ind_4h(**over):
    base = {"close": 100.0, "bullish_kumo": False, "bearish_kumo": False,
            "bullish_candle_4h": False, "bearish_candle_4h": False, "candle_ts": 1_700_000_000_000}
    base.update(over)
    return base


def _eval(ind_4h=None, ind_1d=None, live_o=100.0, live_c=100.0, adx=30.0, rsi=50.0, min_adx=22.0):
    return SwingTrader._evaluate_swing_candidate(
        ind_4h or _ind_4h(), ind_1d or {"bullish_alignment": True, "bearish_alignment": True},
        live_o, live_c, adx, rsi, min_adx)


class TestEvaluateSwingCandidate(unittest.TestCase):
    def test_full_long_checklist_passes(self):
        r = _eval(_ind_4h(bullish_kumo=True, bullish_candle_4h=True))
        self.assertEqual(r["direction"], "LONG")
        self.assertEqual(r["long_failed"], [])
        self.assertFalse(r["near_miss"])

    def test_pattern_present_but_rsi_blocks_is_near_miss(self):
        r = _eval(_ind_4h(bullish_kumo=True, bullish_candle_4h=True), rsi=75.0)
        self.assertIsNone(r["direction"])
        self.assertEqual(r["long_failed"], ["rsi_overbought"])
        self.assertTrue(r["near_miss"])

    def test_no_pattern_side_is_not_near_miss(self):
        r = _eval(_ind_4h(), adx=10.0, rsi=80.0)
        self.assertIsNone(r["direction"])
        self.assertFalse(r["near_miss"])

    def test_checklist_matches_original_gate_semantics(self):
        # Mirror of the original inline AND-conditions, over a spread of states.
        import itertools
        flags = [True, False]
        for bk, bc, k_br, c_br, lo, lc, adx_v, rsi_v in itertools.product(
                flags, flags, flags, flags, (100.0, 99.0), (100.2, 99.0), (10.0, 30.0), (20.0, 50.0, 80.0)):
            ind = _ind_4h(bullish_kumo=bk, bullish_candle_4h=bc, bearish_kumo=k_br, bearish_candle_4h=c_br)
            ind_1d = {"bullish_alignment": True, "bearish_alignment": True}
            original = None
            if (bk and bc and lc >= lo * 0.9985 and lc >= 100.0 * 0.9970 and adx_v >= 22.0 and rsi_v <= 70.0):
                original = "LONG"
            elif (k_br and c_br and lc <= lo * 1.0015 and lc <= 100.0 * 1.0030 and adx_v >= 22.0 and rsi_v >= 35.0):
                original = "SHORT"
            got = _eval(ind, ind_1d, lo, lc, adx_v, rsi_v)["direction"]
            self.assertEqual(got, original, f"mismatch bk={bk} bc={bc} k_br={k_br} c_br={c_br} lo={lo} lc={lc} adx={adx_v} rsi={rsi_v}")

    def test_short_direction_and_failed(self):
        r = _eval(_ind_4h(bearish_kumo=True, bearish_candle_4h=True),
                  {"bullish_alignment": False, "bearish_alignment": True},
                  live_o=100.0, live_c=99.0, rsi=40.0)
        self.assertEqual(r["direction"], "SHORT")


class TestChecklistRejectShadowRow(unittest.TestCase):
    def setUp(self):
        self.old_cwd = os.getcwd()
        self.tmp = tempfile.mkdtemp(prefix="checklist_reject_")
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self.old_cwd)
        try:
            import shutil
            shutil.rmtree(self.tmp, ignore_errors=True)
        except Exception:
            pass

    def test_reject_written_once_per_candle(self):
        trader = SwingTrader.__new__(SwingTrader)
        key = "BTC/USDT|CHECKLIST_L|1700000000000"
        trader._record_shadow_signal_sync("SWING_4H", "BTC/USDT", "LONG", 100.0,
                                          veto_reason="CHECKLIST-REJECT: adx_below_22,1d_bullish_alignment",
                                          candle_key=key)
        trader._record_shadow_signal_sync("SWING_4H", "BTC/USDT", "LONG", 101.0,
                                          veto_reason="CHECKLIST-REJECT: again", candle_key=key)
        with sqlite3.connect("trading_data.db") as conn:
            rows = conn.execute("SELECT direction, veto_reason, entry_price FROM shadow_signals").fetchall()
        self.assertEqual(len(rows), 1, "unique candle_key must dedupe repeat rejects")
        self.assertEqual(rows[0][0], "LONG")
        self.assertTrue(rows[0][1].startswith("CHECKLIST-REJECT:"))

    def test_full_sizing_path_untouched_by_helpers(self):
        # smoke: statics added for #27/#28 do not shadow instance methods
        trader = SwingTrader.__new__(SwingTrader)
        self.assertTrue(callable(trader._compute_1d_trend))


class TestOneDTrendFailClosed(unittest.TestCase):
    def test_insufficient_history_fails_closed(self):
        df = pd.DataFrame(_rows(0, 90), columns=["timestamp", "open", "high", "low", "close", "volume"])
        out = SwingTrader.__new__(SwingTrader)._compute_1d_trend(df)
        self.assertFalse(out["bullish_alignment"])
        self.assertFalse(out["bearish_alignment"])
        self.assertEqual(out["daily_trend"], "INSUFFICIENT")

    def test_sufficient_history_bullish_still_passes(self):
        # Mostly flat at 100 with a late uptick: price above EMA200 and EMA50 near it.
        bars = [[i * DAY_MS, 100.0, 100.0, 100.0, 100.0, 10.0] for i in range(250)]
        for i in range(240, 250):
            bars[i][4] = 101.0 + (i - 240)
        df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
        out = SwingTrader.__new__(SwingTrader)._compute_1d_trend(df)
        self.assertTrue(out["bullish_alignment"])
        self.assertEqual(out["daily_trend"], "BULLISH")


if __name__ == "__main__":
    unittest.main()

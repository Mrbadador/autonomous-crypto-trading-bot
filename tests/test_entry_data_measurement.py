"""Entry measurement (item #2a): the 1D trend filter must run on real daily data.

Live bug class: Bitget swap OHLCV caps at 90 bars per request even for 1d, so
fetch_ohlcv(limit=210) returned ~90 rows, `_compute_1d_trend` hit its
`len < 200` stub and fail-OPENED with bullish_alignment=True for every symbol
(and fail-closed for shorts — asymmetric dead gate). Fix: page forward with
`since` until >=200 bars, and make the stub fail CLOSED when history is short.
"""
import asyncio
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


class TestOneDTrendFailClosed(unittest.TestCase):
    def test_insufficient_history_fails_closed(self):
        df = pd.DataFrame(_rows(0, 90), columns=["timestamp", "open", "high", "low", "close", "volume"])
        out = SwingTrader.__new__(SwingTrader)._compute_1d_trend(df)
        self.assertFalse(out["bullish_alignment"])
        self.assertFalse(out["bearish_alignment"])
        self.assertEqual(out["daily_trend"], "INSUFFICIENT")

    def test_sufficient_history_bullish_still_passes(self):
        df = pd.DataFrame(_rows(0, 250), columns=["timestamp", "open", "high", "low", "close", "volume"])
        out = SwingTrader.__new__(SwingTrader)._compute_1d_trend(df)
        self.assertTrue(out["bullish_alignment"])
        self.assertEqual(out["daily_trend"], "BULLISH")


if __name__ == "__main__":
    unittest.main()

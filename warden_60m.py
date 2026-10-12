# warden_60m.py — V20.1 60-Minute Macro Trend Warden (ccxt Live Feed)
#
# Fetches the 50 most recent 1-hour candles directly from Binance via ccxt
# and calculates the macro trend using a 9/21 EMA crossover stack.
# The SQLite market_data dependency has been removed entirely.

import socket
import pandas as pd
import aiohttp
import dns_fix  # noqa: F401 — patches ccxt.Exchange.open() with ThreadedResolver
import ccxt.async_support as ccxt
import logging
import config

log = logging.getLogger("warden_60m")


class Warden60Minute:
    """
    V20.1 Multi-Timeframe Directional Lock (ccxt edition).

    Fetches live 1h candles from Binance and returns one of three regime strings:
        LONG_ONLY  — Uptrend confirmed. Only LONG entries are permitted.
        SHORT_ONLY — Downtrend confirmed. Only SHORT entries are permitted.
        CHOP_GRID_ACTIVE — Market is ranging. Ghost-Grid Armed.
    """

    def __init__(self, symbol: str = "BTC/USDT", db_path: str = None):
        # db_path kept as a no-op kwarg so existing callers don't break
        self.symbol = symbol
        self._session  = None
        self.exchange  = None
        log.info(f"[60M-WARDEN] Initialized (Async). Guarding the Macro Trend via Live Binance API ({symbol}).")

    async def _ensure_session(self):
        """Lazy-initialize the aiohttp session and ccxt exchange inside the event loop.

        ClientSession must be created inside a running event loop, so we cannot
        do this in __init__.  Calling this at the start of every coroutine is
        safe because it is a no-op after the first call.
        """
        if self._session is None or self._session.closed:
            # DNS FIX: ThreadedResolver delegates to the OS socket layer
            # (getaddrinfo) so we bypass the broken aiodns/c-ares resolver.
            connector = aiohttp.TCPConnector(
                resolver=aiohttp.ThreadedResolver(),
                family=socket.AF_INET,
                use_dns_cache=True,
                ttl_dns_cache=300,
            )
            self._session = aiohttp.ClientSession(connector=connector)

        if self.exchange is None:
            ex_id = getattr(config, "EXCHANGE_ID", "bitget").lower()
            ex_class = getattr(ccxt, ex_id, None) or ccxt.bitget
            self.exchange = ex_class({
                "enableRateLimit": True,
                "options": {
                    "defaultType": "swap" if ex_id == "bitget" else "future",
                    "adjustForTimeDifference": True,
                },
                "aiohttp_session": self._session,
            })

    async def close(self):
        """Closes the ccxt exchange and underlying aiohttp session."""
        if self.exchange:
            await self.exchange.close()
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch_1h_data(self) -> pd.DataFrame | None:
        """Pulls the latest 50 1-hour candles directly from the exchange."""
        await self._ensure_session()
        try:
            # Returns list of [timestamp, open, high, low, close, volume]
            bars = await self.exchange.fetch_ohlcv(self.symbol, timeframe="1h", limit=50)
            df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close_price", "volume"])
            return df
        except Exception as e:
            log.error(f"[60M-WARDEN] 1h API Fetch Error: {e}")
            return None

    async def fetch_15m_data(self) -> pd.DataFrame | None:
        """Pulls the latest 30 15-minute candles directly from the exchange."""
        await self._ensure_session()
        try:
            bars = await self.exchange.fetch_ohlcv(self.symbol, timeframe="15m", limit=30)
            df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close_price", "volume"])
            return df
        except Exception as e:
            log.error(f"[60M-WARDEN] 15m API Fetch Error: {e}")
            return None

    async def determine_regime(self) -> str:
        """
        Calculates the macro trend using multi-horizon dual-timeframe analysis:
        - 1H 9/21 EMA stack (Macro trend)
        - 15M 20/50 EMA structure + 1-hour rolling Rate of Change (Velocity Shock Breaker)

        Returns:
            str: "LONG_ONLY", "SHORT_ONLY", or "CHOP_GRID_ACTIVE"
        """
        df_1h = await self.fetch_1h_data()
        df_15m = await self.fetch_15m_data()

        if df_1h is None or len(df_1h) < 21:
            log.warning("[60M-WARDEN] Insufficient 1H data (fail-closed). Defaulting to CAPITAL_PRESERVATION.")
            return "CAPITAL_PRESERVATION"

        # 1H Vectorized dual-EMA stack
        df_1h["EMA_9"]  = df_1h["close_price"].ewm(span=9,  adjust=False).mean()
        df_1h["EMA_21"] = df_1h["close_price"].ewm(span=21, adjust=False).mean()

        current_close = float(df_1h["close_price"].iloc[-1])
        current_ema9  = float(df_1h["EMA_9"].iloc[-1])
        current_ema21 = float(df_1h["EMA_21"].iloc[-1])

        # Fast 15M / 1H Velocity Analysis
        fast_bearish = False
        fast_bullish = False

        if df_15m is not None and len(df_15m) >= 20:
            df_15m["EMA_20"] = df_15m["close_price"].ewm(span=20, adjust=False).mean()
            df_15m["EMA_50"] = df_15m["close_price"].ewm(span=50, adjust=False).mean()
            
            c_15m = float(df_15m["close_price"].iloc[-1])
            ema20_15m = float(df_15m["EMA_20"].iloc[-1])
            ema50_15m = float(df_15m["EMA_50"].iloc[-1])
            
            # 1-hour rolling price velocity (4 bars on 15m)
            prev4 = float(df_15m["close_price"].iloc[-5]) if len(df_15m) >= 5 else float(df_15m["close_price"].iloc[0])
            roc_1h = ((c_15m - prev4) / prev4) * 100

            # Velocity Shock or Fast Structural Breakdown
            if roc_1h <= -0.65 or (c_15m < ema20_15m and c_15m < ema50_15m and c_15m < current_ema9):
                fast_bearish = True
                log.info(f"[60M-WARDEN] Fast Velocity/Structure Breakdown: 1h ROC={roc_1h:+.2f}% | Close={c_15m:.1f} < 15m EMA20 ({ema20_15m:.1f}) & EMA50 ({ema50_15m:.1f})")
            elif roc_1h >= +0.65 or (c_15m > ema20_15m and c_15m > ema50_15m and c_15m > current_ema9 and current_ema9 > current_ema21):
                fast_bullish = True
                log.info(f"[60M-WARDEN] Fast Velocity/Structure Breakout: 1h ROC={roc_1h:+.2f}% | Close={c_15m:.1f} > 15m EMA20 ({ema20_15m:.1f}) & EMA50 ({ema50_15m:.1f})")

        # Multi-Horizon Decision Matrix
        if fast_bearish:
            log.info("[60M-WARDEN] Regime Set: SHORT_ONLY (Fast Breakdown Activated)")
            return "SHORT_ONLY"
        elif fast_bullish and current_ema9 > current_ema21 and current_close > current_ema9:
            log.info("[60M-WARDEN] Regime Set: LONG_ONLY (Uptrend Confirmed)")
            return "LONG_ONLY"
        elif current_ema9 > current_ema21 and current_close > current_ema9 and not fast_bearish:
            log.info("[60M-WARDEN] Regime Set: LONG_ONLY (Uptrend Confirmed by 1H Stack)")
            return "LONG_ONLY"
        elif current_ema9 < current_ema21 and current_close < current_ema9:
            log.info("[60M-WARDEN] Regime Set: SHORT_ONLY (Downtrend Confirmed by 1H Stack)")
            return "SHORT_ONLY"
        else:
            trend_dist = ((current_close - current_ema9) / current_ema9) * 100
            log.info(f"[WARDEN-INSIGHT] EMA Distance {trend_dist:+.2f}% signals chop (no clear stack).")
            log.warning("[60M-WARDEN] Regime Set: CHOP_GRID_ACTIVE (Market is ranging. Ghost-Grid Armed.)")
            return "CHOP_GRID_ACTIVE"


# ---------------------------------------------------------------------------
# Standalone diagnostic (python warden_60m.py [SYMBOL])
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import asyncio
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC/USDT"
    warden = Warden60Minute(symbol)

    async def main():
        try:
            print(f"\n[60M-WARDEN] Fetching live 1h data for {symbol}...")
            regime = await warden.determine_regime()
            print(f"\n{'='*50}")
            print(f"  Symbol : {symbol}")
            print(f"  Regime : {regime}")
            print(f"{'='*50}\n")
        finally:
            await warden.close()

    asyncio.run(main())

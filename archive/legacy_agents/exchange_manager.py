import ccxt.async_support as ccxt
import pandas as pd
import config
import logging
import os
from typing import Optional
from simulator import SimulatedExchange
from market_replay import ReplayManager

log = logging.getLogger(__name__)

import aiohttp

class ExchangeManager:
    def __init__(self):
        # 1. Public Data Backend (Unauthenticated Spot)
        exchange_class = getattr(ccxt, config.EXCHANGE_ID.lower())
        self.ex = exchange_class({
            "apiKey": getattr(config, f"{config.EXCHANGE_ID.upper()}_API_KEY", ""),
            "secret": getattr(config, f"{config.EXCHANGE_ID.upper()}_SECRET_KEY", ""),
            "enableRateLimit": True,
            "sandbox": config.SANDBOX,
            "timeout": 15000,
            "aiohttp_config": {
                "resolver": aiohttp.ThreadedResolver()
            }
        })
        # PATCH: ENABLE AUTOMATIC TIMESTAMP ADJUSTMENT
        self.ex.enableRateLimit = True
        self.ex.options['adjustForTimeDifference'] = True
        log.info(f"CCXT {config.EXCHANGE_ID} initialized with 15s timeout.")

        
        # 2. Replay Manager (Phase 9)
        self.replay = ReplayManager(data_dir=config.DATA_DIR)
        
        # 3. Virtual Transaction Backend
        self.sim = SimulatedExchange(starting_balance=config.INITIAL_BALANCE)
        
        log.info(f"ExchangeManager Initialized. Mode: {config.MODE.upper()}")

    async def close(self):
        """Cleanly close the underlying CCXT async session."""
        if hasattr(self, "ex") and self.ex:
            await self.ex.close()
            log.info("CCXT Exchange session closed.")

    def init_replay(self, symbols, timeframe):
        """Initializes historical buffers for replay mode."""
        if not os.path.exists(config.DATA_DIR):
            os.makedirs(config.DATA_DIR, exist_ok=True)
            
        for s in symbols:
            try:
                self.replay.load_symbol(s, timeframe)
            except Exception as e:
                log.error(f"Failed to load replay for {s}: {e}")

    @property
    def positions(self):
        """Returns all open positions across all symbols."""
        return list(self.sim.positions.values())

    def get_positions_by_symbol(self, symbol):
        """Returns all open positions for a specific symbol."""
        return [p for p in self.sim.positions.values() if p.symbol == symbol]

    @property
    def balance(self):
        return self.sim.get_balance()

    def update_sim_price(self, symbol, price):
        self.sim.update_price(symbol, price)

    def open_sim_pos(self, symbol, side, qty, price, leverage, **metadata):
        return self.sim.open_position(symbol, side, qty, price, leverage, **metadata)

    def close_sim_pos(self, pos_id: int, price: Optional[float] = None):
        """Closes a specific position by ID."""
        return self.sim.close_position(pos_id, price)

    def close_partial_sim_pos(self, pos_id: int, fraction: float, price: Optional[float] = None) -> Optional[float]:
        """Closes a fraction of a position. Returns realised PnL or None."""
        return self.sim.close_partial_position(pos_id, fraction, price)

    def set_leverage(self, symbol, leverage):
        log.info(f"Local Leverage set to {leverage} in Phase 10 Simulator.")

    async def fetch_ohlcv(self, symbol, timeframe, limit):
        if config.MODE == "replay":
            try:
                return self.replay.get_window(symbol, timeframe, limit)
            except KeyError:
                if timeframe == config.TIMEFRAME:
                    log.error(f"Replay buffer missing for primary timeframe {symbol} {timeframe}")
                    return None

        import time
        import random
        import asyncio
        for attempt in range(5): # Increased retries
            try:
                # Fetch data
                raw = await self.ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                if not raw:
                    log.warning(f"[{symbol}] Attempt {attempt+1}: Empty data returned.")
                    continue
                    
                df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df.set_index("timestamp", inplace=True)
                return df
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "rate limit" in err_str or "429" in err_str
                is_timeout = "timeout" in err_str or "network" in err_str
                
                backoff_intervals = [5, 15, 60] # 5s, 15s, 60s
                
                if is_rate_limit:
                    wait_time = (2 ** attempt) + random.random()
                    log.warning(f"[{symbol}] Rate limit hit. Retrying in {wait_time:.2f}s... (Attempt {attempt+1})")
                    await asyncio.sleep(wait_time)
                elif is_timeout and attempt < len(backoff_intervals):
                    wait_time = backoff_intervals[attempt]
                    log.warning(f"[{symbol}] Network timeout. Retrying in {wait_time}s... (Attempt {attempt+1})")
                    await asyncio.sleep(wait_time)
                else:
                    log.error(f"[{symbol}] Attempt {attempt+1} failed: {e}")
                    await asyncio.sleep(1)
        
        if not hasattr(self, "_last_fetch_crit_log"): self._last_fetch_crit_log = {}
        last_log = self._last_fetch_crit_log.get(symbol)
        if not last_log or (time.time() - last_log) > 3600:
            log.critical(f"!!! CRITICAL: All fetch attempts failed for {symbol}.")
            self._last_fetch_crit_log[symbol] = time.time()
        return None


    async def get_funding_rate(self, symbol, timestamp=None):
        if config.MODE == "replay":
            # Deterministic but fluctuating funding based on simulation time
            import math
            t = timestamp.timestamp() if timestamp else 0
            rate = 0.0001 * math.sin(t / 28800) + 0.00005 # Fluctuates every 8h period
            return {"rate": rate, "next": None}
        try:
            f = await self.ex.fetch_funding_rate(symbol)
            return {"rate": f.get("fundingRate", 0), "next": f.get("fundingTimestamp", f.get("nextFundingTime", None))}
        except Exception as e:
            log.warning(f"Failed to fetch funding rate for {symbol} ({e}). Returning fallback rate of 0.0001.")
            return {"rate": 0.0001, "next": None}

    def calculate_global_margin_usage(self):
        bal = self.sim.get_balance()
        return (bal.used / bal.total * 100) if bal.total > 0 else 0
    async def fetch_bbo(self, symbol):
        """Fetches the Best Bid and Best Offer to simulate Limit Order entries."""
        if config.MODE == "replay":
            # In replay, we estimate the spread as 0.05% of the close price
            price = self.sim.last_prices.get(symbol, 0)
            if price == 0:
                log.critical(f"DANGEROUS FALLBACK: Replay price for {symbol} is 0. Skipping cycle.")
                return None
            spread = price * 0.0005 
            return {"bid": price - spread, "ask": price + spread}
        
        try:
            # Fetch live top of book (level 1 orderbook)
            ob = await self.ex.fetch_order_book(symbol, limit=5)
            best_bid = ob['bids'][0][0] if len(ob['bids']) > 0 else None
            best_ask = ob['asks'][0][0] if len(ob['asks']) > 0 else None
            
            if not best_bid or not best_ask or best_bid == 0 or best_ask == 0:
                log.critical(f"DANGEROUS FALLBACK: Live BBO for {symbol} is 0 or missing. Skipping cycle.")
                return None

            return {"bid": best_bid, "ask": best_ask}
        except Exception as e:
            log.warning(f"[{symbol}] Failed to fetch BBO: {e}")
            price = self.sim.last_prices.get(symbol, 0)
            if price == 0:
                log.critical(f"DANGEROUS FALLBACK: All price sources for {symbol} are 0. Skipping cycle.")
                return None
            return {"bid": price, "ask": price} # Fallback to last price

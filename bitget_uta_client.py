"""
bitget_uta_client.py — Sovereign Swarm Unified Bitget UTA (V3) Execution Client
Directly handles live order placement, cancellation, TP/SL, and balance queries
on Bitget Unified Trading Accounts (UTA / V3) without legacy Classic API errors.
"""
import os
import time
import math
import json
import logging
import asyncio
import ccxt.async_support as ccxt_async
from dotenv import load_dotenv

log = logging.getLogger("bitget_uta")

class BitgetUTAClient:
    def __init__(self, api_key: str = None, secret: str = None, passphrase: str = None):
        load_dotenv(".env")
        self.api_key = api_key or os.getenv("BITGET_API_KEY") or os.getenv("API_KEY")
        self.secret = secret or os.getenv("BITGET_API_SECRET") or os.getenv("API_SECRET")
        self.passphrase = passphrase or os.getenv("BITGET_PASSPHRASE") or os.getenv("API_PASSWORD")
        
        self.exchange = ccxt_async.bitget({
            "apiKey": self.api_key,
            "secret": self.secret,
            "password": self.passphrase,
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",
                "adjustForTimeDifference": True,
                "recvWindow": 20000,
            }
        })
        self._markets_loaded = False
        self._time_synced = False
        self._lock = asyncio.Lock()

    def _normalize_symbol(self, symbol: str) -> str:
        """Converts symbol like 'BTC/USDT' or 'BTC/USDT:USDT' to Bitget format 'BTCUSDT'."""
        if not isinstance(symbol, str) or not symbol.strip() or symbol.strip().isdigit():
            raise TypeError(f"Expected valid symbol string in _normalize_symbol, got {type(symbol).__name__}: {symbol!r}")
        clean = symbol.split(":")[0].replace("/", "").upper()
        return clean

    def format_price_precision(self, symbol: str, price: float) -> str:
        """Formats price to exact Bitget UTA tick precision per asset to prevent API error 25610."""
        if price is None or price <= 0:
            return ""
        clean_sym = self._normalize_symbol(symbol)

        # 1. Inspect loaded CCXT market precision if available
        if self._markets_loaded and self.exchange.markets:
            ccxt_sym = f"{clean_sym[:-4]}/USDT:USDT" if clean_sym.endswith("USDT") else clean_sym
            market = self.exchange.markets.get(ccxt_sym) or self.exchange.markets.get(symbol)
            if market and "precision" in market and market["precision"].get("price") is not None:
                p_prec = market["precision"]["price"]
                try:
                    # If precision is float tick size (e.g. 0.001)
                    if isinstance(p_prec, float) and p_prec < 1.0:
                        dec = max(0, int(round(-math.log10(p_prec))))
                        return f"{price:.{dec}f}"
                    elif isinstance(p_prec, int):
                        return f"{price:.{p_prec}f}"
                except Exception:
                    pass

        # 2. Explicit per-asset tick precision rules for Bitget USDT-Futures
        if clean_sym in ("BTCUSDT", "BTC"):
            return f"{price:.1f}"
        elif clean_sym in ("ETHUSDT", "ETH", "BNBUSDT", "BNB", "AVAXUSDT", "AVAX"):
            return f"{price:.2f}"
        elif clean_sym in ("SOLUSDT", "SOL"):
            return f"{price:.3f}"
        elif clean_sym in ("NEARUSDT", "NEAR", "LINKUSDT", "LINK", "ADAUSDT", "ADA", "XRPUSDT", "XRP", "DOGEUSDT", "DOGE"):
            return f"{price:.4f}"

        # 3. Dynamic fallback based on price magnitude
        if price >= 10000:
            return f"{price:.1f}"
        elif price >= 10:
            return f"{price:.2f}"
        elif price >= 1:
            return f"{price:.3f}"
        else:
            return f"{price:.4f}"

    def format_qty_precision(self, symbol: str, qty: float) -> float:
        """Cleans and rounds order quantity to meet Bitget UTA futures contract step size & minimums."""
        if qty is None or qty <= 0:
            return 0.0
        clean_sym = self._normalize_symbol(symbol)

        # 1. Inspect loaded CCXT market amount precision & limits
        if self._markets_loaded and self.exchange.markets:
            ccxt_sym = f"{clean_sym[:-4]}/USDT:USDT" if clean_sym.endswith("USDT") else clean_sym
            market = self.exchange.markets.get(ccxt_sym) or self.exchange.markets.get(symbol)
            if market:
                a_prec = market.get("precision", {}).get("amount")
                min_amt = market.get("limits", {}).get("amount", {}).get("min") or 0.0
                if a_prec is not None:
                    try:
                        if isinstance(a_prec, float) and a_prec < 1.0:
                            dec = max(0, int(round(-math.log10(a_prec))))
                            rounded_qty = round(qty, dec)
                        elif isinstance(a_prec, int):
                            rounded_qty = round(qty, a_prec)
                        else:
                            rounded_qty = round(qty, 2)
                        return max(rounded_qty, float(min_amt) if min_amt > 0 else rounded_qty)
                    except Exception:
                        pass

        # 2. Explicit per-asset step size and minimums
        if "BTC" in clean_sym:
            return max(round(qty, 4), 0.0001)
        elif "ETH" in clean_sym or "BNB" in clean_sym:
            return max(round(qty, 2), 0.01)
        elif "SOL" in clean_sym or "AVAX" in clean_sym:
            return max(round(qty, 1), 0.1)
        elif any(c in clean_sym for c in ["LINK", "ADA", "XRP", "DOGE", "NEAR"]):
            return max(float(round(qty)), 1.0)
        else:
            return max(round(qty, 2), 0.1)

    def format_quantity_precision(self, symbol: str, qty: float) -> str:
        """Alias for format_qty_precision returning clean string representation."""
        val = self.format_qty_precision(symbol, qty)
        return str(int(val)) if val == int(val) else str(val)

    async def ensure_time_sync(self, force: bool = False):
        """Synchronizes exchange clock offset with Bitget server time to prevent 40008 expired timestamp errors."""
        if not self._time_synced or force:
            try:
                await self.exchange.load_time_difference()
                self._time_synced = True
                offset = self.exchange.options.get("timeDifference", 0)
                log.info(f"[BITGET-UTA] Server time synchronized: offset={offset}ms ({offset/1000.0:.2f}s)")
            except Exception as e:
                log.warning(f"[BITGET-UTA] load_time_difference warning: {e}")

    async def ensure_markets(self):
        await self.ensure_time_sync()
        if not self._markets_loaded:
            async with self._lock:
                if not self._markets_loaded:
                    try:
                        await self.exchange.load_markets()
                        self._markets_loaded = True
                        log.info(f"[BITGET-UTA] Successfully loaded {len(self.exchange.markets)} markets.")
                    except Exception as e:
                        log.error(f"[BITGET-UTA] Failed to load markets: {e}")

    async def fetch_balance(self) -> float:
        """Fetches total available USDT equity from Bitget Unified Account."""
        await self.ensure_time_sync()
        try:
            res = await self.exchange.private_uta_get_v3_account_assets()
            data = res.get("data", {})
            usdt_equity = float(data.get("usdtEquity", 0.0) or data.get("accountEquity", 0.0) or 0.0)
            return usdt_equity
        except Exception as e:
            if "40008" in str(e):
                await self.ensure_time_sync(force=True)
                try:
                    res = await self.exchange.private_uta_get_v3_account_assets()
                    data = res.get("data", {})
                    return float(data.get("usdtEquity", 0.0) or data.get("accountEquity", 0.0) or 0.0)
                except Exception:
                    pass
            log.error(f"[BITGET-UTA] Failed to fetch balance: {e}")
            return 0.0

    async def fetch_positions(self, symbol: str = None) -> list:
        """Fetches active futures positions from Bitget Unified Account."""
        await self.ensure_time_sync()
        try:
            params = {"category": "USDT-FUTURES"}
            if symbol:
                params["symbol"] = self._normalize_symbol(symbol)
            res = await self.exchange.private_uta_get_v3_position_current_position(params)
            data = res.get("data", {})
            raw_list = data.get("list") if isinstance(data, dict) else data
            return raw_list or []
        except Exception as e:
            if "40008" in str(e):
                await self.ensure_time_sync(force=True)
                try:
                    res = await self.exchange.private_uta_get_v3_position_current_position(params)
                    data = res.get("data", {})
                    raw_list = data.get("list") if isinstance(data, dict) else data
                    return raw_list or []
                except Exception as retry_err:
                    log.error(f"[BITGET-UTA] Retry fetch_positions failed: {retry_err}")
                    raise retry_err
            log.error(f"[BITGET-UTA] Failed to fetch positions: {e}")
            raise e

    async def set_leverage(self, leverage: int = None, symbol: str = "BTCUSDT", margin_mode: str = None) -> bool:
        """Sets leverage for positions on Bitget Unified Account (V3) with retry and caching."""
        import config
        try:
            clean_sym = self._normalize_symbol(symbol)
            if leverage is None:
                leverage = config.get_asset_leverage(clean_sym) if hasattr(config, "get_asset_leverage") else getattr(config, "LEVERAGE", 5)
            lev_int = int(leverage)
            if lev_int <= 0:
                raise ValueError(f"Invalid non-positive leverage: {leverage!r}")
            lev = str(lev_int)
            if margin_mode is None:
                margin_mode = getattr(config, "MARGIN_MODE", "isolated").lower()
            else:
                margin_mode = str(margin_mode).lower()

            # Initialize cache attribute if missing
            if not hasattr(self, "_leverage_cache"):
                self._leverage_cache = {}

            cache_key = f"{clean_sym}:{margin_mode}"
            now = time.time()
            cached_val, cached_time = self._leverage_cache.get(cache_key, (None, 0.0))
            if cached_val == lev_int and (now - cached_time) < 1800.0:  # 30-min cache
                log.debug(f"[BITGET-UTA] Leverage already confirmed at {lev}x on {clean_sym} ({margin_mode.upper()} cached)")
                return True

            params = {
                "category": "USDT-FUTURES",
                "symbol": clean_sym,
                "leverage": lev,
                "marginCoin": "USDT",
                "marginMode": margin_mode,
                "longLeverage": lev,
                "shortLeverage": lev
            }

            last_err = None
            for attempt in range(3):
                try:
                    res = await self.exchange.private_uta_post_v3_account_set_leverage(params)
                    log.info(f"[BITGET-UTA] Leverage successfully anchored to {lev}x on {clean_sym} ({margin_mode.upper()} DOUBLE_SIDE): {res.get('msg', 'success')}")
                    self._leverage_cache[cache_key] = (lev_int, now)
                    return True
                except Exception as e:
                    last_err = e
                    log.warning(f"[BITGET-UTA] set_leverage attempt {attempt+1}/3 failed for {clean_sym} ({lev}x): {e}")
                    if attempt < 2:
                        await asyncio.sleep(0.75 * (attempt + 1))

            log.error(f"[BITGET-UTA] set_leverage exhausted 3 attempts for {clean_sym}: {last_err}")
            return False
        except Exception as e:
            log.warning(f"[BITGET-UTA] Note on set_leverage for {symbol}: {e}")
            return False


    async def create_market_order(
        self, symbol: str, side: str, qty: float, client_oid: str = None, sl: float = 0.0, tp: float = 0.0, leverage_val: int = None
    ) -> dict:
        """
        Places a live Market Order on Bitget Unified Account with native TP/SL stop orders.
        side: 'long' (buy open) or 'short' (sell open)
        """
        await self.ensure_time_sync()
        import config
        margin_mode = getattr(config, "MARGIN_MODE", "isolated")
        clean_sym = self._normalize_symbol(symbol)
        if leverage_val is None:
            leverage_val = config.get_asset_leverage(clean_sym) if hasattr(config, "get_asset_leverage") else getattr(config, "LEVERAGE", 5)

        side_lower = side.lower()
        
        is_long = "long" in side_lower or side_lower == "buy"
        order_side = "buy" if is_long else "sell"
        pos_side = "long" if is_long else "short"
        
        # Ensure per-asset leverage is anchored prior to placement
        await self.set_leverage(leverage_val, clean_sym)

        if client_oid is None:
            client_oid = f"BW_{int(time.time() * 1000)}"

        clean_qty = self.format_qty_precision(symbol, qty)
        params = {
            "category": "USDT-FUTURES",
            "symbol": clean_sym,
            "marginMode": margin_mode,
            "marginCoin": "USDT",
            "qty": str(clean_qty),
            "side": order_side,
            "posSide": pos_side,
            "tradeSide": "open",
            "orderType": "market",
            "force": "normal",
            "clientOid": client_oid[:32]
        }
        
        if sl and sl > 0.0:
            params["presetStopLossPrice"] = self.format_price_precision(symbol, sl)
        if tp and tp > 0.0:
            params["presetTakeProfitPrice"] = self.format_price_precision(symbol, tp)
        
        try:
            log.info(f"[BITGET-UTA] Placing Live Market Order: {clean_sym} {pos_side.upper()} Qty={qty} Mode={margin_mode.upper()} Lev={leverage_val}x SL={sl} TP={tp} (clientOid={client_oid})")
            res = await self.exchange.private_uta_post_v3_trade_place_order(params)
            data = res.get("data", {})
            order_id = data.get("orderId") if isinstance(data, dict) else None
            log.info(f"[BITGET-UTA] Live Market Order SUCCESS! Order ID: {order_id}")

            # Register native exchange-side Strategy Orders (TP/SL) on Bitget's matching engine
            opp_side = "sell" if is_long else "buy"
            strat_ids = []
            if sl and sl > 0.0:
                # F-08 FIX: Give Bitget matching engine 1.5s to settle the filled position before attaching strategy orders
                await asyncio.sleep(1.5)
                sl_attached = False
                last_strat_err = None
                for sl_attempt in range(5):
                    try:
                        # Re-confirm actual filled position size from exchange to prevent 25610 precision mismatch
                        effective_qty = qty
                        try:
                            live_pos = await self.fetch_positions(symbol=clean_sym)
                            pos_match = [p for p in (live_pos or []) if str(p.get("posSide", "long")).lower() == pos_side]
                            if pos_match:
                                real_tot = float(pos_match[0].get("total", 0.0) or pos_match[0].get("contracts", 0.0) or 0.0)
                                if real_tot > 0:
                                    effective_qty = real_tot
                        except Exception as pos_fetch_err:
                            log.debug(f"[BITGET-UTA] Note fetching live position for SL precision: {pos_fetch_err}")

                        sl_str = self.format_price_precision(symbol, sl)
                        qty_str = self.format_quantity_precision(symbol, effective_qty)
                        sl_params = {
                            "category": "USDT-FUTURES",
                            "symbol": clean_sym,
                            "marginMode": margin_mode,
                            "marginCoin": "USDT",
                            "planType": "pos_loss",
                            "posSide": pos_side,
                            "side": opp_side,
                            "triggerPrice": sl_str,
                            "stopLoss": sl_str,
                            "triggerType": "mark_price",
                            "size": qty_str
                        }
                        strat_res = await self.exchange.private_uta_post_v3_trade_place_strategy_order(sl_params)
                        sl_id = strat_res.get("data", {}).get("orderId")
                        strat_ids.append(f"SL:{sl_id}")
                        log.info(f"[BITGET-UTA] [SHIELD] Native Stop Loss REGISTERED on Bitget! ID: {sl_id} (SL={sl_str}, Qty={qty_str})")
                        sl_attached = True
                        break
                    except Exception as strat_err:
                        last_strat_err = strat_err
                        log.warning(f"[BITGET-UTA] Native Stop Loss attempt {sl_attempt+1}/5 failed: {strat_err}")
                        await asyncio.sleep(1.5 + 1.0 * sl_attempt)

                if not sl_attached:
                    log.critical(f"[BITGET-UTA] 🚨 CRITICAL STOP-LOSS FAILURE: Could not register native SL on {clean_sym} after 5 attempts ({last_strat_err})! Aborting position to eliminate naked capital exposure.")
                    close_success = False
                    abort_err_str = "None"
                    try:
                        await self.close_position_order(symbol=clean_sym, side=pos_side, qty=qty)
                        log.warning(f"[BITGET-UTA] Emergency position abort SUCCESS for {clean_sym}.")
                        close_success = True
                    except Exception as abort_err:
                        abort_err_str = str(abort_err)
                        log.error(f"[BITGET-UTA] Emergency position abort failed for {clean_sym}: {abort_err}")
                    
                    # M-1 FIX: Write emergency state row when BOTH SL and close fail
                    if not close_success:
                        try:
                            import sqlite3
                            from datetime import datetime, timezone
                            now_iso = datetime.now(timezone.utc).isoformat()
                            
                            with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                                conn.execute("""
                                    INSERT INTO system_state (key, value, updated_at)
                                    VALUES (?, ?, ?)
                                    ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?
                                """, (
                                    f"emergency_sl_failure_{clean_sym}",
                                    json.dumps({
                                        "symbol": clean_sym,
                                        "side": pos_side,
                                        "qty": qty,
                                        "sl_error": str(last_strat_err),
                                        "close_error": abort_err_str,
                                        "timestamp": now_iso,
                                        "status": "UNPROTECTED_NAKED_POSITION"
                                    }),
                                    now_iso,
                                    json.dumps({
                                        "symbol": clean_sym,
                                        "side": pos_side,
                                        "qty": qty,
                                        "sl_error": str(last_strat_err),
                                        "close_error": abort_err_str,
                                        "timestamp": now_iso,
                                        "status": "UNPROTECTED_NAKED_POSITION"
                                    }),
                                    now_iso
                                ))
                                conn.commit()
                                log.critical(f"[EMERGENCY-STATE] Persisted critical failure record for {clean_sym} — manual intervention required!")
                        except Exception as state_err:
                            log.critical(f"[EMERGENCY-STATE] Failed to persist emergency state for {clean_sym}: {state_err}")
                    
                    raise RuntimeError(f"Native Stop Loss placement failed on {clean_sym} after 3 retries ({last_strat_err}). Position was {'aborted' if close_success else 'NOT aborted - UNPROTECTED'} to prevent naked loss.")

            if tp and tp > 0.0:
                try:
                    tp_str = self.format_price_precision(symbol, tp)
                    qty_str = self.format_quantity_precision(symbol, qty)
                    tp_params = {
                        "category": "USDT-FUTURES",
                        "symbol": clean_sym,
                        "marginMode": margin_mode,
                        "marginCoin": "USDT",
                        "planType": "pos_profit",
                        "posSide": pos_side,
                        "side": opp_side,
                        "triggerPrice": tp_str,
                        "takeProfit": tp_str,
                        "triggerType": "mark_price",
                        "size": qty_str
                    }
                    strat_res_tp = await self.exchange.private_uta_post_v3_trade_place_strategy_order(tp_params)
                    tp_id = strat_res_tp.get("data", {}).get("orderId")
                    strat_ids.append(f"TP:{tp_id}")
                    log.info(f"[BITGET-UTA] [TARGET] Native Take Profit REGISTERED on Bitget! ID: {tp_id} (TP={tp})")
                except Exception as tp_err:
                    log.warning(f"[BITGET-UTA] Native Take Profit placement note: {tp_err}")

            return {"id": order_id, "orderId": order_id, "symbol": clean_sym, "side": pos_side, "qty": qty, "raw": res}
        except Exception as e:
            log.error(f"[BITGET-UTA] CRITICAL: Live Market Order Failed on {clean_sym}: {e}")
            raise e

    async def create_limit_order(
        self, symbol: str, side: str, qty: float, price: float, client_oid: str = None, sl: float = 0.0, tp: float = 0.0, leverage_val: int = None
    ) -> dict:
        """Places a Limit Order on Bitget Unified Account with native TP/SL stop orders."""
        await self.ensure_time_sync()
        import config
        margin_mode = getattr(config, "MARGIN_MODE", "isolated")
        clean_sym = self._normalize_symbol(symbol)
        if leverage_val is None:
            leverage_val = config.get_asset_leverage(clean_sym) if hasattr(config, "get_asset_leverage") else getattr(config, "LEVERAGE", 5)

        side_lower = side.lower()
        
        is_long = "long" in side_lower or side_lower == "buy"
        order_side = "buy" if is_long else "sell"
        pos_side = "long" if is_long else "short"
        
        # Ensure per-asset leverage is anchored prior to placement
        await self.set_leverage(leverage_val, clean_sym)

        if client_oid is None:
            client_oid = f"BW_{int(time.time() * 1000)}"

        clean_qty = self.format_qty_precision(symbol, qty)
        clean_price = self.format_price_precision(symbol, price)

        params = {
            "category": "USDT-FUTURES",
            "symbol": clean_sym,
            "marginMode": margin_mode,
            "marginCoin": "USDT",
            "qty": str(clean_qty),
            "price": clean_price,
            "side": order_side,
            "posSide": pos_side,
            "tradeSide": "open",
            "orderType": "limit",
            "force": "normal",
            "clientOid": client_oid[:32]
        }
        
        if sl and sl > 0.0:
            params["presetStopLossPrice"] = self.format_price_precision(symbol, sl)
        if tp and tp > 0.0:
            params["presetTakeProfitPrice"] = self.format_price_precision(symbol, tp)
        
        try:
            res = await self.exchange.private_uta_post_v3_trade_place_order(params)
            data = res.get("data", {})
            order_id = data.get("orderId") if isinstance(data, dict) else None
            return {"id": order_id, "symbol": clean_sym, "side": pos_side, "qty": qty, "price": price, "raw": res}
        except Exception as e:
            log.error(f"[BITGET-UTA] Limit Order Failed on {clean_sym}: {e}")
            raise e

    async def fetch_order(self, symbol: str, order_id: str, category: str = "USDT-FUTURES") -> dict:
        """
        Fetches status, fill quantity, and execution details of an order on Bitget Unified Account.
        Used to prevent race-condition orphan fills before issuing cancellations.
        """
        await self.ensure_time_sync()
        clean_sym = self._normalize_symbol(symbol)
        params = {
            "category": category,
            "symbol": clean_sym,
            "orderId": str(order_id)
        }
        try:
            res = await self.exchange.private_uta_get_v3_trade_order_info(params)
            data = res.get("data", {})
            if isinstance(data, dict):
                # Standardize fill metrics across UTA endpoints
                cum_exec = float(data.get("cumExecQty", 0.0) or data.get("dealQty", 0.0) or (data.get("baseVolume", 0.0) if data.get("status") == "filled" else 0.0))
                avg_p = float(data.get("dealAvgPrice", 0.0) or data.get("price", 0.0) or 0.0)
                status = data.get("status", "unknown")
                data["filled_qty"] = cum_exec
                data["deal_avg_price"] = avg_p
                data["order_status"] = status
                return data
            return {}
        except Exception as e:
            try:
                ccxt_order = await self.exchange.fetch_order(str(order_id), symbol=clean_sym, params={"category": category})
                if ccxt_order:
                    ccxt_order["filled_qty"] = float(ccxt_order.get("filled", 0.0) or 0.0)
                    ccxt_order["deal_avg_price"] = float(ccxt_order.get("average", 0.0) or ccxt_order.get("price", 0.0) or 0.0)
                    ccxt_order["order_status"] = ccxt_order.get("status", "unknown")
                    return ccxt_order
            except Exception:
                pass
            log.error(f"[BITGET-UTA] Failed to fetch order {order_id} on {clean_sym}: {e}")
            raise e

    async def create_post_only_order(
        self, symbol: str, side: str, qty: float, price: float = None,
        client_oid: str = None, sl: float = 0.0, tp: float = 0.0, leverage_val: int = None, margin_mode: str = None
    ) -> dict:
        """
        Places a Maker Post-Only Limit Order on Bitget Unified Account.
        Post-only means the order cannot cross as a taker: it rests on the book or the exchange cancels it.
        This method books no fee of its own — the calling engine does, at the maker rate (0.02%, against the
        0.06% taker rate it books on a market close; `swing_trader.py:2073`, `:2877-2878`).
        If the order would execute immediately as a taker, the exchange cancels it automatically.
        """
        await self.ensure_time_sync()
        import config
        if margin_mode is None:
            margin_mode = getattr(config, "MARGIN_MODE", "isolated")
        clean_sym = self._normalize_symbol(symbol)
        if leverage_val is None:
            leverage_val = config.get_asset_leverage(clean_sym) if hasattr(config, "get_asset_leverage") else getattr(config, "LEVERAGE", 5)

        side_lower = side.lower()
        is_long = "long" in side_lower or side_lower == "buy"
        order_side = "buy" if is_long else "sell"
        pos_side = "long" if is_long else "short"

        # If limit price is not provided, fetch current order book inside best bid/ask
        if price is None or price <= 0.0:
            try:
                ob = await self.exchange.fetch_order_book(clean_sym, limit=5)
                bids = ob.get("bids", [])
                asks = ob.get("asks", [])
                if is_long and bids:
                    price = float(bids[0][0])
                elif not is_long and asks:
                    price = float(asks[0][0])
                else:
                    ticker = await self.exchange.fetch_ticker(clean_sym)
                    price = float(ticker.get("bid", ticker.get("last", 0.0))) if is_long else float(ticker.get("ask", ticker.get("last", 0.0)))
            except Exception as pe:
                log.warning(f"[BITGET-UTA] Failed to fetch inside spread for post-only order: {pe}")
                ticker = await self.exchange.fetch_ticker(clean_sym)
                price = float(ticker.get("last", 0.0))

        await self.set_leverage(leverage_val, clean_sym, margin_mode=margin_mode)

        if client_oid is None:
            client_oid = f"BW_PO_{int(time.time() * 1000)}"

        # Clean quantity for minimums
        if "BTC" in clean_sym:
            qty = max(round(qty, 4), 0.0001)
        elif "ETH" in clean_sym or "BNB" in clean_sym:
            qty = max(round(qty, 2), 0.01)
        elif "SOL" in clean_sym or "AVAX" in clean_sym:
            qty = max(round(qty, 1), 0.1)
        elif any(c in clean_sym for c in ["LINK", "ADA", "XRP", "DOGE", "NEAR"]):
            qty = max(float(round(qty)), 1.0)
        else:
            qty = max(round(qty, 2), 0.1)

        price_str = self.format_price_precision(symbol, price)

        params = {
            "category": "USDT-FUTURES",
            "symbol": clean_sym,
            "marginMode": margin_mode.lower(),
            "marginCoin": "USDT",
            "qty": str(qty),
            "price": price_str,
            "side": order_side,
            "posSide": pos_side,
            "tradeSide": "open",
            "orderType": "limit",
            "force": "post_only",
            "clientOid": client_oid[:32]
        }

        if sl and sl > 0.0:
            params["presetStopLossPrice"] = self.format_price_precision(symbol, sl)
        if tp and tp > 0.0:
            params["presetTakeProfitPrice"] = self.format_price_precision(symbol, tp)

        try:
            log.info(f"[BITGET-UTA] Placing Maker Post-Only Order: {clean_sym} {pos_side.upper()} Qty={qty} Price={price_str} Lev={leverage_val}x (force=post_only)")
            res = await self.exchange.private_uta_post_v3_trade_place_order(params)
            data = res.get("data", {})
            order_id = data.get("orderId") if isinstance(data, dict) else None
            log.info(f"[BITGET-UTA] Post-Only Order SUCCESS! Order ID: {order_id}")
            return {
                "id": order_id,
                "symbol": clean_sym,
                "side": pos_side,
                "qty": qty,
                "price": float(price_str),
                "type": "post_only",
                "clientOid": client_oid[:32],
                "raw": res
            }
        except Exception as e:
            log.error(f"[BITGET-UTA] Post-Only Order Failed on {clean_sym}: {e}")
            raise e

    async def close_position_order(
        self, symbol: str, side: str, qty: float
    ) -> dict:
        """
        Places a Market Close Order on Bitget Unified Account.
        side: 'long' (closes by selling) or 'short' (closes by buying)
        """
        await self.ensure_time_sync()
        import config
        margin_mode = getattr(config, "MARGIN_MODE", "isolated")

        clean_sym = self._normalize_symbol(symbol)
        side_lower = side.lower()
        
        is_long = "long" in side_lower or side_lower == "buy"
        order_side = "sell" if is_long else "buy"
        pos_side = "long" if is_long else "short"
        client_oid = f"BW_CLS_{int(time.time() * 1000)}"

        # Clamp and clean quantity to Bitget lot size and live position amount
        try:
            live_positions = await self.fetch_positions(symbol=clean_sym)
            for lp in live_positions:
                lp_side = (lp.get("posSide") or lp.get("holdSide") or "").lower()
                if lp_side == pos_side:
                    live_total = float(lp.get("total", 0.0) or lp.get("available", 0.0) or 0.0)
                    if live_total > 0.0:
                        # Only clamp up to live_total when intent is a full or near-full close (within rounding tolerance)
                        if qty >= (live_total * 0.98):
                            qty = live_total
                        break
        except Exception as q_err:
            log.warning(f"[BITGET-UTA] Position size alignment note: {q_err}")

        # Format precision per coin for Bitget futures minimums
        if "BTC" in clean_sym:
            qty = max(round(qty, 4), 0.0001)
        elif "ETH" in clean_sym or "BNB" in clean_sym:
            qty = max(round(qty, 2), 0.01)
        elif "SOL" in clean_sym or "AVAX" in clean_sym:
            qty = max(round(qty, 1), 0.1)
        elif any(c in clean_sym for c in ["LINK", "ADA", "XRP", "DOGE", "NEAR"]):
            qty = max(float(round(qty)), 1.0)
        else:
            qty = max(round(qty, 2), 0.1)

        params = {
            "category": "USDT-FUTURES",
            "symbol": clean_sym,
            "marginMode": margin_mode,
            "marginCoin": "USDT",
            "qty": str(qty),
            "side": order_side,
            "posSide": pos_side,
            "tradeSide": "close",
            "orderType": "market",
            "force": "normal",
            "clientOid": client_oid[:32]
        }
        
        try:
            log.info(f"[BITGET-UTA] Closing Position: {clean_sym} {pos_side.upper()} Qty={qty}")
            res = await self.exchange.private_uta_post_v3_trade_place_order(params)
            data = res.get("data", {})
            order_id = data.get("orderId") if isinstance(data, dict) else None
            log.info(f"[BITGET-UTA] Position Close SUCCESS! Order ID: {order_id}")
            return {"id": order_id, "symbol": clean_sym, "side": pos_side, "qty": qty, "raw": res}
        except Exception as e:
            err_str = str(e)
            if "25227" in err_str or "no position available to close" in err_str.lower():
                log.info(f"[BITGET-UTA] Position on {clean_sym} ({pos_side.upper()}) already closed on exchange (code 25227). Proceeding to ledger settlement.")
                return {"id": None, "already_closed": True, "symbol": clean_sym, "side": pos_side, "qty": qty, "raw": {"code": "25227"}}
            log.error(f"[BITGET-UTA] Close Position Failed on {clean_sym}: {e}")
            raise e

    async def create_spot_order(
        self, symbol: str, side: str, qty: float, price: float = None, post_only: bool = True, client_oid: str = None
    ) -> dict:
        """
        Places a Spot Order on Bitget Unified Account (V3).
        side: 'buy' or 'sell'
        Supports maker post_only limit order or market order.
        """
        await self.ensure_time_sync()
        clean_sym = self._normalize_symbol(symbol)
        side_lower = side.lower()
        order_side = "buy" if "buy" in side_lower or "long" in side_lower else "sell"
        
        # Format precision per coin for Bitget spot minimums
        if "BTC" in clean_sym:
            qty = max(round(qty, 4), 0.0001)
        elif "ETH" in clean_sym:
            qty = max(round(qty, 3), 0.001)
        else:
            qty = round(qty, 2)

        if client_oid is None:
            client_oid = f"BW_SPOT_{int(time.time() * 1000)}"

        params = {
            "category": "SPOT",
            "symbol": clean_sym,
            "side": order_side,
            "orderType": "limit" if (price and price > 0.0) else "market",
            "force": "post_only" if (post_only and price and price > 0.0) else "normal",
            "qty": str(qty),
            "clientOid": client_oid[:32]
        }
        if price and price > 0.0:
            params["price"] = self.format_price_precision(symbol, price)

        try:
            log.info(f"[BITGET-UTA] Placing Spot Order: {clean_sym} {order_side.upper()} Qty={qty} Price={params.get('price', 'MARKET')} (force={params['force']})")
            res = await self.exchange.private_uta_post_v3_trade_place_order(params)
            data = res.get("data", {})
            order_id = data.get("orderId") if isinstance(data, dict) else None
            log.info(f"[BITGET-UTA] Spot Order SUCCESS! Order ID: {order_id}")
            return {
                "id": order_id,
                "symbol": clean_sym,
                "side": order_side,
                "qty": qty,
                "price": float(params.get("price", 0.0)),
                "type": "spot",
                "clientOid": client_oid[:32],
                "raw": res
            }
        except Exception as e:
            log.error(f"[BITGET-UTA] Spot Order Failed on {clean_sym}: {e}")
            raise e

    async def close_spot_order(
        self, symbol: str, qty: float, price: float = None, post_only: bool = False
    ) -> dict:
        """Sells spot holdings to unwind position."""
        return await self.create_spot_order(symbol, side="sell", qty=qty, price=price, post_only=post_only)

    async def cancel_order(self, symbol: str, order_id: str, category: str = "USDT-FUTURES") -> dict:
        """Cancels an open order on Bitget Unified Account."""
        clean_sym = self._normalize_symbol(symbol)
        try:
            res = await self.exchange.private_uta_post_v3_trade_cancel_order({
                "category": category,
                "symbol": clean_sym,
                "orderId": str(order_id)
            })
            return res
        except Exception as e:
            log.error(f"[BITGET-UTA] Cancel Order Failed for {order_id} on {clean_sym}: {e}")
            raise e

    async def get_order_fills(self, symbol: str, order_id: str = None, client_oid: str = None) -> list[dict]:
        """
        Fetches individual execution fills for a specific order on Bitget Unified Account (V3).
        Enables exact aggregation of multi-tranche / split fills.
        """
        await self.ensure_time_sync()
        clean_sym = self._normalize_symbol(symbol)
        try:
            params = {"category": "USDT-FUTURES", "symbol": clean_sym, "limit": "50"}
            if order_id:
                params["orderId"] = str(order_id)
            if client_oid:
                params["clientOid"] = str(client_oid)[:32]
            res = await self.exchange.private_uta_get_v3_trade_fills(params)
            data = res.get("data", {})
            raw_fills = data.get("list", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            fills = [
                f for f in raw_fills
                if self._normalize_symbol(str(f.get("symbol", ""))) == clean_sym
            ]
            return fills
        except Exception as e:
            log.warning(f"[BITGET-UTA] get_order_fills note for {clean_sym} ({order_id}): {e}")
            return []

    async def get_recent_fills(self, symbol: str, lookback_mins: int = 30, since_ms: int = None) -> list[dict]:
        """
        Fetches all recent execution fills for a symbol within lookback window or since explicit timestamp.
        """
        await self.ensure_time_sync()
        clean_sym = self._normalize_symbol(symbol)
        try:
            if since_ms is None:
                since_ms = int((time.time() - (lookback_mins * 60)) * 1000)
            params = {"category": "USDT-FUTURES", "symbol": clean_sym, "startTime": str(since_ms), "limit": "50"}
            res = await self.exchange.private_uta_get_v3_trade_fills(params)
            data = res.get("data", {})
            raw_fills = data.get("list", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            fills = [
                f for f in raw_fills
                if self._normalize_symbol(str(f.get("symbol", ""))) == clean_sym
            ]
            return fills
        except Exception as e:
            log.warning(f"[BITGET-UTA] get_recent_fills note for {clean_sym}: {e}")
            return []

    @staticmethod
    def _plan_family(order: dict) -> str:
        """Bitget rests each leg as its own plan; classify it so a leg is only ever
        rewritten by the caller that owns it."""
        blob = f"{order.get('planType', '')} {order.get('orderType', '')}".lower()
        if "profit" in blob:
            return "profit"
        if "track" in blob or "moving" in blob:
            return "trailing"
        if "loss" in blob:
            return "loss"
        return "other"

    async def _safe_cancel(self, clean_sym: str, order_id: str) -> None:
        try:
            await self.exchange.private_uta_post_v3_trade_cancel_order({
                "category": "USDT-FUTURES", "symbol": clean_sym, "orderId": str(order_id)})
            log.warning(f"[BITGET-UTA] Dropped stale plan {order_id} on {clean_sym} before re-registering")
        except Exception as e:
            log.error(f"[BITGET-UTA] Could not cancel stale plan {order_id} on {clean_sym}: {e}")

    async def _rebuild_strategy_leg(self, symbol: str, clean_sym: str, family: str,
                                    field: str, priced: str, plans: list) -> bool:
        """Re-register a leg that is no longer resting, sizing it against the LIVE
        position so a partial take-profit is reflected rather than the entry size."""
        import config
        sibling = next((o for o in plans if isinstance(o, dict) and o.get("posSide")), None)
        pos_side = str(sibling.get("posSide")).lower() if sibling else None
        size = 0.0
        try:
            positions = await self.fetch_positions(symbol=clean_sym)
            live = [p for p in (positions or [])
                    if float(p.get("total", 0.0) or p.get("contracts", 0.0) or 0.0) > 0]
            if pos_side:
                matching = [p for p in live if str(p.get("posSide", "")).lower() == pos_side]
                if matching:
                    live = matching
            if live:
                pos_side = str(live[0].get("posSide", pos_side or "long")).lower()
                size = float(live[0].get("total", 0.0) or live[0].get("contracts", 0.0) or 0.0)
        except Exception as e:
            log.warning(f"[BITGET-UTA] Live position lookup failed for {clean_sym}: {e}")
        if size <= 0 and sibling:
            try:
                size = float(sibling.get("size", 0.0) or 0.0)
            except (TypeError, ValueError):
                size = 0.0

        if not pos_side or size <= 0:
            log.critical(f"[BITGET-UTA] 🚨 [STOP-COVERAGE] {clean_sym} has no resting {family} plan "
                         f"and no live position to re-register one against. The {family} leg at "
                         f"${priced} is NOT protected on-exchange — manual check required.")
            return False

        params = {
            "category": "USDT-FUTURES",
            "symbol": clean_sym,
            "marginMode": getattr(config, "MARGIN_MODE", "isolated"),
            "marginCoin": "USDT",
            "planType": "pos_loss" if family == "loss" else "pos_profit",
            "posSide": pos_side,
            "side": "sell" if pos_side == "long" else "buy",
            "triggerPrice": priced,
            field: priced,
            "triggerType": "mark_price",
            "size": self.format_quantity_precision(clean_sym, size)
        }
        try:
            res = await self.exchange.private_uta_post_v3_trade_place_strategy_order(params)
            code = str(res.get("code", ""))
        except Exception as exc:
            code, res = str(exc), {"error": str(exc)}

        if code in ("00000", "25600"):
            new_id = (res.get("data") or {}).get("orderId") if isinstance(res.get("data"), dict) else None
            log.warning(f"[BITGET-UTA] [REBUILD] {clean_sym} {family} plan was missing — "
                        f"re-registered at ${priced} (OrderId: {new_id})")
            return True

        log.critical(f"[BITGET-UTA] 🚨 [STOP-COVERAGE] {clean_sym} could not re-register the "
                     f"{family} plan at ${priced} (code {code}): {res}")
        return False

    async def _sync_strategy_leg(self, symbol: str, clean_sym: str, family: str,
                                 field: str, price: float, plans: list) -> bool:
        """Move one leg of the shield. Only plans of the same family are touched, and
        triggerPrice travels with the leg value because that is what the engine fires on."""
        priced = self.format_price_precision(symbol, price)
        targets = [o for o in plans if self._plan_family(o) == family]

        if not targets:
            if family == "loss" and any(self._plan_family(o) == "trailing" for o in plans):
                log.warning(f"[BITGET-UTA] {clean_sym} stop leg is held by a TRAILING plan; "
                            f"not overwriting it with a flat stop (${priced}).")
                return False
            return await self._rebuild_strategy_leg(symbol, clean_sym, family, field, priced, plans)

        synced = False
        rejected = []
        for o in targets:
            order_id = o.get("orderId")
            mod_params = {
                "category": "USDT-FUTURES",
                "symbol": clean_sym,
                "orderId": order_id,
                "triggerPrice": priced,
                field: priced
            }
            try:
                res = await self.exchange.private_uta_post_v3_trade_modify_strategy_order(mod_params)
                code = str(res.get("code", ""))
            except Exception as exc:
                code, res = str(exc), {"error": str(exc)}

            if code == "00000":
                log.info(f"[BITGET-UTA] [SHIELD-SYNC] {family} plan {order_id} on {clean_sym} moved to ${priced}")
                synced = True
            elif code == "25600":
                log.debug(f"[BITGET-UTA] [SHIELD-SYNC] {family} plan {order_id} on {clean_sym} already at ${priced}")
                synced = True
            else:
                log.warning(f"[BITGET-UTA] {family} plan modify rejected on {clean_sym} "
                            f"(OrderId {order_id}, code {code}): {res}")
                rejected.append(order_id)

        for order_id in rejected:
            await self._safe_cancel(clean_sym, order_id)

        if synced:
            return True
        return await self._rebuild_strategy_leg(symbol, clean_sym, family, field, priced, plans)

    async def get_resting_plan_families(self, symbol: str) -> set:
        """
        Returns which TP/SL plan families are currently resting on the exchange for a
        symbol — e.g. {"loss", "profit"}. Callers use this to confirm the on-exchange
        shield is actually live before relying on it. Raises if the book cannot be read.
        """
        clean_sym = self._normalize_symbol(symbol)
        unfilled = await self.exchange.private_uta_get_v3_trade_unfilled_strategy_orders(
            {"category": "USDT-FUTURES", "symbol": clean_sym})
        raw = unfilled.get("data", []) if isinstance(unfilled, dict) else []
        if isinstance(raw, dict):
            raw = raw.get("list") or raw.get("orders") or []
        return {self._plan_family(o) for o in (raw or [])
                if isinstance(o, dict) and o.get("symbol") == clean_sym}

    async def modify_strategy_order(self, symbol: str, sl: float = None, tp: float = None) -> bool:
        """
        Syncs the native exchange-side TP/SL plans for a symbol so ratcheted stops and
        moved targets are live on the Bitget matching engine, eliminating polling lag.

        The two legs rest as separate plans (pos_loss / pos_profit), so each requested
        leg is routed only to its own family. A leg with no resting plan is rebuilt from
        the live position; a leg that cannot be rebuilt is reported as stop-coverage loss.
        """
        await self.ensure_time_sync()
        clean_sym = self._normalize_symbol(symbol)

        legs = []
        if sl is not None and sl > 0:
            legs.append(("loss", "stopLoss", float(sl)))
        if tp is not None and tp > 0:
            legs.append(("profit", "takeProfit", float(tp)))
        if not legs:
            return False

        try:
            unfilled = await self.exchange.private_uta_get_v3_trade_unfilled_strategy_orders(
                {"category": "USDT-FUTURES", "symbol": clean_sym})
        except Exception as e:
            log.error(f"[BITGET-UTA] Cannot read strategy plans for {clean_sym}: {e}")
            return False

        raw = unfilled.get("data", []) if isinstance(unfilled, dict) else []
        if isinstance(raw, dict):
            raw = raw.get("list") or raw.get("orders") or []
        plans = [o for o in (raw or []) if isinstance(o, dict) and o.get("symbol") == clean_sym]

        if not plans:
            log.debug(f"[BITGET-UTA] No unfilled strategy plans listed for {clean_sym}; "
                      f"re-registering the requested legs.")

        all_synced = True
        for family, field, price in legs:
            if not await self._sync_strategy_leg(symbol, clean_sym, family, field, price, plans):
                all_synced = False
        return all_synced

    async def get_funding_history(self, symbol: str, since_ms: int = None, limit: int = 50) -> list[dict]:
        """
        Fetches historical funding fee payments for symbol since since_ms.
        Returns list of funding records or empty list if unavailable.
        """
        await self.ensure_time_sync()
        clean_sym = self._normalize_symbol(symbol)
        try:
            ccxt_sym = f"{clean_sym[:-4]}/USDT:USDT" if clean_sym.endswith("USDT") else clean_sym
            if hasattr(self.exchange, "fetch_funding_history"):
                res = await self.exchange.fetch_funding_history(symbol=ccxt_sym, since=since_ms, limit=limit)
                return res or []
        except Exception as e:
            log.warning(f"[BITGET-UTA] fetch_funding_history warning for {clean_sym}: {e}")
        return []

    async def close(self):
        """Closes the underlying aiohttp connection."""
        try:
            await self.exchange.close()
        except:
            pass

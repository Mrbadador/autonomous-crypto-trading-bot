"""
bitget_uta_client.py — Sovereign Swarm Unified Bitget UTA (V3) Execution Client
Directly handles live order placement, cancellation, TP/SL, and balance queries
on Bitget Unified Trading Accounts (UTA / V3) without legacy Classic API errors.
"""
import os
import time
import math
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
        """Sets leverage for positions on Bitget Unified Account (V3)."""
        import config
        clean_sym = self._normalize_symbol(symbol)
        if leverage is None:
            leverage = config.get_asset_leverage(clean_sym) if hasattr(config, "get_asset_leverage") else getattr(config, "LEVERAGE", 5)
        lev = str(leverage)
        if margin_mode is None:
            margin_mode = getattr(config, "MARGIN_MODE", "isolated").lower()
        else:
            margin_mode = margin_mode.lower()
        try:
            params = {
                "category": "USDT-FUTURES",
                "symbol": clean_sym,
                "leverage": lev,
                "marginCoin": "USDT",
                "marginMode": margin_mode,
                "longLeverage": lev,
                "shortLeverage": lev
            }
            res = await self.exchange.private_uta_post_v3_account_set_leverage(params)
            log.info(f"[BITGET-UTA] Leverage successfully anchored to {lev}x on {clean_sym} ({margin_mode.upper()} DOUBLE_SIDE): {res.get('msg', 'success')}")
            return True
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
                sl_attached = False
                last_strat_err = None
                for sl_attempt in range(3):
                    try:
                        sl_str = self.format_price_precision(symbol, sl)
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
                            "size": str(qty)
                        }
                        strat_res = await self.exchange.private_uta_post_v3_trade_place_strategy_order(sl_params)
                        sl_id = strat_res.get("data", {}).get("orderId")
                        strat_ids.append(f"SL:{sl_id}")
                        log.info(f"[BITGET-UTA] [SHIELD] Native Stop Loss REGISTERED on Bitget! ID: {sl_id} (SL={sl_str})")
                        sl_attached = True
                        break
                    except Exception as strat_err:
                        last_strat_err = strat_err
                        log.warning(f"[BITGET-UTA] Native Stop Loss attempt {sl_attempt+1}/3 failed: {strat_err}")
                        await asyncio.sleep(0.5 * (sl_attempt + 1))

                if not sl_attached:
                    log.critical(f"[BITGET-UTA] 🚨 CRITICAL STOP-LOSS FAILURE: Could not register native SL on {clean_sym} after 3 attempts ({last_strat_err})! Aborting position to eliminate naked capital exposure.")
                    try:
                        await self.close_position_order(symbol=clean_sym, side=pos_side, qty=qty)
                        log.warning(f"[BITGET-UTA] Emergency position abort SUCCESS for {clean_sym}.")
                    except Exception as abort_err:
                        log.error(f"[BITGET-UTA] Emergency position abort failed for {clean_sym}: {abort_err}")
                    raise RuntimeError(f"Native Stop Loss placement failed on {clean_sym} after 3 retries ({last_strat_err}). Position was aborted to prevent naked loss.")

            if tp and tp > 0.0:
                try:
                    tp_str = self.format_price_precision(symbol, tp)
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
                        "size": str(qty)
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
        Guarantees maker rebate capture and eliminates taker fee drag (0.02% maker vs 0.04% taker).
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
                        # If closing full or close to full position, use live_total
                        if qty >= (live_total * 0.8) or qty < 0.01:
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
            fills = data.get("list", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            return fills
        except Exception as e:
            log.warning(f"[BITGET-UTA] get_order_fills note for {clean_sym} ({order_id}): {e}")
            return []

    async def get_recent_fills(self, symbol: str, lookback_mins: int = 30) -> list[dict]:
        """
        Fetches all recent execution fills for a symbol within lookback window.
        """
        await self.ensure_time_sync()
        clean_sym = self._normalize_symbol(symbol)
        try:
            since_ms = int((time.time() - (lookback_mins * 60)) * 1000)
            params = {"category": "USDT-FUTURES", "symbol": clean_sym, "startTime": str(since_ms), "limit": "50"}
            res = await self.exchange.private_uta_get_v3_trade_fills(params)
            data = res.get("data", {})
            fills = data.get("list", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            return fills
        except Exception as e:
            log.warning(f"[BITGET-UTA] get_recent_fills note for {clean_sym}: {e}")
            return []

    async def modify_strategy_order(self, symbol: str, sl: float = None, tp: float = None) -> bool:
        """
        Updates native exchange-side strategy order (TP/SL) on Bitget UTA matching engine.
        Ensures ratcheted trailing stops are active directly on-exchange, eliminating polling lag.
        """
        await self.ensure_time_sync()
        clean_sym = self._normalize_symbol(symbol)
        try:
            params = {"category": "USDT-FUTURES", "symbol": clean_sym}
            unfilled = await self.exchange.private_uta_get_v3_trade_unfilled_strategy_orders(params)
            orders = unfilled.get("data", [])
            matching_orders = [o for o in orders if o.get("symbol") == clean_sym]
            if not matching_orders:
                log.debug(f"[BITGET-UTA] No unfilled strategy order found to modify for {clean_sym}")
                return False

            success = False
            for o in matching_orders:
                order_id = o.get("orderId")
                mod_params = {
                    "category": "USDT-FUTURES",
                    "symbol": clean_sym,
                    "orderId": order_id
                }
                if sl is not None and sl > 0:
                    mod_params["stopLoss"] = self.format_price_precision(symbol, sl)
                if tp is not None and tp > 0:
                    mod_params["takeProfit"] = self.format_price_precision(symbol, tp)

                res = await self.exchange.private_uta_post_v3_trade_modify_strategy_order(mod_params)
                code = str(res.get("code", ""))
                if code == "00000":
                    log.info(f"[BITGET-UTA] [SHIELD-SYNC] Successfully synced exchange-side Stop Loss to ${sl} for {clean_sym} (OrderId: {order_id})")
                    success = True
                elif code == "25600":
                    log.debug(f"[BITGET-UTA] [SHIELD-SYNC] Exchange-side TP/SL already up to date on Bitget for {clean_sym}")
                    success = True
                else:
                    log.warning(f"[BITGET-UTA] Strategy order modify response for {clean_sym}: {res}")
            return success
        except Exception as e:
            if "25600" in str(e):
                log.debug(f"[BITGET-UTA] [SHIELD-SYNC] Exchange-side TP/SL already up to date on Bitget for {clean_sym}")
                return True
            log.error(f"[BITGET-UTA] Error modifying strategy order for {clean_sym}: {e}")
            return False

    async def close(self):
        """Closes the underlying aiohttp connection."""
        try:
            await self.exchange.close()
        except:
            pass

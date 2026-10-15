import asyncio
import sqlite3
import os
import logging
import time
from datetime import datetime, UTC
from dotenv import load_dotenv
import ccxt.async_support as ccxt
from warden_60m import Warden60Minute

# ENV CUSTODIAN: Force-load .env from the project root before ANY exchange objects are created.
# This ensures credentials are available even when this subprocess is spawned independently.
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
import dns_fix  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("recovery_agent")

async def run_recovery_loop():
    import config
    if config.SANDBOX is True or os.getenv("PAPER_TRADE", "True").lower() == "true":
        log.critical("[RECOVERY] FIREWALL: config.SANDBOX is True or PAPER_TRADE is True. DCA recovery loops are disabled to protect funds.")
        while True:
            await asyncio.sleep(3600)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "trading_data.db")
    ledger_path = os.path.join(base_dir, "trading_ledger.db")
    
    # V30.1: Shared Warden Cache to prevent instantiation leaks in heartbeat loop
    warden_cache = {}
    
    # P1-PATCH: DCA re-entry guard. Tracks the last DCA timestamp per symbol.
    # Prevents unbounded position amplification (max 1 DCA per position per 4 hours).
    last_dca_time = {}

    from exchange_factory import create_async_exchange, get_bitget_uta_client
    ex_id = getattr(config, "EXCHANGE_ID", "bitget").lower()
    active_key = getattr(config, "ACTIVE_API_KEY", "")
    if not active_key:
        log.critical(f"[RECOVERY] FATAL: ACTIVE_API_KEY for {ex_id} is empty. Check .env file. Aborting.")
        return
    log.info(f"[RECOVERY] ENV OK — {ex_id.upper()} API key loaded (last 4: ...{active_key[-4:]}).")

    exchange = create_async_exchange()
    try:
        await exchange.load_time_difference()
    except Exception as e:
        log.warning(f"[RECOVERY] load_time_difference notice: {e}")

    log.info("[RECOVERY] Agent online. Scanning open positions for underwater DCA targets every 60s...")
    
    try:
        while True:
            try:
                conn = sqlite3.connect(db_path, timeout=10.0)
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA busy_timeout=30000;")
                # Scan both Sniper and Berserker positions
                positions = []
                
                try:
                    cursor = conn.execute("SELECT symbol, side, entry_price, qty, stop_loss FROM open_positions")
                    for row in cursor.fetchall():
                        positions.append(("open_positions", *row))
                except sqlite3.OperationalError as oe:
                    log.warning(f"[RECOVERY] Sniper positions read failed: {oe}")
                    
                try:
                    cursor = conn.execute("SELECT symbol, side, entry_price, qty, stop_loss FROM berserker_open_positions")
                    for row in cursor.fetchall():
                        positions.append(("berserker_open_positions", *row))
                except sqlite3.OperationalError as oe:
                    log.warning(f"[RECOVERY] Berserker positions read failed: {oe}")
                
                conn.close()
                
                for table_name, symbol, side, entry_price, qty, stop_loss in positions:
                    try:
                        # N-12 FIX: Normalize side to lowercase immediately upon row unpacking
                        side_norm = str(side).strip().lower()
                        is_long_side = (side_norm in ("long", "buy"))

                        fetch_sym = symbol
                        if ex_id == "bitget" and ":USDT" not in fetch_sym:
                            fetch_sym = f"{symbol}:USDT"
                        ticker = await exchange.fetch_ticker(fetch_sym)
                        current_price = float(ticker['last'])
                        
                        if is_long_side:
                            pnl_pct = (current_price - entry_price) / entry_price
                        else:
                            pnl_pct = (entry_price - current_price) / entry_price
                            
                        if pnl_pct < -0.03:
                            # Position is > 3% underwater
                            # Dual Regime Verification: Check both global BTC macro AND the specific asset's own trend
                            # Re-use cached Warden60Minute instances to prevent aiohttp ClientSession & OHLCV churn
                            if "BTC/USDT" not in warden_cache:
                                warden_cache["BTC/USDT"] = Warden60Minute(symbol='BTC/USDT')
                            btc_regime = await warden_cache["BTC/USDT"].determine_regime()

                            if symbol not in warden_cache:
                                warden_cache[symbol] = Warden60Minute(symbol=symbol)
                            asset_regime = await warden_cache[symbol].determine_regime()

                            # Require both global BTC and the specific asset to be aligned in trend
                            regime_aligned = (
                                (is_long_side and btc_regime == "LONG_ONLY" and asset_regime == "LONG_ONLY") or
                                (not is_long_side and btc_regime == "SHORT_ONLY" and asset_regime == "SHORT_ONLY")
                            )

                            if not regime_aligned:
                                log.warning(f"[RECOVERY] DCA Veto for {symbol} ({side_norm}): Regime misaligned (BTC={btc_regime}, Asset={asset_regime}). Blocking DCA to avoid averaging into structural trend collapse.")
                                continue

                            # P1-PATCH: Persist DCA count & cooldown in SQLite system_state to survive daemon restarts
                            DCA_COOLDOWN_SECONDS = 4 * 3600  # 4 hours
                            dca_state_key = f"dca_state_{symbol}_{side_norm}"
                            dca_meta = {"count": 0, "last_ts": 0.0}
                            try:
                                conn_st = sqlite3.connect(db_path, timeout=5.0)
                                row_st = conn_st.execute("SELECT value FROM system_state WHERE key = ?", (dca_state_key,)).fetchone()
                                if row_st and row_st[0]:
                                    import json as _json
                                    dca_meta = _json.loads(row_st[0])
                                conn_st.close()
                            except Exception:
                                pass

                            time_since_last_dca = time.time() - float(dca_meta.get("last_ts", 0.0))
                            current_dca_count = int(dca_meta.get("count", 0))

                            if time_since_last_dca < DCA_COOLDOWN_SECONDS:
                                remaining_mins = (DCA_COOLDOWN_SECONDS - time_since_last_dca) / 60
                                log.info(f"[RECOVERY] DCA cooldown active for {symbol} ({side}). "
                                         f"Next DCA eligible in {remaining_mins:.0f}m. Skipping.")
                            elif current_dca_count >= 3:
                                log.warning(f"[RECOVERY] MAX DCA LIMIT (3) reached for {symbol} ({side}). Aborting further DCAs to prevent liquidation.")
                            else:
                                dca_qty = qty * 0.5
                                new_qty = qty + dca_qty
                                new_entry_price = ((entry_price * qty) + (current_price * dca_qty)) / new_qty
                                
                                # R-3 FIX: Case-insensitive side normalization to prevent inverted stop losses
                                side_norm = str(side).strip().lower()
                                is_long_pos = (side_norm in ("long", "buy"))

                                # Proportional Stop Loss Recalculation (precomputed before order execution)
                                sl_val = float(stop_loss) if (stop_loss is not None and float(stop_loss) > 0) else (entry_price * 0.97 if is_long_pos else entry_price * 1.03)
                                old_sl_dist_pct = abs(entry_price - sl_val) / entry_price if entry_price > 0 else 0.03
                                new_sl = new_entry_price * (1.0 - old_sl_dist_pct) if is_long_pos else new_entry_price * (1.0 + old_sl_dist_pct)
                                
                                # Execute DCA Recovery Order on the Exchange!
                                try:
                                    order_side = "buy" if is_long_pos else "sell"
                                    if ex_id == "bitget":
                                        uta = get_bitget_uta_client()
                                        await uta.create_market_order(symbol, "long" if is_long_pos else "short", dca_qty)
                                        await uta.close()
                                    else:
                                        await exchange.create_market_order(symbol, order_side, dca_qty)
                                    log.info(f"[RECOVERY] Exchange Order Executed: {order_side.upper()} {dca_qty} {symbol}")
                                except Exception as e:
                                    log.error(f"[RECOVERY] Exchange Order FAILED for {symbol}. Aborting DB update. Err: {e}")
                                    continue # Skip DB update if the actual trade fails
                                
                                # CRITICAL: Immediately persist DCA cooldown & counter in DB to eliminate crash loops
                                conn_dca = None
                                try:
                                    conn_dca = sqlite3.connect(db_path, timeout=10.0)
                                    conn_dca.execute("PRAGMA journal_mode=WAL;")
                                    conn_dca.execute("PRAGMA busy_timeout=30000;")
                                    dca_meta["count"] = current_dca_count + 1
                                    dca_meta["last_ts"] = time.time()
                                    import json as _json
                                    conn_dca.execute(
                                        "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                                        (dca_state_key, _json.dumps(dca_meta))
                                    )
                                    conn_dca.execute(
                                        f"UPDATE {table_name} SET qty = ?, entry_price = ?, stop_loss = ? WHERE symbol = ? AND LOWER(side) = ?",
                                        (new_qty, new_entry_price, new_sl, symbol, side.lower())
                                    )
                                    conn_dca.commit()
                                except Exception as dca_db_err:
                                    log.error(f"[RECOVERY] Failed to persist DCA state to DB for {symbol}: {dca_db_err}")
                                finally:
                                    if conn_dca:
                                        conn_dca.close()

                                # Update native strategy stop loss order on Bitget
                                if ex_id == "bitget":
                                    try:
                                        uta = get_bitget_uta_client()
                                        await uta.modify_strategy_order(symbol, sl=new_sl)
                                        await uta.close()
                                        log.info(f"[RECOVERY] Exchange Stop Loss updated to ${new_sl:,.4f} for {symbol} {side.upper()}")
                                    except Exception as sl_mod_err:
                                        log.warning(f"[RECOVERY] Could not modify exchange SL after DCA: {sl_mod_err}")

                                # Log AFTER the DB update to confirm end-to-end success
                                log.info(f"[RECOVERY] Position {symbol} ({side}) underwater. Macro valid. Safety DCA #{dca_meta['count']} completed.")
                                log.info(f"           -> Old Entry: {entry_price:.4f} | New Entry: {new_entry_price:.4f} | New SL: {new_sl:.4f} | Qty: {new_qty:.4f}")
                    except Exception as e:
                        log.error(f"[RECOVERY] Error processing {symbol}: {e}")
                        
            except Exception as e:
                log.error(f"[RECOVERY] Loop error: {e}")
                
            await asyncio.sleep(60)
    finally:
        log.info("[RECOVERY] Closing recovery resources...")
        if hasattr(exchange, "close"):
            try:
                await exchange.close()
            except Exception as e:
                log.error(f"[RECOVERY] Error closing async exchange: {e}")
        for symbol, warden in warden_cache.items():
            try:
                await warden.close()
            except Exception as e:
                log.error(f"[RECOVERY] Error closing warden for {symbol}: {e}")

if __name__ == "__main__":
    # Silence ccxt base exchange warning
    logging.getLogger("ccxt").setLevel(logging.ERROR)
    
    # Custom asyncio exception handler to silence unclosed sessions warning
    def custom_handler(loop, context):
        msg = context.get("message", "")
        if "Unclosed client session" in msg or "unclosed" in msg.lower():
            return
        loop.default_exception_handler(context)
        
    async def main():
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(custom_handler)
        await run_recovery_loop()
        
    asyncio.run(main())

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
                    cursor = conn.execute("SELECT symbol, side, entry_price, qty FROM open_positions")
                    for row in cursor.fetchall():
                        positions.append(("open_positions", *row))
                except sqlite3.OperationalError as oe:
                    log.warning(f"[RECOVERY] Sniper positions read failed: {oe}")
                    
                try:
                    cursor = conn.execute("SELECT symbol, side, entry_price, qty FROM berserker_open_positions")
                    for row in cursor.fetchall():
                        positions.append(("berserker_open_positions", *row))
                except sqlite3.OperationalError as oe:
                    log.warning(f"[RECOVERY] Berserker positions read failed: {oe}")
                
                conn.close()
                
                for table_name, symbol, side, entry_price, qty in positions:
                    try:
                        fetch_sym = symbol
                        if ex_id == "bitget" and ":USDT" not in fetch_sym:
                            fetch_sym = f"{symbol}:USDT"
                        ticker = await exchange.fetch_ticker(fetch_sym)
                        current_price = float(ticker['last'])
                        
                        if side == "long":
                            pnl_pct = (current_price - entry_price) / entry_price
                        else:
                            pnl_pct = (entry_price - current_price) / entry_price
                            
                        if pnl_pct < -0.03:
                            # Position is > 3% underwater
                            if symbol not in warden_cache:
                                # P1-PATCH: Always evaluate global BTC macro trend, not the asset's own
                                # trend, to avoid circular logic (e.g. using ETH regime to validate ETH DCA).
                                warden_cache[symbol] = Warden60Minute(symbol='BTC/USDT')
                            
                            warden = warden_cache[symbol]
                            regime = await warden.determine_regime()
                            
                            if (side == "long" and regime == "LONG_ONLY") or (side == "short" and regime == "SHORT_ONLY"):
                                # P1-PATCH: 4-hour DCA re-entry guard.
                                # Prevents loop from stacking 10+ DCAs on a sustained drawdown.
                                DCA_COOLDOWN_SECONDS = 4 * 3600  # 4 hours
                                last_dca = last_dca_time.get(symbol, 0)
                                time_since_last_dca = time.time() - last_dca
                                
                                if time_since_last_dca < DCA_COOLDOWN_SECONDS:
                                    remaining_mins = (DCA_COOLDOWN_SECONDS - time_since_last_dca) / 60
                                    log.info(f"[RECOVERY] DCA cooldown active for {symbol}. "
                                             f"Next DCA eligible in {remaining_mins:.0f}m. Skipping.")
                                else:
                                    dca_qty = qty * 0.5
                                    
                                    # dca_count is tracked on the warden object.
                                    # The warden is guaranteed initialized above.
                                    if not hasattr(warden, 'dca_count'):
                                        warden.dca_count = 0
                                        
                                    if warden.dca_count >= 3:
                                        log.warning(f"[RECOVERY] MAX DCA LIMIT (3) reached for {symbol}. Aborting further DCAs to prevent liquidation.")
                                        continue
                                    
                                    # Execute DCA Recovery Order on the Exchange!
                                    try:
                                        order_side = "buy" if side == "long" else "sell"
                                        if ex_id == "bitget":
                                            uta = get_bitget_uta_client()
                                            await uta.create_market_order(symbol, side, dca_qty)
                                            await uta.close()
                                        else:
                                            await exchange.create_market_order(symbol, order_side, dca_qty)
                                        log.info(f"[RECOVERY] Exchange Order Executed: {order_side.upper()} {dca_qty} {symbol}")
                                    except Exception as e:
                                        log.error(f"[RECOVERY] Exchange Order FAILED for {symbol}. Aborting DB update. Err: {e}")
                                        continue # Skip DB update if the actual trade fails
                                    
                                    new_qty = qty + dca_qty
                                    new_entry_price = ((entry_price * qty) + (current_price * dca_qty)) / new_qty
                                    
                                    # Execute DCA Recovery DB Update
                                    conn_dca = sqlite3.connect(db_path, timeout=10.0)
                                    conn_dca.execute("PRAGMA journal_mode=WAL;")
                                    conn_dca.execute("PRAGMA busy_timeout=30000;")
                                    conn_dca.execute(f"UPDATE {table_name} SET qty = ?, entry_price = ? WHERE symbol = ?",
                                                 (new_qty, new_entry_price, symbol))
                                    conn_dca.commit()
                                    conn_dca.close()
                                    
                                    # Record DCA timestamp to enforce cooldown on next iteration
                                    last_dca_time[symbol] = time.time()
                                    warden.dca_count += 1
                                    
                                    # Log AFTER the DB update to confirm end-to-end success
                                    log.info(f"[RECOVERY] Position {symbol} underwater. Macro valid. Safety DCA completed.")
                                    log.info(f"           -> Old Entry: {entry_price:.2f} | New Entry: {new_entry_price:.2f} | Qty: {new_qty:.4f} | DCA #{warden.dca_count}")
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

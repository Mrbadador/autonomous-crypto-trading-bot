import os
import sys
import time
import asyncio
import logging
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, UTC
from dotenv import load_dotenv

load_dotenv()
import dns_fix  # noqa: F401
import ccxt.async_support as ccxt
import config
from live_paper_exchange import LivePaperExchange

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [GHOST_GRID] %(message)s")
log = logging.getLogger("mean_reversion")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DATA = os.path.join(BASE_DIR, "trading_data.db")
DB_LEDGER = os.path.join(BASE_DIR, "trading_ledger.db")

SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"
MAX_GRID_TIERS = 3
TOTAL_POSITION_CAP = 60.00  # Total allocated margin across all 3 tiers ($20 per tier)
TIER_ALLOCATIONS = [0.33, 0.33, 0.34]
FLASH_CRASH_CONFIRMATION_SECS = getattr(config, "FLASH_CRASH_CONFIRMATION_SECS", 60)

def init_db_schema():
    """Ensure mean_reversion tables support grid tiers, hard_sl, and flash-crash timers."""
    try:
        with sqlite3.connect(DB_DATA, timeout=15.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=15000;")
            
            # Ensure mean_reversion_open_positions has required columns
            cursor = conn.cursor()
            existing_cols = [d[1] for d in cursor.execute("PRAGMA table_info(mean_reversion_open_positions)").fetchall()]
            
            new_cols = {
                "hard_sl": "REAL DEFAULT 0.0",
                "sl_breach_time": "REAL",
                "grid_tier": "INTEGER DEFAULT 1",
                "target_tp": "REAL",
                "target_sl": "REAL"
            }
            for col, col_def in new_cols.items():
                if col not in existing_cols:
                    try:
                        conn.execute(f"ALTER TABLE mean_reversion_open_positions ADD COLUMN {col} {col_def}")
                        log.info(f"Added column '{col}' to mean_reversion_open_positions.")
                    except Exception as e:
                        log.debug(f"Column {col} already present: {e}")
                        
            # Ensure mean_reversion_trades exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mean_reversion_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    side TEXT,
                    type TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    qty REAL,
                    pnl REAL,
                    balance REAL,
                    regime TEXT,
                    phase TEXT,
                    quality REAL,
                    notes TEXT,
                    status TEXT,
                    ai_rationale TEXT,
                    slippage_bps REAL,
                    action TEXT
                )
            """)
            conn.commit()
    except Exception as e:
        log.error(f"Failed to initialize mean_reversion schema: {e}")

def calculate_technical_indicators(df, period=14):
    """Calculates RSI, Rolling VWAP, Standard Deviation Bands, and ATR."""
    # 1. RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))

    # 2. VWAP & Rolling Standard Deviation (over 24 candles = 2 hours on 5m)
    v_window = min(24, len(df))
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    vol = df['volume'] + 1e-8
    df['cum_pv'] = (typical_price * vol).rolling(window=v_window).sum()
    df['cum_vol'] = vol.rolling(window=v_window).sum()
    df['vwap'] = df['cum_pv'] / df['cum_vol']
    df['vwap_std'] = df['close'].rolling(window=v_window).std()

    # 3. ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(window=period).mean()

    return df

def get_market_context():
    """Fetches real-time regime, Hurst exponent, ADX, fear & greed, and macro status."""
    hurst = 0.50
    adx = 18.0
    regime = "CHOP"
    fear_greed = 50.0
    spy_health = 0.5
    
    try:
        with sqlite3.connect(DB_DATA, timeout=10.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            row = conn.execute("SELECT regime, hurst, adx FROM market_regime ORDER BY id DESC LIMIT 1").fetchone()
            if row:
                regime = str(row[0] or "CHOP").upper()
                if row[1] is not None: hurst = float(row[1])
                if row[2] is not None: adx = float(row[2])
    except Exception as e:
        log.error(f"Error fetching market regime: {e}")

    try:
        if os.path.exists(DB_LEDGER):
            with sqlite3.connect(DB_LEDGER, timeout=10.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                row = conn.execute("SELECT fear_greed_index FROM market_context ORDER BY id DESC LIMIT 1").fetchone()
                if row and row[0] is not None: fear_greed = float(row[0])
                
                row2 = conn.execute("SELECT spy_health FROM macro_logs ORDER BY id DESC LIMIT 1").fetchone()
                if row2 and row2[0] is not None: spy_health = float(row2[0])
    except Exception:
        pass

    return regime, hurst, adx, fear_greed, spy_health

def get_dynamic_parameters():
    """Pulls auto-tuned parameter overrides from bot_parameters table in trading_data.db."""
    params = {
        "rsi_buy_l1": 36.0,
        "rsi_buy_l2": 28.0,
        "rsi_buy_l3": 20.0,
        "rsi_sell_l1": 64.0,
        "rsi_sell_l2": 72.0,
        "rsi_sell_l3": 80.0,
        "take_profit_pct": 1.006,  # +0.6% baseline
        "stop_loss_pct": 0.985,    # -1.5% soft SL baseline
        "grid_spacing_atr": 0.5    # 0.5x ATR grid step spacing
    }
    try:
        with sqlite3.connect(DB_DATA, timeout=10.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            rows = conn.execute("SELECT param_name, param_value FROM bot_parameters WHERE bot_name = 'mean_reversion'").fetchall()
            for r in rows:
                p_name, p_val = r[0], float(r[1])
                if p_name in params:
                    params[p_name] = p_val
                elif p_name == "rsi_buy":
                    params["rsi_buy_l1"] = p_val
                elif p_name == "rsi_sell":
                    params["rsi_sell_l1"] = p_val
    except Exception as e:
        log.error(f"Error fetching dynamic params: {e}")
    return params

async def execute_grid_tier(paper_ex, side, price, tier_idx, current_atr, params):
    """Executes a specific grid tier allocation with dynamic TP/SL levels routed through LivePaperExchange."""
    if not price or price <= 0:
        return False
        
    tier_fraction = TIER_ALLOCATIONS[tier_idx - 1] if tier_idx <= len(TIER_ALLOCATIONS) else 0.33
    allocated_usd = TOTAL_POSITION_CAP * tier_fraction
    qty = allocated_usd / price
    
    # Calculate Dynamic TP & Dual-Layer SL
    atr_val = current_atr if current_atr > 0 else price * 0.005
    if side == "LONG":
        tp_price = price * params["take_profit_pct"]
        soft_sl = price - (atr_val * 2.0)
        hard_sl = price - (atr_val * 3.5)
    else: # SHORT
        tp_price = price * (2.0 - params["take_profit_pct"])
        soft_sl = price + (atr_val * 2.0)
        hard_sl = price + (atr_val * 3.5)
        
    try:
        # Route live order via LivePaperExchange (creates real Bitget order if live mode enabled)
        pos = await paper_ex.open_position(
            symbol=SYMBOL,
            side=side.lower(),
            qty=qty,
            price=price,
            sl=soft_sl,
            tp=tp_price,
            hard_sl=hard_sl,
            metadata={
                "grid_tier": tier_idx,
                "notes": f"Ghost-Grid Tier {tier_idx}",
                "quality": 0.85
            }
        )
        if pos:
            # Update grid_tier column in open positions table
            with sqlite3.connect(DB_DATA, timeout=10.0) as conn:
                conn.execute("UPDATE mean_reversion_open_positions SET grid_tier = ? WHERE symbol = ?", (tier_idx, SYMBOL))
            log.info(
                f"[GHOST-GRID] 🎯 Live Order Routed: {side} Tier {tier_idx} ({allocated_usd:.2f} USDT) at ${price:,.2f} | "
                f"TP: ${tp_price:,.2f} | Soft SL: ${soft_sl:,.2f} | Hard SL: ${hard_sl:,.2f}"
            )
            return True
    except Exception as e:
        log.error(f"[GHOST-GRID] Failed to execute grid tier: {e}")
    return False

async def monitor_open_grid_positions(paper_ex, current_price):
    """Monitors active grid tiers for TP targets, Hard SL, and Soft SL Flash-Crash confirmation."""
    try:
        with sqlite3.connect(DB_DATA, timeout=15.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=15000;")
            cursor = conn.cursor()
            
            rows = cursor.execute("""
                SELECT row_id, side, entry_price, qty, take_profit, stop_loss, hard_sl, sl_breach_time, grid_tier, entry_time 
                FROM mean_reversion_open_positions
            """).fetchall()
            
            if not rows:
                return
                
            now_ts = time.time()
            
            for row in rows:
                row_id, side_val, entry_p, qty, tp, soft_sl, hard_sl, breach_time, tier, entry_t = row
                side = str(side_val or "LONG").upper()
                entry_p = float(entry_p or 1.0)
                qty = float(qty or 0.0)
                tp = float(tp or entry_p * 1.01)
                soft_sl = float(soft_sl or entry_p * 0.985)
                hard_sl = float(hard_sl or entry_p * 0.97)
                
                close_reason = None
                
                # 1. Take-Profit Check (Instant exit)
                if side == "LONG" and current_price >= tp:
                    close_reason = f"tp_hit_tier_{tier}"
                elif side == "SHORT" and current_price <= tp:
                    close_reason = f"tp_hit_tier_{tier}"
                    
                # 2. Hard Stop-Loss Check (Instant unconditional exit)
                elif side == "LONG" and hard_sl > 0 and current_price <= hard_sl:
                    close_reason = f"hard_sl_hit_tier_{tier}"
                elif side == "SHORT" and hard_sl > 0 and current_price >= hard_sl:
                    close_reason = f"hard_sl_hit_tier_{tier}"
                    
                # 3. Soft Stop-Loss Flash-Crash Filter Check
                else:
                    is_soft_breached = (side == "LONG" and current_price <= soft_sl) or (side == "SHORT" and current_price >= soft_sl)
                    
                    if is_soft_breached:
                        if breach_time is None:
                            # Start 60s flash-crash confirmation clock
                            cursor.execute("UPDATE mean_reversion_open_positions SET sl_breach_time = ? WHERE row_id = ?", (now_ts, row_id))
                            conn.commit()
                            log.warning(
                                f"[FLASH FILTER] ⚡ Soft SL breached for {side} Tier {tier} (Entry: ${entry_p:,.2f}, Curr: ${current_price:,.2f}). "
                                f"Suppressing close. Waiting {FLASH_CRASH_CONFIRMATION_SECS}s to confirm true move vs exchange wick."
                            )
                        else:
                            elapsed = now_ts - float(breach_time)
                            if elapsed >= FLASH_CRASH_CONFIRMATION_SECS:
                                # Confirmed sustained breach -> execute close
                                close_reason = f"confirmed_sl_tier_{tier}"
                                log.warning(
                                    f"[FLASH FILTER] ✅ Soft SL confirmed for {side} Tier {tier} after {elapsed:.1f}s. Executing stop-loss."
                                )
                            else:
                                log.info(
                                    f"[FLASH FILTER] ⏳ Soft SL breach active for {side} Tier {tier} ({elapsed:.1f}s / {FLASH_CRASH_CONFIRMATION_SECS}s). Holding..."
                                )
                    else:
                        if breach_time is not None:
                            # Price recovered back inside safe range!
                            cursor.execute("UPDATE mean_reversion_open_positions SET sl_breach_time = NULL WHERE row_id = ?", (row_id,))
                            conn.commit()
                            log.info(
                                f"[FLASH FILTER] 🛡️ Price RECOVERED above Soft SL for {side} Tier {tier}! Resetting breach clock."
                            )

                # Execute position closure via LivePaperExchange if triggered
                if close_reason:
                    await paper_ex.close_position(SYMBOL, price=current_price, t_type=close_reason)
                    log.info(f"[GHOST-GRID] 🏁 Closed {side} Tier {tier} at ${current_price:,.2f} | Reason: {close_reason}")
    except Exception as e:
        log.error(f"[GHOST-GRID] Error monitoring positions: {e}")

async def run_bot():
    """Main execution loop for Adaptive Ghost-Grid Scalper."""
    log.info("Starting Adaptive Ghost-Grid Mean Reversion Scalper (V27)...")
    init_db_schema()

    # Initialize LivePaperExchange for real order execution on Bitget UTA
    paper_ex = LivePaperExchange(initial_balance=50.0, account_name="mean_reversion")
    await paper_ex.sync_positions_from_db()

    ex_id = getattr(config, "EXCHANGE_ID", "bitget").lower()
    ex_class = getattr(ccxt, ex_id, None) or ccxt.bitget
    exchange = ex_class({
        'apiKey': config.ACTIVE_API_KEY,
        'secret': config.ACTIVE_SECRET_KEY,
        'password': config.ACTIVE_PASSPHRASE,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap' if ex_id == "bitget" else 'future',
            'adjustForTimeDifference': True,
            'fetchCurrencies': False,
        }
    })

    while True:
        try:
            # 1. Fetch live market candles
            ohlcv = await exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=40)
            if not ohlcv or len(ohlcv) < 20:
                await asyncio.sleep(10)
                continue

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = calculate_technical_indicators(df)

            current_price = float(df['close'].iloc[-1])
            current_rsi = float(df['rsi'].iloc[-1])
            current_vwap = float(df['vwap'].iloc[-1]) if 'vwap' in df and not pd.isna(df['vwap'].iloc[-1]) else current_price
            current_std = float(df['vwap_std'].iloc[-1]) if 'vwap_std' in df and not pd.isna(df['vwap_std'].iloc[-1]) else (current_price * 0.003)
            current_atr = float(df['atr'].iloc[-1]) if 'atr' in df and not pd.isna(df['atr'].iloc[-1]) else (current_price * 0.004)

            # 2. Get dynamic auto-tuned parameters from database
            params = get_dynamic_parameters()

            # 3. Monitor active positions (TP, Soft SL with Flash-Crash Timer, Hard SL)
            await monitor_open_grid_positions(paper_ex, current_price)

            # 4. Check Regime & Breakout Kill-Switch
            regime, hurst, adx, fear_greed, spy_health = get_market_context()

            # Emergency Macro Brakes
            if fear_greed < 20.0 or spy_health < -0.6:
                log.warning(f"[MACRO-BRAKE] Extreme market fear detected (F&G={fear_greed:.1f}, SPY={spy_health:.2f}). Pausing grid entries.")
                await asyncio.sleep(60)
                continue

            # Strict Breakout Handoff Gate (Option 2 Kill-Switch)
            is_trending = ("TREND" in regime and "CHOP" not in regime) or (hurst > 0.56 and adx > 22.0)
            if is_trending:
                log.info(f"[REGIME-HANDOFF] Market is TRENDING (Regime={regime}, Hurst={hurst:.3f}, ADX={adx:.1f}). Ghost-Grid disarmed to let Sniper run.")
                await asyncio.sleep(60)
                continue

            # 5. Evaluate Multi-Tier Ghost-Grid Entries
            existing_tiers = []
            with sqlite3.connect(DB_DATA, timeout=10.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                rows = conn.execute("SELECT grid_tier, side FROM mean_reversion_open_positions WHERE symbol = ?", (SYMBOL,)).fetchall()
                existing_tiers = [(r[0], r[1]) for r in rows]

            num_active_tiers = len(existing_tiers)
            active_side = str(existing_tiers[0][1]).upper() if existing_tiers else None

            # Calculate Dynamic Standard Deviation Bands from 1h VWAP
            band_l1_long = current_vwap - (1.2 * current_std)
            band_l2_long = current_vwap - (1.8 * current_std)
            band_l3_long = current_vwap - (2.5 * current_std)

            band_l1_short = current_vwap + (1.2 * current_std)
            band_l2_short = current_vwap + (1.8 * current_std)
            band_l3_short = current_vwap + (2.5 * current_std)

            # Determine if new tier should trigger
            if num_active_tiers < MAX_GRID_TIERS:
                active_tier_ids = [t[0] for t in existing_tiers]

                # --- LONG GRID SCALING ---
                if active_side in (None, "LONG"):
                    # Tier 1 Trigger: Price < VWAP - 1.2σ OR RSI < rsi_buy_l1
                    if 1 not in active_tier_ids and (current_price <= band_l1_long or current_rsi < params["rsi_buy_l1"]):
                        log.info(f"[GHOST-GRID] 🟢 Long Tier 1 Signal: Price ${current_price:,.2f} <= L1 Band ${band_l1_long:,.2f} | RSI: {current_rsi:.1f}")
                        await execute_grid_tier(paper_ex, "LONG", current_price, 1, current_atr, params)

                    # Tier 2 Trigger: Price < VWAP - 1.8σ OR RSI < rsi_buy_l2
                    elif 1 in active_tier_ids and 2 not in active_tier_ids and (current_price <= band_l2_long or current_rsi < params["rsi_buy_l2"]):
                        log.info(f"[GHOST-GRID] 🟢 Long Tier 2 Signal: Price ${current_price:,.2f} <= L2 Band ${band_l2_long:,.2f} | RSI: {current_rsi:.1f}")
                        await execute_grid_tier(paper_ex, "LONG", current_price, 2, current_atr, params)

                    # Tier 3 Trigger: Deep Value @ VWAP - 2.5σ OR RSI < rsi_buy_l3
                    elif 2 in active_tier_ids and 3 not in active_tier_ids and (current_price <= band_l3_long or current_rsi < params["rsi_buy_l3"]):
                        log.info(f"[GHOST-GRID] 🟢 Long Tier 3 Deep-Value Signal: Price ${current_price:,.2f} <= L3 Band ${band_l3_long:,.2f} | RSI: {current_rsi:.1f}")
                        await execute_grid_tier(paper_ex, "LONG", current_price, 3, current_atr, params)

                # --- SHORT GRID SCALING ---
                if active_side in (None, "SHORT") and num_active_tiers == 0:
                    # Tier 1 Trigger: Price > VWAP + 1.2σ OR RSI > rsi_sell_l1
                    if 1 not in active_tier_ids and (current_price >= band_l1_short or current_rsi > params["rsi_sell_l1"]):
                        log.info(f"[GHOST-GRID] 🔴 Short Tier 1 Signal: Price ${current_price:,.2f} >= L1 Band ${band_l1_short:,.2f} | RSI: {current_rsi:.1f}")
                        await execute_grid_tier(paper_ex, "SHORT", current_price, 1, current_atr, params)

                    # Tier 2 Trigger: Price > VWAP + 1.8σ OR RSI > rsi_sell_l2
                    elif 1 in active_tier_ids and 2 not in active_tier_ids and (current_price >= band_l2_short or current_rsi > params["rsi_sell_l2"]):
                        log.info(f"[GHOST-GRID] 🔴 Short Tier 2 Signal: Price ${current_price:,.2f} >= L2 Band ${band_l2_short:,.2f} | RSI: {current_rsi:.1f}")
                        await execute_grid_tier(paper_ex, "SHORT", current_price, 2, current_atr, params)

                    # Tier 3 Trigger: Overbought Peak @ VWAP + 2.5σ OR RSI > rsi_sell_l3
                    elif 2 in active_tier_ids and 3 not in active_tier_ids and (current_price >= band_l3_short or current_rsi > params["rsi_sell_l3"]):
                        log.info(f"[GHOST-GRID] 🔴 Short Tier 3 Overbought Signal: Price ${current_price:,.2f} >= L3 Band ${band_l3_short:,.2f} | RSI: {current_rsi:.1f}")
                        await execute_grid_tier(paper_ex, "SHORT", current_price, 3, current_atr, params)

            await asyncio.sleep(30) # High-resolution 30s monitoring loop

        except Exception as e:
            log.error(f"[GHOST-GRID] Error in main loop: {e}")
            await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(run_bot())


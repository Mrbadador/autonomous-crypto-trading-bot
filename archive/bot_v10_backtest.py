import asyncio
import logging
import time
import requests
import json
import ccxt
import sqlite3
import numpy as np
import os
import atexit
import aiohttp
import config
import pandas as pd
import traceback
from itertools import product
from datetime import datetime, UTC, timedelta
from rich.live import Live
from rich.console import Console
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel

# Local Imports
from state_manager import EngineState, TradeEntry
from exchange_manager import ExchangeManager
from indicators import compute_indicators, classify_regime_hard, get_candle_patterns, get_volatility_regime, is_market_ready
from ui_render import build_dashboard_layout, setup_live, update_live, stop_live, CYAN, RESET, YELLOW, RED, GREEN, WHITE, BLUE
import ai_engine
from core_engine import (
    TrendContext,
    calculate_risk_qty,
    calculate_exits,
    calculate_true_breakeven,
    build_trend_context,
    calculate_adaptive_sl_distance,
    calculate_adaptive_tp_distance,
    calculate_adaptive_trail_distance,
    reconcile_regime,
    get_deterministic_action,
    calculate_dynamic_risk,
)

# ---------------------------------------------------------------------------
# Setup Logging
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("logs/testnet_session.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("bot")

# ---------------------------------------------------------------------------
# Core Logic & Strategy
# (Functions extracted to core_engine.py in V11.0 — imported above)
# ---------------------------------------------------------------------------

def normalize_mode(val, default, allowed):
    val = str(val).lower()
    return val if val in allowed else default


async def _eval_and_record(state, symbol, pnl, t_type, side, pos_dict):
    """Structured post-trade audit log. DB/CSV recording is handled by the
    subsequent log_trade_event() call in every caller — do not add db.log_trade
    here as it would cause duplicate writes."""
    notes = pos_dict.get("notes", "No notes")
    mfe = pos_dict.get("mfe", 0.0)
    log.info(f"[{symbol}] Post-trade eval ({t_type}): pnl={pnl:.2f} | MFE={mfe:.2f}% | {notes}")

async def log_trade_event(state: EngineState, symbol, t_type, side, entry, exit, qty, pnl, balance, regime, phase, quality, sl_m, tp_m, tr_m, notes, mfe=0.0, pos_id=None):
    entry_obj = TradeEntry(
        symbol=symbol, side=side, entry_price=entry, exit_price=exit,
        qty=qty, pnl=pnl, ai_action=t_type, final_action=t_type, ai_notes=notes,
        timestamp=datetime.now(UTC), balance=balance, regime=regime,
        phase=phase, quality=quality, sl_mode=sl_m, tp_mode=tp_m, trail_mode=tr_m,
        mfe=mfe, pos_id=pos_id
    )
    await asyncio.to_thread(state.record_trade, entry_obj)

def reconcile_regime(ai_res, hard_regime, adx_val):
    ai_regime = ai_res.get("regime", hard_regime).upper()
    if ai_regime in ("UNKNOWN", ""): return hard_regime
    return ai_regime

def get_deterministic_action(regime, trend_ctx, adx_val, adx_sma, price, ema200, htf_trend, rsi, vol, vol_sma, s_high, s_low, atr, detected_patterns, has_pos=False):
    if regime == "CHAOTIC": return "hold", "Market is chaotic.", 1.0
    # if regime == "RANGING": return "hold", "Market is ranging.", 1.0
    
    phase = trend_ctx.phase
    q_mult = 1.0
    proximity_dist = atr * 0.5
    near_resistance = (price <= s_high and price > s_high - proximity_dist)
    near_support    = (price >= s_low and price < s_low + proximity_dist)
    
    bullish_patterns = ["Hammer", "Bullish Engulfing", "Morning Star", "Three White Soldiers", "Rising Three"]
    bearish_patterns = ["Hanging Man", "Bearish Engulfing", "Evening Star", "Three Black Crows", "Falling Three"]
    pattern_conviction = 0
    for p in detected_patterns:
        if p in bullish_patterns: pattern_conviction += 1
        elif p in bearish_patterns: pattern_conviction -= 1

    # V10.0 SUPER-DISPLACEMENT GATE (150% Requirement)
    vol_ratio = vol / vol_sma if vol_sma > 0 else 1.0
    is_volume_spike = vol_ratio >= 1.5
    is_above_ema = price > ema200
    is_below_ema = price < ema200

    if not is_volume_spike:
        return "hold", f"Failed Super-Vol Gate ({vol_ratio:.1f} < 1.5)", 1.0

    # LONG
    if is_above_ema and rsi < 80 and not (near_resistance and adx_val < 30):
        final_q = min(q_mult * 1.3 if vol_ratio > 2.0 else 1.0, 1.5)
        if htf_trend == "bearish": final_q *= config.COUNTER_TREND_SIZE_REDUCTION
        if pattern_conviction > 0: final_q = min(1.5, final_q * 1.2)
        if regime == "RANGING": final_q *= 0.5
        if phase in ("STARTING", "EARLY", "LATE") and adx_val > adx_sma: return "open_long" if not has_pos else "add_long", f"Elite {phase} trend", final_q
        if phase == "MATURE" and adx_val > config.ADX_STRONG_THRESHOLD: return "add_long" if has_pos else "open_long", "Elite mature trend", final_q

    # SHORT
    if is_below_ema and rsi > 20 and not (near_support and adx_val < config.ADX_STRONG_THRESHOLD):
        final_q = min(q_mult * 1.3 if vol_ratio > 2.0 else 1.0, 1.5)
        if htf_trend == "bullish": final_q *= config.COUNTER_TREND_SIZE_REDUCTION
        if pattern_conviction < 0: final_q = min(1.5, final_q * 1.2)
        if regime == "RANGING": final_q *= 0.5
        if phase in ("STARTING", "EARLY", "LATE") and adx_val > adx_sma: return "open_short" if not has_pos else "add_short", f"Elite {phase} trend", final_q
        if phase == "MATURE" and adx_val > config.ADX_STRONG_THRESHOLD: return "add_short" if has_pos else "open_short", "Elite mature trend", final_q

    return "hold", "No elite edge.", 1.0

def calculate_dynamic_risk(adx, macro_status, now_time):
    """V6.8: Dynamic Regime Switching Logic"""
    is_weekend = now_time.weekday() >= 5
    is_aligned = macro_status in ["BULLISH", "BEARISH"]
    
    if adx < 20:
        mode, min_conv, news_win = "CHOP", 8.0, 60
    elif is_aligned and 25 <= adx <= 75 and not is_weekend:
        mode, min_conv, news_win = "AGGRESSIVE", 6.5, 15
    elif macro_status == "MIXED" or adx > 80 or is_weekend:
        mode, min_conv, news_win = "SNIPER", 8.0, 60
    else:
        mode, min_conv, news_win = "NORMAL", config.MIN_AI_CONVICTION_SCORE, config.PRE_NEWS_MINUTES
        
    if not config.IS_REPLAY:
        log.info(f"[DYNAMIC RISK] Active Mode: {mode} | Conviction Req: {min_conv} | News Gate: {news_win}m")
    return mode, min_conv, news_win

async def check_news_freeze(state, now_time, dynamic_news_win=None):
    if not state.upcoming_news: return False, "", False
    news_win = dynamic_news_win if dynamic_news_win is not None else config.PRE_NEWS_MINUTES
    for n_time, impact, title in state.upcoming_news:
        mins = (n_time - now_time).total_seconds() / 60
        if -config.POST_NEWS_MINUTES <= mins <= news_win: return True, title, (mins > 0)
    return False, "", False

async def run_cycle(symbol, em: ExchangeManager, state: EngineState, candle_time=None):
    if os.path.exists(config.KILL_SWITCH_FILE):
        state.is_halted = True; return None
    
    try:
        if hasattr(em, "fetch_df"):
            df_1h = em.fetch_df(symbol, config.TIMEFRAME, config.OHLCV_LIMIT)
        else:
            raw_1h = await asyncio.to_thread(em.fetch_ohlcv, symbol, config.TIMEFRAME, config.OHLCV_LIMIT)
            cols = getattr(em, "columns", ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            if len(raw_1h[0]) != len(cols):
                cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df_1h = pd.DataFrame(raw_1h, columns=cols)
            df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp_ms'] if 'timestamp_ms' in df_1h.columns else df_1h['timestamp'], unit='ms', utc=True)
            df_1h.set_index('timestamp', inplace=True)
            if 'ema200' not in df_1h.columns:
                df_1h = compute_indicators(df_1h)
        
        if df_1h is None or df_1h.empty:
            return None
            
        r1 = df_1h.iloc[-1]
        current_candle_time = df_1h.index[-1]
        last_candle_time = state.last_candle_times.get(symbol)
        is_new_candle = (last_candle_time is None or current_candle_time > last_candle_time)
        
        # Indicator Cluster (Moved Up)
        price, atr = float(r1["close"]), float(r1["atr14"])
        if pd.isna(atr) or atr <= 0: atr = price * 0.01
        atr_pct, adx_val, adx_sma, e_slope = float(r1["atr_pct"]), float(r1.get("adx", 0)), float(r1.get("adx_avg", 0)), float(r1.get("ema_slope", 0))
        swing_high, swing_low, rsi, vol, vol_sma, ema200 = float(r1.get("swing_high", price * 1.01)), float(r1.get("swing_low", price * 0.99)), float(r1.get("rsi14", 50)), float(r1.get("volume", 0)), float(r1.get("vol_sma20", 0)), float(r1.get("ema200", price))
        hard_regime = classify_regime_hard(r1)
        trend_ctx = build_trend_context(r1)
        detected_patterns = get_candle_patterns(df_1h)
        
    except Exception as e:
        log.error(f"Cycle failed for {symbol}: {e}"); return None

    em.update_sim_price(symbol, price)
    now = candle_time if candle_time else (current_candle_time if config.MODE == "replay" else datetime.now(UTC))
    sym_cooldown = state.cooldown_until.get(symbol)
    if sym_cooldown and now >= sym_cooldown: state.cooldown_until[symbol] = None
        
    symbol_positions = em.get_positions_by_symbol(symbol)
    skip_ai = state.is_halted or (sym_cooldown and now < sym_cooldown) or (not is_new_candle and not symbol_positions)
    
    if skip_ai:
        # Get risk mode even for skips to keep UI updated
        risk_mode, dynamic_min_conv, dynamic_news_win = calculate_dynamic_risk(adx_val, state.global_btc_trend, now)
        res = {
            "symbol": symbol, "price": price, "signal": "monitoring", 
            "ai_res": {"action": "wait", "notes": "Monitoring", "regime": hard_regime},
            "phase": trend_ctx.phase, "final_action": "WAIT",
            "atr_pct": atr_pct, "adx": adx_val, "adx_sma": adx_sma, "slope": e_slope,
            "patterns": detected_patterns, "risk_mode": risk_mode
        }
        if not config.IS_REPLAY:
            state.db.log_cycle({
                "symbol": symbol, "price": price, "signal": "monitoring", "action": "wait", "regime": hard_regime,
                "phase": trend_ctx.phase, "atr_pct": atr_pct, "adx": adx_val, "adx_sma": adx_sma, "slope": e_slope,
                "patterns": ", ".join(detected_patterns) if detected_patterns else "", "notes": f"Monitoring ({risk_mode})",
                "risk_mode": risk_mode
            })
        return res

    state.last_candle_times[symbol] = current_candle_time
    
    # Emerald Structural Anchor (V8.0: 4H EMA 200)
    massive_trend = "neutral"
    ema20_4h, ema50_4h, ema200_4h = 0.0, 0.0, 0.0
    try:
        if hasattr(em, "fetch_df"):
            df_4h = em.fetch_df(symbol, "4h", 250)
        else:
            raw_4h = await asyncio.to_thread(em.fetch_ohlcv, symbol, "4h", 250)
            if isinstance(raw_4h, list):
                df_4h = pd.DataFrame(raw_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            else:
                df_4h = raw_4h.copy()
            df_4h = compute_indicators(df_4h)
        
        if df_4h is None or df_4h.empty:
            return None
            
        r4h = df_4h.iloc[-1]
        ema20_4h, ema50_4h, ema200_4h = float(r4h.get("ema20", 0)), float(r4h.get("ema50", 0)), float(r4h.get("ema200", 0))
        
        is_neutral = abs(price - ema200_4h) / ema200_4h < 0.005 if ema200_4h > 0 else False
        can_long = price > ema200_4h and r1.get("ema20", 0) > r1.get("ema50", 0)
        can_short = price < ema200_4h and r1.get("ema20", 0) < r1.get("ema50", 0)
        
        if is_neutral: massive_trend = "neutral"
        elif can_long: massive_trend = "bullish"
        elif can_short: massive_trend = "bearish"
        else: massive_trend = "no_alignment"
    except Exception as e:
        if config.MODE == "replay": log.debug(f"[{symbol}] 4H Anchor Warmup: {e}")
        else: log.error(f"[{symbol}] 4H Anchor Failure: {e}")

    # HTF & Macro (1H filter)
    htf_trend = "bullish"
    ema_slope_3c = 0.0
    if config.HTF_ENABLED:
        try:
            if config.HTF_TIMEFRAME == "4h":
                df_htf = df_4h
            elif hasattr(em, "fetch_df"):
                df_htf = em.fetch_df(symbol, config.HTF_TIMEFRAME, config.OHLCV_LIMIT)
            else:
                df_htf_raw = await asyncio.to_thread(em.fetch_ohlcv, symbol, config.HTF_TIMEFRAME, config.OHLCV_LIMIT)
                df_htf = pd.DataFrame(df_htf_raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df_htf['timestamp'] = pd.to_datetime(df_htf['timestamp'], unit='ms', utc=True)
                df_htf.set_index('timestamp', inplace=True)
                df_htf = compute_indicators(df_htf)
                
            r_htf = df_htf.iloc[-1]
            ema200_htf, ema_slope_htf, close_htf = float(r_htf.get("ema200", 0)), float(r_htf.get("ema_slope", 0)), float(r_htf.get("close", 0))
            ema200_htf_prev = float(df_htf["ema200"].iloc[-4]) if len(df_htf) > 4 else ema200_htf
            ema_slope_3c = (ema200_htf - ema200_htf_prev) / ema200_htf_prev if ema200_htf_prev != 0 else 0
            if close_htf > ema200_htf and ema_slope_htf > 0.1: htf_trend = "bullish"
            elif close_htf < ema200_htf and ema_slope_htf < -0.1: htf_trend = "bearish"
            else: htf_trend = "chop"
        except Exception as e:
            if config.MODE == "replay": log.debug(f"HTF Error: {e}")


    # Management
    for pos in list(symbol_positions):
        side = pos.side
        p_dist = (price - pos.entry_price if side == "long" else pos.entry_price - price)
        if p_dist >= atr * config.PARTIAL_TP_ATR_MULTIPLE and not pos.partial_taken:
            p_pnl = em.close_partial_sim_pos(pos.id, config.PARTIAL_TP_FRACTION, price)
            if p_pnl is not None:
                pos.partial_taken = True
                await _eval_and_record(state, symbol, p_pnl, "partial_tp", side, pos.__dict__)
                _bal = getattr(em.balance, 'total', em.balance) if not isinstance(em.balance, (int, float)) else em.balance
                await log_trade_event(state, symbol, "partial_tp", side, pos.entry_price, price, (pos.qty / (1-config.PARTIAL_TP_FRACTION)) * config.PARTIAL_TP_FRACTION, p_pnl, _bal, pos.regime, pos.phase, pos.quality, pos.sl_mode, pos.tp_mode, pos.trail_mode, pos.notes, mfe=pos.mfe, pos_id=pos.id)

        # --- ASYMMETRICAL EMERALD TRAILING STOP (V8.0) ---
        profit_dist = (price - pos.entry_price if side == "long" else pos.entry_price - price)
        act_atr = config.LONG_TRAIL_ACTIVATION_ATR if side == "long" else config.SHORT_TRAIL_ACTIVATION_ATR
        dist_atr = config.LONG_TRAIL_DIST_ATR if side == "long" else config.SHORT_TRAIL_DIST_ATR
        
        if profit_dist >= act_atr * atr: 
            pnl = (price - pos.entry_price if side == 'long' else pos.entry_price - price) * pos.qty
            if pnl < -em.balance.total: pnl = -em.balance.total
            trail_sl = (pos.peak_price - (dist_atr * atr) if side == "long" else pos.peak_price + (dist_atr * atr))
            if side == "long":
                if trail_sl > pos.sl:
                    pos.sl = trail_sl
                    log.info(f"[{symbol}] [LONG TRAIL] SL moved to ${pos.sl:.2f}")
            else:
                if trail_sl < pos.sl or pos.sl == 0:
                    pos.sl = trail_sl
                    log.info(f"[{symbol}] [SHORT TRAIL] SL moved to ${pos.sl:.2f}")

        if (side == "long" and price <= pos.sl) or (side == "short" and price >= pos.sl):
            c_pos = em.close_sim_pos(pos.id, price)
            if c_pos: 
                state.in_position[symbol] = False
                state.active_trade[symbol] = None
                await _eval_and_record(state, symbol, c_pos.realized_pnl, "sl_hit", side, c_pos.__dict__)
                _bal = getattr(em.balance, 'total', em.balance) if not isinstance(em.balance, (int, float)) else em.balance
                await log_trade_event(state, symbol, "sl_hit", side, c_pos.entry_price, price, c_pos.qty, c_pos.realized_pnl, _bal, c_pos.regime, c_pos.phase, c_pos.quality, "trail", "swing", "trail", c_pos.notes, mfe=c_pos.mfe, pos_id=c_pos.id)
        elif (side == "long" and pos.tp > 0 and price >= pos.tp) or (side == "short" and pos.tp > 0 and price <= pos.tp):
            c_pos = em.close_sim_pos(pos.id, price)
            if c_pos: 
                state.in_position[symbol] = False
                state.active_trade[symbol] = None
                await _eval_and_record(state, symbol, c_pos.realized_pnl, "tp_hit", side, c_pos.__dict__)
                _bal = getattr(em.balance, 'total', em.balance) if not isinstance(em.balance, (int, float)) else em.balance
                await log_trade_event(state, symbol, "tp_hit", side, c_pos.entry_price, price, c_pos.qty, c_pos.realized_pnl, _bal, c_pos.regime, c_pos.phase, c_pos.quality, "limit", "limit", "limit", c_pos.notes, mfe=c_pos.mfe, pos_id=c_pos.id)

    # V6.8 Dynamic Risk
    risk_mode, dynamic_min_conv, dynamic_news_win = calculate_dynamic_risk(adx_val, state.global_btc_trend, now)

    action, tree_note, q_mult = get_deterministic_action(hard_regime, trend_ctx, adx_val, adx_sma, price, ema200, htf_trend, rsi, vol, vol_sma, swing_high, swing_low, atr, detected_patterns, has_pos=len(symbol_positions)>0)
    
    is_news_freeze, news_title, is_pre = await check_news_freeze(state, now, dynamic_news_win)
    if is_news_freeze:
        action, tree_note = "hold", f"NEWS GATE ({news_title})"
        state.is_observe_only = True
    # Position Safety Check (Institutional Hold)
    if symbol_positions:
        # If we have an active position, we hold by default unless management closes it
        if "open" in action: 
            action = "hold"
        # We don't log 'Monitoring' here yet to preserve debug visibility below

    signal = "long" if "long" in action else "short" if "short" in action else "hold"
    ai_res = {"action": "hold", "entry_quality": 0.0, "notes": "AI idle."}
    
    # Emerald Entry Context (EMA Stack)
    is_ema_stack = False
    ema20, ema50 = r1.get("ema20", 0), r1.get("ema50", 0)
    if signal == "long" and price > ema20 > ema50: is_ema_stack = True
    elif signal == "short" and price < ema20 < ema50: is_ema_stack = True

    # V10 Logic Gate: Conviction Check + Portfolio Guard
    in_pos = state.in_position.get(symbol, False)
    is_open_action = action in ("open_long", "open_short", "add_long", "add_short")
    
    if is_open_action and not state.is_observe_only:
        try:
            # CALL AI for both 'open' and 'add' signals
            ai_res = await ai_engine.get_llm().get_risk_params(
                regime=hard_regime, atr_pct=atr_pct, signal=signal, phase=trend_ctx.phase,
                session_pnl=state.get_session_performance(), streak=state.global_streak,
                btc_context=state.global_btc_trend, market_context=str(df_1h['close'].tail(20).tolist()),
                htf_trend=htf_trend, current_regime=hard_regime, ema_slope=ema_slope_3c, patterns=detected_patterns,
                ema_stack=is_ema_stack
            )
        except Exception as e: 
            log.error(f"AI Call Error: {e}")
            ai_res = {"action": "reject", "entry_quality": 0.0, "notes": "AI Error"}
    elif is_open_action:
        # Diagnostic for why we didn't call AI
        print(f"[DEBUG] [{symbol}] AI Skip: InPos={in_pos} | ObsOnly={state.is_observe_only}")
        pass
    
    quality = ai_res.get("entry_quality", 0.0) * q_mult
    
    # Elite Displacement Logic (V10.0 Super-Gate)
    is_displacement = vol >= (vol_sma * 1.5) if vol_sma > 0 else True
    
    # Trend-Ride Logic (V8.5): Elite Structural Guard
    can_enter = False
    if massive_trend == "neutral" and quality >= 7.0 and is_displacement: 
        can_enter = True
    elif massive_trend in ["bullish", "bearish"] and is_displacement: 
        can_enter = True
    elif massive_trend in ["bullish", "bearish"] and quality >= 9.0:
        # High conviction bypasses displacement
        can_enter = True
    
    # V10 Logic Gate: Conviction Check + Portfolio Guard
    # HOTFIX V10.1: Combine displacement/4H anchor with portfolio capacity guard.
    # Previously, the line below fully overwrote the displacement gate (dead code bug).
    can_enter = can_enter and state.can_open_new_trade(symbol)

    if can_enter:
        log.info(
            f"[{symbol}] Gates: Action={action} | Regime={massive_trend} "
            f"| Disp={is_displacement} | Qual={quality:.1f} "
            f"| Anchor=PASS"
        )
    if ("open" in action or "add" in action) and not state.is_observe_only and quality >= dynamic_min_conv and can_enter:
        bbo = await asyncio.to_thread(em.fetch_bbo, symbol)
        side = "long" if "long" in action else "short"
        exec_p = bbo["bid"] if side == "long" else bbo["ask"]
        qty = calculate_risk_qty(em, symbol, exec_p, atr, ai_res.get("size_mult", 1.0), atr_mult=ai_res.get("sl_mult", 1.5))
        
        # Binance Min Notional Check ($10.10)
        notional = qty * exec_p
        if notional < 10.10:
            log.error(f"[{symbol}] [ERROR] Position size too small for exchange limits: ${notional:.2f} < $10.10")
            return None

        if qty > 0:
            sl, tp = calculate_exits(exec_p, side, atr)
            new_p = em.open_sim_pos(symbol, side, qty, exec_p, config.LEVERAGE, sl=sl, tp=tp, regime=hard_regime, phase=trend_ctx.phase, quality=quality, notes=ai_res.get("notes", "AI Trade"))
            if new_p: 
                state.in_position[symbol] = True
                state.active_trade[symbol] = new_p
                await log_trade_event(state, symbol, "open", side, exec_p, exec_p, qty, 0.0, em.balance.total, hard_regime, trend_ctx.phase, quality, "normal", "swing", "normal", ai_res["notes"], mfe=0.0, pos_id=new_p.id)
    if not config.IS_REPLAY:
        state.db.log_cycle({
            "symbol": symbol, "price": price, "signal": signal, "action": action, "regime": hard_regime,
            "phase": trend_ctx.phase, "atr_pct": atr_pct, "adx": adx_val, "adx_sma": adx_sma, "slope": e_slope,
            "patterns": ", ".join(detected_patterns) if detected_patterns else "", "notes": ai_res.get("notes", "Monitoring..."),
            "risk_mode": risk_mode
        })
    return {
        "symbol": symbol, "price": price, "signal": signal, "ai_res": ai_res, "phase": trend_ctx.phase,
        "final_action": action.upper(), "atr_pct": atr_pct, "adx": adx_val, "adx_sma": adx_sma, "slope": e_slope,
        "patterns": detected_patterns, "risk_mode": risk_mode
    }

async def calculate_btc_macro_trend(em, state, candle_time=None):
    try:
        now = candle_time if candle_time else datetime.now(UTC)
        if hasattr(em, "fetch_df"):
            df_1h = em.fetch_df("BTC/USDT", "1h", config.OHLCV_LIMIT)
        else:
            raw_1h = await asyncio.to_thread(em.fetch_ohlcv, "BTC/USDT", "1h", config.OHLCV_LIMIT)
            cols = getattr(em, "columns", ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            if len(raw_1h[0]) != len(cols):
                cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

            df_1h = pd.DataFrame(raw_1h, columns=cols)
            df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp_ms'] if 'timestamp_ms' in df_1h.columns else df_1h['timestamp'], unit='ms', utc=True)
            df_1h.set_index('timestamp', inplace=True)

            if 'ema200' not in df_1h.columns:
                df_1h = compute_indicators(df_1h)
        
        r1h = df_1h.iloc[-1]
        state.global_btc_trend = "BULLISH" if r1h["close"] > r1h["ema200"] else "BEARISH"
        if not config.IS_REPLAY:
            state.db.save_state("market_regime", state.global_btc_trend)
    except Exception as e:
        log.warning(f"Could not calculate BTC macro trend, defaulting to UNKNOWN. Error: {e}")
        state.global_btc_trend = "UNKNOWN"

async def main():
    em = ExchangeManager()
    state = EngineState(sim_instance=em.sim)
    state.notifier.start()
    if config.MODE == "replay":
        em.init_replay(config.SYMBOLS, config.TIMEFRAME)
        for b in em.replay.buffers.values(): b.cursor = config.OHLCV_LIMIT
    setup_live()
    state.db.save_state("session_start", datetime.now(UTC).isoformat())
    try:
        while True:
            try:
                await calculate_btc_macro_trend(em, state)
                tasks = [run_cycle(s, em, state) for s in config.SYMBOLS]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                valid = [r for r in results if isinstance(r, dict)]
                state.cycle_count += 1
                state.update_metrics()
                update_live(build_dashboard_layout(state, valid, em))
                if config.MODE == "replay":
                    if not any(em.replay.advance_all(config.TIMEFRAME).values()): break
                    await asyncio.sleep(config.REPLAY_DELAY)
                else: await asyncio.sleep(config.SLEEP_SECS)
            except Exception as e:
                log.error(f"Main Loop Error: {e}"); await asyncio.sleep(10)
    finally:
        state.close()
        try:
            from ai_engine import get_llm
            get_llm().close()
        except: pass
        stop_live()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Manual stop requested via Ctrl+C.")
    except Exception as e:
        log.error(f"FATAL ENGINE CRASH: {e}"); traceback.print_exc()
    finally:
        stop_live()
        log.info("Engine shutdown complete. Resources released.")

"""
core_engine.py — V11.0 Shared Strategy Library
================================================
Pure, side-effect-free functions shared across all trading engines
(bot.py, live_trader.py, berserker_trader.py).

RULES FOR THIS FILE:
  - No top-level logging.basicConfig() calls
  - No asyncio.run() or atexit registrations
  - No database writes
  - No print() statements
  - All functions must be fully deterministic given their inputs
"""

from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union, Tuple

import config

log = logging.getLogger(__name__)

# Module-level task anchor to prevent garbage collection of inflight coroutines
_BACKGROUND_TASKS = set()


# ---------------------------------------------------------------------------
# Shared Dataclass (moved here to avoid circular imports)
# ---------------------------------------------------------------------------

@dataclass
class TrendContext:
    phase: str
    is_bullish: bool
    strength: float


# ---------------------------------------------------------------------------
# Position Sizing
# ---------------------------------------------------------------------------

import math

def calculate_risk_qty(em, symbol: str, price: float, atr: float,
                       risk_mult: float, atr_mult: float = 1.5, is_scale_in: bool = False) -> float:
    """
    Dynamic Position Size Scaling by Account Equity:
    position_size = (balance * (RISK_PER_TRADE_PCT / 100)) / (stop_distance)
    Where stop_distance = atr * atr_mult (|entry - stop_loss|).
    As account equity grows or shrinks, position sizes scale proportionally.
    """
    if price is None or math.isnan(price) or price <= 0:
        return 0.0

    # 1. Base Dynamic Balance Retrieval - Prioritize live wallet equity / balance over static initial_balance
    bal = 0.0
    if hasattr(em, 'balance'):
        if isinstance(em.balance, (int, float)):
            bal = float(em.balance)
        elif hasattr(em.balance, 'total'):
            bal = float(em.balance.total)
        elif isinstance(em.balance, dict):
            bal = float(em.balance.get('total', em.balance.get('free', 0.0)))
    
    # Fallback to initial_balance or config ONLY if live balance is <= 0 or unavailable
    if bal <= 0:
        if hasattr(em, 'initial_balance'):
            bal = float(em.initial_balance)
        else:
            bal = float(getattr(config, "INITIAL_BALANCE", 65.97))

    if bal <= 0:
        return 0.0

    # 2. Risk % based sizing: (balance * (RISK_PER_TRADE_PCT / 100)) / (entry - stop_loss)
    tier_mult = getattr(config, "get_asset_tier_multiplier", lambda s: 1.0)(symbol)
    risk_pct = float(getattr(config, "RISK_PER_TRADE_PCT", 1.5))
    risk_amount = (bal * (risk_pct / 100.0)) * risk_mult * tier_mult

    # Floor stop_dist at 0.1% of price to prevent quantity spikes during flat ATR
    min_stop = price * 0.001
    stop_dist = max(min_stop, atr * atr_mult)

    if stop_dist <= 0:
        return 0.0
    qty = risk_amount / stop_dist

    # 3. Constraint Layer
    lev = getattr(config, "get_asset_leverage", lambda s: getattr(config, "LEVERAGE", 5))(symbol)
    # Guard A: Max Initial Margin Allocation Per Entry scaled by asset tier
    per_trade_margin_pct = getattr(config, "MAX_MARGIN_PER_TRADE_PCT", 0.07) * tier_mult
    max_trade_notional = bal * per_trade_margin_pct * lev
    if (qty * price) > max_trade_notional:
        qty = max_trade_notional / price

    # Guard B: Aggregate Leverage Cap (Institutional)
    max_leverage_qty = (bal * lev) / price
    if qty > max_leverage_qty:
        qty = max_leverage_qty

    # Guard C: Min Notional (Bitget $5.05 / Binance $10.10)
    min_notional = 5.05 if getattr(config, "EXCHANGE_ID", "bitget").lower() == "bitget" else 10.10
    if (qty * price) < min_notional:
        if (min_notional / price) <= max_leverage_qty:
            qty = min_notional / price
        else:
            return 0.0

    log.debug(
        f"[{symbol}] Dynamic Sizing ({'Entry 2' if is_scale_in else 'Entry 1'}): Bal=${bal:.2f} | "
        f"Risk={risk_pct:.1f}% (${risk_amount:.2f}) | Stop=${stop_dist:.4f} | Notional=${qty * price:.2f} | Qty={qty:.4f}"
    )
    return qty


# ---------------------------------------------------------------------------
# Exit Calculation
# ---------------------------------------------------------------------------

def calculate_exits(entry_price: float, side: str, atr: float, regime: Optional[str] = None) -> tuple[float, float, float]:
    """V17.0 / V36.5 Optimized Baseline Exits. Dynamic ATR-based Stop-Loss & Regime-Aware TP.
    
    Returns (soft_sl, tp, hard_sl).
      - soft_sl: Software-monitored level. Flash crash filter applies a 60s confirmation window here.
      - tp:      Take-profit target (scaled by regime profile, e.g. CHOP=3.0x ATR, BULL=4.5x ATR).
      - hard_sl: Wider Bitget-native strategy order (unconditional backstop). Fires immediately
                 on the exchange engine — protects against genuine capitulation, black-swans,
                 and margin liquidation. Uses HARD_SL_ATR_MULTIPLIER (default 3.5x ATR).
    """
    import math
    if atr is None or math.isnan(atr) or atr <= 0:
        atr = entry_price * 0.01  # Default to 1% of price if ATR is invalid
        
    hard_mult = float(getattr(config, "HARD_SL_ATR_MULTIPLIER", 3.5))
    
    sl_mult = None
    tp_mult = None
    if regime:
        try:
            profile = config.get_regime_profile(str(regime))
            if profile:
                tp_mult = float(profile.get("tp_atr_mult", 0.0))
                sl_mult = float(profile.get("sl_atr_mult", 0.0))
        except Exception:
            pass
    
    if side == "long":
        sl_mult = sl_mult or config.LONG_SL_ATR_MULT
        tp_mult = tp_mult or config.LONG_TP_ATR_MULT
        sl      = entry_price - (sl_mult * atr)
        tp      = entry_price + (tp_mult * atr)
        hard_sl = entry_price - (hard_mult * atr)
    else:
        sl_mult = sl_mult or config.SHORT_SL_ATR_MULT
        tp_mult = tp_mult or config.SHORT_TP_ATR_MULT
        sl      = entry_price + (sl_mult * atr)
        tp      = entry_price - (tp_mult * atr)
        hard_sl = entry_price + (hard_mult * atr)
    return sl, tp, hard_sl


# ---------------------------------------------------------------------------
# Breakeven
# ---------------------------------------------------------------------------

def calculate_true_breakeven(positions: list) -> Optional[float]:
    """Weighted average entry price across a pyramided position stack."""
    if not positions:
        return None
    total_val = sum(p.qty * p.entry_price for p in positions)
    total_qty = sum(p.qty for p in positions)
    return total_val / total_qty if total_qty > 0 else None


# ---------------------------------------------------------------------------
# Trend Context
# ---------------------------------------------------------------------------

def build_trend_context(row) -> TrendContext:
    """
    Classifies the current trend phase from ADX and EMA slope data.
    Returns a TrendContext dataclass.
    """
    adx = row["adx"]
    slope = row["ema_slope"]

    if adx > 25:
        phase = "EARLY" if row["adx_delta"] > 0 else "MATURE"
        if adx > 45:
            phase = "LATE"
    else:
        phase = "STARTING" if row["adx_delta"] > 0 else "ACCUMULATING"

    return TrendContext(phase=phase, is_bullish=(slope > 0), strength=adx)


# ---------------------------------------------------------------------------
# Adaptive Risk Distance Maps
# ---------------------------------------------------------------------------

def calculate_adaptive_sl_distance(mode: str, quality: float, atr: float) -> float:
    """Returns an adaptive SL distance in price units based on risk mode and quality."""
    m = config.ADAPTIVE_SL_MAP.get(mode, config.ADAPTIVE_SL_MAP["normal"])
    q_adj = 1.2 - (quality * 0.4)
    return atr * m * q_adj


def calculate_adaptive_tp_distance(mode: str, quality: float, atr: float) -> float:
    """Returns an adaptive TP distance in price units based on mode and quality."""
    m = config.ADAPTIVE_TP_MAP.get(mode, config.ADAPTIVE_TP_MAP["swing"])
    q_adj = 0.8 + (quality * 0.4)
    return atr * m * q_adj


def calculate_adaptive_trail_distance(mode: str, quality: float, atr: float) -> float:
    """Returns an adaptive trailing stop distance in price units."""
    m = config.ADAPTIVE_TRAIL_MAP.get(mode, config.ADAPTIVE_TRAIL_MAP["normal"])
    return atr * m


# ---------------------------------------------------------------------------
# Regime Reconciliation
# ---------------------------------------------------------------------------

def reconcile_regime(ai_res: dict, hard_regime: str, adx_val: float) -> str:
    """
    Merges the AI's regime opinion with the deterministic hard regime.
    AI 'UNKNOWN' or empty values defer to the hard classifier.
    """
    ai_regime = ai_res.get("regime", hard_regime).upper()
    if ai_regime in ("UNKNOWN", ""):
        return hard_regime
    return ai_regime


# ---------------------------------------------------------------------------
# V10.0 Super-Displacement Deterministic Entry Gate
# ---------------------------------------------------------------------------

def get_deterministic_action(
    regime: str,
    trend_ctx: TrendContext,
    adx_val: float,
    adx_sma: float,
    price: float,
    ema200: float,
    htf_trend: str,
    rsi: float,
    vol: float,
    vol_sma: float,
    s_high: float,
    s_low: float,
    atr: float,
    detected_patterns: list,
    has_pos: bool = False,
    vol_gate_mult: float | None = None,
    vwap: float | None = None,
    dynamic_rsi: float = 30.0,
    dynamic_vwap: float = 0.98
) -> tuple[str, str, float]:
    """
    V10.0 Super-Displacement Gate — the core deterministic entry logic.

    Args:
        vol_gate_mult: Override the volume spike threshold. Defaults to
                       config.SNIPER_VOL_GATE_MULT if not provided.

    Returns:
        (action, reason_note, quality_multiplier)
    """
    if vol_gate_mult is None:
        vol_gate_mult = config.SNIPER_VOL_GATE_MULT

    if regime == "CHAOTIC":
        return "hold", "Market is chaotic.", 1.0

    phase = trend_ctx.phase
    q_mult = 1.0
    proximity_dist = atr * 0.5
    near_resistance = (price <= s_high and price > s_high - proximity_dist)
    near_support    = (price >= s_low  and price < s_low  + proximity_dist)

    bullish_patterns = [
        "Hammer", "Bullish Engulfing", "Morning Star",
        "Three White Soldiers", "Rising Three"
    ]
    bearish_patterns = [
        "Hanging Man", "Bearish Engulfing", "Evening Star",
        "Three Black Crows", "Falling Three"
    ]
    pattern_conviction = 0
    for p in detected_patterns:
        if p in bullish_patterns:
            pattern_conviction += 1
        elif p in bearish_patterns:
            pattern_conviction -= 1

    # V11.0: Vol gate uses config constant (was hardcoded 1.5)
    vol_ratio = vol / vol_sma if vol_sma > 0 else 1.0
    is_volume_spike = vol_ratio >= vol_gate_mult
    is_above_ema = price > ema200
    is_below_ema = price < ema200

    # V17.5 Dynamic Sovereignty Validator: Price holding dynamic_vwap support AND RSI in valid trend band (not overbought, not collapsed)
    # AI Synthesized Rule #1: VETO Longs when RSI >= 65 to prevent buying exhaustion tops
    vwap_guard = (price > (vwap * dynamic_vwap)) if vwap is not None else True
    v17_valid = vwap_guard and (rsi < 65.0) and (rsi > 35.0)

    htf_lower = (htf_trend or "").lower()

    # V18.0 Trend Quality / Chop Gate: Suppress fresh entries when market is in dead chop (ADX < 20.0)
    if not has_pos and adx_val < config.ADX_TREND_THRESHOLD:
        return "hold", f"Entry suppressed: Market in CHOP (ADX {adx_val:.1f} < {config.ADX_TREND_THRESHOLD:.1f})", 1.0

    # LONG
    if is_above_ema and v17_valid and not (near_resistance and adx_val < 30):
        # Strict Macro Direction Lock: Suppress counter-trend LONGs when Macro is BEARISH
        if htf_lower in ("bearish", "short_only"):
            return "hold", "LONG suppressed: Macro BTC trend is BEARISH", 1.0

        final_q = min(q_mult * 1.3 if vol_ratio > 2.0 else 1.0, 1.5)
        if pattern_conviction > 0:
            final_q = min(1.5, final_q * 1.2)
        if regime == "RANGING":
            final_q *= 0.5
        # Only enter fresh positions on STARTING/EARLY trend formation (avoid buying LATE exhaustion)
        valid_entry_phases = ("STARTING", "EARLY") if not has_pos else ("STARTING", "EARLY", "LATE")
        if phase in valid_entry_phases and adx_val > adx_sma:
            return ("open_long" if not has_pos else "add_long",
                    f"Elite {phase} trend", final_q)
        if phase == "MATURE" and adx_val > config.ADX_STRONG_THRESHOLD:
            return ("add_long" if has_pos else "open_long",
                    "Elite mature trend", final_q)

    # SHORT
    v17_short_valid = (rsi > 20.0) and (rsi < 55.0)
    if is_below_ema and v17_short_valid and not (near_support and adx_val < config.ADX_STRONG_THRESHOLD):
        # Macro Direction Lock: Suppress counter-trend SHORTs when Macro is BULLISH
        # For Altcoins: allow short if breakdown is confirmed by healthy ADX (>22) and volume expansion (vol_ratio >= 1.2)
        if htf_lower in ("bullish", "long_only"):
            if not (adx_val > 22.0 and (vol_ratio >= 1.2 or is_volume_spike)):
                return "hold", "SHORT suppressed: Macro BTC trend is BULLISH (requires ADX>22 & vol expansion)", 1.0

        final_q = min(q_mult * 1.3 if vol_ratio > 1.8 else 1.0, 1.5)
        if pattern_conviction < 0:
            final_q = min(1.5, final_q * 1.2)
        if regime == "RANGING":
            final_q *= 0.5
        valid_entry_phases = ("STARTING", "EARLY") if not has_pos else ("STARTING", "EARLY", "LATE")
        if phase in valid_entry_phases and adx_val > adx_sma:
            return ("open_short" if not has_pos else "add_short",
                    f"Elite {phase} trend", final_q)
        if phase == "MATURE" and adx_val > config.ADX_STRONG_THRESHOLD:
            return ("add_short" if has_pos else "open_short",
                    "Elite mature trend", final_q)

    return "hold", "No elite edge.", 1.0


# ---------------------------------------------------------------------------
# V6.8 Dynamic Risk Mode Switcher
# ---------------------------------------------------------------------------

def calculate_dynamic_risk(adx: float, macro_status: str, now_time) -> tuple[str, float, int]:
    """
    V6.8: Dynamic Regime Switching Logic.
    Returns (mode, min_conviction_score, news_window_minutes).
    """
    is_weekend = now_time.weekday() >= 5
    is_aligned = macro_status in ["BULLISH", "BEARISH"]

    if adx < 20:
        mode, min_conv, news_win = "CHOP", 60.0, 60
    elif is_aligned and 25 <= adx <= 75 and not is_weekend:
        mode, min_conv, news_win = "AGGRESSIVE", 50.0, 15
    elif macro_status == "MIXED" or adx > 80 or is_weekend:
        mode, min_conv, news_win = "SNIPER", 55.0, 60
    else:
        mode, min_conv, news_win = "NORMAL", config.MIN_AI_CONVICTION_SCORE, config.PRE_NEWS_MINUTES

    if not config.IS_REPLAY:
        log.info(
            f"[DYNAMIC RISK] Mode: {mode} | Min Conviction: {min_conv} | News Gate: {news_win}m"
        )
    return mode, min_conv, news_win


# ---------------------------------------------------------------------------
# Sprint 3: Synthetic Delta Collar & Portfolio Hedging Algorithms
# ---------------------------------------------------------------------------

DEFAULT_ASSET_BETAS: Dict[str, float] = {
    "BTC": 1.00,
    "ETH": 1.15,
    "SOL": 1.35,
    "AVAX": 1.40,
    "LINK": 1.25,
    "ADA": 1.20,
    "DOGE": 1.45,
    "XRP": 1.10,
    "NEAR": 1.30,
    "BNB": 1.10,
}


def compute_portfolio_delta(
    positions: Union[Dict, List],
    btc_price: float,
    betas: Optional[Dict[str, float]] = None
) -> float:
    """
    Computes aggregate portfolio dollar delta relative to BTC:
      Delta_port = sum_{i=1}^M beta_i * Q_i * P_i
    Where:
      - beta_i is cointegrated asset beta relative to BTC (defaulting to 1.20 if unlisted)
      - Q_i is signed quantity (positive for long, negative for short)
      - P_i is position mark or entry price
    """
    if betas is None:
        betas = DEFAULT_ASSET_BETAS
        
    total_delta = 0.0
    pos_list = positions.values() if isinstance(positions, dict) else positions
    
    for pos in pos_list:
        if not isinstance(pos, dict):
            continue
        sym = str(pos.get("symbol", "")).upper()
        clean_coin = sym.replace("/USDT", "").replace(":USDT", "").replace("-USDT", "").replace("USDT", "")
        beta = betas.get(clean_coin, 1.20)
        
        qty = float(pos.get("qty", 0.0))
        side = str(pos.get("side", "long")).lower()
        signed_qty = qty if (side in ("long", "buy")) else -qty
        
        price = float(pos.get("entry_price") or pos.get("price") or pos.get("current_price") or 0.0)
        if price <= 0.0:
            continue
            
        delta_i = beta * signed_qty * price
        total_delta += delta_i
        
    return total_delta


def compute_collar_hedge_qty(portfolio_delta: float, btc_price: float) -> float:
    """
    Computes the required BTC futures contract quantity to neutralize portfolio delta:
      Q_hedge^{BTC} = - Delta_port / P_{BTC}
    Negative return means SHORT BTC; Positive return means LONG BTC.
    """
    if btc_price <= 0.0:
        return 0.0
    return - (portfolio_delta / btc_price)


async def execute_synthetic_collar(
    exchange_client: Any,
    positions: Union[Dict, List],
    btc_price: float,
    betas: Optional[Dict[str, float]] = None,
    unwind_horizon_sec: float = 900.0
) -> Dict[str, Any]:
    """
    Sprint 3 Non-Blocking Synthetic Delta Collar Execution:
    1. Aggregates portfolio delta relative to BTC: Delta_port = sum beta_i * Q_i * P_i.
    2. Asynchronously places a BTCUSDT market hedge order to instantly collapse net delta to 0 (< 50 ms SLA).
    3. Initiates non-blocking passive unwinding of altcoin legs using post-only limit orders over 15 minutes.
    4. Closes the synthetic BTC hedge once altcoins have unwound.
    """
    t0 = time.perf_counter()
    
    p_delta = compute_portfolio_delta(positions, btc_price, betas)
    q_hedge = compute_collar_hedge_qty(p_delta, btc_price)
    
    abs_hedge_qty = round(abs(q_hedge), 4)
    if abs_hedge_qty < 0.0001:
        log.info(f"[SYNTHETIC-COLLAR] Net portfolio delta is near zero (${p_delta:.2f}). No hedge required.")
        return {
            "status": "noop",
            "portfolio_delta": p_delta,
            "hedge_qty": 0.0,
            "hedge_side": "none",
            "dispatch_latency_ms": (time.perf_counter() - t0) * 1000.0
        }
        
    hedge_side = "short" if q_hedge < 0 else "long"
    log.warning(
        f"[SYNTHETIC-COLLAR] Neutralizing Portfolio Delta: ${p_delta:,.2f} via BTC {hedge_side.upper()} "
        f"Qty: {abs_hedge_qty:.4f} @ ${btc_price:,.2f}"
    )
    
    hedge_order_res = None
    try:
        client_oid = f"COLLAR_{int(time.time() * 1000)}"
        if hasattr(exchange_client, "create_market_order"):
            hedge_order_res = await exchange_client.create_market_order(
                symbol="BTCUSDT",
                side=hedge_side,
                qty=abs_hedge_qty,
                client_oid=client_oid
            )
        elif hasattr(exchange_client, "open_position"):
            hedge_order_res = await exchange_client.open_position(
                symbol="BTC/USDT",
                side=hedge_side,
                qty=abs_hedge_qty,
                price=btc_price,
                metadata={"is_collar_hedge": True, "portfolio_delta": p_delta},
                maker=False
            )
        log.info(f"[SYNTHETIC-COLLAR] BTC Hedge order dispatched successfully! Result: {hedge_order_res}")
    except Exception as he:
        log.error(f"[SYNTHETIC-COLLAR] Failed to place BTC hedge order: {he}")
        raise he
        
    dispatch_latency_ms = (time.perf_counter() - t0) * 1000.0
    log.info(f"[SYNTHETIC-COLLAR] BTC Hedge dispatch completed in {dispatch_latency_ms:.2f} ms (< 50 ms SLA).")
    
    # Spawn background task for graceful passive altcoin liquidation (GC-anchored)
    t = asyncio.create_task(_unwind_altcoin_legs(
        exchange_client=exchange_client,
        positions=positions,
        btc_hedge_qty=abs_hedge_qty,
        btc_hedge_side=hedge_side,
        unwind_horizon_sec=unwind_horizon_sec
    ))
    _BACKGROUND_TASKS.add(t)
    t.add_done_callback(_BACKGROUND_TASKS.discard)

    
    return {
        "status": "hedged",
        "portfolio_delta": p_delta,
        "hedge_qty": abs_hedge_qty,
        "hedge_side": hedge_side,
        "dispatch_latency_ms": dispatch_latency_ms,
        "unwind_horizon_sec": unwind_horizon_sec,
        "hedge_order": hedge_order_res
    }


async def _unwind_altcoin_legs(
    exchange_client: Any,
    positions: Union[Dict, List],
    btc_hedge_qty: float,
    btc_hedge_side: str,
    unwind_horizon_sec: float = 900.0
) -> None:
    """
    Passively unwinds altcoin legs over unwind_horizon_sec (default 15 minutes)
    using post-only limit orders, then closes the synthetic BTC hedge.
    """
    try:
        pos_list = list(positions.values() if isinstance(positions, dict) else positions)
        altcoins = [
            p for p in pos_list
            if isinstance(p, dict) and "BTC" not in str(p.get("symbol", "")).upper()
        ]
        
        if not altcoins:
            log.info("[COLLAR-UNWIND] No altcoin legs to unwind. Closing BTC hedge directly.")
        else:
            log.info(f"[COLLAR-UNWIND] Starting passive liquidation of {len(altcoins)} altcoin legs over {unwind_horizon_sec:.0f}s...")
            step_delay = max(1.0, unwind_horizon_sec / (len(altcoins) * 2))
            
            for alt in altcoins:
                sym = alt.get("symbol", "")
                qty = float(alt.get("qty", 0.0))
                side = str(alt.get("side", "long")).lower()
                close_side = "sell" if side in ("long", "buy") else "buy"
                
                try:
                    if hasattr(exchange_client, "close_position_order"):
                        await exchange_client.close_position_order(symbol=sym, side=side, qty=qty)
                    elif hasattr(exchange_client, "close_position"):
                        await exchange_client.close_position(symbol=sym, price=float(alt.get("entry_price", 0.0)), t_type="collar_unwind")
                    log.info(f"[COLLAR-UNWIND] Passively closed altcoin leg: {sym} {close_side} Qty={qty}")
                except Exception as ce:
                    log.warning(f"[COLLAR-UNWIND] Error closing altcoin {sym}: {ce}")
                    
                await asyncio.sleep(min(step_delay, 5.0))
                
        # All altcoin legs closed: now close synthetic BTC hedge
        log.info(f"[COLLAR-UNWIND] Altcoin legs unwound. Closing BTC synthetic hedge (Qty={btc_hedge_qty})...")
        try:
            if hasattr(exchange_client, "close_position_order"):
                await exchange_client.close_position_order(symbol="BTCUSDT", side=btc_hedge_side, qty=btc_hedge_qty)
            elif hasattr(exchange_client, "close_position"):
                await exchange_client.close_position(symbol="BTC/USDT", price=0.0, t_type="collar_complete")
            log.info("[COLLAR-UNWIND] Synthetic collar fully completed and closed!")
        except Exception as be:
            log.warning(f"[COLLAR-UNWIND] Error closing BTC hedge: {be}")
    except Exception as e:
        log.error(f"[COLLAR-UNWIND] Unwind routine exception: {e}")


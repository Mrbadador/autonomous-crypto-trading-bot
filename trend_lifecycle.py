from enum import Enum
from dataclasses import dataclass
import config

class TrendPhase(Enum):
    EARLY = "EARLY"
    MATURE = "MATURE"
    LATE = "LATE"
    EXHAUSTED = "EXHAUSTED"
    CHOP = "CHOP"
    UNKNOWN = "UNKNOWN"

@dataclass
class TrendContext:
    phase: TrendPhase
    adx: float
    adx_delta: float
    ema_div: float
    slope: float

def build_trend_context(row) -> TrendContext:
    adx = float(row["adx"])
    adx_delta = float(row["adx_delta"])
    ema_div = float(row["ema_div"])
    slope = float(row["ema_slope"])
    atr_pct_price = float(row.get("atr_pct_price", 1.0)) # Fallback to 1.0 if missing
    
    # Normalized divergence (ATR relative)
    ema_div_atr = ema_div / atr_pct_price if atr_pct_price > 0 else ema_div
    
    # 1. CHOP Detection (Weak ADX)
    if adx < config.ADX_WEAK_THRESHOLD:
        return TrendContext(TrendPhase.CHOP, adx, adx_delta, ema_div, slope)
    
    # 2. EARLY Trend (ADX rising from low levels, slope expanding)
    if config.ADX_WEAK_THRESHOLD <= adx < config.ADX_STRONG_THRESHOLD and adx_delta > 0:
        return TrendContext(TrendPhase.EARLY, adx, adx_delta, ema_div, slope)
    
    # 3. MATURE Trend (Strong ADX, high slope, stable divergence)
    if adx >= config.ADX_STRONG_THRESHOLD and adx_delta >= -0.5:
        return TrendContext(TrendPhase.MATURE, adx, adx_delta, ema_div, slope)
    
    # 4. EXHAUSTED Trend (Very high ADX but falling, massive divergence)
    if adx > config.ADX_EXHAUSTION_THRESHOLD and adx_delta < -0.5:
        return TrendContext(TrendPhase.EXHAUSTED, adx, adx_delta, ema_div, slope)
    
    # 5. LATE Trend (Strong ADX but falling)
    if adx >= config.ADX_STRONG_THRESHOLD and adx_delta < -0.5:
        return TrendContext(TrendPhase.LATE, adx, adx_delta, ema_div, slope)
    
    # 6. Over-extended / High Fatigue (Normalized divergence is extreme)
    if ema_div_atr > config.EMA_DIV_EXTREME: 
        return TrendContext(TrendPhase.EXHAUSTED, adx, adx_delta, ema_div, slope)

    return TrendContext(TrendPhase.MATURE if adx > 15 else TrendPhase.UNKNOWN, adx, adx_delta, ema_div, slope)

def apply_lifecycle_rules(action, phase: TrendPhase):
    """Gates AI actions based on the trend lifecycle."""
    if phase == TrendPhase.EXHAUSTED:
        if "add" in action or "open" in action:
            return "hold", "LIFECYCLE VETO: Trend Exhausted (High ADX falling or Over-extended)"
    
    if phase == TrendPhase.CHOP:
        if "add" in action:
            return "hold", "LIFECYCLE VETO: Pyramid blocked in CHOP phase."

    if phase == TrendPhase.LATE:
        if "open" in action:
            # Allow late entries but tree will already halve the size
            return action, None
            
    return action, None

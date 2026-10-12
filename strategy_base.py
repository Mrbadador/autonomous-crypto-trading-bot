"""
strategy_base.py — Jesse-Inspired Declarative Strategy Architecture
Unifies entry triggers (should_long, should_short), order formulation (go_long, go_short),
and in-flight trailing ratchet lifecycle (update_position, on_open_position) across both
live execution (super_bot.py / swing_trader.py) and statistical validation (quant_validation.py).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import config


@dataclass
class StrategyContext:
    """Immutable snapshot of candle & indicator state passed to declarative strategy methods."""
    symbol: str
    timeframe: str
    current_price: float
    open_price: float = 0.0
    high_price: float = 0.0
    low_price: float = 0.0
    volume: float = 0.0
    atr: float = 0.0
    adx: float = 20.0
    rsi: float = 50.0
    rsi_prev: float = 50.0
    vwap: float = 0.0
    lower_vwap_band: float = 0.0
    upper_vwap_band: float = 0.0
    ema20: float = 0.0
    tenkan: float = 0.0
    kijun: float = 0.0
    cloud_top: float = 0.0
    cloud_bottom: float = 0.0
    vah: float = 0.0
    val: float = 0.0
    td_up: int = 0
    td_down: int = 0
    daily_trend: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    market_regime: str = "CHOP"   # BULL, BEAR, CHOP
    di_spread: float = 0.0
    conviction_score: float = 0.75
    bar_index: int = 0
    timestamp_ms: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderPlan:
    """Declarative order specification returned by go_long() / go_short()."""
    symbol: str
    side: str               # "long" or "short"
    order_type: str         # "limit_post_only" or "market"
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: int
    quality_tier: str = "STANDARD_3X"
    rationale: str = ""


@dataclass
class ExitOrRatchetAction:
    """Action returned by update_position() during in-flight candle evaluation."""
    action: str             # "HOLD", "RATCHET_SL", "CLOSE_TP", "CLOSE_SL", "CLOSE_TIME_DECAY"
    new_sl: Optional[float] = None
    exit_price: Optional[float] = None
    reason: str = ""


class BaseStrategy:
    """
    Jesse-style declarative strategy base class.
    Subclasses implement pure, deterministic indicator logic decoupled from network I/O.
    """
    name: str = "BASE"
    timeframe: str = "4h"

    def should_long(self, ctx: StrategyContext) -> bool:
        return False

    def should_short(self, ctx: StrategyContext) -> bool:
        return False

    def go_long(self, ctx: StrategyContext) -> OrderPlan:
        raise NotImplementedError

    def go_short(self, ctx: StrategyContext) -> OrderPlan:
        raise NotImplementedError

    def on_open_position(self, pos: Dict[str, Any]) -> None:
        """Hook invoked when a position is opened in live or simulated execution."""
        pass

    def update_position(self, ctx: StrategyContext, pos: Dict[str, Any]) -> Optional[ExitOrRatchetAction]:
        """Evaluates active position against current bar for SL/TP hits or trailing ratchets."""
        return None


class SwingKumoStrategy(BaseStrategy):
    """
    4H Ichimoku Kumo Breakout & Breakdown Strategy (Swing Architect V36.6).
    Includes Go-Stock VPVR Value Area clearance, TD-9 exhaustion shield,
    two-stage early breakeven/profit ratchets (+0.8% -> +0.15%, +1.5% -> +0.50%),
    and ATR-calibrated stop-loss geometry.
    """
    name: str = "SWING_4H"
    timeframe: str = "4h"

    def __init__(self):
        self.min_adx = float(getattr(config, "SWING_MIN_ADX", 25.0))
        self.atr_sl_mult = float(getattr(config, "SWING_ATR_SL_MULT", 2.5))
        self.runner_tp_mult = float(getattr(config, "SWING_RUNNER_TP_MULT", 4.0))
        self.min_sl_pct = float(getattr(config, "SWING_MIN_SL_PCT", 0.015))
        self.standard_leverage = int(getattr(config, "SWING_STANDARD_LEVERAGE", 3))
        self.high_quality_leverage = int(getattr(config, "SWING_HIGH_QUALITY_LEVERAGE", 5))
        self.high_quality_adx = float(getattr(config, "SWING_HIGH_QUALITY_ADX_MIN", 30.0))
        self.high_quality_conviction = float(getattr(config, "SWING_HIGH_QUALITY_CONVICTION_MIN", 0.80))
        self.vah_tol = float(getattr(config, "SWING_VOLUME_PROFILE_VAH_TOLERANCE", 0.985))

    def should_long(self, ctx: StrategyContext) -> bool:
        c = ctx.current_price
        bullish_candle = (ctx.open_price <= 0.0) or (c > ctx.open_price)
        bullish_kumo = (
            bullish_candle and
            c > ctx.cloud_top and
            c > ctx.kijun and
            ((ctx.tenkan >= ctx.kijun * 0.995) or (c > ctx.cloud_top * 1.002))
        )
        not_1d_bear = ctx.daily_trend != "BEARISH"
        td_ok = ctx.td_up not in (9, 13)
        return bool(bullish_kumo and ctx.adx >= self.min_adx and not_1d_bear and (40.0 <= ctx.rsi <= 72.0) and td_ok)

    def should_short(self, ctx: StrategyContext) -> bool:
        c = ctx.current_price
        bearish_candle = (ctx.open_price <= 0.0) or (c < ctx.open_price)
        bearish_kumo = (
            bearish_candle and
            c < ctx.cloud_bottom and
            c < ctx.kijun and
            ((ctx.tenkan <= ctx.kijun * 1.005) or (c < ctx.cloud_bottom * 0.998))
        )
        is_1d_bear = ctx.daily_trend == "BEARISH"
        td_ok = ctx.td_down not in (9, 13)
        return bool(bearish_kumo and ctx.adx >= self.min_adx and is_1d_bear and (28.0 <= ctx.rsi <= 60.0) and td_ok)

    def go_long(self, ctx: StrategyContext) -> OrderPlan:
        entry = ctx.current_price
        sl_dist = max(self.atr_sl_mult * ctx.atr, entry * self.min_sl_pct)
        sl = round(entry - sl_dist, 6)
        tp = round(entry + (self.runner_tp_mult * sl_dist), 6)
        is_hq = (
            ctx.conviction_score >= self.high_quality_conviction and
            ctx.adx >= self.high_quality_adx and
            (40.0 <= ctx.rsi <= 65.0)
        )
        lev = self.high_quality_leverage if is_hq else self.standard_leverage
        tier = "PREMIUM_5X" if is_hq else "STANDARD_3X"
        return OrderPlan(
            symbol=ctx.symbol,
            side="long",
            order_type="market",
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            leverage=lev,
            quality_tier=tier,
            rationale=f"4H Kumo Breakout LONG (ADX={ctx.adx:.1f}, RSI={ctx.rsi:.1f}, Tier={tier})"
        )

    def go_short(self, ctx: StrategyContext) -> OrderPlan:
        entry = ctx.current_price
        sl_dist = max(self.atr_sl_mult * ctx.atr, entry * self.min_sl_pct)
        sl = round(entry + sl_dist, 6)
        tp = round(entry - (self.runner_tp_mult * sl_dist), 6)
        is_hq = (
            ctx.conviction_score >= self.high_quality_conviction and
            ctx.adx >= self.high_quality_adx and
            (35.0 <= ctx.rsi <= 60.0)
        )
        lev = self.high_quality_leverage if is_hq else self.standard_leverage
        tier = "PREMIUM_5X" if is_hq else "STANDARD_3X"
        return OrderPlan(
            symbol=ctx.symbol,
            side="short",
            order_type="market",
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            leverage=lev,
            quality_tier=tier,
            rationale=f"4H Kumo Breakdown SHORT (ADX={ctx.adx:.1f}, RSI={ctx.rsi:.1f}, Tier={tier})"
        )

    def update_position(self, ctx: StrategyContext, pos: Dict[str, Any]) -> Optional[ExitOrRatchetAction]:
        side = str(pos.get("side", "long")).lower()
        entry = float(pos["entry_price"])
        sl = float(pos["stop_loss"])
        tp = float(pos["take_profit"])
        high_p = ctx.high_price or ctx.current_price
        low_p = ctx.low_price or ctx.current_price

        if side == "long":
            if low_p <= sl:
                return ExitOrRatchetAction(action="CLOSE_SL", exit_price=sl, reason="STOP_LOSS")
            if high_p >= tp:
                return ExitOrRatchetAction(action="CLOSE_TP", exit_price=tp, reason="TAKE_PROFIT")

            peak = max(float(pos.get("peak_price", entry)), high_p)
            pos["peak_price"] = peak
            gain_pct = (peak - entry) / entry if entry > 0 else 0.0

            new_sl = sl
            if gain_pct >= 0.015:
                new_sl = max(new_sl, entry * 1.005)
            elif gain_pct >= 0.008:
                new_sl = max(new_sl, entry * 1.0015)

            if ctx.atr > 0 and (peak - entry) >= (2.0 * ctx.atr):
                chandelier_sl = peak - (2.2 * ctx.atr)
                new_sl = max(new_sl, chandelier_sl)

            if new_sl > sl and new_sl < ctx.current_price:
                pos["stop_loss"] = new_sl
                return ExitOrRatchetAction(action="RATCHET_SL", new_sl=new_sl, reason="SWING_RATCHET")
        else:
            if high_p >= sl:
                return ExitOrRatchetAction(action="CLOSE_SL", exit_price=sl, reason="STOP_LOSS")
            if low_p <= tp:
                return ExitOrRatchetAction(action="CLOSE_TP", exit_price=tp, reason="TAKE_PROFIT")

            trough = min(float(pos.get("peak_price", entry)), low_p)
            pos["peak_price"] = trough
            gain_pct = (entry - trough) / entry if entry > 0 else 0.0

            new_sl = sl
            if gain_pct >= 0.015:
                new_sl = min(new_sl, entry * 0.995)
            elif gain_pct >= 0.008:
                new_sl = min(new_sl, entry * 0.9985)

            if ctx.atr > 0 and (entry - trough) >= (2.0 * ctx.atr):
                chandelier_sl = trough + (2.2 * ctx.atr)
                new_sl = min(new_sl, chandelier_sl)

            if new_sl < sl and new_sl > ctx.current_price:
                pos["stop_loss"] = new_sl
                return ExitOrRatchetAction(action="RATCHET_SL", new_sl=new_sl, reason="SWING_RATCHET")

        return ExitOrRatchetAction(action="HOLD")


class ChopGridMakerStrategy(BaseStrategy):
    """
    5m Mean-Reversion Maker Dip-Buy Strategy (CHOP Regime).
    Enters on RSI < 30 & Close <= 24-period VWAP Lower Band (-2.0 std) with ADX < 24.
    """
    name: str = "GRID_MAKER"
    timeframe: str = "5m"

    def __init__(self):
        self.rsi_thresh = float(getattr(config, "GRID_RSI_THRESHOLD", 30.0))
        self.tp_pct = float(getattr(config, "GRID_TP_PCT", 0.0135))
        self.sl_pct = float(getattr(config, "GRID_SL_PCT", 0.0115))
        self.max_adx = float(getattr(config, "GRID_MAX_5M_ADX", 24.0))
        self.leverage = int(getattr(config, "GRID_LEVERAGE", 3))

    def should_long(self, ctx: StrategyContext) -> bool:
        if ctx.adx >= self.max_adx:
            return False
        if ctx.adx >= 22.0 and ctx.di_spread < -4.0:
            return False
        c_range = max(ctx.high_price - ctx.low_price, 1e-6)
        lower_wick_ratio = (min(ctx.open_price, ctx.current_price) - ctx.low_price) / c_range
        rebound_hook = (ctx.current_price >= ctx.open_price) or (lower_wick_ratio >= 0.25) or (ctx.rsi > ctx.rsi_prev)
        return bool(ctx.rsi <= self.rsi_thresh and ctx.current_price <= ctx.lower_vwap_band and rebound_hook)

    def go_long(self, ctx: StrategyContext) -> OrderPlan:
        entry = ctx.current_price
        sl = round(entry * (1.0 - self.sl_pct), 6)
        tp = round(entry * (1.0 + self.tp_pct), 6)
        return OrderPlan(
            symbol=ctx.symbol,
            side="long",
            order_type="limit_post_only",
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            leverage=self.leverage,
            quality_tier="MAKER_3X",
            rationale=f"5m Chop Dip-Buy (RSI={ctx.rsi:.1f} <= {self.rsi_thresh:.0f}, Price <= Lower VWAP Band)"
        )

    def update_position(self, ctx: StrategyContext, pos: Dict[str, Any]) -> Optional[ExitOrRatchetAction]:
        entry = float(pos["entry_price"])
        sl = float(pos["stop_loss"])
        tp = float(pos["take_profit"])
        high_p = ctx.high_price or ctx.current_price
        low_p = ctx.low_price or ctx.current_price

        if low_p <= sl:
            return ExitOrRatchetAction(action="CLOSE_SL", exit_price=sl, reason="STOP_LOSS")
        if high_p >= tp:
            return ExitOrRatchetAction(action="CLOSE_TP", exit_price=tp, reason="TAKE_PROFIT")

        peak = max(float(pos.get("peak_price", entry)), high_p)
        pos["peak_price"] = peak
        gain_pct = (peak - entry) / entry if entry > 0 else 0.0

        new_sl = sl
        if gain_pct >= 0.008:
            new_sl = max(new_sl, entry * 1.0015)

        if new_sl > sl and new_sl < ctx.current_price:
            pos["stop_loss"] = new_sl
            return ExitOrRatchetAction(action="RATCHET_SL", new_sl=new_sl, reason="GRID_BE_RATCHET")

        return ExitOrRatchetAction(action="HOLD")


class BearScalpShortStrategy(BaseStrategy):
    """
    Quick Turnover Scalp Short Strategy (CHOP Range-Top Fade & BEAR Relief Bounce Fade).
    Enforces +1.20% TP, 1.00% SL, +0.50% fast breakeven ratchet, and 45-minute time-decay exit.
    """
    name: str = "SCALP_SHORT"
    timeframe: str = "5m"

    def __init__(self):
        self.tp_pct = float(getattr(config, "SCALP_SHORT_TP_PCT", 0.0120))
        self.sl_pct = float(getattr(config, "SCALP_SHORT_SL_PCT", 0.0100))
        self.be_trigger = float(getattr(config, "SCALP_SHORT_BE_TRIGGER_PCT", 0.0050))
        self.rsi_overbought = float(getattr(config, "SCALP_SHORT_RSI_OVERBOUGHT", 68.0))
        self.min_rsi = float(getattr(config, "SCALP_SHORT_MIN_RSI", 38.0))
        self.max_adx = float(getattr(config, "SCALP_SHORT_MAX_5M_ADX", 22.0))
        self.leverage = int(getattr(config, "GRID_LEVERAGE", 3))
        self.max_hold_bars_5m = 9  # 45 minutes (9 x 5m bars)

    def should_short(self, ctx: StrategyContext) -> bool:
        if ctx.rsi < self.min_rsi:
            return False

        c_range = max(ctx.high_price - ctx.low_price, 1e-6)
        upper_wick = ctx.high_price - max(ctx.open_price, ctx.current_price)
        upper_wick_ratio = upper_wick / c_range
        is_red = ctx.current_price < ctx.open_price
        is_shooting_star = upper_wick_ratio >= 0.35
        is_rsi_falling = ctx.rsi < (ctx.rsi_prev - 0.5)
        has_bearish_reversal = (is_red and upper_wick_ratio >= 0.25) or (is_shooting_star and is_rsi_falling)

        if ctx.market_regime == "BEAR":
            bounced = (ctx.current_price >= ctx.vwap * 0.997) or (ctx.current_price >= ctx.ema20 * 0.998)
            return bool(bounced and (45.0 <= ctx.rsi <= 68.0) and has_bearish_reversal)
        else:
            # CHOP Range Top Fade
            if ctx.adx >= self.max_adx:
                return False
            return bool(
                ctx.current_price >= (ctx.upper_vwap_band * 0.998) and
                ctx.rsi >= self.rsi_overbought and
                has_bearish_reversal
            )

    def go_short(self, ctx: StrategyContext) -> OrderPlan:
        entry = ctx.current_price
        sl = round(entry * (1.0 + self.sl_pct), 6)
        tp = round(entry * (1.0 - self.tp_pct), 6)
        return OrderPlan(
            symbol=ctx.symbol,
            side="short",
            order_type="limit_post_only",
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            leverage=self.leverage,
            quality_tier="SCALP_SHORT_3X",
            rationale=f"Quick Scalp Short ({ctx.market_regime} Fade, RSI={ctx.rsi:.1f})"
        )

    def update_position(self, ctx: StrategyContext, pos: Dict[str, Any]) -> Optional[ExitOrRatchetAction]:
        entry = float(pos["entry_price"])
        sl = float(pos["stop_loss"])
        tp = float(pos["take_profit"])
        high_p = ctx.high_price or ctx.current_price
        low_p = ctx.low_price or ctx.current_price

        if high_p >= sl:
            return ExitOrRatchetAction(action="CLOSE_SL", exit_price=sl, reason="STOP_LOSS")
        if low_p <= tp:
            return ExitOrRatchetAction(action="CLOSE_TP", exit_price=tp, reason="TAKE_PROFIT")

        bars_held = ctx.bar_index - int(pos.get("entry_bar", ctx.bar_index))
        if bars_held >= self.max_hold_bars_5m:
            return ExitOrRatchetAction(action="CLOSE_TIME_DECAY", exit_price=ctx.current_price, reason="TIME_DECAY_45M")

        trough = min(float(pos.get("peak_price", entry)), low_p)
        pos["peak_price"] = trough
        gain_pct = (entry - trough) / entry if entry > 0 else 0.0

        if gain_pct >= self.be_trigger:
            new_sl = min(sl, entry * 0.9995)
            if new_sl < sl and new_sl > ctx.current_price:
                pos["stop_loss"] = new_sl
                return ExitOrRatchetAction(action="RATCHET_SL", new_sl=new_sl, reason="SCALP_SHORT_BE_RATCHET")

        return ExitOrRatchetAction(action="HOLD")

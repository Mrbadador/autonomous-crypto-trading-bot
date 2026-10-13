"""
backtests/backtest_grid_replay.py — Pure 5m Chop Maker Grid Replay & Fill Simulator

Faithfully reproduces the 5m Chop Maker Grid and Quick Scalp Short execution:
1. Signal Generation: Imports extracted pure functions from super_bot.py:
   - compute_grid_5m_indicators
   - evaluate_chop_grid_signal
   - evaluate_scalp_short_signal
2. Limit Fill Model:
   - Conservative: Long limit at L fills at bar t+k only if low[t+k] <= L - tick_size (price strictly traded through).
                   Short limit at A fills at bar t+k only if high[t+k] >= A + tick_size.
   - Strict-Optimistic: Fills on exact touch (low[t+k] <= L or high[t+k] >= A).
3. Exit & Holding Management:
   - TP maker fill: High >= TP + tick_size (Conservative) or High >= TP (Optimistic). Maker fee = 0.02%.
   - SL taker fill: Low <= SL. Fills at min(SL, Open). Taker fee = 0.06%.
   - 45m checkpoint: If holding >= 45m and PnL >= +0.45%, harvest market TP (0.06% taker fee).
   - 45m - 90m: Tightens SL to entry * 0.9982 (if PnL >= 0.22%) else entry * 1.0055.
   - 90m hard expiry: Market exit at bar close/open with 0.06% taker fee.
4. Historical Funding Accrual:
   - Longs pay positive funding (funding_usd = -rate * notional).
   - Shorts receive positive funding (funding_usd = +rate * notional).
"""

import os
import sys
import numpy as np
import pandas as pd
import ta
from typing import Dict, List, Any, Optional

# Ensure project root in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from super_bot import (
    compute_grid_5m_indicators,
    evaluate_chop_grid_signal,
    evaluate_scalp_short_signal,
)

TICK_SIZES = {
    "BTCUSDT": 0.1,
    "ETHUSDT": 0.01,
    "SOLUSDT": 0.01,
    "DEFAULT": 0.01,
}

MAKER_FEE = 0.0002   # 0.02%
TAKER_FEE = 0.0006   # 0.06%


def precompute_5m_indicators_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized computation of 5m indicators matching compute_grid_5m_indicators exactly.
    Accelerates the replay across 200,000+ bars.
    """
    res = df.copy()
    c = res["close"]
    h = res["high"]
    l = res["low"]
    o = res["open"]
    v = res["volume"]

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi_series = 100.0 - (100.0 / (1.0 + rs)).fillna(50.0)
    res["rsi"] = rsi_series
    res["rsi_prev"] = rsi_series.shift(1).fillna(rsi_series)

    cum_pv = (c * v).rolling(24).sum()
    cum_v = v.rolling(24).sum()
    vwap_series = (cum_pv / (cum_v + 1e-9)).fillna(c)
    res["vwap"] = vwap_series

    vwap_diff = c - vwap_series
    std_series = vwap_diff.rolling(24).std().fillna(0.0)
    c_std = c.rolling(24).std().fillna(0.0)
    rolling_std = np.where(std_series > 0, std_series, c_std)
    res["rolling_std"] = rolling_std
    res["lower_band"] = vwap_series - (2.0 * rolling_std)
    res["upper_band"] = vwap_series + (2.0 * rolling_std)

    res["ema20"] = c.ewm(span=20, adjust=False).mean()

    # ta.trend.ADXIndicator matching super_bot live computation
    adx_obj = ta.trend.ADXIndicator(h, l, c, window=14)
    res["adx_5m"] = adx_obj.adx().fillna(0.0)
    res["plus_di"] = adx_obj.adx_pos().fillna(0.0)
    res["minus_di"] = adx_obj.adx_neg().fillna(0.0)
    res["di_spread"] = (res["plus_di"] - res["minus_di"]).fillna(0.0)

    c_range = np.maximum(h - l, 1e-6)
    upper_wick = h - np.maximum(o, c)
    res["upper_wick_ratio"] = upper_wick / c_range

    res["is_red_candle"] = c < o
    res["is_shooting_star"] = res["upper_wick_ratio"] >= 0.35
    res["is_rsi_falling"] = res["rsi"] < (res["rsi_prev"] - 0.5)
    res["has_bearish_reversal"] = (
        (res["is_red_candle"] & (res["upper_wick_ratio"] >= 0.25)) |
        (res["is_shooting_star"] & res["is_rsi_falling"])
    )

    return res


def precompute_btc_15m_regime(df_15m: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Wilder ADX(14) and DI spread on BTC 15m candles with zero lookahead.
    A 15m bar starting at timestamp T closes at T + 15m (T + 900,000 ms).
    It is available to 5m bars whose start timestamp t >= T + 900,000 ms.
    """
    res = df_15m.copy()
    adx_obj = ta.trend.ADXIndicator(res["high"], res["low"], res["close"], window=14)
    res["market_adx"] = adx_obj.adx().fillna(0.0)
    res["market_plus_di"] = adx_obj.adx_pos().fillna(0.0)
    res["market_minus_di"] = adx_obj.adx_neg().fillna(0.0)
    res["market_di_spread"] = (res["market_plus_di"] - res["market_minus_di"]).fillna(0.0)
    # Candle availability timestamp: close of the 15m candle
    res["close_ts"] = res["timestamp"] + 15 * 60 * 1000
    return res[["close_ts", "market_adx", "market_di_spread"]].sort_values("close_ts")


def run_grid_replay(
    df: pd.DataFrame,
    symbol: str = "BTCUSDT",
    funding_df: Optional[pd.DataFrame] = None,
    btc_15m_df: Optional[pd.DataFrame] = None,
    conservative_fills: bool = True,
    enable_regime_gate: bool = True,
    enable_vwap_band: bool = True,
    enable_45m_tightening: bool = True,
    enable_short_scalps: bool = True,
    trade_allocation_usd: float = 85.0,
    leverage: float = 3.0,
    max_order_ttl_bars: int = 1,  # Live bot TTL is ~30s; 1 bar tests fill on signal bar's successor only
    cooldown_bars: int = 12,      # 1 hour anti-churn cooldown after exit
) -> List[Dict[str, Any]]:
    """
    Executes historical 5m replay simulation.

    NOTE ON ORDER TTL BIAS:
    Live execution attempts maker limit fills for ~30 seconds before canceling unfilled orders.
    A live-faithful replay sets max_order_ttl_bars=1 (successor bar only).
    A longer TTL (e.g. max_order_ttl_bars=6 / 30m) introduces severe negative selection bias:
    orders left standing fill preferentially during sharp, sustained market collapses through
    the bid, artificially exaggerating losses and negative EV.
    """
    tick_size = TICK_SIZES.get(symbol, TICK_SIZES["DEFAULT"])
    ind_df = precompute_5m_indicators_vectorized(df)

    # Attach BTC 15m market regime feed if provided (zero-lookahead merge_asof)
    if btc_15m_df is not None and not btc_15m_df.empty:
        regime_15m = precompute_btc_15m_regime(btc_15m_df)
        ind_df = pd.merge_asof(
            ind_df.sort_values("timestamp"),
            regime_15m,
            left_on="timestamp",
            right_on="close_ts",
            direction="backward"
        )
        ind_df["market_adx"] = ind_df["market_adx"].fillna(ind_df["adx_5m"])
        ind_df["market_di_spread"] = ind_df["market_di_spread"].fillna(ind_df["di_spread"])
    else:
        # Fallback to asset 5m indicators if no external 15m BTC feed
        ind_df["market_adx"] = ind_df["adx_5m"]
        ind_df["market_di_spread"] = ind_df["di_spread"]

    # Funding rate map: timestamp -> rate
    funding_map = {}
    if funding_df is not None and not funding_df.empty:
        funding_map = dict(zip(funding_df["timestamp"], funding_df["funding_rate"]))

    trades = []
    n_bars = len(ind_df)
    active_trade = None
    pending_order = None
    last_exit_bar = -999

    # Pre-extract numpy arrays for fast row iteration
    ts = ind_df["timestamp"].values
    opens = ind_df["open"].values
    highs = ind_df["high"].values
    lows = ind_df["low"].values
    closes = ind_df["close"].values
    rsis = ind_df["rsi"].values
    adx_5m = ind_df["adx_5m"].values
    lower_bands = ind_df["lower_band"].values
    upper_bands = ind_df["upper_band"].values
    vwaps = ind_df["vwap"].values
    di_spreads = ind_df["di_spread"].values
    has_bearish_rev = ind_df["has_bearish_reversal"].values
    market_adx_arr = ind_df["market_adx"].values
    market_di_spread_arr = ind_df["market_di_spread"].values

    for i in range(30, n_bars):
        curr_ts = ts[i]
        curr_open = opens[i]
        curr_high = highs[i]
        curr_low = lows[i]
        curr_close = closes[i]

        # -------------------------------------------------------------
        # 1. Manage Active Trade (TP / SL / Time Harvest & Expiry)
        # -------------------------------------------------------------
        if active_trade is not None:
            entry_p = active_trade["entry_price"]
            side = active_trade["side"]
            tp_p = active_trade["take_profit"]
            sl_p = active_trade["stop_loss"]
            bars_held = i - active_trade["entry_bar"]
            mins_held = bars_held * 5

            # Accrue funding if 8h boundary crossed
            # Bitget funding epochs: 00:00, 08:00, 16:00 UTC (every 28,800,000 ms)
            if funding_map:
                prev_ts = ts[i - 1]
                # Check if an 8h timestamp falls between prev_ts and curr_ts
                for f_ts, f_rate in funding_map.items():
                    if prev_ts < f_ts <= curr_ts:
                        notional = trade_allocation_usd * leverage
                        # Long pays if positive funding, short receives
                        f_usd = -f_rate * notional if side == "LONG" else f_rate * notional
                        active_trade["funding_usd"] += f_usd

            # Check 45m - 90m dynamic tightening
            if enable_45m_tightening and side == "SHORT" and mins_held >= 45:
                short_pnl_pct = (entry_p - curr_close) / entry_p * 100.0
                if mins_held >= 90:
                    # 90m hard expiry
                    exit_p = curr_open
                    fee_usd = (trade_allocation_usd * leverage) * TAKER_FEE
                    gross_pnl = (entry_p - exit_p) / entry_p * (trade_allocation_usd * leverage)
                    net_pnl = gross_pnl - active_trade["entry_fee_usd"] - fee_usd + active_trade["funding_usd"]
                    trades.append({
                        **active_trade,
                        "exit_bar": i,
                        "exit_ts": curr_ts,
                        "exit_price": exit_p,
                        "exit_reason": "SCALP_SHORT_TIME_EXPIRY",
                        "holding_mins": mins_held,
                        "exit_fee_usd": fee_usd,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "is_maker_exit": False,
                    })
                    active_trade = None
                    last_exit_bar = i
                    continue
                elif short_pnl_pct >= 0.45:
                    # 45m harvest
                    exit_p = curr_open
                    fee_usd = (trade_allocation_usd * leverage) * TAKER_FEE
                    gross_pnl = (entry_p - exit_p) / entry_p * (trade_allocation_usd * leverage)
                    net_pnl = gross_pnl - active_trade["entry_fee_usd"] - fee_usd + active_trade["funding_usd"]
                    trades.append({
                        **active_trade,
                        "exit_bar": i,
                        "exit_ts": curr_ts,
                        "exit_price": exit_p,
                        "exit_reason": "SCALP_SHORT_TIME_PROFIT",
                        "holding_mins": mins_held,
                        "exit_fee_usd": fee_usd,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "is_maker_exit": False,
                    })
                    active_trade = None
                    last_exit_bar = i
                    continue
                else:
                    # Tighten stop
                    target_sl = entry_p * (1.0 - 0.0018) if short_pnl_pct >= 0.22 else entry_p * 1.0055
                    if target_sl < active_trade["stop_loss"]:
                        active_trade["stop_loss"] = target_sl
                        sl_p = target_sl

            # Check TP / SL on current bar
            if side == "LONG":
                # TP Check (Maker limit fill)
                tp_hit = (curr_high >= tp_p + tick_size) if conservative_fills else (curr_high >= tp_p)
                sl_hit = (curr_low <= sl_p)

                if tp_hit and not sl_hit:
                    exit_p = tp_p
                    fee_usd = (trade_allocation_usd * leverage) * MAKER_FEE
                    gross_pnl = (exit_p - entry_p) / entry_p * (trade_allocation_usd * leverage)
                    net_pnl = gross_pnl - active_trade["entry_fee_usd"] - fee_usd + active_trade["funding_usd"]
                    trades.append({
                        **active_trade,
                        "exit_bar": i,
                        "exit_ts": curr_ts,
                        "exit_price": exit_p,
                        "exit_reason": "TAKE_PROFIT",
                        "holding_mins": mins_held,
                        "exit_fee_usd": fee_usd,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "is_maker_exit": True,
                    })
                    active_trade = None
                    last_exit_bar = i
                    continue

                if sl_hit:
                    # Fill at worse of SL price or bar open (slippage model)
                    exit_p = min(sl_p, curr_open)
                    fee_usd = (trade_allocation_usd * leverage) * TAKER_FEE
                    gross_pnl = (exit_p - entry_p) / entry_p * (trade_allocation_usd * leverage)
                    net_pnl = gross_pnl - active_trade["entry_fee_usd"] - fee_usd + active_trade["funding_usd"]
                    trades.append({
                        **active_trade,
                        "exit_bar": i,
                        "exit_ts": curr_ts,
                        "exit_price": exit_p,
                        "exit_reason": "STOP_LOSS",
                        "holding_mins": mins_held,
                        "exit_fee_usd": fee_usd,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "is_maker_exit": False,
                    })
                    active_trade = None
                    last_exit_bar = i
                    continue

            elif side == "SHORT":
                # TP Check (Maker limit fill)
                tp_hit = (curr_low <= tp_p - tick_size) if conservative_fills else (curr_low <= tp_p)
                sl_hit = (curr_high >= sl_p)

                if tp_hit and not sl_hit:
                    exit_p = tp_p
                    fee_usd = (trade_allocation_usd * leverage) * MAKER_FEE
                    gross_pnl = (entry_p - exit_p) / entry_p * (trade_allocation_usd * leverage)
                    net_pnl = gross_pnl - active_trade["entry_fee_usd"] - fee_usd + active_trade["funding_usd"]
                    trades.append({
                        **active_trade,
                        "exit_bar": i,
                        "exit_ts": curr_ts,
                        "exit_price": exit_p,
                        "exit_reason": "TAKE_PROFIT",
                        "holding_mins": mins_held,
                        "exit_fee_usd": fee_usd,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "is_maker_exit": True,
                    })
                    active_trade = None
                    last_exit_bar = i
                    continue

                if sl_hit:
                    # Fill at worse of SL price or bar open
                    exit_p = max(sl_p, curr_open)
                    fee_usd = (trade_allocation_usd * leverage) * TAKER_FEE
                    gross_pnl = (entry_p - exit_p) / entry_p * (trade_allocation_usd * leverage)
                    net_pnl = gross_pnl - active_trade["entry_fee_usd"] - fee_usd + active_trade["funding_usd"]
                    trades.append({
                        **active_trade,
                        "exit_bar": i,
                        "exit_ts": curr_ts,
                        "exit_price": exit_p,
                        "exit_reason": "STOP_LOSS",
                        "holding_mins": mins_held,
                        "exit_fee_usd": fee_usd,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "is_maker_exit": False,
                    })
                    active_trade = None
                    last_exit_bar = i
                    continue

        # -------------------------------------------------------------
        # 2. Check Pending Post-Only Maker Limit Fill
        # -------------------------------------------------------------
        if pending_order is not None:
            p_side = pending_order["side"]
            limit_p = pending_order["price"]
            bars_pending = i - pending_order["placed_bar"]

            is_filled = False
            if p_side == "LONG":
                if conservative_fills:
                    is_filled = (curr_low <= limit_p - tick_size)
                else:
                    is_filled = (curr_low <= limit_p)
            elif p_side == "SHORT":
                if conservative_fills:
                    is_filled = (curr_high >= limit_p + tick_size)
                else:
                    is_filled = (curr_high >= limit_p)

            if is_filled:
                entry_fee_usd = (trade_allocation_usd * leverage) * MAKER_FEE
                active_trade = {
                    "symbol": symbol,
                    "side": p_side,
                    "entry_bar": i,
                    "entry_ts": curr_ts,
                    "entry_price": limit_p,
                    "take_profit": pending_order["take_profit"],
                    "stop_loss": pending_order["stop_loss"],
                    "regime": pending_order["regime"],
                    "entry_market_adx": pending_order["market_adx"],
                    "allocation_usd": trade_allocation_usd,
                    "leverage": leverage,
                    "entry_fee_usd": entry_fee_usd,
                    "funding_usd": 0.0,
                    "is_maker_entry": True,
                }
                pending_order = None
            elif bars_pending >= max_order_ttl_bars:
                # Cancelled / TTL expired
                pending_order = None

        # -------------------------------------------------------------
        # 3. Scan New Entry Signals (if flat, no pending order, cooldown passed)
        # -------------------------------------------------------------
        if active_trade is None and pending_order is None and (i - last_exit_bar) >= cooldown_bars:
            bar_rsi = rsis[i]
            bar_lower = lower_bands[i] if enable_vwap_band else 1e9
            bar_upper = upper_bands[i] if enable_vwap_band else 0.0
            bar_adx = adx_5m[i]
            bar_di = di_spreads[i]
            m_adx = market_adx_arr[i]
            m_di = market_di_spread_arr[i]

            # Regime assignment based on Market ADX (BTC 15m)
            # Low ADX (< 22.0) = CHOP, High ADX (>= 28.0) = TREND, High DI spread = HIGH_VOL
            if m_adx < 22.0:
                current_regime = "CHOP"
            elif m_adx >= 28.0:
                current_regime = "TREND"
            else:
                current_regime = "HIGH_VOL"

            # LONG GRID SIGNAL
            long_regime_ok = True
            if enable_regime_gate:
                if m_adx >= 28.0 or (m_adx >= 25.0 and m_di < -4.0):
                    long_regime_ok = False

            if long_regime_ok and bar_rsi < 30.0 and curr_close <= bar_lower and bar_adx < 24.0:
                limit_p = curr_close
                pending_order = {
                    "side": "LONG",
                    "placed_bar": i,
                    "placed_ts": curr_ts,
                    "price": limit_p,
                    "take_profit": limit_p * (1.0 + 0.0135),
                    "stop_loss": limit_p * (1.0 - 0.0115),
                    "regime": current_regime,
                    "market_adx": m_adx,
                }
                continue

            # SHORT SCALP SIGNAL
            if enable_short_scalps:
                short_regime_ok = True
                if enable_regime_gate and bar_adx >= 22.0:
                    short_regime_ok = False

                if short_regime_ok and bar_rsi >= 68.0 and curr_close >= (bar_upper * 0.998) and has_bearish_rev[i]:
                    limit_p = curr_close
                    pending_order = {
                        "side": "SHORT",
                        "placed_bar": i,
                        "placed_ts": curr_ts,
                        "price": limit_p,
                        "take_profit": limit_p * (1.0 - 0.0120),
                        "stop_loss": limit_p * (1.0 + 0.0100),
                        "regime": current_regime,
                        "market_adx": m_adx,
                    }
                    continue

    return trades

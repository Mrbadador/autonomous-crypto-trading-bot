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
    "SOLUSDT": 0.001,
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


def evaluate_limit_fill(
    side: str,
    limit_price: float,
    curr_open: float,
    curr_high: float,
    curr_low: float,
    tick_size: float,
    fill_model: str = "one_tick",
    depth_ratio: float = 0.5,
    seed: Optional[int] = None
) -> bool:
    """
    Evaluates whether a passive limit order is filled according to the specified fill model:
    - 'touch': Fills immediately upon price touching the limit level (Strict-Optimistic).
    - 'one_tick': Fills only if price strictly trades through the limit level by >= 1 tick (Conservative).
    - 'logprob': Probabilistic queue fill model: P(fill) = 1 - exp(-k * penetration_ratio).
    - 'squareprob': Probabilistic queue fill model: P(fill) = (penetration_ratio)^2 clipped [0, 1].

    NOTE: Probabilistic models (logprob, squareprob) represent queue depletion dynamics.
    Until calibrated against empirical L2 books15 recorded depth snapshots, touch and one_tick
    serve as upper and lower bounds on execution expectancy.
    """
    if side == "LONG":
        # Price must reach or breach the limit bid
        if curr_low > limit_price:
            return False
        if fill_model == "touch":
            return True
        if fill_model == "one_tick":
            return curr_low <= (limit_price - tick_size)

        # Depth-through penetration
        penetration = max(0.0, limit_price - curr_low)
        bar_range = max(curr_high - curr_low, tick_size)
        rel_penetration = penetration / bar_range

        rng = np.random.default_rng(seed)
        u = rng.uniform(0.0, 1.0)

        if fill_model == "logprob":
            # LogProb queue model: P(fill) = 1 - exp(-3.0 * rel_penetration)
            p_fill = 1.0 - np.exp(-3.0 * rel_penetration)
            return u <= p_fill
        elif fill_model == "squareprob":
            # SquareProb queue model: P(fill) = min(1.0, (rel_penetration * 1.5)^2)
            p_fill = min(1.0, (rel_penetration * 1.5) ** 2)
            return u <= p_fill
        return curr_low <= (limit_price - tick_size)

    elif side == "SHORT":
        if curr_high < limit_price:
            return False
        if fill_model == "touch":
            return True
        if fill_model == "one_tick":
            return curr_high >= (limit_price + tick_size)

        penetration = max(0.0, curr_high - limit_price)
        bar_range = max(curr_high - curr_low, tick_size)
        rel_penetration = penetration / bar_range

        rng = np.random.default_rng(seed)
        u = rng.uniform(0.0, 1.0)

        if fill_model == "logprob":
            p_fill = 1.0 - np.exp(-3.0 * rel_penetration)
            return u <= p_fill
        elif fill_model == "squareprob":
            p_fill = min(1.0, (rel_penetration * 1.5) ** 2)
            return u <= p_fill
        return curr_high >= (limit_price + tick_size)

    return False


def run_grid_replay(
    df: pd.DataFrame,
    symbol: str = "BTCUSDT",
    funding_df: Optional[pd.DataFrame] = None,
    btc_15m_df: Optional[pd.DataFrame] = None,
    conservative_fills: bool = True,
    fill_model: Optional[str] = None,  # 'touch', 'one_tick', 'logprob', 'squareprob'
    enable_regime_gate: bool = True,
    enable_vwap_band: bool = True,
    enable_45m_tightening: bool = True,
    enable_short_scalps: bool = True,
    trade_allocation_usd: float = 85.0,
    leverage: float = 3.0,
    max_order_ttl_bars: int = 1,
    cooldown_bars: int = 12,
    enable_funding_regime_gate: bool = False,  # D3 ablation arm: suppress longs if trailing funding > 80th pct
    seeded_positions: Optional[List[Dict[str, Any]]] = None,  # FIX 1: boardroom veto shadow — pre-place a position at entry_ts (taker market entry), skip signal scanning
    scan_signals: bool = True,  # FIX 1: set False for pure exit-management replays
) -> List[Dict[str, Any]]:
    """
    Executes historical 5m replay simulation.

    NOTE ON ORDER TTL BIAS:
    Live execution attempts maker limit fills for ~30 seconds before canceling unfilled orders.
    A live-faithful replay sets max_order_ttl_bars=1 (successor bar only).
    A longer TTL (e.g. max_order_ttl_bars=6 / 30m) introduces severe negative selection bias:
    orders left standing fill preferentially during sharp, sustained market collapses through
    the bid, artificially exaggerating losses and negative EV.

    SEEDED POSITIONS (FIX 1, counterfactual shadow replay):
    Each seed dict: {side, entry_price, take_profit, stop_loss, entry_ts, symbol, label}.
    The position is opened taker at the entry bar's close (entry_price must equal that close;
    the caller is responsible for reconstruction), then managed by the identical TP/SL/funding
    state machine used by signal-driven replays. scan_signals=False disables new signal entry.
    """
    tick_size = TICK_SIZES.get(symbol, TICK_SIZES["DEFAULT"])
    effective_fill_model = fill_model if fill_model is not None else ("one_tick" if conservative_fills else "touch")
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
        # FIX 9: Missing market regime must block entries (treat as ADX 99.0, DI spread 0.0), no fallback contamination
        missing_count = int(ind_df["market_adx"].isna().sum())
        ind_df["market_adx"] = ind_df["market_adx"].fillna(99.0)
        ind_df["market_di_spread"] = ind_df["market_di_spread"].fillna(0.0)
        if missing_count > 0:
            print(f"[{symbol}] Blocked {missing_count} unaligned bars at start of series with market_adx=99.0")
    else:
        # If no external 15m BTC feed, block entries by default or treat as unaligned
        ind_df["market_adx"] = 99.0
        ind_df["market_di_spread"] = 0.0

    # Funding rate map: timestamp -> rate
    # FIX 10: Bitget funding dataset starts in mid-2026. Prepend synthetic 8h settlements
    # at symbol's median observed rate across the missing ~22-month window.
    funding_map = {}
    if funding_df is not None and not funding_df.empty:
        rates = funding_df["funding_rate"].values
        med_rate = float(np.median(rates))
        min_real_ts = int(funding_df["timestamp"].min())
        first_bar_ts = int(ind_df["timestamp"].min())
        funding_map = dict(zip(funding_df["timestamp"].astype(int), funding_df["funding_rate"].astype(float)))

        # Prepend synthetic 8h settlements (00:00, 08:00, 16:00 UTC = every 28,800,000 ms)
        synth_count = 0
        cur_synth_ts = (first_bar_ts // 28800000 + 1) * 28800000
        while cur_synth_ts < min_real_ts:
            funding_map[cur_synth_ts] = med_rate
            synth_count += 1
            cur_synth_ts += 28800000
        print(f"[FUNDING-FILL] {symbol}: prepended {synth_count:,} synthetic 8h funding settlements @ {med_rate:.6e} median rate")

    trades = []
    n_bars = len(ind_df)
    active_trade = None
    pending_order = None
    last_loss_bar = -999

    # FIX 1: instantiate seeded positions (single seed per replay run)
    seed_meta = None
    if seeded_positions:
        if len(seeded_positions) != 1:
            raise ValueError("run_grid_replay supports exactly one seeded position per run")
        seed = seeded_positions[0]
        ts_index = ind_df["timestamp"].astype("int64")
        entry_bar_arr = np.where(ts_index.values == int(seed["entry_ts"]))[0]
        if len(entry_bar_arr) == 0:
            seed_meta = {**seed, "outcome": "NO_BAR", "net_pnl": None}
        else:
            e_idx = int(entry_bar_arr[0])
            if e_idx >= n_bars - 1:
                seed_meta = {**seed, "outcome": "EDGE_OF_DATA", "net_pnl": None}
            else:
                close_at_entry = float(ind_df["close"].iloc[e_idx])
                notional = trade_allocation_usd * leverage
                active_trade = {
                    "symbol": seed.get("symbol", symbol),
                    "side": seed["side"],
                    "entry_bar": e_idx,
                    "entry_ts": int(seed["entry_ts"]),
                    "entry_price": float(seed["entry_price"]),
                    "take_profit": float(seed["take_profit"]),
                    "stop_loss": float(seed["stop_loss"]),
                    "regime": seed.get("regime", "SEEDED"),
                    "label": seed.get("label", ""),
                    "allocation_usd": trade_allocation_usd,
                    "leverage": leverage,
                    "entry_fee_usd": notional * TAKER_FEE,
                    "funding_usd": 0.0,
                    "is_maker_entry": False,
                }
                seed_meta = {**seed, "outcome": "OPEN", "net_pnl": None, "signal_close_at_entry": close_at_entry}

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
                if (prev_ts // 28800000) != (curr_ts // 28800000):
                    boundary_ts = int((curr_ts // 28800000) * 28800000)
                    f_rate = funding_map.get(boundary_ts, 0.0)
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
                    if net_pnl < 0:
                        last_loss_bar = i
                    active_trade = None
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
                    if net_pnl < 0:
                        last_loss_bar = i
                    active_trade = None
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
                if effective_fill_model:
                    tp_hit = evaluate_limit_fill(
                        side="SHORT",  # Exiting a long by selling at limit tp_p
                        limit_price=tp_p,
                        curr_open=curr_open,
                        curr_high=curr_high,
                        curr_low=curr_low,
                        tick_size=tick_size,
                        fill_model=effective_fill_model,
                        seed=i,
                    )
                else:
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
                    if net_pnl < 0:
                        last_loss_bar = i
                    active_trade = None
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
                    last_loss_bar = i
                    active_trade = None
                    continue

            elif side == "SHORT":
                # TP Check (Maker limit fill)
                if effective_fill_model:
                    tp_hit = evaluate_limit_fill(
                        side="LONG",  # Exiting a short by buying to cover at limit tp_p
                        limit_price=tp_p,
                        curr_open=curr_open,
                        curr_high=curr_high,
                        curr_low=curr_low,
                        tick_size=tick_size,
                        fill_model=effective_fill_model,
                        seed=i,
                    )
                else:
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
                    if net_pnl < 0:
                        last_loss_bar = i
                    active_trade = None
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
                    last_loss_bar = i
                    active_trade = None
                    continue

        # -------------------------------------------------------------
        # 2. Check Pending Post-Only Maker Limit Fill
        # -------------------------------------------------------------
        if pending_order is not None:
            p_side = pending_order["side"]
            limit_p = pending_order["price"]
            bars_pending = i - pending_order["placed_bar"]

            is_filled = evaluate_limit_fill(
                side=p_side,
                limit_price=limit_p,
                curr_open=curr_open,
                curr_high=curr_high,
                curr_low=curr_low,
                tick_size=tick_size,
                fill_model=effective_fill_model,
                seed=i,
            )

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
        # 3. Scan New Entry Signals (if flat, no pending order, loss cooldown passed)
        # -------------------------------------------------------------
        if scan_signals and active_trade is None and pending_order is None and (i - last_loss_bar) >= cooldown_bars:
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

            # D3: Funding Regime Gate ablation arm: suppress longs if trailing funding is excessively elevated
            if enable_funding_regime_gate and funding_map:
                cur_boundary_ts = int((curr_ts // 28800000) * 28800000)
                cur_f_rate = funding_map.get(cur_boundary_ts, 0.0)
                if cur_f_rate > 0.0001:  # > 1 bp per 8h
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

    # FIX 1: seeded-position lifecycle accounting
    if seeded_positions:
        if active_trade is not None:
            # Still open at edge of data: mark out at last close (taker), flagged unresolved
            exit_p = float(closes[-1])
            notional = trade_allocation_usd * leverage
            fee_usd = notional * TAKER_FEE
            if active_trade["side"] == "LONG":
                gross_pnl = (exit_p - active_trade["entry_price"]) / active_trade["entry_price"] * notional
            else:
                gross_pnl = (active_trade["entry_price"] - exit_p) / active_trade["entry_price"] * notional
            net_pnl = gross_pnl - active_trade["entry_fee_usd"] - fee_usd + active_trade["funding_usd"]
            trades.append({
                **active_trade,
                "exit_bar": n_bars - 1,
                "exit_ts": int(ts[-1]),
                "exit_price": exit_p,
                "exit_reason": "OPEN_AT_EDGE_MARK_OUT",
                "holding_mins": (n_bars - 1 - active_trade["entry_bar"]) * 5,
                "exit_fee_usd": fee_usd,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "is_maker_exit": False,
                "unresolved": True,
            })
        if seed_meta is not None and seed_meta["outcome"] in ("NO_BAR", "EDGE_OF_DATA"):
            trades.append({
                "symbol": seed_meta.get("symbol", symbol),
                "side": seed_meta["side"],
                "label": seed_meta.get("label", ""),
                "entry_ts": int(seed_meta["entry_ts"]),
                "entry_price": float(seed_meta["entry_price"]),
                "take_profit": float(seed_meta["take_profit"]),
                "stop_loss": float(seed_meta["stop_loss"]),
                "exit_ts": int(seed_meta["entry_ts"]),
                "exit_price": float(seed_meta["entry_price"]),
                "exit_reason": seed_meta["outcome"],
                "gross_pnl": 0.0,
                "net_pnl": 0.0,
                "funding_usd": 0.0,
                "entry_fee_usd": 0.0,
                "exit_fee_usd": 0.0,
                "holding_mins": 0,
                "unresolved": True,
            })

    return trades


def run_fleet_grid_replay(
    dfs: Dict[str, pd.DataFrame],
    funding_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    btc_15m_df: Optional[pd.DataFrame] = None,
    conservative_fills: bool = True,
    fill_model: Optional[str] = None,  # 'touch', 'one_tick', 'logprob', 'squareprob'
    enable_regime_gate: bool = True,
    enable_vwap_band: bool = True,
    enable_45m_tightening: bool = True,
    enable_short_scalps: bool = True,
    trade_allocation_usd: float = 85.0,
    leverage: float = 3.0,
    max_order_ttl_bars: int = 1,
    cooldown_bars: int = 12,
    max_long_cap: int = 2,  # FIX 7: Live-faithful fleet cap (super_bot.py max 2 concurrent grid LONGs across fleet)
    market_adx_ceiling: Optional[float] = None,  # Pre-registered arm: e.g. 20.0 (strict chop filter)
    enable_funding_regime_gate: bool = False,
) -> Dict[str, Any]:
    """
    Executes historical 5m multi-asset fleet simulation respecting fleet-wide position caps.

    Tracks:
    - Fleet-wide concurrent open and pending LONG positions (< max_long_cap, default 2).
    - Signal-to-fill % (filled limit orders / total placed limit orders).
    - TP exit share (TP maker exits / total completed trades).
    """
    if funding_dfs is None:
        funding_dfs = {}

    effective_fill_model = fill_model if fill_model is not None else ("one_tick" if conservative_fills else "touch")

    # Precompute BTC 15m regime
    regime_15m = None
    if btc_15m_df is not None and not btc_15m_df.empty:
        regime_15m = precompute_btc_15m_regime(btc_15m_df)

    # Precompute indicators and align timestamps across all symbols
    symbol_data = {}
    all_timestamps = set()

    for sym, df in dfs.items():
        tick_size = TICK_SIZES.get(sym, TICK_SIZES["DEFAULT"])
        ind_df = precompute_5m_indicators_vectorized(df)
        if regime_15m is not None:
            ind_df = pd.merge_asof(
                ind_df.sort_values("timestamp"),
                regime_15m,
                left_on="timestamp",
                right_on="close_ts",
                direction="backward"
            )
            missing_count = int(ind_df["market_adx"].isna().sum())
            ind_df["market_adx"] = ind_df["market_adx"].fillna(99.0)
            ind_df["market_di_spread"] = ind_df["market_di_spread"].fillna(0.0)
            if missing_count > 0:
                print(f"[{sym}] Blocked {missing_count} unaligned bars at start of series with market_adx=99.0")
        else:
            ind_df["market_adx"] = 99.0
            ind_df["market_di_spread"] = 0.0

        # Funding rate map
        f_df = funding_dfs.get(sym)
        funding_map = {}
        if f_df is not None and not f_df.empty:
            rates = f_df["funding_rate"].values
            med_rate = float(np.median(rates))
            min_real_ts = int(f_df["timestamp"].min())
            first_bar_ts = int(ind_df["timestamp"].min())
            funding_map = dict(zip(f_df["timestamp"].astype(int), f_df["funding_rate"].astype(float)))
            synth_count = 0
            cur_synth_ts = (first_bar_ts // 28800000 + 1) * 28800000
            while cur_synth_ts < min_real_ts:
                funding_map[cur_synth_ts] = med_rate
                synth_count += 1
                cur_synth_ts += 28800000
            print(f"[FUNDING-FILL] {sym}: prepended {synth_count:,} synthetic 8h funding settlements @ {med_rate:.6e} median rate")

        ind_df = ind_df.set_index("timestamp")
        symbol_data[sym] = {
            "ind_df": ind_df,
            "tick_size": tick_size,
            "funding_map": funding_map,
            "active_trade": None,
            "pending_order": None,
            "last_loss_idx": -999,
            "order_placed_count": 0,
            "order_filled_count": 0,
        }
        all_timestamps.update(ind_df.index)

    sorted_timestamps = sorted(list(all_timestamps))
    completed_trades: List[Dict[str, Any]] = []

    for t_idx, curr_ts in enumerate(sorted_timestamps):
        if t_idx < 30:
            continue

        # -------------------------------------------------------------
        # Step 1: Manage active positions for each symbol
        # -------------------------------------------------------------
        for sym, sdata in symbol_data.items():
            at = sdata["active_trade"]
            if at is None:
                continue

            ind_df = sdata["ind_df"]
            if curr_ts not in ind_df.index:
                continue
            row = ind_df.loc[curr_ts]

            curr_open = row["open"]
            curr_high = row["high"]
            curr_low = row["low"]
            curr_close = row["close"]
            tick_sz = sdata["tick_size"]
            f_map = sdata["funding_map"]

            entry_p = at["entry_price"]
            side = at["side"]
            tp_p = at["take_profit"]
            sl_p = at["stop_loss"]
            bars_held = t_idx - at["entry_idx"]
            mins_held = bars_held * 5

            # Accrue funding if 8h boundary crossed
            if f_map:
                prev_ts = sorted_timestamps[t_idx - 1]
                if (prev_ts // 28800000) != (curr_ts // 28800000):
                    boundary_ts = int((curr_ts // 28800000) * 28800000)
                    f_rate = f_map.get(boundary_ts, 0.0)
                    notional = trade_allocation_usd * leverage
                    f_usd = -f_rate * notional if side == "LONG" else f_rate * notional
                    at["funding_usd"] += f_usd

            # Check 45m - 90m dynamic tightening for shorts
            if enable_45m_tightening and side == "SHORT" and mins_held >= 45:
                short_pnl_pct = (entry_p - curr_close) / entry_p * 100.0
                if mins_held >= 90:
                    exit_p = curr_open
                    fee_usd = (trade_allocation_usd * leverage) * TAKER_FEE
                    gross_pnl = (entry_p - exit_p) / entry_p * (trade_allocation_usd * leverage)
                    net_pnl = gross_pnl - at["entry_fee_usd"] - fee_usd + at["funding_usd"]
                    completed_trades.append({
                        **at,
                        "exit_idx": t_idx,
                        "exit_ts": curr_ts,
                        "exit_price": exit_p,
                        "exit_reason": "SCALP_SHORT_TIME_EXPIRY",
                        "holding_mins": mins_held,
                        "exit_fee_usd": fee_usd,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "is_maker_exit": False,
                    })
                    if net_pnl < 0:
                        sdata["last_loss_idx"] = t_idx
                    sdata["active_trade"] = None
                    continue
                elif short_pnl_pct >= 0.45:
                    exit_p = curr_open
                    fee_usd = (trade_allocation_usd * leverage) * TAKER_FEE
                    gross_pnl = (entry_p - exit_p) / entry_p * (trade_allocation_usd * leverage)
                    net_pnl = gross_pnl - at["entry_fee_usd"] - fee_usd + at["funding_usd"]
                    completed_trades.append({
                        **at,
                        "exit_idx": t_idx,
                        "exit_ts": curr_ts,
                        "exit_price": exit_p,
                        "exit_reason": "SCALP_SHORT_TIME_PROFIT",
                        "holding_mins": mins_held,
                        "exit_fee_usd": fee_usd,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "is_maker_exit": False,
                    })
                    if net_pnl < 0:
                        sdata["last_loss_idx"] = t_idx
                    sdata["active_trade"] = None
                    continue
                else:
                    target_sl = entry_p * (1.0 - 0.0018) if short_pnl_pct >= 0.22 else entry_p * 1.0055
                    if target_sl < at["stop_loss"]:
                        at["stop_loss"] = target_sl
                        sl_p = target_sl

            # TP / SL checks
            if side == "LONG":
                if effective_fill_model:
                    tp_hit = evaluate_limit_fill(
                        side="SHORT",
                        limit_price=tp_p,
                        curr_open=curr_open,
                        curr_high=curr_high,
                        curr_low=curr_low,
                        tick_size=tick_sz,
                        fill_model=effective_fill_model,
                        seed=t_idx,
                    )
                else:
                    tp_hit = (curr_high >= tp_p + tick_sz) if conservative_fills else (curr_high >= tp_p)
                sl_hit = (curr_low <= sl_p)

                if tp_hit and not sl_hit:
                    exit_p = tp_p
                    fee_usd = (trade_allocation_usd * leverage) * MAKER_FEE
                    gross_pnl = (exit_p - entry_p) / entry_p * (trade_allocation_usd * leverage)
                    net_pnl = gross_pnl - at["entry_fee_usd"] - fee_usd + at["funding_usd"]
                    completed_trades.append({
                        **at,
                        "exit_idx": t_idx,
                        "exit_ts": curr_ts,
                        "exit_price": exit_p,
                        "exit_reason": "TAKE_PROFIT",
                        "holding_mins": mins_held,
                        "exit_fee_usd": fee_usd,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "is_maker_exit": True,
                    })
                    if net_pnl < 0:
                        sdata["last_loss_idx"] = t_idx
                    sdata["active_trade"] = None
                    continue

                if sl_hit:
                    exit_p = min(sl_p, curr_open)
                    fee_usd = (trade_allocation_usd * leverage) * TAKER_FEE
                    gross_pnl = (exit_p - entry_p) / entry_p * (trade_allocation_usd * leverage)
                    net_pnl = gross_pnl - at["entry_fee_usd"] - fee_usd + at["funding_usd"]
                    completed_trades.append({
                        **at,
                        "exit_idx": t_idx,
                        "exit_ts": curr_ts,
                        "exit_price": exit_p,
                        "exit_reason": "STOP_LOSS",
                        "holding_mins": mins_held,
                        "exit_fee_usd": fee_usd,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "is_maker_exit": False,
                    })
                    sdata["last_loss_idx"] = t_idx
                    sdata["active_trade"] = None
                    continue

            elif side == "SHORT":
                if effective_fill_model:
                    tp_hit = evaluate_limit_fill(
                        side="LONG",
                        limit_price=tp_p,
                        curr_open=curr_open,
                        curr_high=curr_high,
                        curr_low=curr_low,
                        tick_size=tick_sz,
                        fill_model=effective_fill_model,
                        seed=t_idx,
                    )
                else:
                    tp_hit = (curr_low <= tp_p - tick_sz) if conservative_fills else (curr_low <= tp_p)
                sl_hit = (curr_high >= sl_p)

                if tp_hit and not sl_hit:
                    exit_p = tp_p
                    fee_usd = (trade_allocation_usd * leverage) * MAKER_FEE
                    gross_pnl = (entry_p - exit_p) / entry_p * (trade_allocation_usd * leverage)
                    net_pnl = gross_pnl - at["entry_fee_usd"] - fee_usd + at["funding_usd"]
                    completed_trades.append({
                        **at,
                        "exit_idx": t_idx,
                        "exit_ts": curr_ts,
                        "exit_price": exit_p,
                        "exit_reason": "TAKE_PROFIT",
                        "holding_mins": mins_held,
                        "exit_fee_usd": fee_usd,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "is_maker_exit": True,
                    })
                    if net_pnl < 0:
                        sdata["last_loss_idx"] = t_idx
                    sdata["active_trade"] = None
                    continue

                if sl_hit:
                    exit_p = max(sl_p, curr_open)
                    fee_usd = (trade_allocation_usd * leverage) * TAKER_FEE
                    gross_pnl = (entry_p - exit_p) / entry_p * (trade_allocation_usd * leverage)
                    net_pnl = gross_pnl - at["entry_fee_usd"] - fee_usd + at["funding_usd"]
                    completed_trades.append({
                        **at,
                        "exit_idx": t_idx,
                        "exit_ts": curr_ts,
                        "exit_price": exit_p,
                        "exit_reason": "STOP_LOSS",
                        "holding_mins": mins_held,
                        "exit_fee_usd": fee_usd,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "is_maker_exit": False,
                    })
                    sdata["last_loss_idx"] = t_idx
                    sdata["active_trade"] = None
                    continue

        # -------------------------------------------------------------
        # Step 2: Check pending limit order fills for each symbol
        # -------------------------------------------------------------
        for sym, sdata in symbol_data.items():
            po = sdata["pending_order"]
            if po is None:
                continue

            ind_df = sdata["ind_df"]
            if curr_ts not in ind_df.index:
                continue
            row = ind_df.loc[curr_ts]

            curr_open = row["open"]
            curr_high = row["high"]
            curr_low = row["low"]
            tick_sz = sdata["tick_size"]

            p_side = po["side"]
            limit_p = po["price"]
            bars_pending = t_idx - po["placed_idx"]

            is_filled = evaluate_limit_fill(
                side=p_side,
                limit_price=limit_p,
                curr_open=curr_open,
                curr_high=curr_high,
                curr_low=curr_low,
                tick_size=tick_sz,
                fill_model=effective_fill_model,
                seed=t_idx,
            )

            if is_filled:
                sdata["order_filled_count"] += 1
                entry_fee_usd = (trade_allocation_usd * leverage) * MAKER_FEE
                sdata["active_trade"] = {
                    "symbol": sym,
                    "side": p_side,
                    "entry_idx": t_idx,
                    "entry_ts": curr_ts,
                    "entry_price": limit_p,
                    "take_profit": po["take_profit"],
                    "stop_loss": po["stop_loss"],
                    "regime": po["regime"],
                    "entry_market_adx": po["market_adx"],
                    "allocation_usd": trade_allocation_usd,
                    "leverage": leverage,
                    "entry_fee_usd": entry_fee_usd,
                    "funding_usd": 0.0,
                    "is_maker_entry": True,
                }
                sdata["pending_order"] = None
            elif bars_pending >= max_order_ttl_bars:
                sdata["pending_order"] = None

        # -------------------------------------------------------------
        # Step 3: Scan new entry signals (Fleet-Cap Gated)
        # -------------------------------------------------------------
        # Count open and pending LONGs across the entire fleet
        open_long_count = sum(1 for s in symbol_data.values() if s["active_trade"] is not None and s["active_trade"]["side"] == "LONG")
        pending_long_count = sum(1 for s in symbol_data.values() if s["pending_order"] is not None and s["pending_order"]["side"] == "LONG")

        for sym, sdata in symbol_data.items():
            if sdata["active_trade"] is not None or sdata["pending_order"] is not None:
                continue
            if (t_idx - sdata["last_loss_idx"]) < cooldown_bars:
                continue

            ind_df = sdata["ind_df"]
            if curr_ts not in ind_df.index:
                continue
            row = ind_df.loc[curr_ts]

            bar_rsi = row["rsi"]
            bar_lower = row["lower_band"] if enable_vwap_band else 1e9
            bar_upper = row["upper_band"] if enable_vwap_band else 0.0
            bar_adx = row["adx_5m"]
            curr_close = row["close"]
            m_adx = row["market_adx"]
            m_di = row["market_di_spread"]
            has_b_rev = row["has_bearish_reversal"]

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
            if market_adx_ceiling is not None and m_adx >= market_adx_ceiling:
                long_regime_ok = False

            if enable_funding_regime_gate and sdata["funding_map"]:
                cur_boundary_ts = int((curr_ts // 28800000) * 28800000)
                cur_f_rate = sdata["funding_map"].get(cur_boundary_ts, 0.0)
                if cur_f_rate > 0.0001:
                    long_regime_ok = False

            # Check Fleet-Cap constraint for LONGs
            fleet_long_allowed = (open_long_count + pending_long_count) < max_long_cap

            if fleet_long_allowed and long_regime_ok and bar_rsi < 30.0 and curr_close <= bar_lower and bar_adx < 24.0:
                limit_p = curr_close
                sdata["order_placed_count"] += 1
                sdata["pending_order"] = {
                    "side": "LONG",
                    "placed_idx": t_idx,
                    "placed_ts": curr_ts,
                    "price": limit_p,
                    "take_profit": limit_p * (1.0 + 0.0135),
                    "stop_loss": limit_p * (1.0 - 0.0115),
                    "regime": current_regime,
                    "market_adx": m_adx,
                }
                pending_long_count += 1
                continue

            # SHORT SCALP SIGNAL
            if enable_short_scalps:
                short_regime_ok = True
                if enable_regime_gate and bar_adx >= 22.0:
                    short_regime_ok = False

                if short_regime_ok and bar_rsi >= 68.0 and curr_close >= (bar_upper * 0.998) and has_b_rev:
                    limit_p = curr_close
                    sdata["order_placed_count"] += 1
                    sdata["pending_order"] = {
                        "side": "SHORT",
                        "placed_idx": t_idx,
                        "placed_ts": curr_ts,
                        "price": limit_p,
                        "take_profit": limit_p * (1.0 - 0.0120),
                        "stop_loss": limit_p * (1.0 + 0.0100),
                        "regime": current_regime,
                        "market_adx": m_adx,
                    }
                    continue

    # Compute execution telemetry
    total_placed = sum(s["order_placed_count"] for s in symbol_data.values())
    total_filled = sum(s["order_filled_count"] for s in symbol_data.values())
    signal_to_fill_pct = (total_filled / total_placed * 100.0) if total_placed > 0 else 0.0

    per_symbol_placed = {s: d["order_placed_count"] for s, d in symbol_data.items()}
    per_symbol_filled = {s: d["order_filled_count"] for s, d in symbol_data.items()}

    return {
        "trades": completed_trades,
        "total_placed": total_placed,
        "total_filled": total_filled,
        "signal_to_fill_pct": signal_to_fill_pct,
        "per_symbol_placed": per_symbol_placed,
        "per_symbol_filled": per_symbol_filled,
    }

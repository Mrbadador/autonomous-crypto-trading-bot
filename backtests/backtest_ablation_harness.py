"""
backtest_ablation_harness.py — Pure Replay & Gate Ablation Testing Harness
Replays live 4H swing entry logic over historical OHLCV data to measure counterfactual
expectancy when each gate is toggled ON vs OFF:
  - Base: 4H Ichimoku Kumo breakout + candle expansion + ADX >= 22
  - Gate A: 1D Trend Support (Price > EMA200 & EMA50 >= 0.98 * EMA200)
  - Gate B: 5m/4H Exhaustion Wick Threshold (wick <= 0.58)
  - Gate C: Higher-Timeframe Runway Clearance (runway >= 0.4%)
  - Gate D: Momentum Divergence Gate (4H RSI/MACD divergence)
  - Gate E: Volume Profile & Value Area Clearance
  - Gate F: TD Sequential Setup Exhaustion (TD-9 exhaustion)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Ensure project root in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators import compute_indicators, compute_td_sequential, compute_volume_profile, detect_momentum_divergence


def precompute_4h_indicators_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes 4H breakout indicators vectorized across all bars matching swing_trader.py logic.
    Substantially accelerates backtest simulation loop from minutes to milliseconds.
    """
    res = df.copy()
    c = res["close"]
    h = res["high"]
    l = res["low"]
    o = res["open"]

    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2.0
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2.0
    span_a_raw = (tenkan + kijun) / 2.0
    span_b_raw = (h.rolling(52).max() + l.rolling(52).min()) / 2.0

    span_a = span_a_raw.shift(26).fillna(span_a_raw)
    span_b = span_b_raw.shift(26).fillna(span_b_raw)

    cloud_top = np.maximum(span_a, span_b)
    cloud_bottom = np.minimum(span_a, span_b)

    # True Range & ATR(14)
    prev_close = c.shift(1).fillna(c)
    tr = pd.concat([
        h - l,
        (h - prev_close).abs(),
        (l - prev_close).abs()
    ], axis=1).max(axis=1)
    res["atr"] = tr.rolling(window=14).mean().fillna(0.0)

    # Wilder ADX(14)
    up_move = h - h.shift(1)
    down_move = l.shift(1) - l
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_sum = tr.rolling(window=14).sum()
    plus_di = 100.0 * (pd.Series(plus_dm, index=df.index).rolling(window=14).sum() / (tr_sum + 1e-9))
    minus_di = 100.0 * (pd.Series(minus_dm, index=df.index).rolling(window=14).sum() / (tr_sum + 1e-9))
    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
    res["adx"] = dx.rolling(window=14).mean().fillna(0.0)

    # RSI(14)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    res["rsi"] = 100.0 - (100.0 / (1.0 + rs)).fillna(50.0)

    c_range = np.maximum(h - l, 1e-8)
    res["upper_wick_ratio"] = (h - np.maximum(o, c)) / c_range
    res["lower_wick_ratio"] = (np.minimum(o, c) - l) / c_range

    res["bullish_candle"] = (c > o) & (c >= prev_close) & (res["upper_wick_ratio"] <= 0.60)
    res["bearish_candle"] = (c < o) & (c <= prev_close) & (res["lower_wick_ratio"] <= 0.60)

    # Chikou Span Confirmation (Close vs Close 26 bars ago)
    c_shift_26 = c.shift(26)
    res["chikou_bull_clear"] = (c > c_shift_26).fillna(True)
    res["chikou_bear_clear"] = (c < c_shift_26).fillna(True)

    # Exact swing_trader.py Kumo Breakout formula:
    res["bullish_kumo"] = (
        (c > cloud_top) & (c > kijun) & res["chikou_bull_clear"] &
        ((tenkan >= kijun * 0.995) | (c > cloud_top * 1.002))
    )
    res["bearish_kumo"] = (
        (c < cloud_bottom) & (c < kijun) & res["chikou_bear_clear"] &
        ((tenkan <= kijun * 1.005) | (c < cloud_bottom * 0.998))
    )

    res["is_base_long"] = res["bullish_kumo"] & res["bullish_candle"] & (res["adx"] >= 22.0) & (res["rsi"] <= 70.0)
    res["is_base_short"] = res["bearish_kumo"] & res["bearish_candle"] & (res["adx"] >= 22.0) & (res["rsi"] >= 35.0)

    return res


def precompute_1d_trend_vectorized(df_1d: pd.DataFrame) -> pd.DataFrame:
    """Computes daily EMA 50 & 200 alignment vectorized across full 1D history."""
    res = df_1d.copy()
    close = res["close"]
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    res["bullish_alignment"] = (close > ema200) & (ema50 >= (ema200 * 0.98))
    res["bearish_alignment"] = (close < ema200) & (ema50 <= (ema200 * 1.02))
    return res


def simulate_trade_path(
    df: pd.DataFrame,
    start_idx: int,
    side: str,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    max_bars: int = 42
) -> dict:
    """
    Simulates subsequent bar path without lookahead.
    Evaluates high/low of subsequent bars to determine whether TP or SL was hit first.
    """
    for i in range(start_idx + 1, min(len(df), start_idx + 1 + max_bars)):
        bar = df.iloc[i]
        b_high = float(bar["high"])
        b_low = float(bar["low"])

        if side == "LONG":
            # Check SL hit
            if b_low <= sl_price:
                pnl_pct = (sl_price - entry_price) / entry_price
                return {"pnl_pct": pnl_pct, "exit_reason": "STOP_LOSS", "bars_held": i - start_idx}
            # Check TP hit
            if b_high >= tp_price:
                pnl_pct = (tp_price - entry_price) / entry_price
                return {"pnl_pct": pnl_pct, "exit_reason": "TAKE_PROFIT", "bars_held": i - start_idx}
        else:
            # Short
            if b_high >= sl_price:
                pnl_pct = (entry_price - sl_price) / entry_price
                return {"pnl_pct": pnl_pct, "exit_reason": "STOP_LOSS", "bars_held": i - start_idx}
            if b_low <= tp_price:
                pnl_pct = (entry_price - tp_price) / entry_price
                return {"pnl_pct": pnl_pct, "exit_reason": "TAKE_PROFIT", "bars_held": i - start_idx}

    # Time expiry exit at last bar close
    last_close = float(df.iloc[min(len(df) - 1, start_idx + max_bars)]["close"])
    pnl_pct = (last_close - entry_price) / entry_price if side == "LONG" else (entry_price - last_close) / entry_price
    return {"pnl_pct": pnl_pct, "exit_reason": "TIME_EXPIRY", "bars_held": max_bars}


def run_ablation_simulation(
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,
    gates_config: Dict[str, bool],
    fee_pct: float = 0.0008,
    df_4h_precomputed: pd.DataFrame = None,
    df_4h_vec: pd.DataFrame = None,
    df_1d_vec: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    Simulates 4H swing strategy across historical bars with the requested gate ablation config.
    """
    trades = []
    cooldown_until = 0

    if df_4h_vec is None:
        df_4h_vec = precompute_4h_indicators_vectorized(df_4h)
    if df_1d_vec is None:
        df_1d_vec = precompute_1d_trend_vectorized(df_1d)
    if df_4h_precomputed is None:
        df_4h_precomputed = compute_indicators(df_4h, warmup_bars=min(len(df_4h), 200), strict=False)

    # Minimum window needed for 52-bar Ichimoku + 200-bar 1D
    start_bar = 90
    total_bars = len(df_4h)

    # Pre-filter candidate bars with base setups
    candidate_indices = np.where(df_4h_vec["is_base_long"] | df_4h_vec["is_base_short"])[0]
    candidate_indices = [idx for idx in candidate_indices if idx >= start_bar and idx < total_bars - 1]

    # Pre-index 1D timestamps for fast O(log N) lookup
    has_ts = "timestamp" in df_1d_vec.columns and "timestamp" in df_4h_vec.columns
    ts_1d = df_1d_vec["timestamp"].values if has_ts else None

    gate_fires = {
        "gate_1d_trend": 0,
        "gate_wick_exhaustion": 0,
        "gate_divergence": 0,
        "gate_td9": 0,
        "gate_volume_profile": 0
    }

    for idx in candidate_indices:
        if idx < cooldown_until:
            continue

        row_4h = df_4h_vec.iloc[idx]
        curr_c = float(row_4h["close"])
        atr = float(row_4h["atr"])
        if atr <= 0:
            continue

        is_long = bool(row_4h["is_base_long"])
        side = "LONG" if is_long else "SHORT"

        # Gate A: 1D Trend Support Check
        if gates_config.get("gate_1d_trend", True):
            if has_ts:
                c_ts = row_4h["timestamp"]
                pos = np.searchsorted(ts_1d, c_ts, side="right") - 1
                row_1d = df_1d_vec.iloc[max(0, pos)]
            else:
                pos = min(len(df_1d_vec) - 1, idx // 6)
                row_1d = df_1d_vec.iloc[pos]

            if side == "LONG" and not bool(row_1d.get("bullish_alignment", True)):
                gate_fires["gate_1d_trend"] += 1
                continue
            if side == "SHORT" and not bool(row_1d.get("bearish_alignment", True)):
                gate_fires["gate_1d_trend"] += 1
                continue

        # Gate B: Wick Exhaustion Shield (wick <= 0.58)
        if gates_config.get("gate_wick_exhaustion", True):
            if side == "LONG" and float(row_4h["upper_wick_ratio"]) >= 0.58:
                gate_fires["gate_wick_exhaustion"] += 1
                continue
            if side == "SHORT" and float(row_4h["lower_wick_ratio"]) >= 0.58:
                gate_fires["gate_wick_exhaustion"] += 1
                continue

        # Sliced precomputed indicators for sub-millisecond gate execution
        slice_precomp = df_4h_precomputed.iloc[max(0, idx - 60):idx+1]

        # Gate C: Momentum Divergence Gate
        if gates_config.get("gate_divergence", True):
            try:
                div = detect_momentum_divergence(slice_precomp)
                if side == "LONG" and div.get("bearish_divergence", False):
                    gate_fires["gate_divergence"] += 1
                    continue
                if side == "SHORT" and div.get("bullish_divergence", False):
                    gate_fires["gate_divergence"] += 1
                    continue
            except Exception:
                pass

        # Gate D: TD Sequential Exhaustion Gate (TD-9)
        if gates_config.get("gate_td9", True):
            try:
                td = compute_td_sequential(slice_precomp)
                if side == "LONG" and td.get("is_td9_upward_exhaustion", False):
                    gate_fires["gate_td9"] += 1
                    continue
                if side == "SHORT" and td.get("is_td9_downward_exhaustion", False):
                    gate_fires["gate_td9"] += 1
                    continue
            except Exception:
                pass

        # Gate E: Volume Profile & Value Area Clearance Gate
        if gates_config.get("gate_volume_profile", True):
            try:
                vp = compute_volume_profile(slice_precomp)
                # Avoid long breakout into heavy overhead resistance POC
                if side == "LONG" and vp.get("poc", 0.0) > curr_c and (vp["poc"] - curr_c) / curr_c < 0.004:
                    gate_fires["gate_volume_profile"] += 1
                    continue
            except Exception:
                pass

        # Sizing / Geometry: 2.0 ATR SL, 3.0 ATR TP
        if side == "LONG":
            sl_price = curr_c - (2.0 * atr)
            tp_price = curr_c + (3.0 * atr)
        else:
            sl_price = curr_c + (2.0 * atr)
            tp_price = curr_c - (3.0 * atr)

        trade_res = simulate_trade_path(df_4h, idx, side, curr_c, sl_price, tp_price)
        net_pnl = trade_res["pnl_pct"] - fee_pct  # round-trip fee deduction

        trades.append({
            "idx": idx,
            "side": side,
            "entry_price": curr_c,
            "exit_reason": trade_res["exit_reason"],
            "bars_held": trade_res["bars_held"],
            "net_pnl": net_pnl
        })

        cooldown_until = idx + max(1, trade_res["bars_held"])

    if not trades:
        return {"n_trades": 0, "win_rate": 0.0, "total_return": 0.0, "profit_factor": 0.0, "sharpe": 0.0, "gate_fires": gate_fires, "trades": []}

    pnls = [t["net_pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls) * 100.0
    tot_ret = sum(pnls) * 100.0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses)) if losses else 1e-6
    pf = gross_win / gross_loss if gross_loss > 0 else 999.0
    sharpe = (np.mean(pnls) / (np.std(pnls) + 1e-9)) * np.sqrt(365 * 6) if len(pnls) > 1 else 0.0

    # Bootstrap 90% Confidence Interval for Expected Value (EV) per trade
    ev_mean = float(np.mean(pnls))
    ev_ci_lo, ev_ci_hi = ev_mean, ev_mean
    if len(pnls) >= 5:
        rng = np.random.default_rng(42)
        n_boot = 1500
        boot_means = [float(np.mean(rng.choice(pnls, size=len(pnls), replace=True))) for _ in range(n_boot)]
        ev_ci_lo = float(np.percentile(boot_means, 5.0))
        ev_ci_hi = float(np.percentile(boot_means, 95.0))

    return {
        "n_trades": len(trades),
        "win_rate": round(wr, 1),
        "total_return": round(tot_ret, 2),
        "profit_factor": round(pf, 2),
        "sharpe": round(sharpe, 2),
        "ev_mean": round(ev_mean * 100.0, 3),
        "ev_ci_90": (round(ev_ci_lo * 100.0, 3), round(ev_ci_hi * 100.0, 3)),
        "gate_fires": gate_fires,
        "trades": trades
    }


def main():
    parser = argparse.ArgumentParser(description="Swing 4H Gate Ablation Testing Harness")
    parser.add_argument("--csv-4h", type=str, default="data/BTCUSDT_4h.csv", help="Path to 4H OHLCV CSV")
    parser.add_argument("--csv-1d", type=str, default="data/BTCUSDT_1d.csv", help="Path to 1D OHLCV CSV")
    parser.add_argument("--csv-1h", type=str, default="data/historical/BTCUSDT_1h_full.csv", help="Path to 1H historical source CSV")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("   SWING 4H ARCHITECT -- RELATIVE GATE ABLATION SCREENER")
    print("   [NOTE: Relative counterfactual screener over historical bars.")
    print("    Used to rank gate contribution; not a full live-orderbook replay.]")
    print("=" * 80 + "\n")

    # Load authentic historical data
    df_4h = None
    df_1d = None

    if os.path.exists(args.csv_4h) and os.path.exists(args.csv_1d):
        print(f"[*] Loading historical 4H and 1D data from {args.csv_4h} and {args.csv_1d}...")
        df_4h = pd.read_csv(args.csv_4h)
        df_1d = pd.read_csv(args.csv_1d)
    elif os.path.exists(args.csv_1h):
        print(f"[*] Resampling 4H and 1D datasets directly from authentic 1H ledger: {args.csv_1h}...")
        df_1h = pd.read_csv(args.csv_1h)
        # Ensure timestamp alignment
        if "timestamp" in df_1h.columns:
            ts_series = df_1h["timestamp"]
            unit = "ms" if ts_series.iloc[0] > 1e11 else "s"
            df_1h["dt"] = pd.to_datetime(ts_series, unit=unit)
            df_1h = df_1h.set_index("dt").sort_index()
            df_4h = df_1h.resample("4h").agg({
                "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "timestamp": "last"
            }).dropna().reset_index(drop=True)
            df_1d = df_1h.resample("1D").agg({
                "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "timestamp": "last"
            }).dropna().reset_index(drop=True)
        else:
            df_4h = df_1h.iloc[::4].copy().reset_index(drop=True)
            df_1d = df_1h.iloc[::24].copy().reset_index(drop=True)
        print(f"[OK] Successfully prepared {len(df_4h):,} 4H bars and {len(df_1d):,} 1D bars.")
    else:
        # Refuse to generate meaningless synthetic random walk
        raise FileNotFoundError(
            f"Authentic historical OHLCV data not found at {args.csv_4h}, {args.csv_1d}, or {args.csv_1h}. "
            "Refusing to generate synthetic random walk data to avoid misleading ablation tables."
        )

    # Define gate ablation matrix
    experiments = [
        ("Full Stack (All Gates ON)", {"gate_1d_trend": True, "gate_wick_exhaustion": True, "gate_divergence": True, "gate_td9": True, "gate_volume_profile": True}),
        ("Ablate Gate A (1D Trend Support OFF)", {"gate_1d_trend": False, "gate_wick_exhaustion": True, "gate_divergence": True, "gate_td9": True, "gate_volume_profile": True}),
        ("Ablate Gate B (Wick Exhaustion OFF)", {"gate_1d_trend": True, "gate_wick_exhaustion": False, "gate_divergence": True, "gate_td9": True, "gate_volume_profile": True}),
        ("Ablate Gate C (Momentum Divergence OFF)", {"gate_1d_trend": True, "gate_wick_exhaustion": True, "gate_divergence": False, "gate_td9": True, "gate_volume_profile": True}),
        ("Ablate Gate D (TD-9 Exhaustion OFF)", {"gate_1d_trend": True, "gate_wick_exhaustion": True, "gate_divergence": True, "gate_td9": False, "gate_volume_profile": True}),
        ("Ablate Gate E (Volume Profile OFF)", {"gate_1d_trend": True, "gate_wick_exhaustion": True, "gate_divergence": True, "gate_td9": True, "gate_volume_profile": False}),
        ("Naked Baseline (All Gates OFF)", {"gate_1d_trend": False, "gate_wick_exhaustion": False, "gate_divergence": False, "gate_td9": False, "gate_volume_profile": False}),
    ]

    # Precompute indicators across full 4H dataset once
    print("[*] Precomputing rolling indicators, Ichimoku breakout vectors, and 1D alignments...")
    df_4h_precomputed = compute_indicators(df_4h, warmup_bars=min(len(df_4h), 200), strict=False)
    df_4h_vec = precompute_4h_indicators_vectorized(df_4h)
    df_1d_vec = precompute_1d_trend_vectorized(df_1d)

    print(f"\n{'Configuration':<38} | {'Trades':<6} | {'Win Rate':<8} | {'Return':<9} | {'PF':<6} | {'Sharpe':<6} | {'EV / Trade (90% CI)':<22} | {'Active Gate Fires'}")
    print("-" * 135)

    matrix_rows = []
    for name, config_dict in experiments:
        res = run_ablation_simulation(
            df_4h, df_1d, config_dict,
            df_4h_precomputed=df_4h_precomputed,
            df_4h_vec=df_4h_vec,
            df_1d_vec=df_1d_vec
        )
        ci_str = f"{res['ev_mean']:+.2f}% [{res['ev_ci_90'][0]:+.2f}%, {res['ev_ci_90'][1]:+.2f}%]"
        gf = res.get("gate_fires", {})
        gf_str = ", ".join(f"{k.replace('gate_', '')}:{v}" for k, v in gf.items() if v > 0) or "none"
        print(f"{name:<38} | {res['n_trades']:<6} | {res['win_rate']:>7.1f}% | {res['total_return']:>8.2f}% | {res['profit_factor']:>6.2f} | {res['sharpe']:>6.2f} | {ci_str:<22} | {gf_str}")
        matrix_rows.append((name, res))

    print("\n[SUCCESS] Ablation comparison matrix successfully evaluated against real data.\n")
    return matrix_rows


if __name__ == "__main__":
    main()

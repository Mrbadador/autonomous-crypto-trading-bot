"""
backtest_swing_architect.py — High-Fidelity Multi-Asset Quantitative Backtester for The Swing Architect (4H)
Replays historical 4H and 1D candles to empirically validate:
1. Live strategy rules (Ichimoku Kumo, 1D macro EMA, ADX, RSI, 3x/5x tiered leverage); the
   Ichimoku cloud is an A/B arm -- `--shifted` replays the live engine's 26-bar projected cloud,
   and the default arm is the unshifted comparison, which the live engine does not use.
2. Dynamic Trailing stops (Breakeven at +1.5R, ROE Profit Locks at 15%/30%, Kijun trailing).
3. Realistic Bitget VIP0 fee deduction (0.06% taker per side) + 2 bps slippage + funding drag.
4. Comparative variants to diagnose exactly what "needs work" in the live engine.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def measure_trade_ratios(df_trades):
    """Measure a run's profit factor and average win/loss ratio, or None where there is no denominator.

    Both ratios divide by something the run may not have produced: a losing trade. A run whose
    trades never lost has no denominator, so the ratio is None -- the absence of a measurement --
    and not a hardcoded constant dressed as one. The old code published 99.0 for the profit factor
    and 0.0 for the win/loss ratio, and the comparative audit matrix printed them as measured
    numbers (99.00, 0.00x) next to rows that had actually been measured.

    `df_trades` is the closed-trade frame and must carry `net_pnl` and `gross_pnl`. A ratio that
    was measured is returned as a number, including a measured 0.0 (a run that never won has a
    denominator and no profit in it); only the missing denominator is None.
    """
    if df_trades is None or len(df_trades) == 0:
        return None, None

    wins = df_trades[df_trades["net_pnl"] > 0]
    losses = df_trades[df_trades["net_pnl"] <= 0]

    gross_profit = wins["gross_pnl"].sum() if not wins.empty else 0.0
    gross_loss = abs(losses["gross_pnl"].sum()) if not losses.empty else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

    avg_win = wins["net_pnl"].mean() if not wins.empty else 0.0
    avg_loss = abs(losses["net_pnl"].mean()) if not losses.empty else 0.0
    win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else None

    return profit_factor, win_loss_ratio


def format_profit_factor(profit_factor, n_trades, loss_count=0):
    """Render a run's profit factor, printing absence where the ratio has no denominator.

    The cause is derived from the run's own counts, so the sentence a reader gets is the one that
    actually applied:
      - no closed trades at all -> nothing was measured;
      - closed trades and not one of them a loser -> the ratio has no denominator;
      - closed trades whose non-winners carry no gross loss -> the ratio has no denominator.

    Same shape as the ablation harness's renderer (`format_profit_factor` in
    backtests/backtest_ablation_harness.py), which this repo's reports already follow.
    """
    if profit_factor is not None:
        return f"{profit_factor:.2f}"
    if not n_trades:
        return "n/a (no closed trades, profit factor not measured)"
    if not loss_count:
        return "n/a (no losing trade, profit factor has no denominator)"
    return "n/a (no gross loss to divide by, profit factor has no denominator)"


def format_win_loss_ratio(win_loss_ratio, n_trades, loss_count=0):
    """Render a run's average win/loss ratio, printing absence where it has no denominator.

    A ratio of the average win to the average loss needs a losing trade to average: a run that
    never lost has none, and a run whose non-wins were all exactly breakeven has an average loss
    of zero. Both are absences, not measured zeros.
    """
    if win_loss_ratio is not None:
        return f"{win_loss_ratio:.2f}x"
    if not n_trades:
        return "n/a (no closed trades, win/loss ratio not measured)"
    if not loss_count:
        return "n/a (no losing trade, win/loss ratio has no denominator)"
    return "n/a (every non-win was a breakeven, win/loss ratio has no denominator)"


def compute_4h_indicators(df: pd.DataFrame, shifted_cloud: bool = False) -> pd.DataFrame:
    """Computes Tenkan, Kijun, Senkou Span A/B, ATR(14), ADX(14), and RSI(14)."""
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Tenkan-sen (9): (9-period high + 9-period low) / 2
    tenkan = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2.0

    # Kijun-sen (26): (26-period high + 26-period low) / 2
    kijun = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2.0

    # Senkou Span A: (Tenkan + Kijun) / 2
    span_a_raw = (tenkan + kijun) / 2.0
    # Senkou Span B: (52-period high + 52-period low) / 2
    span_b_raw = (high.rolling(window=52).max() + low.rolling(window=52).min()) / 2.0

    if shifted_cloud:
        # Classical Ichimoku: Cloud is projected 26 bars forward, so today's cloud was calculated 26 bars ago.
        # This is the treatment the live engine uses (swing_trader.py:642-643, quant_validation.py:89-90).
        span_a = span_a_raw.shift(26)
        span_b = span_b_raw.shift(26)
    else:
        # Unshifted comparison arm, kept for the A/B against the live treatment: the cloud is read on
        # the bar that computed it. The live engine does NOT use this arm -- it always shifts 26 bars
        # (swing_trader.py:642-643, quant_validation.py:89-90).
        span_a = span_a_raw
        span_b = span_b_raw

    # ATR(14)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14).mean()

    # ADX(14)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_sum = tr.rolling(window=14).sum()
    plus_di = 100.0 * (pd.Series(plus_dm, index=df.index).rolling(window=14).sum() / tr_sum)
    minus_di = 100.0 * (pd.Series(minus_dm, index=df.index).rolling(window=14).sum() / tr_sum)
    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_14 = dx.rolling(window=14).mean()

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_14 = 100.0 - (100.0 / (1.0 + rs))

    df["tenkan"] = tenkan
    df["kijun"] = kijun
    df["span_a"] = span_a
    df["span_b"] = span_b
    df["cloud_top"] = np.maximum(span_a, span_b)
    df["cloud_bottom"] = np.minimum(span_a, span_b)
    df["atr"] = atr_14
    df["adx"] = adx_14.fillna(0.0)
    df["rsi"] = rsi_14.fillna(50.0)

    # Bullish / Bearish Kumo conditions
    df["bullish_kumo"] = (close > df["cloud_top"]) & (close > kijun) & (tenkan >= kijun)
    df["bearish_kumo"] = (close < df["cloud_bottom"]) & (close < kijun) & (tenkan <= kijun)

    # Fresh breakout: true only if current bar is bullish and previous bar was NOT bullish
    df["fresh_bullish_breakout"] = df["bullish_kumo"] & (~df["bullish_kumo"].shift(1).fillna(False))
    df["fresh_bearish_breakdown"] = df["bearish_kumo"] & (~df["bearish_kumo"].shift(1).fillna(False))

    return df


def compute_1d_indicators(df_1d: pd.DataFrame) -> pd.DataFrame:
    """Computes daily 50 and 200 EMAs."""
    df_1d = df_1d.copy().sort_values("timestamp").reset_index(drop=True)
    df_1d["ema50"] = df_1d["close"].ewm(span=50, adjust=False).mean()
    df_1d["ema200"] = df_1d["close"].ewm(span=200, adjust=False).mean()
    df_1d["bullish_alignment"] = (df_1d["close"] > df_1d["ema200"]) & (df_1d["ema50"] >= (df_1d["ema200"] * 0.98))
    df_1d["bearish_alignment"] = (df_1d["close"] < df_1d["ema200"]) & (df_1d["ema50"] <= (df_1d["ema200"] * 1.02))
    return df_1d


class SwingArchitectBacktester:
    def __init__(
        self,
        symbols: list = None,
        initial_balance: float = 137.0,
        allocation_usd: float = 20.0,
        max_locks: int = 6,
        atr_sl_mult: float = 2.5,
        runner_tp_mult: float = 4.0,
        be_threshold_r: float = 1.5,
        min_adx: float = 25.0,
        high_quality_adx: float = 30.0,
        standard_leverage: float = 3.0,
        high_quality_leverage: float = 5.0,
        use_shifted_cloud: bool = False,  # False = unshifted comparison arm; True = live engine's projected cloud
        require_fresh_breakout: bool = False,
        allow_shorts: bool = True,
        t1_roe: float = 15.0,
        t1_ratio: float = 0.50,
        t2_roe: float = 30.0,
        t2_ratio: float = 0.70,
        fee_rate: float = 0.0006,  # 0.06% Bitget VIP0 taker fee
        slippage_rate: float = 0.0002,  # 2 bps slippage
        funding_rate_8h: float = 0.0001,  # 0.01% per 8h
        cooldown_bars: int = 3,  # 12h cooldown
        min_sl_pct: float = 0.015  # 1.5% min SL floor
    ):
        self.symbols = symbols or [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'LINK/USDT', 
            'AVAX/USDT', 'XRP/USDT', 'ADA/USDT', 'BNB/USDT', 'NEAR/USDT'
        ]
        self.initial_balance = initial_balance
        self.allocation_usd = allocation_usd
        self.max_locks = max_locks
        self.atr_sl_mult = atr_sl_mult
        self.runner_tp_mult = runner_tp_mult
        self.be_threshold_r = be_threshold_r
        self.min_adx = min_adx
        self.high_quality_adx = high_quality_adx
        self.standard_leverage = standard_leverage
        self.high_quality_leverage = high_quality_leverage
        self.use_shifted_cloud = use_shifted_cloud
        self.require_fresh_breakout = require_fresh_breakout
        self.allow_shorts = allow_shorts
        self.t1_roe = t1_roe
        self.t1_ratio = t1_ratio
        self.t2_roe = t2_roe
        self.t2_ratio = t2_ratio
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.funding_rate_8h = funding_rate_8h
        self.cooldown_bars = cooldown_bars
        self.min_sl_pct = min_sl_pct

    def load_data(self) -> dict:
        """Loads and precomputes 4H and 1D data for all symbols."""
        data_bundle = {}
        for sym in self.symbols:
            clean = sym.replace('/', '')
            f_4h = f"data/historical/{clean}_4h_full.csv"
            f_1d = f"data/historical/{clean}_1d_full.csv"
            if not os.path.exists(f_4h):
                f_4h = f"data/historical/{clean}_4h.csv"
            if not os.path.exists(f_1d):
                f_1d = f"data/historical/{clean}_1d.csv"

            if not os.path.exists(f_4h) or not os.path.exists(f_1d):
                print(f"[WARN] Data missing for {sym}, skipping.")
                continue

            df_4h = pd.read_csv(f_4h)
            df_1d = pd.read_csv(f_1d)

            df_4h["dt"] = pd.to_datetime(df_4h["timestamp"], unit="ms", utc=True)
            df_1d["dt"] = pd.to_datetime(df_1d["timestamp"], unit="ms", utc=True)

            df_4h = compute_4h_indicators(df_4h, shifted_cloud=self.use_shifted_cloud)
            df_1d = compute_1d_indicators(df_1d)

            # Map daily alignment to 4H candles using merge_asof (strictly no lookahead)
            df_4h = pd.merge_asof(
                df_4h.sort_values("dt"),
                df_1d[["dt", "bullish_alignment", "bearish_alignment", "ema50", "ema200"]].sort_values("dt"),
                on="dt",
                direction="backward"
            )

            data_bundle[sym] = df_4h
        return data_bundle

    def run(self) -> dict:
        """Simulates fleet portfolio execution on unified timeline."""
        data_bundle = self.load_data()
        if not data_bundle:
            return {"error": "No historical data found"}

        # Build unified timestamps across all symbols
        all_dts = sorted(list(set().union(*[df["dt"].tolist() for df in data_bundle.values()])))

        balance = self.initial_balance
        peak_balance = balance
        max_drawdown = 0.0

        active_positions = {}  # sym -> dict
        cooldowns = {}  # sym -> bar_index_available
        closed_trades = []
        equity_curve = []

        total_bars = len(all_dts)
        start_idx = 60  # warm-up period for 52-period cloud

        for bar_idx in range(start_idx, total_bars):
            curr_dt = all_dts[bar_idx]

            # 1. Manage Active Positions
            for sym in list(active_positions.keys()):
                pos = active_positions[sym]
                df_sym = data_bundle[sym]
                row_matches = df_sym[df_sym["dt"] == curr_dt]
                if row_matches.empty:
                    continue
                row = row_matches.iloc[0]

                high_p = float(row["high"])
                low_p = float(row["low"])
                close_p = float(row["close"])
                is_long = (pos["side"] == "LONG")
                entry_p = pos["entry_price"]
                lev = pos["leverage"]

                # Check Exits FIRST (using stops active coming into this bar)
                exit_price = None
                exit_reason = None

                # Check Take Profit
                if is_long and high_p >= pos["tp"]:
                    exit_price = pos["tp"]
                    exit_reason = "TAKE_PROFIT"
                elif not is_long and low_p <= pos["tp"]:
                    exit_price = pos["tp"]
                    exit_reason = "TAKE_PROFIT"
                # Check Stop Loss
                elif is_long and low_p <= pos["sl"]:
                    exit_price = min(pos["sl"], float(row["open"]))  # Account for gap downs
                    exit_reason = "STOP_LOSS"
                elif not is_long and high_p >= pos["sl"]:
                    exit_price = max(pos["sl"], float(row["open"]))  # Account for gap ups
                    exit_reason = "STOP_LOSS"

                if exit_price is not None:
                    # Apply slippage on exit
                    actual_exit = exit_price * (1.0 - self.slippage_rate) if is_long else exit_price * (1.0 + self.slippage_rate)
                    gross_pnl = (actual_exit - entry_p) * pos["qty"] if is_long else (entry_p - actual_exit) * pos["qty"]
                    exit_fee = pos["notional"] * self.fee_rate
                    pos["cumulative_fees"] += exit_fee
                    net_pnl = gross_pnl - exit_fee
                    balance += net_pnl

                    holding_hours = pos["holding_bars"] * 4.0
                    roe_pct = (net_pnl / pos["margin_used"]) * 100.0 if pos["margin_used"] > 0 else 0.0

                    closed_trades.append({
                        "symbol": sym,
                        "side": pos["side"],
                        "entry_time": pos["entry_time"],
                        "exit_time": curr_dt,
                        "entry_price": entry_p,
                        "exit_price": actual_exit,
                        "gross_pnl": gross_pnl,
                        "total_fees": pos["cumulative_fees"],
                        "net_pnl": net_pnl,
                        "roe_pct": roe_pct,
                        "holding_hours": holding_hours,
                        "exit_reason": exit_reason,
                        "quality_tier": pos["quality_tier"],
                        "leverage": lev,
                        "notes": pos["notes"]
                    })

                    # Activate anti-whipsaw cooldown
                    cooldowns[sym] = bar_idx + self.cooldown_bars
                    del active_positions[sym]
                    continue

                # If position survived this bar, update peak metrics and trailing ratchets for subsequent bars
                # 8h Funding Rate drag (accrued every 2 bars of 4H = 8 hours)
                pos["holding_bars"] += 1
                if pos["holding_bars"] % 2 == 0:
                    funding_cost = pos["notional"] * self.funding_rate_8h
                    balance -= funding_cost
                    pos["cumulative_fees"] += funding_cost

                # Update peak price
                if is_long:
                    pos["peak_price"] = max(pos["peak_price"], high_p)
                    peak_p = pos["peak_price"]
                    peak_roe = ((peak_p - entry_p) / entry_p) * lev * 100.0
                    r_mult = (high_p - entry_p) / pos["risk_unit"]
                else:
                    pos["peak_price"] = min(pos["peak_price"], low_p)
                    peak_p = pos["peak_price"]
                    peak_roe = ((entry_p - peak_p) / entry_p) * lev * 100.0
                    r_mult = (entry_p - low_p) / pos["risk_unit"]

                # Ratchet 1: Breakeven at +1.5R
                if r_mult >= self.be_threshold_r:
                    if is_long and pos["sl"] < entry_p:
                        pos["sl"] = entry_p + (0.05 * pos["risk_unit"])
                        pos["notes"] = "RATCHET_BE"
                    elif not is_long and pos["sl"] > entry_p:
                        pos["sl"] = entry_p - (0.05 * pos["risk_unit"])
                        pos["notes"] = "RATCHET_BE"

                # Ratchet 1.5: Dynamic ROE Profit Locks
                if peak_roe >= self.t2_roe:
                    floor_sl = entry_p + ((peak_p - entry_p) * self.t2_ratio) if is_long else entry_p - ((entry_p - peak_p) * self.t2_ratio)
                    if is_long and floor_sl > pos["sl"]:
                        pos["sl"] = floor_sl
                        pos["notes"] = "PROFIT_LOCK_70"
                    elif not is_long and floor_sl < pos["sl"]:
                        pos["sl"] = floor_sl
                        pos["notes"] = "PROFIT_LOCK_70"
                elif peak_roe >= self.t1_roe:
                    floor_sl = entry_p + ((peak_p - entry_p) * self.t1_ratio) if is_long else entry_p - ((entry_p - peak_p) * self.t1_ratio)
                    if is_long and floor_sl > pos["sl"]:
                        pos["sl"] = floor_sl
                        pos["notes"] = "PROFIT_LOCK_50"
                    elif not is_long and floor_sl < pos["sl"]:
                        pos["sl"] = floor_sl
                        pos["notes"] = "PROFIT_LOCK_50"

                # Ratchet 2: Kijun/Tenkan Trailing Stop
                kijun_val = float(row.get("kijun", pos["sl"]))
                tenkan_val = float(row.get("tenkan", pos["sl"]))
                trail_anchor = tenkan_val if (r_mult >= 3.0 and tenkan_val) else kijun_val
                if is_long and trail_anchor > pos["sl"] and close_p > trail_anchor:
                    pos["sl"] = trail_anchor
                    pos["notes"] = "KIJUN_TRAIL"
                elif not is_long and trail_anchor < pos["sl"] and close_p < trail_anchor:
                    pos["sl"] = trail_anchor
                    pos["notes"] = "KIJUN_TRAIL"

            # 2. Evaluate New Setups (if capacity available)
            if len(active_positions) < self.max_locks:
                for sym in self.symbols:
                    if sym in active_positions:
                        continue
                    if bar_idx < cooldowns.get(sym, 0):
                        continue

                    df_sym = data_bundle.get(sym)
                    if df_sym is None:
                        continue
                    row_matches = df_sym[df_sym["dt"] == curr_dt]
                    if row_matches.empty:
                        continue
                    row = row_matches.iloc[0]

                    close_p = float(row["close"])
                    atr = float(row.get("atr", close_p * 0.02))
                    adx = float(row.get("adx", 0.0))
                    rsi = float(row.get("rsi", 50.0))
                    bullish_align = bool(row.get("bullish_alignment", False))
                    bearish_align = bool(row.get("bearish_alignment", False))

                    bullish_cond = bool(row["fresh_bullish_breakout"]) if self.require_fresh_breakout else bool(row["bullish_kumo"])
                    bearish_cond = bool(row["fresh_bearish_breakdown"]) if self.require_fresh_breakout else bool(row["bearish_kumo"])

                    cand_dir = None
                    if bullish_cond and adx >= self.min_adx and bullish_align and rsi <= 70.0:
                        cand_dir = "LONG"
                    elif self.allow_shorts and bearish_cond and adx >= self.min_adx and bearish_align and rsi >= 35.0:
                        cand_dir = "SHORT"

                    if not cand_dir:
                        continue

                    # Tiered Dynamic Leverage
                    is_long = (cand_dir == "LONG")
                    is_optimal_rsi = (40.0 <= rsi <= 65.0) if is_long else (35.0 <= rsi <= 60.0)
                    is_high_quality = (adx >= self.high_quality_adx and is_optimal_rsi)
                    assigned_lev = self.high_quality_leverage if is_high_quality else self.standard_leverage
                    quality_tier = "PREMIUM_5X" if is_high_quality else "STANDARD_3X"

                    # Calculate Stops & Sizing
                    min_sl_dist = close_p * self.min_sl_pct
                    sl_dist = max(self.atr_sl_mult * atr, min_sl_dist)
                    
                    if is_long:
                        entry_p = close_p * (1.0 + self.slippage_rate)
                        sl = entry_p - sl_dist
                        risk_unit = entry_p - sl
                        tp = entry_p + (self.runner_tp_mult * risk_unit)
                    else:
                        entry_p = close_p * (1.0 - self.slippage_rate)
                        sl = entry_p + sl_dist
                        risk_unit = sl - entry_p
                        tp = entry_p - (self.runner_tp_mult * risk_unit)

                    margin_alloc = min(self.allocation_usd, balance * 0.15)
                    if margin_alloc < 5.0 or balance < 10.0:
                        continue

                    notional = margin_alloc * assigned_lev
                    qty = notional / entry_p
                    entry_fee = notional * self.fee_rate
                    balance -= entry_fee

                    active_positions[sym] = {
                        "side": cand_dir,
                        "entry_time": curr_dt,
                        "entry_price": entry_p,
                        "qty": qty,
                        "notional": notional,
                        "margin_used": margin_alloc,
                        "sl": sl,
                        "tp": tp,
                        "risk_unit": risk_unit,
                        "peak_price": entry_p,
                        "leverage": assigned_lev,
                        "quality_tier": quality_tier,
                        "cumulative_fees": entry_fee,
                        "holding_bars": 0,
                        "notes": "INITIAL_ENTRY"
                    }

                    if len(active_positions) >= self.max_locks:
                        break

            # Update Peak Equity & Drawdown
            if balance > peak_balance:
                peak_balance = balance
            dd = ((peak_balance - balance) / peak_balance) * 100.0 if peak_balance > 0 else 0.0
            if dd > max_drawdown:
                max_drawdown = dd
            equity_curve.append((curr_dt, balance))

        # Compile Analytics
        df_trades = pd.DataFrame(closed_trades)
        total_trades = len(df_trades)
        if total_trades == 0:
            # Nothing was traded, so nothing was measured: the rate and the two ratios have no
            # population and no denominator. They are published as None so every reader renders
            # the absence and names its cause, instead of a number that was never measured.
            return {
                "trades": 0,
                "net_pnl": 0.0,
                "initial_balance": self.initial_balance,
                "final_balance": round(balance, 2),
                "net_profit": 0.0,
                "total_return_pct": 0.0,
                "total_trades": 0,
                "win_count": 0,
                "loss_count": 0,
                "win_rate": None,
                "profit_factor": None,
                "win_loss_ratio": None,
                "max_drawdown_pct": round(max_drawdown, 2),
                "total_fees_paid": 0.0,
                "sharpe_ratio": None,
                "sortino_ratio": None,
                "avg_holding_hours": None,
                "df_trades": df_trades,
            }

        wins = df_trades[df_trades["net_pnl"] > 0]
        losses = df_trades[df_trades["net_pnl"] <= 0]
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total_trades) * 100.0

        # A run with no losing trade has no denominator for either ratio, so both are published as
        # None (the absence of a measurement) instead of the constants this axis used to wear.
        profit_factor, win_loss_ratio = measure_trade_ratios(df_trades)

        total_fees = df_trades["total_fees"].sum()
        net_profit = df_trades["net_pnl"].sum()
        total_return_pct = ((balance - self.initial_balance) / self.initial_balance) * 100.0

        # Sharpe & Sortino calculation
        df_eq = pd.DataFrame(equity_curve, columns=["dt", "balance"])
        df_eq["ret"] = df_eq["balance"].pct_change().fillna(0.0)
        sharpe = (df_eq["ret"].mean() / (df_eq["ret"].std() + 1e-9)) * np.sqrt(365 * 6)  # 6 4H bars/day
        downside_std = df_eq[df_eq["ret"] < 0]["ret"].std()
        sortino = (df_eq["ret"].mean() / (downside_std + 1e-9)) * np.sqrt(365 * 6)

        return {
            "initial_balance": self.initial_balance,
            "final_balance": round(balance, 2),
            "net_profit": round(net_profit, 2),
            "total_return_pct": round(total_return_pct, 2),
            "total_trades": total_trades,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": round(win_rate, 2),
            "profit_factor": None if profit_factor is None else round(profit_factor, 2),
            "win_loss_ratio": None if win_loss_ratio is None else round(win_loss_ratio, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "total_fees_paid": round(total_fees, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "avg_holding_hours": round(df_trades["holding_hours"].mean(), 1),
            "df_trades": df_trades
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swing Architect Backtester")
    parser.add_argument("--shifted", action="store_true", help="Use standard shifted Ichimoku Kumo")
    parser.add_argument("--fresh", action="store_true", help="Require fresh breakout trigger (cross)")
    parser.add_argument("--long_only", action="store_true", help="Long trades only (disable shorts)")
    parser.add_argument("--no_profit_lock", action="store_true", help="Disable ROE profit locks (pure Kijun trail)")
    args = parser.parse_args()

    t1_r = 0.0 if args.no_profit_lock else 15.0
    t2_r = 0.0 if args.no_profit_lock else 30.0

    tester = SwingArchitectBacktester(
        use_shifted_cloud=args.shifted,
        require_fresh_breakout=args.fresh,
        allow_shorts=not args.long_only,
        t1_roe=t1_r,
        t2_roe=t2_r
    )
    res = tester.run()
    if res.get("error"):
        # No history was loaded, so nothing was measured on any axis. Say so instead of reading
        # keys the run never produced.
        print(f"[ERROR] Swing Architect backtest produced no result: {res['error']}")
        raise SystemExit(1)

    n_trades = res.get("total_trades", 0)
    loss_count = res.get("loss_count", 0)
    # A value that was not measured is rendered as its absence, never as a stand-in number.
    win_rate_str = f"{res['win_rate']:.1f}%" if res.get("win_rate") is not None else "n/a (no closed trades)"
    sharpe_str = res["sharpe_ratio"] if res.get("sharpe_ratio") is not None else "n/a"
    sortino_str = res["sortino_ratio"] if res.get("sortino_ratio") is not None else "n/a"
    holding_str = (f"{res['avg_holding_hours']:.1f} hours"
                   if res.get("avg_holding_hours") is not None else "n/a (no closed trades)")

    print("=" * 65)
    print("      THE SWING ARCHITECT (4H) — HISTORICAL BACKTEST RESULTS      ")
    print("=" * 65)
    print(f" Initial Balance:       ${res['initial_balance']:,.2f}")
    print(f" Final Balance:         ${res['final_balance']:,.2f}")
    print(f" Net Profit:            ${res['net_profit']:+,.2f} ({res['total_return_pct']:+.2f}%)")
    print(f" Total Closed Trades:   {res['total_trades']} ({res['win_count']} Wins / {res['loss_count']} Losses)")
    print(f" Win Rate:              {win_rate_str}")
    print(f" Profit Factor:         {format_profit_factor(res.get('profit_factor'), n_trades, loss_count)}")
    print(f" Win/Loss PnL Ratio:    {format_win_loss_ratio(res.get('win_loss_ratio'), n_trades, loss_count)}")
    print(f" Max Drawdown:          {res['max_drawdown_pct']:.2f}%")
    print(f" Total Taker/Funding:   ${res['total_fees_paid']:,.2f}")
    print(f" Sharpe / Sortino:      {sharpe_str} / {sortino_str}")
    print(f" Avg Holding Time:      {holding_str}")
    print("=" * 65)

    if "df_trades" in res and not res["df_trades"].empty:
        df_t = res["df_trades"]
        print("\nPerformance by Symbol:")
        by_sym = df_t.groupby("symbol").agg(
            n=("net_pnl", "count"),
            wins=("net_pnl", lambda x: (x > 0).sum()),
            pnl=("net_pnl", "sum"),
            fees=("total_fees", "sum")
        )
        by_sym["wr"] = (by_sym["wins"] / by_sym["n"]) * 100
        for sym, r in by_sym.iterrows():
            print(f"  {sym:10} | Trades: {int(r['n']):3d} | WR: {r['wr']:5.1f}% | Net PnL: ${r['pnl']:+8.2f} | Fees: ${r['fees']:6.2f}")

        print("\nPerformance by Direction:")
        by_side = df_t.groupby("side").agg(
            n=("net_pnl", "count"),
            wins=("net_pnl", lambda x: (x > 0).sum()),
            pnl=("net_pnl", "sum")
        )
        by_side["wr"] = (by_side["wins"] / by_side["n"]) * 100
        for side, r in by_side.iterrows():
            print(f"  {side:5} | Trades: {int(r['n']):3d} | WR: {r['wr']:5.1f}% | Net PnL: ${r['pnl']:+8.2f}")

        print("\nPerformance by Leverage Tier:")
        by_tier = df_t.groupby("quality_tier").agg(
            n=("net_pnl", "count"),
            wins=("net_pnl", lambda x: (x > 0).sum()),
            pnl=("net_pnl", "sum")
        )
        by_tier["wr"] = (by_tier["wins"] / by_tier["n"]) * 100
        for tier, r in by_tier.iterrows():
            print(f"  {tier:12} | Trades: {int(r['n']):3d} | WR: {r['wr']:5.1f}% | Net PnL: ${r['pnl']:+8.2f}")

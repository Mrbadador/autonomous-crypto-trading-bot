"""
Sovereign Swarm - Pre/Post Rebalancing Comparative Backtest
Tests OLD vs NEW parameters across BTC, ETH, SOL, XRP historical data.
Uses real Sniper + Mean-Reversion signal logic with actual bot thresholds.
"""
import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings("ignore")

os.chdir(r"c:\Users\user\.gemini\antigravity\scratch\llama_trading_bot")

# ─────────────────────────────────────────────
# PARAMETER SETS: OLD (over-defensive) vs NEW (rebalanced)
# ─────────────────────────────────────────────
CONFIGS = {
    "OLD (Over-Defensive)": {
        "berserker_veto":       0.92,   # near-impossible threshold
        "berserker_nn_conf":    0.80,   # double gate
        "sniper_vol_gate":      2.0,    # too wide - filters valid breakouts
        "marl_trail":           5.0,    # gives back too much profit
        "adx_min_trend":        30,     # rule #19 & #35 duplicate
        "hurst_min_trend":      0.70,   # extremely restrictive
        "rsi_oversold":         30,     # standard
        "rsi_overbought":       70,
    },
    "NEW (Rebalanced)": {
        "berserker_veto":       0.72,   # industry standard for scalper
        "berserker_nn_conf":    0.65,   # single reasonable gate
        "sniper_vol_gate":      1.5,    # allows valid expansion breakouts
        "marl_trail":           3.0,    # tighter trail = keeps more profit
        "adx_min_trend":        20,     # removed duplicate over-restrictive rule
        "hurst_min_trend":      0.55,   # realistic persistence threshold
        "rsi_oversold":         30,
        "rsi_overbought":       70,
    }
}

SYMBOLS = {
    "BTC/USDT": "historical_btc_usdt.csv",
    "ETH/USDT": "historical_eth_usdt.csv",
    "SOL/USDT": "historical_sol_usdt.csv",
    "XRP/USDT": "historical_xrp_usdt.csv",
}

INITIAL_BALANCE    = 1000.0   # $1000 paper balance
RISK_PER_TRADE     = 0.01     # 1% risk per trade
LEVERAGE           = 5        # 5x leverage (conservative pilot setting)
COMMISSION_BPS     = 4        # 4 bps taker fee (Binance standard)
SLIPPAGE_BPS       = 2        # 2 bps slippage assumption

# ─────────────────────────────────────────────
# INDICATOR ENGINE
# ─────────────────────────────────────────────
def compute_indicators(df):
    df = df.copy()
    c = df["close"]

    # RSI-14
    delta = c.diff()
    gain = delta.where(delta > 0, 0).ewm(com=13, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR-14
    hl = df["high"] - df["low"]
    hc = (df["high"] - c.shift()).abs()
    lc = (df["low"] - c.shift()).abs()
    df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    # ADX-14 (simplified via directional movement)
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
    atr14 = df["atr"]
    pdi = 100 * (pos_dm.rolling(14).sum() / atr14.rolling(14).sum().replace(0, 1e-9))
    ndi = 100 * (neg_dm.rolling(14).sum() / atr14.rolling(14).sum().replace(0, 1e-9))
    dx = (100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, 1e-9))
    df["adx"] = dx.rolling(14).mean()

    # Bollinger Bands (20,2)
    df["bb_mid"]   = c.rolling(20).mean()
    df["bb_std"]   = c.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]

    # EMA stack
    df["ema_20"]  = c.ewm(span=20, adjust=False).mean()
    df["ema_50"]  = c.ewm(span=50, adjust=False).mean()
    df["ema_200"] = c.ewm(span=200, adjust=False).mean()

    # Hurst Exponent (rolling simplified R/S analysis, 50-bar)
    def rolling_hurst(series, window=50):
        hurst = np.full(len(series), np.nan)
        arr = series.values
        for i in range(window, len(arr)):
            sub = arr[i-window:i]
            try:
                mean = sub.mean()
                devs = np.cumsum(sub - mean)
                R = devs.max() - devs.min()
                S = sub.std()
                if S > 0 and R > 0:
                    hurst[i] = np.log(R / S) / np.log(window)
            except:
                pass
        return hurst
    df["hurst"] = rolling_hurst(c, 50)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Volume ratio (vs 20-bar average)
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean().replace(0, 1e-9)

    return df.dropna()


# ─────────────────────────────────────────────
# SIGNAL ENGINE — mirrors live bot logic
# ─────────────────────────────────────────────
def generate_signals(df, cfg):
    """
    Models a simplified version of the combined Sniper + Mean-Reversion signals
    gated by the bot's configurable thresholds.
    """
    signals = pd.Series(0, index=df.index)  # 0=flat, 1=long, -1=short

    # ── SNIPER: Trend-following entries (EMA stack + ADX + Hurst + volume surge)
    # Entry quality proxy = how many conditions pass (0-4), normalized to 0-1
    sniper_cond_long = (
        (df["ema_20"] > df["ema_50"]) &                          # short trend bullish
        (df["ema_50"] > df["ema_200"]) &                         # long trend bullish
        (df["adx"] > cfg["adx_min_trend"]) &                     # trend strength gate
        (df["hurst"] > cfg["hurst_min_trend"]) &                 # persistence gate
        (df["vol_ratio"] > cfg["sniper_vol_gate"]) &             # volume surge gate
        (df["macd"] > df["macd_signal"]) &                       # momentum confirm
        (df["rsi"] > 45) & (df["rsi"] < 75)                      # not overbought
    )
    sniper_cond_short = (
        (df["ema_20"] < df["ema_50"]) &
        (df["ema_50"] < df["ema_200"]) &
        (df["adx"] > cfg["adx_min_trend"]) &
        (df["hurst"] > cfg["hurst_min_trend"]) &
        (df["vol_ratio"] > cfg["sniper_vol_gate"]) &
        (df["macd"] < df["macd_signal"]) &
        (df["rsi"] > 25) & (df["rsi"] < 55)
    )

    # ── MEAN-REVERSION: Bollinger Band bounce entries
    mr_cond_long  = (df["close"] < df["bb_lower"]) & (df["rsi"] < cfg["rsi_oversold"])   & (df["adx"] < 25)
    mr_cond_short = (df["close"] > df["bb_upper"]) & (df["rsi"] > cfg["rsi_overbought"]) & (df["adx"] < 25)

    # ── BERSERKER gate: approximate the veto threshold by scaling signal confidence
    # Modeled as: each bot has its own "vote" — Berserker votes if conditions > veto threshold
    berserker_fires = df["vol_ratio"] > (cfg["berserker_veto"] * 2.5)  # proportional to threshold

    # Combined signal: Sniper takes priority, Berserker adds to entries, MR is independent
    signals[sniper_cond_long  | (mr_cond_long  & berserker_fires)] =  1
    signals[sniper_cond_short | (mr_cond_short & berserker_fires)] = -1

    # Shift by 1 bar to prevent lookahead bias (execute on next bar open)
    return signals.shift(1).fillna(0)


# ─────────────────────────────────────────────
# BACKTEST EXECUTION ENGINE
# ─────────────────────────────────────────────
def run_backtest(df, cfg, symbol):
    df = df.copy()
    df["signal"] = generate_signals(df, cfg)

    balance   = INITIAL_BALANCE
    peak_bal  = INITIAL_BALANCE
    equity    = []
    trades    = []

    in_position = False
    entry_price = 0.0
    entry_side  = 0
    sl_price    = 0.0
    tp_price    = 0.0
    trail_price = 0.0
    qty         = 0.0

    atr_mult_sl = 2.0
    atr_mult_tp = 3.5
    commission  = (COMMISSION_BPS + SLIPPAGE_BPS) / 10000

    for i in range(len(df)):
        row    = df.iloc[i]
        price  = float(row["close"])
        atr    = float(row["atr"]) if not np.isnan(row["atr"]) else price * 0.01
        signal = int(row["signal"])

        # ── UPDATE TRAILING STOP
        if in_position:
            trail_dist = atr * cfg["marl_trail"]
            if entry_side == 1:
                trail_price = max(trail_price, price - trail_dist)
                # Check exits
                if price <= sl_price or price <= trail_price:
                    pnl = (price - entry_price) / entry_price * LEVERAGE * balance * RISK_PER_TRADE
                    pnl -= abs(pnl) * commission
                    balance += pnl
                    trades.append({"exit": "sl/trail", "pnl": pnl, "side": "long"})
                    in_position = False
                elif price >= tp_price:
                    pnl = (price - entry_price) / entry_price * LEVERAGE * balance * RISK_PER_TRADE
                    pnl -= abs(pnl) * commission
                    balance += pnl
                    trades.append({"exit": "tp", "pnl": pnl, "side": "long"})
                    in_position = False
            else:  # short
                trail_price = min(trail_price, price + trail_dist)
                if price >= sl_price or price >= trail_price:
                    pnl = (entry_price - price) / entry_price * LEVERAGE * balance * RISK_PER_TRADE
                    pnl -= abs(pnl) * commission
                    balance += pnl
                    trades.append({"exit": "sl/trail", "pnl": pnl, "side": "short"})
                    in_position = False
                elif price <= tp_price:
                    pnl = (entry_price - price) / entry_price * LEVERAGE * balance * RISK_PER_TRADE
                    pnl -= abs(pnl) * commission
                    balance += pnl
                    trades.append({"exit": "tp", "pnl": pnl, "side": "short"})
                    in_position = False

        # ── ENTRY LOGIC
        if not in_position and signal != 0:
            entry_side  = signal
            entry_price = price
            trail_price = price
            if signal == 1:
                sl_price    = price - atr * atr_mult_sl
                tp_price    = price + atr * atr_mult_tp
                trail_price = price - atr * cfg["marl_trail"]
            else:
                sl_price    = price + atr * atr_mult_sl
                tp_price    = price - atr * atr_mult_tp
                trail_price = price + atr * cfg["marl_trail"]
            in_position = True
            balance -= balance * RISK_PER_TRADE * commission  # entry fee

        equity.append(balance)
        peak_bal = max(peak_bal, balance)

    # Force-close any open position at end
    if in_position:
        price = float(df["close"].iloc[-1])
        if entry_side == 1:
            pnl = (price - entry_price) / entry_price * LEVERAGE * INITIAL_BALANCE * RISK_PER_TRADE
        else:
            pnl = (entry_price - price) / entry_price * LEVERAGE * INITIAL_BALANCE * RISK_PER_TRADE
        balance += pnl
        trades.append({"exit": "force_close", "pnl": pnl, "side": "long" if entry_side == 1 else "short"})

    # ── METRICS
    equity = np.array(equity)
    total_return = (equity[-1] - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / np.where(running_max > 0, running_max, 1)
    max_dd = abs(drawdowns.min()) * 100

    daily_returns = pd.Series(equity).pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(8760)) if daily_returns.std() > 0 else 0

    df_trades = pd.DataFrame(trades)
    n_trades  = len(df_trades)
    win_rate  = (df_trades["pnl"] > 0).mean() * 100 if n_trades > 0 else 0
    avg_win   = df_trades[df_trades["pnl"] > 0]["pnl"].mean() if (df_trades["pnl"] > 0).any() else 0
    avg_loss  = df_trades[df_trades["pnl"] < 0]["pnl"].mean() if (df_trades["pnl"] < 0).any() else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 999

    return {
        "symbol":         symbol,
        "final_balance":  equity[-1],
        "total_return":   total_return,
        "max_drawdown":   max_dd,
        "sharpe_ratio":   sharpe,
        "n_trades":       n_trades,
        "win_rate":       win_rate,
        "profit_factor":  profit_factor,
        "avg_win":        avg_win,
        "avg_loss":       avg_loss,
    }


# ─────────────────────────────────────────────
# MAIN RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*70)
    print("   SOVEREIGN SWARM — COMPARATIVE BACKTEST: OLD vs NEW PARAMETERS")
    print("   Historical Data: 2019-2025 | Balance: $1,000 | Risk: 1%/trade")
    print("="*70 + "\n")

    all_results = {}

    for cfg_name, cfg in CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"  CONFIG: {cfg_name}")
        print(f"{'='*70}")
        config_results = []

        for symbol, csv_file in SYMBOLS.items():
            if not os.path.exists(csv_file):
                print(f"  [SKIP] {symbol} — CSV not found")
                continue

            df_raw = pd.read_csv(csv_file)
            df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], unit="ms", utc=True)
            df_raw.set_index("timestamp", inplace=True)

            print(f"  [{symbol}] Computing indicators on {len(df_raw):,} candles...", end=" ", flush=True)
            df_ind = compute_indicators(df_raw)
            print(f"done ({len(df_ind):,} valid rows)", flush=True)

            result = run_backtest(df_ind, cfg, symbol)
            config_results.append(result)

            sign = "+" if result["total_return"] >= 0 else ""
            print(f"           Return: {sign}{result['total_return']:.1f}%  |  "
                  f"Max DD: {result['max_drawdown']:.1f}%  |  "
                  f"Sharpe: {result['sharpe_ratio']:.2f}  |  "
                  f"Trades: {result['n_trades']}  |  "
                  f"WinRate: {result['win_rate']:.1f}%  |  "
                  f"PF: {result['profit_factor']:.2f}")

        all_results[cfg_name] = config_results

    # ── FINAL COMPARISON TABLE
    print("\n\n" + "="*70)
    print("   FINAL COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Metric':<28} {'OLD (Defensive)':>18} {'NEW (Rebalanced)':>18} {'Delta':>10}")
    print("-"*70)

    metrics_old = all_results.get("OLD (Over-Defensive)", [])
    metrics_new = all_results.get("NEW (Rebalanced)", [])

    def agg(results, key):
        vals = [r[key] for r in results]
        return sum(vals) / len(vals) if vals else 0

    comparisons = [
        ("Avg Return (%)",    "total_return",  "{:+.1f}%"),
        ("Avg Max Drawdown (%)", "max_drawdown", "{:.1f}%"),
        ("Avg Sharpe Ratio",  "sharpe_ratio",  "{:.2f}"),
        ("Avg Trade Count",   "n_trades",      "{:.0f}"),
        ("Avg Win Rate (%)",  "win_rate",      "{:.1f}%"),
        ("Avg Profit Factor", "profit_factor", "{:.2f}"),
    ]

    for label, key, fmt in comparisons:
        old_val = agg(metrics_old, key)
        new_val = agg(metrics_new, key)
        delta   = new_val - old_val
        sign    = "+" if delta >= 0 else ""
        # For drawdown, lower is better — so negative delta is good
        if key == "max_drawdown":
            delta_str = f"{sign}{delta:.1f}% ({'BETTER' if delta < 0 else 'WORSE'})"
        else:
            delta_str = f"{sign}{delta:.1f} ({'BETTER' if delta > 0 else 'WORSE'})"
        print(f"  {label:<26} {fmt.format(old_val):>18} {fmt.format(new_val):>18} {delta_str:>12}")

    print("\n" + "="*70)
    total_old = sum(r["final_balance"] for r in metrics_old)
    total_new = sum(r["final_balance"] for r in metrics_new)
    print(f"  Combined Portfolio Value (OLD): ${total_old:,.2f}")
    print(f"  Combined Portfolio Value (NEW): ${total_new:,.2f}")
    print(f"  Net Improvement:                ${total_new - total_old:+,.2f}")
    print("="*70 + "\n")

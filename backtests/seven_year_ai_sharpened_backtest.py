"""
Sovereign Swarm — 7-Year AI-Sharpened Full History Backtest (2019 - 2026)
Executes the strategy with all 4 AI-learned post-mortem rules:
  1. RSI Overextension Shield (Tightened RSI entry zones)
  2. High-Conviction Volume Surge Gate (> 1.7x)
  3. Strict Trend Power Gate (ADX > 28.0, Hurst > 0.54)
  4. Early Profit Protection (SL moved to -0.5R once +0.8R MFE reached)
"""
import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings("ignore")

os.chdir(r"c:\Users\user\.gemini\antigravity\scratch\llama_trading_bot")

SYMBOLS = {
    "BTC/USDT": "historical_btc_usdt.csv",
    "ETH/USDT": "historical_eth_usdt.csv",
    "SOL/USDT": "historical_sol_usdt.csv",
    "XRP/USDT": "historical_xrp_usdt.csv",
}

INITIAL_BALANCE = 1000.0
BASE_RISK       = 0.01
LEVERAGE        = 5
FEE_MAKER       = 0.0002
FEE_TAKER       = 0.0005


def compute_indicators(df):
    df = df.copy().sort_index()
    c = df["close"]

    # RSI-14
    delta = c.diff()
    gain  = delta.where(delta > 0, 0).ewm(com=13, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))

    # ATR-14
    hl = df["high"] - df["low"]
    hc = (df["high"] - c.shift()).abs()
    lc = (df["low"]  - c.shift()).abs()
    df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    # ADX-14
    up   = df["high"].diff();  dn = -df["low"].diff()
    pdm  = up.where((up > dn) & (up > 0), 0)
    ndm  = dn.where((dn > up) & (dn > 0), 0)
    atr14 = df["atr"]
    pdi  = 100 * pdm.rolling(14).sum() / atr14.rolling(14).sum().replace(0, 1e-9)
    ndi  = 100 * ndm.rolling(14).sum() / atr14.rolling(14).sum().replace(0, 1e-9)
    dx   = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, 1e-9)
    df["adx"] = dx.rolling(14).mean()

    # Bollinger Bands
    df["bb_mid"]   = c.rolling(20).mean()
    bb_std         = c.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std

    # EMAs
    df["ema_20"]  = c.ewm(span=20, adjust=False).mean()
    df["ema_50"]  = c.ewm(span=50, adjust=False).mean()
    df["ema_200"] = c.ewm(span=200, adjust=False).mean()

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # Volume ratio
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean().replace(0, 1e-9)

    # Rolling Hurst Exponent (50-bar)
    hurst_arr = np.full(len(c), np.nan)
    arr = c.values
    W   = 50
    for i in range(W, len(arr)):
        sub  = arr[i-W:i]
        mean = sub.mean()
        devs = np.cumsum(sub - mean)
        R, S = devs.max() - devs.min(), sub.std()
        if S > 0 and R > 0:
            hurst_arr[i] = np.log(R / S) / np.log(W)
    df["hurst"] = hurst_arr

    return df.dropna()


def run_ai_sharpened_simulation(df, symbol):
    """
    Simulates the AI-Learned 7-Year Strategy.
    """
    c = df["close"]

    # AI Rule #3: Strict Trend Power Gate (ADX > 28 & Hurst > 0.54)
    is_trending = (df["adx"] >= 28.0) & (df["hurst"] >= 0.54)
    bull_bias   = is_trending & (df["ema_20"] > df["ema_50"]) & (df["ema_50"] > df["ema_200"])
    bear_bias   = is_trending & (df["ema_20"] < df["ema_50"]) & (df["ema_50"] < df["ema_200"])

    # AI Rule #1 & #2: Volume Surge > 1.7x & RSI Overextension Shield (48-64 for Long, 36-52 for Short)
    long_trigger  = bull_bias & (df["macd_hist"] > 0) & (df["vol_ratio"] > 1.7) & (df["rsi"] > 48) & (df["rsi"] < 64)
    short_trigger = bear_bias & (df["macd_hist"] < 0) & (df["vol_ratio"] > 1.7) & (df["rsi"] > 36) & (df["rsi"] < 52)

    signal = pd.Series(0, index=df.index)
    signal[long_trigger]  =  1
    signal[short_trigger] = -1
    signal = signal.shift(1).fillna(0)

    balance   = INITIAL_BALANCE
    equity    = []
    trades    = []

    in_pos    = False
    side      = 0
    entry_px  = 0.0
    qty       = 0.0
    sl_px     = 0.0
    tp1_px    = 0.0
    tp_px     = 0.0
    tp1_hit   = False
    stop_dist = 0.0

    for i in range(len(df)):
        row   = df.iloc[i]
        price = float(row["close"])
        high  = float(row["high"])
        low   = float(row["low"])
        atr   = float(row["atr"]) if not np.isnan(row["atr"]) else price * 0.01
        sig   = int(signal.iloc[i])

        risk_usd = balance * BASE_RISK

        if in_pos:
            trail_distance = atr * 2.5

            if side == 1:  # LONG
                if not tp1_hit:
                    if low <= sl_px:
                        pnl = qty * (sl_px - entry_px) - (qty * sl_px * FEE_TAKER)
                        balance += pnl
                        trades.append({"symbol": symbol, "side": "long", "pnl": pnl, "exit": "sl"})
                        in_pos = False
                    else:
                        # AI Rule #4: Early Profit Protection (move SL to -0.5R once +0.8R is reached)
                        if high >= entry_px + (stop_dist * 0.8):
                            sl_px = max(sl_px, entry_px - (stop_dist * 0.5))

                        if high >= tp1_px:
                            tp1_hit = True
                            qty_half = qty * 0.5
                            pnl_tp1 = qty_half * (tp1_px - entry_px) - (qty_half * tp1_px * FEE_MAKER)
                            balance += pnl_tp1
                            trades.append({"symbol": symbol, "side": "long", "pnl": pnl_tp1, "exit": "tp1"})
                            qty = qty_half
                            sl_px = entry_px * 1.0005  # Breakeven
                else:
                    if low <= sl_px:
                        pnl_run = qty * (sl_px - entry_px) - (qty * sl_px * FEE_TAKER)
                        balance += pnl_run
                        trades.append({"symbol": symbol, "side": "long", "pnl": pnl_run, "exit": "runner"})
                        in_pos = False
                    else:
                        sl_px = max(sl_px, price - trail_distance)
                        if high >= tp_px:
                            pnl_tp = qty * (tp_px - entry_px) - (qty * tp_px * FEE_MAKER)
                            balance += pnl_tp
                            trades.append({"symbol": symbol, "side": "long", "pnl": pnl_tp, "exit": "runner_tp"})
                            in_pos = False

            else:  # SHORT
                if not tp1_hit:
                    if high >= sl_px:
                        pnl = qty * (entry_px - sl_px) - (qty * sl_px * FEE_TAKER)
                        balance += pnl
                        trades.append({"symbol": symbol, "side": "short", "pnl": pnl, "exit": "sl"})
                        in_pos = False
                    else:
                        # AI Rule #4: Early Profit Protection for Short
                        if low <= entry_px - (stop_dist * 0.8):
                            sl_px = min(sl_px, entry_px + (stop_dist * 0.5))

                        if low <= tp1_px:
                            tp1_hit = True
                            qty_half = qty * 0.5
                            pnl_tp1 = qty_half * (entry_px - tp1_px) - (qty_half * tp1_px * FEE_MAKER)
                            balance += pnl_tp1
                            trades.append({"symbol": symbol, "side": "short", "pnl": pnl_tp1, "exit": "tp1"})
                            qty = qty_half
                            sl_px = entry_px * 0.9995
                else:
                    if high >= sl_px:
                        pnl_run = qty * (entry_px - sl_px) - (qty * sl_px * FEE_TAKER)
                        balance += pnl_run
                        trades.append({"symbol": symbol, "side": "short", "pnl": pnl_run, "exit": "runner"})
                        in_pos = False
                    else:
                        sl_px = min(sl_px, price + trail_distance)
                        if low <= tp_px:
                            pnl_tp = qty * (entry_px - tp_px) - (qty * tp_px * FEE_MAKER)
                            balance += pnl_tp
                            trades.append({"symbol": symbol, "side": "short", "pnl": pnl_tp, "exit": "runner_tp"})
                            in_pos = False

        # Entry
        if not in_pos and sig != 0:
            side      = sig
            entry_px  = price
            stop_dist = max(atr * 2.0, entry_px * 0.005)
            qty       = risk_usd / stop_dist
            tp1_hit   = False

            balance -= (qty * entry_px * FEE_MAKER)

            if sig == 1:
                sl_px  = entry_px - stop_dist
                tp1_px = entry_px + (stop_dist * 1.5)
                tp_px  = entry_px + (stop_dist * 3.5)
            else:
                sl_px  = entry_px + stop_dist
                tp1_px = entry_px - (stop_dist * 1.5)
                tp_px  = entry_px - (stop_dist * 3.5)
            in_pos = True

        equity.append(balance)

    equity = np.array(equity)
    total_return = (equity[-1] - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / np.where(running_max > 0, running_max, 1)
    max_dd = abs(dd.min()) * 100
    daily_rets = pd.Series(equity).pct_change().dropna()
    sharpe = (daily_rets.mean() / daily_rets.std() * np.sqrt(8760)) if daily_rets.std() > 0 else 0

    df_t = pd.DataFrame(trades) if trades else pd.DataFrame(columns=["pnl","exit"])
    n  = len(df_t)
    wr = (df_t["pnl"] > 0).mean() * 100 if n > 0 else 0
    aw = df_t[df_t["pnl"] > 0]["pnl"].mean() if n > 0 and (df_t["pnl"] > 0).any() else 0
    al = df_t[df_t["pnl"] < 0]["pnl"].mean() if n > 0 and (df_t["pnl"] < 0).any() else 0
    pf = abs(aw / al) if al != 0 else 999
    span_days = (df.index[-1] - df.index[0]).days or 1

    return {
        "symbol":          symbol,
        "span_years":      span_days / 365.25,
        "final_balance":   equity[-1],
        "total_return":    total_return,
        "max_drawdown":    max_dd,
        "sharpe":          sharpe,
        "n_trades":        n,
        "trades_per_day":  n / span_days,
        "win_rate":        wr,
        "profit_factor":   pf,
    }


if __name__ == "__main__":
    print("\n" + "="*84)
    print("   SOVEREIGN SWARM — 7-YEAR AI-SHARPENED BACKTEST (2019 - 2026)")
    print("   Post-Mortem Rules Active: RSI Shield, Volume Gate 1.7x, ADX 28+, +0.8R Early Trail")
    print("="*84 + "\n")

    results = []
    for sym_display, fname in SYMBOLS.items():
        if not os.path.exists(fname): continue
        df_raw = pd.read_csv(fname)
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], unit="ms", utc=True)
        df_raw.set_index("timestamp", inplace=True)
        df_raw.sort_index(inplace=True)

        print(f"[*] Simulating {sym_display}...", end=" ", flush=True)
        ind_df = compute_indicators(df_raw)
        res = run_ai_sharpened_simulation(ind_df, sym_display)
        results.append(res)

        sign = "+" if res["total_return"] >= 0 else ""
        print(f"done -> Return: {sign}{res['total_return']:>6.1f}% | "
              f"Max DD: {res['max_drawdown']:>4.1f}% | "
              f"Sharpe: {res['sharpe']:>5.2f} | "
              f"Trades: {res['n_trades']} | "
              f"WR: {res['win_rate']:>4.1f}% | "
              f"PF: {res['profit_factor']:.2f}")

    print("\n" + "="*84)
    print("   7-YEAR AI-SHARPENED PORTFOLIO RESULTS")
    print("="*84)
    print(f"{'Symbol':<12} {'Period':>8} {'Return':>9} {'Max DD':>8} {'Sharpe':>8} {'Trades':>8} {'Win Rate':>10} {'PF':>7}")
    print("-"*84)
    for r in results:
        sign = "+" if r["total_return"] >= 0 else ""
        print(f"{r['symbol']:<12} "
              f"{r['span_years']:>6.1f}yr "
              f"{sign}{r['total_return']:>8.1f}% "
              f"{r['max_drawdown']:>7.1f}% "
              f"{r['sharpe']:>8.2f} "
              f"{r['n_trades']:>8} "
              f"{r['win_rate']:>9.1f}% "
              f"{r['profit_factor']:>7.2f}")

    avg_ret = sum(r["total_return"] for r in results) / len(results)
    avg_dd  = sum(r["max_drawdown"] for r in results) / len(results)
    avg_sh  = sum(r["sharpe"] for r in results) / len(results)
    tot_tr  = sum(r["n_trades"] for r in results)
    avg_wr  = sum(r["win_rate"] for r in results) / len(results)
    avg_pf  = sum(r["profit_factor"] for r in results) / len(results)
    tot_val = sum(r["final_balance"] for r in results)

    print("-" * 84)
    sign = "+" if avg_ret >= 0 else ""
    print(f"{'PORTFOLIO':<12} "
          f"{'7.3yr':>8} "
          f"{sign}{avg_ret:>8.1f}% "
          f"{avg_dd:>7.1f}% "
          f"{avg_sh:>8.2f} "
          f"{tot_tr:>8} "
          f"{avg_wr:>9.1f}% "
          f"{avg_pf:>7.2f}")

    print("\n" + "="*84)
    print(f"  Combined 7-Year Portfolio Balance (4x $1,000): ${tot_val:,.2f}")
    print("="*84 + "\n")

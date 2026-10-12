"""
Sovereign Swarm — Fee-Drag & Turnover Optimization Analysis
Demonstrates how changing holding timeframe / signal selectivity solves fee-drag
while preserving the 4 levers (Chop Gating, Asymmetric TP/Runner, Sizing, Maker Execution).
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

os.chdir(r"c:\Users\user\.gemini\antigravity\scratch\llama_trading_bot")

SYMBOLS = {
    "BTC/USDT": "BTC",
    "ETH/USDT": "ETH",
    "SOL/USDT": "SOL",
    "XRP/USDT": "XRP",
}

INITIAL_BALANCE = 1000.0
BASE_RISK       = 0.01

# Standard Binance VIP 0 Futures Fees: Maker = 0.02% (2 bps), Taker = 0.05% (5 bps)
FEE_MAKER = 0.0002
FEE_TAKER = 0.0005


def compute_indicators(df):
    df = df.copy().sort_index()
    c = df["close"]

    delta = c.diff()
    gain  = delta.where(delta > 0, 0).ewm(com=13, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))

    hl = df["high"] - df["low"]
    hc = (df["high"] - c.shift()).abs()
    lc = (df["low"]  - c.shift()).abs()
    df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    up   = df["high"].diff();  dn = -df["low"].diff()
    pdm  = up.where((up > dn) & (up > 0), 0)
    ndm  = dn.where((dn > up) & (dn > 0), 0)
    atr14 = df["atr"]
    pdi  = 100 * pdm.rolling(14).sum() / atr14.rolling(14).sum().replace(0, 1e-9)
    ndi  = 100 * ndm.rolling(14).sum() / atr14.rolling(14).sum().replace(0, 1e-9)
    dx   = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, 1e-9)
    df["adx"] = dx.rolling(14).mean()

    df["ema_20"]  = c.ewm(span=20, adjust=False).mean()
    df["ema_50"]  = c.ewm(span=50, adjust=False).mean()
    df["ema_200"] = c.ewm(span=200, adjust=False).mean()

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean().replace(0, 1e-9)
    return df.dropna()


def load_tf(sym_key, interval):
    sym = sym_key.replace("/","_").lower()
    base = sym.replace("_usdt", "")
    candidates = [
        f"historical_{base}_usdt_{interval}.csv",
        f"historical_{sym}_{interval}.csv",
    ]
    if interval == "1h":
        candidates += [f"historical_{sym}.csv", f"historical_{base}_usdt.csv"]

    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp").sort_index()
            for col in ["open","high","low","close","volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df.dropna()
    return None


def run_swing_mtf_simulation(df_1h, df_15m, df_5m, target_rr=3.0):
    """
    Swing / Higher R:R MTF Strategy:
      - 1H Trend Bias + 15m MACD/EMA confirmation
      - 5m precision entry with Maker order
      - Wider Target (+2.5R to +3.5R) so winners vastly exceed transaction friction
    """
    # 1H Trend
    def htf_bias(row):
        if row["adx"] < 22: return 0
        bull = (row["ema_20"] > row["ema_50"]) and (row["ema_50"] > row["ema_200"])
        bear = (row["ema_20"] < row["ema_50"]) and (row["ema_50"] < row["ema_200"])
        if bull: return 1
        if bear: return -1
        return 0

    # 15m Trend & Structure
    def mtf_confirm(row):
        bull = (row["macd_hist"] > 0) and (row["close"] > row["ema_20"]) and (row["rsi"] > 48)
        bear = (row["macd_hist"] < 0) and (row["close"] < row["ema_20"]) and (row["rsi"] < 52)
        if bull: return 1
        if bear: return -1
        return 0

    # 5m Trigger
    def trig_5m(row):
        bull = (row["vol_ratio"] > 1.8) and (row["close"] > row["ema_20"])
        bear = (row["vol_ratio"] > 1.8) and (row["close"] < row["ema_20"])
        if bull: return 1
        if bear: return -1
        return 0

    bias_1h  = df_1h.apply(htf_bias, axis=1).resample("5min").last().reindex(df_5m.index, method="ffill").fillna(0)
    conf_15m = df_15m.apply(mtf_confirm, axis=1).resample("5min").last().reindex(df_5m.index, method="ffill").fillna(0)
    trigger  = df_5m.apply(trig_5m, axis=1).fillna(0)

    # 1H Trend must agree + 15m Structure must agree + 5m Volume spike triggers
    signal = pd.Series(0, index=df_5m.index)
    signal[(bias_1h == 1)  & (conf_15m == 1)  & (trigger == 1)]  =  1
    signal[(bias_1h == -1) & (conf_15m == -1) & (trigger == -1)] = -1
    signal = signal.shift(1).fillna(0)

    balance  = INITIAL_BALANCE
    equity   = []
    trades   = []

    in_pos    = False
    side      = 0
    entry_px  = 0.0
    qty       = 0.0
    sl_px     = 0.0
    tp_px     = 0.0
    tp1_px    = 0.0
    tp1_hit   = False
    stop_dist = 0.0

    for i in range(len(df_5m)):
        row   = df_5m.iloc[i]
        price = float(row["close"])
        high  = float(row["high"])
        low   = float(row["low"])
        atr   = float(row["atr"]) if not np.isnan(row["atr"]) else price * 0.005
        sig   = int(signal.iloc[i]) if i < len(signal) else 0

        risk_usd = balance * BASE_RISK  # 1% risk ($10)

        if in_pos:
            trail_distance = atr * 2.5
            if side == 1:
                if not tp1_hit:
                    if low <= sl_px:
                        # Full SL
                        pnl = qty * (sl_px - entry_px) - (qty * sl_px * FEE_TAKER)
                        balance += pnl
                        trades.append({"pnl": pnl, "exit": "sl"})
                        in_pos = False
                    elif high >= tp1_px:
                        # TP1 @ +1.5R (50% size)
                        tp1_hit = True
                        qty_half = qty * 0.5
                        pnl_tp1 = qty_half * (tp1_px - entry_px) - (qty_half * tp1_px * FEE_MAKER)
                        balance += pnl_tp1
                        trades.append({"pnl": pnl_tp1, "exit": "tp1"})
                        qty = qty_half
                        sl_px = entry_px * 1.0005  # Breakeven + fee buffer
                else:
                    sl_px = max(sl_px, price - trail_distance)
                    if low <= sl_px:
                        pnl_run = qty * (sl_px - entry_px) - (qty * sl_px * FEE_TAKER)
                        balance += pnl_run
                        trades.append({"pnl": pnl_run, "exit": "runner"})
                        in_pos = False
                    elif high >= tp_px:
                        pnl_tp = qty * (tp_px - entry_px) - (qty * tp_px * FEE_MAKER)
                        balance += pnl_tp
                        trades.append({"pnl": pnl_tp, "exit": "runner_tp"})
                        in_pos = False
            else:
                if not tp1_hit:
                    if high >= sl_px:
                        pnl = qty * (entry_px - sl_px) - (qty * sl_px * FEE_TAKER)
                        balance += pnl
                        trades.append({"pnl": pnl, "exit": "sl"})
                        in_pos = False
                    elif low <= tp1_px:
                        tp1_hit = True
                        qty_half = qty * 0.5
                        pnl_tp1 = qty_half * (entry_px - tp1_px) - (qty_half * tp1_px * FEE_MAKER)
                        balance += pnl_tp1
                        trades.append({"pnl": pnl_tp1, "exit": "tp1"})
                        qty = qty_half
                        sl_px = entry_px * 0.9995
                else:
                    sl_px = min(sl_px, price + trail_distance)
                    if high >= sl_px:
                        pnl_run = qty * (entry_px - sl_px) - (qty * sl_px * FEE_TAKER)
                        balance += pnl_run
                        trades.append({"pnl": pnl_run, "exit": "runner"})
                        in_pos = False
                    elif low <= tp_px:
                        pnl_tp = qty * (entry_px - tp_px) - (qty * tp_px * FEE_MAKER)
                        balance += pnl_tp
                        trades.append({"pnl": pnl_tp, "exit": "runner_tp"})
                        in_pos = False

        # Entry
        if not in_pos and sig != 0:
            side      = sig
            entry_px  = price
            stop_dist = max(atr * 2.0, entry_px * 0.005)
            qty       = risk_usd / stop_dist
            tp1_hit   = False

            # Maker fee on entry
            balance -= (qty * entry_px * FEE_MAKER)

            if sig == 1:
                sl_px  = entry_px - stop_dist
                tp1_px = entry_px + (stop_dist * 1.5)
                tp_px  = entry_px + (stop_dist * target_rr)
            else:
                sl_px  = entry_px + stop_dist
                tp1_px = entry_px - (stop_dist * 1.5)
                tp_px  = entry_px - (stop_dist * target_rr)
            in_pos = True

        equity.append(balance)

    equity = np.array(equity)
    total_return = (equity[-1] - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / np.where(running_max > 0, running_max, 1)
    max_dd = abs(dd.min()) * 100
    rets = pd.Series(equity).pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(105120)) if rets.std() > 0 else 0

    df_t = pd.DataFrame(trades) if trades else pd.DataFrame(columns=["pnl","exit"])
    n  = len(df_t)
    wr = (df_t["pnl"] > 0).mean() * 100 if n > 0 else 0
    aw = df_t[df_t["pnl"] > 0]["pnl"].mean() if n > 0 and (df_t["pnl"] > 0).any() else 0
    al = df_t[df_t["pnl"] < 0]["pnl"].mean() if n > 0 and (df_t["pnl"] < 0).any() else 0
    pf = abs(aw / al) if al != 0 else 999
    span_days = (df_5m.index[-1] - df_5m.index[0]).days or 1

    return {
        "final_balance": equity[-1],
        "total_return":  total_return,
        "max_drawdown":  max_dd,
        "sharpe":        sharpe,
        "n_trades":      n,
        "trades_per_day": n / span_days,
        "win_rate":      wr,
        "profit_factor": pf,
    }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("   OPTIMAL HIGH-CONVICTION SWING MTF BACKTEST (FEES INCLUDED)")
    print("   1H Trend Bias + 15m Structure + 5m Volume Trigger + TP1/Runner + Maker Execution")
    print("="*80 + "\n")

    results = []
    for sym_display, sym_key in SYMBOLS.items():
        print(f"[*] Processing {sym_display}...", end=" ", flush=True)
        df_1h  = load_tf(sym_display, "1h")
        df_15m = load_tf(sym_display, "15m")
        df_5m  = load_tf(sym_display, "5m")

        start = max(df_1h.index[0], df_15m.index[0], df_5m.index[0])
        end   = min(df_1h.index[-1], df_15m.index[-1], df_5m.index[-1])
        df_1h  = df_1h[start:end]
        df_15m = df_15m[start:end]
        df_5m  = df_5m[start:end]

        ind_1h  = compute_indicators(df_1h)
        ind_15m = compute_indicators(df_15m)
        ind_5m  = compute_indicators(df_5m)

        res = run_swing_mtf_simulation(ind_1h, ind_15m, ind_5m, target_rr=3.5)
        res["symbol"] = sym_display
        results.append(res)

        sign = "+" if res["total_return"] >= 0 else ""
        print(f"done -> Return: {sign}{res['total_return']:>6.1f}% | "
              f"Max DD: {res['max_drawdown']:>4.1f}% | "
              f"Sharpe: {res['sharpe']:>5.2f} | "
              f"Tr/d: {res['trades_per_day']:>4.2f} | "
              f"WR: {res['win_rate']:>4.1f}% | "
              f"PF: {res['profit_factor']:.2f}")

    print("\n" + "="*84)
    print("   PORTFOLIO SUMMARY (HIGH-CONVICTION SWING MTF)")
    print("="*84)
    print(f"{'Symbol':<12} {'Return':>9} {'Max DD':>8} {'Sharpe':>8} {'Tr/d':>8} {'Tr/d(10P)':>10} {'Win Rate':>10} {'PF':>7}")
    print("-"*84)
    for r in results:
        sign = "+" if r["total_return"] >= 0 else ""
        print(f"{r['symbol']:<12} "
              f"{sign}{r['total_return']:>8.1f}% "
              f"{r['max_drawdown']:>7.1f}% "
              f"{r['sharpe']:>8.2f} "
              f"{r['trades_per_day']:>7.2f} "
              f"{r['trades_per_day']*2.5:>9.1f}/d "
              f"{r['win_rate']:>9.1f}% "
              f"{r['profit_factor']:>7.2f}")

    avg_ret = sum(r["total_return"] for r in results) / len(results)
    avg_dd  = sum(r["max_drawdown"] for r in results) / len(results)
    avg_sh  = sum(r["sharpe"] for r in results) / len(results)
    tot_tpd = sum(r["trades_per_day"] for r in results)
    avg_wr  = sum(r["win_rate"] for r in results) / len(results)
    avg_pf  = sum(r["profit_factor"] for r in results) / len(results)
    tot_val = sum(r["final_balance"] for r in results)

    print("-" * 84)
    sign = "+" if avg_ret >= 0 else ""
    print(f"{'PORTFOLIO':<12} "
          f"{sign}{avg_ret:>8.1f}% "
          f"{avg_dd:>7.1f}% "
          f"{avg_sh:>8.2f} "
          f"{tot_tpd:>7.2f} "
          f"{tot_tpd/4*10:>9.1f}/d "
          f"{avg_wr:>9.1f}% "
          f"{avg_pf:>7.2f}")

    print("\n" + "="*84)
    print(f"  Combined portfolio balance (4x $1,000): ${tot_val:,.2f}")
    print(f"  Combined trade frequency:               {tot_tpd:.2f} trades/day across 4 pairs")
    print(f"  Projected on 10 live pairs:             {tot_tpd/4*10:.1f} trades/day")
    print("="*84 + "\n")

"""
Sovereign Swarm — Multi-Timeframe Alignment Gate Comparison Backtest
Compares:
  Mode A: Strict 3-of-3 (1H AND 15m AND 5m)
  Mode B: 2-of-3 Majority (Any 2 of the 3 TFs agree)
  Mode C: HTF Anchored (1H Trend Bias AND (15m Confirmation OR 5m Trigger))

Runs across full real 1H + 15m + 5m historical datasets for BTC, ETH, SOL, XRP.
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

os.chdir(r"c:\Users\user\.gemini\antigravity\scratch\llama_trading_bot")

CFG = {
    "adx_min_trend":     20,
    "hurst_min_trend":   0.55,
    "sniper_vol_gate":   1.5,
    "berserker_veto":    0.72,
    "berserker_nn_conf": 0.65,
    "marl_trail":        3.0,
    "rsi_oversold":      30,
    "rsi_overbought":    70,
}

SYMBOLS = {
    "BTC/USDT": "BTC",
    "ETH/USDT": "ETH",
    "SOL/USDT": "SOL",
    "XRP/USDT": "XRP",
}

INITIAL_BALANCE  = 1000.0
RISK_PER_TRADE   = 0.01
LEVERAGE         = 5
COMMISSION_BPS   = 4
SLIPPAGE_BPS     = 2
COMMISSION       = (COMMISSION_BPS + SLIPPAGE_BPS) / 10000

def compute_indicators(df, label=""):
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

    # Bollinger Bands (20,2)
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

    # Hurst exponent (rolling 50-bar R/S)
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

def build_raw_signals_by_tf(df_1h, df_15m, df_5m, cfg):
    def htf_bias(row):
        bull = (row["ema_20"] > row["ema_50"] and
                row["ema_50"] > row["ema_200"] and
                row["adx"]    > cfg["adx_min_trend"] and
                row.get("hurst", 0) > cfg["hurst_min_trend"])
        bear = (row["ema_20"] < row["ema_50"] and
                row["ema_50"] < row["ema_200"] and
                row["adx"]    > cfg["adx_min_trend"] and
                row.get("hurst", 0) > cfg["hurst_min_trend"])
        if bull: return 1
        if bear: return -1
        return 0

    def mtf_confirm(row):
        long_conf  = (row["macd_hist"] > 0 and
                      row["rsi"] > 45 and row["rsi"] < 72 and
                      row["close"] > row["bb_mid"])
        short_conf = (row["macd_hist"] < 0 and
                      row["rsi"] > 28 and row["rsi"] < 55 and
                      row["close"] < row["bb_mid"])
        mr_long  = row["close"] < row["bb_lower"] and row["rsi"] < cfg["rsi_oversold"]
        mr_short = row["close"] > row["bb_upper"] and row["rsi"] > cfg["rsi_overbought"]
        if mr_long:  return 1
        if mr_short: return -1
        if long_conf:  return 1
        if short_conf: return -1
        return 0

    def trigger(row):
        fires_long  = (row["vol_ratio"] > cfg["sniper_vol_gate"] and
                       row["macd_hist"] > 0 and
                       row["close"] > row["ema_20"])
        fires_short = (row["vol_ratio"] > cfg["sniper_vol_gate"] and
                       row["macd_hist"] < 0 and
                       row["close"] < row["ema_20"])
        if fires_long:  return 1
        if fires_short: return -1
        return 0

    bias_1h  = df_1h.apply(htf_bias, axis=1).resample("5min").last().reindex(df_5m.index, method="ffill").fillna(0)
    conf_15m = df_15m.apply(mtf_confirm, axis=1).resample("5min").last().reindex(df_5m.index, method="ffill").fillna(0)
    trig_5m  = df_5m.apply(trigger, axis=1).fillna(0)

    return bias_1h, conf_15m, trig_5m

def combine_signals(bias_1h, conf_15m, trig_5m, mode="strict_3_of_3"):
    signal = pd.Series(0, index=trig_5m.index)

    if mode == "strict_3_of_3":
        signal[(bias_1h == 1)  & (conf_15m == 1)  & (trig_5m == 1)]  =  1
        signal[(bias_1h == -1) & (conf_15m == -1) & (trig_5m == -1)] = -1

    elif mode == "2_of_3_majority":
        # Score sum across timeframes
        score = bias_1h + conf_15m + trig_5m
        signal[score >= 2] =  1
        signal[score <= -2] = -1

    elif mode == "htf_anchored":
        # 1H Macro Trend must align, plus at least 15m OR 5m must trigger/confirm
        signal[(bias_1h == 1)  & ((conf_15m == 1)  | (trig_5m == 1))]  =  1
        signal[(bias_1h == -1) & ((conf_15m == -1) | (trig_5m == -1))] = -1

    # Shift 1 bar to eliminate lookahead bias (execute at next bar open)
    return signal.shift(1).fillna(0)

def run_backtest(df_5m, signals, cfg, symbol):
    balance    = INITIAL_BALANCE
    equity     = []
    trades     = []

    in_pos     = False
    side       = 0
    entry_px   = 0.0
    sl_px      = 0.0
    tp_px      = 0.0
    trail_px   = 0.0

    ATR_SL = 2.0
    ATR_TP = 3.5

    for i in range(len(df_5m)):
        row    = df_5m.iloc[i]
        price  = float(row["close"])
        atr    = float(row["atr"]) if not np.isnan(row["atr"]) else price * 0.005
        sig    = int(signals.iloc[i]) if i < len(signals) else 0
        trail_dist = atr * cfg["marl_trail"]

        # Position exit checking
        if in_pos:
            if side == 1:
                trail_px = max(trail_px, price - trail_dist)
                if price <= sl_px or price <= trail_px:
                    pnl = (price - entry_px) / entry_px * LEVERAGE * balance * RISK_PER_TRADE
                    pnl -= abs(pnl) * COMMISSION
                    balance += pnl
                    trades.append({"pnl": pnl, "exit": "sl", "side": "long"})
                    in_pos = False
                elif price >= tp_px:
                    pnl = (price - entry_px) / entry_px * LEVERAGE * balance * RISK_PER_TRADE
                    pnl -= abs(pnl) * COMMISSION
                    balance += pnl
                    trades.append({"pnl": pnl, "exit": "tp", "side": "long"})
                    in_pos = False
            else:
                trail_px = min(trail_px, price + trail_dist)
                if price >= sl_px or price >= trail_px:
                    pnl = (entry_px - price) / entry_px * LEVERAGE * balance * RISK_PER_TRADE
                    pnl -= abs(pnl) * COMMISSION
                    balance += pnl
                    trades.append({"pnl": pnl, "exit": "sl", "side": "short"})
                    in_pos = False
                elif price <= tp_px:
                    pnl = (entry_px - price) / entry_px * LEVERAGE * balance * RISK_PER_TRADE
                    pnl -= abs(pnl) * COMMISSION
                    balance += pnl
                    trades.append({"pnl": pnl, "exit": "tp", "side": "short"})
                    in_pos = False

        # Entry
        if not in_pos and sig != 0:
            side     = sig
            entry_px = price
            balance -= balance * RISK_PER_TRADE * COMMISSION
            if sig == 1:
                sl_px    = price - atr * ATR_SL
                tp_px    = price + atr * ATR_TP
                trail_px = price - trail_dist
            else:
                sl_px    = price + atr * ATR_SL
                tp_px    = price - atr * ATR_TP
                trail_px = price + trail_dist
            in_pos = True

        equity.append(balance)

    if in_pos:
        price = float(df_5m["close"].iloc[-1])
        if side == 1:
            pnl = (price - entry_px) / entry_px * LEVERAGE * INITIAL_BALANCE * RISK_PER_TRADE
        else:
            pnl = (entry_px - price) / entry_px * LEVERAGE * INITIAL_BALANCE * RISK_PER_TRADE
        balance += pnl
        trades.append({"pnl": pnl, "exit": "force_close", "side": "long" if side==1 else "short"})

    equity = np.array(equity)
    total_return = (equity[-1] - INITIAL_BALANCE) / INITIAL_BALANCE * 100

    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / np.where(running_max > 0, running_max, 1)
    max_dd = abs(dd.min()) * 100

    rets = pd.Series(equity).pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(105120)) if rets.std() > 0 else 0

    df_t = pd.DataFrame(trades) if trades else pd.DataFrame(columns=["pnl","exit","side"])
    n  = len(df_t)
    wr = (df_t["pnl"] > 0).mean() * 100 if n > 0 else 0
    aw = df_t[df_t["pnl"] > 0]["pnl"].mean() if n > 0 and (df_t["pnl"] > 0).any() else 0
    al = df_t[df_t["pnl"] < 0]["pnl"].mean() if n > 0 and (df_t["pnl"] < 0).any() else 0
    pf = abs(aw / al) if al != 0 else 999

    span_days = (df_5m.index[-1] - df_5m.index[0]).days or 1
    trades_per_day = n / span_days

    return {
        "symbol":          symbol,
        "days":            span_days,
        "final_balance":   equity[-1],
        "total_return":    total_return,
        "max_drawdown":    max_dd,
        "sharpe":          sharpe,
        "n_trades":        n,
        "trades_per_day":  trades_per_day,
        "win_rate":        wr,
        "profit_factor":   pf,
    }

if __name__ == "__main__":
    print("\n" + "="*76)
    print("   SOVEREIGN SWARM — MULTI-TIMEFRAME GATE ALIGNMENT COMPARISON")
    print("   Real 1H + 15m + 5m Tick-Accurate Binance Data (617 Days)")
    print("   1% Risk / Trade | 5x Leverage | $1,000 Starting Capital")
    print("="*76 + "\n")

    MODES = [
        ("strict_3_of_3",    "A. Strict 3-of-3 (1H AND 15m AND 5m)"),
        ("2_of_3_majority",  "B. 2-of-3 Majority (Any 2 TFs Agree)"),
        ("htf_anchored",     "C. HTF Anchored (1H Bias AND (15m OR 5m Trigger))"),
    ]

    # Preload and calculate indicators once per symbol
    prepared_data = {}
    for sym_display, sym_key in SYMBOLS.items():
        print(f"[*] Processing indicators for {sym_display}...", end=" ", flush=True)
        df_1h  = load_tf(sym_display, "1h")
        df_15m = load_tf(sym_display, "15m")
        df_5m  = load_tf(sym_display, "5m")

        start = max(df_1h.index[0], df_15m.index[0], df_5m.index[0])
        end   = min(df_1h.index[-1], df_15m.index[-1], df_5m.index[-1])
        df_1h  = df_1h[start:end]
        df_15m = df_15m[start:end]
        df_5m  = df_5m[start:end]

        ind_1h  = compute_indicators(df_1h, "1h")
        ind_15m = compute_indicators(df_15m, "15m")
        ind_5m  = compute_indicators(df_5m, "5m")

        bias_1h, conf_15m, trig_5m = build_raw_signals_by_tf(ind_1h, ind_15m, ind_5m, CFG)
        prepared_data[sym_display] = {
            "df_5m": ind_5m,
            "bias_1h": bias_1h,
            "conf_15m": conf_15m,
            "trig_5m": trig_5m,
        }
        print(f"done ({len(ind_5m):,} 5m bars)")

    print("\n" + "="*76)
    print("   RUNNING COMPARATIVE SIMULATIONS")
    print("="*76 + "\n")

    mode_summaries = {}

    for mode_key, mode_name in MODES:
        print(f"\n>>> TESTING: {mode_name}")
        mode_results = []
        for sym_display, data in prepared_data.items():
            signals = combine_signals(data["bias_1h"], data["conf_15m"], data["trig_5m"], mode=mode_key)
            res = run_backtest(data["df_5m"], signals, CFG, sym_display)
            mode_results.append(res)
            sign = "+" if res["total_return"] >= 0 else ""
            print(f"    {sym_display:<10} Return: {sign}{res['total_return']:>6.1f}% | "
                  f"Max DD: {res['max_drawdown']:>4.1f}% | "
                  f"Sharpe: {res['sharpe']:>5.2f} | "
                  f"Trades: {res['n_trades']:>4} ({res['trades_per_day']:.1f}/d) | "
                  f"WR: {res['win_rate']:>4.1f}% | "
                  f"PF: {res['profit_factor']:.2f}")

        avg_ret = sum(r["total_return"] for r in mode_results) / len(mode_results)
        avg_dd  = sum(r["max_drawdown"] for r in mode_results) / len(mode_results)
        avg_sh  = sum(r["sharpe"] for r in mode_results) / len(mode_results)
        tot_tpd = sum(r["trades_per_day"] for r in mode_results)
        avg_wr  = sum(r["win_rate"] for r in mode_results) / len(mode_results)
        avg_pf  = sum(r["profit_factor"] for r in mode_results) / len(mode_results)
        tot_val = sum(r["final_balance"] for r in mode_results)

        mode_summaries[mode_name] = {
            "avg_ret": avg_ret,
            "avg_dd":  avg_dd,
            "avg_sh":  avg_sh,
            "tot_tpd": tot_tpd,
            "live_10_tpd": tot_tpd / 4 * 10,
            "avg_wr":  avg_wr,
            "avg_pf":  avg_pf,
            "tot_val": tot_val,
        }

    print("\n\n" + "="*84)
    print("   HEAD-TO-HEAD MTF ALIGNMENT MODE COMPARISON SUMMARY")
    print("="*84)
    print(f"{'Alignment Strategy':<38} {'Avg Ret':>9} {'Max DD':>8} {'Sharpe':>8} {'4P Tr/d':>9} {'10P Tr/d':>10} {'Avg PF':>8}")
    print("-"*84)
    for name, s in mode_summaries.items():
        sign = "+" if s["avg_ret"] >= 0 else ""
        print(f"{name:<38} "
              f"{sign}{s['avg_ret']:>8.1f}% "
              f"{s['avg_dd']:>7.1f}% "
              f"{s['avg_sh']:>8.2f} "
              f"{s['tot_tpd']:>8.1f}/d "
              f"{s['live_10_tpd']:>9.1f}/d "
              f"{s['avg_pf']:>8.2f}")
    print("="*84 + "\n")

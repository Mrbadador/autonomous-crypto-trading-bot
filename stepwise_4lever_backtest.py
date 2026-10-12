"""
Sovereign Swarm — Multi-Lever Stepwise Optimization Backtest
Tests:
  1. Baseline Mode C (HTF Anchored)
  2. Mode C + Lever 1 (Dynamic 1H Chop Filter: ADX > 25, Hurst > 0.53)
  3. Mode C + Lever 1 & 2 (Chop Filter + Asymmetric TP1 @ 1.5R & Trailing Runner)
  4. Mode C + Full 4 Levers (Chop Filter + TP1/Runner + Dynamic Sharpe Weighting + Maker Fees)

Runs on real 617-day 1H+15m+5m Binance tick datasets.
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
BASE_RISK       = 0.01  # 1% risk per trade
LEVERAGE        = 5

TAKER_FEE = (4.0 + 2.0) / 10000.0  # 6 bps total taker friction
MAKER_FEE = (2.0 + 0.5) / 10000.0  # 2.5 bps total maker friction


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

    # Hurst Exponent (50-bar rolling)
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


def generate_signals(df_1h, df_15m, df_5m, use_chop_filter=False):
    adx_thresh   = 25.0 if use_chop_filter else 20.0
    hurst_thresh = 0.53 if use_chop_filter else 0.50

    def htf_bias(row):
        if use_chop_filter:
            if row["adx"] < adx_thresh or row.get("hurst", 0) < hurst_thresh:
                return 0
        else:
            if row["adx"] < adx_thresh:
                return 0

        bull = (row["ema_20"] > row["ema_50"]) and (row["ema_50"] > row["ema_200"])
        bear = (row["ema_20"] < row["ema_50"]) and (row["ema_50"] < row["ema_200"])
        if bull: return 1
        if bear: return -1
        return 0

    def mtf_confirm(row):
        long_conf  = (row["macd_hist"] > 0 and row["rsi"] > 45 and row["rsi"] < 70 and row["close"] > row["bb_mid"])
        short_conf = (row["macd_hist"] < 0 and row["rsi"] > 30 and row["rsi"] < 55 and row["close"] < row["bb_mid"])
        if long_conf:  return 1
        if short_conf: return -1
        return 0

    def trigger(row):
        fires_long  = (row["vol_ratio"] > 1.5 and row["macd_hist"] > 0 and row["close"] > row["ema_20"])
        fires_short = (row["vol_ratio"] > 1.5 and row["macd_hist"] < 0 and row["close"] < row["ema_20"])
        if fires_long:  return 1
        if fires_short: return -1
        return 0

    bias_1h  = df_1h.apply(htf_bias, axis=1).resample("5min").last().reindex(df_5m.index, method="ffill").fillna(0)
    conf_15m = df_15m.apply(mtf_confirm, axis=1).resample("5min").last().reindex(df_5m.index, method="ffill").fillna(0)
    trig_5m  = df_5m.apply(trigger, axis=1).fillna(0)

    # Mode C: 1H Bias AND (15m Conf OR 5m Trig)
    signal = pd.Series(0, index=df_5m.index)
    signal[(bias_1h == 1)  & ((conf_15m == 1)  | (trig_5m == 1))]  =  1
    signal[(bias_1h == -1) & ((conf_15m == -1) | (trig_5m == -1))] = -1

    return signal.shift(1).fillna(0)


def simulate_trading(df_5m, signals, symbol, use_asymmetric_exit=False, use_dynamic_sizing=False, use_maker_fee=False):
    balance   = INITIAL_BALANCE
    equity    = []
    trades    = []

    in_pos     = False
    side       = 0
    entry_px   = 0.0
    qty        = 0.0
    sl_px      = 0.0
    tp1_px     = 0.0
    tp1_hit    = False
    trail_dist = 0.0

    recent_pnl = []

    fee_entry = MAKER_FEE if use_maker_fee else TAKER_FEE
    fee_tp    = MAKER_FEE if use_maker_fee else TAKER_FEE
    fee_sl    = TAKER_FEE

    for i in range(len(df_5m)):
        row   = df_5m.iloc[i]
        price = float(row["close"])
        high  = float(row["high"])
        low   = float(row["low"])
        atr   = float(row["atr"]) if not np.isnan(row["atr"]) else price * 0.005
        sig   = int(signals.iloc[i]) if i < len(signals) else 0

        # Lever 3: Dynamic Sizing
        if use_dynamic_sizing and len(recent_pnl) >= 20:
            arr = np.array(recent_pnl[-30:])
            std = arr.std()
            if std > 0:
                sh = arr.mean() / std
                weight = 1.5 if sh > 0.5 else (0.5 if sh < -0.2 else 1.0)
            else:
                weight = 1.0
        else:
            weight = 1.0

        current_risk_usd = balance * BASE_RISK * weight

        # ── MANAGE ACTIVE POSITION
        if in_pos:
            stop_distance = atr * 2.0
            trail_distance = atr * 3.0

            if side == 1:  # LONG
                if not use_asymmetric_exit:
                    # Single-target standard trail
                    sl_px = max(sl_px, price - trail_distance)
                    tp_target = entry_px + (stop_distance * 1.75)
                    if low <= sl_px:
                        pnl = qty * (sl_px - entry_px) - (qty * sl_px * fee_sl)
                        balance += pnl
                        trades.append({"pnl": pnl, "exit": "sl"})
                        recent_pnl.append(pnl)
                        in_pos = False
                    elif high >= tp_target:
                        pnl = qty * (tp_target - entry_px) - (qty * tp_target * fee_tp)
                        balance += pnl
                        trades.append({"pnl": pnl, "exit": "tp"})
                        recent_pnl.append(pnl)
                        in_pos = False
                else:
                    # Asymmetric TP1 + Runner
                    if not tp1_hit:
                        if low <= sl_px:
                            # Full SL hit
                            pnl = qty * (sl_px - entry_px) - (qty * sl_px * fee_sl)
                            balance += pnl
                            trades.append({"pnl": pnl, "exit": "full_sl"})
                            recent_pnl.append(pnl)
                            in_pos = False
                        elif high >= tp1_px:
                            # 50% TP1 hit @ +1.5R
                            tp1_hit = True
                            qty_half = qty * 0.5
                            pnl_tp1 = qty_half * (tp1_px - entry_px) - (qty_half * tp1_px * fee_tp)
                            balance += pnl_tp1
                            trades.append({"pnl": pnl_tp1, "exit": "tp1"})
                            recent_pnl.append(pnl_tp1)
                            qty = qty_half  # remaining 50%
                            sl_px = entry_px * 1.0005  # Breakeven + fee buffer
                    else:
                        # Runner trailing
                        sl_px = max(sl_px, price - trail_distance)
                        if low <= sl_px:
                            pnl_run = qty * (sl_px - entry_px) - (qty * sl_px * fee_sl)
                            balance += pnl_run
                            trades.append({"pnl": pnl_run, "exit": "runner"})
                            recent_pnl.append(pnl_run)
                            in_pos = False

            else:  # SHORT
                if not use_asymmetric_exit:
                    sl_px = min(sl_px, price + trail_distance)
                    tp_target = entry_px - (stop_distance * 1.75)
                    if high >= sl_px:
                        pnl = qty * (entry_px - sl_px) - (qty * sl_px * fee_sl)
                        balance += pnl
                        trades.append({"pnl": pnl, "exit": "sl"})
                        recent_pnl.append(pnl)
                        in_pos = False
                    elif low <= tp_target:
                        pnl = qty * (entry_px - tp_target) - (qty * tp_target * fee_tp)
                        balance += pnl
                        trades.append({"pnl": pnl, "exit": "tp"})
                        recent_pnl.append(pnl)
                        in_pos = False
                else:
                    if not tp1_hit:
                        if high >= sl_px:
                            pnl = qty * (entry_px - sl_px) - (qty * sl_px * fee_sl)
                            balance += pnl
                            trades.append({"pnl": pnl, "exit": "full_sl"})
                            recent_pnl.append(pnl)
                            in_pos = False
                        elif low <= tp1_px:
                            tp1_hit = True
                            qty_half = qty * 0.5
                            pnl_tp1 = qty_half * (entry_px - tp1_px) - (qty_half * tp1_px * fee_tp)
                            balance += pnl_tp1
                            trades.append({"pnl": pnl_tp1, "exit": "tp1"})
                            recent_pnl.append(pnl_tp1)
                            qty = qty_half
                            sl_px = entry_px * 0.9995
                    else:
                        sl_px = min(sl_px, price + trail_distance)
                        if high >= sl_px:
                            pnl_run = qty * (entry_px - sl_px) - (qty * sl_px * fee_sl)
                            balance += pnl_run
                            trades.append({"pnl": pnl_run, "exit": "runner"})
                            recent_pnl.append(pnl_run)
                            in_pos = False

        # ── ENTRY
        if not in_pos and sig != 0:
            side     = sig
            entry_px = price
            stop_dist = max(atr * 2.0, entry_px * 0.003)
            qty      = (current_risk_usd / stop_dist)
            tp1_hit  = False

            # Entry fee
            balance -= (qty * entry_px * fee_entry)

            if sig == 1:
                sl_px  = entry_px - stop_dist
                tp1_px = entry_px + (stop_dist * 1.5)
            else:
                sl_px  = entry_px + stop_dist
                tp1_px = entry_px - (stop_dist * 1.5)
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
    trades_per_day = n / span_days

    return {
        "symbol":          symbol,
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
    print("\n" + "="*84)
    print("   SOVEREIGN SWARM — 4-LEVER STEPWISE OPTIMIZATION BACKTEST")
    print("   Testing Incremental Alpha of Each Institutional Lever (617 Days)")
    print("="*84 + "\n")

    # Step 1: Pre-load and compute indicators once
    data_cache = {}
    for sym_display, sym_key in SYMBOLS.items():
        print(f"[*] Caching {sym_display}...", end=" ", flush=True)
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

        sig_raw     = generate_signals(ind_1h, ind_15m, ind_5m, use_chop_filter=False)
        sig_filtered = generate_signals(ind_1h, ind_15m, ind_5m, use_chop_filter=True)

        data_cache[sym_display] = {
            "df_5m": ind_5m,
            "sig_raw": sig_raw,
            "sig_filtered": sig_filtered,
        }
        print("done")

    EXPERIMENTS = [
        ("1. Baseline Mode C",                 False, False, False, False),
        ("2. + Lever 1 (1H Chop Filter)",       True,  False, False, False),
        ("3. + Lever 2 (Asymmetric TP/Runner)", True,  True,  False, False),
        ("4. + Lever 3 & 4 (Full 4-Levers)",    True,  True,  True,  True),
    ]

    exp_summaries = {}

    for exp_name, use_chop, use_asym, use_dyn, use_mkr in EXPERIMENTS:
        print(f"\n>>> Running: {exp_name}")
        exp_results = []
        for sym_display, d in data_cache.items():
            sig = d["sig_filtered"] if use_chop else d["sig_raw"]
            res = simulate_trading(d["df_5m"], sig, sym_display,
                                   use_asymmetric_exit=use_asym,
                                   use_dynamic_sizing=use_dyn,
                                   use_maker_fee=use_mkr)
            exp_results.append(res)
            sign = "+" if res["total_return"] >= 0 else ""
            print(f"    {sym_display:<10} Return: {sign}{res['total_return']:>6.1f}% | "
                  f"Max DD: {res['max_drawdown']:>4.1f}% | "
                  f"Sharpe: {res['sharpe']:>5.2f} | "
                  f"Tr/d: {res['trades_per_day']:>4.1f} | "
                  f"WR: {res['win_rate']:>4.1f}% | "
                  f"PF: {res['profit_factor']:.2f}")

        avg_ret = sum(r["total_return"] for r in exp_results) / len(exp_results)
        avg_dd  = sum(r["max_drawdown"] for r in exp_results) / len(exp_results)
        avg_sh  = sum(r["sharpe"] for r in exp_results) / len(exp_results)
        tot_tpd = sum(r["trades_per_day"] for r in exp_results)
        avg_wr  = sum(r["win_rate"] for r in exp_results) / len(exp_results)
        avg_pf  = sum(r["profit_factor"] for r in exp_results) / len(exp_results)
        tot_val = sum(r["final_balance"] for r in exp_results)

        exp_summaries[exp_name] = {
            "avg_ret": avg_ret,
            "avg_dd":  avg_dd,
            "avg_sh":  avg_sh,
            "tot_tpd": tot_tpd,
            "live_10": tot_tpd / 4 * 10,
            "avg_wr":  avg_wr,
            "avg_pf":  avg_pf,
            "tot_val": tot_val,
        }

    print("\n\n" + "="*88)
    print("   INCREMENTAL ALPHA COMPARISON: THE 4 LEVERS IN ACTION")
    print("="*88)
    print(f"{'Optimization Configuration':<40} {'Avg Ret':>9} {'Max DD':>8} {'Sharpe':>8} {'Tr/d(10P)':>10} {'Win Rate':>10} {'PF':>7}")
    print("-"*88)
    for name, s in exp_summaries.items():
        sign = "+" if s["avg_ret"] >= 0 else ""
        print(f"{name:<40} "
              f"{sign}{s['avg_ret']:>8.1f}% "
              f"{s['avg_dd']:>7.1f}% "
              f"{s['avg_sh']:>8.2f} "
              f"{s['live_10']:>9.1f}/d "
              f"{s['avg_wr']:>9.1f}% "
              f"{s['avg_pf']:>7.2f}")
    print("="*88 + "\n")

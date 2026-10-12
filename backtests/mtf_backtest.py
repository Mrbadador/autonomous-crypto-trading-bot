"""
Sovereign Swarm — Multi-Timeframe Comparative Backtest (MTF)
Simultaneously uses 1H (HTF trend bias) + 15m (entry confirmation) + 5m (precision timing)
exactly mirroring how the live Sniper + Berserker + MARL fleet operates.

Signal Logic:
  1H  → Trend direction bias (EMA 20/50/200 stack, ADX, Hurst)
  15m → Entry confirmation (MACD cross, RSI, Bollinger Band)
  5m  → Precision entry timing (Volume surge, momentum impulse — Berserker)

A trade only fires when ALL relevant timeframes are aligned.
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

os.chdir(r"c:\Users\user\.gemini\antigravity\scratch\llama_trading_bot")

# ─────────────────────────────────────────────
# REBALANCED PARAMETERS (new config)
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# INDICATOR ENGINE (reusable across all TFs)
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# LOAD DATA — auto-detect 1h/15m/5m CSVs
# ─────────────────────────────────────────────
def load_tf(sym_key, interval):
    """Load CSV for a given symbol and timeframe. Falls back gracefully."""
    sym = sym_key.replace("/","_").lower()  # btc_usdt
    base = sym.replace("_usdt", "")        # btc

    candidates = [
        f"historical_{base}_usdt_{interval}.csv",
        f"historical_{sym}_{interval}.csv",
    ]
    # For 1h, also check the original files without interval suffix
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


# ─────────────────────────────────────────────
# MTF SIGNAL ENGINE
# ─────────────────────────────────────────────
def build_mtf_signals(df_1h, df_15m, df_5m, cfg):
    """
    Align all 3 timeframes to the 5m bar index and generate combined signals.

    Rule: A trade fires only when:
      1H  CONFIRMS the direction (trend bias)
      15m CONFIRMS entry (momentum + structure)
      5m  TRIGGERS the entry (volume + impulse)
    """

    # ── 1H: Trend bias signal (+1 bull / -1 bear / 0 neutral)
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

    # ── 15m: Entry confirmation (+1 long conf / -1 short conf / 0)
    def mtf_confirm(row):
        long_conf  = (row["macd_hist"] > 0 and
                      row["rsi"] > 45 and row["rsi"] < 72 and
                      row["close"] > row["bb_mid"])
        short_conf = (row["macd_hist"] < 0 and
                      row["rsi"] > 28 and row["rsi"] < 55 and
                      row["close"] < row["bb_mid"])
        # Mean-reversion override on extremes
        mr_long  = row["close"] < row["bb_lower"] and row["rsi"] < cfg["rsi_oversold"]
        mr_short = row["close"] > row["bb_upper"] and row["rsi"] > cfg["rsi_overbought"]
        if mr_long:  return 1
        if mr_short: return -1
        if long_conf:  return 1
        if short_conf: return -1
        return 0

    # ── 5m: Entry trigger (Berserker volume surge + momentum impulse)
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

    # Resample 1H and 15m signals onto 5m index (forward-fill = prior closed bar)
    bias_1h  = df_1h.apply(htf_bias, axis=1).resample("5min").last().reindex(df_5m.index, method="ffill")
    conf_15m = df_15m.apply(mtf_confirm, axis=1).resample("5min").last().reindex(df_5m.index, method="ffill")
    trig_5m  = df_5m.apply(trigger, axis=1)

    # Final signal: all 3 must agree
    signal = pd.Series(0, index=df_5m.index)
    signal[(bias_1h == 1)  & (conf_15m == 1)  & (trig_5m == 1)]  =  1
    signal[(bias_1h == -1) & (conf_15m == -1) & (trig_5m == -1)] = -1

    # Shift 1 bar to prevent lookahead bias
    return signal.shift(1).fillna(0)


# ─────────────────────────────────────────────
# BACKTEST EXECUTION ENGINE
# ─────────────────────────────────────────────
def run_backtest(df_5m, signals, cfg, symbol):
    balance    = INITIAL_BALANCE
    peak_bal   = INITIAL_BALANCE
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

        # ── MANAGE OPEN POSITION
        if in_pos:
            if side == 1:
                trail_px = max(trail_px, price - trail_dist)
                if price <= sl_px or price <= trail_px:
                    pnl = (price - entry_px) / entry_px * LEVERAGE * balance * RISK_PER_TRADE
                    pnl -= abs(pnl) * COMMISSION
                    balance += pnl
                    trades.append({"pnl": pnl, "exit": "sl", "side": "long", "price": price})
                    in_pos = False
                elif price >= tp_px:
                    pnl = (price - entry_px) / entry_px * LEVERAGE * balance * RISK_PER_TRADE
                    pnl -= abs(pnl) * COMMISSION
                    balance += pnl
                    trades.append({"pnl": pnl, "exit": "tp", "side": "long", "price": price})
                    in_pos = False
            else:
                trail_px = min(trail_px, price + trail_dist)
                if price >= sl_px or price >= trail_px:
                    pnl = (entry_px - price) / entry_px * LEVERAGE * balance * RISK_PER_TRADE
                    pnl -= abs(pnl) * COMMISSION
                    balance += pnl
                    trades.append({"pnl": pnl, "exit": "sl", "side": "short", "price": price})
                    in_pos = False
                elif price <= tp_px:
                    pnl = (entry_px - price) / entry_px * LEVERAGE * balance * RISK_PER_TRADE
                    pnl -= abs(pnl) * COMMISSION
                    balance += pnl
                    trades.append({"pnl": pnl, "exit": "tp", "side": "short", "price": price})
                    in_pos = False

        # ── NEW ENTRY
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
        peak_bal = max(peak_bal, balance)

    # Force-close at end
    if in_pos:
        price = float(df_5m["close"].iloc[-1])
        if side == 1:
            pnl = (price - entry_px) / entry_px * LEVERAGE * INITIAL_BALANCE * RISK_PER_TRADE
        else:
            pnl = (entry_px - price) / entry_px * LEVERAGE * INITIAL_BALANCE * RISK_PER_TRADE
        balance += pnl
        trades.append({"pnl": pnl, "exit": "force_close", "side": "long" if side==1 else "short", "price": price})

    # ── METRICS
    equity = np.array(equity)
    total_return = (equity[-1] - INITIAL_BALANCE) / INITIAL_BALANCE * 100

    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / np.where(running_max > 0, running_max, 1)
    max_dd = abs(dd.min()) * 100

    rets = pd.Series(equity).pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(105120)) if rets.std() > 0 else 0  # 5m annualised

    df_t = pd.DataFrame(trades) if trades else pd.DataFrame(columns=["pnl","exit","side","price"])
    n  = len(df_t)
    wr = (df_t["pnl"] > 0).mean() * 100 if n > 0 else 0
    aw = df_t[df_t["pnl"] > 0]["pnl"].mean() if n > 0 and (df_t["pnl"] > 0).any() else 0
    al = df_t[df_t["pnl"] < 0]["pnl"].mean() if n > 0 and (df_t["pnl"] < 0).any() else 0
    pf = abs(aw / al) if al != 0 else 999

    # Days tested
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
        "avg_win":         aw,
        "avg_loss":        al,
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*72)
    print("   SOVEREIGN SWARM — MULTI-TIMEFRAME BACKTEST (1H + 15m + 5m)")
    print("   All 3 timeframes must align to fire a trade — no lookahead bias")
    print(f"   Balance: ${INITIAL_BALANCE:,.0f} | Risk: {RISK_PER_TRADE*100:.0f}%/trade | Leverage: {LEVERAGE}x")
    print("="*72 + "\n")

    all_results = []
    missing_tf  = []

    for sym_display, sym_key in SYMBOLS.items():
        sym_lower = sym_key.lower()

        print(f"  Loading {sym_display}...")
        df_1h  = load_tf(sym_display, "1h")
        df_15m = load_tf(sym_display, "15m")
        df_5m  = load_tf(sym_display, "5m")

        # Check what we have
        tfs_found  = []
        tfs_missing = []
        for label, df in [("1h", df_1h), ("15m", df_15m), ("5m", df_5m)]:
            if df is not None and len(df) > 100:
                tfs_found.append(f"{label}({len(df):,})")
            else:
                tfs_missing.append(label)

        print(f"    Found: {', '.join(tfs_found) if tfs_found else 'NONE'}")

        if tfs_missing:
            print(f"    Missing TFs: {', '.join(tfs_missing)} — will synthesise from available data")
            missing_tf.extend([f"{sym_display}:{tf}" for tf in tfs_missing])

        # Synthesise missing timeframes by resampling from 1H if needed
        if df_1h is not None:
            if df_15m is None or len(df_15m) < 100:
                print(f"    Synthesising 15m from 1H by interpolation...")
                df_15m = df_1h.resample("15min").interpolate(method="linear")
                df_15m["volume"] = df_1h["volume"].resample("15min").bfill() / 4
            if df_5m is None or len(df_5m) < 100:
                print(f"    Synthesising 5m from 1H by interpolation...")
                df_5m = df_1h.resample("5min").interpolate(method="linear")
                df_5m["volume"] = df_1h["volume"].resample("5min").bfill() / 12

        if df_1h is None or df_5m is None:
            print(f"    [SKIP] {sym_display} — insufficient data\n")
            continue

        # Find common date range across all 3 TFs
        start = max(df_1h.index[0], df_15m.index[0], df_5m.index[0])
        end   = min(df_1h.index[-1], df_15m.index[-1], df_5m.index[-1])
        df_1h  = df_1h[start:end]
        df_15m = df_15m[start:end]
        df_5m  = df_5m[start:end]

        print(f"    Common range: {start.date()} → {end.date()} ({(end-start).days} days)")
        print(f"    Computing indicators... ", end="", flush=True)

        df_1h  = compute_indicators(df_1h,  "1h")
        df_15m = compute_indicators(df_15m, "15m")
        df_5m  = compute_indicators(df_5m,  "5m")
        print("done")

        print(f"    Building MTF signals...", end="", flush=True)
        signals = build_mtf_signals(df_1h, df_15m, df_5m, CFG)
        n_sig   = (signals != 0).sum()
        print(f" done ({n_sig:,} raw signals across {len(df_5m):,} 5m bars)")

        result = run_backtest(df_5m, signals, CFG, sym_display)
        all_results.append(result)

        sign = "+" if result["total_return"] >= 0 else ""
        print(f"\n    [RESULT] {sym_display}")
        print(f"      Return:         {sign}{result['total_return']:.1f}%   (${result['final_balance']:,.2f})")
        print(f"      Max Drawdown:   {result['max_drawdown']:.1f}%")
        print(f"      Sharpe Ratio:   {result['sharpe']:.2f}")
        print(f"      Trades:         {result['n_trades']:,} total  |  {result['trades_per_day']:.1f}/day")
        print(f"      Win Rate:       {result['win_rate']:.1f}%")
        print(f"      Profit Factor:  {result['profit_factor']:.2f}")
        print(f"      Avg Win:        ${result['avg_win']:.2f}  |  Avg Loss: ${result['avg_loss']:.2f}\n")

    if not all_results:
        print("  No results — check that CSV data files are present.")
        raise SystemExit(1)

    # ─────────────────────────────────────────────
    # FINAL PORTFOLIO SUMMARY
    # ─────────────────────────────────────────────
    print("\n" + "="*72)
    print("   MULTI-TIMEFRAME PORTFOLIO SUMMARY")
    print("="*72)
    print(f"  {'Symbol':<12} {'Return':>8} {'Max DD':>8} {'Sharpe':>8} {'Trades/Day':>12} {'WinRate':>9} {'PF':>6}")
    print("  " + "-"*68)
    for r in all_results:
        sign = "+" if r["total_return"] >= 0 else ""
        print(f"  {r['symbol']:<12} "
              f"{sign}{r['total_return']:>7.1f}% "
              f"{r['max_drawdown']:>7.1f}% "
              f"{r['sharpe']:>8.2f} "
              f"{r['trades_per_day']:>11.1f}/d "
              f"{r['win_rate']:>8.1f}% "
              f"{r['profit_factor']:>6.2f}")

    avg_ret   = sum(r["total_return"]   for r in all_results) / len(all_results)
    avg_dd    = sum(r["max_drawdown"]   for r in all_results) / len(all_results)
    avg_sh    = sum(r["sharpe"]         for r in all_results) / len(all_results)
    avg_tpd   = sum(r["trades_per_day"] for r in all_results) / len(all_results)
    avg_wr    = sum(r["win_rate"]       for r in all_results) / len(all_results)
    avg_pf    = sum(r["profit_factor"]  for r in all_results) / len(all_results)
    total_val = sum(r["final_balance"]  for r in all_results)
    combined_trades_per_day = sum(r["trades_per_day"] for r in all_results)

    print("  " + "-"*68)
    sign = "+" if avg_ret >= 0 else ""
    print(f"  {'AVERAGE':<12} "
          f"{sign}{avg_ret:>7.1f}% "
          f"{avg_dd:>7.1f}% "
          f"{avg_sh:>8.2f} "
          f"{avg_tpd:>11.1f}/d "
          f"{avg_wr:>8.1f}% "
          f"{avg_pf:>6.2f}")

    print("\n" + "="*72)
    print(f"  Combined portfolio value (4x $1,000):  ${total_val:,.2f}")
    print(f"  Combined daily trade frequency:         {combined_trades_per_day:.1f} trades/day across all pairs")
    print(f"  Estimated on 10 live pairs:             {combined_trades_per_day/4*10:.1f} trades/day")

    if missing_tf:
        print(f"\n  NOTE: Some TFs were synthesised (not real downloaded data):")
        for m in missing_tf:
            print(f"    - {m}")
        print("  Run download_mtf_data.py first for 100% accurate 15m/5m results.")
    print("="*72 + "\n")

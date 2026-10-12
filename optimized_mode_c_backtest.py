"""
Sovereign Swarm — Optimized Mode C+ Institutional Quant Engine
Incorporates 4 Levers:
  1. Dynamic Regime Gating (ADX > 25 & Hurst > 0.53 on 1H)
  2. Asymmetric Two-Tier Exits (50% TP1 @ 1.5R + Breakeven SL, 50% Runner on 15m EMA20)
  3. Dynamic Rolling Performance Asset Weighting (0.5x to 1.5x risk scaling)
  4. Maker Fee & Mid-Spread Execution Model
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

os.chdir(r"c:\Users\user\.gemini\antigravity\scratch\llama_trading_bot")

CFG_BASE = {
    "adx_min_trend":     25.0,  # Lever 1: Raised from 20 -> 25 to kill flat chop
    "hurst_min_trend":   0.53,  # Lever 1: Strict trend persistence gate
    "sniper_vol_gate":   1.5,
    "rsi_oversold":      30,
    "rsi_overbought":    70,
}

SYMBOLS = {
    "BTC/USDT": "BTC",
    "ETH/USDT": "ETH",
    "SOL/USDT": "SOL",
    "XRP/USDT": "XRP",
}

INITIAL_BALANCE = 1000.0
BASE_RISK       = 0.01  # 1% base risk per trade
LEVERAGE        = 5

# Lever 4: Passive Maker Fee (0.02%) + near-zero execution slippage (0.5 bps)
MAKER_FEE_BPS   = 2.0
MAKER_SLIPPAGE  = 0.5
COMMISSION_MAKER = (MAKER_FEE_BPS + MAKER_SLIPPAGE) / 10000.0

# Standard Taker fee for stop-loss hits
COMMISSION_TAKER = (4.0 + 2.0) / 10000.0


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

    # Hurst exponent (50-bar rolling)
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


def generate_optimized_signals(df_1h, df_15m, df_5m, cfg):
    """
    Lever 1: Dynamic 1H Chop Filter
    1H must have clear trend (ADX > 25, Hurst > 0.53, EMA Stack aligned).
    """
    def htf_bias(row):
        is_trending = (row["adx"] >= cfg["adx_min_trend"]) and (row.get("hurst", 0) >= cfg["hurst_min_trend"])
        if not is_trending:
            return 0  # FROZEN: Chop detected on 1H
        
        bull = (row["ema_20"] > row["ema_50"]) and (row["ema_50"] > row["ema_200"])
        bear = (row["ema_20"] < row["ema_50"]) and (row["ema_50"] < row["ema_200"])
        if bull: return 1
        if bear: return -1
        return 0

    def mtf_confirm(row):
        long_conf  = (row["macd_hist"] > 0 and
                      row["rsi"] > 45 and row["rsi"] < 70 and
                      row["close"] > row["bb_mid"])
        short_conf = (row["macd_hist"] < 0 and
                      row["rsi"] > 30 and row["rsi"] < 55 and
                      row["close"] < row["bb_mid"])
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

    # Mode C Logic: 1H Trend Anchored AND (15m OR 5m trigger)
    signal = pd.Series(0, index=df_5m.index)
    signal[(bias_1h == 1)  & ((conf_15m == 1)  | (trig_5m == 1))]  =  1
    signal[(bias_1h == -1) & ((conf_15m == -1) | (trig_5m == -1))] = -1

    return signal.shift(1).fillna(0), df_15m["ema_20"].resample("5min").last().reindex(df_5m.index, method="ffill")


def run_advanced_simulation(df_5m, signals, ema20_15m, symbol):
    """
    Simulates:
      - Lever 2: Asymmetric 2-tier scale-out (50% TP1 @ 1.5R + Breakeven SL, 50% Runner on 15m EMA20)
      - Lever 3: Dynamic rolling asset Sharpe risk weighting (0.5x - 1.5x)
      - Lever 4: Maker entry execution fees
    """
    balance  = INITIAL_BALANCE
    equity   = []
    trades   = []

    # Position State
    in_pos     = False
    side       = 0
    entry_px   = 0.0
    initial_r  = 0.0
    sl_px      = 0.0
    tp1_px     = 0.0
    tp1_hit    = False
    position_size_weight = 1.0

    # 30-day rolling performance buffer for Lever 3
    recent_trade_pnl = []

    for i in range(len(df_5m)):
        row      = df_5m.iloc[i]
        price    = float(row["close"])
        high     = float(row["high"])
        low      = float(row["low"])
        atr      = float(row["atr"]) if not np.isnan(row["atr"]) else price * 0.005
        ema15m   = float(ema20_15m.iloc[i]) if not np.isnan(ema20_15m.iloc[i]) else price
        sig      = int(signals.iloc[i]) if i < len(signals) else 0

        # Lever 3: Dynamic Risk Weighting based on rolling 30-trade Sharpe
        if len(recent_trade_pnl) >= 20:
            arr = np.array(recent_trade_pnl[-30:])
            std = arr.std()
            if std > 0:
                sharpe_proxy = arr.mean() / std
                if sharpe_proxy > 0.8:
                    position_size_weight = 1.5   # 1.5x boost for hot assets
                elif sharpe_proxy < 0.0:
                    position_size_weight = 0.5   # 0.5x throttle for lagging assets
                else:
                    position_size_weight = 1.0
        else:
            position_size_weight = 1.0

        effective_risk = BASE_RISK * position_size_weight

        # ── MANAGE ACTIVE POSITION
        if in_pos:
            if side == 1:  # LONG
                # Check initial SL before TP1
                if not tp1_hit:
                    if low <= sl_px:
                        # Full stop-loss hit
                        pnl = -effective_risk * balance
                        pnl -= abs(pnl) * COMMISSION_TAKER
                        balance += pnl
                        trades.append({"pnl": pnl, "exit": "full_sl", "side": "long"})
                        recent_trade_pnl.append(pnl)
                        in_pos = False
                    elif high >= tp1_px:
                        # LEVER 2: TP1 HIT (Take 50% profit @ +1.5R)
                        tp1_hit = True
                        pnl_tp1 = 0.5 * (effective_risk * balance * 1.5)
                        pnl_tp1 -= abs(pnl_tp1) * COMMISSION_MAKER
                        balance += pnl_tp1
                        trades.append({"pnl": pnl_tp1, "exit": "tp1_half", "side": "long"})
                        recent_trade_pnl.append(pnl_tp1)
                        # MOVE SL TO BREAK-EVEN + FEES
                        sl_px = entry_px * 1.001
                else:
                    # Runner active (50% position): Trail with 15m EMA20 or Breakeven SL
                    trailing_sl = max(sl_px, ema15m)
                    if low <= trailing_sl:
                        pnl_runner = 0.5 * ((price - entry_px) / entry_px * LEVERAGE * balance * effective_risk)
                        pnl_runner -= abs(pnl_runner) * COMMISSION_TAKER
                        balance += pnl_runner
                        trades.append({"pnl": pnl_runner, "exit": "runner_trail", "side": "long"})
                        recent_trade_pnl.append(pnl_runner)
                        in_pos = False

            else:  # SHORT
                if not tp1_hit:
                    if high >= sl_px:
                        pnl = -effective_risk * balance
                        pnl -= abs(pnl) * COMMISSION_TAKER
                        balance += pnl
                        trades.append({"pnl": pnl, "exit": "full_sl", "side": "short"})
                        recent_trade_pnl.append(pnl)
                        in_pos = False
                    elif low <= tp1_px:
                        tp1_hit = True
                        pnl_tp1 = 0.5 * (effective_risk * balance * 1.5)
                        pnl_tp1 -= abs(pnl_tp1) * COMMISSION_MAKER
                        balance += pnl_tp1
                        trades.append({"pnl": pnl_tp1, "exit": "tp1_half", "side": "short"})
                        recent_trade_pnl.append(pnl_tp1)
                        sl_px = entry_px * 0.999
                else:
                    trailing_sl = min(sl_px, ema15m)
                    if high >= trailing_sl:
                        pnl_runner = 0.5 * ((entry_px - price) / entry_px * LEVERAGE * balance * effective_risk)
                        pnl_runner -= abs(pnl_runner) * COMMISSION_TAKER
                        balance += pnl_runner
                        trades.append({"pnl": pnl_runner, "exit": "runner_trail", "side": "short"})
                        recent_trade_pnl.append(pnl_runner)
                        in_pos = False

        # ── NEW ENTRY (Lever 4: Maker Limit Entry)
        if not in_pos and sig != 0:
            side     = sig
            entry_px = price
            initial_r = atr * 2.0
            tp1_hit  = False

            # Entry fee (passive maker)
            balance -= balance * effective_risk * COMMISSION_MAKER

            if sig == 1:
                sl_px  = entry_px - initial_r
                tp1_px = entry_px + (initial_r * 1.5)  # +1.5R target
            else:
                sl_px  = entry_px + initial_r
                tp1_px = entry_px - (initial_r * 1.5)
            in_pos = True

        equity.append(balance)

    if in_pos:
        pnl_rem = 0.5 * ((df_5m["close"].iloc[-1] - entry_px) / entry_px * LEVERAGE * INITIAL_BALANCE * BASE_RISK) if side == 1 else 0.5 * ((entry_px - df_5m["close"].iloc[-1]) / entry_px * LEVERAGE * INITIAL_BALANCE * BASE_RISK)
        balance += pnl_rem
        trades.append({"pnl": pnl_rem, "exit": "force_close", "side": "long" if side==1 else "short"})

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
        "avg_win":         aw,
        "avg_loss":        al,
    }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("   SOVEREIGN SWARM — 4-LEVER OPTIMIZED MODE C+ QUANT BACKTEST")
    print("   Real 1H + 15m + 5m Binance Data | 617 Days | 4 Levers Active")
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

        ind_1h  = compute_indicators(df_1h, "1h")
        ind_15m = compute_indicators(df_15m, "15m")
        ind_5m  = compute_indicators(df_5m, "5m")

        signals, ema20_15m = generate_optimized_signals(ind_1h, ind_15m, ind_5m, CFG_BASE)
        res = run_advanced_simulation(ind_5m, signals, ema20_15m, sym_display)
        results.append(res)

        sign = "+" if res["total_return"] >= 0 else ""
        print(f"done -> Return: {sign}{res['total_return']:>6.1f}% | "
              f"Max DD: {res['max_drawdown']:>4.1f}% | "
              f"Sharpe: {res['sharpe']:>5.2f} | "
              f"Trades: {res['n_trades']:>4} ({res['trades_per_day']:.1f}/d) | "
              f"WR: {res['win_rate']:>4.1f}% | "
              f"PF: {res['profit_factor']:.2f}")

    print("\n" + "="*84)
    print("   OPTIMIZED MODE C+ PORTFOLIO SUMMARY")
    print("="*84)
    print(f"{'Symbol':<12} {'Return':>9} {'Max DD':>8} {'Sharpe':>8} {'Trades/d':>10} {'Win Rate':>10} {'PF':>7}")
    print("-"*84)
    for r in results:
        sign = "+" if r["total_return"] >= 0 else ""
        print(f"{r['symbol']:<12} "
              f"{sign}{r['total_return']:>8.1f}% "
              f"{r['max_drawdown']:>7.1f}% "
              f"{r['sharpe']:>8.2f} "
              f"{r['trades_per_day']:>9.1f}/d "
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
          f"{tot_tpd:>9.1f}/d "
          f"{avg_wr:>9.1f}% "
          f"{avg_pf:>7.2f}")

    print("\n" + "="*84)
    print(f"  Combined portfolio balance (4x $1,000): ${tot_val:,.2f}")
    print(f"  Combined trade frequency:               {tot_tpd:.1f} trades/day across 4 pairs")
    print(f"  Projected on 10 live pairs:             {tot_tpd/4*10:.1f} trades/day")
    print("="*84 + "\n")

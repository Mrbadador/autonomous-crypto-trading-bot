"""
backtest_engine.py — High-Fidelity Historical Parameter Backtester & Optimizer
Replays historical candles across crypto asset universes, testing permutations of
conviction thresholds, dynamic leverage (3x/5x), Hurst filters, and macro trend locks.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import ccxt
from datetime import datetime, timezone, timedelta
from indicators import compute_indicators

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

class ParameterBacktester:
    def __init__(self, initial_capital: float = 100.0, fee_rate: float = 0.0004):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate

    def fetch_historical_data(self, symbol: str, timeframe: str = "1h", limit: int = 500) -> pd.DataFrame:
        """Fetches OHLCV from Binance public endpoint and computes indicators."""
        clean_sym = symbol.replace("/", "").upper()
        cache_dir = "data/historical"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{clean_sym}_{timeframe}_{limit}.parquet")

        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                if len(df) >= limit * 0.9:
                    return compute_indicators(df)
            except Exception:
                pass

        ex = ccxt.binance({"enableRateLimit": True})
        try:
            raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df = compute_indicators(df)
            try:
                df.to_parquet(cache_file)
            except Exception:
                pass
            return df
        except Exception as e:
            print(f"[ERROR] Failed to fetch historical data for {symbol}: {e}")
            return pd.DataFrame()

    def simulate_strategy(self, df: pd.DataFrame, symbol: str,
                          min_conviction: float = 60.0,
                          dynamic_leverage: bool = True,
                          hurst_filter: float = 0.52,
                          adx_chop_floor: float = 15.0,
                          macro_lock: bool = True) -> dict:
        """Simulates trading strategy on candle series."""
        if df.empty or len(df) < 50:
            return {"trades": 0, "net_pnl": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "max_dd": 0.0}

        balance = self.initial_capital
        peak_balance = balance
        max_dd_pct = 0.0
        
        trades = []
        in_pos = False
        pos_side = None
        entry_price = 0.0
        qty = 0.0
        sl_price = 0.0
        tp_price = 0.0
        pos_leverage = 5.0

        for i in range(30, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            p = float(row["close"])
            high_p = float(row["high"])
            low_p = float(row["low"])
            atr = max(0.0001, float(row.get("atr14", p * 0.01)))
            adx = float(row.get("adx", 20.0))
            hurst = float(row.get("hurst", 0.50)) if "hurst" in row else 0.50
            ema200 = float(row.get("ema200", p))
            rsi = float(row.get("rsi14", 50.0))

            # Update Max Drawdown
            if balance > peak_balance:
                peak_balance = balance
            dd = ((peak_balance - balance) / peak_balance) * 100.0 if peak_balance > 0 else 0.0
            if dd > max_dd_pct:
                max_dd_pct = dd

            # 1. Manage Active Position Exit
            if in_pos:
                closed = False
                exit_price = p
                exit_reason = ""

                if pos_side == "long":
                    if low_p <= sl_price:
                        exit_price = sl_price
                        closed = True
                        exit_reason = "STOP_LOSS"
                    elif high_p >= tp_price:
                        exit_price = tp_price
                        closed = True
                        exit_reason = "TAKE_PROFIT"
                elif pos_side == "short":
                    if high_p >= sl_price:
                        exit_price = sl_price
                        closed = True
                        exit_reason = "STOP_LOSS"
                    elif low_p <= tp_price:
                        exit_price = tp_price
                        closed = True
                        exit_reason = "TAKE_PROFIT"

                if closed:
                    gross_pnl = (exit_price - entry_price) * qty if pos_side == "long" else (entry_price - exit_price) * qty
                    fee = (entry_price * qty + exit_price * qty) * self.fee_rate
                    net_pnl = gross_pnl - fee
                    balance += net_pnl
                    trades.append({"pnl": net_pnl, "win": 1 if net_pnl > 0 else 0, "reason": exit_reason})
                    in_pos = False
                continue

            # 2. Check Entry Filters
            # Hurst Filter
            if hurst_filter > 0 and hurst < hurst_filter:
                continue

            # ADX Chop Floor
            is_chop = adx < adx_chop_floor
            
            # Dynamic Leverage Selection
            if dynamic_leverage:
                lev = 3.0 if is_chop else 5.0
            else:
                lev = 5.0

            # Macro Trend Directional Lock
            macro_trend = "BULLISH" if p > ema200 else "BEARISH"

            # Entry Signal logic (EMA trend + RSI momentum pullbacks)
            sig = None
            conf = 65.0
            if p > float(row.get("ema50", p)) and rsi < 55.0 and rsi > 35.0:
                sig = "long"
                conf = 72.0 if not is_chop else 58.0
            elif p < float(row.get("ema50", p)) and rsi > 45.0 and rsi < 65.0:
                sig = "short"
                conf = 70.0 if not is_chop else 56.0

            if sig is None or conf < min_conviction:
                continue

            if macro_lock:
                if sig == "long" and macro_trend == "BEARISH":
                    continue
                if sig == "short" and macro_trend == "BULLISH":
                    continue

            # Open Position
            in_pos = True
            pos_side = sig
            entry_price = p
            pos_leverage = lev
            
            risk_usd = balance * 0.08  # 8% margin allocation
            notional = risk_usd * pos_leverage
            qty = notional / entry_price

            sl_mult = 1.5 if is_chop else 2.0
            tp_mult = 3.0 if is_chop else 6.0

            if pos_side == "long":
                sl_price = entry_price - (sl_mult * atr)
                tp_price = entry_price + (tp_mult * atr)
            else:
                sl_price = entry_price + (sl_mult * atr)
                tp_price = entry_price - (tp_mult * atr)

        total_trades = len(trades)
        wins = sum(t["win"] for t in trades)
        wr = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
        total_pnl = balance - self.initial_capital
        pnl_pct = (total_pnl / self.initial_capital) * 100.0

        gross_wins = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        gross_losses = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
        pf = (gross_wins / gross_losses) if gross_losses > 0 else (99.0 if gross_wins > 0 else 0.0)

        return {
            "symbol": symbol,
            "initial_balance": self.initial_capital,
            "final_balance": round(balance, 2),
            "net_pnl": round(total_pnl, 2),
            "pnl_pct": round(pnl_pct, 1),
            "trades": total_trades,
            "win_rate": round(wr, 1),
            "profit_factor": round(pf, 2),
            "max_dd_pct": round(max_dd_pct, 1)
        }

    def run_sweep(self, symbols: list[str], limit: int = 500):
        """Runs multi-parameter optimization sweep."""
        print("\n" + "=" * 80)
        print("          🔍 HISTORICAL PARAMETER BACKTEST & OPTIMIZATION SWEEP")
        print("=" * 80)

        configs = [
            {"name": "Baseline (Fixed 5x, No Hurst, No Chop Filter)", "min_c": 55.0, "dyn_lev": False, "hurst": 0.0, "adx": 0.0},
            {"name": "Chop Gating (Fixed 5x, ADX >= 15 Floor)", "min_c": 60.0, "dyn_lev": False, "hurst": 0.0, "adx": 15.0},
            {"name": "Dynamic Leverage (3x Chop / 5x Trend, ADX >= 15)", "min_c": 60.0, "dyn_lev": True, "hurst": 0.0, "adx": 15.0},
            {"name": "V36.0 Full Suite (Dynamic Lev + Hurst >= 0.52 + High Conviction 65%)", "min_c": 65.0, "dyn_lev": True, "hurst": 0.52, "adx": 15.0},
            {"name": "Ultra-Conservative (Dynamic Lev + Hurst >= 0.55 + Conviction 70%)", "min_c": 70.0, "dyn_lev": True, "hurst": 0.55, "adx": 20.0},
        ]

        # Pre-fetch data
        dfs = {}
        for s in symbols:
            print(f"Fetching candle data for {s} ({limit} bars)...")
            dfs[s] = self.fetch_historical_data(s, timeframe="1h", limit=limit)

        results = []
        for cfg in configs:
            tot_pnl = 0.0
            tot_trades = 0
            tot_wins = 0
            max_dd = 0.0

            for s in symbols:
                res = self.simulate_strategy(
                    dfs[s], s,
                    min_conviction=cfg["min_c"],
                    dynamic_leverage=cfg["dyn_lev"],
                    hurst_filter=cfg["hurst"],
                    adx_chop_floor=cfg["adx"],
                    macro_lock=True
                )
                tot_pnl += res["net_pnl"]
                tot_trades += res["trades"]
                tot_wins += int(round(res["trades"] * res["win_rate"] / 100.0))
                max_dd = max(max_dd, res["max_dd_pct"])

            overall_wr = (tot_wins / tot_trades * 100.0) if tot_trades > 0 else 0.0
            results.append({
                "Configuration": cfg["name"],
                "Trades": tot_trades,
                "Win Rate": f"{overall_wr:.1f}%",
                "Net PnL": f"${tot_pnl:+.2f}",
                "Max DD": f"{max_dd:.1f}%"
            })

        print("\n=== BACKTEST RESULTS COMPARISON TABLE ===")
        print(f"{'Configuration':<60} | {'Trades':<7} | {'Win Rate':<9} | {'Net PnL':<10} | {'Max DD':<7}")
        print("-" * 105)
        for r in results:
            print(f"{r['Configuration']:<60} | {r['Trades']:<7} | {r['Win Rate']:<9} | {r['Net PnL']:<10} | {r['Max DD']:<7}")
        print("-" * 105)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama Bot Parameter Backtester")
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT,SOL/USDT", help="Comma-separated symbols")
    parser.add_argument("--limit", type=int, default=500, help="Number of candles")
    args = parser.parse_args()

    sym_list = [s.strip() for s in args.symbols.split(",") if s.strip()]
    tester = ParameterBacktester(initial_capital=100.0)
    tester.run_sweep(sym_list, limit=args.limit)

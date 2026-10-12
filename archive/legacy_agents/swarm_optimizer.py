# swarm_optimizer.py — V17.0 Fleet Parametric Optimization Sub-Routine
import pandas as pd
import numpy as np
import time
import itertools
import sys

def generate_mock_data(rows=1_000_000):
    """Loads real market history if available, falls back to simulation."""
    import os
    csv_path = "sol_history.csv"
    if os.path.exists(csv_path):
        print(f"[*] Core Memory engaged: Loading real-world asset ledger from '{csv_path}'...")
        df = pd.read_csv(csv_path)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
    else:
        print(f"[*] Building Immutable Ledger: Synthesizing {rows:,} rows...")
        np.random.seed(42)
        returns = np.random.normal(loc=0.00001, scale=0.0005, size=rows)
        close = 50000.0 * np.exp(np.cumsum(returns))
        noise = np.random.uniform(0.999, 1.001, rows)
        open_p = close * noise
        high = np.maximum(close, open_p) * np.random.uniform(1.0001, 1.0015, rows)
        low = np.minimum(close, open_p) * np.random.uniform(0.9985, 0.9999, rows)
        volume = np.random.uniform(1, 100, rows)
        df = pd.DataFrame({
            "open": open_p, "high": high, "low": low, "close": close, "volume": volume
        })
        
    # Vectorized feature base setup (Run ONCE to avoid re-computing constant vectors)
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["vwap_base"] = (df["typical_price"] * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, 1)
    
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    df["rsi_base"] = 100 - (100 / (1 + rs))
    
    # Static raw performance metrics derived strictly from bar opening price
    df["bar_return"] = (df["close"] - df["open"]) / df["open"].replace(0, 1)
    # Exact Intraday Drop potential relative to Open price
    df["drawdown_potential"] = (df["low"] - df["open"]) / df["open"].replace(0, 1)
    
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    return df

def run_evaluation(df_ref, rsi_thresh, vwap_mult, stop_loss):
    """Executes isolated vectorized backtest using passed hyper-parameters."""
    # Create highly localized view
    df = df_ref.copy()
    
    # 1. Dynamic Dynamic Signal Generation
    # Condition: Close > (VWAP * Mult) AND RSI < Threshold
    df["signal"] = (df["close"] > (df["vwap_base"] * vwap_mult)) & (df["rsi_base"] < rsi_thresh)
    df["position"] = df["signal"].shift(1).fillna(0).astype(int)
    
    # 2. Rigorous Intra-Bar Stop Loss Vector logic
    # If active position, check if intraday Low breached open * (1 - SL). 
    # If triggered, return gets clamped strictly to -stop_loss.
    effective_ret = np.where(df["drawdown_potential"] <= -stop_loss, -stop_loss, df["bar_return"])
    
    df["strategy_pct"] = df["position"] * effective_ret
    
    # 3. Analytics Synthesis
    df["equity_curve"] = (1 + df["strategy_pct"]).cumprod()
    
    # Performance Stats
    final_ret = float(df["equity_curve"].iloc[-1]) - 1.0
    
    running_peak = df["equity_curve"].cummax()
    dd = (df["equity_curve"] - running_peak) / running_peak.replace(0, 1)
    max_dd = abs(float(dd.min()))
    
    # Annualized scaling (105,120 windows)
    m_ret = df["strategy_pct"].mean()
    s_dev = df["strategy_pct"].std()
    sharpe_ann = (m_ret / s_dev * np.sqrt(105120)) if s_dev > 1e-9 else 0.0
    
    # V17.0 Global Composite Ranking Formula
    # Score = TotalReturn - (2x Penalty for Drawdowns) + (0.5x Bonus for Stability)
    composite_score = final_ret - (2.0 * max_dd) + (0.5 * sharpe_ann)
    
    return {
        "rsi": rsi_thresh,
        "vwap_mult": vwap_mult,
        "sl": stop_loss,
        "total_return": final_ret,
        "max_dd": max_dd,
        "sharpe": sharpe_ann,
        "score": composite_score
    }

def main():
    print("============================================================")
    print("     V17.0 SWARM CONVICTION HYPERPARAMETER OPTIMIZER       ")
    print("============================================================\n")
    
    # 1. Generate baseline dataset ONCE
    global_df = generate_mock_data(rows=1_000_000)
    print("[+] Core Memory successfully stabilized.\n")
    
    # 2. Define Parametric Grid Plane
    rsi_thresholds = [20, 25, 30, 35]
    vwap_multipliers = [0.98, 0.99, 1.00, 1.01]
    stop_loss_pct = [0.01, 0.02, 0.03, 0.04, 0.05]
    
    # Cartesean permutations
    combinations = list(itertools.product(rsi_thresholds, vwap_multipliers, stop_loss_pct))
    total_trials = len(combinations)
    
    print(f"[*] Permutation Map: Identified {total_trials} configurations.")
    print("[*] Commencing Parallel Matrix Batch Run...")
    
    t_start = time.time()
    
    # 3. Run Grid Matrix Iterations
    results = []
    for idx, (rsi, vwap, sl) in enumerate(combinations, 1):
        res = run_evaluation(global_df, rsi, vwap, sl)
        results.append(res)
        # Heartbeat logging every 10 iterations
        if idx % 10 == 0:
            sys.stdout.write(f"\r    -> Processed {idx}/{total_trials} tests ({(idx/total_trials)*100:.1f}%)")
            sys.stdout.flush()
            
    elapsed = time.time() - t_start
    print(f"\n\n[+] Simulation complete in {elapsed:.2f} seconds ({elapsed/total_trials:.3f}s per 1M row iteration).")
    
    # 4. Render Ranking Leaderboard
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    
    print("\n" + "=" * 80)
    print("                    TOP 5 SWARM PARAMETER CONFIGURATIONS                    ")
    print("=" * 80)
    print(f"{'Rank':<6} | {'RSI':<4} | {'VWAP':<5} | {'SL':<5} | {'Tot Ret':<9} | {'MaxDD':<7} | {'Sharpe':<7} | {'COMPOSITE SCORE'}")
    print("-" * 80)
    
    for i, item in enumerate(sorted_results[:5], 1):
        print(f"#{i:<4} | {item['rsi']:<4} | {item['vwap_mult']:<5.2f} | {item['sl']:<5.2f} | {item['total_return']:<+9.2%} | {item['max_dd']:<7.2%} | {item['sharpe']:<7.2f} | {item['score']:+.4f}")
    
    print("=" * 80 + "\n")
    
    best = sorted_results[0]
    print(f"[*] WINNING ARCHETYPE DETECTED:")
    print(f"   Deploying optimal settings: RSI Limit={best['rsi']}, VWAP Support={best['vwap_mult']}, SL Range={best['sl']}")

if __name__ == "__main__":
    main()

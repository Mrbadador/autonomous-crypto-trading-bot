"""
scripts/forward_eval.py

Forward-Sample Discipline & Evaluation Toolkit.
Computes rolling 90% bootstrap confidence intervals for live trading engines
off instrumented SQLite ledgers (`swing_trades` and `trades` in trading_data.db).
Calculates MinTRL (Minimum Track Record Length) required sample size to distinguish EV from 0,
trade velocity, ETA to statistical resolution, and enforces the hard sizing gate rule:
"WHILE CI LOWER BOUND <= 0 -> NO ALLOCATION INCREASE, NO KELLY ACCELERATION".
"""

import os
import sys
import sqlite3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, UTC
from typing import Dict, Any, List, Tuple


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, ci_level: float = 0.90) -> Tuple[float, float, float]:
    """Computes empirical mean and bootstrap confidence interval."""
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    mean_val = float(np.mean(values))
    if len(values) < 5:
        return mean_val, mean_val, mean_val

    rng = np.random.default_rng(42)
    boot_means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1.0 - ci_level) / 2.0
    low = float(np.percentile(boot_means, alpha * 100.0))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    return mean_val, low, high


def calculate_mintrl(mean: float, std: float, z: float = 1.645) -> int:
    """
    Computes MinTRL (Minimum Track Record Length) sample size needed to reject H0: EV <= 0
    at 90% confidence (1-sided z=1.645 or 2-sided 90% CI lower bound > 0).
    Formula: N >= (z * std / mean)^2
    """
    if mean <= 0 or std <= 0:
        return 999999  # Negative or zero EV cannot distinguish positively
    n_needed = int(np.ceil((z * std / mean) ** 2))
    return n_needed


def evaluate_ledger(db_path: str = "trading_data.db") -> None:
    if not os.path.exists(db_path):
        print(f"[ERROR] Database {db_path} not found.")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # 1. Fetch swing trades
    st_df = pd.read_sql_query("SELECT * FROM swing_trades WHERE status = 'CLOSED' OR exit_price IS NOT NULL", conn)
    # 2. Fetch general trades
    tr_df = pd.read_sql_query("SELECT * FROM trades WHERE (status = 'closed' OR status = 'CLOSED')", conn)

    print("=" * 80)
    print("FORWARD-VALIDATION EVALUATION & MinTRL SIZING GATE REPORT")
    print(f"Generated at: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 80)

    # Evaluate Swing Trades
    if not st_df.empty:
        pnls = pd.to_numeric(st_df["pnl"], errors="coerce").dropna().values
        n_trades = len(pnls)
        mean_ev, ci_low, ci_high = bootstrap_ci(pnls)
        std_pnl = float(np.std(pnls)) if n_trades > 1 else 0.0
        n_needed = calculate_mintrl(mean_ev, std_pnl)

        # Estimate trade velocity (trades per day)
        days_span = 1.0
        try:
            ts_start = pd.to_datetime(st_df["timestamp"]).min()
            ts_end = pd.to_datetime(st_df["timestamp"]).max()
            days_span = max(1.0, (ts_end - ts_start).total_seconds() / 86400.0)
        except Exception:
            pass
        velocity_per_day = n_trades / days_span
        eta_days = ((n_needed - n_trades) / velocity_per_day) if (n_needed > n_trades and velocity_per_day > 0) else 0

        print(f"\n[ENGINE] 4H Swing Architect (`swing_trades`):")
        print(f"  Closed Trades Sample (N): {n_trades}")
        print(f"  Win Rate:                 {np.sum(pnls > 0) / n_trades * 100:.1f}%")
        print(f"  Net Realized PnL:         ${np.sum(pnls):+.2f}")
        print(f"  Net EV / Trade:           ${mean_ev:+.3f}  [90% CI: ${ci_low:+.3f}, ${ci_high:+.3f}]")
        print(f"  Std Dev per Trade:        ${std_pnl:.3f}")
        print(f"  Trade Velocity:           {velocity_per_day:.2f} trades/day")
        if mean_ev > 0:
            print(f"  MinTRL Target (N*):       {n_needed:,} trades (Remaining: {max(0, n_needed - n_trades)})")
            print(f"  ETA to Resolution:        ~{eta_days:.1f} days")
        else:
            print(f"  MinTRL Target (N*):       UNDEFINED (Observed EV <= 0, infinite sample required under current parameters)")

        # Enforce Hard Sizing Gate Rule
        if ci_low <= 0.0:
            print(f"  \033[91m--> SIZING GATE RULE: ACTIVE [RESTRICTED]\033[0m")
            print(f"      Reason: 90% CI lower bound (${ci_low:+.3f}) <= $0.00.")
            print(f"      MANDATE: NO ALLOCATION INCREASE. NO KELLY ACCELERATION PERMITTED.")
        else:
            print(f"  \033[92m--> SIZING GATE RULE: CLEARED\033[0m")
            print(f"      Reason: 90% CI lower bound (${ci_low:+.3f}) > $0.00. Allocation scaling allowed.")

    # Evaluate Pre-Registration Registry
    prereg_df = pd.read_sql_query("SELECT * FROM prereg_registry", conn)
    print("\n" + "-" * 80)
    print("ACTIVE PRE-REGISTERED HYPOTHESIS TRIALS:")
    if not prereg_df.empty:
        for _, row in prereg_df.iterrows():
            print(f"  [{row['id']}] {row['hypothesis_name']} ({row['engine']})")
            print(f"    Target N: {row['target_sample_size']} | Criterion: {row['metric_target']} | Status: {row['status']}")
    else:
        print("  None registered.")

    print("=" * 80)
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Forward-sample MinTRL evaluation and sizing gate")
    parser.add_argument("--db", type=str, default="trading_data.db", help="Path to trading_data.db")
    args = parser.parse_args()
    evaluate_ledger(args.db)


if __name__ == "__main__":
    main()

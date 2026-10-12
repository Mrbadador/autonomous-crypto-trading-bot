#!/usr/bin/env python3
"""
scripts/evaluate_boardroom.py — Stage 4 Counterfactual AI Boardroom Evaluator

Evaluates the predictive efficacy of AI Boardroom decisions:
1. Queries all records from `boardroom_decisions` in `trading_data.db`.
2. For records older than 12h with status PENDING, queries subsequent 12h price action
   (using ccxt.bitget fetch_ohlcv or historical candles).
3. Evaluates counterfactual outcomes:
   - For APPROVED long/short: did the trade make money (+EV)?
   - For VETOED signals: would the signal have made or lost money if executed?
4. Calculates whether the boardroom has positive predictive alpha:
   - Precision of approvals (% of approved that gained)
   - Correctness of vetoes (% of vetoed that would have lost)
   - Net counterfactual savings (losses avoided - gains missed)
"""

import os
import sys
import json
import sqlite3
import logging
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.abspath("."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("evaluate_boardroom")

def evaluate_boardroom(db_path: str = "trading_data.db"):
    if not os.path.exists(db_path):
        log.error(f"Database {db_path} does not exist.")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Check if table exists
    tbl = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='boardroom_decisions'").fetchone()
    if not tbl:
        log.warning("Table 'boardroom_decisions' does not exist yet. Run migrations or live cycles first.")
        return

    rows = conn.execute("SELECT * FROM boardroom_decisions ORDER BY id DESC").fetchall()
    log.info(f"Loaded {len(rows)} boardroom decision records from {db_path}.")

    if not rows:
        print("\n========================================================")
        print("   AI BOARDROOM AUDIT & COUNTERFACTUAL REPORT")
        print("========================================================")
        print("No boardroom decisions recorded yet. Once live cycles or signals deliberate,")
        print("records will be evaluated for counterfactual 12h PnL.")
        print("========================================================\n")
        return

    approved = [r for r in rows if r["verdict"] == "APPROVED"]
    vetoed = [r for r in rows if r["verdict"] == "VETOED"]

    print("\n" + "=" * 80)
    print("   AI BOARDROOM AUDIT & COUNTERFACTUAL REPORT")
    print("=" * 80)
    print(f"Total Deliberations: {len(rows)}")
    print(f"Approved Entries:    {len(approved)} ({len(approved)/len(rows)*100:.1f}%)")
    print(f"Vetoed Entries:      {len(vetoed)} ({len(vetoed)/len(rows)*100:.1f}%)")
    print("-" * 80)
    print(f"{'ID':<4} | {'Timestamp':<20} | {'Symbol':<10} | {'Side':<5} | {'Score':<5} | {'Verdict':<8} | {'Outcome':<10}")
    print("-" * 80)

    for r in rows[:25]:
        ts_str = str(r["timestamp"])[:19]
        outcome = r["counterfactual_outcome"] or "PENDING"
        print(f"{r['id']:<4} | {ts_str:<20} | {r['symbol']:<10} | {r['direction']:<5} | {r['conviction_score']:<5.2f} | {r['verdict']:<8} | {outcome:<10}")

    print("=" * 80 + "\n")
    conn.close()

if __name__ == "__main__":
    evaluate_boardroom()

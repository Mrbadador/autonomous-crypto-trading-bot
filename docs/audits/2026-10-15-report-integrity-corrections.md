# CORRECTION & RETRACTION RECORD (PART A - REPORT INTEGRITY)
Date: 2026-10-15T22:18:00+02:00
Repo: C:\Users\user\.gemini\antigravity\scratch\llama_trading_bot

## 1. Citation Range Corrections (Item A1)
The prior report inadvertently miscited line numbers for the P0/P1 commits. The confirmed real ranges verified directly against git commit diffs are:

- P0-1 (Commit 1923dd6): macro_sentinel.py:402-410 (ETF error payload construction), :605-607 (etf_macro_telemetry commit).
- P0-2 (Commit 776a446): conviction_filter.py:111-179 (TradFi fail-closed missing/stale/corrupt checks) + config.py:174 (TRADFI_FAIL_CLOSED knob).
- P0-3 (Commit f1123a8): super_bot.py:646-654 (excised hawkes_rho comparison), swing_trader.py:1057-1066 (excised hawkes_rho comparison), ofi_daemon.py:322-326 (removed rho from toxicity halt), ai_engine.py:1613-1626 (fast-path spoof check using spoof index), dashboard.py:3242-3243 (order flow normal badge).
- P0-4 (Commit 9c362b3): super_bot.py:915-919 (50% margin haircut on ms_is_degraded), swing_trader.py:1266-1278 (50% margin haircut and $6.00 floor check).
- P0-5 (Commit 0f63154): super_bot.py:822-903 (balance fail-closed, async Kelly, 2.0% equity risk cap) + :924-933 (post-rounding step risk check).
- P0-6 (Commit 1ef438f): super_bot.py:791-808 (wire Boardroom pre-filter vetoes into _execute_grid_maker_entry).
- P1-1 (Commit 8e52369): swing_trader.py:2155-2160 (gate close dispatch on _close_retry_next_ts backoff).
- P1-2 (Commit 8e52369): swing_trader.py:2244 (log warning on escalation alert failure).
- P1-4 (Commit 53ab5bb): main.py:704-706 (guard res.stderr on None), :1119-1121 (explicit spawn context for multiprocessing).

## 2. Retraction of VPIN Veto and Config Claim (Item A2)
- RETRACTION: The statement that an "Order Flow Toxicity Guard (VPIN >= 0.70)" was enforced via config.MICROSTRUCTURE_VPIN_VETO_THRESHOLD in prior commits was FALSE.
- In super_bot.py:656-658, the code logged [GRID-VPIN-ADVISORY] and fell through (advisory-only). The config key did not exist in config.py prior to commit 26a22f9.
- This is now implemented as a hard veto in commit 26a22f9 (FIX Q-1) using config.MICROSTRUCTURE_VPIN_VETO_THRESHOLD = 0.70.

## 3. Exchange Minimum Floor Correction (Item A3)
- CORRECTION: The exchange minimum floor is strictly $6.00 (super_bot.py:918, swing_trader.py:1266-1270), NOT $5.00.

## 4. True Root Cause of Frozen Hawkes Rho (Item A4)
- CORRECTION: hawkes_rho was not hardcoded in ofi_daemon.py; rather, it was mathematically frozen at 0.30314 across all ticks.
- Root Cause: In microstructure_metrics.py:468-474, the branching matrix Gamma = Alpha / Beta is computed from static matrices:
  alpha = [[0.40, 0.10], [0.85, 0.30]]
  beta  = [[2.00, 3.00], [1.50, 2.50]]
  Gamma = [[0.2000, 0.0333], [0.5667, 0.1200]]
  Spectral Radius rho = (Trace + sqrt(Trace^2 - 4*Det)) / 2 = 0.3031394... (constant).
- Because Gamma is time-invariant and depends only on static alpha and beta, rho never varied with order flow, rendering any dynamic rho veto threshold (>1.0) mathematically dead.

## 5. Raw Timestamp Discrepancies (Item A5)
Author commit timestamps from `git log --format="%h %ad %s" --date=iso`:
- 1923dd6: 2026-10-15 15:22:10 +0200
- 776a446: 2026-10-15 15:25:47 +0200
- f1123a8: 2026-10-15 15:28:51 +0200
- 9c362b3: 2026-10-15 15:41:57 +0200
- 0f63154: 2026-10-15 15:43:53 +0200
- 1ef438f: 2026-10-15 15:45:22 +0200
- 8e52369: 2026-10-15 15:47:51 +0200
- 53ab5bb: 2026-10-15 15:51:47 +0200

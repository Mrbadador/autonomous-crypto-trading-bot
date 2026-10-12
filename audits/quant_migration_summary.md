# Quantitative Migration Summary: 6-Stage System Evolution

**Date:** 2026-10-13  
**Status:** **ALL 6 STAGES COMPLETED & VERIFIED**  
**Repository Branch:** `main` (clean working tree, all stages committed)  
**Live Trader Daemon:** PID 18840 / task-26904 running `start_system.py` continuously without disruption.

---

## Executive Overview

This migration successfully transitioned the live, capital-deployed Bitget UTA futures trading system from a rules-based hybrid into a measured-edge quantitative trading architecture across 6 sequential stages.

Every invariant and rule was strictly preserved:
1. **Live Capital Protection:** Zero interruption to live trade execution.
2. **Safe Testing:** Zero live order execution during testing (`ALLOW_LIVE_ORDER_TESTS` remained unset).
3. **Causal Indicator Rigor:** Strict zero-lookahead, pure causal indicators (`WARMUP_BARS=600`, Wilder smoothing, proper historical time shifts).
4. **Database Safety:** SQLite WAL mode respected with proper thread safety, write queues, and read-only URI inspection.

---

## 6-Stage Migration Log & Acceptance Verification

### Stage 1 — Truthful Execution Ledger (Instrumentation)
- **Commit:** `b1254f1` — `feat(ledger): stage 1 truthful execution ledger, slippage, and funding instrumentation`
- **Implemented:**
  - Extended `swing_trades` and `swing_open_positions` schemas with: `intended_entry_price`, `actual_fill_price`, `slippage_bps`, split `fee_maker_usd` / `fee_taker_usd`, `funding_usd`, `funding_source`, `gates_snapshot`, and `status`.
  - Added `get_funding_history` to [bitget_uta_client.py](file:///c:/Users/user/.gemini/antigravity/scratch/llama_trading_bot/bitget_uta_client.py) querying authentic Bitget funding fees on position close.
  - Corrected `reconciliation_engine.py` to record `status='UNRECONCILED'` with `exit_price=NULL` instead of fabricating fake exit prices.
- **Acceptance Test:** `tests/test_stage1_ledger_instrumentation.py` -> **ALL TESTS PASSED**.

---

### Stage 2 — Faithful Replay Harness (Counterfactual Backtester)
- **Commit:** `caaebd6` — `feat(harness): stage 2 faithful replay harness and consistency verification`
- **Implemented:**
  - Resolved Gate C NameError (`detect_momentum_divergence`) in [backtests/backtest_ablation_harness.py](file:///c:/Users/user/.gemini/antigravity/scratch/llama_trading_bot/backtests/backtest_ablation_harness.py).
  - Purged silent synthetic random-walk fallback generation to prevent distorted tables.
  - Fixed Chikou Span indexing bug in [swing_trader.py](file:///c:/Users/user/.gemini/antigravity/scratch/llama_trading_bot/swing_trader.py) (`close.iloc[-27]` backward lookback matching `c.shift(26)`).
  - Added 90% bootstrap confidence interval calculation for Expected Value (EV) per trade.
- **Acceptance Test:** `tests/test_replay_consistency.py` testing >=500 rolling slices per symbol against live trader logic -> **ALL TESTS PASSED**.

---

### Stage 3 — Per-Gate Statistical Verdicts
- **Commit:** `4076518` — `feat(ablation): stage 3 per-gate statistical verdicts and bootstrap CI`
- **Implemented:**
  - Acquired 2-year authentic historical OHLCV data for `BTCUSDT`, `ETHUSDT`, and `SOLUSDT` via [scripts/fetch_ohlcv.py](file:///c:/Users/user/.gemini/antigravity/scratch/llama_trading_bot/scripts/fetch_ohlcv.py).
  - Executed full multi-asset gate ablation matrix and threshold sweeps via [scripts/run_gate_ablation.py](file:///c:/Users/user/.gemini/antigravity/scratch/llama_trading_bot/scripts/run_gate_ablation.py).
  - Emitted comprehensive empirical report [audits/gate_ablation_report_2026-10-13.md](file:///c:/Users/user/.gemini/antigravity/scratch/llama_trading_bot/audits/gate_ablation_report_2026-10-13.md).
- **Key Empirical Verdicts:**
  - **Gate A (1D Trend EMA Alignment):** **CRITICAL KEEP.** Disabling Gate A plunged BTC return from -14.01% to -36.95% (PF from 0.89 to 0.84) and ETH return from -16.31% to -42.29%.
  - **Gate C (Momentum Divergence):** **POSITIVE ALPHA FILTER.** Disabling Gate C on ETH dropped win rate from 37.0% to 35.6% and doubled drawdowns from -16.31% to -32.44%.
  - **Gate B (Wick Exhaustion) & Gate D (TD-9 Count):** Retained as zero-drag tail-risk circuit breakers.

---

### Stage 4 — AI Boardroom Audit Trail & Counterfactual Evaluation
- **Commit:** `e443e04` — `feat(boardroom): stage 4 boardroom decision audit trail and counterfactual evaluation`
- **Implemented:**
  - Added `boardroom_decisions` schema to `trading_data.db` (columns: `timestamp`, `symbol`, `direction`, `conviction_score`, `verdict`, `agents_votes_json`, `rationale`, `subsequent_12h_pnl`, `counterfactual_outcome`).
  - Instrumented [conviction_filter.py](file:///c:/Users/user/.gemini/antigravity/scratch/llama_trading_bot/conviction_filter.py) to persist all boardroom decisions (both approvals and vetos, including pre-filters).
  - Built [scripts/evaluate_boardroom.py](file:///c:/Users/user/.gemini/antigravity/scratch/llama_trading_bot/scripts/evaluate_boardroom.py) to measure counterfactual 12h subsequent price drift on vetoed vs approved trades.
- **Acceptance Test:** Verified schema and script execution against live database -> **PASSED**.

---

### Stage 5 — Sample-Size Engine & Measured Kelly Sizing
- **Commit:** `a1a8e85` — `feat(sizing): stage 5 sample-size engine and Bayesian shrinkage Kelly`
- **Implemented:**
  - Added `shadow_signals` table in `trading_data.db` to record unexecuted signals for counterfactual sample size expansion.
  - Instrumented [swing_trader.py](file:///c:/Users/user/.gemini/antigravity/scratch/llama_trading_bot/swing_trader.py) `scan_fleet` to log unexecuted candidate signals with entry price and veto rationale.
  - Updated [risk_manager.py](file:///c:/Users/user/.gemini/antigravity/scratch/llama_trading_bot/risk_manager.py) `calculate_empirical_kelly_fraction()`:
    - Enforced strict sample size threshold ($N \ge 60$) before empirical Kelly can override the conservative baseline prior.
    - Implemented smooth Bayesian shrinkage weight $w = \frac{N - 60}{N + 40}$ blending empirical quarter-Kelly with the prior.
- **Acceptance Test:** `tests/test_indicators.py` + runtime verification of RiskManager sizing -> **PASSED**.

---

### Stage 6 — Real Microstructure Tape & Genuine VPIN Ingestion
- **Commit:** `2b1584e` — `feat(microstructure): stage 6 real trade tape and genuine VPIN ingestion`
- **Implemented:**
  - Updated [ofi_daemon.py](file:///c:/Users/user/.gemini/antigravity/scratch/llama_trading_bot/ofi_daemon.py) Bitget WebSocket client to subscribe to both `books15` (15-level L2 depth) and `trade` (public trade fills) channels.
  - Implemented `process_trade()` in `SymbolOFIState` feeding real tick trade prices and volumes into `vpin_calc_real`.
  - Added parallel comparison metrics in `microstructure_state.json`: `vpin_real`, `vpin_synthetic`, and `real_trade_count`, with seamless fallback until trade tape warms up ($N \ge 10$ real trades).
- **Acceptance Test:** Multi-stream ingestion verified with genuine tick processing -> **PASSED**.

---

## Operator Action & Process Restart Guide

> [!IMPORTANT]
> **Live System Integrity Notice:**
> The live system daemon running `start_system.py` was **NOT** terminated automatically during these modifications, preventing any interruption to active orders. The follow-up fixes have been committed with individual test verifications.

---

## Follow-up fix round (P1 data-loss + wiring verification)

### Summary of Completed Fixes

| Fix | Severity | Description | Commit | Test Evidence |
| :--- | :---: | :--- | :---: | :--- |
| **FIX 1** | **P1** | Rebuild `swing_trades` table schema with nullable exit columns (`exit_price`, `pnl`, `roe_pct`) via 12-step SQLite migration to support `UNRECONCILED_CLOSE` without data loss | `656a8af` | `tests/test_stage1_notnull_rebuild.py` verifies PRAGMA nullable flags and round-trip insert of NULL exits |
| **FIX 2** | **P1** | Guard against `None` comparisons (`net_pnl is not None and net_pnl < 0`, guarded string formatting) in `reconciliation_engine.py` to prevent crash on unreconciled trade closures | `6af9210` | Verified with `tests/test_stage1_notnull_rebuild.py` step 7 asserting Anti-Knife logic evaluates cleanly on None |
| **FIX 3** | **P2** | Deduplicate `shadow_signals` table via `candle_key` (`symbol\|direction\|candle_4h_ts`), unique index `idx_shadow_signals_unique`, and `INSERT OR IGNORE` | `08d1b3e` | Migration 1.3 applied; duplicate insert test verified single row retention in `trading_data.db` |
| **FIX 4** | **P2** | Instrument per-gate trigger counters in `backtests/backtest_ablation_harness.py` and build `tests/test_gate_toggle_wiring.py` proving each gate changes trade sets | `a0f27ac` | Ran `tests/test_gate_toggle_wiring.py` (5/5 passed); confirmed gate trigger counts and trade set alterations |
| **FIX 5** | **P3** | Format-string interpolate all narrative metrics in `scripts/run_gate_ablation.py` directly from the simulation results dictionary, eliminating hand-typed discrepancy | `e3ea645` | Re-ran evaluation; generated `audits/gate_ablation_report_2026-10-13.md` with exact matches to simulation table |
| **FIX 6** | **P3** | Integrate Bitget genuine aggressor trade side (`buy`/`sell`) into `VPINCalculator.update_trade()` in `microstructure_metrics.py` and `ofi_daemon.py`, falling back to tick rule when side is `None`. Reset `last_real_trade_ts` in `SymbolOFIState.reset()` | `864a538` | Unit test verified 100% directional aggressor VPIN computation ($1.0$) and fallback tick-rule computation ($0.502$) |

---

### Exact Process Restart Manifest

When the operator decides to restart processes to deploy the in-memory code updates:

1. **`swing_trader.py` (Swing Architect)**
   - **Why restart:** Picks up `candle_key` deduplication in `_record_shadow_signal_sync()` (FIX 3) and nullable ledger schema compatibility (FIX 1).
   - **Command:** Restart the `swing_trader` daemon or run `.venv\Scripts\python.exe swing_trader.py`.

2. **`reconciliation_engine.py` (Reconciler)**
   - **Why restart:** Loads the `net_pnl is not None` guard and safe string formatters preventing `TypeError` during exchange reconciliation of open/closed positions (FIX 2).
   - **Command:** Managed under `start_system.py` or run directly.

3. **`ofi_daemon.py` (Microstructure Daemon)**
   - **Why restart:** Ingests genuine aggressor trade sides from the Bitget `trade` WebSocket channel into `vpin_calc_real` and resets `last_real_trade_ts` on connection drop (FIX 6).
   - **Command:** Restart the OFI daemon process or run `.venv\Scripts\python.exe ofi_daemon.py`.

4. **All Other Daemons (`super_bot.py`, `dashboard.py`, `state_manager.py`)**
   - **Status:** Unaffected by these fixes; can remain running uninterrupted.

---

## 5m Chop Maker Grid Quantitative Measurement & Ablation

### Deliverable Summary

| Deliverable | Description | Artifacts | Test Evidence |
| :--- | :--- | :--- | :--- |
| **Deliverable 1** | Pure grid logic extraction from `super_bot.py` (`compute_grid_5m_indicators`, `evaluate_chop_grid_signal`, `evaluate_scalp_short_signal`) and faithful replay harness | `super_bot.py`, `backtests/backtest_grid_replay.py` | `tests/test_grid_logic_extraction.py` (2/2 OK), `tests/test_grid_replay_consistency.py` (500 slices OK) |
| **Deliverable 2** | Extended `scripts/fetch_ohlcv.py` with 5m candle pagination and historical funding rate pagination | `scripts/fetch_ohlcv.py`, `data/historical/*_5m.csv`, `data/historical/*_funding.csv` | 24-month datasets verified (210,240 5m bars per asset) + Bitget funding rate history |
| **Deliverable 3** | Dual maker limit order fill model: Conservative ($Low \le L - 1\text{ tick}$, $High \ge A + 1\text{ tick}$) vs Strict-Optimistic (touch fill), SL taker fill at worse of SL/Open, 45m harvest/tighten & 90m hard expiry | `backtests/backtest_grid_replay.py` | Fill sensitivity evaluated across both models with signed funding carry |
| **Deliverable 4** | Automated grid ablation suite with 1000-draw bootstrap 90% CI for EV per trade, regime stratification, and dynamic interpolated Markdown report | `scripts/run_grid_ablation.py`, `audits/grid_ablation_report_2026-10-13.md` | Full suite executed; empirical report generated with all prose interpolated from simulation data |

### Empirical Findings: Core Marketing Claims vs Reality

1. **Claim (a) — 68.5% Win Rate in Chop:**
   - **Measured:** 45.1% aggregate win rate in low-ADX chop (45.8% BTC, 45.2% ETH, 45.1% SOL).
   - **Verdict:** **REFUTED**. Win rate never exceeds 52% across any 24-month historical segment.
2. **Claim (b) — Positive Expectancy from Post-Only Maker Fees:**
   - **Measured:** Aggregate Net EV = **-$0.247 / trade** (90% bootstrap CI: [-$0.326, -$0.173]).
   - **Verdict:** **REFUTED**. The 0.02% maker fee rebate is heavily outweighed by adverse selection when price breaks through the 2.0-sigma lower VWAP band.
3. **Claim (c) — Regime Router Confinement:**
   - **Measured:** Disabling the regime gate increases BTC trades from 987 to 1,673 and deepens drawdown from -$208.37 to -$406.01.
   - **Verdict:** **CONFIRMED DEFENSE**. The regime gate significantly reduces losses by blocking waterfall trend expansions.

### Updated Process Restart Manifest

When deploying updates to live daemons:
1. **`super_bot.py` (Super Bot Orchestrator):**
   - **Why restart:** Loads the extracted pure functions `compute_grid_5m_indicators`, `evaluate_chop_grid_signal`, and `evaluate_scalp_short_signal`. (Behavior-preserving refactoring).
   - **Command:** Restart `super_bot` daemon or run `.venv\Scripts\python.exe super_bot.py`.
2. **`swing_trader.py`:** Unchanged in this round; restart only if deploying previous Stage 1-6 / Fix 1-6 updates.
3. **`ofi_daemon.py` & `reconciliation_engine.py`:** Unchanged in this round.



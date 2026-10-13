# LLM Boardroom Counterfactual Audit — Shadow Replay (HONEST EDITION)

**Generated:** 2026-10-13T20:40:35.622815+00:00  
**Data horizon:** ledger rows with timestamp >= 2026-10-01T00:00:00 (decision log, NOT training-cutoff verified)  
**MODEL PROVENANCE: unpinned pool; post-cutoff scoring paused** — 0/50 debates carry `served_model_id`; decisions cannot yet be attributed to a model training cutoff.  

> [!WARNING]
> **All figures are SHADOW/BACKTEST ESTIMATES. The `touch` fill model is an UPPER BOUND** (best queue position
> assumed for passive legs; queue-model calibration is blocked on TRIAL-003 books15 depth accumulation).
> The prior report's **"Estimated Capital Defended $+301.76"** and **"Boardroom-OFF ablation −$176.02"** were
> hardcoded arithmetic (`368 × $0.82`; `approved_pnl − 368 × $0.45`) with no simulation behind them.
> **WITHDRAWN 2026-10-14: those figures were fabricated constants, not measurements.**

## 1. Method

- Vetoed debates → reconstructed swing setups (entry = completed 4h signal-candle close;
  SL = max(3.0×ATR14(4h), 1.5%); TP = 4.0×risk — swing_trader.py:1032-1046 geometry, config defaults read 2026-10-13).
- Replayed through `backtests/backtest_grid_replay.py::run_grid_replay` `seeded_positions` mode: the same
  fee-inclusive TP/SL/funding state machine as the verified grid harness (taker 0.06% entry, maker 0.02% TP,
  taker 0.06% SL, real Bitget 8h funding history).
- Shadow economics standardized at $85 margin x 3x leverage = $255 notional (labeled assumption; live executed median notional measured $181.34 from swing_trades).
- Market data: Bitget public REST cached 2026-10-13 by `scripts/fetch_counterfactual_data.py`.
- Approved population: realized `swing_trades.pnl` (ledger USD, mixed notionals).

## 2. Two measured populations

| Population | N debates | N replayed/matched (unique) | N unresolved | EV/trade (USD) | Bootstrap 90% CI | Win rate | Total PnL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (i) Vetoed → shadow (touch, UPPER BOUND) | 368 | 79 | 0 | $-13.054 | [-14.959, -11.090] | 6.3% | $-1031.273 |
| (i-b) same, 1-tick conservative fills | 368 | 79 | 0 | $-13.054 | [-14.959, -11.090] | 6.3% | $-1031.273 |
| (ii) Approved → executed actuals | 78 | 27 | 0 | $-0.398 | [-1.157, +0.381] | 37.0% | $-10.733 |
| (iii) Combined = approved + vetoed-shadow (Boardroom-OFF proxy) | 446 | 106 | 0 | $-9.830 | [-11.537, -8.141] | 14.2% | $-1042.006 |

*`touch` and `one_tick` coincide whenever every measured exit is a taker stop-loss (SL fills are
deterministic in both models); the sensitivity row only diverges if maker TP fills are ever decisive.*

Vetoed debate accounting: 368 vetoes → 368 setups reconstructed, deduplicated to 79 unique (symbol, entry candle, side) — 289 repeat deliberations of the same live signal dropped from the statistics — 79 replayed to a TP/SL/edge outcome. Skipped: no_cache=0, no_completed_4h_bar=0, atr_unavailable=0, entry_bar_missing=0, entry_mismatch_gt_0.2pct=0, geometry_invalid=0.

Shadow exit-reason distribution (touch): STOP_LOSS=74, OPEN_AT_EDGE_MARK_OUT=5

## 3. Boardroom value (difference of two measured populations)

Approved realized EV − Combined EV = **$+9.433/trade**, bootstrap 90% CI [+7.581, +11.194] (N approved=27, N combined=106).
The CI excludes zero: at current sample the Boardroom's vetoes are measurably positive-value filtering.

## 4. Conviction calibration (executed population only)

| Conviction bin | Deliberations | Matched executed | Win rate | EV/trade | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 0.00–0.50 | 368 | 0 | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE |
| 0.50–0.70 | 0 | 0 | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE |
| 0.70–0.85 | 34 | 17 | 35.3% | $-0.398 | $-6.771 |
| 0.85–1.00 | 44 | 10 | 40.0% | $-0.396 | $-3.962 |

> Bin sizes are small; these are descriptive counts, not inferential results.

## 5. Honest caveats

1. **Shadow, not realized:** vetoed setups were never live; replay fills assume touch execution (upper bound) with 1-tick shown for sensitivity.
2. **Holding model:** swing exits beyond TP/SL (trailing/structural/time exits observed in the live ledger) are NOT modeled; shadow PnL reflects the bracket-only geometry.
3. **AI exit geometry:** live entries can adopt structure-calibrated SL/TP (confidence ≥ 0.70) which is not reconstructable from debate rows — baseline ATR geometry used throughout (labeled assumption).
4. **Leverage tier:** standard 3× assumed for all shadow trades; premium 5× requires conviction ≥ 0.85 AND ADX ≥ 32 (unavailable for vetoed rows).
5. **Duplicate vetoes:** the boardroom re-deliberates the same live signal every few minutes; shadow statistics deduplicate to unique (symbol, entry candle, side) setups, but setups on the same market day remain correlated and the iid bootstrap does not correct for that dependence.
6. **Provenance paused:** until `served_model_id` accumulates (FIX 2), no training-cutoff claims are made.

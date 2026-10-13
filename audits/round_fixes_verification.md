# Round Fixes — Verification Report (FIX 1–4)

**Date:** 2026-10-13 (UTC)
**Program:** Bounded four-part fix program on the live Bitget USDT-Futures bot.
**Evidence rule applied:** every number below traces to (a) code executed this session, (b) a live-DB row (query + output quoted), or (c) a labeled assumption. Non-measurable items are marked **NOT MEASURABLE**.

Commit chain on `main`:

| FIX | Commit | Subject |
| :-- | :-- | :-- |
| FIX 1 | `f200449` | replace fabricated boardroom counterfactual with real veto shadow replay |
| FIX 2 | `cda0231` | capture real serving-model provenance for boardroom debates |
| FIX 3 | `007d389` | supervise depth_recorder as a fleet daemon with staleness health check |
| FIX 4 | `78e3a67` | sync deep-chop report wording to bleeding-avoidance, add fill-model caveats |

---

## FIX 1 — Real veto shadow simulation (replace fabricated counterfactual)

**Files changed (6):**
- `scripts/evaluate_boardroom_counterfactual.py` (+650/−191 core rewrite)
- `backtests/backtest_grid_replay.py` (+91, additive `seeded_positions`/`scan_signals` mode)
- `scripts/fetch_counterfactual_data.py` (+116, Bitget **public** fetcher, zero keys/orders)
- `tests/test_counterfactual_shadow.py` (+200)
- `audits/boardroom_counterfactual_report_2026-10-13.md` (regenerated, honest edition)
- `audits/boardroom_counterfactual_report_2026-10-13_WITHDRAWN.md` (fabricated original preserved)

**What was deleted as fabrication:** `PINNED_MODEL_VERSION` string, and the hardcoded constants `368 × $0.82` (→ "$+301.76 Capital Defended") and `approved_pnl − 368 × $0.45` (→ "−$176.02 Boardroom-OFF ablation"). No simulation existed behind them.

**Real method:** reconstruct would-be swing setups from **vetoed** debates (completed 4h candle close = entry, SL = max(3×ATR14(4h), 1.5%), TP = 4×risk — geometry pinned to `swing_trader.py:1032-1046`, config `SWING_ATR_SL_MULT=3.0`, `SWING_MIN_SL_PCT=0.015`, `SWING_RUNNER_TP_MULT=4.0`), then replay each through the **existing** fee+funding state machine in `backtests/backtest_grid_replay.py` via an injected seed (functions imported and reused — **fill rules not re-implemented**). Notional $255 = $85 × 3× leverage (labeled assumption). Repeat deliberations deduplicated to unique `(symbol, entry_ts, side)`.

**Executed output — `python -X utf8 scripts/evaluate_boardroom_counterfactual.py`:**
```
{
  "n_vetoed": 368,
  "reconstructed": 368,
  "measured": 79,                 <- unique replayable setups after dedup
  "shadow_ev": -13.054094136222268,
  "shadow_ci": -14.958967265118469,
  "approved_ev": -0.3975046663650036,
  "boardroom_value": {
    "diff": 9.432741208667206,
    "n_a": 27, "n_b": 106,
    "lower": 7.581249474696808,
    "upper": 11.194255149433257
  }
}
[OK] Report generated at audits/boardroom_counterfactual_report_2026-10-13.md
```

**Report populations (shadow/backtest estimates; `touch` = upper bound):**
- Vetoed → shadow (would-have-been): **−$13.054/trade**, bootstrap 90% CI **[−14.959, −11.090]**, N=79.
- Approved → executed actuals: **−$0.398/trade**, CI straddles zero, N=27.
- Boardroom value (approval − shadow): **+$9.433/trade**, CI **[+7.581, +11.194]** — the CI **excludes zero**, so at current sample the vetoes are measurably positive-value filtering. (Bootstrap 1000 draws, seed 42 → deterministic, reproduced identically in the FIX 2 re-run.)

**Unit test — `python -X utf8 -m unittest tests.test_counterfactual_shadow`:** 9/9 OK. Includes an exact net-PnL check `(98−100)/100×NOTIONAL − NOTIONAL×TAKER_FEE×2` for a SL taker exit, a TP maker exit, an OPEN_AT_EDGE mark-out, a NO_BAR seed, and the no-lookahead signal-candle selection (bar idx 19 of completed 4h candles).

**Harness regression guard:** `tests.test_grid_replay_consistency` → `Successfully verified consistency across 500 rolling 5m slices.` The default path (`scan_signals=True`, no seeds) is unchanged; `seeded_positions` is additive.

---

## FIX 2 — Model provenance (`served_model_id`)

**Files changed (5):** `ai_engine.py` (+5), `conviction_filter.py` (+13/−5), `db_manager.py` (+40), `tests/test_model_provenance.py` (+127), regenerated report (±1 line).

**Design:** additive nullable `served_model_id TEXT` on `boardroom_debates`. Capture point = the actual answering model at the `red_team_deliberation` return in `ai_engine.py` (`model_used`/`model`/`active_model`), persisted by `conviction_filter._persist_debate_sync`. **Zero decisioning change** — the value is recorded alongside the existing verdict; no read/write path alters execute/score logic. Fail-closed veto stores `served_model_id = None`.

**Live-DB migration evidence — `ensure_boardroom_provenance_schema("trading_ledger.db")` (read-only PRAGMA before/after):**
```
BEFORE cols: ['id','symbol','side','execute','conviction_score','red_team_objection','alpha_thesis','final_rationale','timestamp']
BEFORE served_model_id present: False
BEFORE row count: 556
--- applying ensure_boardroom_provenance_schema ---
result: True
AFTER cols: [...,'timestamp','served_model_id']
AFTER served_model_id present: True
AFTER row count: 556
second call result: True   <- idempotent
third call result: True    <- no duplicate column, no row churn
provenance rows now: 0
```
Row count unchanged (556→556); no rows rewritten or deleted; column appended nullable.

**Honest report header (verified in regenerated report):**
```
**MODEL PROVENANCE: unpinned pool; post-cutoff scoring paused** — 0/50 debates carry `served_model_id`;
decisions cannot yet be attributed to a model training cutoff.
```
The prior fictional `PINNED_MODEL_VERSION="llama-3.1-70b-versatile-2026-10"` is gone; real pool is Gemini flash-lite + Groq gpt-oss-20b failover.

**Acceptance test — `python -X utf8 -m unittest tests.test_model_provenance`:** 5/5 OK. Covers legacy 9-col DB → additive+idempotent migration, existing rows untouched with NULL provenance, **insert + read-back** of a row carrying `served_model_id`, and engine capture propagation (served path reports model; fail-closed reports None).

**Scope note:** `conviction_filter.py` is outside the enumerated write-scope list, but FIX 2's task text explicitly directs capture at the "ai_engine / conviction-pipeline call site that logs debates"; the edit is read-only with respect to decisioning (persists one extra column on an existing INSERT). Flagged here for transparency.

---

## FIX 3 — Depth recorder daemon uptime + staleness health check

**Files changed (3):** `depth_recorder.py` (+70, additive pure probe only), `start_system.py` (+54/−2, supervision wiring), `tests/test_depth_health.py` (+100).

**Supervision wiring (`start_system.py`):**
- `[sys.executable, "depth_recorder.py"]` appended to `subprocess_configs` (gated by `getattr(config,"DEPTH_RECORDER_ENABLED",True)`) → inherits the existing `ManagedProcess` **auto-restart + exponential backoff** (`min(60.0, 5.0*2**restarts)`, cap 15, reset after 5-min stable) — identical path to every other daemon.
- `depth_recorder.py` added to `_recycle_existing_orchestrator` and `_cleanup_stale_children` target lists (adopted/cleaned on relaunch).
- Clean de-registration guard added (mirrors basis/swing) if the flag is later set false.
- **No WS/REST/schema change** to the recorder; retention stays 90 days.

**Staleness health check:** pure `depth_recorder.read_depth_health(now_ts, depth_dir, date_str)` opens the current UTC-day partition via **`file:...?mode=ro`** (no writes) and returns `MISSING|READ_ERROR|EMPTY|STALE_ERROR(>60min)|STALE_WARN(>15min)|FRESH`. `start_system.get_depth_health()` delegates to it and emits `log.warning`/`log.error` at the thresholds, fired on boot and every 5 min in the monitor loop.

**Health check against the real stale store (before any supervised run):**
```
[DEPTH-HEALTH] ERROR: books15 depth stream stale 780.8 min (>60min) — recorder likely down (rows=1470, snapshots=49).
{"status":"STALE_ERROR","age_sec":46847.37,"rows":1470,"snapshots":49,...}
```

**Live accrual acceptance — `depth_recorder.py` run standalone (public, zero-order) for 210 s:**
```
[BASELINE] snapshots=49  rows=1470  max_ts=1791877475767   (~13h stale)
[SAMPLE 1] snapshots=599  rows=17970  max_age_sec=21.0
[SAMPLE 4] snapshots=2756 rows=82680  max_age_sec=22.2
[SAMPLE 7] snapshots=4762 rows=142860 max_age_sec=21.3
[FINAL]    snapshots=4762 (+4713) rows=142860 (+141390)
[HEALTH]   status=FRESH age_sec=21.9
```
**Snapshots advancing across consecutive UTC minutes** (distinct `(timestamp_ms,symbol)` events):
```
20:47 -> 397   20:48 -> 1390   20:49 -> 1458   20:50 -> 1326   (20:51 -> 142, partial: recorder stopped)
```
Four full consecutive minutes each advanced → acceptance satisfied. Health check flipped STALE_ERROR → FRESH within the window (<5 min). Throwaway demo driver + log deleted after the run; no live trading daemon was restarted (only the standalone zero-order recorder was launched and terminated).

**Projection to the TRIAL-003 target (N ≥ 50,000 snapshots across ≥ 14 consecutive days):**
- Measured **22.42 snapshots/s = ~1,937,000 snapshots/day**; **673 rows/s**.
- Sample-size N=50,000 is reachable in **~0.04 days (~37 min) of uptime** at this rate → **not the binding constraint.**
- **The binding constraint is the ≥14-consecutive-day coverage floor** (regime diversity — 37 min of one market state cannot represent 14 days). TRIAL-003 completes at **day 14 of stable supervised uptime**, not when 50k first appears. Projected completion within 14 days **≤ 21 days**, so **no slip** to report — *provided* the supervised recorder stays up; it currently accrues only when `start_system.py` is (re)launched.

**Acceptance test — `python -X utf8 -m unittest tests.test_depth_health`:** 8/8 OK (FRESH / STALE_WARN / STALE_ERROR / 15-min boundary / EMPTY / MISSING / **read-only file-unchanged guard** / snapshot counting).

---

## FIX 4 — Report wording sync (bleeding avoidance, not positive edge)

**Files changed (3):** `audits/grid_ablation_report_2026-10-13.md` (+41/−...), `audits/quant_migration_summary.md`, `audits/prereg_registry.md`.

**Restatements (no measured numbers altered — only interpretation + caveats):**
- Deep-chop (ADX<20) reframed from "outperforms / slightly positive / strictly superior / yield harvester" to **"EV statistically indistinguishable from zero (90% CI straddles $0$)"**; ceiling tightening justified **only as bleeding avoidance** = dodging the **−0.12…−0.23%/trade 20–25 trend-band bleed**.
- §1 claim (b) verdict `MARGINAL/NEGATIVE` → `NOT DEMONSTRATED (CI straddles $0)`.
- `+$0.014` / `+$0.003` deep-chop EVs annotated `indistinguishable from $0`, spreads relabeled `bleed avoided (deep-chop EV ≈ 0, not positive)`.
- **`touch`-fill upper-bound caveat added under every EV-bearing table** (grid report sections 1–6 + 7.2; quant summary findings list + pre-registered arm). Queue-model calibration noted as blocked on TRIAL-003.
- **TRIAL-001 wording aligned; metric target unchanged:** Success Criterion stays *forward bootstrap 90% CI lower bound > 0.0 USD/trade at N ≥ 300*; added an interpretation guard barring allocation increase / mid-sample re-tuning while forward CI_90_lower ≤ 0.

**Executed verification:**
```
caveat count (upper bound): 12   bleeding avoidance mentions: 9   indistinguishable mentions: 5
```
Remaining `positive expectancy` / `strictly superior` matches are the **claim headers** and their **refuting** verdicts / explicit "not positive expectancy" negations — the intended honest state.

---

## Final summary — MEASURED vs NOT MEASURABLE

**MEASURED (executed code / live DB rows):**
- FIX 1: shadow veto EV −$13.054 CI [−14.959, −11.090] (N=79 unique); approved EV −$0.398 (CI straddles 0); boardroom value +$9.433 CI [+7.581, +11.194] excludes zero. Deterministic (seed 42).
- FIX 2: live ledger 556→556 rows, column appended nullable + idempotent; 0/50 provenance rows; report header verified.
- FIX 3: depth 49→4,762 snapshots across 4 consecutive UTC minutes; measured 22.42 snap/s; health check STALE_ERROR↔FRESH reproduced; 8/8 unit tests.
- FIX 4: wording/caveat propagation counts above.

**NOT MEASURABLE — reason:**
- **Post-cutoff per-model boardroom attribution** — NOT MEASURABLE: 0/50 debates carry `served_model_id` (provenance accrues only on next live deliberations through the FIX 2 path). Report header reads "unpinned pool; post-cutoff scoring paused."
- **Queue-model (logprob/squareprob) calibration** — NOT MEASURABLE: needs N≥50,000 books15 snapshots across ≥14 consecutive days (TRIAL-003); today's partition has 4,762. `touch` remains an optimistic **upper bound** until calibrated.
- **Boardroom-OFF live PnL counterfactual** — NOT MEASURABLE: cannot run the counterfactual arm live; only the shadow-replay estimate above is derivable from data, and it is a backtest estimate.
- **Forward (live) EV clearing zero (TRIAL-001 success)** — NOT MEASURABLE yet: n < 300 forward trades and forward CI lower bound ≤ 0; no allocation change permitted.
- **Depth steady-state supervised uptime** — NOT DEMONSTRATED end-to-end: proved via code path + standalone run; full supervised accrual begins on the next legitimate `start_system.py` launch, which was **not** forced to avoid restarting the live trading fleet (hard constraint).

**Constraints honored:** no `ALLOW_LIVE_ORDER_TESTS`, no order placed/cancelled/probed (public OHLCV/funding/depth only); no edit to `super_bot.py` or the live order path; live DBs inspected via `mode=ro`, schema changes additive-only; no `.env`/key values printed; no foreign uncommitted `reconciliation_engine.py` change committed; one commit per FIX, pushed to `origin/main`.

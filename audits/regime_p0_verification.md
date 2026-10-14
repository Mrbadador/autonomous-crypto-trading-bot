# Regime P0 — implementation and verification report

Date: 2026-10-14 (all timestamps UTC unless marked "local")
Repo: `C:\Users\user\.gemini\antigravity\scratch\llama_trading_bot` (live, real-money Bitget UTA bot)
Scope: TASK 1–7 of the "Regime P0" prompt, implemented by this agent, one local commit per numbered fix.
Nothing here is pushed. Nothing here was restarted. **None of these seven commits is live yet** — see §7.

Rules as briefed, and how they were honoured:

| Rule | Status |
|---|---|
| No live orders (never `ALLOW_LIVE_ORDER_TESTS`, no place/cancel/close) | honoured — no exchange write call was made by any probe or test |
| Never run `start_system.py` or any engine, never restart | honoured — 26 fleet processes are untouched (see §7) |
| Read-only DB (`file:…?mode=ro`, `uri=True`); fixtures in a temp dir | honoured — every probe used `mode=ro`; every test fixture used `tempfile.mkdtemp()` |
| Never print `.env` or keys | honoured |
| One commit per numbered fix, local, no amend/rebase | honoured — 7 commits, `git log` verified |
| No threshold / allocation / leverage / config-default changes | honoured — verified: no added or removed line in `regime_classifier.py` / `super_bot.py` / `macro_sentinel.py` touches the `0.40` Hurst, `32.0` ADX, `8.0` DI-spread, `25`/`28` market-ADX constants or any allocation/leverage default. The only config addition is the new `REGIME_MAX_AGE_SEC = 900` key TASK 2 itself specified (env-overridable, no existing default altered). |
| Never `git push` | honoured — see §1, this environment *could* push |

---

## 1. Environment blockers you should know about first

Three things about this workspace contradict the prompt's assumptions. They matter for how much
weight to give everything below.

1. **This "separate research workspace" is a git worktree of the live repo.** It has no own
   `.git` — only a 194-byte `.git` *file* pointing at the parent repo's `.git/worktrees/f60964a8`.
   The prompt's "you have local file access only, never `git push`" is therefore not enforced by
   the environment: from here `git push` would reach `origin`. All commits were made with
   `git commit` only, no `push` was run, and no remote was contacted.
2. **`C:` drive appeared 100 % full (0 bytes free) mid-task**, which broke every tool call for a
   while. Roughly 32 MB was reclaimed from *my own* caches only (`%LOCALAPPDATA%\pip\cache`,
   `%LOCALAPPDATA%\pnpm-cache`, `%TMP%`); no bot data, model, DB or log was deleted.
   **That reading was wrong, or the condition cleared on its own**: re-measured at 21:25Z with
   `shutil.disk_usage("C:/")` the volume is **465.1 GB total, 377.65 GB free (81.2 %)**. Treat the
   disk warning as a git-bash `df` artifact, not a real risk — but see §8 item 7 for why it is
   worth monitoring on a box running WAL-mode SQLite.
3. **There is no usable system `python`** (the `WindowsApps` stub). A CPython 3.12 payload was
   placed in `Python/` inside this workspace so probes could import `pandas`/`numpy`. No `pip`
   install was performed. Side effect: probes that import repo modules write `__pycache__/*.pyc`
   into the **live repo**, which the running trainer daemon logs as `unauthorized, marked reject`
   (observed at 06:29:45Z for `model_calibration_tracker` and `policy_model`). Harmless to the bot's
   behaviour, but it is noise in the live log and it is a write into the repo tree.

Interpreters involved, because it changed one conclusion: this workspace's `python` is 3.12.12 /
SQLite 3.50.4; the bot's own `.venv` is **Python 3.14.3 / SQLite 3.50.4**. Both reject
`UPDATE … ORDER BY … LIMIT` (verified directly in the `.venv`), so any such statement would fail
identically in production — none exists in the tree (checked: no `UPDATE` with `ORDER BY`+`LIMIT`,
and no `regime_entry` column in any live table).

---

## 2. The seven fixes

| # | Commit | Files | New test file | Tests |
|---|---|---|---|---|
| 1 | `37c4279` | `regime_classifier.py`, `executive_briefing.py` | `tests/test_regime_history_append.py` | 8 / 8 OK |
| 2 | `e224961` | `super_bot.py`, `config.py` | `tests/test_regime_fail_closed_routing.py` | 17 / 17 OK |
| 3 | `98f9d04` | `regime_classifier.py` | `tests/test_regime_closed_candles.py` | 16 / 16 OK |
| 4 | `ecc98df` | `regime_classifier.py` | `tests/test_regime_adx_parity.py` | 5 / 5 OK |
| 5 | `3b721fb` | `regime_classifier.py` | `tests/test_ai_payload_validation.py` | 17 / 17 OK |
| 6 | `f358fc7` | `reconciliation_engine.py` | `tests/test_reconciler_ledger_guard.py` | 5 / 5 OK |
| 7 | `a54f2dc` | `macro_sentinel.py`, `ai_engine.py` | `tests/test_macro_fail_closed_chain.py` | 37 / 37 OK |

Pre-existing from the earlier reconciler round: `93f7c8a` (prune only after a confirmed ledger
write) and `345f08a` (test isolation for the event gate).

### TASK 1 — append-only regime history (`37c4279`)
The classifier used to `DELETE` all but one row and `INSERT OR REPLACE` the survivor, so the regime
verdict trail was a single row — no regime-dependent EV study was possible at all. Now it appends,
keeps `timestamp` as the *publication* time, and separates the *bar* time (`bar_end_ts`,
`source_close_ts`) so a late-arriving candle cannot look like a fresh verdict. `symbol` is carried
per row. `executive_briefing.py` reads the newest row per symbol instead of the newest row overall.

Evidence: `tests/test_regime_history_append.py` (8) — appends instead of replacing, legacy row
survives the migration, `bar_end_ts ≠ timestamp` on a backfilled row, per-symbol newest wins.

### TASK 2 — fail-closed regime consumer, new entries only (`e224961`)
`super_bot` treated an absent or stale regime row as permission to trade. It now requires a row
whose `updated_at` is inside `config.REGIME_MAX_AGE_SEC` (new key, default 900 s — matching the
5-minute producer cadence) and blocks **entry** only. Exits, stop/TP management, position recovery
and reconciliation were deliberately left reachable: a fail-closed rule that also blocks exiting
turns a data outage into a forced liquidation at the worst possible price. That distinction is
asserted in the test file, not just in prose.

Evidence: `tests/test_regime_fail_closed_routing.py` (17) — stale row blocks a new entry, missing
row blocks a new entry, same conditions still allow an exit, `CAPITAL_PRESERVATION` path intact.

### TASK 3 — closed candles only + Hurst windowing (`98f9d04`)
The classifier fed the newest, still-forming candle into EMA/ADX/Hurst. Bitget returns that row and
**rewrites it in place** (verified: identical `open_time`, moving `close/high/low`) — so the
verdict was computed on a bar that had not finished, and it re-issued an indistinguishable row for
every intra-bar tick. Hurst was also computed on `N // lag` *disjoint* subsets starting at index 0,
which discards most of the window and depends on the anchor. Now only closed bars are used and the
Hurst estimator uses all overlapping lags.

Measured impact of the Hurst change on 900 rolling 100-bar BTC 15m windows (re-run 06:39Z,
`regime_p0/hurst_label_impact.py`):

```
legacy   mean=0.602  p05=0.415 p25=0.520 p50=0.592 p75=0.673 p95=0.822  share<0.40 = 4.3 %
fixed    mean=0.602  p05=0.443 p25=0.535 p50=0.585 p75=0.670 p95=0.808  share<0.40 = 2.4 %
mean |delta| = 0.0687      gate flips at hurst<0.40 = 43 of 900 windows (4.8 %)
```

Forming-bar effect on the newest closed window: `ADX 13.04 (+DI 20.64 −DI 25.15, spread −4.51)`
with the forming bar vs `ADX 13.30 (+DI 21.23 −DI 24.00, spread −2.77)` without — a 1.74 shift in
DI spread. In this sample both sides stay negative, so no sign flip was reproduced; the spread can
cross zero on other windows and after a restart that is exactly what the router gate keys on.

### TASK 4 — ADX seeds (`ecc98df`)
The Wilder smoothing fallback seeded `+DM/-DM/TR` sums from the *first* bar instead of the first
`period` bars and seeded ATR from one bar. **This fallback is the production path**: `pandas_ta` is
not installed in the bot's own `.venv` (verified: `ModuleNotFoundError`, with `pandas 3.0.2`,
`numpy 2.4.4`, Python 3.14.3), so every live ADX/`+DI`/`-DI` value came out of the hand-written
smoother. The fix is correctness plus visibility: the fallback is now seeded over the first
`period` bars, and the import error is logged instead of swallowed.

Measured, reproducible numbers:
* parity against `pandas_ta` (run in an interpreter that has it): `ADX 25.03190874 vs 25.03190874`,
  `45.06431195 vs 45.06431195`, `15.73895219 vs 15.73895219` on three seeds — worst error
  `2.1e-14`. Same seed also shows why the *bar set* matters more than the smoother: shifting the
  reference by one bar moves ADX by `0.12–1.13`.
* numeric effect of the old seed alone, 8 synthetic 200-bar windows (chop / mild / trend-up /
  trend-down × 3 seeds): `|ΔADX|` median `0.154`, max `0.479`; `|Δ+DI|` max `0.010`;
  **regime-gate crossings caused by the seed bug: 0 / 24 at ADX 25 and 0 / 24 at ADX 28.**

Correction, stated plainly: earlier in this task I reported `21.95` vs `36.40` (a ~14-point ADX gap)
from a probe on a live 1H window. **That number does not reproduce** — the script that produced it
no longer exists in `regime_p0/`, and the surviving parity probes bound the defect's effect at
`|ΔADX| ≤ 0.48` with zero gate crossings. TASK 4 should therefore be read as a correctness and
telemetry fix, not as a demonstrated regime-flip fix. The regime-labelling sensitivity is TASK 3's
forming-bar and Hurst effects, below.

### TASK 5 — AI payload validation (`3b721fb`)
The classifier consumed the local regime-AI verdict with direct `float()`/`.get()` on every field.
A single `null` (a model hiccup, an empty JSON key) raised inside the daemon loop and killed the
cycle, and out-of-range numbers were cached as verdicts. Every numeric field is now bounded
(clamp or reject-to-neutral), booleans go through a strict parser, and the *neutral* fallback never
writes a protective state as if it were a measured one. This also fixed the `float(None)` crash
that was stopping the regime daemon's cycle.

### TASK 6 — reconciler records on-exchange closures (`f358fc7`)
The finalize site issued a generic ledger `INSERT` with `type/qty/balance/action` columns that
`swing_trades` does not have; the insert raised, the P2 guard deferred the prune, and the closure
was **never** recorded — an on-exchange stop-out left no ledger row. Routed through the real swing
column set (`entry/exit`, `size`, `pnl`, `roe_pct`, `holding_hours`, `exit_reason`, fees, funding,
`gates_snapshot`), and the `recent_close` de-dup probe was windowed to 15 minutes so a months-old
ledger row can no longer mark a write as "already done" and prune without recording. `pruned_count`
now increments only on a confirmed prune.

Live shape of that table today (`swing_trades`, 165 rows, newest `2026-10-14T00:14:35Z`):
`pnl` present, `SUM(pnl) = -35.571`, 94 of 165 rows negative, long 120 / short 45, and **`gates_snapshot` is populated in 0 / 165 rows** — the provenance
column exists but nothing has ever been written into it.

### TASK 7 — macro chain (`a54f2dc`) — the one that was actually fail-open

Headline defect: the sentinel *computed* `global_killswitch` from the SPY/DXY packet but wrote it
only inside the `data_ok` branch. When DXY was unresolvable, `data_ok` was false, the regime was
forced to `UNKNOWN`, and `macro_global_killswitch` kept its previous `"0"`. `swing_trader`
compares that key literally against `"1"`, so a SPY −1.42 % dump during a DXY outage armed nothing.

Before/after, same synthetic one-cycle probe (`regime_p0/macro_killswitch_probe.py`, SPY −1.42 %
with DXY unmeasurable; HEAD files restored by md5 after the "before" run):

```
BEFORE (HEAD a54f2dc~1)                    AFTER (fixed)
macro_global_killswitch   = 0              macro_global_killswitch   = 1
macro_data_ok             = 0              macro_killswitch_state    = MEASURED_PARTIAL
macro_regime              = UNKNOWN        macro_regime              = RISK_OFF_KILLSWITCH
macro_killswitch_state    = <absent>       telemetry row             = (1, 'RISK_OFF_KILLSWITCH', -1.42, None)
telemetry row             = (1,'UNKNOWN',None,None)
VERDICT: KILLSWITCH LOST — swing_trader still sees '0'    VERDICT: KILLSWITCH PERSISTED (fail-closed)
```

The rest of TASK 7 applies one asymmetric rule to the whole chain: **a missing or partial
measurement may arm a protection, never discharge one.** Discharge requires a fully valid cycle;
an unreadable latch reads as armed; an AI outage inherits the last protective verdict (via
`synthesis_source` / `anchor_source` markers, so a provider fallback cannot be mistaken for a
"clean" verdict); `bool("false")` can no longer arm the skew circuit breaker; the
`BULL_MOMENTUM_EXPANSION` playbook can no longer *loosen* posture; and the regime cache is chosen by
row freshness instead of dictionary order.

Right-hand-side check on the live system at 06:27Z: `btc_anchor_state = DISTRIBUTION_DUMP_RISK`
and `alt_long_pause = 1` — the protective state the hold logic will now freeze in place during an
AI outage is currently armed.

---

## 3. Acceptance criteria

* `py_compile` over every tracked `.py` file in the repo (`git ls-files`, 259 files — the union of
  root, `backtests/`, `scripts/`, `tests/`, `core/`, `audits/`, `archive/`; the untracked `Python/`
  payload is excluded). Result: **259 OK, 0 FAIL**, written to a temp dir so nothing was imported
  or executed. Per-scope: `backtests/` 18/18, `scripts/` 76/76, `tests/` 48/48.
* Each new test file run individually: **105 tests, all OK** (8 + 17 + 16 + 5 + 17 + 5 + 37).
  The whole `tests/` directory was **not** run — see §9, U6.
* Before/after evidence per fix: §2 (probe outputs for 3, 4, 7; assertion names for 1, 2, 5, 6).
* Live DB untouched by tests: the new fixtures patch `_get_db_path`, `BASE_DIR` and every DB
  connection to a temp dir; `TestNoLiveWrites` asserts every path touched equals one of the two
  fixture DBs inside the temp dir.

---

## 4. What is *not* fixed: the measurement the P0 premise depends on

The premise was "41 % of closed positions entered under CHOP, loss rate 25 % → 51 %". That number
is **not in the live DB and cannot be computed from it**, for three independent reasons, all
verified read-only at 06:30–06:40Z:

1. **No regime-at-entry tag exists on trades.** Every regime-labelled table in
   `trading_data.db`/`trading_ledger.db`: `trades` (365 rows: `TRENDING 241`, `UNKNOWN 64`, blank
   `54`, `RANGING 4`, `DESYNC_CLEANUP 2` — **zero CHOP rows**, newest row `2026-10-11 14:38:26`),
   `legacy_trades` (1147, no CHOP), `mean_reversion_trades` (44: 3 `CHOP_GRID`, 39 blank, 2
   `UNKNOWN`), `swing_trades` (165, **no regime column at all**), `open_positions` (0 rows, no
   regime column). There is no `regime_entry` column in any live table.
2. **The label vocabularies do not match.** The classifier emits `CHOP / BULL_TREND / BEAR_TREND`;
   `super_bot_telemetry` (20 993 rows) mixes both: `CHOP 8545`, `DEFENSIVE_CHOP 5865`,
   `NEUTRAL 4032`, `BEAR_TREND 1761`, `BULL_TREND 420`, `TRENDING 370`; the trade tables emit a
   third set. A per-regime EV join across those tables measures nothing.
3. **No regime history to join to.** `trading_data.db.market_regime` = **1 row**, still on the
   legacy column set (no `symbol`, no `bar_end_ts`, no `source_close_ts`), and
   `trading_ledger.db` has **no `market_regime` table at all**.

The 51–54 % figures do exist — in the synthetic grid **replay** CSVs
(`grid_replay/trades_*.csv`: CHOP loss rate 51.3 %–54.1 % over 1 150–1 581 simulated trades), not in
live fills. Quoting them as live performance was the D5 defect; it is withdrawn and §7 of the
boardroom review's methodology is still unfixed.

Consequence: TASK 1 makes the classifier's *own* verdict history measurable, which is necessary,
but **regime-conditional live EV is still unmeasurable** until (a) a regime-at-entry column is
written at open time, and (b) the vocabulary is unified. Both are schema/behaviour changes outside
this prompt's scope, so they were not attempted.

---

## 5. Dead and advisory-only keys found (no fix applied)

| Key | Finding |
|---|---|
| `super_bot_regime` | read by the sentinel's packet, **no writer anywhere in the tree** — permanently absent |
| `warden_60m_regime` | absent in the live DB; the warden computes its verdict but the cached key has no writer |
| `btc_price` (state) | read every cycle, never written by any code |
| `ai_macro_posture`, `macro_skew_posture` | dashboard/advisory only — no enforcing consumer exists |
| `AGGRESSIVE_EXPANSION` | the only posture an AI could return that the old override *loosened*; removal costs nothing because nothing enforces posture |
| `reconciliation_engine.py:647` | reads `atr` from a `SELECT` list that never returns `atr` |
| `market_regime` (live) | 1 row, legacy schema — proof the running process predates these fixes |

Freshness note: all three regime producers write every ~300 s (`regime_classifier`, `main.py:735-770`
warden block, `live_oracle.py:573`), so `REGIME_MAX_AGE_SEC = 900` is ~3 missed cycles. That budget
is now configurable rather than hard-coded, but the value is a default, not a tuned number.

---

## 6. Standalone consumers not covered

`swing_trader._check_macro_safety` reads `macro_global_killswitch`, `ai_macro_bias`,
`market_regime_label`, `ai_event_playbook` with **no `updated_at` check** — it fail-closes on a DB
*error* but happily trusts a three-day-old row. `conviction_filter.py` (BTC anchor / alt-long
pause) has the same gap. Both were left alone: adding an age check there changes live entry
behaviour beyond what the prompt scoped, and it is a decision you should make with the restart in
front of you. `recovery_agent`'s DCA gate calls the warden live and already fails closed
(`CAPITAL_PRESERVATION`), so its cache is effectively advisory.

---

## 7. Restart requirement — decision is yours, nothing was applied

`start_system.py` (PID 18840, spawned 2026-10-12 06:40 UTC) is still the parent of 25 child
processes: `regime_classifier` 24944, `macro_sentinel` 22252, `super_bot` 26584, `live_oracle`
26060, `reconciliation_engine` 21108, `ofi_daemon` 22224, `asset_scout` 12728, `recovery_agent`
13408, `portfolio_risk_agent` 21492, `post_mortem_agent` 18740, `db_maintenance_daemon` 14384,
`prometheus_exporter` 14432, `antigravity_agent_manager` 23768, plus the streamlit dashboard.
All are ~2 days old, so **all seven commits are inert.** The live producers are still healthy on
the old code (state rows written 06:23:22Z and 06:27:42–48Z, ~5 minutes before the probe).

A restart is required for every one of the seven fixes, and it will additionally:

* run the TASK 1 migration on the **live** `trading_data.db` (`ALTER TABLE ADD COLUMN symbol /
  bar_end_ts / source_close_ts`, index create, and the first `CREATE TABLE` of `market_regime` in
  `trading_ledger.db`, which has no such table today). That is a DDL write to the live DB at boot —
  flagging it explicitly rather than assuming you want it.
* start emitting the new keys (`macro_killswitch_state`, `macro_killswitch_asof`,
  `ai_macro_posture_held`, `ai_macro_posture_asof`, `btc_anchor_held`), all of which are `<absent>`
  now;
* begin **appending** regime rows (~1 per symbol per 5 min instead of 1 row total);
* apply the TASK 3 closed-bar input, which per §2 moves Hurst on 4.8 % of windows across the
  `0.40` boundary and shifts DI spread by ~1.7 — i.e. the router will pick a different lane more
  often at first, with **no threshold re-tuned** (thresholds were off-limits). The 43/900 flip rate
  is the honest size of that surprise.
* make `UNKNOWN`/stale regime actually block new entries (TASK 2), which will reduce trade
  frequency during any classifier outage — that is the intended fail-closed behaviour, not a bug,
  but it will look like the bot going quiet.

Recommendation: restart when you can watch it for one regime cycle (~15 min) and when there is no
open position you rely on the running process to manage. I will not do it.

---

## 8. Suggested next steps (no code written for these)

1. Unified regime vocabulary + a `regime_entry` column written **at open** (by `super_bot` /
   `swing_trader`, not the reconciler), so EV by regime becomes computable. This is the actual P0.
2. Re-run the CHOP grid EV from `market_regime` history only after ≥100 append-only rows exist
   (~8 h per symbol post-restart).
3. Shadow replay (the D5 replacement, FIX 1–4) stays the only place where the 51 % CHOP loss figure
   may be quoted, and it must be labelled simulated.
4. Either wire an enforcing consumer for posture keys or delete them; today they are dashboard text.
5. Fix `reconciliation_engine.py:647` (`atr` never selected) and give `super_bot_regime` /
   `warden_60m_regime` a writer or drop the reads.
6. Decide on the `swing_trader` / `conviction_filter` freshness gap (§6) — same fail-closed rule,
   entry-only, one small commit each.
7. Disk: `C:` measured 81 % free at 21:25Z, so this is a monitoring note, not a live risk — a
   WAL-mode SQLite ledger on a near-full volume would be one.

---

## 9. Unverified / out of scope — stated plainly

* **U1** No end-to-end live-cycle test of the fixes: nothing was restarted, so no fix has been
  observed running against real exchange data. Every claim above is unit-test or read-only-probe
  evidence.
* **U2** TASK 7's AI-hold path was tested with stubbed providers (`AsyncMock`); the real Gemini →
  Groq failover ordering, rate-limit behaviour and prompt-contract compliance are untested.
* **U3** The `REGIME_MAX_AGE_SEC = 900` budget is a reasoned default (3 × the 300 s producer
  cadence), not validated against the observed miss rate of the live producers.
* **U4** The 43/900 Hurst flip and the 1.74 DI-spread shift are single-sample measurements over the
  most recent 1 000 × 15m BTCUSDT bars. No multi-symbol, multi-period sensitivity was run.
* **U5** The boardroom §3 methodology (mixing simulated and live numbers) was **not** touched.
* **U6** `tests/test_scalp_short_engine.py` writes the live `trading_data.db`; combined with
  ~20 unreviewed `verify_*.py` scripts in `tests/`, the directory was deliberately not run as a
  suite. That file is one `pytest` away from corrupting the live ledger.
* **U7** Resolved during writing this report, so no longer a gap: `pandas_ta` is **not installed**
  in the bot's `.venv` (verified directly with that interpreter). The hand-written Wilder fallback
  is what produces every live ADX/DI number, which is why TASK 4 mattered at all — but see TASK 4
  for the corrected, much smaller size of its effect.
* **U8** The disk reading in §1.2 was a git-bash `df` artifact — `shutil.disk_usage` shows 81 % of
  `C:` free, so disk headroom is *not* currently a risk. The `git push` reachability from this
  worktree (§1.1) and the `__pycache__` provenance noise (§1.3) remain real.
* **U9** `market_regime` history depth: after restart the first rows will be back-to-back closed
  bars only; there is no backfill, so the EV study in step 2 waits on wall-clock, not on code.

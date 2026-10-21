# Audit Remediation 2026-10-22 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the confirmed measurement-integrity defects found by the 2026-10-22 external audit so that no reported statistic is produced from invented data, and no live trade is suppressed or sized by a label instead of a number.

**Architecture:** Four of the six tasks repair analytics/audit code paths that currently substitute constants for measurements (`quant_validation.py` Monte Carlo, `scripts/run_gate_ablation.py` verdicts, `backtests/backtest_ablation_harness.py` Sharpe, `backtests/batch_backtest_learner.py` exemplar clusters). One repairs a live exit-path behaviour (`swing_trader.py` anti-whipsaw cooldown) and one repairs documentation (`README.md`). Every fix follows the same rule: absence of data becomes a loud, visible state, never a plausible-looking default.

**Tech Stack:** Python 3, sqlite3 (read-mostly), pandas/numpy for the backtest corpus, pytest for tests. No new dependencies.

**Spec:** the verified findings recorded in project memory `project-external-audit-verification-2026-10-22` and the task-list entries #40–#45. Every file:line anchor in this plan was re-read on branch `fix/audit-remediation-2026-10-22` at base commit `e9af6a7` before writing.

## Global Constraints

- **No remote mutation.** The production fleet runs on EC2. Nothing in this plan may SSH, deploy, restart, or reconfigure the remote box. No `git push` — the user has not authorised pushing, and it is irreversible.
- **No new runtime dependencies.** `requirements.txt` and `requirements-dev.txt` are not touched by any task.
- **Tests never touch live databases.** Production state lives in `trading_data.db`, `trading_ledger.db`, `live_paper_trades.db` at the repo root. Every test builds its own sqlite file under `tmp_path` and uses `monkeypatch.chdir(tmp_path)` so the relative paths the code under test uses (`"trading_data.db"`) resolve to the fixture, not to production. `tests/conftest.py` already exists to stop the suite from acting on the live fleet; do not weaken it.
- **Fail-closed, visibly.** Where code currently substitutes a constant for a measurement, the replacement must surface the absence: raise, or emit an explicit `status: "unavailable"` plus a reason. A silent default is the defect being fixed, so a fix that swaps one silent default for another fails review.
- **Existing call signatures are preserved.** `generate_full_validation_report(force_refresh=False)` is called by `dashboard.py:4307`; the report dict is read by `super_bot.py:550` and `swing_trader.py:1512`, both of which consume `finrl_ensemble_selector.strategy_weights[...].allocation_weight_mult` and clamp to `[0.75, 1.25]`, falling back to `1.0` on any exception. Keys those readers expect must keep existing.
- **Commit hygiene.** Stage only the named files in each step. Never `git add -A` or `git add .`. The branch starts clean, but untracked scratch and DB files exist in the working directory and must never enter a commit.
- **Comments.** Do not add explanatory comments unless a hidden constraint would otherwise surprise a reader. Two places in this plan explicitly call for one; everywhere else, none.
- **Schema reality (verified, do not re-derive):** `swing_trades` columns are `id, timestamp, symbol, side, entry_price, exit_price, size, pnl, roe_pct, holding_hours, exit_reason, notes, intended_entry_price, actual_fill_price, slippage_bps, fee_paid_usd, fee_maker_usd, fee_taker_usd, funding_usd, funding_source, gates_snapshot, status` — **there is no `leverage` column**, and `roe_pct` is populated (non-NULL) for all 181 rows, range −23.746 … +73.636 (percent). `learned_rules` columns are `rule_id, category, rule_text, confidence, created_at` — the primary key is `rule_id`, **not `id`**. `bot_parameters` columns are `id, bot_name, param_name, param_value, last_updated`; its distinct `bot_name` values today are `Berserker, Marl, Mean Reversion, Sniper, berserker, marl, mean_reversion, sniper`.
- **Two premises from the source audit are false and are deliberately NOT implemented.** (1) `post_mortem_agent.py:391` filters `action IN ('sl_hit','tp_hit','partial_tp','closed','close')` against `conn_paper` = `live_paper_trades.db`, where `action` *is* the populated close-label column and there is **no `type` column at all** — switching it to `type` would raise, be swallowed by the `except sqlite3.OperationalError` at `:395`, and silently disable the paper post-mortem path. (2) The claim that `quant_validation.py` uses "SMA-ATR instead of Wilder ATR" is not a fidelity gap: live `swing_trader.py:644` also computes `tr.rolling(window=14).mean()`. Do not "fix" either of these; if an implementer or reviewer proposes doing so, reject with this note.

---

## Task ordering and why

Tasks are sequenced so that code which *changes measurements* lands before prose that *quotes* them.

1. Task 1 (#40) — Monte Carlo data honesty. Highest severity: the report feeds live sizing multipliers.
2. Task 2 (#42) — anti-whipsaw cooldown keyed on PnL. Touches the live exit path; smallest, most valuable.
3. Task 3 (#45) — quarantine the fabricated-scenario and unbounded-parameter writers, make rule selection deterministic.
4. Task 4 (#41) — ablation verdicts derived from confidence intervals.
5. Task 5 (#44) — Sharpe annualization, three lookahead bugs, Ichimoku shift.
6. Task 6 (#43) — README. Last, because it quotes the state the others produce.

---

## Task 1: Make the Monte Carlo validator report only what it actually measured

**Files:**
- Modify: `quant_validation.py:572-620` (`run_trade_sequence_monte_carlo`), `quant_validation.py:680-731` (`generate_full_validation_report`, `__main__` block)
- Modify: `scripts/quant_validation.py` (byte-identical duplicate of the root module)
- Test: `tests/test_quant_validation_mc_data_source.py` (create)

**Interfaces:**
- Produces: `class InsufficientValidationData(RuntimeError)` in `quant_validation`, raised by `run_trade_sequence_monte_carlo` when fewer than `min_trades` real per-trade returns can be loaded. Returns dict gains two keys consumed by later tasks and by humans: `"data_source": "swing_trades"` on the success path, `"status": "ok"`. On the unavailable path `generate_full_validation_report` puts `"monte_carlo": {"status": "unavailable", "data_source": "none", "reason": <str>}` in the report.
- `run_trade_sequence_monte_carlo` keeps its existing signature `(initial_equity: float = 500.0, n_simulations: int = 5000, horizon_trades: int = 60, seed: int = 42)` — verify against the current definition at `:572-576` and do not change parameter names or defaults.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_quant_validation_mc_data_source.py`. The three tests below pin behaviour, not implementation:

```python
import sqlite3
import pytest

from quant_validation import (
    InsufficientValidationData,
    run_trade_sequence_monte_carlo,
)


def _seed_ledger(path, rows, with_balance=True):
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE swing_trades (
               id INTEGER PRIMARY KEY, pnl REAL, entry_price REAL,
               size REAL, roe_pct REAL)"""
    )
    conn.execute("CREATE TABLE system_state (key TEXT PRIMARY KEY, value TEXT)")
    conn.executemany(
        "INSERT INTO swing_trades (id, pnl, entry_price, size, roe_pct) VALUES (?,?,?,?,?)",
        rows,
    )
    if with_balance:
        conn.execute("INSERT INTO system_state VALUES ('hwm_balance', '545.24')")
    conn.commit()
    conn.close()


def test_empty_ledger_raises_instead_of_using_synthetic_returns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed_ledger("trading_data.db", [])
    with pytest.raises(InsufficientValidationData):
        run_trade_sequence_monte_carlo(n_simulations=50, horizon_trades=10)


def test_missing_database_raises_instead_of_using_synthetic_returns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(InsufficientValidationData):
        run_trade_sequence_monte_carlo(n_simulations=50, horizon_trades=10)


def test_reported_sample_and_win_rate_come_from_the_ledger(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = [(i, 1.0, 100.0, 1.0, 10.0) for i in range(1, 9)]
    rows += [(i, -1.0, 100.0, 1.0, -5.0) for i in range(9, 13)]
    _seed_ledger("trading_data.db", rows)
    res = run_trade_sequence_monte_carlo(n_simulations=200, horizon_trades=12)
    assert res["data_source"] == "swing_trades"
    assert res["historical_sample_size"] == 12
    assert res["empirical_win_rate_pct"] == pytest.approx(66.6667, abs=0.01)
    assert res["initial_equity_usdt"] == pytest.approx(545.24)
```

The win rate must come from `roe_pct` signs (8 of 12 positive = 66.67%), *not* from `pnl` signs, so the two are deliberately consistent here; add one extra test where they disagree (`pnl > 0` but `roe_pct < 0`) asserting `roe_pct` wins, and name it `test_roe_pct_is_the_return_series_not_pnl_sign`.

- [ ] **Step 2: Add the fence against the synthetic array ever returning**

In the same file:

```python
def test_source_contains_no_synthetic_return_fallback():
    import pathlib
    src = pathlib.Path("quant_validation.py").read_text(encoding="utf-8")
    assert "0.036, -0.024, 0.042" not in src
    assert "if len(trade_pct_returns) < 10:" not in src
```

This is a regression fence, not the primary proof; Steps 3-5 carry the behaviour.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_quant_validation_mc_data_source.py -v`
Expected: collection error on `ImportError: cannot import name 'InsufficientValidationData'`.

- [ ] **Step 4: Implement the honest data load**

Add the exception next to the other module-level definitions near the top of `quant_validation.py`:

```python
class InsufficientValidationData(RuntimeError):
    """Raised when a validation statistic cannot be computed from real recorded trades."""
```

Replace the body of `run_trade_sequence_monte_carlo` from `initial_equity = 500.0` (`:583`) through the synthetic fallback block ending at `:619` with:

```python
    rng = np.random.default_rng(seed)
    initial_equity = 500.0
    trade_pct_returns: List[float] = []
    min_trades = 10

    if not os.path.exists("trading_data.db"):
        raise InsufficientValidationData(
            "Monte Carlo validation requires trading_data.db in the working directory; none found."
        )

    try:
        with sqlite3.connect("file:trading_data.db?mode=ro", uri=True, timeout=5.0) as conn:
            row_bal = conn.execute(
                "SELECT value FROM system_state WHERE key IN ('live_wallet_balance', 'wallet_balance', 'hwm_balance') LIMIT 1"
            ).fetchone()
            if row_bal and row_bal[0]:
                try:
                    b_val = float(row_bal[0])
                    if b_val > 50.0:
                        initial_equity = b_val
                except (TypeError, ValueError):
                    pass

            rows = conn.execute(
                "SELECT pnl, roe_pct, entry_price, size FROM swing_trades "
                "WHERE pnl IS NOT NULL ORDER BY id DESC LIMIT 200"
            ).fetchall()
    except sqlite3.Error as exc:
        raise InsufficientValidationData(f"Monte Carlo validation could not read trading_data.db: {exc}") from exc

    for pnl, roe_pct, entry_price, size in rows:
        if roe_pct is not None:
            trade_pct_returns.append(float(roe_pct) / 100.0)
            continue
        notional = float(entry_price or 0.0) * float(size or 0.0)
        if notional > 0:
            trade_pct_returns.append(float(pnl) / notional)

    if len(trade_pct_returns) < min_trades:
        raise InsufficientValidationData(
            f"Monte Carlo validation needs at least {min_trades} closed trades with a recorded return; "
            f"found {len(trade_pct_returns)} in swing_trades. Refusing to substitute synthetic returns."
        )
```

The per-trade return is `roe_pct / 100`. `entry_price * size` is the **notional**, used only when `roe_pct` is NULL; there is no leverage column, so the removed `lev_f` divisor and its `or 3.0` default are gone, and the `margin = max(15.0, ...)` floor and the `max(-0.25, min(0.40, ...))` clamp go with them — real returns are no longer truncated toward a fabricated band. Add one short comment above the loop, since the missing column is exactly the kind of constraint that surprises a reader:

```python
    # swing_trades has no leverage column; roe_pct is the recorded return on margin.
```

Then set the two new keys on the returned dict (which already contains `historical_sample_size`, `empirical_win_rate_pct`, `initial_equity_usdt`):

```python
        "status": "ok",
        "data_source": "swing_trades",
```

- [ ] **Step 5: Make the report state unavailability instead of pretending**

In `generate_full_validation_report` (`:693`), wrap only the Monte Carlo call — the bootstrap and ensemble arms are computed independently and must still be produced:

```python
    try:
        mc_res = run_trade_sequence_monte_carlo(n_simulations=5000, horizon_trades=60)
    except InsufficientValidationData as exc:
        mc_res = {"status": "unavailable", "data_source": "none", "reason": str(exc)}
```

In the `__main__` block (`:726-731`), print the unavailability instead of raising a `KeyError` on `mc['n_simulations']`:

```python
    mc = rep["monte_carlo"]
    if mc.get("status") == "unavailable":
        print(f"[MONTE CARLO] UNAVAILABLE — {mc.get('reason')}")
    else:
        print(
            f"[MONTE CARLO N={mc['n_simulations']}] 1/4 Kelly={mc['quarter_kelly_fraction_pct']:.1f}% | "
            f"Median DD={mc['median_max_dd_pct']:.2f}% | MaxDD_95={mc['max_dd_95_pct']:.2f}% | "
            f"MaxDD_99={mc['max_dd_99_pct']:.2f}% | Median Terminal Equity=${mc['terminal_equity_p50']:,.2f}"
        )
```

- [ ] **Step 6: Collapse the duplicate module**

`scripts/quant_validation.py` is a second copy of the whole engine, so a fix applied to one silently diverges from the other. Replace its entire contents with a re-export shim (do not delete the file — a deletion also removes the `python scripts/quant_validation.py` entry point):

```python
"""Compatibility entry point. The validation engine lives at the repository root."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_validation import *  # noqa: F401,F403
from quant_validation import (  # noqa: F401
    InsufficientValidationData,
    generate_full_validation_report,
    run_trade_sequence_monte_carlo,
)

if __name__ == "__main__":
    import quant_validation

    quant_validation.main()
```

The root module currently guards with `if __name__ == "__main__":` inline rather than a `main()` function. If Step 4-5 did not introduce one, extract the existing `__main__` body into `def main() -> None:` at the root module and call it from `if __name__ == "__main__": main()` — that is a mechanical move, no logic change.

- [ ] **Step 7: Run the full test file, then the suite**

Run: `python -m pytest tests/test_quant_validation_mc_data_source.py -v`
Expected: PASS (all tests).
Run: `python -m pytest tests/ -q -x`
Expected: no new failures. Record any pre-existing failure in the report; do not fix unrelated tests.

- [ ] **Step 8: Verify against the real ledger**

Run: `python quant_validation.py`
Expected: a `[MONTE CARLO N=5000]` line whose report shows `historical_sample_size: 181` and `empirical_win_rate_pct` near 44.2 (the real `roe_pct`-derived rate), **not** 20 and 60.0. Confirm with:
`python -c "import json;r=json.load(open('cache/quant_validation_report.json'));mc=r['monte_carlo'];print(mc['data_source'], mc['historical_sample_size'], mc['empirical_win_rate_pct'])"`
Expected: `swing_trades 181 44.2` (±0.1). Note `cache/quant_validation_report.json` is a regenerable artefact; overwriting it is expected.

- [ ] **Step 9: Commit**

```bash
git add quant_validation.py scripts/quant_validation.py tests/test_quant_validation_mc_data_source.py
git commit -m "fix(validation): compute Monte Carlo risk from recorded ROE, refuse synthetic returns"
```

---

## Task 2: Key the anti-whipsaw cooldown on realised PnL, not on the exit label

**Files:**
- Modify: `swing_trader.py:3069` and `swing_trader.py:246`
- Test: `tests/test_swing_cooldown_pnl_keyed.py` (create)

**Interfaces:**
- Consumes: `self._set_cooldown_sync(symbol, hours, reason)` and `self._resolve_cooldown_until(...)` exactly as they exist. Only the *predicate that decides whether a cooldown applies* changes.
- Produces: no new public names. Behaviour: a closed swing trade triggers the cooldown **iff its realised PnL is negative**.

**Measured impact (from the real ledger, 181 closed swing trades):** 24 profitable exits — 22 labelled `STOP_LOSS`, 2 labelled `sl_hit` — triggered a 12-hour anti-whipsaw cooldown purely because of their label. That is 13% of the ledger suppressing re-entry into trades that made money. Conversely the label test is not needed to catch losers: `pnl < 0` alone covers all 105 losing trades, and the label test catches `breakeven_hit` (10 rows, net −$0.44) which is not a loss worth cooling.

- [ ] **Step 1: Write the failing test**

Create `tests/test_swing_cooldown_pnl_keyed.py`:

```python
import pytest


@pytest.mark.parametrize(
    "reason, pnl, should_cooldown",
    [
        ("STOP_LOSS", -1.25, True),
        ("AI_STRUCTURAL_INVALIDATION", -3.0, True),
        ("SCALP_SHORT_TIME_EXPIRY", -0.5, True),
        ("STOP_LOSS", 0.75, False),
        ("sl_hit", 0.10, False),
        ("breakeven_hit", 0.0, False),
        ("TAKE_PROFIT", 2.5, False),
    ],
)
def test_cooldown_predicate_is_pnl_keyed(reason, pnl, should_cooldown):
    from swing_trader import _cooldown_should_trigger

    assert _cooldown_should_trigger(reason, pnl) is should_cooldown
```

Before writing it, grep `swing_trader.py` for `_cooldown_should_trigger` to confirm the name is free; if the module already defines something with that name, choose `should_apply_whipsaw_cooldown` and update both call sites and the test consistently.

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_swing_cooldown_pnl_keyed.py -v`
Expected: FAIL with `ImportError: cannot import name '_cooldown_should_trigger'` (or `cannot import name 'swing_trader'` if the module import pulls in network config — see Step 3 note).

- [ ] **Step 3: Add the module-level predicate and use it**

`swing_trader.py` is a large async module whose import may require configuration. Put the predicate at module top level (next to the other module-level helpers, before `class SwingTrader`) so it is importable without instantiating the trader; if importing `swing_trader` in a test triggers side effects, the test may instead import the function via `importlib` after monkeypatching, or the predicate may be placed in the lowest-dependency module the two call sites already import — state the choice in the report rather than silently restructuring the file.

```python
def _cooldown_should_trigger(exit_reason: str, pnl: float) -> bool:
    """Anti-whipsaw cooldown applies to realised losses only, never to a label."""
    return pnl < 0
```

At `:3069`, replace:

```python
        if "stop" in reason.lower() or "sl" in reason.lower() or pnl < 0:
```

with:

```python
        if _cooldown_should_trigger(reason, pnl):
```

At `:246`, replace:

```python
                    is_stop_or_loss = ("stop" in exit_reason.lower() or "sl" in exit_reason.lower() or last_pnl < 0)
```

with:

```python
                    is_stop_or_loss = _cooldown_should_trigger(exit_reason, last_pnl)
```

`is_stop_or_loss` keeps its name (it is read at `:247`); rename it to `should_apply_cooldown` only if the rename is limited to these two lines, and say so in the report.

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_swing_cooldown_pnl_keyed.py -v`
Expected: PASS, 7 passed.

- [ ] **Step 5: Confirm no other site keys a cooldown on a label**

Run: `grep -n '"stop" in\|"sl" in' swing_trader.py super_bot.py`
Expected: no remaining cooldown predicates. A hit in an unrelated context (log text, order-type construction) is acceptable — list each remaining hit and what it does in the report; do not change one that is not a cooldown gate.

- [ ] **Step 6: Regression-check the suite and commit**

Run: `python -m pytest tests/ -q -k "cooldown or swing"`
Expected: no new failures.

```bash
git add swing_trader.py tests/test_swing_cooldown_pnl_keyed.py
git commit -m "fix(swing): trigger anti-whipsaw cooldown on realised loss, not exit label"
```

---

## Task 3: Quarantine invented training data, close the unbounded parameter path, make rule selection deterministic

**Files:**
- Modify: `backtests/batch_backtest_learner.py:4` (docstring), `:92-134` (`extract_historical_scenarios`), `:192-199` (rule insert), `:202-213` (parameter writer)
- Modify: `scripts/deep_brain_trainer.py:250-257` (rule insert), `:259-270` (parameter writer)
- Modify: `post_mortem_agent.py:60-81` (`enforce_parameter_bounds`)
- Modify: `ai_engine.py:1049`
- Test: `tests/test_rule_and_param_hygiene.py` (create)

**Context the implementer needs:** `learned_rules` holds 579 rows, every one at confidence ≥ 0.70, and `ai_engine._get_top_rules_sync` selects `ORDER BY confidence DESC LIMIT 15` with no tiebreak — so the 15 rows injected into every boardroom prompt are whatever sqlite happens to return first among hundreds of ties, and the ordering is not stable across writes. The three earliest rows (`rule_id` 1-3, dated 2026-08-15) carry text that matches hand-written exemplars in `batch_backtest_learner.py` almost word for word (`atr_mult: 3.2, dxy_move: "+0.45%"` → "ATR > 2.5x and DXY move > 0.4%"). These modules are **not imported by the running fleet** — `batch_backtest_learner` is reachable only as a script and `deep_brain_trainer` only as a script — so this is quarantine, not a live-fire exercise. Do not delete rows from `trading_ledger.db`: purging production data is the user's call, and this task makes the writers honest instead.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rule_and_param_hygiene.py`:

```python
import sqlite3

import pytest


def test_unknown_parameter_is_never_mutated():
    from post_mortem_agent import enforce_parameter_bounds

    assert enforce_parameter_bounds("trail_dist", 2.0, 5.0) == 2.0
    assert enforce_parameter_bounds("veto_threshold", 50.0, 200.0) == 95.0
    assert enforce_parameter_bounds("stop_loss_pct", 0.97, -0.5) == 0.95
    assert enforce_parameter_bounds("made_up_knob", 1.0, 999.0) == 1.0


def test_unparsable_suggestion_is_rejected():
    from post_mortem_agent import enforce_parameter_bounds

    assert enforce_parameter_bounds("veto_threshold", 50.0, "not a number") == 50.0


def test_scenarios_are_labelled_exemplars_not_history():
    import pathlib
    src = pathlib.Path("backtests/batch_backtest_learner.py").read_text(encoding="utf-8")
    assert "Runs a fast historical backtest trade extraction" not in src
    assert "synthetic exemplar" in src.lower()


def test_learner_writes_do_not_shadow_existing_bot_names(tmp_path):
    db = tmp_path / "trading_data.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE bot_parameters (id INTEGER PRIMARY KEY, bot_name TEXT,"
        " param_name TEXT, param_value REAL, last_updated TEXT,"
        " UNIQUE(bot_name, param_name))"
    )
    conn.execute("INSERT INTO bot_parameters (bot_name, param_name, param_value) VALUES ('sniper','trail_dist',2.0)")
    conn.commit()
    conn.close()

    from batch_backtest_learner import write_parameter_adjustments

    write_parameter_adjustments(str(db), [{"bot_name": "SNIPER", "param_name": "trail_dist", "suggested_value": 1.0}])

    conn = sqlite3.connect(db)
    names = [r[0] for r in conn.execute("SELECT bot_name FROM bot_parameters").fetchall()]
    conn.close()
    assert names == ["sniper"]
```

`write_parameter_adjustments` does not exist yet — Step 5 creates it by extracting the loop at `:202-213`. If extraction proves more invasive than moving those 12 lines into a function of that name, report the deviation rather than inventing a second writer.

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_rule_and_param_hygiene.py -v`
Expected: Step-1 tests FAIL (the fallthrough returns the unbounded value; the docstring is still the false one; the import of `write_parameter_adjustments` fails).

- [ ] **Step 3: Close the bounds fallthrough**

In `post_mortem_agent.py`, replace `:81`:

```python
    return new_val
```

with an explicit refusal, so an unrecognised knob can never be written by an LLM suggestion:

```python
    # Unlisted knobs have no measured safe range: refuse rather than write an unbounded value.
    return current_val
```

Also tighten the type guard so a non-numeric suggestion cannot slip through the `except` branch and be written by the caller: `:61-65` already returns `current_val` when the *inputs* are unparsable — verify the caller at `:180-190` and `:285-295` writes only what this function returns, and if either caller re-applies `suggested_adj` itself, remove that so the clamped value is the only value persisted.

- [ ] **Step 4: Rename the function and label its output honestly**

In `backtests/batch_backtest_learner.py`, rename `extract_historical_scenarios` to `exemplar_scenarios` and update its call sites within the file. Replace the module docstring line 4 and the function docstring so the file says what it is:

```python
"""Hand-written exemplar scenarios used to seed the rule-learning loop.

These are illustrative trades authored in this file, not trades extracted from any
backtest or live ledger. Rules distilled from them are tagged
source='synthetic_exemplar' so they can be told apart from measured lessons.
"""
```

Do not delete the clusters: with the module unreachable from the fleet and its output now labelled, removing them would also remove the seed of the only test that keeps the labelling honest.

- [ ] **Step 5: Tag rule writes with a source, and stop shadow-writing parameters**

Add a `source` column defensively — it must not break a database that lacks it, and the live reader must never select on it:

```python
def ensure_learned_rules_source_column(conn):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(learned_rules)")}
    if "source" not in cols:
        conn.execute("ALTER TABLE learned_rules ADD COLUMN source TEXT DEFAULT 'measured'")
```

Call it before the insert at `batch_backtest_learner.py:192` and `scripts/deep_brain_trainer.py:250`, and extend both inserts to write the source:

```python
            conn.execute("""
                INSERT INTO learned_rules (category, rule_text, confidence, created_at, source)
                VALUES (?, ?, ?, datetime('now'), 'synthetic_exemplar')
            """, (
                cluster_key,
                data.get("defense_rule", "Strict risk management enforced."),
                float(data.get("confidence_score", 0.90))
            ))
```

For the parameter writers: extract `batch_backtest_learner.py:202-213` into the module-level `write_parameter_adjustments(db_path, adjustments)` from Step 1, and inside it (a) normalise `bot = str(adj.get("bot_name", "sniper")).strip().lower().replace(" ", "_")` so a suggestion for `SNIPER` updates `sniper` instead of creating a shadow row, and (b) route the value through `post_mortem_agent.enforce_parameter_bounds` before writing, dropping any adjustment for a knob that module does not bound. Apply the same two changes inline at `scripts/deep_brain_trainer.py:259-270`.

- [ ] **Step 6: Make the live rule selection deterministic and recency-aware**

At `ai_engine.py:1049`, replace:

```python
                        "SELECT category, rule_text, confidence FROM learned_rules WHERE confidence >= 0.70 ORDER BY confidence DESC LIMIT 15"
```

with:

```python
                        "SELECT category, rule_text, confidence FROM learned_rules "
                        "WHERE confidence >= 0.70 "
                        "ORDER BY confidence DESC, created_at DESC, rule_id DESC LIMIT 15"
```

`rule_id` and `created_at` are the real column names (there is no `id`). The selection stays confidence-ordered; newest wins among ties, so a measured lesson written today displaces a 2025 exemplar instead of being starved by it. Do not exclude `'synthetic_exemplar'` rows from this query — the boardroom block is advisory text (`:1064` says explicitly "NOT as rigid mechanical vetoes"), and silently filtering the prompt is a different decision than the one this plan authorises.

- [ ] **Step 7: Run tests and commit**

Run: `python -m pytest tests/test_rule_and_param_hygiene.py -v`
Expected: PASS.
Run: `python -m pytest tests/ -q -k "rule or param or post_mortem or ai_engine"`
Expected: no new failures.

```bash
git add post_mortem_agent.py ai_engine.py backtests/batch_backtest_learner.py scripts/deep_brain_trainer.py tests/test_rule_and_param_hygiene.py
git commit -m "fix(learning): label exemplar rules, bound parameter writes, order rule selection by recency"
```

---

## Task 4: Derive gate-ablation verdicts from the confidence intervals the harness already computes

**Files:**
- Modify: `scripts/run_gate_ablation.py:91-119`
- Test: `tests/test_gate_ablation_verdicts.py` (create)

**Interfaces:**
- Consumes: the arm dicts returned by `backtest_ablation_harness.run_ablation_simulation`, whose keys are `n_trades, win_rate, total_return, profit_factor, sharpe, ev_mean, ev_ci_90, gate_fires, trades` (verified at `:323-333`), where `trades` is a list of dicts each carrying `net_pnl`.
- Produces: `verdict_for_gate(full, ablated)` returning a `(verdict, evidence)` string pair consumed by the report writer.

**The defect:** lines `:93`, `:103`, `:111`, `:115` write literal verdicts — `"KEEP (CRITICAL ALPHA DEFENSE)"`, `"KEEP (POSITIVE STATISTICAL FILTER)"`, `"RETAIN AS STRUCTURAL TAIL-RISK BRAKES"`, `"KEEP"` — regardless of what the bootstrap found. The numbers interpolated into the surrounding prose are real; the conclusion is hardcoded. Every arm's EV confidence interval straddles zero, so the honest verdict is `INCONCLUSIVE` in all four sections.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gate_ablation_verdicts.py`:

```python
import pytest


def _arm(pnls, ci=None):
    return {"n_trades": len(pnls), "trades": [{"net_pnl": p} for p in pnls],
            "ev_mean": 0.0, "ev_ci_90": ci or (0.0, 0.0)}


def test_gate_that_clearly_raises_ev_is_keep():
    from run_gate_ablation import verdict_for_gate
    full = _arm([0.5] * 40 + [-0.4] * 40)
    ablated = _arm([-0.5] * 40 + [0.4] * 40)
    verdict, evidence = verdict_for_gate(full, ablated)
    assert verdict.startswith("KEEP")
    assert "ΔEV" in evidence


def test_gate_that_clearly_lowers_ev_is_rethink():
    from run_gate_ablation import verdict_for_gate
    full = _arm([-0.5] * 40 + [0.4] * 40)
    ablated = _arm([0.5] * 40 + [-0.4] * 40)
    verdict, _ = verdict_for_gate(full, ablated)
    assert verdict.startswith("RETHINK")


def test_noisy_difference_is_inconclusive():
    from run_gate_ablation import verdict_for_gate
    full = _arm([0.1, -0.1] * 40)
    ablated = _arm([0.1, -0.1] * 40)
    verdict, evidence = verdict_for_gate(full, ablated)
    assert verdict == "INCONCLUSIVE"
    assert "straddles" in evidence.lower()


def test_underpowered_arms_are_inconclusive():
    from run_gate_ablation import verdict_for_gate
    verdict, evidence = verdict_for_gate(_arm([0.9, 0.9]), _arm([-0.9, -0.9]))
    assert verdict == "INCONCLUSIVE"
```

**Verified import mechanism (do not re-derive):** `scripts/` has no `__init__.py`, but Python 3 namespace packages make `from scripts.run_gate_ablation import ...` work once the repo root is on `sys.path` — which `tests/conftest.py:33-34` already does, and `tests/test_counterfactual_shadow.py:24` proves the pattern with `from scripts.evaluate_boardroom_counterfactual import ...`. So use the fully-qualified `from scripts.run_gate_ablation import verdict_for_gate` form (the bare `from run_gate_ablation import ...` above is shorthand — qualify it), and do not add another `sys.path` insert unless collection actually fails.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_gate_ablation_verdicts.py -v`
Expected: FAIL — `ImportError: cannot import name 'verdict_for_gate'`.

- [ ] **Step 3: Implement the paired-resample verdict**

Add to `scripts/run_gate_ablation.py` above the report writer:

```python
def _bootstrap_delta_ev(full_pnls, ablated_pnls, n_boot=1500, seed=42, alpha=10.0):
    rng = np.random.default_rng(seed)
    if len(full_pnls) < 5 or len(ablated_pnls) < 5:
        return None, None
    deltas = []
    for _ in range(n_boot):
        f = rng.choice(full_pnls, size=len(full_pnls), replace=True)
        a = rng.choice(ablated_pnls, size=len(ablated_pnls), replace=True)
        deltas.append(float(np.mean(f) - np.mean(a)))
    return float(np.percentile(deltas, alpha / 2)), float(np.percentile(deltas, 100 - alpha / 2))


def verdict_for_gate(full, ablated):
    full_pnls = [t["net_pnl"] for t in full.get("trades", [])]
    ablated_pnls = [t["net_pnl"] for t in ablated.get("trades", [])]
    lo, hi = _bootstrap_delta_ev(full_pnls, ablated_pnls)
    delta = float(np.mean(full_pnls) - np.mean(ablated_pnls)) if full_pnls and ablated_pnls else 0.0
    evidence = (
        f"ΔEV (gate ON minus gate OFF) = {delta * 100:+.3f}% per trade, "
        f"90% CI [{(lo or 0) * 100:+.3f}%, {(hi or 0) * 100:+.3f}%] "
        f"over N={len(full_pnls)}/{len(ablated_pnls)} trades"
    )
    if lo is None:
        return "INCONCLUSIVE — both arms have fewer than 5 trades; ΔEV is not estimable", evidence
    if lo > 0:
        return "KEEP — removing this gate lowers expected value and the 90% CI excludes zero", evidence
    if hi < 0:
        return "RETHINK — removing this gate raises expected value and the 90% CI excludes zero", evidence
    return "INCONCLUSIVE — the 90% CI for ΔEV straddles zero; this sample cannot measure the gate's contribution", evidence
```

`np` must be imported in `run_gate_ablation.py`; if it is not already, add `import numpy as np` to the existing import block. Then replace the four hardcoded verdict writes (`:93`, `:103`, `:111`, `:115`) and their prose with a loop over gate sections, each defined by `(heading, param_name, full_row, ablated_row)`, calling `verdict_for_gate` and writing both the returned verdict and evidence. Keep the four section headings; replace the assertions in their bodies. Sections whose ablated arm is missing from `all_results` must yield `INCONCLUSIVE — the ablation arm did not run`, not a verdict borrowed from a neighbour. Append one line to the report footer:

```python
        f.write("Confidence intervals are an iid bootstrap over closed trades and do not correct for serial dependence between consecutive entries.\n")
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_gate_ablation_verdicts.py -v`
Expected: PASS, 4 passed.

- [ ] **Step 5: Do not regenerate the report in this task**

Re-running `scripts/run_gate_ablation.py` needs `data/historical/*_4h.csv` for every symbol in its sweep and takes minutes; the numbers it would produce change again in Task 5. Record in the report that the next run of the script will overwrite `audits/gate_ablation_report_<date>.md` with derived verdicts, and leave the existing dated file untouched.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_gate_ablation.py tests/test_gate_ablation_verdicts.py
git commit -m "fix(audit): derive gate-ablation verdicts from bootstrap delta-EV instead of hardcoded strings"
```

---

## Task 5: Fix the Sharpe annualization and the three lookahead bugs

**Files:**
- Modify: `backtests/backtest_ablation_harness.py:181, 311, 323-333`
- Modify: `backtests/seven_year_backtest.py:176-181, 216-221`
- Modify: `backtests/seven_year_ai_sharpened_backtest.py:162-167, 194-199`
- Modify: `scripts/compare_trigger_modes_840k.py:119-120, 133`
- Modify: `quant_validation.py:77-88`
- Test: `tests/test_measurement_fidelity.py` (create)

**Interfaces:**
- Produces: `annualized_sharpe(pnls, bars, hours_per_bar)` in `backtests/backtest_ablation_harness.py` — a pure function so the arithmetic is testable without the 2-year CSV corpus.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_measurement_fidelity.py`:

```python
import re
from pathlib import Path

import numpy as np
import pytest

# Verified: repo root is on sys.path via tests/conftest.py:33-34, and
# tests/test_gate_toggle_wiring.py:15 already imports `from backtests.backtest_ablation_harness import ...`
# as a namespace package. Use that qualified form; add no extra sys.path handling.
from backtests.backtest_ablation_harness import annualized_sharpe


def test_sharpe_annualizes_by_measured_trade_frequency():
    pnls = [0.01, -0.005] * 10
    bars = 2190 * 2  # two years of 4H bars
    per_year = annualized_sharpe(pnls, bars=bars, hours_per_bar=4.0)
    mean, std = np.mean(pnls), np.std(pnls)
    trades_per_year = 20.0 / 2.0
    assert per_year == pytest.approx((mean / std) * np.sqrt(trades_per_year), rel=1e-6)


def test_sharpe_of_dense_and_sparse_trade_series_differ():
    pnls = [0.01, -0.005] * 10
    dense = annualized_sharpe(pnls, bars=2190, hours_per_bar=4.0)
    sparse = annualized_sharpe(pnls, bars=2190 * 20, hours_per_bar=4.0)
    assert dense > sparse


def test_degenerate_inputs_annualize_to_zero():
    assert annualized_sharpe([], bars=2190, hours_per_bar=4.0) == 0.0
    assert annualized_sharpe([0.01], bars=2190, hours_per_bar=4.0) == 0.0
    assert annualized_sharpe([0.01, -0.01], bars=0, hours_per_bar=4.0) == 0.0


@pytest.mark.parametrize("relpath", [
    "backtests/seven_year_backtest.py",
    "backtests/seven_year_ai_sharpened_backtest.py",
])
def test_runner_tests_carried_stop_before_ratcheting_it(relpath):
    """The intrabar lookahead fix is an ordering property; guard the order."""
    src = Path(relpath).read_text(encoding="utf-8").splitlines()
    ratchet = [i for i, ln in enumerate(src) if re.search(r"^\s*sl_px = (max|min)\(sl_px,", ln)]
    assert ratchet, f"{relpath}: no trailing ratchet found"
    for i in ratchet:
        window = src[i + 1:i + 4]
        assert not any(re.search(r"if (low <= sl_px|high >= sl_px):", ln) for ln in window), (
            f"{relpath}:{i + 1} ratchets sl_px and then tests it against the same bar"
        )
```

**Why the third test is a guard rather than a behavioural test (deliberate, review this):** the ratchet-and-test sequence lives inside `run_full_history_simulation` in a nested `if in_pos → if side == 1 → else:` block that mutates `balance`, `trades`, `qty`, `sl_px` and `in_pos` as loop locals (`backtests/seven_year_backtest.py:144-186`, and `:206-226` for shorts). Extracting it into a pure function to make it behaviourally testable is a real restructure of a research script's state machine — outside this plan's scope and not worth the regression risk for a script that trades nothing. So Step 4 is verified by this ordering guard plus the reviewer reading the reorder; **nobody behaviourally proves the reorder changes results until the corpus is re-run**, which is already listed as excluded follow-up work. Say so plainly in the report; do not quietly claim the fix is empirically validated.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_measurement_fidelity.py -v`
Expected: FAIL — `annualized_sharpe` does not exist; the Sharpe factor is a constant.

- [ ] **Step 3: Annualize by measured frequency**

Add to `backtests/backtest_ablation_harness.py`:

```python
def annualized_sharpe(pnls, bars, hours_per_bar):
    """Annualize a per-trade Sharpe by the trade frequency the run actually produced."""
    if len(pnls) < 2 or bars <= 0:
        return 0.0
    years = (bars * hours_per_bar) / (365.0 * 24.0)
    if years <= 0:
        return 0.0
    trades_per_year = len(pnls) / years
    return float(np.mean(pnls) / (np.std(pnls) + 1e-9) * np.sqrt(trades_per_year))
```

Replace `:311` with a call that uses the bar count already in scope (`total_bars`, `:181`) and the 4H timeframe the harness replays:

```python
    sharpe = annualized_sharpe(pnls, bars=total_bars, hours_per_bar=4.0)
```

Add to the returned dict (`:323-333`) so a reader can see the basis rather than infer it:

```python
        "sharpe_trades_per_year": round(len(pnls) / max((total_bars * 4.0) / 8760.0, 1e-9), 2),
```

- [ ] **Step 4: Fix the intrabar trailing-stop lookahead (four sites)**

In `backtests/seven_year_backtest.py` the long runner at `:176-181` raises the stop from the current bar's close and then tests that raised stop against the current bar's low — exits happen up to one bar later than reality, favourably. Reorder so the carried stop is tested first:

```python
                else:
                    if low <= sl_px:
                        pnl_run = qty * (sl_px - entry_px) - (qty * sl_px * FEE_TAKER)
                        balance += pnl_run
                        trades.append({"symbol": symbol, "side": "long", "pnl": pnl_run, "exit": "runner"})
                        in_pos = False
                    else:
                        sl_px = max(sl_px, price - trail_distance)
                        if high >= tp_px:
                            pnl_tp = qty * (tp_px - entry_px) - (qty * tp_px * FEE_MAKER)
                            balance += pnl_tp
                            trades.append({"symbol": symbol, "side": "long", "pnl": pnl_tp, "exit": "runner_tp"})
                            in_pos = False
```

Apply the mirror to the short branch at `:216-221` (`if high >= sl_px:` first, then `sl_px = min(sl_px, price + trail_distance)`, then `elif low <= tp_px:`). Apply the identical two swaps to `backtests/seven_year_ai_sharpened_backtest.py:162-167` and `:194-199`. Keep the SL-before-TP precedence inside a bar — when both are touched it assumes the worse fill, which is conservative; do not change it.

- [ ] **Step 5: Fix the higher-timeframe resample lookahead**

`scripts/compare_trigger_modes_840k.py:119-120` resamples an hourly series onto 5-minute bins and forward-fills, so a bar timestamped `T` — whose close is only known at `T+1h` — is visible from `T+5m`. Shift by one HTF period before the reindex instead:

```python
    bias_1h  = ind_1h.apply(htf_bias, axis=1).shift(1).reindex(ind_5m.index, method="ffill").fillna(0)
    conf_15m = ind_15m.apply(mtf_confirm, axis=1).shift(1).reindex(ind_5m.index, method="ffill").fillna(0)
```

`.shift(1)` on the hourly/15-minute index moves each value back by exactly one of its own periods, which is what "only completed HTF bars may be used" means. Apply the same to line `:133`'s 15m EMA export:

```python
    return signal.shift(1).fillna(0), ind_15m["ema_20"].shift(1).reindex(ind_5m.index, method="ffill")
```

- [ ] **Step 6: Make the validator's cloud match the live engine's**

`quant_validation.py:80-83` compares today's close against a cloud built from today's highs and lows. The live engine (`swing_trader.py:627-635`) projects the spans 26 bars forward, so the cloud at the current bar was drawn 26 bars ago. Mirror the live guard:

```python
    span_a_raw = (df["tenkan"] + df["kijun"]) / 2.0
    span_b_raw = (high.rolling(52).max() + low.rolling(52).min()) / 2.0
    if len(df) >= 78:
        span_a = span_a_raw.shift(26)
        span_b = span_b_raw.shift(26)
    else:
        span_a = span_a_raw
        span_b = span_b_raw
    df["cloud_top"] = np.maximum(span_a, span_b)
    df["cloud_bottom"] = np.minimum(span_a, span_b)
```

And at `:88`, drop the back-fill so early bars stay NaN instead of being assigned a fabricated ATR:

```python
    df["atr"] = tr.rolling(14).mean()
```

Trace the consequence: any gate that reads `cloud_top`/`cloud_bottom`/`atr` now sees NaN in the first 78/14 bars. Confirm those gates treat NaN as "no entry" (fail-closed) rather than "compare and pass"; if one compares NaN directly, add an explicit `isna()` guard in that gate and list it in the report.

- [ ] **Step 7: Make silently-skipped arms visible**

`quant_validation.py:385` asks for `SOLUSDT_4h_full.csv`, `ADAUSDT_4h_full.csv`, `AVAXUSDT_4h_full.csv`, `NEARUSDT_4h_full.csv`; none exist in `data/historical/` (only `BTCUSDT_4h.csv`, `ETHUSDT_4h.csv`, `SOLUSDT_4h.csv`), so the swing arm `continue`s at `:466-467` and contributes nothing while the report looks complete. Where the arm file list is iterated, collect the names that did not resolve and include them in the bootstrap result as `"missing_datasets": [...]`, then print them in the `__main__` block:

```python
    if v.get("missing_datasets"):
        print(f"   [DATA GAP] {k}: no dataset found for {v['missing_datasets']}")
```

This makes the gap a reported fact. Provisioning the missing history is **not** in this plan — it needs a data download the user must authorise.

- [ ] **Step 8: Run tests and commit**

Run: `python -m pytest tests/test_measurement_fidelity.py tests/test_quant_validation_mc_data_source.py -v`
Expected: PASS.
Run: `python -m pytest tests/ -q`
Expected: no new failures.

```bash
git add backtests/backtest_ablation_harness.py backtests/seven_year_backtest.py backtests/seven_year_ai_sharpened_backtest.py scripts/compare_trigger_modes_840k.py quant_validation.py tests/test_measurement_fidelity.py
git commit -m "fix(backtests): annualize Sharpe by measured frequency and remove three lookahead bugs"
```

---

## Task 6: Correct the README's metrics and remove the methodology claims nothing implements

**Files:**
- Modify: `README.md:28-29, 64, 66, 77, 81, 85-87`

**What was verified about the README, and why the fix is narrow:** every number in the live-telemetry table reproduces to the cent from the first 149 rows of `swing_trades` (2026-09-20 → 2026-10-12): 149 trades, +$77.64 / −$112.08, net −$34.44, 40.3% (60W/89L), 3.59h average hold, max drawdown −$64.56. The table is an honest snapshot that was never refreshed, not an invention. Only two cells are wrong, and they are overstated: the payoff ratio is **1.027** in that window (avgWin $1.294 / avgLoss $1.259), not 1.47, and **0.942** over the current 181 trades; take-profit efficiency is **88.4%** — 5 of 43 `tp_hit`-labelled exits closed negative — not 100%. Separately, four methodology claims have no implementation anywhere in the repository: `walk_forward`, `TimeSeriesSplit`, `train_test_split` and `embargo` each return zero hits across all `.py` files, and "180-day market regimes" appears only in README.md.

- [ ] **Step 1: Re-measure before writing, do not copy this plan's numbers blind**

Run:

```bash
python - <<'EOF'
import sqlite3
c = sqlite3.connect("file:trading_data.db?mode=ro", uri=True)
rows = c.execute("SELECT pnl, roe_pct, entry_price, size, exit_reason FROM swing_trades WHERE pnl IS NOT NULL ORDER BY id").fetchall()
first149 = rows[:149]
for label, rs in (("first149", first149), ("all", rows)):
    wins = [p for p, *_ in rs if p > 0]
    losses = [p for p, *_ in rs if p <= 0]
    rwin = [r for _, r, *_ in rs if r > 0]
    rloss = [r for _, r, *_ in rs if r <= 0]
    tp = [(p, x) for p, _, _, _, x in rs if "tp" in (x or "").lower()]
    tp_neg = [(p, x) for p, x in tp if p <= 0]
    dollar_payoff = (sum(wins) / len(wins)) / abs(sum(losses) / len(losses)) if wins and losses else float("nan")
    roe_payoff = (sum(rwin) / len(rwin)) / abs(sum(rloss) / len(rloss)) if rwin and rloss else float("nan")
    print(label, "n=", len(rs),
          "net=", round(sum(p for p, *_ in rs), 2),
          "wr_pnl=", round(100 * len(wins) / len(rs), 1),
          "wr_roe=", round(100 * len(rwin) / len(rs), 1),
          "payoff_DOLLAR=", round(dollar_payoff, 3),
          "payoff_ROE=", round(roe_payoff, 3),
          "tp_eff=", round(100 * (len(tp) - len(tp_neg)) / len(tp), 1) if tp else "n/a")
EOF
```

Use what it prints. If it disagrees with the figures above, the database has moved and the database wins.

**Verified during planning — the payoff ratio has two legitimate and very different answers, and the README must say which it is quoting.** Over the full 181-trade ledger: the **dollar** payoff (average $ won ÷ average $ lost) is **0.94**, because winning trades were smaller-notional than losing ones; the **return-on-equity** payoff (`roe_pct` series, which is what the Monte Carlo and Kelly math now use) is **1.69**. Both are true. A "1.47 R:R payoff" cell in a strategy report reads as a per-trade edge advantage and is closer to neither unless labelled. Do not pick the flattering one; quote the dollar payoff for the trade-summary table and, if the ROE payoff appears anywhere, label it "payoff on return, not on dollars" in the same cell or footnote.

- [ ] **Step 2: Date the telemetry table and correct the two cells**

Add a snapshot line under the table heading — `**Snapshot as of 2026-10-12 over the first 149 closed swing trades.**` — and set the payoff cell and the take-profit-efficiency cell to the Step-1 values, each with the window it describes. If the current-ledger values differ from the 149-row window, show the current one and keep the 2026-10-12 one labelled.

- [ ] **Step 3: Remove the four unimplemented methodology claims**

Delete or rewrite so nothing claims an experiment that does not exist:
- `:64` "evaluated across historical walk-forward simulations" → state what is actually done: bootstrap resampling of recorded trades, and an iid caveat.
- `:66` the "Walk-Forward" row label in the live telemetry table → rename to "Live forward trading", which is what it is.
- `:81` "180-day market regimes" → drop the number; describe the regime detection that exists (`market_regime` classification with ADX/Hurst gates and turbulence thresholds).
- `:28-29` "Zero lookahead bias" → qualify: the live indicator path (`indicators.py`) is causal; the backtest corpus had intrabar and resample lookahead, fixed on this branch, and results generated before the fix were favourable by that margin.

- [ ] **Step 4: Replace the benchmark Sharpe table with audited numbers**

The Sharpe values 1.82 / 2.14 / 3.40 appear **only** in README.md — no code emits them. Replace that block with the audited swing-4H result and label every figure with the artefact it came from:
- win rate 46.2%, EV per trade −$0.230 with 90% CI [−0.373, −0.087], from `audits/boardroom_counterfactual_report_2026-10-13.md`
- Sharpe ≈ −0.34 for the BTC full-stack arm, with the arithmetic shown: the ablation harness reported −2.49 using an assumed 2,190 trades/year, against roughly 40 trades/year in a 2-year arm with 81 trades, so the correctly annualized figure is −2.49 × √(40/2190). Add that the harness annualization was corrected on this branch and that arm-level numbers should be re-run before being quoted again.
- The gates' contribution is `INCONCLUSIVE`: the ΔEV confidence intervals straddle zero.

Before writing any of those numbers, open the two named artefacts and confirm they still say what is claimed; if an artefact does not contain the figure, drop the claim rather than keeping it.

- [ ] **Step 5: Verify no fabricated metric survives, and commit**

Run: `grep -n "walk-forward\|Walk-Forward\|180-day\|1.47\|1.82\|2.14\|3.40\|100% on target" README.md`
Expected: no matches other than deliberately-labelled historical notes saying a figure was superseded.

```bash
git add README.md
git commit -m "docs(readme): date the telemetry snapshot, correct payoff and TP efficiency, drop unimplemented methodology claims"
```

---

## Follow-up work this plan deliberately excludes

Recorded so nobody thinks the audit is closed:

- Re-running the 2-year ablation and the 7-year backtests after Task 5 to regenerate every report with corrected annualization and no lookahead. Needs the CSV corpus and minutes of runtime.
- Provisioning the four missing `*_4h_full.csv` datasets so the swing arm is actually validated.
- Revoking the EC2 security-group inbound `0.0.0.0/0` rules on 8501/8502, and removing the leftover `dashboard-proxy.service` unit file (one `systemctl start` away from re-exposing the dashboard). Both are user actions.
- The Kelly sizer's prior table (`risk_manager.py:111-117` blends WR 0.6587 from `trades`/`berserker_trades`/`marl_live_trades` while the live strategy's own win rate is 0.4199) and its hardcoded `b=1.5`: real, separately scoped, and the highest-PnL item remaining.
- Block/cluster bootstrap in place of the iid bootstrap, multiple-hypothesis correction across the 72-config sweeps, and a purged walk-forward harness.
- The 26 uppercase/space-duplicated `bot_parameters` rows already in `trading_data.db` (`Berserker`, `Marl`, `Mean Reversion`, `Sniper`) created by one-off scripts in `scratch/`; Task 3 stops new ones but does not merge existing rows, which is a production data migration.

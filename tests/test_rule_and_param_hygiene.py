"""
tests/test_rule_and_param_hygiene.py — Task 3, 2026-10-22 audit remediation.

Covers four hygiene defects in the AI self-optimisation loop:

  1. `post_mortem_agent.enforce_parameter_bounds` ended in `return new_val`, so ANY knob name
     an LLM invented was written back completely unbounded.
  2. `backtests/batch_backtest_learner.py` claimed in its module docstring to "run a fast
     historical backtest trade extraction" while its scenarios were hand-written exemplars,
     and rules distilled from them landed in `learned_rules` indistinguishable from measured
     lessons.
  3. Both learner parameter writers took `bot_name` verbatim (a suggestion for `SNIPER`
     created a shadow row beside `sniper`) and wrote any `suggested_value` for any knob.
  4. `ai_engine._get_top_rules_sync` selected `ORDER BY confidence DESC LIMIT 15` with no
     tiebreak over a table whose rows are all >= 0.70, so the 15 rules injected into every
     boardroom prompt were an arbitrary tie subset and a 2025 exemplar could starve a lesson
     written today.

Fix round 1 of the review adds the signal-purity test (importing `post_mortem_agent` must not
take over the process signal table), the delegation tests (the trainer calls the learner's
parameter writer instead of re-implementing the upsert), and the refusal-reason tests (every
dropped adjustment is logged with the reason that actually applies).

Every database here is a fixture built under ``tmp_path`` with the working directory moved
there. The production ``trading_data.db``, ``trading_ledger.db`` and ``live_paper_trades.db``
are never opened, written, or read. ``scripts/deep_brain_trainer.py`` is imported for the
behavioural write-path tests with its ``DATA_DB``/``LEDGER_DB`` module constants repointed at
the fixture (its defaults name ``scripts/trading_data.db``), and asserted structurally with
``ast`` wherever the point is which helper the script calls rather than what a row ends up
containing.
"""

import ast
import asyncio
import importlib
import json
import logging
import pathlib
import signal
import sqlite3
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
LEARNER_SRC = REPO_ROOT / "backtests" / "batch_backtest_learner.py"
TRAINER_SRC = REPO_ROOT / "scripts" / "deep_brain_trainer.py"
AI_ENGINE_SRC = REPO_ROOT / "ai_engine.py"
POST_MORTEM_SRC = REPO_ROOT / "post_mortem_agent.py"

# The seven measured safe ranges, spelled out independently of PARAM_BOUNDS so that editing
# the table in the source cannot quietly move the target this test checks.
EXPECTED_BOUNDS = {
    "rsi_buy": (10.0, 40.0),
    "rsi_sell": (60.0, 90.0),
    "take_profit_pct": (1.001, 1.05),
    "stop_loss_pct": (0.95, 0.999),
    "min_nn_confidence": (0.60, 0.95),
    "vol_gate_mult": (0.8, 3.0),
    "veto_threshold": (40.0, 95.0),
}


def make_bot_parameters(db_path, rows=()):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE bot_parameters (id INTEGER PRIMARY KEY, bot_name TEXT,"
        " param_name TEXT, param_value REAL, last_updated TEXT,"
        " UNIQUE(bot_name, param_name))"
    )
    for bot, param, val in rows:
        conn.execute(
            "INSERT INTO bot_parameters (bot_name, param_name, param_value) VALUES (?, ?, ?)",
            (bot, param, val),
        )
    conn.commit()
    conn.close()


def make_learned_rules(db_path, with_source=False):
    conn = sqlite3.connect(db_path)
    cols = (
        "rule_id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT, rule_text TEXT,"
        " confidence REAL, created_at TEXT"
    )
    if with_source:
        cols += ", source TEXT DEFAULT 'measured'"
    conn.execute(f"CREATE TABLE learned_rules ({cols})")
    conn.commit()
    conn.close()


def make_trade_lessons(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE trade_lessons (id INTEGER PRIMARY KEY AUTOINCREMENT, trade_id INTEGER,"
        " bot_type TEXT, symbol TEXT, pnl REAL, original_rationale TEXT, lesson TEXT, timestamp TEXT)"
    )
    conn.commit()
    conn.close()


def refusal_lines(caplog_text):
    """Every `[PARAM REFUSED]` line the writers logged, so a test can pin the *reason* text."""
    return [line for line in caplog_text.splitlines() if "[PARAM REFUSED]" in line]


def live_top_rules_sql():
    """Return the exact SELECT string `ai_engine._get_top_rules_sync` executes."""
    tree = ast.parse(AI_ENGINE_SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_get_top_rules_sync":
            for call in ast.walk(node):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "execute"
                    and call.args
                    and isinstance(call.args[0], ast.Constant)
                    and isinstance(call.args[0].value, str)
                    and "learned_rules" in call.args[0].value
                ):
                    return call.args[0].value
    raise AssertionError("Could not locate the learned_rules SELECT in _get_top_rules_sync")


# ---------------------------------------------------------------------------
# Step 1 tests (assertions verbatim from the brief; only the import form and the
# source-path resolution follow the controller corrections).
# ---------------------------------------------------------------------------

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
    src = LEARNER_SRC.read_text(encoding="utf-8")
    assert "Runs a fast historical backtest trade extraction" not in src
    assert "synthetic exemplar" in src.lower()


def test_learner_writes_do_not_shadow_existing_bot_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "trading_data.db"
    make_bot_parameters(db, [("sniper", "trail_dist", 2.0)])

    from backtests.batch_backtest_learner import write_parameter_adjustments

    write_parameter_adjustments(str(db), [{"bot_name": "SNIPER", "param_name": "trail_dist", "suggested_value": 1.0}])

    conn = sqlite3.connect(db)
    names = [r[0] for r in conn.execute("SELECT bot_name FROM bot_parameters").fetchall()]
    conn.close()
    assert names == ["sniper"]


# ---------------------------------------------------------------------------
# Corrected Step 5 — PARAM_BOUNDS is the single source of truth, and the delta
# function keeps byte-identical behaviour for every bounded knob.
# ---------------------------------------------------------------------------

def test_bounds_table_matches_the_measured_bands():
    from post_mortem_agent import PARAM_BOUNDS

    assert PARAM_BOUNDS == EXPECTED_BOUNDS


@pytest.mark.parametrize("knob", sorted(EXPECTED_BOUNDS))
def test_every_band_edge_round_trips_through_the_delta_function(knob):
    from post_mortem_agent import enforce_parameter_bounds

    lo, hi = EXPECTED_BOUNDS[knob]
    # A zero delta at either edge leaves the value untouched.
    assert enforce_parameter_bounds(knob, lo, 0.0) == lo
    assert enforce_parameter_bounds(knob, hi, 0.0) == hi
    # A delta that would overshoot clamps to the edge it is heading for.
    assert enforce_parameter_bounds(knob, lo, -100000.0) == lo
    assert enforce_parameter_bounds(knob, hi, 100000.0) == hi
    # A delta landing mid-band is applied unchanged.
    mid = (lo + hi) / 2.0
    assert enforce_parameter_bounds(knob, lo, mid - lo) == pytest.approx(mid)


@pytest.mark.parametrize(
    "knob,current,delta,expected",
    [
        ("rsi_buy", 20.0, 0.0, 20.0),
        ("rsi_buy", 25.0, 20.0, 40.0),
        ("rsi_sell", 70.0, -15.0, 60.0),
        ("take_profit_pct", 1.02, 0.05, 1.05),
        ("stop_loss_pct", 0.97, -0.05, 0.95),
        ("min_nn_confidence", 0.70, 0.30, 0.95),
        ("vol_gate_mult", 1.2, 5.0, 3.0),
        ("veto_threshold", 50.0, 45.0, 95.0),
    ],
)
def test_delta_function_results_are_byte_identical_to_the_old_chain(knob, current, delta, expected):
    from post_mortem_agent import enforce_parameter_bounds

    assert enforce_parameter_bounds(knob, current, delta) == expected


def test_delta_function_still_returns_current_for_unparsable_inputs():
    from post_mortem_agent import enforce_parameter_bounds

    # `current_val` coerced, `suggested_adj` did not -> the coerced current value comes back.
    assert enforce_parameter_bounds("rsi_buy", "25", "junk") == 25.0
    # Neither coerces -> the raw object is returned unchanged, exactly as the old chain did.
    assert enforce_parameter_bounds("rsi_buy", None, 1.0) is None


def test_absolute_clamp_never_adds_to_the_current_value():
    """The writers hold an ABSOLUTE suggested_value; a clamp must not behave like a delta."""
    from post_mortem_agent import clamp_parameter_value

    assert clamp_parameter_value("vol_gate_mult", 99.0) == 3.0
    assert clamp_parameter_value("trail_dist", 1.0) is None
    assert clamp_parameter_value("veto_threshold", "junk") is None
    # An absolute 90 for veto_threshold is 90, not current+90.
    assert clamp_parameter_value("veto_threshold", 90.0) == 90.0
    assert clamp_parameter_value("stop_loss_pct", 0.9) == 0.95
    assert clamp_parameter_value("made_up_knob", 0.5) is None
    assert clamp_parameter_value("rsi_buy", None) is None


# ---------------------------------------------------------------------------
# Corrected Step 5 — the learner parameter writers.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("SNIPER", "sniper"),
        ("  sniper ", "sniper"),
        ("Mean Reversion", "mean_reversion"),
        ("MEAN_REVERSION", "mean_reversion"),
        ("Berserker", "berserker"),
        ("sniper_v2", "sniper_v2"),
    ],
)
def test_bot_name_normalisation_is_shared(raw, expected):
    from backtests.batch_backtest_learner import normalise_bot_name

    assert normalise_bot_name(raw) == expected


def test_normalise_bot_name_falls_back_to_sniper():
    from backtests.batch_backtest_learner import normalise_bot_name

    assert normalise_bot_name(None) == "sniper"
    assert normalise_bot_name("") == "sniper"
    assert normalise_bot_name("   ") == "sniper"


def test_learner_drops_a_knob_with_no_measured_band(tmp_path, monkeypatch):
    """trail_dist is deliberately unbounded, so a learner suggestion for it is refused."""
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "trading_data.db"
    make_bot_parameters(db, [("sniper", "trail_dist", 2.0)])

    from backtests.batch_backtest_learner import write_parameter_adjustments

    write_parameter_adjustments(str(db), [
        {"bot_name": "sniper", "param_name": "trail_dist", "suggested_value": 99.0},
        {"bot_name": "sniper", "param_name": "invented_knob", "suggested_value": 1.0},
    ])

    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT bot_name, param_name, param_value FROM bot_parameters").fetchall()
    conn.close()
    assert rows == [("sniper", "trail_dist", 2.0)]


def test_learner_clamps_an_out_of_band_absolute_suggestion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "trading_data.db"
    make_bot_parameters(db, [("berserker", "veto_threshold", 75.0)])

    from backtests.batch_backtest_learner import write_parameter_adjustments

    write_parameter_adjustments(str(db), [
        {"bot_name": "BERSERKER", "param_name": "veto_threshold", "suggested_value": 5000.0},
    ])

    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT bot_name, param_value FROM bot_parameters").fetchall()
    conn.close()
    assert rows == [("berserker", 95.0)]


def test_learner_merges_shadow_case_into_the_canonical_row(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "trading_data.db"
    make_bot_parameters(db, [("sniper", "min_nn_confidence", 0.70), ("SNIPER", "min_nn_confidence", 0.80)])

    from backtests.batch_backtest_learner import write_parameter_adjustments

    write_parameter_adjustments(str(db), [
        {"bot_name": "Sniper", "param_name": "min_nn_confidence", "suggested_value": 0.75},
    ])

    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT bot_name, param_value FROM bot_parameters").fetchall()
    conn.close()
    # Canonical `sniper` is the row that moves. The pre-existing legacy shadow row is left
    # untouched (this task rewrites no history) and no NEW shadow row appears.
    assert rows == [("sniper", 0.75), ("SNIPER", 0.80)]


def test_writer_does_not_invent_a_missing_schema(tmp_path, monkeypatch):
    """A write against a DB without bot_parameters must fail loudly, not silently create one."""
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "empty.db"
    sqlite3.connect(db).close()

    from backtests.batch_backtest_learner import write_parameter_adjustments

    with pytest.raises(sqlite3.OperationalError):
        write_parameter_adjustments(str(db), [
            {"bot_name": "sniper", "param_name": "vol_gate_mult", "suggested_value": 1.5},
        ])


# ---------------------------------------------------------------------------
# Step 5 — the `source` column, added defensively.
# ---------------------------------------------------------------------------

def test_source_column_helper_adds_the_column_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "trading_ledger.db"
    make_learned_rules(db, with_source=False)

    from backtests.batch_backtest_learner import ensure_learned_rules_source_column

    conn = sqlite3.connect(db)
    ensure_learned_rules_source_column(conn)
    conn.execute(
        "INSERT INTO learned_rules (category, rule_text, confidence, created_at)"
        " VALUES ('RISK', 'legacy writer row', 0.80, datetime('now'))"
    )
    cols = {r[1] for r in conn.execute("PRAGMA table_info(learned_rules)")}
    src = conn.execute("SELECT source FROM learned_rules").fetchall()
    conn.commit()
    conn.close()
    assert "source" in cols
    assert src == [("measured",)]


def test_source_column_helper_is_a_noop_when_the_column_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "trading_ledger.db"
    make_learned_rules(db, with_source=True)

    from backtests.batch_backtest_learner import ensure_learned_rules_source_column

    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO learned_rules (category, rule_text, confidence, created_at, source)"
        " VALUES ('RISK', 'already tagged', 0.80, datetime('now'), 'synthetic_exemplar')"
    )
    ensure_learned_rules_source_column(conn)  # must not raise, must not re-add
    ensure_learned_rules_source_column(conn)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(learned_rules)")]
    kept = conn.execute("SELECT source FROM learned_rules").fetchall()
    conn.commit()
    conn.close()
    assert cols.count("source") == 1
    assert kept == [("synthetic_exemplar",)]


def test_exemplar_rule_writer_tags_its_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "trading_ledger.db"
    make_learned_rules(db, with_source=False)

    from backtests.batch_backtest_learner import write_learned_rule

    conn = sqlite3.connect(db)
    write_learned_rule(conn, "VOLATILITY_SPIKES", "ATR > 2.5x and DXY move > 0.4%", 0.94)
    row = conn.execute(
        "SELECT category, rule_text, confidence, source FROM learned_rules"
    ).fetchone()
    conn.commit()
    conn.close()
    assert row == ("VOLATILITY_SPIKES", "ATR > 2.5x and DXY move > 0.4%", 0.94, "synthetic_exemplar")


def test_rule_writer_can_label_a_measured_lesson(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "trading_ledger.db"
    make_learned_rules(db, with_source=True)

    from backtests.batch_backtest_learner import write_learned_rule

    conn = sqlite3.connect(db)
    write_learned_rule(conn, "RISK", "measured lesson", 0.77, source="measured")
    write_learned_rule(conn, "RISK", "exemplar lesson", 0.77)
    rows = conn.execute("SELECT rule_text, source FROM learned_rules ORDER BY rule_id").fetchall()
    conn.commit()
    conn.close()
    assert rows == [("measured lesson", "measured"), ("exemplar lesson", "synthetic_exemplar")]


# ---------------------------------------------------------------------------
# Step 6 — the live reader must keep working on either schema shape, and the
# ordering must be deterministic and recency-aware.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("with_source", [False, True])
def test_live_reader_query_runs_on_both_schema_shapes(tmp_path, monkeypatch, with_source):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "trading_ledger.db"
    make_learned_rules(db, with_source=with_source)
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO learned_rules (category, rule_text, confidence, created_at)"
        " VALUES ('RISK', 'a rule', 0.85, datetime('now'))"
    )
    conn.commit()
    conn.close()

    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(live_top_rules_sql()).fetchall()
    finally:
        conn.close()
    assert rows == [("RISK", "a rule", 0.85)]


def test_live_reader_query_is_recency_ordered_and_never_filters_on_source():
    sql = live_top_rules_sql()
    assert "SELECT category, rule_text, confidence FROM learned_rules " in sql
    assert "WHERE confidence >= 0.70 " in sql
    assert "ORDER BY confidence DESC, created_at DESC, rule_id DESC LIMIT 15" in sql
    # `source` is written by the learners but must never constrain the advisory prompt.
    assert "source" not in sql


def test_newest_rule_displaces_a_stale_exemplar_among_confidence_ties(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "trading_ledger.db"
    make_learned_rules(db, with_source=True)

    conn = sqlite3.connect(db)
    # 20 stale exemplars at the same confidence, then one measured lesson written today.
    for i in range(20):
        conn.execute(
            "INSERT INTO learned_rules (category, rule_text, confidence, created_at, source)"
            " VALUES ('VOLATILITY_SPIKES', ?, 0.90, '2025-08-15 12:00:00', 'synthetic_exemplar')",
            (f"stale exemplar {i}",),
        )
    conn.execute(
        "INSERT INTO learned_rules (category, rule_text, confidence, created_at, source)"
        " VALUES ('RISK', 'lesson written today', 0.90, datetime('now'), 'measured')"
    )
    conn.commit()
    texts = [r[1] for r in conn.execute(live_top_rules_sql()).fetchall()]
    conn.close()

    assert len(texts) == 15
    assert "lesson written today" in texts
    assert "stale exemplar 19" in texts
    assert "stale exemplar 0" not in texts


def test_rule_selection_is_stable_across_repeated_reads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "trading_ledger.db"
    make_learned_rules(db)
    conn = sqlite3.connect(db)
    for i in range(40):
        conn.execute(
            "INSERT INTO learned_rules (category, rule_text, confidence, created_at)"
            " VALUES ('RISK', ?, 0.70, ?)",
            (f"rule {i}", f"2026-01-01 00:00:{i:02d}"),
        )
    conn.commit()
    sql = live_top_rules_sql()
    first = conn.execute(sql).fetchall()
    for _ in range(5):
        assert conn.execute(sql).fetchall() == first
    conn.close()
    assert [r[1] for r in first] == [f"rule {i}" for i in range(39, 24, -1)]


# ---------------------------------------------------------------------------
# Guards for the steps that touch scripts/ (not importable from a test).
# ---------------------------------------------------------------------------

def test_exemplar_clusters_are_renamed_but_still_seeded():
    """Step 4 renames the producer; the clusters stay, as the seed of the honesty test."""
    import backtests.batch_backtest_learner as learner

    clusters = learner.exemplar_scenarios()
    assert set(clusters) == {"CHOP_FAILURES", "VOLATILITY_SPIKES", "PREMATURE_EXITS", "ASYMMETRIC_WINNERS"}
    assert not hasattr(learner, "extract_historical_scenarios")
    assert "exemplar" in (learner.exemplar_scenarios.__doc__ or "").lower()
    # The false history claim must be gone from the whole file, docstrings included.
    assert "Simulates multi-asset historical trade executions" not in LEARNER_SRC.read_text(encoding="utf-8")


def test_deep_brain_trainer_delegates_the_parameter_upsert_to_the_learner():
    """Finding 2/4: one writer owns the upsert, so the two learners cannot drift again."""
    tree = ast.parse(TRAINER_SRC.read_text(encoding="utf-8"))
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)

    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append((node.module, {alias.name for alias in node.names}))

    assert ("backtests.batch_backtest_learner",
            {"write_learned_rule", "write_parameter_adjustments"}) in imported

    # Both shared writers are actually invoked, not merely imported.
    for helper in ("write_learned_rule", "write_parameter_adjustments"):
        assert helper in calls, f"{helper} is imported but never called"
    # Finding 4: `write_learned_rule` calls the source-column helper itself, so a second
    # call in front of it is dead code.
    assert "ensure_learned_rules_source_column" not in calls
    # Finding 2: normalising and clamping belong to the shared writer, not to this script.
    assert "normalise_bot_name" not in calls
    assert "clamp_parameter_value" not in calls

    # The trainer must not keep its own copy of the upsert or its own copy of the refusal text.
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "process_training_scenario"
    )
    body = ast.unparse(fn)
    assert "INSERT INTO bot_parameters" not in body
    assert "ON CONFLICT" not in body
    assert "[PARAM REFUSED]" not in TRAINER_SRC.read_text(encoding="utf-8")


def run_trainer_scenario(tmp_path, monkeypatch, caplog, adjustment, seed=(), target_bot="sniper"):
    """Drive deep_brain_trainer's real per-scenario write path with a canned LLM reply.

    Returns (bot_parameters rows, captured log text). Both DB paths are repointed at
    ``tmp_path`` fixtures because the script's own module constants name
    ``scripts/trading_data.db`` / ``scripts/trading_ledger.db``, and the script's ``DATA_DB``
    is handed straight to the writer.
    """
    trainer = importlib.import_module("scripts.deep_brain_trainer")
    monkeypatch.chdir(tmp_path)
    data_db = tmp_path / "trading_data.db"
    ledger_db = tmp_path / "trading_ledger.db"
    make_bot_parameters(data_db, seed)
    make_learned_rules(ledger_db, with_source=False)
    make_trade_lessons(ledger_db)

    # Guard the real script-local databases rather than trusting the monkeypatch alone.
    real_dbs = [
        path for path in (
            REPO_ROOT / "scripts" / "trading_data.db",
            REPO_ROOT / "scripts" / "trading_ledger.db",
        ) if path.exists()
    ]
    before = {path: (path.stat().st_mtime_ns, path.stat().st_size) for path in real_dbs}

    monkeypatch.setattr(trainer, "DATA_DB", str(data_db))
    monkeypatch.setattr(trainer, "LEDGER_DB", str(ledger_db))

    reply = {
        "master_lesson": "Veto breakout entries while ADX is under 20.",
        "defense_rule": "ADX < 20 -> veto",
        "confidence_score": 0.90,
        "parameter_adjustment": adjustment,
    }

    async def fake_query_cloud_llm(*args, **kwargs):
        return json.dumps(reply)

    monkeypatch.setattr(trainer, "query_cloud_llm", fake_query_cloud_llm)
    item = {
        "category": "CHOP_DEFENSE",
        "asset": "BTC/USDT",
        "title": "Low-Momentum Asian Session Range Compression",
        "scenario": "ADX=11.5, Hurst=0.34.",
        "failure_mode": "4 false breakout trades.",
        "target_bot": target_bot,
    }
    with caplog.at_level(logging.INFO):
        result = asyncio.run(trainer.process_training_scenario(1, item))
    # process_training_scenario swallows exceptions into a None return, so a fixture mistake
    # must not read as "nothing was refused".
    assert result is not None, f"scenario errored out; log was:\n{caplog.text}"

    for path, signature in before.items():
        current = (path.stat().st_mtime_ns, path.stat().st_size)
        assert current == signature, f"{path} was written by the trainer test"

    conn = sqlite3.connect(data_db)
    try:
        rows = conn.execute(
            "SELECT bot_name, param_name, param_value FROM bot_parameters ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    return rows, caplog.text


def test_trainer_writer_updates_the_canonical_row_instead_of_shadowing(tmp_path, monkeypatch, caplog):
    """Finding 2: the trainer's own write path, exercised for real, not ast-walked."""
    rows, _ = run_trainer_scenario(
        tmp_path, monkeypatch, caplog,
        {"bot_name": "SNIPER", "param_name": "veto_threshold", "suggested_value": 90.0,
         "rationale": "veto earlier"},
        seed=[("sniper", "veto_threshold", 75.0)],
    )
    assert rows == [("sniper", "veto_threshold", 90.0)]


def test_trainer_writer_clamps_an_out_of_band_absolute_suggestion(tmp_path, monkeypatch, caplog):
    rows, text = run_trainer_scenario(
        tmp_path, monkeypatch, caplog,
        {"bot_name": "BERSERKER", "param_name": "veto_threshold", "suggested_value": 5000.0,
         "rationale": "raise the gate"},
        seed=[("berserker", "veto_threshold", 75.0)],
        target_bot="berserker",
    )
    assert rows == [("berserker", "veto_threshold", 95.0)]
    assert "[PARAM OPTIMIZED]" in text


def test_trainer_writer_refuses_a_knob_with_no_measured_band(tmp_path, monkeypatch, caplog):
    rows, text = run_trainer_scenario(
        tmp_path, monkeypatch, caplog,
        {"bot_name": "sniper", "param_name": "trail_dist", "suggested_value": 99.0,
         "rationale": "tighten the trail"},
        seed=[("sniper", "trail_dist", 2.0)],
    )
    assert rows == [("sniper", "trail_dist", 2.0)]
    refusals = refusal_lines(text)
    assert len(refusals) == 1
    assert "no measured safe range" in refusals[0]


def test_trainer_writer_defaults_the_bot_to_the_scenario_target(tmp_path, monkeypatch, caplog):
    """A suggestion with no bot_name must not silently retarget `sniper`."""
    rows, _ = run_trainer_scenario(
        tmp_path, monkeypatch, caplog,
        {"param_name": "min_nn_confidence", "suggested_value": 0.80, "rationale": "sharpen entry"},
        seed=[("berserker", "min_nn_confidence", 0.65)],
        target_bot="berserker",
    )
    assert rows == [("berserker", "min_nn_confidence", 0.80)]


def test_deep_brain_trainer_no_longer_writes_raw_bot_names():
    tree = ast.parse(TRAINER_SRC.read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "process_training_scenario"
    )
    body = ast.unparse(fn)
    # Canonicalising and clamping now live in the shared writer; the script hands over the
    # suggestion plus the scenario's target bot as the fallback name, and nothing else.
    assert "write_parameter_adjustments" in body
    assert "bot_parameters" not in body
    # The raw accessor form that created the shadow rows must be gone.
    assert 'adj.get("bot_name"' not in body


def test_live_fleet_rule_writers_are_not_relabelled():
    """Only the two learner writers get the exemplar tag. The fleet's own post-mortem writers
    keep their untagged INSERT — relabelling the live lesson producers is a different decision
    than the one this task authorises."""
    tree = ast.parse(POST_MORTEM_SRC.read_text(encoding="utf-8"))
    inserts = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "INSERT INTO learned_rules" in node.value
    ]
    assert inserts, "post_mortem_agent should still write learned rules"
    for sql in inserts:
        assert "source" not in sql


def test_enforce_parameter_bounds_callers_were_not_rewired():
    """Step 3's caller-cleanup clause is void: both live callers already pass a delta and gate
    on `new_val != cur_val`. Prove the call sites still look exactly alike."""
    src = POST_MORTEM_SRC.read_text(encoding="utf-8")
    assert src.count("new_val = enforce_parameter_bounds(p_name, cur_val, p_adj)") == 2
    assert src.count("if new_val != cur_val:") == 2


# ---------------------------------------------------------------------------
# Fix round 1, Finding 1 — importing post_mortem_agent must not take over the
# process signal table.
# ---------------------------------------------------------------------------

def _signal_registrations(node):
    """The signal names registered by `signal.signal(signal.SIGX, ...)` under an AST node."""
    found = []
    for call in ast.walk(node):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "signal"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "signal"
            and call.args
            and isinstance(call.args[0], ast.Attribute)
        ):
            found.append(call.args[0].attr)
    return found


def _module_scope_signal_registrations(tree):
    found = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        found += _signal_registrations(stmt)
    return found


def test_importing_post_mortem_agent_leaves_the_process_signal_table_alone():
    """`import post_mortem_agent` must not replace the importer's SIGINT/SIGTERM handlers.

    `swing_trader.py` registers graceful-shutdown handlers (`:3471-3472`) and then lazily
    imports post_mortem_agent from the position-close path (`:3082`), so an import-time
    registration silently disarms Ctrl-C on the live trader after the first losing swing exit.
    `backtests/batch_backtest_learner.py` and `scripts/deep_brain_trainer.py` import it at
    module scope, which made both multi-minute human-run LLM loops uninterruptible.
    """
    def sentinel(signum, frame):
        return None

    watched = (signal.SIGINT, signal.SIGTERM)
    original = {sig: signal.getsignal(sig) for sig in watched}
    saved_module = sys.modules.pop("post_mortem_agent", None)
    try:
        # Install a distinctive handler FIRST. Without this the test is order-dependent: if an
        # earlier test already imported a handler-hijacking module, the recorded baseline would
        # BE that hijacked handler and the assertion would pass by accident.
        for sig in watched:
            signal.signal(sig, sentinel)
        importlib.import_module("post_mortem_agent")
        for sig in watched:
            assert signal.getsignal(sig) is sentinel, (
                f"importing post_mortem_agent overwrote the process {sig.name} handler"
            )
    finally:
        for sig, handler in original.items():
            signal.signal(sig, handler)
        if saved_module is not None:
            sys.modules["post_mortem_agent"] = saved_module


def test_post_mortem_agent_installs_its_shutdown_handlers_in_run_agent():
    """The registrations must MOVE into run_agent(), not be deleted along with the module scope."""
    tree = ast.parse(POST_MORTEM_SRC.read_text(encoding="utf-8"))
    assert _module_scope_signal_registrations(tree) == []
    run_agent = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_agent"
    )
    assert sorted(_signal_registrations(run_agent)) == ["SIGINT", "SIGTERM"]


# ---------------------------------------------------------------------------
# Fix round 1, Findings 3 and 5 — every refusal says why, and nothing is skipped
# in silence.
# ---------------------------------------------------------------------------

def test_refusal_log_names_the_real_cause(tmp_path, monkeypatch, caplog):
    """A bounded knob fed an unparsable value must not be reported as a knob with no band."""
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "trading_data.db"
    make_bot_parameters(db, [("sniper", "trail_dist", 2.0)])

    from backtests.batch_backtest_learner import write_parameter_adjustments

    with caplog.at_level(logging.INFO):
        write_parameter_adjustments(str(db), [
            {"bot_name": "sniper", "param_name": "vol_gate_mult", "suggested_value": "banana"},
            {"bot_name": "sniper", "param_name": "trail_dist", "suggested_value": 9.0},
        ])

    refusals = refusal_lines(caplog.text)
    assert len(refusals) == 2
    unparsable = next(line for line in refusals if "vol_gate_mult" in line)
    unbounded = next(line for line in refusals if "trail_dist" in line)
    assert "not a number" in unparsable
    assert "no measured safe range" not in unparsable
    assert "no measured safe range" in unbounded
    assert "not a number" not in unbounded

    # Fail-closed behaviour is untouched by the wording change.
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT bot_name, param_name, param_value FROM bot_parameters").fetchall()
    conn.close()
    assert rows == [("sniper", "trail_dist", 2.0)]


def test_malformed_adjustment_is_logged_not_silently_dropped(tmp_path, monkeypatch, caplog):
    """Finding 5: a skipped adjustment has to be visible; a silent skip is the defect class."""
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "trading_data.db"
    make_bot_parameters(db, [("sniper", "vol_gate_mult", 1.2)])

    from backtests.batch_backtest_learner import write_parameter_adjustments

    with caplog.at_level(logging.INFO):
        write_parameter_adjustments(str(db), [
            {"bot_name": "sniper", "suggested_value": 2.0},
            {"bot_name": "sniper", "param_name": "vol_gate_mult"},
        ])

    refusals = refusal_lines(caplog.text)
    assert len(refusals) == 2
    assert any("carries no param_name" in line for line in refusals)
    assert any("carries no suggested_value" in line for line in refusals)

    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT bot_name, param_name, param_value FROM bot_parameters").fetchall()
    conn.close()
    assert rows == [("sniper", "vol_gate_mult", 1.2)]


def test_trainer_writer_refuses_an_unparsable_suggestion_and_says_why(tmp_path, monkeypatch, caplog):
    """The same two distinguishable reasons through the trainer's own write path."""
    rows, text = run_trainer_scenario(
        tmp_path, monkeypatch, caplog,
        {"bot_name": "sniper", "param_name": "vol_gate_mult", "suggested_value": "high",
         "rationale": "loosen the vol gate"},
        seed=[("sniper", "vol_gate_mult", 1.2)],
    )
    assert rows == [("sniper", "vol_gate_mult", 1.2)]
    refusals = refusal_lines(text)
    assert len(refusals) == 1
    assert "not a number" in refusals[0]
    assert "no measured safe range" not in refusals[0]

"""tests/test_inflight_shepherd_verdict_absence.py — F6 of the 2026-10-22 audit remediation.

The branch's defect class is a label that claims more than the measurement supports. The in-flight
shepherd used to publish `"HOLDING_NORMAL"` with `confidence 0.50` when the AI call returned nothing
at all — an algorithmic baseline dressed up as an AI verdict — and every consumer painted that
absence green.

These pins hold the fix in place:

  * the producer publishes `config.VERDICT_ABSENT` (a non-null token, because the ledger's
    `verdict` column is `TEXT NOT NULL`) with `tape_condition` / `confidence` as `None`;
  * a decision payload with no verdict key yields the token, and a payload that carried one yields
    exactly that verdict;
  * the derived verdicts — measured off the position's own R/ROE and the climax/reversal detectors —
    still overwrite the token;
  * the ledger path accepts a token row (temp sqlite under `tmp_path`, never a production DB);
  * no control flow moved: every `verdict` comparison in the module answers identically for the
    token and for `"HOLDING_NORMAL"`;
  * the four consumers render the absence in grey words — "no AI verdict measured" — and never the
    raw token, "HOLDING_NORMAL", "BALANCED" or a literal `None`.

Import rules honoured here: `inflight_shepherd` is imported only through `_load_shepherd`, which
moves CWD into a tmp dir BEFORE the import (its module body opens `trading_ledger.db` by relative
path), and `dashboard` is never imported — its renderer is compiled out of the source file by the
AST harness in `tests/test_measurement_fidelity.py`.
"""

import ast
import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

import config
from tests.test_measurement_fidelity import _dashboard_helper

REPO_ROOT = Path(__file__).resolve().parent.parent
SHEPHERD_SRC = REPO_ROOT / "inflight_shepherd.py"

DERIVED_VERDICTS = {
    "HARVEST_CLIMAX_EXECUTED",
    "MOMENTUM_STALL",
    "EARLY_REVERSAL_EXIT",
    "FUNDING_CARRY_HARVEST",
}

ABSENT_WORDS = ("HOLDING_NORMAL", "MONITORED", "BALANCED", config.VERDICT_ABSENT)


def _load_shepherd(monkeypatch, tmp_path):
    """Import `inflight_shepherd` with a tmp CWD already in place, and return it.

    Its module body connects to `trading_ledger.db` by relative path, so importing it from the repo
    root would open a PRODUCTION database. Every test in this file goes through here.
    """
    monkeypatch.chdir(tmp_path)
    import inflight_shepherd
    return inflight_shepherd


def _shepherd_function(name="evaluate_active_positions"):
    tree = ast.parse(SHEPHERD_SRC.read_text(encoding="utf-8"), filename=str(SHEPHERD_SRC))
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"inflight_shepherd.py has no function named {name}")


# ── the producer ───────────────────────────────────────────────────────────────────────────────

def test_ai_unavailable_fallback_publishes_the_absence(monkeypatch, tmp_path):
    """No AI answer is an absence — never a `HOLDING_NORMAL` all-clear with an invented 0.50."""
    shepherd = _load_shepherd(monkeypatch, tmp_path)
    import ai_engine

    class _FakeClient:
        session = None

    async def _gemini_down(*_args, **_kwargs):
        return None

    async def _groq_down(*_args, **_kwargs):
        return None

    monkeypatch.setattr(ai_engine, "get_llm", lambda *_args, **_kwargs: _FakeClient())
    monkeypatch.setattr(ai_engine, "_ensure_active_session", lambda *_args, **_kwargs: "fake-session")
    monkeypatch.setattr(ai_engine, "_query_gemini_cloud", _gemini_down)
    monkeypatch.setattr(ai_engine, "_query_groq_cloud_with_rotation", _groq_down)

    res = asyncio.run(
        shepherd.InflightPositionShepherd()._query_ai_shepherd(
            {"symbol": "BTC/USDT", "side": "long"}, timeout_sec=0.1
        )
    )

    assert res["verdict"] == config.VERDICT_ABSENT
    assert res["tape_condition"] is None
    assert res["confidence"] is None
    assert "no verdict measured" in res["diagnosis"].lower()
    # The inert fields and the POLICY target are unchanged: no action on a verdict nobody measured.
    assert res["runner_mode"] is False
    assert res["tighten_sl"] is False
    assert res["suggested_sl"] is None
    assert res["target_r"] == 2.5
    blob = json.dumps(res)
    for fabricated in ("HOLDING_NORMAL", "BALANCED", "Standard algorithmic trailing baseline"):
        assert fabricated not in blob, fabricated


def test_decision_without_a_verdict_key_yields_the_token(monkeypatch, tmp_path):
    """A payload that never carried a verdict measured nothing, so its default is the token."""
    word = _load_shepherd(monkeypatch, tmp_path)._verdict_word
    for decision in ({}, {"confidence": 0.9}, {"verdict": None}, {"verdict": ""},
                     {"verdict": "   "}, None, "not-a-dict"):
        assert word(decision) == config.VERDICT_ABSENT, decision


def test_decision_with_a_verdict_yields_that_verdict(monkeypatch, tmp_path):
    """A measured verdict keeps its own word — the absence default must not swallow it."""
    word = _load_shepherd(monkeypatch, tmp_path)._verdict_word
    assert word({"verdict": "HOLDING_NORMAL"}) == "HOLDING_NORMAL"
    assert word({"verdict": "momentum_stall"}) == "MOMENTUM_STALL"
    assert word({"verdict": "RUNNER_EXPANSION"}) == "RUNNER_EXPANSION"
    assert word({"verdict": config.VERDICT_ABSENT}) == config.VERDICT_ABSENT


def test_unmeasured_tape_publishes_none_not_the_word_balanced(monkeypatch, tmp_path):
    """The tape is published as an absence when neither the AI payload nor the tape packet carried a
    word — `"BALANCED"` used to be a reading the producer invented and its consumers coloured."""
    tape_word = _load_shepherd(monkeypatch, tmp_path)._tape_word

    assert tape_word({}, {}) is None
    assert tape_word({"tape_condition": None}, {"tape_condition": None}) is None
    assert tape_word({"tape_condition": ""}, {}) is None
    assert tape_word({"tape_condition": "BULLISH_ABSORPTION"}, {}) == "BULLISH_ABSORPTION"
    assert tape_word({}, {"tape_condition": "WALL_RESISTANCE"}) == "WALL_RESISTANCE"
    assert tape_word({"tape_condition": "BEARISH_ABSORPTION"},
                     {"tape_condition": "WALL_RESISTANCE"}) == "BEARISH_ABSORPTION"


def test_absence_confidence_does_not_reach_a_float_call(monkeypatch, tmp_path):
    """The loop's confidence read must be None-aware: the fallback publishes None and
    `float(None)` would raise inside the live scan, so the published value stays None."""
    src = SHEPHERD_SRC.read_text(encoding="utf-8")
    loop = src[src.index("decision = await self._query_ai_shepherd"):]
    loop = loop[:loop.index("tape_cond =")]
    assert "float(decision.get(\"confidence\"" not in loop, "the trap is back: float() on a None"
    assert "if raw_conf is not None else None" in loop


# ── the derived (measured) verdicts still win ──────────────────────────────────────────────────

def test_derived_verdicts_still_overwrite_the_published_default():
    """The four derived verdicts come from the position's own R/ROE and the climax/reversal
    detectors. They are measurements, so they must keep overwriting the absence token — and the
    token must never be one of them."""
    fn = _shepherd_function()
    literals = []
    default_line = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "verdict" for t in node.targets):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                literals.append((node.value.value, node.lineno))
            elif (isinstance(node.value, ast.Call)
                  and getattr(node.value.func, "id", "") == "_verdict_word"):
                default_line = node.lineno

    assert default_line is not None, "the published verdict no longer comes from _verdict_word()"
    assert {v for v, _line in literals} == DERIVED_VERDICTS
    assert config.VERDICT_ABSENT not in {v for v, _line in literals}
    for verdict, line in literals:
        assert line > default_line, f"the {verdict} branch must assign after the default to win"


# ── control-flow proof ─────────────────────────────────────────────────────────────────────────

def test_every_verdict_comparison_answers_identically_for_the_token(monkeypatch, tmp_path):
    """Swapping the default for the token may not move one branch.

    Every `Compare` node in `evaluate_active_positions` that touches `verdict` is compiled out of
    the shipped source and evaluated twice — once with the absence token, once with the word it
    replaced — and both answers must match. This is the proof, not a restatement of it.
    """
    _load_shepherd(monkeypatch, tmp_path)  # keeps this file's import rule in one place
    fn = _shepherd_function()

    comparisons = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Compare):
            if any(isinstance(n, ast.Name) and n.id == "verdict" for n in ast.walk(node)):
                comparisons.append(node)

    assert len(comparisons) >= 4, f"expected the guard chain, found {len(comparisons)}"

    env = {"r_multiple": 1.5, "roe_pct": 9.0, "tighten_sl": True, "suggested_sl": 100.0}
    for node in comparisons:
        code = compile(ast.Expression(body=node), "<verdict-comparison>", "eval")
        answers = []
        for verdict in (config.VERDICT_ABSENT, "HOLDING_NORMAL"):
            scope = dict(env)
            scope["verdict"] = verdict
            answers.append(eval(code, {"__builtins__": {}}, scope))  # the repo's own AST node
        assert answers[0] == answers[1], f"control flow moved: {ast.unparse(node)}"

    # And the word the token replaced is never a comparison operand anywhere in the module.
    whole = ast.parse(SHEPHERD_SRC.read_text(encoding="utf-8"), filename=str(SHEPHERD_SRC))
    for node in ast.walk(whole):
        if isinstance(node, ast.Compare):
            strings = {c.value for c in ast.walk(node)
                       if isinstance(c, ast.Constant) and isinstance(c.value, str)}
            assert "HOLDING_NORMAL" not in strings, ast.unparse(node)


def test_the_absence_token_joins_no_ratchet_guard(monkeypatch, tmp_path):
    """The guards that suppress the ratchet list measured exit verdicts only; the token must not
    join them (that would be a behaviour change) and must not be treated as one of them."""
    _load_shepherd(monkeypatch, tmp_path)
    fn = _shepherd_function()
    guards = 0
    for node in ast.walk(fn):
        if isinstance(node, ast.Compare) and isinstance(node.comparators[0], (ast.Tuple, ast.List)):
            if any(isinstance(n, ast.Name) and n.id == "verdict" for n in ast.walk(node)):
                members = {c.value for c in ast.walk(node.comparators[0])
                           if isinstance(c, ast.Constant) and isinstance(c.value, str)}
                assert members and members <= DERIVED_VERDICTS, members
                guards += 1
    assert guards >= 3, f"the guard chain shrank: {guards}"


# ── the ledger column is NOT NULL, so the absence travels as a word ─────────────────────────────

def _report(verdict_value, tape_value, confidence_value) -> dict:
    return {
        "symbol": "BTC/USDT", "side": "long", "entry_price": 60000.0, "current_price": 61000.0,
        "unrealized_roe_pct": 1.66, "r_multiple": 0.83, "holding_mins": 42.0,
        "verdict": verdict_value, "runner_mode": False, "tape_condition": tape_value,
        "suggested_sl": None, "current_sl": 59000.0, "sl_ratcheted": False,
        "target_r": 2.5, "diagnosis": "AI shepherd unreachable: no verdict measured.",
        "confidence": confidence_value,
    }


def test_ledger_accepts_a_row_whose_verdict_is_the_token(monkeypatch, tmp_path):
    """The published row must survive `verdict TEXT NOT NULL`. The token does; a raw `None` raises,
    and the module's own `except` would swallow that — silently losing every ledger row for an
    unmeasured verdict. Written into a temp DB under `tmp_path`, never a production file."""
    shepherd = _load_shepherd(monkeypatch, tmp_path)
    shepherd._ensure_shepherd_tables()

    ledger = Path("trading_ledger.db").absolute()
    assert ledger.parent == tmp_path.resolve() or str(ledger).startswith(str(tmp_path)), ledger

    shepherd.InflightPositionShepherd()._persist_shepherd_report(
        _report(config.VERDICT_ABSENT, None, None))

    with sqlite3.connect(ledger) as conn:
        rows = conn.execute(
            "SELECT verdict, tape_condition, confidence FROM inflight_shepherd_logs").fetchall()
    assert len(rows) == 1, rows
    verdict, tape, confidence = rows[0]
    assert verdict == config.VERDICT_ABSENT
    assert tape is None, "an unmeasured tape publishes None — the column is nullable"
    assert confidence is None, "an unmeasured confidence publishes None — the column is nullable"

    # The NOT NULL proof on the same schema and the same column list: None is refused, the token is
    # not, which is exactly why the absence travels as a word.
    insert = ("INSERT INTO inflight_shepherd_logs (timestamp, symbol, side, entry_price, "
              "current_price, unrealized_roe_pct, r_multiple, holding_mins, verdict, runner_mode, "
              "tape_condition, suggested_sl, current_sl, sl_ratcheted, target_r, diagnosis, "
              "confidence) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")
    args = ("2026-10-22T12:00:00+00:00", "BTC/USDT", "long", 60000.0, 61000.0, 1.66, 0.83, 42.0,
            None, 0, None, None, 59000.0, 0, 2.5, "no verdict measured", None)
    with pytest.raises(sqlite3.IntegrityError):
        with sqlite3.connect(ledger) as conn:
            conn.execute(insert, args)
            conn.commit()

    with sqlite3.connect(ledger) as conn:
        conn.execute(insert, args[:8] + (config.VERDICT_ABSENT,) + args[9:])
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM inflight_shepherd_logs").fetchone()[0] == 2


def test_state_row_and_ledger_row_agree_on_the_token(monkeypatch, tmp_path):
    """The JSON payload keeps the token as its verdict value, so the system_state row and the
    ledger row say the same thing (and neither says `None`)."""
    shepherd = _load_shepherd(monkeypatch, tmp_path)
    shepherd._ensure_shepherd_tables()
    with sqlite3.connect("trading_data.db") as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS system_state "
                     "(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
        conn.commit()

    shepherd.InflightPositionShepherd()._persist_shepherd_report(
        _report(config.VERDICT_ABSENT, None, None))

    with sqlite3.connect("trading_data.db") as conn:
        raw = conn.execute("SELECT value FROM system_state WHERE key = 'ai_shepherd_BTCUSDT'").fetchone()
    assert raw, "the sentinel's own state row was not written"
    payload = json.loads(raw[0])
    assert payload["verdict"] == config.VERDICT_ABSENT
    assert payload["tape_condition"] is None
    assert payload["confidence"] is None


# ── the consumers render the absence ───────────────────────────────────────────────────────────

_shepherd_verdict_display = _dashboard_helper("_shepherd_verdict_display")


def test_dashboard_helper_renders_the_absence_in_grey_words():
    disp = _shepherd_verdict_display({"verdict": config.VERDICT_ABSENT, "diagnosis": None})
    assert disp["text"] == "no AI verdict measured"
    assert disp["color"] == "#94a3b8"
    assert disp["tape_text"] == "not measured"
    assert disp["absent"] is True
    blob = json.dumps(disp)
    for fabricated in ("HOLDING_NORMAL", "MONITORED", "BALANCED", "#00ff66", config.VERDICT_ABSENT):
        assert fabricated not in blob, fabricated


@pytest.mark.parametrize("payload", [{}, {"verdict": None}, {"verdict": ""}, {"verdict": "   "},
                                     None, "not-a-dict", 7])
def test_dashboard_helper_treats_every_flavour_of_missing_as_absent(payload):
    disp = _shepherd_verdict_display(payload)
    assert disp["text"] == "no AI verdict measured"
    assert disp["color"] == "#94a3b8"
    assert disp["icon"] == "", "no shield for a health check that never ran"
    assert disp["absent"] is True


def test_dashboard_helper_keeps_a_measured_verdict_exactly_as_it_was():
    normal = _shepherd_verdict_display({"verdict": "HOLDING_NORMAL", "tape_condition": "BALANCED",
                                        "diagnosis": "Healthy oscillation."})
    assert normal["text"] == "HOLDING_NORMAL"
    assert normal["color"] == "#00ff66"
    assert normal["icon"] == "🛡️"
    assert normal["tape_text"] == "BALANCED"
    assert normal["absent"] is False

    runner = _shepherd_verdict_display({"verdict": "RUNNER_EXPANSION", "runner_mode": True})
    assert (runner["color"], runner["icon"]) == ("#00e5ff", "🏃")

    stall = _shepherd_verdict_display({"verdict": "MOMENTUM_STALL"})
    assert (stall["color"], stall["icon"]) == ("#ff9900", "⚠️")

    adverse = _shepherd_verdict_display({"verdict": "ADVERSE_RISK"})
    assert adverse["color"] == "#ff003c"

    # A measured verdict with an unmeasured tape still says "not measured" about the tape.
    no_tape = _shepherd_verdict_display({"verdict": "ADVERSE_RISK", "tape_condition": None})
    assert no_tape["text"] == "ADVERSE_RISK"
    assert no_tape["tape_text"] == "not measured"


def _hourly_payload(shep_verdict) -> dict:
    return {
        "generated_at_utc": "2026-10-22T12:00:00+00:00",
        "cycles_count": 12, "start_cycle_num": 1, "end_cycle_num": 12,
        "equity_delta": 0.0, "equity_delta_pct": 0.0,
        "end_equity": 152.10, "end_margin": 12.0, "max_margin": 20.0, "min_margin": 4.0,
        "slots_occupied": 1, "total_accrued_funding": 0.0,
        "dominant_regime": "CHOP", "latest_stance": "Enforce capital preservation.",
        "latest_rationale": "Chop regime.", "latest_macro_narrative": "Flat macro.",
        "latest_watch_levels": "60000 / 61000",
        "directives": [], "closed_trades": [],
        "active_positions": [{
            "symbol": "BTC/USDT", "clean_symbol": "BTCUSDT", "side": "LONG", "qty": 0.01,
            "entry_p": 60000.0, "mark_p": 60100.0, "sl": 59500.0, "hard_sl": 59400.0,
            "tp": 62000.0, "lev": 5.0, "margin": 120.0, "unr_pnl": 1.0, "unr_roi": 0.83,
            "funding": 0.0, "true_net": 1.0,
            "shep_verdict": shep_verdict, "shep_diag": "", "shep_ratchet": False,
            "shep_target_r": 2.5,
        }],
    }


def test_hourly_briefing_renders_the_absence(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    import hourly_briefing

    label = hourly_briefing._shep_verdict_display(config.VERDICT_ABSENT)
    assert label == {"text": "no AI verdict measured", "color": "#94a3b8"}
    assert hourly_briefing._shep_verdict_display(None)["text"] == "no AI verdict measured"
    assert hourly_briefing._shep_verdict_display("HOLDING_NORMAL")["text"] == "HOLDING_NORMAL"

    absent_html = hourly_briefing.build_hourly_briefing_html(_hourly_payload(config.VERDICT_ABSENT))
    assert "no AI verdict measured" in absent_html
    for fabricated in ABSENT_WORDS:
        assert fabricated not in absent_html, fabricated

    measured_html = hourly_briefing.build_hourly_briefing_html(_hourly_payload("HOLDING_NORMAL"))
    assert "HOLDING_NORMAL" in measured_html
    assert "no AI verdict measured" not in measured_html


def test_executive_briefing_renders_the_absence(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    import executive_briefing

    absent = executive_briefing._shepherd_badge_and_tape(
        {"verdict": config.VERDICT_ABSENT, "tape_condition": None})
    assert "no AI verdict measured" in absent["badge"]
    assert absent["tape"] == "not measured"
    assert "🛡️" not in absent["badge"], "the shield implies a health check that did not happen"
    blob = absent["badge"] + absent["tape"]
    for fabricated in ABSENT_WORDS + ("None",):
        assert fabricated not in blob, fabricated

    missing = executive_briefing._shepherd_badge_and_tape({})
    assert "no AI verdict measured" in missing["badge"]
    assert missing["tape"] == "not measured"

    measured = executive_briefing._shepherd_badge_and_tape(
        {"verdict": "HOLDING_NORMAL", "tape_condition": "BALANCED"})
    assert measured["badge"] == "🛡️ **HOLDING_NORMAL**"
    assert measured["tape"] == "BALANCED"

    runner = executive_briefing._shepherd_badge_and_tape(
        {"verdict": "RUNNER_EXPANSION", "runner_mode": True, "tape_condition": "WALL_RESISTANCE"})
    assert runner["badge"] == "🏃 **RUNNER EXPANSION**"
    assert runner["tape"] == "WALL_RESISTANCE"


def test_executive_briefing_imports_the_shared_token_owner(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    import executive_briefing
    assert executive_briefing.config.VERDICT_ABSENT == config.VERDICT_ABSENT


# ── the shared token, and the F6a writer rule ──────────────────────────────────────────────────

def test_absence_token_is_a_shared_non_null_word():
    assert config.VERDICT_ABSENT == "NO_AI_VERDICT"
    assert isinstance(config.VERDICT_ABSENT, str)


def test_the_health_probe_no_longer_writes_the_sentinel_key():
    """F6a: the swing trader's AI-health probe must not clobber `ai_shepherd_{sym}` — the row the
    sentinel owns. A health-schema payload written there gave the sentinel card a health verdict and
    fabricated target/tape defaults."""
    swing = (REPO_ROOT / "swing_trader.py").read_text(encoding="utf-8")
    writer = swing[swing.index("def _save_ai_health"):]
    writer = writer[:writer.index("await asyncio.to_thread(_save_ai_health")]

    assert writer.count("INSERT OR REPLACE") == 1, "the health probe must write exactly one key"
    code = [ln for ln in writer.splitlines() if not ln.strip().startswith("#")]
    assert any("ai_health_" in ln for ln in code), "the health probe still publishes its own row"
    assert not any("ai_shepherd_" in ln for ln in code), "the health probe writes the sentinel's key"
    assert "ai_shepherd_" in writer, "the WHY comment must name the key it refuses to write"


def test_the_sentinel_is_the_only_writer_of_the_sentinel_key():
    """`ai_shepherd_{sym}` is the in-flight sentinel's own state row: only inflight_shepherd's
    persistence path may write it. Everyone else reads it."""
    writers = []
    for path in sorted(REPO_ROOT.glob("*.py")):
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines):
            if "ai_shepherd_" not in line or line.strip().startswith("#"):
                continue
            window = "\n".join(lines[max(0, i - 6):i + 1])
            if "INSERT" in window and "query_system_state" not in line:
                writers.append((path.name, i + 1, line.strip()))
    assert {name for name, _line, _text in writers} <= {"inflight_shepherd.py"}, writers
    assert any(name == "inflight_shepherd.py" for name, _l, _t in writers), \
        "the sentinel's own write went missing — the scan is broken"


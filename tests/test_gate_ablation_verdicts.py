"""
tests/test_gate_ablation_verdicts.py — Task 4 of the 2026-10-22 audit remediation.

Covers ``scripts/run_gate_ablation.py::verdict_for_gate``, the derived replacement for the four
verdict strings the report writer used to emit as literals ("KEEP (CRITICAL ALPHA DEFENSE)",
"KEEP (POSITIVE STATISTICAL FILTER)", "RETAIN AS STRUCTURAL TAIL-RISK BRAKES", "KEEP") no matter
what the harness's bootstrap found.

Arms are built here as plain dicts with the same keys ``run_ablation_simulation`` returns
(n_trades, win_rate, total_return, profit_factor, sharpe, ev_mean, ev_ci_90, gate_fires, trades;
``trades`` a list of dicts carrying ``net_pnl``), so nothing in this file runs the multi-minute
historical sweep, reads ``data/historical/*.csv``, or touches a ``.db``.
"""

from pathlib import Path

import pytest


def _arm(pnls, ci=None):
    return {"n_trades": len(pnls), "trades": [{"net_pnl": p} for p in pnls],
            "ev_mean": 0.0, "ev_ci_90": ci or (0.0, 0.0)}


# The brief's KEEP/RETHINK fixtures were [0.5]*40+[-0.4]*40 vs its mirror: a +10%/trade point
# estimate whose UNPAIRED 90% bootstrap interval is [-2.375%, +22.375%] and therefore straddles zero,
# so the brief's own CI rule returns INCONCLUSIVE for them — the snippet and its tests disagree. The
# rule is the deliverable, so the fixtures are widened to a genuinely significant pair (same shape,
# N=80/arm) rather than the rule being loosened to manufacture a KEEP. See
# test_a_large_point_estimate_that_is_not_significant_is_not_a_keep, which pins the original pair.
def test_gate_that_clearly_raises_ev_is_keep():
    from scripts.run_gate_ablation import verdict_for_gate
    full = _arm([0.6] * 40 + [-0.1] * 40)
    ablated = _arm([-0.6] * 40 + [0.1] * 40)
    verdict, evidence = verdict_for_gate(full, ablated)
    assert verdict.startswith("KEEP")
    assert "ΔEV" in evidence


def test_gate_that_clearly_lowers_ev_is_rethink():
    from scripts.run_gate_ablation import verdict_for_gate
    full = _arm([-0.6] * 40 + [0.1] * 40)
    ablated = _arm([0.6] * 40 + [-0.1] * 40)
    verdict, _ = verdict_for_gate(full, ablated)
    assert verdict.startswith("RETHINK")


def test_noisy_difference_is_inconclusive():
    from scripts.run_gate_ablation import verdict_for_gate
    full = _arm([0.1, -0.1] * 40)
    ablated = _arm([0.1, -0.1] * 40)
    verdict, evidence = verdict_for_gate(full, ablated)
    assert verdict == "INCONCLUSIVE"
    assert "straddles" in evidence.lower()


def test_underpowered_arms_are_inconclusive():
    from scripts.run_gate_ablation import verdict_for_gate
    verdict, evidence = verdict_for_gate(_arm([0.9, 0.9]), _arm([-0.9, -0.9]))
    assert verdict == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Deviation 1 of the task brief: a non-estimable ΔEV must surface its absence.
# The brief's snippet formats `{(lo or 0) * 100:+.3f}%` BEFORE the None check, so an
# underpowered pair prints "90% CI [+0.000%, +0.000%]" — a zero-width interval that reads
# like a precise measurement of "no effect". These tests pin the required behaviour instead.
# ---------------------------------------------------------------------------

def test_underpowered_arms_print_no_ci_and_no_delta():
    from scripts.run_gate_ablation import verdict_for_gate
    verdict, evidence = verdict_for_gate(_arm([0.9, 0.9]), _arm([-0.9, -0.9]))
    assert verdict == "INCONCLUSIVE"
    assert "CI [" not in evidence
    assert "+0.000%" not in evidence
    assert "ΔEV (gate ON minus gate OFF)" not in evidence
    assert "not estimable" in evidence


def test_underpowered_arms_name_actual_sizes_when_only_one_arm_is_short():
    from scripts.run_gate_ablation import verdict_for_gate
    verdict, evidence = verdict_for_gate(_arm([0.2] * 12 + [-0.2] * 12), _arm([0.7, -0.6, 0.3]))
    assert verdict == "INCONCLUSIVE"
    assert "CI [" not in evidence
    assert "not estimable" in evidence
    # Both real arm sizes are stated, and neither is presented as a measurement of zero.
    assert "24" in evidence and "3" in evidence


def test_missing_arm_says_it_did_not_run():
    from scripts.run_gate_ablation import verdict_for_gate
    verdict, evidence = verdict_for_gate(_arm([0.2] * 20), None)
    assert verdict == "INCONCLUSIVE"
    assert "did not run" in evidence
    assert "CI [" not in evidence
    verdict2, evidence2 = verdict_for_gate(None, _arm([0.2] * 20))
    assert verdict2 == "INCONCLUSIVE" and "did not run" in evidence2
    # The message names whichever side is absent instead of blaming the wrong arm.
    assert "gate-OFF ablation" in evidence and "gate-ON reference" not in evidence
    assert "gate-ON reference" in evidence2 and "gate-OFF ablation" not in evidence2
    _both_verdict, both_evidence = verdict_for_gate(None, None)
    assert "gate-ON reference and gate-OFF ablation" in both_evidence


def test_a_large_point_estimate_that_is_not_significant_is_not_a_keep():
    """Honesty guard: the verdict follows the interval, never the size of the point estimate."""
    from scripts.run_gate_ablation import verdict_for_gate
    verdict, evidence = verdict_for_gate(
        _arm([0.5] * 40 + [-0.4] * 40), _arm([-0.5] * 40 + [0.4] * 40))
    # ΔEV point estimate is +10% per trade, but the unpaired 90% bootstrap interval straddles
    # zero, so the derived verdict cannot be KEEP.
    assert verdict == "INCONCLUSIVE"
    assert "straddles" in evidence.lower()


def test_verdict_and_evidence_are_deterministic():
    from scripts.run_gate_ablation import verdict_for_gate
    full = _arm([0.6] * 60 + [-0.1] * 60)
    ablated = _arm([-0.6] * 60 + [0.1] * 60)
    first = verdict_for_gate(full, ablated)
    second = verdict_for_gate(full, ablated)
    assert first == second


def test_estimable_pair_reports_a_real_interval_and_delta():
    from scripts.run_gate_ablation import verdict_for_gate
    verdict, evidence = verdict_for_gate(
        _arm([0.6] * 60 + [-0.1] * 60), _arm([-0.6] * 60 + [0.1] * 60))
    assert verdict == "KEEP"
    assert "ΔEV (gate ON minus gate OFF)" in evidence
    assert "CI [" in evidence
    assert "N=120/120" in evidence


# ---------------------------------------------------------------------------
# I-1: the (heading, flag, arm) triple in GATE_SECTIONS is the only thing connecting a printed
# verdict to the measurement behind it. Swap two arm names and every section still prints a
# fully-derived, plausible, about-the-wrong-gate verdict, so the pairing is pinned here against
# EXPERIMENTS — the executable source of truth for which arm ablates which flag.
# ---------------------------------------------------------------------------

def test_every_gate_section_is_wired_to_the_arm_that_ablates_only_that_flag():
    from scripts.run_gate_ablation import EXPERIMENTS, FULL_STACK_NAME, GATE_SECTIONS

    arms = dict(EXPERIMENTS)
    gate_flags = sorted(arms[FULL_STACK_NAME])
    assert len(gate_flags) == 5, "the full stack is expected to define exactly five gate flags"
    assert len(GATE_SECTIONS) == len(gate_flags), (
        "every gate flag needs its own section, or a gate goes unreported")

    covered = set()
    for heading, param_name, ablated_name in GATE_SECTIONS:
        assert param_name in gate_flags, f"{heading} names an unknown flag {param_name!r}"
        assert param_name in heading, f"{heading} does not name its own flag {param_name!r}"
        assert ablated_name in arms, f"{heading} reads arm {ablated_name!r}, which EXPERIMENTS does not run"

        cfg = arms[ablated_name]
        assert cfg[param_name] is False, (
            f"{heading} compares against {ablated_name!r}, which leaves {param_name} ON")
        assert all(cfg[other] is True for other in gate_flags if other != param_name), (
            f"{heading} compares against {ablated_name!r}, which ablates more than {param_name}")
        covered.add(param_name)

    assert covered == set(gate_flags), f"sections cover {sorted(covered)}, expected all gate flags"
    assert len({name for _, _, name in GATE_SECTIONS}) == len(GATE_SECTIONS), (
        "two sections share one ablation arm")


# ---------------------------------------------------------------------------
# M-5: a constant PnL arm makes every bootstrap resample identical, so lo == hi and the "90% CI"
# is a zero-width interval printed from a variance-free sample — maximally confident and
# information-free. That must not be able to produce a KEEP.
# ---------------------------------------------------------------------------

def test_degenerate_zero_width_interval_is_inconclusive():
    from scripts.run_gate_ablation import verdict_for_gate
    verdict, evidence = verdict_for_gate(_arm([0.5] * 20), _arm([-0.5] * 20))
    assert verdict == "INCONCLUSIVE"
    assert "degenerate" in evidence
    assert "CI [" not in evidence
    assert "+0.000%" not in evidence


def test_degenerate_interval_does_not_leak_into_a_variable_pair():
    """The degenerate branch must not suppress real intervals."""
    from scripts.run_gate_ablation import verdict_for_gate
    verdict, evidence = verdict_for_gate(_arm([0.6] * 40 + [-0.1] * 40), _arm([-0.6] * 40 + [0.1] * 40))
    assert verdict == "KEEP"
    assert "CI [" in evidence
    assert "degenerate" not in evidence


# ---------------------------------------------------------------------------
# I-2: the report writer is where Step 3's heading, footer and missing-arm requirements live, so
# the writer itself must be under test — grepping the source for three deleted literals cannot
# catch a regression in the section loop or the footer. These tests call the writer with synthetic
# arms inside a temp cwd, so nothing can land in the repo's audits/ directory.
# ---------------------------------------------------------------------------

FULL_PNLS = [0.6] * 40 + [-0.1] * 40


def _row(name, pnls, fires):
    row = _arm(pnls)
    row.update({
        "name": name,
        "win_rate": 50.0,
        "total_return": round(sum(pnls) * 100.0, 2),
        "profit_factor": 1.2,
        "sharpe": 1.0,
        "gate_fires": fires,
    })
    return row


def _synthetic_rows():
    """One asset's arm rows, shaped exactly like evaluate_all() builds them.

    Covers all five verdict paths and all three veto-count paths at once: Gate A a significant
    positive ΔEV (KEEP), Gate C's arm absent (did not run), Gate B a three-trade arm whose gate
    never fires (not estimable + zero vetoes), Gate D an identical arm that never fires (straddles
    + zero vetoes), Gate E a significant negative ΔEV whose flag has no recorded fire count
    (RETHINK + not recorded).
    """
    from scripts.run_gate_ablation import FULL_STACK_NAME
    fires = {"gate_1d_trend": 12, "gate_divergence": 5, "gate_wick_exhaustion": 0, "gate_td9": 0}
    return [
        _row(FULL_STACK_NAME, FULL_PNLS, fires),
        _row("Ablate Gate A (1D Trend Support OFF)", [-0.6] * 40 + [0.1] * 40, {}),
        _row("Ablate Gate B (Wick Exhaustion OFF)", [0.6, -0.1, 0.3], {}),
        _row("Ablate Gate D (TD-9 Exhaustion OFF)", list(FULL_PNLS), {}),
        _row("Ablate Gate E (Volume Profile OFF)", [1.2] * 40 + [0.5] * 40, {}),
    ]


def _render_report(monkeypatch, tmp_path):
    import scripts.run_gate_ablation as rga
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rga, "SYMBOLS", ["TESTUSDT"])
    path = rga.write_report({"TESTUSDT": _synthetic_rows()})
    text = Path(path).read_text(encoding="utf-8")
    return rga, Path(path), text


def _sections(text):
    blocks = {}
    for chunk in text.split("### ")[1:]:
        title, _, body = chunk.partition("\n")
        blocks[title.strip()] = body
    return blocks


def test_report_writer_renders_derived_verdicts_per_section(monkeypatch, tmp_path):
    rga, path, text = _render_report(monkeypatch, tmp_path)
    sections = _sections(text)

    assert [heading for heading, _, _ in rga.GATE_SECTIONS] == [
        heading for heading in sections if heading[:1].isdigit()]
    assert path.parent.name == "audits"
    assert str(tmp_path) in str(path.resolve()), "the writer must not be able to touch the repo audits/"
    assert "## Asset: TESTUSDT" in text
    assert "| Gate Configuration | Trades (N) |" in text
    for literal in ("CRITICAL ALPHA DEFENSE", "POSITIVE STATISTICAL FILTER",
                    "RETAIN AS STRUCTURAL TAIL-RISK BRAKES"):
        assert literal not in text

    gate_a = sections["1. Gate A — 1D EMA Trend Alignment (`gate_1d_trend`)"]
    assert "**TESTUSDT — Verdict:** **KEEP**" in gate_a
    assert "ΔEV (gate ON minus gate OFF)" in gate_a and "CI [" in gate_a
    assert "12 candidate entries rejected by `gate_1d_trend`" in gate_a

    gate_e = sections["5. Gate E — Volume Profile & Value Area Clearance (`gate_volume_profile`)"]
    assert "**TESTUSDT — Verdict:** **RETHINK**" in gate_e
    assert "not recorded for `gate_volume_profile`" in gate_e


def test_report_writer_reports_no_interval_for_missing_and_underpowered_sections(monkeypatch, tmp_path):
    _, _, text = _render_report(monkeypatch, tmp_path)
    sections = _sections(text)

    missing = sections["2. Gate C — Momentum Divergence (`gate_divergence`)"]
    assert "**TESTUSDT — Verdict:** **INCONCLUSIVE**" in missing
    assert "did not run" in missing
    assert "CI [" not in missing

    underpowered = sections["3. Gate B — Wick Exhaustion (`gate_wick_exhaustion`)"]
    assert "**TESTUSDT — Verdict:** **INCONCLUSIVE**" in underpowered
    assert "not estimable" in underpowered and "closed 3" in underpowered
    assert "CI [" not in underpowered

    straddling = sections["4. Gate D — TD-9 Count Exhaustion (`gate_td9`)"]
    assert "CI [" in straddling and "straddles" in straddling.lower()

    iid_footer = ("- Confidence intervals are an iid bootstrap over closed trades and do not "
                  "correct for serial dependence between consecutive entries.")
    assert iid_footer in text


# ---------------------------------------------------------------------------
# M-3: the zero-veto line used to point at "the ΔEV above" even when the line above it said no
# ΔEV was reported. It may only reference a figure the section actually printed.
# ---------------------------------------------------------------------------

def test_zero_veto_count_does_not_point_at_an_unreported_delta(monkeypatch, tmp_path):
    _, _, text = _render_report(monkeypatch, tmp_path)
    block = _sections(text)["3. Gate B — Wick Exhaustion (`gate_wick_exhaustion`)"]
    assert "not estimable" in block
    assert "the ΔEV above" not in block
    assert "no ΔEV interval was reported above" in block


def test_zero_veto_count_still_describes_a_reported_delta_as_structural(monkeypatch, tmp_path):
    _, _, text = _render_report(monkeypatch, tmp_path)
    block = _sections(text)["4. Gate D — TD-9 Count Exhaustion (`gate_td9`)"]
    assert "CI [" in block
    assert "0 candidate entries were rejected by `gate_td9`" in block
    assert "the ΔEV above is structural" in block


# ---------------------------------------------------------------------------
# M-4: the methodology paragraph used to state "1500-resample iid bootstrap (seed 42)" as prose.
# Those figures live in module constants, so the label must be generated from them; if it is not,
# changing a constant silently makes the report describe a run that did not happen.
# ---------------------------------------------------------------------------

def test_methodology_paragraph_quotes_the_bootstrap_constants(monkeypatch, tmp_path):
    rga, _, text = _render_report(monkeypatch, tmp_path)
    assert f"{rga.BOOTSTRAP_RESAMPLES}-resample iid bootstrap (seed {rga.BOOTSTRAP_SEED})" in text


def test_methodology_paragraph_follows_the_constants_when_they_move(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.run_gate_ablation.BOOTSTRAP_RESAMPLES", 7, raising=False)
    monkeypatch.setattr("scripts.run_gate_ablation.BOOTSTRAP_SEED", 99, raising=False)
    _, _, text = _render_report(monkeypatch, tmp_path)
    assert "7-resample iid bootstrap (seed 99)" in text
    assert "1500" not in text


def test_bootstrap_resample_count_feeds_the_computation_not_only_the_label(monkeypatch):
    from scripts.run_gate_ablation import verdict_for_gate
    full, ablated = _arm(FULL_PNLS), _arm([-0.6] * 40 + [0.1] * 40)
    assert verdict_for_gate(full, ablated)[0] == "KEEP"
    monkeypatch.setattr("scripts.run_gate_ablation.BOOTSTRAP_RESAMPLES", 1, raising=False)
    assert verdict_for_gate(full, ablated)[0] == "INCONCLUSIVE"


@pytest.mark.parametrize("hardcoded", [
    "KEEP (CRITICAL ALPHA DEFENSE)",
    "KEEP (POSITIVE STATISTICAL FILTER)",
    "RETAIN AS STRUCTURAL TAIL-RISK BRAKES",
])
def test_report_writer_no_longer_emits_hardcoded_verdicts(hardcoded):
    src = (Path(__file__).resolve().parent.parent / "scripts" / "run_gate_ablation.py").read_text(
        encoding="utf-8")
    assert hardcoded not in src

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


@pytest.mark.parametrize("hardcoded", [
    "KEEP (CRITICAL ALPHA DEFENSE)",
    "KEEP (POSITIVE STATISTICAL FILTER)",
    "RETAIN AS STRUCTURAL TAIL-RISK BRAKES",
])
def test_report_writer_no_longer_emits_hardcoded_verdicts(hardcoded):
    src = (Path(__file__).resolve().parent.parent / "scripts" / "run_gate_ablation.py").read_text(
        encoding="utf-8")
    assert hardcoded not in src

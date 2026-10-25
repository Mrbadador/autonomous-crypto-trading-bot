"""
test_quant_validation_sharpe_absence.py — the rolling Sharpe publishes an absence, not a floor artefact.

The defect this pins: `quant_validation.py` divided the rolling validation window's mean by
`max(std, 0.10)` and by `max(downside_rms, 0.10)` — or by an invented `0.15` when the window held
no losing trade at all. Below those floors the published Sharpe was a number the FLOOR produced,
and `compute_finrl_rolling_sharpe_selector` turned it into an `allocation_weight_mult` that the
live engine sizes real capital with (`super_bot.py:556`, `swing_trader.py:1572`).

The floor is now a REFUSAL threshold: below it the window publishes an absence (None), the
selector sizes an unmeasured arm at the most conservative multiplier and names the absence in
its posture, and the report prints the absence in words instead of raising a TypeError.

Nothing here touches a live database: the end-to-end test builds its own CSV corpus under
tmp_path and `monkeypatch.chdir(tmp_path)`.
"""

import re

import numpy as np

import quant_validation
from quant_validation import (
    MIN_ARM_BARS,
    ROLLING_SHARPE_MIN_DISPERSION,
    _rolling_window_sharpes,
    compute_finrl_rolling_sharpe_selector,
    format_rolling_sharpe,
)


# A window of ten returns whose dispersion clears the threshold, chosen so the old code's
# arithmetic can be checked BY HAND rather than by calling the old code:
#   values            -> five at -1.0 and five at +3.0
#   mean              -> (-5 + 15) / 10 = 1.0
#   population std    -> sqrt((5*(-2)^2 + 5*(2)^2) / 10) = sqrt(4) = 2.0
#   old rec_std       -> max(2.0, 0.10) = 2.0
#   old Sharpe        -> round(sqrt(10) * (1.0 / 2.0), 2) = round(1.5811388300841898, 2) = 1.58
#   downside slice    -> the five -1.0 values, rms = sqrt(mean([1,1,1,1,1])) = 1.0
#   old downside_std  -> max(1.0, 0.10) = 1.0
#   old Sortino       -> round(sqrt(10) * (1.0 / 1.0), 2) = round(3.1622776601683795, 2) = 3.16
MEASURED_WINDOW = [-1.0, -1.0, -1.0, -1.0, -1.0, 3.0, 3.0, 3.0, 3.0, 3.0]
EXPECTED_SHARPE = 1.58
EXPECTED_SORTINO = 3.16

# A window whose dispersion is below the threshold: every return identical, so the standard
# deviation is exactly 0.0 and there is no denominator to divide by.
FLAT_WINDOW = [0.5] * 10


def test_a_window_below_the_dispersion_threshold_publishes_no_sharpe_and_no_sortino():
    window = _rolling_window_sharpes(np.array(FLAT_WINDOW, dtype=float))
    assert window["rolling_sharpe"] is None, (
        "a window whose dispersion is at or below the floor has no denominator; publishing a "
        "number here publishes the floor's artefact as a measurement")
    assert window["rolling_sortino"] is None, (
        "a flat window holds no losing trade, so the downside deviation has no denominator either")
    assert window["rolling_validation_window"] == len(FLAT_WINDOW)


def test_a_window_without_a_losing_trade_publishes_no_sortino_but_keeps_a_measured_sharpe():
    # Every return is positive but they are NOT identical, so the dispersion clears the floor:
    # mean 1.0, deviations +/-0.5 -> std 0.5.
    window = _rolling_window_sharpes(np.array([0.5, 1.5] * 5, dtype=float))
    assert window["rolling_sharpe"] is not None
    assert window["rolling_sharpe"] == round(float(np.sqrt(10) * (1.0 / 0.5)), 2)
    assert window["rolling_sortino"] is None, (
        "the Sortino needs a losing trade to measure downside dispersion; without one the old "
        "code divided by an invented 0.15")


def test_a_measured_window_still_publishes_the_number_the_old_code_published():
    window = _rolling_window_sharpes(np.array(MEASURED_WINDOW, dtype=float))
    assert window["rolling_sharpe"] == EXPECTED_SHARPE, (
        "the arithmetic on the measured path must not move: sqrt(N) * (mean / std), rounded to 2")
    assert window["rolling_sortino"] == EXPECTED_SORTINO, (
        "the arithmetic on the measured path must not move: sqrt(N) * (mean / downside_rms)")


def test_the_dispersion_threshold_is_a_refusal_and_not_a_divisor():
    # Exactly at the threshold the window refuses; the old code would have divided by the floor.
    at_floor = np.array([0.0, ROLLING_SHARPE_MIN_DISPERSION] * 5, dtype=float)
    window = _rolling_window_sharpes(at_floor)
    assert window["rolling_sharpe"] is None
    assert window["rolling_sortino"] is None


def test_the_selector_sizes_an_unmeasured_arm_conservatively_and_names_the_absence():
    bootstrap_res = {
        "UNMEASURED_ARM": {
            "strategy_name": "UNMEASURED_ARM",
            "rolling_sharpe": None,
            "rolling_sortino": None,
            "rolling_recent_roe_pct": 0.4,
            "z_score": 3.0,
            "significant_edge": True,
        },
    }
    result = compute_finrl_rolling_sharpe_selector(bootstrap_res)
    record = result["strategy_weights"]["UNMEASURED_ARM"]
    assert record["allocation_weight_mult"] == 0.75, (
        "an arm whose Sharpe was not measured takes the most conservative multiplier")
    assert isinstance(record["allocation_weight_mult"], float), (
        "the live engine reads this value as a float")
    assert "UNMEASURED" in record["posture"], (
        "the posture must name the absence instead of reading like a measured throttle")
    assert record["rolling_sharpe"] is None
    assert record["rolling_sortino"] is None
    assert result["winner_sharpe"] is None, (
        "the winner's Sharpe was not measured, so the report must not publish a number for it")


def test_a_measured_arm_outranks_an_unmeasured_arm_even_when_the_unmeasured_z_is_higher():
    bootstrap_res = {
        "MEASURED_ARM": {
            "strategy_name": "MEASURED_ARM",
            "rolling_sharpe": EXPECTED_SHARPE,
            "rolling_sortino": EXPECTED_SORTINO,
            "rolling_recent_roe_pct": 1.0,
            "z_score": 0.1,
            "significant_edge": True,
        },
        "UNMEASURED_ARM": {
            "strategy_name": "UNMEASURED_ARM",
            "rolling_sharpe": None,
            "rolling_sortino": None,
            "rolling_recent_roe_pct": 9.9,
            "z_score": 99.0,
            "significant_edge": True,
        },
    }
    result = compute_finrl_rolling_sharpe_selector(bootstrap_res)
    weights = result["strategy_weights"]
    assert weights["MEASURED_ARM"]["rank"] == 1, (
        "a measured arm must rank above an unmeasured one, whatever the unmeasured z-score is")
    assert weights["UNMEASURED_ARM"]["rank"] == 2
    assert result["ensemble_winner"] == "MEASURED_ARM"
    assert result["winner_sharpe"] == EXPECTED_SHARPE
    assert weights["UNMEASURED_ARM"]["allocation_weight_mult"] == 0.75
    assert "UNMEASURED" in weights["UNMEASURED_ARM"]["posture"]


def test_the_selector_does_not_raise_on_a_present_but_null_sharpe_key():
    # The `.get(key, default)` trap: a key that is PRESENT with the value None returns None,
    # NOT the default. `float(None)` would raise TypeError.
    bootstrap_res = {
        "ARM": {
            "strategy_name": "ARM",
            "rolling_sharpe": None,
            "rolling_sortino": None,
            "rolling_recent_roe_pct": None,
            "z_score": None,
            "significant_edge": False,
        },
    }
    result = compute_finrl_rolling_sharpe_selector(bootstrap_res)
    record = result["strategy_weights"]["ARM"]
    assert record["rolling_sharpe"] is None
    assert record["rolling_sortino"] is None
    assert record["rolling_recent_roe_pct"] is None
    assert result["winner_sharpe"] is None


def test_an_empty_bootstrap_result_names_no_winner():
    result = compute_finrl_rolling_sharpe_selector({})
    assert result["ensemble_winner"] is None, (
        "no arm measured nothing, so no arm won; naming SWING_4H would be a label the report "
        "never earned")
    assert result["winner_sharpe"] is None
    assert result["strategy_weights"] == {}


def _cli_report_block() -> str:
    # The CLI block cannot be CALLED from a test: main() runs
    # generate_full_validation_report(force_refresh=True), which rewrites the cache the live
    # engines size capital from. So the render path is pinned as source, the way
    # test_measurement_fidelity.py pins the rest of this module's printing.
    source = quant_validation.__file__
    with open(source, encoding="utf-8") as handle:
        return handle.read().split("def main()")[-1]


def test_the_cli_report_renders_both_sharpes_through_the_absence_aware_renderer():
    block = _cli_report_block()
    assert block.count("format_rolling_sharpe(") >= 2, (
        "both the per-arm rolling Sharpe and the ensemble winner's Sharpe must go through the "
        "renderer; `f\"{None:+.2f}\"` raises TypeError and takes the whole report down")
    assert "rolling_sharpe']:+.2f" not in block, (
        "a Sharpe formatted straight off the dict key cannot render an absence")
    assert "winner_sharpe']:+.2f" not in block, (
        "a Sharpe formatted straight off the dict key cannot render an absence")


def test_the_absence_is_rendered_in_words_not_as_a_zero():
    rendered = format_rolling_sharpe(None)
    assert "n/a" in rendered
    assert "0.00" not in rendered
    assert "+0.00" not in rendered
    assert format_rolling_sharpe(EXPECTED_SHARPE) == "+1.58"


def test_the_arm_minimum_is_one_constant_used_by_the_guard_and_the_reason():
    assert MIN_ARM_BARS == 120
    source = quant_validation.__file__
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    assert "if len(df) < MIN_ARM_BARS:" in text, (
        "the guard must read the named constant, not a literal that can drift from the reason")
    assert "below the 120-bar minimum this arm needs" not in text, (
        "the reason string must interpolate the constant, not restate the number")


def test_the_selector_docstring_does_not_restate_the_multipliers_it_does_not_own():
    # A docstring that prints tier multipliers is a second source of truth for numbers that live
    # in the branches below it. It drifted (claimed "1.00x Standard" where the code gives 1.05 and
    # 0.90) and named no posture for an arm whose Sharpe was never measured.
    doc = compute_finrl_rolling_sharpe_selector.__doc__ or ""
    assert not re.search(r"\b\d\.\d{2}x", doc), (
        "the multipliers belong in the branches, not restated in prose")


# ---------------------------------------------------------------------------
# End to end: the results dict itself carries the absence
# ---------------------------------------------------------------------------
# The pure helper proves the window refuses to divide. This proves the refusal REACHES the
# published results dict, which is what the dashboard and the selector read.

_CSV_FILES = [
    "SOLUSDT_4h_full.csv", "ADAUSDT_4h_full.csv", "AVAXUSDT_4h_full.csv", "NEARUSDT_4h_full.csv",
    "SOLUSDT_1h_full.csv", "ADAUSDT_1h.csv", "ETHUSDT_1h_full.csv", "LINKUSDT_1h_full.csv",
    "BNBUSDT_1h.csv",
]


def _write_corpus(root, rows=300):
    hist = root / "data" / "historical"
    hist.mkdir(parents=True, exist_ok=True)
    header = "timestamp,open,high,low,close,volume\n"
    body = []
    for i in range(rows):
        close = 100.0 + (i % 7) * 0.1
        body.append(f"{i},{close - 0.2},{close + 0.3},{close - 0.4},{close},10.0")
    payload = header + "\n".join(body) + "\n"
    for name in _CSV_FILES:
        (hist / name).write_text(payload, encoding="utf-8")


def test_the_results_dict_publishes_the_absence_for_a_zero_dispersion_window(tmp_path, monkeypatch):
    _write_corpus(tmp_path)
    monkeypatch.chdir(tmp_path)

    # No network: the 5m dataset fetcher is never allowed to reach an exchange from a test.
    monkeypatch.setattr(quant_validation, "_ensure_5m_validation_datasets", lambda *a, **k: [])
    # Every replayed signal returns the same R, so the window's dispersion is exactly zero.
    monkeypatch.setattr(
        quant_validation, "_simulate_single_entry_fast",
        lambda *a, **k: (0.01, False),
    )
    monkeypatch.setattr(
        quant_validation, "_extract_strategy_signals",
        lambda df, strategy, symbol: [(i, "long") for i in range(60, 150, 10)],
    )

    missing = {}
    results = quant_validation.run_bootstrap_entry_significance(
        b_iterations=50, noise_sigma=0.15, seed=7, missing_datasets=missing,
    )

    assert results, "no arm produced a result, so the run did not exercise the publishing path"
    for arm, payload in results.items():
        assert payload["rolling_sharpe"] is None, (
            f"{arm} published a rolling Sharpe for a zero-dispersion window")
        assert payload["rolling_sortino"] is None, (
            f"{arm} published a rolling Sortino for a window with no losing trade")
        assert payload["rolling_validation_window"] >= 10

    # The selector must survive the published absence without raising.
    selector = compute_finrl_rolling_sharpe_selector(results)
    for arm, record in selector["strategy_weights"].items():
        assert record["rolling_sharpe"] is None
        assert record["allocation_weight_mult"] == 0.75
        assert "UNMEASURED" in record["posture"]

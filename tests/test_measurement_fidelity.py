"""tests/test_measurement_fidelity.py — Task 5 of the 2026-10-22 audit remediation.

Pins the measurement machinery itself: a Sharpe ratio annualized by the trade frequency a run
actually produced (not by a constant), trailing stops that may only be tested against the bar they
are carried into, higher-timeframe values that may only be read once their bar has closed, indicator
warm-up windows that stay NaN instead of being back-filled, and datasets an arm asked for and did
not find being reported as missing instead of silently skipped.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Verified: repo root is on sys.path via tests/conftest.py:33-34, and
# tests/test_gate_toggle_wiring.py:15 already imports `from backtests.backtest_ablation_harness import ...`
# as a namespace package. Use that qualified form; add no extra sys.path handling.
from backtests.backtest_ablation_harness import annualized_sharpe

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS = REPO_ROOT / "backtests" / "backtest_ablation_harness.py"
GATE_ABLATION = REPO_ROOT / "scripts" / "run_gate_ablation.py"
TRIGGER_COMPARISON = REPO_ROOT / "scripts" / "compare_trigger_modes_840k.py"
DASHBOARD = REPO_ROOT / "dashboard.py"

ALL_GATES_OFF = {
    "gate_1d_trend": False,
    "gate_wick_exhaustion": False,
    "gate_divergence": False,
    "gate_td9": False,
    "gate_volume_profile": False,
}


def _frame(closes):
    c = np.asarray(closes, dtype=float)
    o = np.concatenate([[c[0]], c[:-1]])
    return pd.DataFrame({
        "open": o,
        "high": np.maximum(o, c) * 1.004,
        "low": np.minimum(o, c) * 0.996,
        "close": c,
        "volume": np.full(len(c), 100.0),
        "timestamp": np.arange(len(c)) * 4,
    })


def _fixture_closes(n):
    rng = np.random.default_rng(0)
    steps = rng.normal(0.004, 0.012, n)
    return list(100.0 * np.exp(np.cumsum(steps)))


def _md_table_rows(text):
    """Parses the rendered report's markdown table into one {header: cell} dict per arm row."""
    lines = [ln for ln in text.splitlines() if ln.startswith("| ")]
    assert lines, "the rendered report carries no markdown table"
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    return [dict(zip(header, [c.strip() for c in ln.strip("|").split("|")])) for ln in lines[2:]]


def _arm(n):
    """Runs the real harness over a deterministic in-memory frame; n selects the arm's trade count."""
    from backtests.backtest_ablation_harness import run_ablation_simulation
    return run_ablation_simulation(
        _frame(_fixture_closes(n)), _frame([100.0] * 10), ALL_GATES_OFF
    )


# ---------------------------------------------------------------------------
# Step 3 — annualization
# ---------------------------------------------------------------------------

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


def test_degenerate_inputs_have_no_measurable_sharpe():
    """Below five closed trades there is no Sharpe on this axis, so it is absence, not a number.

    The previous version of this test pinned a MEASURED-LOOKING ``0.0`` for degenerate input, which
    is the same defect the EV axis already refuses: a reader cannot tell "the ratio is zero" from
    "the ratio was never measurable".
    """
    assert annualized_sharpe([], bars=2190, hours_per_bar=4.0) is None
    assert annualized_sharpe([0.01], bars=2190, hours_per_bar=4.0) is None
    assert annualized_sharpe([0.01, -0.01], bars=0, hours_per_bar=4.0) is None


@pytest.mark.parametrize("pnls", [
    [0.02, -0.01],
    [0.02, -0.01, 0.05],
    [0.02, -0.01, 0.05, -0.03],
])
def test_sharpe_axis_refuses_the_sample_size_the_ev_axis_refuses(pnls):
    """A dispersed series is still too small: 2..MIN-1 trades gate the Sharpe like the EV."""
    from backtests.backtest_ablation_harness import MIN_BOOTSTRAP_TRADES

    assert len(pnls) < MIN_BOOTSTRAP_TRADES
    assert annualized_sharpe(pnls, bars=2190, hours_per_bar=4.0) is None


def test_sharpe_axis_measures_the_sample_the_ev_axis_measures():
    from backtests.backtest_ablation_harness import MIN_BOOTSTRAP_TRADES

    pnls = [0.02, -0.01, 0.05, -0.03, 0.01]
    assert len(pnls) == MIN_BOOTSTRAP_TRADES
    assert annualized_sharpe(pnls, bars=2190, hours_per_bar=4.0) is not None


def test_dispersion_between_the_guard_and_the_divisor_is_refused_not_scaled():
    """The guard floor and the divisor are one threshold, so no band escapes both.

    The old body guarded at ``eps * max(|mean|, 1)`` (~2.2e-16) and divided by ``std + 1e-9``, so a
    series with ``2.2e-16 < std <= 1e-9`` passed the guard and was divided by the epsilon: a
    ~1e6 magnitude published as a measured Sharpe.
    """
    pnls = [0.001, 0.001 + 2e-10, 0.001 - 2e-10, 0.001 + 1e-10, 0.001 - 1e-10]
    std = float(np.std(pnls))
    assert np.finfo(float).eps < std <= 1e-9, f"fixture drifted out of the band: std={std:g}"
    assert annualized_sharpe(pnls, bars=2190, hours_per_bar=4.0) is None


def test_harness_sharpe_comes_from_the_pure_function_and_not_the_old_constant():
    arm = _arm(200)
    pnls = [t["net_pnl"] for t in arm["trades"]]
    assert len(pnls) >= 2
    assert arm["sharpe"] == round(annualized_sharpe(pnls, bars=200, hours_per_bar=4.0), 2)
    constant_factor = np.mean(pnls) / (np.std(pnls) + 1e-9) * np.sqrt(365 * 6)
    assert arm["sharpe"] != round(float(constant_factor), 2)


def test_harness_publishes_the_annualization_basis_it_used():
    arm = _arm(200)
    n = len(arm["trades"])
    assert arm["sharpe_trades_per_year"] == round(n / ((200 * 4.0) / 8760.0), 2)


# ---------------------------------------------------------------------------
# Review finding I-1 — the Sharpe axis needs absence handling, not an epsilon default
# ---------------------------------------------------------------------------

def test_variance_free_trade_series_has_no_measurable_sharpe():
    """A sample with no dispersion has no Sharpe: its denominator is zero, whatever the epsilon."""
    for pnls in ([0.05] * 6, [0.0] * 6, [-0.02] * 3):
        assert annualized_sharpe(pnls, bars=2190, hours_per_bar=4.0) is None


@pytest.mark.parametrize("n_trades", [6, 20])
def test_variance_free_sharpe_never_reaches_a_reader_as_a_number(n_trades):
    from backtests.backtest_ablation_harness import format_sharpe

    text = format_sharpe(annualized_sharpe([0.05] * n_trades, bars=2190, hours_per_bar=4.0), n_trades)
    assert text.startswith("n/a"), text
    assert "100000000000" not in text
    assert "0.00" not in text


@pytest.mark.parametrize("sharpe, n_trades, reason", [
    (None, 0, "no closed trades"),
    (None, 1, "single closed trade"),
    (None, 12, "variance-free"),
])
def test_sharpe_formatter_prints_absence_as_absence_and_names_the_cause(sharpe, n_trades, reason):
    from backtests.backtest_ablation_harness import format_sharpe

    text = format_sharpe(sharpe, n_trades)
    assert text.startswith("n/a"), text
    assert reason in text, f"{text!r} does not say why nothing was measured"


def test_sharpe_formatter_prints_a_measured_value_as_a_plain_number():
    from backtests.backtest_ablation_harness import format_sharpe

    assert format_sharpe(0.87, 80) == "0.87"
    assert format_sharpe(-1.5, 80) == "-1.50"


@pytest.mark.parametrize("n_trades, expected_fragment", [
    (0, "no closed trades"),
    (1, "single closed trade"),
    (2, "only 2 closed trades"),
    (4, "only 4 closed trades"),
    (12, "variance-free"),
])
def test_sharpe_formatter_names_the_cause_that_actually_applies(n_trades, expected_fragment):
    """The cause list is exhaustive: an arm floored for sample size is not labelled variance-free.

    The formatter infers the cause from ``n_trades``, so a 2-trade arm (whose dispersion is real)
    used to be described as a "variance-free trade series" — a stated cause that was not the cause.
    """
    from backtests.backtest_ablation_harness import format_sharpe

    text = format_sharpe(None, n_trades)
    assert text.startswith("n/a"), text
    assert expected_fragment in text, text
    if n_trades < 2 or 1 < n_trades < 5:
        assert "variance-free" not in text, text


def test_small_sample_sharpe_absence_reaches_the_report_naming_its_own_count(monkeypatch, tmp_path):
    from backtests.backtest_ablation_harness import run_ablation_simulation

    small = _arm(120)
    assert 0 < small["n_trades"] < 5, f"fixture drifted: {small['n_trades']} trades"
    assert small["sharpe"] is None
    rows = _render_one_arm_table(
        monkeypatch, tmp_path,
        [_arm_row("Full Stack (All Gates ON)", small)])
    assert f"only {small['n_trades']} closed trades" in rows[0]["Sharpe"], rows[0]

    zero = run_ablation_simulation(_frame([100.0] * 120), _frame([100.0] * 10), ALL_GATES_OFF)
    rows = _render_one_arm_table(
        monkeypatch, tmp_path, [_arm_row("Full Stack (All Gates ON)", zero)])
    assert "no closed trades" in rows[0]["Sharpe"], rows[0]


def test_zero_trade_arm_publishes_sharpe_absence_alongside_its_ev_absence():
    zero = _arm(100)
    assert zero["n_trades"] == 0
    assert zero["sharpe"] is None
    assert zero["sharpe_trades_per_year"] is None


def test_single_trade_arm_publishes_sharpe_absence():
    """One closed trade has no dispersion: a 0.00 Sharpe there is a measured zero for no measurement."""
    from backtests.backtest_ablation_harness import format_sharpe

    single = _arm(110)
    assert single["n_trades"] == 1, f"fixture drifted: {single['n_trades']} trades, expected 1"
    assert single["sharpe"] is None
    assert format_sharpe(single["sharpe"], single["n_trades"]).startswith("n/a")


def test_the_trades_per_year_basis_is_computed_in_one_place():
    """annualized_sharpe and the published basis must not carry two copies of the same window."""
    src = HARNESS.read_text(encoding="utf-8")
    assert "8760.0" not in src, (
        "the annual-hours window is restated outside the shared helper, so it can drift from the "
        "factor annualized_sharpe actually applied")
    assert "max((total_bars * 4.0)" not in src, "the basis still recomputes trades/year inline"


def test_published_basis_is_the_factor_the_sharpe_used():
    from backtests.backtest_ablation_harness import measured_trades_per_year

    arm = _arm(200)
    n = len(arm["trades"])
    per_year = measured_trades_per_year(n, 200, 4.0)
    assert arm["sharpe_trades_per_year"] == round(per_year, 2)
    pnls = [t["net_pnl"] for t in arm["trades"]]
    reconstructed = float(np.mean(pnls) / np.std(pnls) * np.sqrt(per_year))
    assert arm["sharpe"] == pytest.approx(round(reconstructed, 2), abs=0.02)


def test_measured_trades_per_year_reports_no_frequency_for_an_empty_arm():
    from backtests.backtest_ablation_harness import measured_trades_per_year

    assert measured_trades_per_year(0, 2190, 4.0) is None
    assert measured_trades_per_year(10, 0, 4.0) is None
    assert measured_trades_per_year(10, 2190, 4.0) == pytest.approx(10 / ((2190 * 4.0) / (365.0 * 24.0)))


@pytest.mark.parametrize("relpath", [
    "backtests/seven_year_backtest.py",
    "backtests/seven_year_ai_sharpened_backtest.py",
    "scripts/compare_trigger_modes_840k.py",
])
def test_runner_tests_carried_stop_before_ratcheting_it(relpath):
    """The intrabar lookahead fix is an ordering property; guard the order.

    Anchored on REPO_ROOT like every other source-level test here: a bare relative path makes this
    guard pass or fail depending on the directory pytest was invoked from.
    """
    src = (REPO_ROOT / relpath).read_text(encoding="utf-8").splitlines()
    ratchet = [i for i, ln in enumerate(src) if re.search(r"^\s*sl_px = (max|min)\(sl_px,", ln)]
    assert ratchet, f"{relpath}: no trailing ratchet found"
    for i in ratchet:
        window = src[i + 1:i + 4]
        assert not any(re.search(r"if (low <= sl_px|high >= sl_px):", ln) for ln in window), (
            f"{relpath}:{i + 1} ratchets sl_px and then tests it against the same bar"
        )


# ---------------------------------------------------------------------------
# Step 5 — higher-timeframe resample lookahead
# ---------------------------------------------------------------------------

def test_htf_series_are_shifted_by_one_of_their_own_periods_before_reindex():
    src = TRIGGER_COMPARISON.read_text(encoding="utf-8")
    aligned = [ln for ln in src.splitlines() if ".reindex(ind_5m.index" in ln]
    assert len(aligned) >= 3, (
        "expected the 1H bias, the 15m confirm and the exported 15m EMA to be aligned onto the 5m index")
    for ln in aligned:
        assert ".shift(1).reindex(" in ln, (
            f"HTF series reaches the 5m grid without a one-period lag: {ln.strip()}")
        assert 'resample("5min")' not in ln, (
            f"forward-filling a resampled HTF close publishes a bar before it closes: {ln.strip()}")


# ---------------------------------------------------------------------------
# Step 6 — indicator warm-up windows stay NaN
# ---------------------------------------------------------------------------

def _indicator_frame(n=200):
    from quant_validation import _prepare_indicator_dataframe
    return _prepare_indicator_dataframe(_frame(_fixture_closes(n)))


def test_atr_is_nan_during_warmup_instead_of_backfilled():
    df = _indicator_frame()
    assert df["atr"].isna().sum() == 13
    assert np.isfinite(df["atr"].iloc[13])
    assert df["atr"].iloc[13:].notna().all()


def test_cloud_is_projected_26_bars_ahead_like_the_live_engine():
    df = _indicator_frame()
    assert df["cloud_top"].isna().sum() == 77
    tenkan = (df["high"].rolling(9).max() + df["low"].rolling(9).min()) / 2.0
    kijun = (df["high"].rolling(26).max() + df["low"].rolling(26).min()) / 2.0
    same_bar_span_a = (tenkan + kijun) / 2.0
    same_bar_span_b = (df["high"].rolling(52).max() + df["low"].rolling(52).min()) / 2.0
    same_bar_cloud_top = np.maximum(same_bar_span_a, same_bar_span_b)
    assert df["cloud_top"].iloc[100] == pytest.approx(float(same_bar_cloud_top.iloc[74]))
    assert df["cloud_top"].iloc[100] != pytest.approx(float(same_bar_cloud_top.iloc[100]))


# The projection used to be conditional: `if len(df) >= 78` shifted, `else` published the raw span at
# the bar that produced it — the same-bar lookahead the shift exists to remove, handed back to every
# short frame. The guard is gone, so the 26-bar projection must hold at every length: a frame too short
# to have measured a cloud says so with NaN instead of with a value read off its own bar.
@pytest.mark.parametrize(
    "n, expected_valid_cloud_cells",
    [
        (20, 0),    # shorter than the 52-bar span itself
        (52, 0),    # span_b becomes computable on the last bar, so it is still 26 bars from being publishable
        (60, 0),    # the brief's short frame: no cloud, not a same-bar cloud
        (77, 0),    # the bar immediately below the deleted guard's threshold
        (78, 1),    # the threshold bar: one projected cell, not two
        (104, 27),
        (200, 123),
    ],
)
def test_the_cloud_is_projected_26_bars_at_every_frame_length(n, expected_valid_cloud_cells):
    df = _indicator_frame(n)
    assert len(df) == n, "the fixture must survive indicator preparation intact for this assertion to mean anything"
    for col in ("cloud_top", "cloud_bottom"):
        valid = int(df[col].notna().sum())
        assert valid == expected_valid_cloud_cells, (
            f"{col}: a {n}-bar frame published {valid} cloud cells, expected {expected_valid_cloud_cells}. "
            "A count above max(0, n - 77) means the cloud reached the frame unshifted, i.e. the same-bar "
            "lookahead the 26-bar projection is there to remove."
        )
        if expected_valid_cloud_cells == 0:
            assert df[col].isna().all(), f"{col}: a {n}-bar frame must carry no cloud value at all"


@pytest.mark.parametrize("field", ["atr", "cloud_top", "cloud_bottom"])
def test_swing_entry_gate_treats_a_missing_indicator_as_no_entry(field):
    from dataclasses import replace

    from strategy_base import StrategyContext, SwingKumoStrategy
    strat = SwingKumoStrategy()
    ctx = StrategyContext(
        symbol="TEST/USDT", timeframe="4h", current_price=100.0, open_price=99.0,
        high_price=101.0, low_price=98.5, volume=10.0, atr=1.5, adx=40.0, rsi=60.0,
        tenkan=99.0, kijun=98.0, cloud_top=97.0, cloud_bottom=96.0,
    )
    assert strat.should_long(ctx) is True, "the fixture must be a valid entry before the field is removed"
    broken = replace(ctx, **{field: float("nan")})
    assert strat.should_long(broken) is False
    assert strat.should_short(broken) is False


# ---------------------------------------------------------------------------
# Step 7 — silently-skipped datasets become a reported fact
# ---------------------------------------------------------------------------

SWING_DATASETS = [
    "SOLUSDT_4h_full.csv",
    "ADAUSDT_4h_full.csv",
    "AVAXUSDT_4h_full.csv",
    "NEARUSDT_4h_full.csv",
]


def _run_bootstrap_with_one_present_dataset(tmp_path, monkeypatch):
    import quant_validation

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "historical").mkdir(parents=True)
    _frame(_fixture_closes(300)).to_csv(
        tmp_path / "data" / "historical" / "BTCUSDT_1h_full.csv", index=False
    )
    monkeypatch.setattr(quant_validation, "_ensure_5m_validation_datasets", lambda *a, **k: ["BTCUSDT_1h_full.csv"])
    monkeypatch.setattr(
        quant_validation, "_extract_strategy_signals",
        lambda df, strategy, symbol: [(70, "long"), (90, "long"), (110, "long")],
    )
    gaps = {}
    res = quant_validation.run_bootstrap_entry_significance(
        b_iterations=50, noise_sigma=0.0, missing_datasets=gaps
    )
    return res, gaps


def test_bootstrap_arms_that_asked_for_absent_datasets_record_them(tmp_path, monkeypatch):
    res, gaps = _run_bootstrap_with_one_present_dataset(tmp_path, monkeypatch)
    assert "SWING_4H" not in res, "an arm with no usable dataset must not publish a measured result"
    assert gaps["SWING_4H"] == SWING_DATASETS
    assert gaps["GRID_MAKER"] == [
        "SOLUSDT_1h_full.csv", "ADAUSDT_1h.csv", "ETHUSDT_1h_full.csv",
        "LINKUSDT_1h_full.csv", "BNBUSDT_1h.csv",
    ]
    assert "SCALP_SHORT" not in gaps, "an arm whose datasets all resolved must not claim a gap"


def test_missing_datasets_are_additive_to_the_arm_result(tmp_path, monkeypatch):
    res, gaps = _run_bootstrap_with_one_present_dataset(tmp_path, monkeypatch)
    for name, arm in res.items():
        assert arm["missing_datasets"] == gaps.get(name, [])
        for key in ("Sampled Signals (K)", "z_score", "p_value", "rolling_sharpe",
                    "strategy_name", "significant_edge", "ratchet_survival_rate_pct"):
            assert key in arm, f"{name} lost {key}, which existing readers consume"
    assert res["GRID_MAKER"]["missing_datasets"] == gaps["GRID_MAKER"]


def test_bootstrap_call_signature_is_unchanged_for_existing_callers(tmp_path, monkeypatch):
    import quant_validation

    _run_bootstrap_with_one_present_dataset(tmp_path, monkeypatch)
    res = quant_validation.run_bootstrap_entry_significance(b_iterations=50, noise_sigma=0.0)
    assert isinstance(res, dict)
    assert res


def test_report_carries_the_dataset_gap_map(tmp_path, monkeypatch):
    import quant_validation

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        quant_validation, "run_bootstrap_entry_significance",
        lambda *args, **kwargs: kwargs["missing_datasets"].update({"SWING_4H": SWING_DATASETS}) or {"STUB": {}},
    )
    monkeypatch.setattr(
        quant_validation, "compute_finrl_rolling_sharpe_selector",
        lambda bootstrap_res, *a, **k: {"ensemble_winner": "STUB", "strategy_weights": {}},
    )
    report = quant_validation.generate_full_validation_report(force_refresh=True)
    assert report["missing_datasets"] == {"SWING_4H": SWING_DATASETS}
    assert Path("cache", "quant_validation_report.json").exists()


def test_report_main_block_prints_the_dataset_gap():
    main_body = REPO_ROOT.joinpath("quant_validation.py").read_text(encoding="utf-8").split("def main()")[-1]
    assert "[DATA GAP]" in main_body, "the CLI report must print the gap, not only carry it in the dict"
    assert "missing_datasets" in main_body


# ---------------------------------------------------------------------------
# Review finding M-3 — a dataset that exists but cannot be used is still a gap
# ---------------------------------------------------------------------------

def _bootstrap_gap_map(tmp_path, monkeypatch, datasets):
    """Runs the real arm loop over an in-memory data/historical tree and returns (results, gap map).

    `datasets` maps a file name to a DataFrame (present and readable) or to a str (present with that
    exact content). Files an arm asks for that are in neither set stay absent.
    """
    import quant_validation

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "historical").mkdir(parents=True)
    for fname, payload in datasets.items():
        target = tmp_path / "data" / "historical" / fname
        if isinstance(payload, str):
            target.write_text(payload, encoding="utf-8")
        else:
            payload.to_csv(target, index=False)
    present = sorted(datasets)
    monkeypatch.setattr(quant_validation, "_ensure_5m_validation_datasets", lambda *a, **k: present)
    monkeypatch.setattr(
        quant_validation, "_extract_strategy_signals",
        lambda df, strategy, symbol: [(70, "long"), (90, "long"), (110, "long")],
    )
    gaps = {}
    res = quant_validation.run_bootstrap_entry_significance(
        b_iterations=50, noise_sigma=0.0, missing_datasets=gaps
    )
    return res, gaps


def _gap_entries_for(gaps, arm, fname):
    return [e for e in gaps.get(arm, []) if e.startswith(fname)]


def test_present_but_too_short_dataset_is_recorded_with_a_reason(tmp_path, monkeypatch):
    _, gaps = _bootstrap_gap_map(
        tmp_path, monkeypatch, {"BTCUSDT_1h_full.csv": _frame(_fixture_closes(60))})
    entries = _gap_entries_for(gaps, "SCALP_SHORT", "BTCUSDT_1h_full.csv")
    assert entries, (
        f"a 60-bar dataset vanished from the gap map, so missing_datasets: {gaps} reads as "
        "'this arm got everything'")
    assert "unusable" in entries[0] and "120" in entries[0], entries[0]


def test_present_but_unreadable_dataset_is_recorded_with_a_reason(tmp_path, monkeypatch):
    broken = "timestamp,open,high,low,close,volume\n1,1,1,1,1\n2,2,2\n"
    _, gaps = _bootstrap_gap_map(tmp_path, monkeypatch, {"BTCUSDT_1h_full.csv": broken})
    entries = _gap_entries_for(gaps, "SCALP_SHORT", "BTCUSDT_1h_full.csv")
    assert entries, f"a dataset that raised on load vanished from the gap map: {gaps}"
    assert "unusable" in entries[0], entries[0]
    assert entries[0] != "BTCUSDT_1h_full.csv", "the gap must carry a reason, not only the name"


def test_absent_datasets_are_still_recorded_as_bare_names(tmp_path, monkeypatch):
    _, gaps = _bootstrap_gap_map(tmp_path, monkeypatch, {"BTCUSDT_1h_full.csv": _frame(_fixture_closes(300))})
    assert "BTCUSDT_1h_full.csv" not in gaps.get("SCALP_SHORT", []), (
        "a usable dataset must not be reported as a gap")
    assert gaps["SWING_4H"] == SWING_DATASETS, (
        "absent datasets must keep the bare-name shape the bootstrap result already publishes")


def test_report_main_block_describes_a_gap_without_claiming_the_file_is_absent():
    main_body = REPO_ROOT.joinpath("quant_validation.py").read_text(encoding="utf-8").split("def main()")[-1]
    assert "[DATA GAP]" in main_body
    assert "no dataset found for" not in main_body, (
        "the gap map now also lists datasets that exist but are unusable; printing them as 'no dataset "
        "found' states the opposite of what was measured")


# ---------------------------------------------------------------------------
# Step 7b — the dashboard panel renders the gap map, not only the CLI
# ---------------------------------------------------------------------------
# What is proven below and what is not: `_missing_dataset_warnings` is a pure function that turns the
# report's arm -> reasons map into sentences, so these tests can call the shipped code and check every
# decision. The `st.warning(...)` loop in dashboard.py's validation panel is NOT executed by any of
# them — the committed suite has no Streamlit AppTest harness, and importing dashboard.py runs a
# 4,500-line top-level script against the real DB and live price endpoints. A helper that returns a
# string cannot lie about what it decided; only the rendering one line further out is unverified, and
# test_panel_wires_the_gap_helper_into_a_warning below is a source-text check of that line, nothing more.

def _dashboard_helper(name):
    """Compiles ONE pure top-level helper out of dashboard.py and returns it, without booting the app.

    dashboard.py cannot be imported in a test (module-level DB reads, HTTP fetches and chart rendering),
    and its rendering cannot be exercised without an AppTest harness this suite does not have. The
    helpers under test are pure and read only module-level constants, so their own AST node — plus any
    module-level assignment they load by name — is compiled in an empty namespace and called for real.
    That executes the shipped body, which is a stronger claim than matching its source text.
    """
    import ast

    tree = ast.parse(DASHBOARD.read_text(encoding="utf-8"), filename=str(DASHBOARD))
    target = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
    assert target is not None, f"dashboard.py has no module-level function named {name}"

    loaded = {n.id for n in ast.walk(target) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    ns = {}
    for node in [target] + [
        n for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in loaded for t in n.targets)
    ]:
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(DASHBOARD), "exec"), ns)
    fn = ns[name]
    assert callable(fn), f"{name} is not callable after extraction"
    return fn


_missing_dataset_warnings = _dashboard_helper("_missing_dataset_warnings")


@pytest.mark.parametrize("gap_map", [
    None,                                             # key absent from the report entirely
    {},                                               # every arm got everything it asked for
    {"SWING_4H": []},                                 # an arm whose list is empty
    {"SWING_4H": None},
    {"SWING_4H": ""},
    {"SWING_4H": [None]},                             # JSON nulls are not reasons
    {"SWING_4H": [None, ""]},
    {"SWING_4H": 7},                                  # a bare number names no dataset
    {"SWING_4H": True},
    "SWING_4H",                                       # the whole map malformed
    ["SWING_4H"],
    5,
    True,
])
def test_an_absent_or_malformed_gap_map_prints_nothing(gap_map):
    """Silence is the only honest output here: a warning about a gap nobody measured would be the same
    defect class in the other direction."""
    assert _missing_dataset_warnings(gap_map) == []


def test_one_gap_message_names_the_arm_and_the_reason_the_report_recorded():
    msgs = _missing_dataset_warnings({"SWING_4H": ["SOLUSDT_4h_full.csv"]})
    assert len(msgs) == 1, msgs
    assert "SWING_4H" in msgs[0], "the sentence has to name the arm whose numbers are affected"
    assert "SOLUSDT_4h_full.csv" in msgs[0], "and the dataset it asked for and did not get"
    assert "fewer datasets" in msgs[0], (
        f"the panel must say what the gap does to the arm's numbers, not just that data is missing: {msgs[0]}")


def test_every_reason_of_an_arm_is_in_that_arms_message():
    unusable = ("NEARUSDT_4h_full.csv (unusable: 60 bars after indicator preparation, "
                "below the 120-bar minimum this arm needs)")
    msgs = _missing_dataset_warnings({"SWING_4H": list(SWING_DATASETS) + [unusable]})
    assert len(msgs) == 1, msgs
    for reason in SWING_DATASETS + [unusable]:
        assert reason in msgs[0], f"{reason} was recorded as a gap and dropped from what the panel says"
    # Both gap kinds the map carries must be covered by the wording: absent files and present-but-unusable
    # ones. Asserting only the file names would let the sentence go back to claiming everything is absent.
    assert "absent" not in msgs[0].lower(), (
        f"the wording must not assert absence for a file that exists but was unusable: {msgs[0]}")


def test_one_message_per_arm_that_has_gaps_and_none_for_arms_without_them():
    msgs = _missing_dataset_warnings({
        "SWING_4H": SWING_DATASETS,
        "SCALP_SHORT": ["BTCUSDT_1h_full.csv"],
        "CHOP_GRID": [],
    })
    assert len(msgs) == 2, f"an arm with no gaps must not be described as having one: {msgs}"
    swing = next(m for m in msgs if "SWING_4H" in m)
    scalp = next(m for m in msgs if "SCALP_SHORT" in m)
    assert "CHOP_GRID" not in swing and "CHOP_GRID" not in scalp
    assert "SCALP_SHORT" not in swing and "SWING_4H" not in scalp, (
        "one arm's gap must not be described as another arm's")


def test_panel_wires_the_gap_helper_into_a_warning():
    """Source-text check of the one line the helper tests cannot reach — see the section note above.

    This proves the panel reads the key from the same report object it already loads and feeds the
    helper's output to st.warning inside the try block. It does not prove Streamlit renders it.
    """
    src = DASHBOARD.read_text(encoding="utf-8")
    assert "missing_datasets" in src, (
        "the panel dropped the report's dataset-gap map, so a reader cannot learn an arm ran on a subset")
    panel = src[src.index("def _missing_dataset_warnings"):]
    wired = [ln for ln in panel.splitlines()
             if "_missing_dataset_warnings(jv_report" in ln and 'get("missing_datasets")' in ln]
    assert len(wired) == 1, wired
    call_site = src[src.index(wired[0]):]
    assert "st.warning(" in call_site.splitlines()[1], (
        "the helper's sentences must reach the panel as warnings, not be computed and dropped")



# ---------------------------------------------------------------------------
# Handed over by Task 4 — the fabricated zero-width CI and the zero-trade crash
# ---------------------------------------------------------------------------

def _arm_row(name, arm):
    """Maps an arm dict exactly the way scripts.run_gate_ablation.evaluate_all does."""
    return {
        "name": name,
        "n_trades": arm["n_trades"],
        "win_rate": arm["win_rate"],
        "total_return": arm["total_return"],
        "profit_factor": arm["profit_factor"],
        "sharpe": arm["sharpe"],
        "sharpe_trades_per_year": arm["sharpe_trades_per_year"],
        "ev_mean": arm["ev_mean"],
        "ev_ci_90": arm["ev_ci_90"],
        "trades": arm["trades"],
        "gate_fires": arm["gate_fires"],
    }


def test_zero_trade_and_populated_arms_expose_the_same_keys():
    populated = _arm(200)
    zero = _arm(100)
    assert zero["n_trades"] == 0
    assert populated["n_trades"] > 0
    assert set(zero) == set(populated), (
        f"zero-trade arm keys {sorted(set(zero) ^ set(populated))} diverged from the populated arm")


def test_zero_trade_arm_reports_absence_rather_than_a_measured_zero():
    zero = _arm(100)
    assert zero["ev_mean"] is None
    assert zero["ev_ci_90"] is None


def test_arm_too_small_to_bootstrap_reports_no_interval():
    small = _arm(120)
    assert 0 < small["n_trades"] < 5
    assert small["ev_mean"] is not None
    assert small["ev_ci_90"] is None


def test_arm_large_enough_to_bootstrap_reports_a_real_interval():
    arm = _arm(200)
    assert arm["n_trades"] >= 5
    lo, hi = arm["ev_ci_90"]
    assert lo < hi


@pytest.mark.parametrize("ev_mean, ev_ci_90, has_interval", [
    (None, None, False),
    (1.25, None, False),
    (1.25, (0.5, 0.5), False),
    (1.25, (-0.5, 2.5), True),
])
def test_ev_ci_formatter_prints_absence_as_absence(ev_mean, ev_ci_90, has_interval):
    from backtests.backtest_ablation_harness import format_ev_ci
    text = format_ev_ci(ev_mean, ev_ci_90)
    assert ("[" in text) is has_interval
    assert ("n/a" in text) is (not has_interval)
    assert "+0.00%" not in text
    if has_interval:
        assert text == "+1.25% [-0.50%, +2.50%]"


def test_both_renderers_print_absence_instead_of_a_degenerate_interval(monkeypatch, tmp_path):
    from backtests.backtest_ablation_harness import run_ablation_simulation
    import scripts.run_gate_ablation as rga
    from scripts.run_gate_ablation import FULL_STACK_NAME

    flat = _frame([100.0] * 120)
    zero_arm = run_ablation_simulation(flat, _frame([100.0] * 10), ALL_GATES_OFF)
    small_arm = _arm(120)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rga, "SYMBOLS", ["TESTUSDT"])
    path = rga.write_report({"TESTUSDT": [
        _arm_row(FULL_STACK_NAME, zero_arm),
        _arm_row("Ablate Gate A (1D Trend Support OFF)", small_arm),
    ]})
    text = Path(path).read_text(encoding="utf-8")
    table_rows = [ln for ln in text.splitlines() if ln.startswith("| ") and "Trades (N)" not in ln and ":---" not in ln]
    assert len(table_rows) == 2
    for row in table_rows:
        assert "n/a" in row, f"the per-asset table renders a measured CI where none was estimated: {row}"
    assert "[+0.00%, +0.00%]" not in text


def test_both_report_printers_route_through_the_shared_formatter():
    harness_src = HARNESS.read_text(encoding="utf-8")
    ablation_src = GATE_ABLATION.read_text(encoding="utf-8")
    assert 'format_ev_ci(res["ev_mean"], res["ev_ci_90"])' in harness_src
    assert "{res['ev_ci_90'][0]" not in harness_src
    assert "format_ev_ci(" in ablation_src
    assert "{r['ev_ci_90'][0]" not in ablation_src


# ---------------------------------------------------------------------------
# Review findings I-1 + M-1 — the Sharpe axis renders through a shared formatter, and the
# annualization basis it was computed on reaches a human-readable table
# ---------------------------------------------------------------------------

def test_both_report_printers_route_through_the_shared_sharpe_formatter():
    harness_src = HARNESS.read_text(encoding="utf-8")
    ablation_src = GATE_ABLATION.read_text(encoding="utf-8")
    assert 'format_sharpe(res["sharpe"], res["n_trades"])' in harness_src
    assert "{res['sharpe']:>6.2f}" not in harness_src, "the console still formats a possibly-absent Sharpe as a number"
    assert 'format_sharpe(r.get("sharpe"), r.get("n_trades", 0))' in ablation_src
    assert "{r['sharpe']:.2f}" not in ablation_src, "the markdown table still formats a possibly-absent Sharpe as a number"
    for label, src in (("harness", harness_src), ("ablation writer", ablation_src)):
        assert "format_sharpe(" in src, f"{label} does not render Sharpe through the shared formatter"


def test_arm_to_row_mapping_carries_the_annualization_basis():
    src = GATE_ABLATION.read_text(encoding="utf-8")
    assert '"sharpe_trades_per_year": res["sharpe_trades_per_year"]' in src, (
        "evaluate_all drops the basis key, so no reader ever sees the frequency the Sharpe used")


def _render_one_arm_table(monkeypatch, tmp_path, rows):
    import scripts.run_gate_ablation as rga

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rga, "SYMBOLS", ["TESTUSDT"])
    path = rga.write_report({"TESTUSDT": rows})
    return _md_table_rows(Path(path).read_text(encoding="utf-8"))


def test_report_table_names_the_annualization_basis_a_reader_can_use(monkeypatch, tmp_path):
    arm = _arm(200)
    assert arm["n_trades"] >= 2
    rows = _render_one_arm_table(monkeypatch, tmp_path, [_arm_row("Full Stack (All Gates ON)", arm)])
    assert len(rows) == 1
    assert rows[0]["Sharpe"] == f"{arm['sharpe']:.2f}"
    assert rows[0]["Sharpe Basis (trades/yr)"] == f"{arm['sharpe_trades_per_year']:.1f} trades/yr"


def test_report_table_prints_sharpe_absence_for_a_zero_trade_arm(monkeypatch, tmp_path):
    from backtests.backtest_ablation_harness import run_ablation_simulation

    zero_arm = run_ablation_simulation(_frame([100.0] * 120), _frame([100.0] * 10), ALL_GATES_OFF)
    assert zero_arm["n_trades"] == 0
    rows = _render_one_arm_table(monkeypatch, tmp_path, [_arm_row("Full Stack (All Gates ON)", zero_arm)])
    assert rows[0]["Sharpe"].startswith("n/a"), rows[0]
    assert rows[0]["Sharpe"] != "0.00"
    assert rows[0]["Sharpe Basis (trades/yr)"] == "n/a (no closed trades)"


def test_console_header_names_the_sharpe_basis_column():
    src = HARNESS.read_text(encoding="utf-8")
    assert "Sharpe Basis" in src.split("def main()")[-1], (
        "the console table has no basis column, so the frequency stays invisible")


# ---------------------------------------------------------------------------
# Review finding M-2 — one definition of the bootstrap minimum, interpolated into both sentences
# ---------------------------------------------------------------------------

def test_the_bootstrap_minimum_is_defined_exactly_once():
    pattern = re.compile(r"^MIN_BOOTSTRAP_TRADES\s*=", re.M)
    definers = sorted(p.name for p in (HARNESS, GATE_ABLATION)
                      if pattern.search(p.read_text(encoding="utf-8")))
    assert definers == ["backtest_ablation_harness.py"], (
        f"{definers} each restate the minimum, so moving one threshold yields two contradictory "
        "sentences in the same report")

    import backtests.backtest_ablation_harness as harness
    import scripts.run_gate_ablation as rga
    assert rga.MIN_BOOTSTRAP_TRADES == harness.MIN_BOOTSTRAP_TRADES



# ---------------------------------------------------------------------------
# Final dispatch, item 2 — the profit-factor axis gets the same absence convention
# ---------------------------------------------------------------------------

QUANT_VALIDATION = REPO_ROOT / "quant_validation.py"


def test_lossless_arm_publishes_no_profit_factor_instead_of_a_huge_ratio():
    """A series with no losing trade has no denominator: gross_win / 1e-6 is not a measurement."""
    from backtests.backtest_ablation_harness import run_ablation_simulation

    arm = _arm(120)
    assert 0 < arm["n_trades"] < 5, f"fixture drifted: {arm['n_trades']} trades"
    pnls = [t["net_pnl"] for t in arm["trades"]]
    assert all(p > 0 for p in pnls), f"fixture is no longer lossless: {pnls}"
    assert arm["profit_factor"] is None
    assert arm["profit_factor"] != 0.0

    zero = run_ablation_simulation(_frame([100.0] * 120), _frame([100.0] * 10), ALL_GATES_OFF)
    assert zero["n_trades"] == 0
    assert zero["profit_factor"] is None


def test_profit_factor_formatter_names_the_absence_and_its_cause():
    from backtests.backtest_ablation_harness import format_profit_factor

    assert format_profit_factor(1.23, 80) == "1.23"
    assert format_profit_factor(None, 0).startswith("n/a")
    assert "no closed trades" in format_profit_factor(None, 0)
    assert format_profit_factor(None, 8).startswith("n/a")
    assert "no losing" in format_profit_factor(None, 8)
    assert "999" not in format_profit_factor(None, 8)


def test_both_renderers_print_profit_factor_absence_instead_of_a_number(monkeypatch, tmp_path):
    arm = _arm(120)
    rows = _render_one_arm_table(monkeypatch, tmp_path, [_arm_row("Full Stack (All Gates ON)", arm)])
    cell = rows[0]["Profit Factor"]
    assert cell.startswith("n/a"), rows[0]
    assert cell != "0.00"
    assert "68893" not in cell and "1000000" not in cell


def test_both_report_printers_route_through_the_shared_profit_factor_formatter():
    harness_src = HARNESS.read_text(encoding="utf-8")
    ablation_src = GATE_ABLATION.read_text(encoding="utf-8")
    assert 'format_profit_factor(res["profit_factor"], res["n_trades"])' in harness_src
    assert "{res['profit_factor']:>6.2f}" not in harness_src
    assert 'format_profit_factor(r.get("profit_factor"), r.get("n_trades", 0))' in ablation_src
    assert "{r['profit_factor']:.2f}" not in ablation_src


def test_no_padded_profit_factor_denominator_survives_anywhere():
    """The 1e-6 stand-in denominator, the 999.0 cap and quant_validation's 2.5 are all deleted."""
    harness_src = HARNESS.read_text(encoding="utf-8")
    assert "else 1e-6" not in harness_src, "a lossless arm still gets a synthetic denominator"
    assert "999.0" not in harness_src, "the unreachable 999.0 cap is still published as a ratio"

    qv_src = QUANT_VALIDATION.read_text(encoding="utf-8")
    assert re.search(r"gross_loss[^:\n]*else 2\.5", qv_src) is None, (
        "observed_profit_factor still falls back to a fabricated 2.5 for an arm with no losing trade")


def test_quant_validation_publishes_no_profit_factor_for_an_arm_without_losses():
    """The bootstrap significance arm's own PF: no losing signal means no ratio, not 2.5."""
    from quant_validation import run_bootstrap_entry_significance  # noqa: F401  (import path guard)

    # The full arm needs a historical corpus, so the contract is pinned at the source level above;
    # this test pins the published key's None-awareness through the renderer that reads it.
    from backtests.backtest_ablation_harness import format_profit_factor

    assert format_profit_factor(None, 3).startswith("n/a")

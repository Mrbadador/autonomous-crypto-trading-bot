"""tests/test_measurement_fidelity.py — Task 5 of the 2026-10-22 audit remediation.

Pins the measurement machinery itself: a Sharpe ratio annualized by the trade frequency a run
actually produced (not by a constant), trailing stops that may only be tested against the bar they
are carried into, higher-timeframe values that may only be read once their bar has closed, indicator
warm-up windows that stay NaN instead of being back-filled, and datasets an arm asked for and did
not find being reported as missing instead of silently skipped.
"""

import json
import re
from datetime import datetime, timedelta, timezone
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


# The live engine carried the same conditional projection `quant_validation.py` just lost, plus a
# second fallback at the cloud read: the spans were shifted only when the frame had >= 78 bars,
# every shorter frame published the *raw* span, and a NaN shifted span was replaced by that same
# raw span again. Both routes handed a short frame a cloud built from its own bar. A frame too
# short to project the cloud 26 bars ahead must now REFUSE — no indicators at all — rather than
# publish a same-bar cloud the entry checklist would read as a breakout.
@pytest.mark.parametrize("n", [55, 60, 77])
def test_the_live_engine_refuses_a_4h_frame_too_short_to_project_the_cloud(n):
    from swing_trader import SwingArchitectTrader

    trader = SwingArchitectTrader.__new__(SwingArchitectTrader)
    ind = trader._compute_4h_indicators(_frame(_fixture_closes(n)))
    assert ind == {}, (
        f"a {n}-bar 4H frame published {ind!r}; it cannot know a cloud projected 26 bars ahead, "
        "so it must refuse (the empty indicator dict) instead of publishing a same-bar cloud"
    )


def test_the_live_engine_publishes_the_projected_cloud_not_the_same_bar_cloud():
    from swing_trader import SwingArchitectTrader

    trader = SwingArchitectTrader.__new__(SwingArchitectTrader)
    df = _frame(_fixture_closes(120))
    ind = trader._compute_4h_indicators(df)
    assert ind, "a 120-bar 4H frame can project the cloud 26 bars ahead and must publish it"

    high, low = df["high"], df["low"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2.0
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2.0
    span_a_raw = (tenkan + kijun) / 2.0
    span_b_raw = (high.rolling(52).max() + low.rolling(52).min()) / 2.0
    # The cloud published at the last bar is the one projected 26 bars earlier.
    projected_top = float(np.maximum(span_a_raw.iloc[-27], span_b_raw.iloc[-27]))
    projected_bottom = float(np.minimum(span_a_raw.iloc[-27], span_b_raw.iloc[-27]))
    same_bar_top = float(np.maximum(span_a_raw.iloc[-1], span_b_raw.iloc[-1]))
    assert ind["cloud_top"] == pytest.approx(projected_top)
    assert ind["cloud_bottom"] == pytest.approx(projected_bottom)
    assert not np.isclose(ind["cloud_top"], same_bar_top), (
        "the published cloud is the same-bar cloud — the 26-bar projection did not happen"
    )


def test_a_refused_4h_frame_yields_no_candidate_in_either_direction():
    """The empty indicator dict is the "no entry" sentinel, not a missing-key crash."""
    from swing_trader import SwingArchitectTrader

    cand = SwingArchitectTrader._evaluate_swing_candidate(
        {}, {"bullish_alignment": True, "bearish_alignment": True},
        100.0, 100.0, 30.0, 50.0, 22.0,
    )
    assert cand["direction"] is None, "a frame with no known cloud must not produce a candidate"
    assert cand["near_miss"] is False


# The ablation harness had the same defect by a different route: `.shift(26).fillna(span_raw)`
# back-filled the leading warm-up window with the *unshifted* span, so the arm's opening bars
# were scored against a cloud built from their own highs and lows. The cloud at bar i is the one
# projected 26 bars earlier, so a frame of n bars has a cloud on exactly max(0, n - 77) of them.
@pytest.mark.parametrize(
    "n, expected_valid_cloud_cells",
    [
        (20, 0),    # shorter than the 52-bar span itself
        (52, 0),    # span_b becomes computable, but it is still 26 bars from being publishable
        (60, 0),
        (77, 0),    # the bar immediately below the first publishable one
        (78, 1),    # the first publishable bar: one projected cell, not two
        (104, 27),
        (200, 123),
    ],
)
def test_the_ablation_harness_leaves_the_unprojectable_cloud_window_absent(n, expected_valid_cloud_cells):
    from backtests.backtest_ablation_harness import precompute_4h_indicators_vectorized

    res = precompute_4h_indicators_vectorized(_frame(_fixture_closes(n)))
    for col in ("cloud_top", "cloud_bottom"):
        valid = int(res[col].notna().sum())
        assert valid == expected_valid_cloud_cells, (
            f"{col}: a {n}-bar frame published {valid} cloud cells, expected "
            f"{expected_valid_cloud_cells}. A count above max(0, n - 77) means the leading window "
            "was back-filled with the unshifted span — the same-bar lookahead the 26-bar "
            "projection is there to remove."
        )


def test_the_ablation_harness_publishes_the_projected_cloud_not_the_same_bar_cloud():
    from backtests.backtest_ablation_harness import precompute_4h_indicators_vectorized

    df = _frame(_fixture_closes(200))
    res = precompute_4h_indicators_vectorized(df)

    high, low = df["high"], df["low"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2.0
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2.0
    span_a_raw = (tenkan + kijun) / 2.0
    span_b_raw = (high.rolling(52).max() + low.rolling(52).min()) / 2.0
    projected = np.maximum(span_a_raw.shift(26), span_b_raw.shift(26))
    same_bar = np.maximum(span_a_raw, span_b_raw)
    assert res["cloud_top"].iloc[100] == pytest.approx(float(projected.iloc[100]))
    assert not np.isclose(res["cloud_top"].iloc[100], float(same_bar.iloc[100])), (
        "the published cloud is the same-bar cloud — the 26-bar projection did not happen"
    )


def test_the_ablation_harness_scores_no_bar_whose_cloud_is_unprojectable():
    """A bar with no cloud cannot carry a kumo signal, so the leading window contributes no trades.

    The fixture is a steady uptrend, where the same-bar cloud reads as a breakout on every bar.
    Under the back-filled warm-up window the harness therefore published bullish signals on the
    26 bars whose cloud contained their own highs and lows; with the window left absent it must
    publish none of them, while the projected window is unaffected.
    """
    from backtests.backtest_ablation_harness import precompute_4h_indicators_vectorized

    n = 200
    res = precompute_4h_indicators_vectorized(_frame([100.0 * (1.001 ** i) for i in range(n)]))
    assert res["bullish_kumo"].iloc[:77].sum() == 0, (
        "the harness published a kumo signal on a bar with no measurable cloud"
    )
    assert res["bearish_kumo"].iloc[:77].sum() == 0
    assert res["bullish_kumo"].iloc[77:].any(), (
        "the fixture must produce signals on the projected window, or the guard above is vacuous"
    )


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

def _dashboard_helper(name, path=DASHBOARD, extra_ns=None):
    """Compiles ONE pure top-level helper out of a source file and returns it, without booting it.

    dashboard.py cannot be imported in a test (module-level DB reads, HTTP fetches and chart rendering),
    and its rendering cannot be exercised without an AppTest harness this suite does not have. The
    helpers under test are pure and read only module-level constants, so their own AST node — plus any
    module-level assignment they load by name — is compiled in an empty namespace and called for real.
    That executes the shipped body, which is a stronger claim than matching its source text.

    The compiled set is closed over the helper's own dependencies: a module-level function it calls is
    compiled alongside it, and so is a module-level constant that function calls. Without that, a helper
    that calls another helper raises NameError when the test calls it.

    `path` defaults to dashboard.py; the legacy scripts/live_dashboard.py carries its own copies of
    the same pure renderers (it is a standalone Streamlit script and cannot import dashboard.py),
    and they are extracted the same way.

    `extra_ns` (added for G-5b fix #3) is injected AFTER the file's top-level imports and before the
    helper's own node is exec'd. A decorated function needs its decorator present at exec time —
    `query_system_state_row` is `@st.cache_data`-wrapped, so the walk could not extract it while the
    namespace's `st` was the module-level `import streamlit as st`, if that import happens to resolve.
    Injecting after the imports also means the caller's stand-in is never silently replaced by one of
    the file's own import bindings.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    target = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
    assert target is not None, f"{path.name} has no module-level function named {name}"

    def _loads(node):
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}

    needed = set()
    pending = [target]
    while pending:
        node = pending.pop()
        fresh = _loads(node) - needed
        if not fresh:
            continue
        needed |= fresh
        for other in tree.body:
            if isinstance(other, ast.FunctionDef) and other.name in fresh:
                pending.append(other)

    ns = {}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
            except Exception:
                pass
    for _injected, _value in (extra_ns or {}).items():
        ns[_injected] = _value
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in needed for t in node.targets):
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
            except Exception:
                pass
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in needed and node is not target:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    exec(compile(ast.Module(body=[target], type_ignores=[]), str(path), "exec"), ns)
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


# ---------------------------------------------------------------------------
# Final fix D-2 — the dashboard's profit factor is a measurement or an absence, never a constant
# ---------------------------------------------------------------------------
# What is proven below and what is not: `_profit_factor_delta` is a pure function that turns the
# KPI card's profit factor (a ratio, or None when there is no denominator) into the text the card
# shows, so these tests can call the shipped code and check every decision. The `st.metric(...)`
# call in dashboard.py's KPI row is NOT executed by any of them — the committed suite has no
# Streamlit AppTest harness, and importing dashboard.py runs a 4,500-line top-level script against
# the real DB and live price endpoints. A helper that returns a string cannot lie about what it
# decided; only the rendering one line further out is unverified, and
# test_the_kpi_card_routes_through_the_helper below is a source-text check of that line, nothing
# more.

_profit_factor_delta = _dashboard_helper("_profit_factor_delta")


def test_the_profit_factor_card_publishes_no_fabricated_ratio():
    """The card used to publish 2.62 for a fleet with no losing trade, 1.0 for an empty ledger and
    0.0 for the no-data case: three numbers that were never measured."""
    for win_count, loss_count in [(0, 0), (7, 0), (0, 3), (7, 3)]:
        out = _profit_factor_delta(None, win_count, loss_count)
        assert "2.62" not in out, out
        assert "1.00" not in out, out
        assert not any(ch.isdigit() for ch in out), (
            f"an unmeasured profit factor must print no number at all, not a stand-in: {out}")


def test_no_closed_trades_says_so_instead_of_measuring_zero():
    out = _profit_factor_delta(None, 0, 0)
    assert "not measured" in out.lower(), out
    assert "no closed trades" in out, out
    assert "0.00" not in out and "0.0x" not in out, out


def test_closed_trades_with_no_loser_name_the_missing_denominator():
    out = _profit_factor_delta(None, 7, 0)
    assert "not measured" in out.lower(), out
    assert "no losing" in out, out
    assert "denominator" in out, out
    assert "2.62" not in out and "0.00" not in out, out


def test_a_measured_profit_factor_is_published_as_a_number():
    assert _profit_factor_delta(2.35, 3, 1) == "2.35x Profit Factor (3W / 1L)"
    # A ratio with a real denominator is a measurement even when it is zero: no wins at all.
    assert _profit_factor_delta(0.0, 0, 4) == "0.00x Profit Factor (0W / 4L)"


def test_the_card_does_not_call_a_breakeven_population_lossless():
    out = _profit_factor_delta(None, 4, 2)
    assert "not measured" in out.lower(), out
    assert "no losing" not in out, (
        "there were losing trades (breakevens): calling that 'no losing trade' states the "
        f"opposite of what was measured: {out}")


def test_the_kpi_card_routes_through_the_helper():
    """Source-text check of the one line the helper tests cannot reach — see the section note above.

    It proves the panel publishes the ratio as None when there is no denominator and hands the
    card's delta to the helper. It does not prove Streamlit renders it.
    """
    src = DASHBOARD.read_text(encoding="utf-8")
    assert re.search(
        r"profit_factor\s*=\s*\(gross_profit\s*/\s*gross_loss\)\s*if\s+gross_loss\s*>\s*0\s*"
        r"else\s*None", src), (
        "the panel must publish the ratio as None when there is no denominator, not a constant")
    wired = [ln for ln in src.splitlines() if "delta=_profit_factor_delta(profit_factor" in ln]
    assert len(wired) == 1, wired
    assert wired[0].strip().startswith("delta="), (
        f"the helper's text must be the metric's delta argument: {wired[0]!r}")
    call = src[src.rindex("st.metric(", 0, src.index(wired[0])):]
    assert "delta_color=" in call and "delta_arrow=" in call, (
        "an absent ratio must not wear the green up-arrow st.metric gives a non-negative delta "
        "(streamlit/elements/metric.py::_is_negative_delta only looks for a leading '-')")


# ---------------------------------------------------------------------------
# Final fix D-5 — the bootstrap arm's own profit factor, driven for real
# ---------------------------------------------------------------------------
# The grep in test_no_padded_profit_factor_denominator_survives_anywhere pins the SOURCE of
# quant_validation's absence; it never drives the producer that publishes it. The producer is
# driven here instead: `run_bootstrap_entry_significance` runs over a corpus it can actually read
# (a temp data/historical tree, a patched dataset list, a patched signal extractor) exactly as the
# dataset-gap tests above already do. It needs no real corpus, no file list and no large bootstrap
# budget -- 50 iterations over a 300-bar frame is enough -- and the producer's behaviour is not
# changed to make the test pass.

def _bootstrap_arm_over_a_rising_frame(tmp_path, monkeypatch):
    """Runs the real bootstrap loop over a frame that rises 0.4% a bar, and returns (results, gaps).

    A GRID_MAKER long on that frame reaches its +1.35% target (or its ratcheted stop above the
    entry) on the third bar after entry, so every signal the arm produces is a win: the arm has
    no losing trade and therefore no denominator.
    """
    import quant_validation

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "historical").mkdir(parents=True)
    closes = [100.0 * (1.004 ** i) for i in range(300)]
    _frame(closes).to_csv(tmp_path / "data" / "historical" / "BTCUSDT_1h_full.csv", index=False)
    monkeypatch.setattr(
        quant_validation, "_ensure_5m_validation_datasets", lambda *a, **k: ["BTCUSDT_1h_full.csv"])
    monkeypatch.setattr(
        quant_validation, "_extract_strategy_signals",
        lambda df, strategy, symbol: (
            [(70, "long"), (90, "long"), (110, "long")] if strategy.name == "GRID_MAKER" else []),
    )
    gaps = {}
    res = quant_validation.run_bootstrap_entry_significance(
        b_iterations=50, noise_sigma=0.0, missing_datasets=gaps
    )
    return res, gaps


def _bootstrap_arm_over_a_falling_frame(tmp_path, monkeypatch):
    """The same run over a frame that falls 0.4% a bar: the arm's longs hit their stop."""
    import quant_validation

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "historical").mkdir(parents=True)
    closes = [100.0 * (0.996 ** i) for i in range(300)]
    _frame(closes).to_csv(tmp_path / "data" / "historical" / "BTCUSDT_1h_full.csv", index=False)
    monkeypatch.setattr(
        quant_validation, "_ensure_5m_validation_datasets", lambda *a, **k: ["BTCUSDT_1h_full.csv"])
    monkeypatch.setattr(
        quant_validation, "_extract_strategy_signals",
        lambda df, strategy, symbol: (
            [(70, "long"), (90, "long"), (110, "long")] if strategy.name == "GRID_MAKER" else []),
    )
    gaps = {}
    res = quant_validation.run_bootstrap_entry_significance(
        b_iterations=50, noise_sigma=0.0, missing_datasets=gaps
    )
    return res, gaps


def test_the_bootstrap_arm_publishes_no_profit_factor_when_no_signal_ever_loses(tmp_path, monkeypatch):
    """The producer's behaviour, not its source text: a lossless arm has no denominator."""
    res, gaps = _bootstrap_arm_over_a_rising_frame(tmp_path, monkeypatch)
    assert "GRID_MAKER" in res, f"the lossless arm published no result at all: {res}"
    arm = res["GRID_MAKER"]
    assert arm["Sampled Signals (K)"] == 3, arm
    assert arm["observed_win_rate_pct"] == 100.0, (
        f"the fixture is no longer lossless, so the absence below would be vacuous: {arm}")
    assert arm["observed_expectancy_roe_pct"] > 0, arm
    assert arm["observed_profit_factor"] is None, (
        f"a lossless arm has no denominator, so there is no ratio to publish: {arm}")
    assert arm["observed_profit_factor"] != 0.0, (
        "a measured zero is a number nobody measured: this arm never lost a trade")
    assert arm["observed_profit_factor"] != 2.5


def test_the_bootstrap_arm_publishes_a_real_ratio_once_a_signal_loses(tmp_path, monkeypatch):
    """The control: the None above is the absence of a measurement, not a constant the producer
    always publishes. With a losing signal there is a denominator, and the ratio is published."""
    res, gaps = _bootstrap_arm_over_a_falling_frame(tmp_path, monkeypatch)
    assert "GRID_MAKER" in res, f"the losing arm published no result at all: {res}"
    arm = res["GRID_MAKER"]
    assert arm["observed_win_rate_pct"] == 0.0, (
        f"the fixture is no longer loss-making, so the control below would be vacuous: {arm}")
    assert arm["observed_profit_factor"] is not None, (
        "an arm that lost a trade has a denominator and must publish its ratio")
    assert arm["observed_profit_factor"] == 0.0, (
        f"no wins against a loss is a real zero ratio, and it is measured: {arm}")


# ---------------------------------------------------------------------------
# Final fix E-1 — the swing backtester's two ratios are measurements or absences
# ---------------------------------------------------------------------------
# What is proven below and what is not: `measure_trade_ratios` IS the producer's arithmetic (the
# backtester calls it), and `format_profit_factor` / `format_win_loss_ratio` are the renderers the
# audit matrix prints, so these tests call the shipped code and check every decision.
# `SwingArchitectBacktester.run()` itself is NOT executed -- it replays years of local history per
# scenario -- so the wiring between the producer and the two printers is pinned by source-text
# checks, and the matrix row itself is exercised by the E-2 tests below, once the script is
# importable at all.
#
# The defect being closed: the backtester published a hardcoded 99.0 as its profit factor whenever
# the run had no losing trade to divide by, and a 0.0 as its win/loss ratio. The comparative audit
# matrix printed them as measured numbers (99.00, 0.00x) in the same table as rows that had been
# measured.

SWING_BACKTESTER = REPO_ROOT / "backtests" / "backtest_swing_architect.py"
SWING_AUDIT_SCRIPT = REPO_ROOT / "scripts" / "run_swing_comparative_audit.py"

from backtests.backtest_swing_architect import (  # noqa: E402
    format_profit_factor as swing_format_profit_factor,
    format_win_loss_ratio as swing_format_win_loss_ratio,
    measure_trade_ratios,
)


def _closed_trades(rows):
    """A closed-trade frame carrying the columns the producer reads: net_pnl and gross_pnl."""
    return pd.DataFrame(list(rows), columns=["net_pnl", "gross_pnl"])


def test_a_lossless_swing_run_publishes_no_profit_factor():
    """A run whose trades never lost has no denominator: 99.0 was a constant dressed as a ratio."""
    trades = _closed_trades([(5.0, 5.0), (3.0, 3.0)])
    profit_factor, win_loss_ratio = measure_trade_ratios(trades)
    assert profit_factor is None, (
        f"a lossless run has no denominator, so there is no ratio to publish: {profit_factor}")
    assert profit_factor != 99.0
    assert win_loss_ratio is None, (
        f"a lossless run has no losing trade to average: {win_loss_ratio}")
    assert win_loss_ratio != 0.0


def test_a_swing_run_with_a_losing_trade_measures_its_ratios():
    """The control: with a denominator the ratios are measurements, not absences."""
    trades = _closed_trades([(5.0, 5.0), (-2.0, -2.0)])
    profit_factor, win_loss_ratio = measure_trade_ratios(trades)
    assert profit_factor == pytest.approx(2.5)
    assert win_loss_ratio == pytest.approx(2.5)


def test_a_swing_run_that_never_won_still_measures_a_zero_ratio():
    """A measured zero is a real number: the denominator was there and there was no profit in it."""
    trades = _closed_trades([(-2.0, -2.0), (-1.0, -1.0)])
    profit_factor, win_loss_ratio = measure_trade_ratios(trades)
    assert profit_factor == 0.0
    assert win_loss_ratio == 0.0


def test_an_empty_swing_run_publishes_no_ratio_at_all():
    profit_factor, win_loss_ratio = measure_trade_ratios(
        pd.DataFrame(columns=["net_pnl", "gross_pnl"]))
    assert profit_factor is None
    assert win_loss_ratio is None


def test_the_swing_renderers_name_the_absence_and_its_cause():
    """The absence has to say WHICH absence: no denominator, or nothing measured at all."""
    assert swing_format_profit_factor(1.23, 80, 20) == "1.23"
    assert swing_format_profit_factor(None, 0, 0).startswith("n/a")
    assert "no closed trades" in swing_format_profit_factor(None, 0, 0)
    assert "no losing" in swing_format_profit_factor(None, 8, 0)
    assert "denominator" in swing_format_profit_factor(None, 8, 0)
    assert "99" not in swing_format_profit_factor(None, 8, 0)
    assert "no closed trades" in swing_format_profit_factor(None, 0, 0)

    assert swing_format_win_loss_ratio(1.5, 80, 20) == "1.50x"
    assert swing_format_win_loss_ratio(None, 0, 0).startswith("n/a")
    assert "no closed trades" in swing_format_win_loss_ratio(None, 0, 0)
    assert "no losing" in swing_format_win_loss_ratio(None, 8, 0)
    assert "0.00x" not in swing_format_win_loss_ratio(None, 8, 0)


def test_the_swing_renderers_never_stand_in_for_a_missing_measurement():
    """No absence may carry a number that could be read as a measurement."""
    for n_trades, loss_count in [(0, 0), (8, 0), (8, 3)]:
        for out in (swing_format_profit_factor(None, n_trades, loss_count),
                    swing_format_win_loss_ratio(None, n_trades, loss_count)):
            assert "99" not in out, out
            assert "0.00" not in out, out
            assert out.startswith("n/a"), out


def test_the_swing_backtester_no_longer_publishes_a_constant_ratio():
    """Source-text check of the producer's own arithmetic and of the two printers that consume it."""
    src = SWING_BACKTESTER.read_text(encoding="utf-8")
    assert "gross_loss > 0 else 99.0" not in src, (
        "the profit factor must be the absence of a ratio, not a hardcoded constant")
    assert "avg_loss > 0 else 0.0" not in src, (
        "the win/loss ratio must be the absence of a ratio, not a measured zero")
    assert '"profit_factor": None if profit_factor is None else round(profit_factor, 2)' in src
    assert '"win_loss_ratio": None if win_loss_ratio is None else round(win_loss_ratio, 2)' in src
    # The backtester's own printer is a consumer too, and it must render the absence.
    assert "format_profit_factor(res.get('profit_factor'), n_trades, loss_count)" in src
    assert "format_win_loss_ratio(res.get('win_loss_ratio'), n_trades, loss_count)" in src
    assert "{res['profit_factor']:.2f}" not in src
    assert "{res['win_loss_ratio']:.2f}x" not in src


def test_the_audit_matrix_routes_through_the_swing_renderers():
    """Source-text check of the matrix printer — see the section note above."""
    src = SWING_AUDIT_SCRIPT.read_text(encoding="utf-8")
    assert 'format_profit_factor(res.get("profit_factor"), n_trades, loss_count)' in src
    assert 'format_win_loss_ratio(res.get("win_loss_ratio"), n_trades, loss_count)' in src
    assert '{res[\'profit_factor\']:.2f}' not in src
    assert '{res[\'win_loss_ratio\']:.2f}x' not in src


# ---------------------------------------------------------------------------
# Final fix E-2 — the comparative audit script can be run the way it looks runnable
# ---------------------------------------------------------------------------
# `python scripts/run_swing_comparative_audit.py` puts <repo>/scripts on sys.path[0] and NOT the
# repo root, so `backtests/` (a namespace package: it has no __init__.py) is not importable and
# nothing put the root there. The script's bare `from backtest_swing_architect import ...` therefore
# failed under the invocation the script is written for. This is the same failure part D fixed in
# scripts/lesson_distillation_pipeline.py (2ff8045), with the same spelling.

def test_the_audit_script_puts_the_repo_root_on_sys_path_the_way_the_repo_does():
    src = SWING_AUDIT_SCRIPT.read_text(encoding="utf-8")
    assert ("sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))"
            in src), (
        "the repo root has to reach sys.path the way the repo's other hand-run scripts spell it "
        "(scripts/quant_validation.py, scripts/run_grid_ablation.py, "
        "scripts/reconcile_d4_corrupt_exits.py)")
    assert "from backtests.backtest_swing_architect import" in src
    assert "from backtest_swing_architect import" not in src, (
        "the backtester lives in backtests/, so the import has to name it")
    assert "python scripts/run_swing_comparative_audit.py" in src, (
        "the script's own docstring has to document the invocation it can actually run")


def test_the_audit_script_import_resolves_under_its_documented_invocation():
    """The import is driven for real, in a clean interpreter, under the documented invocation.

    The probe rebuilds the sys.path that `python scripts/run_swing_comparative_audit.py` produces
    (sys.path[0] = <repo>/scripts, the repo root NOT on the path) and then executes the script's
    module header -- the docstring, the path setup and the imports -- and nothing else. No backtest
    runs and nothing is written.
    """
    import subprocess
    import sys

    probe = (
        "import ast, os, sys\n"
        "repo = os.path.abspath(sys.argv[1])\n"
        "script = os.path.join(repo, 'scripts', 'run_swing_comparative_audit.py')\n"
        "kept = [p for p in sys.path if p and os.path.abspath(p) != repo]\n"
        "sys.path = [os.path.join(repo, 'scripts')] + kept\n"
        "tree = ast.parse(open(script, encoding='utf-8').read(), filename=script)\n"
        "nodes = []\n"
        "for node in tree.body:\n"
        "    if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'scenarios' for t in node.targets):\n"
        "        break\n"
        "    nodes.append(node)\n"
        "ns = {'__file__': script, '__name__': 'e2_probe'}\n"
        "exec(compile(ast.Module(body=nodes, type_ignores=[]), script, 'exec'), ns)\n"
        "print(ns['SwingArchitectBacktester'].__module__)\n"
    )
    proc = subprocess.run([sys.executable, "-c", probe, str(REPO_ROOT)],
                          capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert proc.returncode == 0, (
        f"the documented hand-run cannot import the backtester:\n{proc.stdout}\n{proc.stderr}")
    assert proc.stdout.strip() == "backtests.backtest_swing_architect", proc.stdout


def test_the_audit_matrix_renders_the_absence_instead_of_the_fabricated_ratio():
    """The consumer, driven for real: a run with no losing trade has no ratio, and the matrix says
    so instead of printing 99.00 next to rows that had been measured."""
    from scripts.run_swing_comparative_audit import arm_row

    lossless = {
        "total_trades": 12, "win_count": 12, "loss_count": 0,
        "win_rate": 100.0, "profit_factor": None, "win_loss_ratio": None,
        "total_return_pct": 4.2, "net_profit": 5.1, "max_drawdown_pct": 1.1,
        "sharpe_ratio": 1.4, "sortino_ratio": 2.0, "total_fees_paid": 1.2,
    }
    row = arm_row("1. arm", lossless)
    assert row["Profit Factor"] == "n/a (no losing trade, profit factor has no denominator)"
    assert row["Win/Loss"] == "n/a (no losing trade, win/loss ratio has no denominator)"
    assert "99" not in row["Profit Factor"]
    assert "0.00x" not in row["Win/Loss"]
    # The cells that were measured are still published as numbers.
    assert row["Win Rate"] == "100.0%"
    assert row["Return %"] == "+4.2%"
    assert row["Trades"] == 12


def test_the_audit_matrix_says_when_a_run_measured_nothing():
    """Two different absences, told apart: a run that closed no trade, and a run with no history."""
    from scripts.run_swing_comparative_audit import arm_row

    no_trades = {
        "total_trades": 0, "win_count": 0, "loss_count": 0,
        "win_rate": None, "profit_factor": None, "win_loss_ratio": None,
        "total_return_pct": 0.0, "net_profit": 0.0, "max_drawdown_pct": 0.0,
        "sharpe_ratio": None, "sortino_ratio": None, "total_fees_paid": 0.0,
    }
    row = arm_row("2. arm", no_trades)
    assert "no closed trades" in row["Win Rate"]
    assert "no closed trades" in row["Profit Factor"]
    assert "no closed trades" in row["Win/Loss"]
    assert "0.0%" not in row["Win Rate"], (
        f"a win rate with no closed trades is not a measured zero: {row['Win Rate']}")
    # The sums and the counts of an empty population are honest zeros; the rate is not.
    assert row["Trades"] == 0
    assert row["Return %"] == "+0.0%"

    no_data = arm_row("3. arm", {"error": "No historical data found"})
    assert "no historical data" in no_data["Profit Factor"]
    assert "no historical data" in no_data["Win Rate"]


# ---------------------------------------------------------------------------
# Final fix E-3 — the dashboard's win rate is a measurement or an absence
# ---------------------------------------------------------------------------
# What is proven below and what is not: `_win_rate_value` is a pure function that turns the KPI
# card's win rate (a rate, or None when there is no population to divide by) into the text the card
# shows, so these tests call the shipped code and check every decision. The `st.metric(...)` call in
# dashboard.py's KPI row is NOT executed by any of them -- the committed suite has no Streamlit
# AppTest harness, and importing dashboard.py runs a 4,500-line top-level script against the real
# DB and live price endpoints. The helper's decision is proven; the rendering one line further out
# is a source-text check, exactly as part D's profit-factor helper was.

_win_rate_value = _dashboard_helper("_win_rate_value")


def test_the_card_publishes_no_measured_zero_win_rate():
    """The card used to read `0.0% Win Rate` whenever nothing had been closed."""
    out = _win_rate_value(None, 0)
    assert "not measured" in out.lower(), out
    assert "no closed trades" in out, out
    assert "0.0%" not in out, out
    assert not any(ch.isdigit() for ch in out), (
        f"an unmeasured win rate must print no number at all, not a stand-in: {out}")


def test_a_measured_zero_win_rate_is_still_published_as_a_number():
    """A fleet that closed trades and won none of them has a real 0.0% win rate."""
    assert _win_rate_value(0.0, 7) == "0.0% Win Rate"
    assert _win_rate_value(43.3, 60) == "43.3% Win Rate"


def test_the_win_rate_helper_never_raises_on_an_unreadable_population():
    """The card around it must survive a population it cannot read."""
    assert "not measured" in _win_rate_value(None, None).lower()
    assert "not measured" in _win_rate_value(None, "not-a-number").lower()
    assert _win_rate_value("43.3", 60) == "43.3% Win Rate"


def test_the_panel_publishes_no_win_rate_for_an_empty_ledger():
    """The else branch published 0.0 for the empty ledger; the panel now publishes None."""
    src = DASHBOARD.read_text(encoding="utf-8")
    assert ("win_rate_pct = (win_count / total_closed_count) * 100.0 if total_closed_count > 0 "
            "else None") in src
    assert ("win_count, loss_count, win_rate_pct, profit_factor, net_realized_pnl = "
            "0, 0, None, None, 0.0") in src


def test_the_kpi_card_routes_the_win_rate_through_the_helper():
    """Source-text check of the one line the helper tests cannot reach — see the section note above.

    It proves the panel hands the card's value to the helper. It does not prove Streamlit renders
    it.
    """
    src = DASHBOARD.read_text(encoding="utf-8")
    wired = [ln for ln in src.splitlines()
             if "value=_win_rate_value(win_rate_pct, total_closed_count)" in ln]
    assert len(wired) == 1, wired
    assert wired[0].strip().startswith("value="), (
        f"the helper's text must be the metric's value argument: {wired[0]!r}")
    assert 'value=f"{win_rate_pct:.1f}% Win Rate"' not in src


# ---------------------------------------------------------------------------
# Final fix F-1 — the calibration panel publishes a measurement or an absence
# ---------------------------------------------------------------------------
# What is proven below and what is not: `_calibration_panel_values` is a pure function that turns
# the tracker's metrics dict into the four strings the panel shows, so these tests call the shipped
# code and check every decision. The `st.metric(...)` calls in dashboard.py's calibration panel are
# NOT executed by any of them -- the committed suite has no Streamlit AppTest harness, and importing
# dashboard.py runs a 4,500-line top-level script against the real DB and live price endpoints. A
# helper that returns a string cannot lie about what it decided; only the rendering one line further
# out is unverified, and test_the_calibration_panel_routes_through_the_helper below is a source-text
# check of that line, nothing more.
#
# The defect being closed: the panel published `0%`, `0 Trades`, `0.0000` and a `GOOD` grade
# whenever the tracker had measured nothing, and a panel with no data at all graded itself GOOD.

CALIBRATION_TRACKER = REPO_ROOT / "model_calibration_tracker.py"

_calibration_panel_values = _dashboard_helper("_calibration_panel_values")


def _tracker_over(tmp_path):
    """A calibration tracker whose two databases are temp files, so no repo DB is ever opened.

    The tracker's constructor CREATEs its log table in whatever ledger it is handed, so pointing it
    at the repo's trading_ledger.db would be a write to a live database. Both paths below live
    inside tmp_path, and the temp `trades` table is empty so the bootstrap step inserts nothing.
    """
    import sqlite3

    from model_calibration_tracker import ModelCalibrationTracker

    ledger = tmp_path / "ledger.db"
    data = tmp_path / "data.db"
    with sqlite3.connect(data) as conn:
        conn.execute(
            "CREATE TABLE trades (id INTEGER, timestamp TEXT, symbol TEXT, side TEXT, "
            "pnl REAL, quality REAL, ai_rationale TEXT)")
        conn.commit()
    tracker = ModelCalibrationTracker(db_ledger=str(ledger), db_data=str(data))
    return tracker, ledger, data


def test_the_tracker_publishes_no_rate_and_no_grade_for_an_empty_calibration_log(tmp_path):
    """The producer's own shape for an empty log, driven for real.

    This is the fact the panel has to render: the sample COUNT is there (a measured zero) and the
    rate and the grade are not there at all -- there is no key for them. The panel's old
    `.get('overall_win_rate', 0)` and `.get('calibration_grade', 'GOOD')` defaults invented both.
    """
    tracker, _, _ = _tracker_over(tmp_path)
    metrics = tracker.compute_calibration_metrics()
    assert metrics["total_samples"] == 0, metrics
    assert "overall_win_rate" not in metrics, (
        f"an empty log published a win rate nobody measured: {metrics}")
    assert "calibration_grade" not in metrics, (
        f"an empty log published a grade nobody earned: {metrics}")
    # The tracker's brier_score of 0.0 stands in for a score with no samples to average over.
    assert metrics["brier_score"] == 0.0, metrics


def test_the_tracker_measures_a_rate_once_it_has_samples(tmp_path):
    """The control: the absences above are the absence of a measurement, not constants the
    producer always publishes. With samples logged there is a rate, a score and a grade."""
    import sqlite3

    tracker, ledger, _ = _tracker_over(tmp_path)
    with sqlite3.connect(ledger) as conn:
        conn.executemany(
            "INSERT INTO model_calibration_log (predicted_confidence, is_win, actual_pnl) "
            "VALUES (?, ?, ?)",
            [(0.8, 1, 1.0), (0.4, 0, -1.0)],
        )
        conn.commit()
    metrics = tracker.compute_calibration_metrics()
    assert metrics["total_samples"] == 2, metrics
    assert metrics["overall_win_rate"] == 50.0, metrics
    assert metrics["brier_score"] == 0.1, metrics
    assert metrics["calibration_grade"] == "EXCELLENT", metrics


def test_the_calibration_panel_publishes_the_sample_count_it_measured():
    """A count of zero is a measurement: nothing was evaluated, and that is the fact.

    The rate, the score and the grade are different -- they need a population to be measured over,
    so with none they are absences.
    """
    values = _calibration_panel_values({"total_samples": 0, "brier_score": 0.0, "bins": []})
    assert values["signals_evaluated"] == "0 Trades", values
    assert values["historical_win_rate"].startswith("n/a"), values
    assert values["brier_score"].startswith("n/a"), values
    assert values["calibration_grade"].startswith("not measured"), values


def test_the_calibration_panel_publishes_no_fabricated_grade_for_an_empty_log():
    """The grade used to default to GOOD, so a panel with no data at all graded itself GOOD."""
    values = _calibration_panel_values({"total_samples": 0, "brier_score": 0.0, "bins": []})
    assert values["calibration_grade"] != "GOOD", values
    assert "GOOD" not in values["calibration_grade"], values
    assert "not measured" in values["calibration_grade"], values
    assert "no signals evaluated yet" in values["calibration_grade"], values


def test_the_calibration_panel_publishes_no_measured_zero_for_an_empty_log():
    """No cell may carry a number that could be read as a measurement."""
    values = _calibration_panel_values({"total_samples": 0, "brier_score": 0.0, "bins": []})
    for key in ("historical_win_rate", "brier_score", "calibration_grade"):
        assert not any(ch.isdigit() for ch in values[key]), (
            f"{key} carries a digit where nothing was measured: {values[key]!r}")
    assert "0%" not in values["historical_win_rate"], values
    assert "0.0000" not in values["brier_score"], values


def test_the_calibration_panel_publishes_an_absence_for_an_unreadable_log():
    """The tracker's other shape: `{}` when the calibration log could not be read.

    There is not even a count here, so nothing is published as a number -- not even a zero.
    """
    values = _calibration_panel_values({})
    for key, text in values.items():
        assert not any(ch.isdigit() for ch in text), (
            f"{key} carries a digit where nothing could be read: {text!r}")
    assert "unavailable" in values["signals_evaluated"], values
    assert "unavailable" in values["historical_win_rate"], values
    assert "unavailable" in values["brier_score"], values
    assert "unavailable" in values["calibration_grade"], values


def test_the_calibration_panel_survives_a_malformed_metrics_dict():
    """The panel must not take down the tab over a shape it did not expect."""
    for bad in (None, 5, "metrics", [], {"total_samples": "many"}, {"brier_score": "n/a"}):
        values = _calibration_panel_values(bad)
        assert set(values) == {"signals_evaluated", "historical_win_rate", "brier_score",
                               "calibration_grade"}, values


def test_a_measured_calibration_is_published_as_numbers():
    """The control: with samples, the panel publishes the measurement it has."""
    values = _calibration_panel_values({
        "total_samples": 2, "overall_win_rate": 50.0, "brier_score": 0.1,
        "calibration_grade": "EXCELLENT", "bins": [],
    })
    assert values["signals_evaluated"] == "2 Trades", values
    assert values["historical_win_rate"] == "50.0%", values
    assert values["brier_score"] == "0.1000", values
    assert values["calibration_grade"] == "EXCELLENT", values


def test_the_calibration_panel_routes_through_the_helper():
    """Source-text check of the one line the helper tests cannot reach — see the section note.

    It proves the panel publishes the helper's strings and no longer defaults the three values it
    used to invent. It does not prove Streamlit renders them.
    """
    src = DASHBOARD.read_text(encoding="utf-8")
    assert "cal_values = _calibration_panel_values(c_metrics)" in src, (
        "the panel must publish the helper's strings, not read the metrics dict itself")
    for gone in ("c_metrics.get('total_samples', 0)", "c_metrics.get('overall_win_rate', 0)",
                 "c_metrics.get('brier_score', 0)", "c_metrics.get('calibration_grade', 'GOOD')"):
        assert gone not in src, (
            f"the panel still defaults {gone}: a key the tracker never produced and a measured "
            "zero look the same on screen")


def _calibration_cli_report(metrics):
    """Runs model_calibration_tracker.py's own __main__ report block against a stubbed tracker.

    The block is compiled out of the module and executed with a stub in place of
    ModelCalibrationTracker, so neither trading_ledger.db nor trading_data.db is ever opened: the
    real constructor CREATEs a table in the ledger, and this dispatch may not write to it. What is
    exercised is the block's own rendering -- the shipped body, not a paraphrase of it.
    """
    import ast
    import contextlib
    import io

    src = CALIBRATION_TRACKER.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(CALIBRATION_TRACKER))
    main_block = next(
        (n for n in tree.body
         if isinstance(n, ast.If) and "__main__" in ast.dump(n.test)),
        None,
    )
    assert main_block is not None, "model_calibration_tracker.py has no __main__ report block"

    class _StubTracker:
        def compute_calibration_metrics(self):
            return metrics

    ns = {"__name__": "__main__", "ModelCalibrationTracker": _StubTracker}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(ast.Module(body=[main_block], type_ignores=[]),
                     str(CALIBRATION_TRACKER), "exec"), ns)
    return buf.getvalue()


def test_the_tracker_report_publishes_absences_for_an_empty_log():
    """The CLI report is the same defect class as the panel: it printed 0% and a 0.0 Brier score
    (and a N/A grade) for a log that held nothing."""
    out = _calibration_cli_report({"total_samples": 0, "brier_score": 0.0, "bins": []})
    assert "Total Evaluated Signals: 0" in out, out
    assert "Overall Historical Win Rate: n/a (no signals evaluated yet)" in out, out
    assert "Grade: not measured" in out, out
    assert "0%" not in out, out
    assert "GOOD" not in out, out


def test_the_tracker_report_publishes_absences_for_an_unreadable_log():
    out = _calibration_cli_report({})
    assert "Total Evaluated Signals: n/a" in out, out
    assert "Overall Historical Win Rate: n/a (calibration metrics unavailable)" in out, out
    assert "Brier Calibration Score: n/a (calibration metrics unavailable)" in out, out


def test_the_tracker_report_publishes_the_measurement_once_it_has_samples():
    out = _calibration_cli_report({
        "total_samples": 2, "overall_win_rate": 50.0, "brier_score": 0.1,
        "calibration_grade": "EXCELLENT", "bins": [],
    })
    assert "Total Evaluated Signals: 2" in out, out
    assert "Overall Historical Win Rate: 50.0%" in out, out
    assert "Brier Calibration Score: 0.1 (Grade: EXCELLENT)" in out, out


def test_the_tracker_report_no_longer_defaults_the_values_it_never_measured():
    """Source-text check of the shipped CLI block's defaults."""
    src = CALIBRATION_TRACKER.read_text(encoding="utf-8")
    for gone in ("metrics.get('overall_win_rate', 0)", "metrics.get('brier_score', 0)",
                 "metrics.get('calibration_grade', 'N/A')"):
        assert gone not in src, (
            f"the report still defaults {gone}: a measurement nobody took printed as a number")


# ---------------------------------------------------------------------------
# Final fix F-2 — the legacy live dashboard publishes no demo ledger
# ---------------------------------------------------------------------------
# scripts/live_dashboard.py is a standalone Streamlit script (it cannot import dashboard.py), so it
# carries its own copies of the same pure renderers. They are extracted with the same helper, so
# the shipped bodies are executed here too.
#
# The defect being closed: the script's no-data branch published a hardcoded demo ledger (43 wins,
# 16 losses, 72.9%, 2.62x, $12.18) as if it were telemetry, and the same file carried a demo
# account balance, four demo holding-time cards and three demo counts.

LEGACY_DASHBOARD = REPO_ROOT / "scripts" / "live_dashboard.py"

_legacy_win_rate_value = _dashboard_helper("_win_rate_value", LEGACY_DASHBOARD)
_legacy_profit_factor_delta = _dashboard_helper("_profit_factor_delta", LEGACY_DASHBOARD)
_legacy_calibration_panel_values = _dashboard_helper("_calibration_panel_values", LEGACY_DASHBOARD)


def _numeric_literals(path):
    """Every numeric constant in a module, so a demo figure cannot hide in a computation."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
            and not isinstance(n.value, bool)]


def _published_strings(path):
    """Every string constant in a module except its docstrings.

    A docstring is prose about the module; a string constant is text that can reach a reader. The
    two have to be told apart, because the fixes below explain themselves in docstrings that quote
    the very figures they removed.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and body:
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                docstrings.add(id(first.value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docstrings]


def test_the_legacy_no_data_branch_publishes_no_demo_ledger():
    """The branch that used to publish 43 wins, 16 losses, 72.9%, 2.62x and $12.18."""
    src = LEGACY_DASHBOARD.read_text(encoding="utf-8")
    assert ("win_count, loss_count, win_rate_pct, profit_factor, net_realized_pnl = "
            "0, 0, None, None, None") in src, (
        "the no-data branch must publish zero trades and no rate, no ratio and no total")
    assert "profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None" in src, (
        "no losing trade means no denominator, so the ratio must be the absence of a ratio")
    assert "else (2.62 if win_count > 0 else 1.0)" not in src, (
        "the profit factor still falls back to a constant nobody measured")


def test_the_legacy_file_publishes_no_demo_telemetry_figures():
    """An audit of the module's numeric constants: the demo ledger and the demo cards are gone."""
    numbers = set(_numeric_literals(LEGACY_DASHBOARD))
    for gone in (72.9, 2.62, 12.18, 155.50, 38.5, 46.2, 27.8, 1.66, 43, 16, 33, 220, 18):
        assert gone not in numbers, (
            f"the legacy dashboard still carries the demo figure {gone!r} as a literal")


def test_the_legacy_file_publishes_no_demo_telemetry_text():
    """And an audit of the text it can show: no demo figure survives in a published string."""
    published = _published_strings(LEGACY_DASHBOARD)
    for gone in ("72.9", "2.62", "12.18", "155.50", "38.5", "46.2", "27.8", "1.66",
                 "33 dynamic", "220 Lessons", "18 Background", "72.9%"):
        hits = [s for s in published if gone in s]
        assert not hits, (
            f"the legacy dashboard still publishes {gone!r}: {hits[:3]}")


def test_the_legacy_win_rate_card_publishes_an_absence_for_an_empty_ledger():
    """The card's value used to read 0.0% (or 72.9%) whether or not anything was measured."""
    out = _legacy_win_rate_value(None, 0)
    assert "not measured" in out.lower(), out
    assert "no closed trades" in out, out
    assert not any(ch.isdigit() for ch in out), (
        f"an unmeasured win rate must print no number at all: {out}")


def test_the_legacy_win_rate_card_publishes_a_measured_rate():
    out = _legacy_win_rate_value(72.9, 59)
    assert out == "72.9% Win Rate", out
    assert _legacy_win_rate_value(0.0, 7) == "0.0% Win Rate", (
        "a fleet that closed trades and won none of them has a real zero rate")


def test_the_legacy_profit_factor_card_publishes_no_constant():
    """The card's delta used to read 2.62x (or 1.00x) for a fleet with no losing trade."""
    for win_count, loss_count in [(0, 0), (43, 0), (0, 16), (43, 16)]:
        out = _legacy_profit_factor_delta(None, win_count, loss_count)
        assert "2.62" not in out, out
        assert "1.00" not in out, out
        assert "not measured" in out.lower(), out
        assert not any(ch.isdigit() for ch in out), (
            f"an unmeasured profit factor must print no number at all: {out}")


def test_the_legacy_profit_factor_card_publishes_a_measured_ratio():
    out = _legacy_profit_factor_delta(2.62, 43, 16)
    assert out == "2.62x Profit Factor (43W / 16L)", out


def test_the_legacy_kpi_card_routes_through_the_helpers():
    """Source-text check of the two lines the helper tests cannot reach.

    It proves the card hands the ledger's numbers to the renderers. It does not prove Streamlit
    renders them.
    """
    src = LEGACY_DASHBOARD.read_text(encoding="utf-8")
    assert "value=_win_rate_value(win_rate_pct, total_closed_count)" in src
    assert "delta=_profit_factor_delta(profit_factor, win_count, loss_count)" in src
    assert 'value=f"{win_rate_pct:.1f}% Win Rate"' not in src
    assert 'delta=f"{profit_factor:.2f}x Profit Factor ({win_count}W / {loss_count}L)"' not in src


def test_the_legacy_calibration_panel_publishes_the_same_absence_as_the_main_dashboard():
    """The two panels are the same panel: the legacy script must publish the same strings."""
    for metrics in ({}, {"total_samples": 0, "brier_score": 0.0, "bins": []},
                    {"total_samples": 2, "overall_win_rate": 50.0, "brier_score": 0.1,
                     "calibration_grade": "EXCELLENT", "bins": []}):
        assert _legacy_calibration_panel_values(metrics) == _calibration_panel_values(metrics), (
            f"the two dashboards disagree about the same calibration metrics: {metrics}")


def test_the_legacy_calibration_panel_publishes_no_fabricated_grade():
    values = _legacy_calibration_panel_values({"total_samples": 0, "brier_score": 0.0, "bins": []})
    assert values["calibration_grade"] != "GOOD", values
    assert "not measured" in values["calibration_grade"], values
    assert values["historical_win_rate"].startswith("n/a"), values
    assert values["brier_score"].startswith("n/a"), values


def test_the_legacy_calibration_panel_routes_through_the_helper():
    src = LEGACY_DASHBOARD.read_text(encoding="utf-8")
    assert "cal_values = _calibration_panel_values(c_metrics)" in src
    for gone in ("c_metrics.get('total_samples', 0)", "c_metrics.get('overall_win_rate', 0)",
                 "c_metrics.get('brier_score', 0)", "c_metrics.get('calibration_grade', 'GOOD')"):
        assert gone not in src, f"the legacy panel still defaults {gone}"


def test_the_legacy_account_balance_is_not_a_demo_balance():
    """The sidebar used to publish $155.50 for an account it had never read."""
    src = LEGACY_DASHBOARD.read_text(encoding="utf-8")
    assert 'query_system_state("balance", "")' in src, (
        "the balance must be read without a demo fallback")
    # G-2 moved this absence sentence into the shared `_profit_display` helper: the legacy app
    # now publishes "not measured (no recorded capital or no balance telemetry)" from there
    # instead of from a literal of its own. The requirement is unchanged -- an unreadable
    # balance is an absence, not a number.
    assert "_profit_display(total_profit, total_roi_pct)" in src, (
        "an unreadable balance must be published as an absence")
    assert "Telemetry Offline" in src


def test_the_legacy_margin_block_publishes_no_ratio_without_a_balance():
    """Margin utilisation and the liquid buffer are ratios against the balance: no balance, no
    ratio."""
    src = LEGACY_DASHBOARD.read_text(encoding="utf-8")
    # G-5c moved this gate into the twin helper `_deck_account_balance` (an unreadable or zero
    # balance is an absence) and routed the ratio itself through `_deck_margin_gauge`, so a margin
    # total built on an assumed leverage is now a second way for the gauge to say nothing. The
    # requirement is unchanged -- no ratio without a balance -- and `margin_measured` survives as
    # the balance half of it.
    assert "_deck_balance = _deck_account_balance(live_bal_val)" in src
    assert "margin_measured = not _deck_is_absent(_deck_balance)" in src
    assert "_deck_margin_gauge(total_margin_locked" in src
    assert 'margin_util_display = _deck_text(margin_util_pct, "{:.1f}%")' in src
    assert '"n/a (no account balance)"' not in src, (
        "the gauge went back to its own absence sentence instead of the house one")


def test_the_legacy_holding_time_cards_publish_no_demo_durations():
    """Four cards used to publish 38.5 mins, 46.2 mins, 27.8 mins and a 1.66x hold ratio that no
    trade was ever measured against."""
    src = LEGACY_DASHBOARD.read_text(encoding="utf-8")
    for label in ("Avg Fleet Duration", "Avg Winner Duration", "Avg Loser Duration",
                  "Hold Ratio (Win/Loss)"):
        assert f'"{label}", "n/a (not measured)"' in src, (
            f"the {label} card must publish an absence, not a demo duration")
    assert "Not measured in this view" in src, (
        "the holding-asymmetry note must not claim an asymmetry nobody measured")


def test_the_legacy_rule_and_lesson_counts_are_measured_not_asserted():
    """The tab used to say '33 dynamic neural rules' and '220 Lessons' without counting."""
    src = LEGACY_DASHBOARD.read_text(encoding="utf-8")
    assert "SELECT COUNT(*) as cnt FROM learned_rules" in src
    assert "SELECT COUNT(*) as cnt FROM trade_lessons" in src
    assert "dynamic neural rules active in long-term memory" in src
    assert "Lessons in Database" in src


def test_the_legacy_daemon_count_is_not_a_fabricated_number():
    """The subsystem header used to claim 18 background processes over a list of 11 rows."""
    src = LEGACY_DASHBOARD.read_text(encoding="utf-8")
    assert "18 Background Processes" not in src
    assert "Autonomous Daemon Subsystems (Unified Super Bot Architecture)" in src


# ---------------------------------------------------------------------------
# Final fix G-1 — the regime gauges and the macro regime publish absences, not stand-in readings
# ---------------------------------------------------------------------------
# What is proven below and what is not: the four helpers under test (`_optional_float`,
# `_regime_readings`, `_macro_regime`, `_gauge_display`) are pure and module-level, so they are
# extracted by AST and executed for real here — the shipped body, not a paraphrase of it. The
# `st.metric(...)` calls they feed are NOT executed: the committed suite has no Streamlit AppTest
# harness, and importing dashboard.py runs a 4,700-line top-level script against the real DB and
# live price endpoints. A helper that returns a dict cannot lie about what it decided; the
# rendering one line further out is pinned by source-text checks, nothing more.
#
# The defect being closed: the regime deck fell back on 28.5 (ADX), 0.62 (Hurst), 0.55
# (volatility) and 0.40 (trend strength) whenever the regime daemon had published nothing, so a
# fleet with no regime row rendered a confident "Trend Strength (ADX) 28.5 / Strong Trend (>25)"
# deck. The DI pair fell back on 0.0 and was then classified as "Balanced" — a measured statement
# about two numbers nobody measured.
# And the macro banner defaulted `btc_macro_trend` to "BULLISH", so a fleet whose macro key had
# never been written announced "REGIME: MACRO BULLISH (LONG ONLY)" and published "Macro
# Direction: LONG ONLY" for the very same absence. The radar's mapping was worse still: it
# labelled every macro regime that was not "BULLISH" as "SHORT ONLY", so a *measured* NEUTRAL
# macro regime was dressed as a short-only trading stance.

_GAUGE_HELPERS = {}


def _gauge_helper(name, path=DASHBOARD):
    """One of the four regime/macro helpers, extracted once per (file, name) and cached.

    The extraction is deliberately lazy: a helper that does not exist yet must fail the test
    that asks for it, not the whole module at collection time.
    """
    key = (str(path), name)
    if key not in _GAUGE_HELPERS:
        _GAUGE_HELPERS[key] = _dashboard_helper(name, path)
    return _GAUGE_HELPERS[key]


def _gauge_block(path):
    """The source of the regime gauge block, so the stand-in readings can be looked for in it.

    The block runs from the "Quantitative Regime Deck Gauges" marker to the next section break,
    so a stand-in reading anywhere inside it is a stand-in reading the deck publishes.

    The terminators are indentation-tolerant: the legacy app's section rules are not all at
    column 0, and a column-anchored terminator would let the extraction run past the rule that
    closes the block and scan a much larger region of the file than the gauge deck.
    """
    import textwrap

    src = path.read_text(encoding="utf-8")
    match = re.search(r"^[ \t]*# Quantitative Regime Deck Gauges", src, re.M)
    assert match is not None, f"{path.name} has no regime gauge block"
    tail = src[match.start():]
    stop = len(tail)
    for pattern in (r'st\.markdown\(\s*"---"\s*\)', r"\n[ \t]*#\s*═"):
        match = re.search(pattern, tail)
        if match is not None:
            stop = min(stop, match.start())
    return textwrap.dedent(tail[:stop])


def _shipped_di_classifier():
    """The DI classifier dashboard.py actually ships, compiled out of its own gauge block.

    The shipped classifier is a lambda that closes over `plus_di` and `minus_di`, so it cannot
    be exec'd on its own; its source segment is, with the two readings bound in the namespace.
    That is the shipped body being called, not a paraphrase of it.
    """
    import ast

    block = _gauge_block(DASHBOARD)
    tree = ast.parse(block, filename=str(DASHBOARD))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_gauge_display" and len(node.args) >= 3):
            continue
        classify = node.args[2]
        if not isinstance(classify, ast.Lambda):
            continue
        names = {n.id for n in ast.walk(classify) if isinstance(n, ast.Name)}
        if {"plus_di", "minus_di"} <= names:
            segment = ast.get_source_segment(block, classify)
            namespace = {"plus_di": 0.0, "minus_di": 0.0}
            exec(f"classify = {segment}", namespace)
            return namespace["classify"]
    raise AssertionError(
        "dashboard.py's regime gauge block has no DI classifier closing over plus_di/minus_di")


def _numeric_constants(text, filename="<gauge block>"):
    """Every numeric constant in a source fragment."""
    import ast

    tree = ast.parse(text, filename=filename)
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
            and not isinstance(n.value, bool)]


@pytest.mark.parametrize("raw, expected", [
    ("55.00", 55.0),
    (55, 55.0),
    (55.5, 55.5),
    ("$1,000.00", 1000.0),
    ("  42  ", 42.0),
    (None, None),
    ("", None),
    ("   ", None),
    ("--", None),
    ("syncing...", None),
    ("none", None),
    ("null", None),
    ("nan", None),
    ("abc", None),
    (float("nan"), None),
    (True, None),
])
def test_optional_float_reads_a_number_and_names_every_absence(raw, expected):
    """A reading that was taken is a number; a reading that was not taken is None.

    None is what tells an absence apart from a measured zero, so it is the only value the
    gauge layer is allowed to see for "nothing was published here".
    """
    assert _gauge_helper("_optional_float")(raw) == expected


def test_regime_readings_are_all_absences_when_no_row_was_published():
    """The fleet whose regime daemon has published nothing has no gauge readings at all."""
    readings = _gauge_helper("_regime_readings")(None, None)
    assert set(readings) == {"adx", "hurst", "plus_di", "minus_di",
                              "volatility", "trend_strength"}, readings
    assert all(value is None for value in readings.values()), readings


def test_regime_readings_read_every_column_of_a_published_row():
    readings = _gauge_helper("_regime_readings")(
        {"adx": 37.5, "hurst": 0.61, "plus_di": 22.0, "minus_di": 14.0},
        {"volatility": 0.44, "trend_strength": 0.52},
    )
    assert readings == {
        "adx": 37.5, "hurst": 0.61, "plus_di": 22.0, "minus_di": 14.0,
        "volatility": 0.44, "trend_strength": 0.52,
    }, readings


def test_regime_readings_leave_a_missing_column_absent_and_read_the_rest():
    """A row that predates a column is not a row with a zero in it."""
    readings = _gauge_helper("_regime_readings")({"hurst": 0.61}, {"volatility": 0.44})
    assert readings["adx"] is None, readings
    assert readings["hurst"] == 0.61, readings
    assert readings["volatility"] == 0.44, readings
    assert readings["plus_di"] is None, readings


def test_regime_readings_treat_null_and_junk_as_absences():
    readings = _gauge_helper("_regime_readings")(
        {"adx": None, "hurst": "nan", "plus_di": ""}, {"volatility": "n/a"})
    assert readings["adx"] is None, readings
    assert readings["hurst"] is None, readings
    assert readings["plus_di"] is None, readings
    assert readings["volatility"] is None, readings


def test_regime_readings_accept_a_numeric_string_as_a_reading():
    """The daemon writes text into SQLite, so "37.5" is a reading and not an absence."""
    readings = _gauge_helper("_regime_readings")({"adx": "37.5"}, None)
    assert readings["adx"] == 37.5, readings


def test_a_measured_zero_di_is_a_reading_not_an_absence():
    """A measured zero is a fact. It must not collapse into "nothing was measured"."""
    readings = _gauge_helper("_regime_readings")({"plus_di": 0.0, "minus_di": 0.0}, None)
    assert readings["plus_di"] == 0.0, readings
    assert readings["minus_di"] == 0.0, readings
    assert readings["plus_di"] is not None, readings


def test_regime_readings_read_a_pandas_series_row():
    """The deck reads a DataFrame row, so a Series must read exactly like the equivalent dict."""
    readings = _gauge_helper("_regime_readings")(
        pd.Series({"adx": 37.5, "hurst": 0.61, "plus_di": 22.0, "minus_di": 14.0}),
        pd.Series({"volatility": 0.44, "trend_strength": 0.52}),
    )
    assert readings == {
        "adx": 37.5, "hurst": 0.61, "plus_di": 22.0, "minus_di": 14.0,
        "volatility": 0.44, "trend_strength": 0.52,
    }, readings


@pytest.mark.parametrize("raw, expected", [
    ("bullish", "BULLISH"),
    (" BEARISH ", "BEARISH"),
    ("NEUTRAL", "NEUTRAL"),
    ("UNKNOWN", "UNKNOWN"),
    ("", "UNKNOWN"),
    (None, "UNKNOWN"),
    ("chop", "UNKNOWN"),
    ("0", "UNKNOWN"),
])
def test_macro_regime_publishes_only_the_stances_the_fleet_publishes(raw, expected):
    """BULLISH / BEARISH / NEUTRAL are the fleet's own states; UNKNOWN is one of them too.

    Anything else — an absent key, an empty string, junk — means the same thing here: no
    stance. The banner must never announce a direction it did not measure.
    """
    assert _gauge_helper("_macro_regime")(raw) == expected


def test_an_absent_gauge_says_the_reading_was_never_taken():
    """The absence carries the reason, with no arrow and no colour.

    An arrow would say "this reading moved", and a colour would say "this reading went up".
    Neither is true of a reading that was never taken.
    """
    out = _gauge_helper("_gauge_display")(
        None, lambda v: f"{v:.1f}", lambda v: "Strong Trend (>25)", "no market_regime row")
    assert out == {"value": "not measured", "delta": "no market_regime row",
                   "arrow": "off", "color": "off"}, out


def test_a_measured_gauge_publishes_the_reading_and_its_classification():
    out = _gauge_helper("_gauge_display")(
        37.5, lambda v: f"{v:.1f}", lambda v: "Strong Trend (>25)", "no market_regime row")
    assert out["value"] == "37.5", out
    assert out["delta"] == "Strong Trend (>25)", out
    assert out["arrow"] == "auto", out
    assert out["color"] == "normal", out


def test_a_gauge_survives_a_formatter_that_raises():
    def _boom(_value):
        raise RuntimeError("formatter exploded")

    out = _gauge_helper("_gauge_display")(37.5, _boom, lambda v: "Strong Trend (>25)", "no row")
    assert out["value"] == "not measured", out


def test_a_gauge_survives_a_classifier_that_raises():
    def _boom(_value):
        raise RuntimeError("classifier exploded")

    out = _gauge_helper("_gauge_display")(37.5, lambda v: f"{v:.1f}", _boom, "no row")
    assert out["delta"] == "measured", out
    assert out["value"] == "37.5", out


def test_the_two_dashboards_publish_the_same_regime_and_macro_readings():
    """scripts/live_dashboard.py is a standalone Streamlit script and cannot import dashboard.py,
    so it carries its own copies of the four helpers. The copies have to agree, or the same
    fleet would be shown two different regimes."""
    cases = [
        ("_optional_float", ("55.00",), {}),
        ("_optional_float", (None,), {}),
        ("_optional_float", ("--",), {}),
        ("_regime_readings", ({"adx": 37.5, "hurst": 0.61}, {"volatility": 0.44}), {}),
        ("_regime_readings", (None, None), {}),
        ("_macro_regime", ("bullish",), {}),
        ("_macro_regime", (None,), {}),
        ("_gauge_display", (None, lambda v: f"{v:.1f}", lambda v: "Strong", "no market_regime row"), {}),
        ("_gauge_display", (37.5, lambda v: f"{v:.1f}", lambda v: "Strong", "no market_regime row"), {}),
    ]
    for name, args, _kwargs in cases:
        main = _gauge_helper(name, DASHBOARD)
        legacy = _gauge_helper(name, LEGACY_DASHBOARD)
        assert main(*args) == legacy(*args), (
            f"the two dashboards disagree about {name}{args}: {main(*args)} != {legacy(*args)}")


def test_the_two_dashboards_carry_byte_identical_copies_of_the_four_helpers():
    """The copies are duplicated on purpose (the legacy script cannot import dashboard.py), so
    they are pinned to be the same source, not merely to behave the same today."""
    import ast

    for name in ("_optional_float", "_regime_readings", "_macro_regime", "_gauge_display"):
        sources = []
        for path in (DASHBOARD, LEGACY_DASHBOARD):
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(path))
            node = next(n for n in tree.body
                        if isinstance(n, ast.FunctionDef) and n.name == name)
            sources.append(ast.get_source_segment(text, node))
        assert sources[0] == sources[1], (
            f"{name} differs between the two dashboards, so the same fleet would be shown two "
            f"different readings")


def test_the_dashboard_files_publish_no_stand_in_regime_readings():
    """The gauges used to fall back on 28.5 (ADX), 0.62 (Hurst), 0.55 (volatility) and 0.40
    (trend strength) when the regime daemon had published nothing at all."""
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        block = _gauge_block(path)
        numbers = _numeric_constants(block, filename=str(path))
        for gone in (28.5, 0.62, 0.55, 0.40):
            assert gone not in numbers, (
                f"{path.name} still falls back on the stand-in reading {gone!r} in its regime "
                f"gauge block")
        # The defect shape is the `or` fallback, not the value 0.0: a measured zero and an
        # absent reading are told apart by None, so a `or 0.0` fallback invents a reading.
        # The forbidden list above cannot catch that, because 0.0 == 0 in Python and the
        # shipped comparison `(plus_di > 0 or minus_di > 0)` carries the integer literal 0.
        assert not re.search(r"\bor\s+0(?:\.0)?\b", block), (
            f"{path.name} still falls back on a zero stand-in in its regime gauge block")
        published = _published_strings(path)
        for gone in ("28.5", "0.62", "0.55"):
            # "0.40" is deliberately not in this list: dashboard.py publishes "0.40" as the
            # `ai_regime_trap_risk` default of an unrelated panel (the AI Cognitive Regime
            # Arbiter), which this dispatch does not touch. The gauge-block check above is
            # what pins the regime deck itself.
            assert gone not in published, (
                f"{path.name} still publishes the stand-in reading {gone!r}")


def test_neither_dashboard_publishes_a_macro_default_of_bullish():
    """The banner used to read REGIME: MACRO BULLISH (LONG ONLY) for a fleet whose macro key
    had never been written.

    "BULLISH" has to survive as a stance the fleet really publishes, so the guard is on the
    DEFAULT and not on the word: the key may only be read with an empty default, and the
    normaliser decides what the fleet's own value means.
    """
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        src = path.read_text(encoding="utf-8")
        assert '"btc_macro_trend", "BULLISH"' not in src, (
            f"{path.name} still defaults the macro key to BULLISH, which invents a direction out "
            "of a reading nobody took")
        defaults = re.findall(
            r'query_system_state\(\s*"btc_macro_trend"\s*,\s*"([^"]*)"\s*\)', src)
        assert defaults, (
            f"{path.name} does not read the macro key through query_system_state at all, so the "
            "default guard below cannot see it")
        assert all(default == "" for default in defaults), (
            f"{path.name} reads the macro key with the default(s) {defaults}; only an empty "
            "default may reach the normaliser")


def test_both_dashboards_publish_no_stance_for_an_unknown_macro_direction():
    """The radar table turned every macro regime that was not "BULLISH" into SHORT ONLY.

    An absent key was not the case it got wrong — the key defaulted to "BULLISH", so an absence
    published LONG ONLY. A *measured* NEUTRAL macro regime published SHORT ONLY, which is the
    worse defect: a reading that was taken was relabelled as a directional trading stance.
    """
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        src = path.read_text(encoding="utf-8")
        assert '"NO STANCE"' in src, (
            f"{path.name} must publish NO STANCE for a macro regime it has no reading for")
        assert "NO STANCE" in _published_strings(path), (
            f"{path.name} must publish NO STANCE as text a reader can see")


def test_the_regime_gauges_and_the_macro_regime_are_wired_to_the_helpers():
    """Source-text check of the wiring the helper tests cannot reach — see the section note.

    It proves each deck reads its readings through the helpers and renders them through
    `_gauge_display`. It does not prove Streamlit renders them.
    """
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        src = path.read_text(encoding="utf-8")
        squashed = re.sub(r"\s+", " ", src)
        assert "readings = _regime_readings(" in src, (
            f"{path.name} does not read its regime readings through the helper")
        assert "macro_trend = _macro_regime(" in src, (
            f"{path.name} does not normalise its macro regime through the helper")
        assert "_gauge_display(" in src, (
            f"{path.name} does not render its gauges through the helper")
        assert ('"Macro Direction": {"BULLISH": "LONG ONLY", "BEARISH": "SHORT ONLY"}'
                '.get(macro_trend, "NO STANCE")') in squashed, (
            f"{path.name} still publishes a macro direction it has no reading for")


def test_neither_dashboard_still_publishes_the_unconditional_macro_direction():
    """The old expression turned every non-BULLISH macro reading into SHORT ONLY."""
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        src = path.read_text(encoding="utf-8")
        assert '"LONG ONLY" if macro_trend == "BULLISH" else "SHORT ONLY"' not in src, (
            f"{path.name} still publishes a direction for a macro reading it may not have")


def test_the_di_spread_is_measured_only_when_both_di_readings_are():
    """The DI gauge used to fall back on 0.0 for both readings and then classified the pair as
    "Balanced" — a measured statement about two numbers nobody measured.

    An absent pair is now an absence; a measured pair keeps the "Balanced" classification when
    both readings are <= 0.
    """
    src = DASHBOARD.read_text(encoding="utf-8")
    assert "di_known = plus_di is not None and minus_di is not None" in src, (
        "the DI spread must be measured only when both DI readings are")
    assert "spread_di = (plus_di - minus_di) if di_known else None" in src, (
        "an absent DI pair must not produce a spread of zero")
    assert 'else "Balanced"' in src, (
        "a measured pair whose readings are both <= 0 keeps the Balanced classification")


def test_a_measured_di_pair_of_zeros_is_still_a_measurement():
    """The control for the absence above: both DI readings at zero is a measurement."""
    classify = _shipped_di_classifier()
    out = _gauge_helper("_gauge_display")(0.0, lambda v: f"{v:+.1f}", classify, "no DI reading")
    assert out["value"] == "+0.0", out
    assert out["delta"] == "Balanced", out
    assert out["arrow"] == "auto", out
    assert out["color"] == "normal", out


def test_the_macro_banner_publishes_a_neutral_regime_and_an_unknown_one_apart():
    """NEUTRAL is a published state the fleet really writes, so it keeps its own banner; an
    absent or UNKNOWN reading must not be dressed as a chop regime either."""
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        src = path.read_text(encoding="utf-8")
        assert 'regime_text = "REGIME: RANGE-BOUND CHOP"' in src, (
            f"{path.name} lost the NEUTRAL banner")
        assert "REGIME: MACRO UNKNOWN (NO STANCE)" in src, (
            f"{path.name} has no banner for a macro regime it has no reading for")
        assert 'elif macro_trend == "NEUTRAL":' in src, (
            f"{path.name} must tell a measured NEUTRAL apart from an UNKNOWN reading")


# ---------------------------------------------------------------------------
# Final fix G-2 — the capital block publishes recorded capital, not a hardcoded deposit schedule
# ---------------------------------------------------------------------------
# What is proven below and what is not: the three helpers under test (`_recorded_amount`,
# `_capital_display`, `_profit_display`) are pure and module-level, so they are extracted by AST
# and executed for real here — the shipped body, not a paraphrase of it. The `st.write(...)`,
# `st.caption(...)`, `st.metric(...)` and `st.markdown(...)` calls they feed are NOT executed: the
# committed suite has no Streamlit AppTest harness, and importing dashboard.py runs a 4,700-line
# top-level script against the real DB and live price endpoints. A helper that returns a dict
# cannot lie about what it decided; the rendering one line further out is pinned by source-text
# checks, nothing more.
#
# The defect being closed: dashboard.py published a deposit schedule nobody measured — $250.00 of
# deposited capital, a $55.00 seed, an $82.00 top-up 1 and a $113.00 top-up 2 — for any fleet that
# had never recorded a deposit, and computed a "Real Net Profit" against that invented capital.
# scripts/live_dashboard.py published the same kind of schedule with a DIFFERENT invented total
# ($137.00), so the two dashboards disagreed about how much capital the fleet had. The invented
# literal also outranked the operator's own configured INITIAL_BALANCE, because
# `num(os.getenv("INITIAL_BALANCE"), 250.0)` returns 250.0 whenever the env var is unset.
#
# The convention: a figure nobody recorded is published as an absence ("not recorded" for a
# capital figure, "not measured" for a profit), never as a number and never as a zero. A recorded
# zero is a fact and stays a number.

_CAPITAL_IDENTIFIERS = (
    "total_deposited", "deposit_t1", "deposit_t2", "deposit_t3",
    "total_profit", "total_roi_pct",
)

_CAPITAL_HELPERS = {}


def _capital_helper(name, path=DASHBOARD):
    """One of the three capital helpers, extracted once per (file, name) and cached.

    The extraction is deliberately lazy: a helper that does not exist yet must fail the test
    that asks for it, not the whole module at collection time.
    """
    key = (str(path), name)
    if key not in _CAPITAL_HELPERS:
        _CAPITAL_HELPERS[key] = _dashboard_helper(name, path)
    return _CAPITAL_HELPERS[key]


def _numeric_literal_sites(path):
    """Every numeric constant that sits inside a module-level function, with that function's name.

    The whole-module `_numeric_literals` helper cannot tell the integer 250 (a token estimate)
    from the float 250.0 (an invented capital figure), because 250 == 250.0 in Python. This one
    keeps the function each constant sits in, so the survivor can be pinned to its one
    legitimate home.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    sites = []
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                    and not isinstance(node.value, bool):
                sites.append((fn.name, node.value))
    return sites


@pytest.mark.parametrize("raw_values, expected", [
    (("55.00",), 55.0),
    (("55.00", "82.00"), 55.0),
    (("", "82.00"), 82.0),
    (("", None, "junk"), None),
    ((), None),
    (("not recorded",), None),
    (("$1,000.00",), 1000.0),
    (("0",), 0.0),
])
def test_recorded_amount_returns_the_first_usable_amount(raw_values, expected):
    """A deposit tranche is an operator figure: recorded in system_state, or configured in the
    environment. The first usable one wins, and nothing usable means the figure was never
    recorded.

    A recorded zero is a fact, not an absence: it must not collapse into None the way an empty
    string or a junk value does.
    """
    assert _capital_helper("_recorded_amount")(*raw_values) == expected


def test_capital_display_publishes_a_recorded_deposit_schedule():
    """The schedule a reader sees when every figure was recorded."""
    out = _capital_helper("_capital_display")(250.0, 55.0, 82.0, 113.0)
    assert out["deposited"] == "$250.00", out
    assert out["deposited_usdt"] == "$250.00 USDT", out
    assert out["seed"] == "$55", out
    assert out["topup1"] == "$82", out
    assert out["topup2"] == "$113", out
    assert out["topup2_shown"] is True, out
    assert out["tranches"] == "$55 seed + $82 top-up 1 + $113 top-up 2", out
    assert out["tranches_plain"] == "$55 + $82 + $113", out


def test_capital_display_publishes_an_absence_for_an_unrecorded_deposit_schedule():
    """A fleet with no recorded deposit has no deposit schedule. It must not render one."""
    out = _capital_helper("_capital_display")(None, None, None, None)
    assert out["deposited"] == "not recorded", out
    assert out["deposited_usdt"] == "not recorded", out
    assert out["seed"] == "not recorded", out
    assert out["topup1"] == "not recorded", out
    assert out["topup2"] == "not recorded", out
    assert out["topup2_shown"] is None, out
    assert "not recorded" in out["tranches"], out
    assert "not recorded" in out["tranches_plain"], out
    assert "$" not in out["tranches"], (
        f"an unrecorded schedule must publish no money at all: {out['tranches']}")


def test_capital_display_publishes_no_third_tranche_for_a_recorded_zero():
    """A third tranche of zero is a recorded fact, but there is no top-up 2 to show."""
    out = _capital_helper("_capital_display")(250.0, 55.0, 82.0, 0.0)
    assert out["topup2_shown"] is False, out
    assert "top-up 2" not in out["tranches"], out


def test_capital_display_keeps_a_recorded_tranche_when_the_total_is_unrecorded():
    """The tranches were recorded even though the total was not: each figure stands on its own."""
    out = _capital_helper("_capital_display")(None, 55.0, 82.0, None)
    assert out["deposited_usdt"] == "not recorded", out
    assert out["seed"] == "$55", out
    assert out["topup1"] == "$82", out


@pytest.mark.parametrize("total_profit, total_roi_pct, expected", [
    (12.5, 5.0, "+$12.50 USDT (+5.00%)"),
    (-3.25, -1.5, "-$3.25 USDT (-1.50%)"),
    (0.0, 0.0, "+$0.00 USDT (+0.00%)"),
])
def test_profit_display_publishes_a_measured_profit(total_profit, total_roi_pct, expected):
    """A measured break-even is a fact, not an absence: it publishes as a number."""
    assert _capital_helper("_profit_display")(total_profit, total_roi_pct) == expected


@pytest.mark.parametrize("total_profit, total_roi_pct", [
    (None, None),
    (None, 5.0),
    (12.5, None),
])
def test_profit_display_says_it_was_not_measured(total_profit, total_roi_pct):
    """A profit needs both a live balance and a recorded capital.

    With either missing there is no profit to publish, and publishing $0.00 would say the fleet
    broke even — a measured statement about a fleet that was never measured.
    """
    out = _capital_helper("_profit_display")(total_profit, total_roi_pct)
    assert "not measured" in out, out


def test_the_two_dashboards_publish_the_same_capital_figures():
    """scripts/live_dashboard.py is a standalone Streamlit script and cannot import dashboard.py,
    so it carries its own copies of the three helpers. The copies have to agree, or the same fleet
    would be shown two different capital schedules."""
    cases = [
        ("_recorded_amount", (("55.00", "82.00"),)),
        ("_recorded_amount", (("", None, "junk"),)),
        ("_recorded_amount", (("0",),)),
        ("_capital_display", (250.0, 55.0, 82.0, 113.0)),
        ("_capital_display", (None, None, None, None)),
        ("_capital_display", (None, 55.0, 82.0, 0.0)),
        ("_profit_display", (12.5, 5.0)),
        ("_profit_display", (None, None)),
    ]
    for name, args in cases:
        main = _capital_helper(name, DASHBOARD)
        legacy = _capital_helper(name, LEGACY_DASHBOARD)
        assert main(*args) == legacy(*args), (
            f"the two dashboards disagree about {name}{args}: {main(*args)} != {legacy(*args)}")


def test_the_two_dashboards_carry_byte_identical_copies_of_the_capital_helpers():
    """The copies are duplicated on purpose (the legacy script cannot import dashboard.py), so
    they are pinned to be the same source, not merely to behave the same today."""
    import ast

    for name in ("_recorded_amount", "_capital_display", "_profit_display"):
        sources = []
        for path in (DASHBOARD, LEGACY_DASHBOARD):
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(path))
            node = next(n for n in tree.body
                        if isinstance(n, ast.FunctionDef) and n.name == name)
            sources.append(ast.get_source_segment(text, node))
        assert sources[0] == sources[1], (
            f"{name} differs between the two dashboards, so the same fleet would be shown two "
            f"different capital schedules")


def test_the_invented_capital_figures_are_gone_from_both_dashboards():
    """The HUD used to publish a deposit schedule nobody measured.

    dashboard.py carried $250.00 / $55.00 / $82.00 / $113.00 and scripts/live_dashboard.py carried
    $137.00 / $55.00 / $82.00, so the two dashboards disagreed about how much capital the fleet
    had — and neither figure was ever measured.
    """
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        published = _published_strings(path)
        for gone in ("55.00", "82.00", "113.00", "137.00"):
            hits = [s for s in published if gone in s]
            assert not hits, f"{path.name} still publishes {gone!r}: {hits[:3]}"
        # The invented total was the FLOAT constant 250.0. dashboard.py legitimately carries the
        # INTEGER 250 elsewhere (the AI telemetry token estimate), and 250 == 250.0 in Python, so
        # the whole-module membership test cannot be used on that file: the float-typed check
        # below is what pins the defect, and the integer is pinned to its one legitimate home by
        # the next test.
        floats = [v for v in _numeric_literals(path) if isinstance(v, float)]
        assert 250.0 not in floats, (
            f"{path.name} still carries the invented 250.0 capital figure as a float constant")
    assert 250.0 not in _numeric_literals(LEGACY_DASHBOARD), (
        "the legacy dashboard still carries the invented 250.0 capital figure")


def test_the_one_remaining_250_in_dashboard_is_not_a_capital_figure():
    """dashboard.py carries the integer literal 250 in the AI usage estimate (the tape
    whisperer's per-call token cost). It is not a capital figure, and it is the only 250 left."""
    leftovers = [fn for fn, value in _numeric_literal_sites(DASHBOARD) if value == 250]
    assert leftovers == ["fetch_ai_usage_stats"], (
        f"dashboard.py carries a 250 outside the AI token estimate: {leftovers}")


def test_the_invented_deposit_schedule_is_gone_from_the_source():
    """Neither file may read a tranche through the old `state_num(key, invented_literal)` shape,
    and both must read the tranches through `_recorded_amount`."""
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        src = path.read_text(encoding="utf-8")
        assert 'state_num("initial_deposit_tranche_1"' not in src, (
            f"{path.name} still reads the seed tranche with an invented fallback")
        assert 'state_num("topup_deposit_tranche_2"' not in src, (
            f"{path.name} still reads the top-up tranche with an invented fallback")
        assert "_recorded_amount(" in src, (
            f"{path.name} does not read its capital figures through the helper")
        assert 'os.getenv("INITIAL_DEPOSIT_TRANCHE_1", "")' in src, (
            f"{path.name} does not read the operator's configured seed tranche")
        # Scoped form of the whole-module 250 check: no invented 250 may sit on a capital line.
        capital_lines = {i for i, line in enumerate(src.splitlines(), start=1)
                         if any(name in line for name in _CAPITAL_IDENTIFIERS)}
        import ast

        tree = ast.parse(src, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                    and not isinstance(node.value, bool) and node.value == 250:
                assert node.lineno not in capital_lines, (
                    f"{path.name}:{node.lineno} still carries an invented 250 on a capital line")


def test_the_capital_block_is_wired_to_the_helpers():
    """Source-text check of the wiring the helper tests cannot reach — see the section note.

    It proves each dashboard reads its capital figures through the helpers and renders them
    through `_capital_display` / `_profit_display`. It does not prove Streamlit renders them.
    """
    main = DASHBOARD.read_text(encoding="utf-8")
    legacy = LEGACY_DASHBOARD.read_text(encoding="utf-8")
    assert "capital = _capital_display(total_deposited, deposit_t1, deposit_t2, deposit_t3)" in main, (
        "dashboard.py does not build its capital figures through the helper")
    assert "capital = _capital_display(total_deposited, deposit_t1, deposit_t2, None)" in legacy, (
        "scripts/live_dashboard.py does not build its capital figures through the helper")
    for src, name in ((main, "dashboard.py"), (legacy, "scripts/live_dashboard.py")):
        assert 'os.getenv("INITIAL_BALANCE", "")' in src, (
            f"{name} does not read the operator's configured capital")
        assert 'os.getenv("TOPUP_DEPOSIT_TRANCHE_2", "")' in src, (
            f"{name} does not read the operator's configured top-up tranche")
        assert "_profit_display(total_profit, total_roi_pct)" in src, (
            f"{name} does not render its profit through the helper")
        assert 'num(os.getenv("INITIAL_BALANCE"), 250.0)' not in src, (
            f"{name} still lets an invented literal outrank the operator's configured capital")
    assert 'os.getenv("TOPUP_DEPOSIT_TRANCHE_3", "")' in main, (
        "dashboard.py does not read the operator's configured third tranche")


# ---------------------------------------------------------------------------
# Final fix G-3 — the state-backed labels publish readings or absences, never stand-ins
# ---------------------------------------------------------------------------
# What is proven below and what is not: the four helpers under test (`_published_state`,
# `_published_flag`, `_stance_color`, `_absence_note`) are pure and module-level, so they are
# extracted by AST and executed for real here — the shipped body, not a paraphrase of it. The
# `st.metric(...)`, `st.markdown(...)` and `st.write(...)` calls they feed are NOT executed: the
# committed suite has no Streamlit AppTest harness, and importing dashboard.py runs a 5,000-line
# top-level script against the real DB and live price endpoints. A helper that returns a dict
# cannot lie about what it decided; the rendering one line further out is pinned by source-text
# checks, nothing more.
#
# The defect being closed: the top banner published a leverage cap, a Fear & Greed, a news gate
# and a halt state that no measurement produced — `RISK_HALT` defaulted to "0", so a fleet whose
# risk engine had published nothing wore a green "SYSTEM ACTIVE" badge next to the controls that
# halt trading. The Microstructure Veto Pulse published a regime, a warden bias and a DI spread it
# had not measured, painted the warden row green whatever the warden said, and published a
# hardcoded green "CLEAR (Normal Flow)" for a volatility cascade that no component measures at
# all. The AI Cognitive Regime Arbiter published a bias, a trap risk, a thesis and a leverage cap
# out of hardcoded defaults — a trap risk nobody measured rendered as "40%" with a green "Safe
# Expansion" delta. And two of those panels interpolated DB- or LLM-derived text into
# `unsafe_allow_html=True` with no escaping at all.
#
# The convention: a figure that was never recorded is published as an absence, never as a number
# and never as a zero. A recorded zero is a fact and stays a reading.

_G3_STATE_HELPERS = {}


def _g3_helper(name, path=DASHBOARD):
    """One of the four state-label helpers, extracted once per (file, name) and cached.

    The extraction is deliberately lazy: a helper that does not exist yet must fail the test
    that asks for it, not the whole module at collection time.
    """
    key = (str(path), name)
    if key not in _G3_STATE_HELPERS:
        _G3_STATE_HELPERS[key] = _dashboard_helper(name, path)
    return _G3_STATE_HELPERS[key]


# Group 1 — `_published_state`: a reading that was taken, or an absence that names itself.
@pytest.mark.parametrize("value", ["", None, "None", "null", "nan", "N/A", "na", "--", "   "])
def test_g3_published_state_reports_an_absence_for_an_unpublished_key(value):
    """A key nobody published is an absence, and the HUD says why it is absent."""
    out = _g3_helper("_published_state")(value, "no regime bias published")
    assert out == {"text": "not measured", "measured": False,
                   "reason": "no regime bias published"}, out


@pytest.mark.parametrize("value", ["0", "BULL_TREND", "63.0", "LONG_ONLY", "ACTIVE"])
def test_g3_published_state_reports_a_published_value_as_a_measurement(value):
    """A value that IS present is published exactly as the fleet wrote it."""
    out = _g3_helper("_published_state")(value, "no regime bias published")
    assert out == {"text": value, "measured": True, "reason": ""}, out


def test_g3_published_state_keeps_a_published_zero_as_a_reading():
    """A published "0" is a fact. It must not collapse into the absence."""
    out = _g3_helper("_published_state")("0", "no recommended leverage published")
    assert out["measured"] is True, out
    assert out["text"] == "0", out
    assert out["reason"] == "", out


# Group 2 — `_published_flag`: a control flag that was never published is NOT "not halted".
@pytest.mark.parametrize("value", ["", None])
def test_g3_published_flag_reports_an_absence_for_an_unpublished_key(value):
    """A halt flag that was never published must not read as "not halted"."""
    out = _g3_helper("_published_flag")(value, "no halt state published")
    assert out == {"flag": False, "measured": False, "reason": "no halt state published"}, out


def test_g3_published_flag_reads_a_published_zero_as_a_measured_off():
    """A published "0" is a measured off — the state was published, and it was published off."""
    out = _g3_helper("_published_flag")("0", "no halt state published")
    assert out == {"flag": False, "measured": True, "reason": ""}, out


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "YES", "on", "On"])
def test_g3_published_flag_reads_a_published_on_in_every_spelling(value):
    """The fleet's own spelling of a set flag must be read as set."""
    out = _g3_helper("_published_flag")(value, "no halt state published")
    assert out == {"flag": True, "measured": True, "reason": ""}, out


@pytest.mark.parametrize("value", ["HALTED", "2", "enabled", "1.0", "paused"])
def test_g3_published_flag_reads_an_unrecognised_published_value_as_engaged(value):
    """A published value the HUD does not recognise is not evidence of "off".

    The pre-fix code read the halt flag as `!= "0"`, so any spelling other than "0" paused
    entries. Reading an unknown spelling as "off" would move that control from fail-safe to
    fail-open and publish a green "SYSTEM ACTIVE" badge nobody measured.
    """
    out = _g3_helper("_published_flag")(value, "no halt state published")
    assert out == {"flag": True, "measured": True, "reason": ""}, out


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "NO", "off", "OFF"])
def test_g3_published_flag_reads_every_explicit_off_spelling_as_off(value):
    """Only an explicit off is off. A published "0" is a measured off, and the other spellings
    mean the same thing."""
    out = _g3_helper("_published_flag")(value, "no halt state published")
    assert out == {"flag": False, "measured": True, "reason": ""}, out


# Group 3 — `_stance_color`: an absent stance is neutral grey, never the colour of a measured one.
@pytest.mark.parametrize("label", ["TRENDING", "LONG_ONLY", "BULL_TREND", "BEAR_TREND",
                                   "SHORT_ONLY", "CHOP_GRID_ACTIVE", ""])
def test_g3_stance_color_paints_an_unmeasured_stance_neutral_grey(label):
    """The defect: the veto pulse defaulted the regime to "TRENDING" and painted the panel green
    off a reading nobody took. An absent stance must never wear a measured stance's colour."""
    out = _g3_helper("_stance_color")(label, False, ("BULL_TREND", "BULLISH", "TRENDING"),
                                      ("BEAR_TREND", "BEARISH"))
    assert out == "#94a3b8", out


@pytest.mark.parametrize("label, expected", [
    ("BULL_TREND", "#00ff66"),
    ("BULLISH", "#00ff66"),
    ("TRENDING", "#00ff66"),
    ("BEAR_TREND", "#ff003c"),
    ("BEARISH", "#ff003c"),
    ("CHOP_GRID_ACTIVE", "#f59e0b"),
    ("LONG_ONLY", "#f59e0b"),
])
def test_g3_stance_color_paints_a_measured_stance_by_what_it_says(label, expected):
    """A measured stance is coloured by its own words. The warden bias is no longer hardcoded
    green, so a published SHORT_ONLY is painted red."""
    out = _g3_helper("_stance_color")(label, True, ("BULL_TREND", "BULLISH", "TRENDING"),
                                      ("BEAR_TREND", "BEARISH"))
    assert out == expected, out


def test_g3_stance_color_normalises_the_case_of_a_stance():
    """The fleet's vocabulary is upper case, but the comparison must not depend on it."""
    helper = _g3_helper("_stance_color")
    bullish = ("BULL_TREND", "BULLISH", "TRENDING")
    bearish = ("BEAR_TREND", "BEARISH")
    assert helper("bull_trend", True, bullish, bearish) == helper("BULL_TREND", True, bullish, bearish)
    assert helper("bear_trend", True, bullish, bearish) == helper("BEAR_TREND", True, bullish, bearish)


def test_g3_stance_color_paints_a_short_only_warden_red():
    """The warden row used to be hardcoded green whatever the warden said."""
    out = _g3_helper("_stance_color")("SHORT_ONLY", True, ("LONG_ONLY",), ("SHORT_ONLY",))
    assert out == "#ff003c", out


# Group 4 — `_absence_note`: the dimmed parenthetical that says why a reading is missing.
def test_g3_absence_note_names_the_reason_it_is_missing():
    out = _g3_helper("_absence_note")("no 60m warden regime published")
    assert "no 60m warden regime published" in out, out
    assert "<span" in out and "</span>" in out, out


def test_g3_absence_note_escapes_the_reason_it_is_missing():
    """The reason reaches a reader through an HTML block, so it may not carry markup."""
    out = _g3_helper("_absence_note")("<b>x</b>")
    assert "<b>" not in out, out
    assert "&lt;b&gt;" in out, out


# Group 5 — the two apps carry byte-identical copies of the four helpers.
def test_g3_the_two_dashboards_carry_byte_identical_copies_of_the_state_helpers():
    """The copies are duplicated on purpose (the legacy script cannot import dashboard.py), so
    they are pinned to be the same source, not merely to behave the same today."""
    import ast

    for name in ("_published_state", "_published_flag", "_stance_color", "_absence_note"):
        sources = []
        for path in (DASHBOARD, LEGACY_DASHBOARD):
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(path))
            node = next(n for n in tree.body
                        if isinstance(n, ast.FunctionDef) and n.name == name)
            sources.append(ast.get_source_segment(text, node))
        assert sources[0] == sources[1], (
            f"{name} differs between the two dashboards, so the same fleet would be shown two "
            f"different readings")


# Group 6 — the same behaviour across the two apps.
def test_g3_the_two_dashboards_publish_the_same_state_readings():
    """scripts/live_dashboard.py is a standalone Streamlit script and cannot import dashboard.py,
    so it carries its own copies of the four helpers. The copies have to agree, or the same fleet
    would be shown two different stances.

    None of the four takes a callable argument, so nothing here can slip through an `except`
    branch and make the comparison vacuous; the arguments are compared exactly as they are.
    """
    bullish = ("BULL_TREND", "BULLISH", "TRENDING")
    bearish = ("BEAR_TREND", "BEARISH")
    cases = [
        ("_published_state", ("", "no regime bias published"), {}),
        ("_published_state", (None, "no regime bias published"), {}),
        ("_published_state", ("BULL_TREND", "no regime bias published"), {}),
        ("_published_state", ("0", "no regime bias published"), {}),
        ("_published_flag", ("", "no halt state published"), {}),
        ("_published_flag", (None, "no halt state published"), {}),
        ("_published_flag", ("0", "no halt state published"), {}),
        ("_published_flag", ("1", "no halt state published"), {}),
        ("_published_flag", ("HALTED", "no halt state published"), {}),
        ("_published_flag", ("off", "no halt state published"), {}),
        ("_published_flag", ("false", "no halt state published"), {}),
        ("_stance_color", ("TRENDING", False), {"bullish": bullish, "bearish": bearish}),
        ("_stance_color", ("BULL_TREND", True), {"bullish": bullish, "bearish": bearish}),
        ("_stance_color", ("BEAR_TREND", True), {"bullish": bullish, "bearish": bearish}),
        ("_stance_color", ("LONG_ONLY", True), {"bullish": ("LONG_ONLY",),
                                                "bearish": ("SHORT_ONLY",)}),
        ("_absence_note", ("no 60m warden regime published",), {}),
    ]
    for name, args, kwargs in cases:
        main = _g3_helper(name, DASHBOARD)
        legacy = _g3_helper(name, LEGACY_DASHBOARD)
        assert main(*args, **kwargs) == legacy(*args, **kwargs), (
            f"the two dashboards disagree about {name}{args}: "
            f"{main(*args, **kwargs)} != {legacy(*args, **kwargs)}")


# Group 7 — no stand-in default survives in the panels that were rewritten.
def _g3_state_reads(src: str, key: str):
    """Every `query_system_state("<key>", <default>)` call in `src`, with its default.

    The key may be quoted with either quote style and the default is captured verbatim, so a
    non-literal default (a name, a constant) is visible to the caller instead of silently
    escaping a literal-only match.
    """
    pattern = r"query_system_state\(\s*(['\"])%s\1\s*,\s*([^)]*)\)" % re.escape(key)
    return re.findall(pattern, src)


_G3_OLD_DEFAULTS = {
    "RISK_HALT": "0",
    "recommended_leverage": "3",
    "fear_greed_index": "50.0",
    "news_event_active": "0",
    "market_regime_label": "TRENDING",
    "warden_60m_regime": "LONG_ONLY",
    "market_regime_di_spread": "0.0",
    "macro_trend_4h": "NEUTRAL",
    "ai_regime_bias": "DEFENSIVE_CHOP",
    "ai_regime_trap_risk": "0.40",
    "ai_regime_thesis": "Consolidation within range; capital shield priority.",
}

# The keys both dashboards read: the banner's four and the veto pulse's two.
_G3_SHARED_KEYS = (
    "RISK_HALT", "recommended_leverage", "fear_greed_index", "news_event_active",
    "market_regime_label", "warden_60m_regime",
)

# The keys only dashboard.py reads: the legacy panel has no DI row, no 4H macro row and no
# arbiter panel at all — they are a different panel, not a stale copy.
_G3_MAIN_ONLY_KEYS = (
    "market_regime_di_spread", "macro_trend_4h",
    "ai_regime_bias", "ai_regime_trap_risk", "ai_regime_thesis",
)


@pytest.mark.parametrize("key", sorted(_G3_OLD_DEFAULTS))
def test_g3_no_stand_in_default_survives_in_the_state_backed_reads(key):
    """Every one of these keys used to be read with an invented default, and the HUD published
    that default as if a measurement had produced it.

    The guard is on the DEFAULT and not on the word: "LONG_ONLY", "TRENDING" and
    "DEFENSIVE_CHOP" survive as the vocabulary the fleet itself publishes (the warden's stance,
    the regime label, a bias), so they may not be banned outright.
    """
    old = _G3_OLD_DEFAULTS[key]
    paths = (DASHBOARD, LEGACY_DASHBOARD) if key in _G3_SHARED_KEYS else (DASHBOARD,)
    for path in paths:
        src = path.read_text(encoding="utf-8")
        assert f'query_system_state("{key}", "{old}")' not in src, (
            f"{path.name} still reads {key} with the invented default {old!r}")
        for _quote, default in _g3_state_reads(src, key):
            assert default.strip() == '""', (
                f"{path.name} reads {key} with the default {default!r}; only an empty literal "
                f"default may reach the publisher")


@pytest.mark.parametrize("key", _G3_SHARED_KEYS)
def test_g3_both_dashboards_read_the_shared_keys_with_an_empty_default(key):
    """The banner's keys and the veto pulse's keys are read by both apps, and both read them
    with an empty default so the publisher can tell "never published" from "published as 0"."""
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        src = path.read_text(encoding="utf-8")
        reads = _g3_state_reads(src, key)
        assert reads, f"{path.name} does not read {key} at all"
        for _quote, default in reads:
            assert default.strip() == '""', (
                f"{path.name} reads {key} with the default {default!r}; only an empty literal "
                f"default may reach the publisher")


@pytest.mark.parametrize("key", _G3_MAIN_ONLY_KEYS)
def test_g3_dashboard_reads_the_panel_only_keys_with_an_empty_default(key):
    """The DI spread, the 4H macro trend and the three arbiter keys live only in dashboard.py."""
    src = DASHBOARD.read_text(encoding="utf-8")
    reads = _g3_state_reads(src, key)
    assert reads, f"dashboard.py does not read {key} at all"
    for _quote, default in reads:
        assert default.strip() == '""', (
            f"dashboard.py reads {key} with the default {default!r}; only an empty literal "
            f"default may reach the publisher")


def test_g3_the_default_guard_cannot_pass_vacuously():
    """The Group 7 guard must fail on a read whose default is not an empty literal.

    Before this round the guard matched only a literal double-quoted default, so a read like
    `query_system_state("RISK_HALT", HALT_DEFAULT)` matched nothing and `all(...)` over an
    empty list passed.

    The guard's own assertion is the thing under test here, so it is expected to raise on the
    non-literal default: the hole this pins is the guard passing vacuously, not the guard
    rejecting a default it can see.
    """
    src = 'x = query_system_state("RISK_HALT", HALT_DEFAULT)\n'
    reads = _g3_state_reads(src, "RISK_HALT")
    assert reads, "a non-literal default must still be visible to the guard"
    with pytest.raises(AssertionError):
        for _quote, default in reads:
            assert default.strip() == '""', (
                f"the guard accepted the non-literal default {default!r}")


def test_g3_the_unused_state_reads_are_gone():
    """`news_event_reason` and `ai_regime_updated_at` were read and read nowhere. Their defaults
    ("No active events") were stand-ins in their own right.

    G-4 put the news half of this guard back to work: the gate line now publishes the reason the
    calendar wrote, so `news_event_reason` is no longer a dead read and banning the key would
    only ban fixing it. What this guard still pins is the thing G-3 was actually about — no dead
    local (`news_reason`), and no stand-in default on the read. `test_g4_the_news_gate_reason_
    reaches_a_published_line` is the other half: the key must be read *and* published.
    """
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        src = path.read_text(encoding="utf-8")
        assert "news_reason" not in src, (
            f"{path.name} still binds the unused news_reason local")
        assert 'query_system_state("news_event_reason", "No active events")' not in src, (
            f"{path.name} gives news_event_reason a stand-in default")
    assert "arb_upd" not in DASHBOARD.read_text(encoding="utf-8"), (
        "dashboard.py still reads the unused ai_regime_updated_at")


def test_g3_the_cascade_row_publishes_no_all_clear():
    """No component in this repository publishes a volatility-cascade state key, so the row has
    nothing to read: it states the absence and why. It is not wired to funding_crash_risk or
    btc_contagion_risk_score, which are different measurements."""
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        src = path.read_text(encoding="utf-8")
        assert "CLEAR (Normal Flow)" not in src, (
            f"{path.name} still publishes a hardcoded all-clear for a measurement nobody took")
        assert "no cascade telemetry is published" in src, (
            f"{path.name} does not say why the cascade row is empty")


def test_g3_the_di_row_publishes_no_green_absence():
    """The DI spread borrows the regime's colour, but an unmeasured spread must not wear the
    regime's green: a green "not measured" is an all-clear the HUD did not measure."""
    src = DASHBOARD.read_text(encoding="utf-8")
    # The veto pulse's bullet row only. The G-1 gauge deck also carries a
    # `st.metric(label="DI Polarity Spread")`, which is a different panel and a different
    # measurement, so the row filter is narrowed to the labelled bullet.
    di_rows = [line for line in src.split("\n") if "**DI Polarity Spread:**" in line]
    assert len(di_rows) == 1, di_rows
    row = di_rows[0]
    assert "{di_color}" in row, row
    assert "{reg_color}" not in row, row
    assert 'di_color = reg_color if di_state is not None else "#94a3b8"' in src, (
        "the DI colour must fall back to neutral grey when the spread was not measured")


# Group 8 — the two HTML injection holes are closed.
def test_g3_the_arbiter_thesis_interpolation_is_escaped():
    """The thesis is model-generated prose interpolated into an unsafe_allow_html=True block."""
    src = DASHBOARD.read_text(encoding="utf-8")
    assert 'html.escape(arb_thesis_state["text"])' in src, (
        "the arbiter thesis is not escaped before it reaches the HTML block")
    assert "{arb_thesis}" not in src, (
        "the arbiter thesis is still interpolated raw into an unsafe_allow_html=True block")
    assert "Arbiter Structural Thesis" in src, (
        "the arbiter thesis card must survive the fix")


def test_g3_the_legacy_veto_pulse_escapes_every_interpolated_value():
    """The legacy app never received the L-1 escaping fix and did not even import html."""
    src = LEGACY_DASHBOARD.read_text(encoding="utf-8")
    assert "import html" in src, (
        "scripts/live_dashboard.py does not import html, so it cannot escape anything")
    assert "safe_market_label = html.escape(" in src, (
        "the legacy veto pulse does not escape the market regime label")
    assert "safe_warden_regime = html.escape(" in src, (
        "the legacy veto pulse does not escape the warden bias")
    assert "safe_last_veto = html.escape(str(last_sniper_veto))" in src, (
        "the legacy veto pulse does not escape the last filter gate")
    assert "{safe_market_label}" in src, (
        "the legacy veto pulse does not interpolate the escaped market regime label")
    assert "{safe_warden_regime}" in src, (
        "the legacy veto pulse does not interpolate the escaped warden bias")
    assert "{veto_preview}" in src, (
        "the legacy veto pulse does not interpolate the escaped last filter gate")
    for raw in ("{market_label}", "{warden_regime}", "{last_sniper_veto}"):
        assert raw not in src, (
            f"the legacy veto pulse still interpolates {raw} raw into an unsafe_allow_html=True "
            "block")
    assert "{last_sniper_veto[:40]}..." not in src, (
        "the legacy veto pulse still prints an ellipsis on every value, even a three-character one")


# Group 9 — the banner cannot claim an all-clear it did not measure.
def test_g3_the_banner_publishes_an_unpublished_halt_state_in_neutral_grey():
    """A halt flag nobody published is not "SYSTEM ACTIVE": it is a grey badge that says the
    halt state was never published."""
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        src = path.read_text(encoding="utf-8")
        assert '"HALT STATE NOT PUBLISHED"' in src, (
            f"{path.name} does not publish an unpublished halt state")
        assert "badge-neutral" in src, (
            f"{path.name} does not use the neutral badge for an unpublished halt state")
        assert ".badge-neutral {" in src, (
            f"{path.name} has no .badge-neutral CSS rule")
        assert 'risk_halt_state = _published_flag(query_system_state("RISK_HALT", "")' in src, (
            f"{path.name} does not read the halt flag through the publisher")
        assert '"SYSTEM ACTIVE"' in src, (
            f"{path.name} no longer publishes the measured-off halt text")
        assert '"ENTRIES PAUSED (CLOSE ONLY)"' in src, (
            f"{path.name} no longer publishes the measured-on halt text")


def test_g3_the_banner_no_longer_claims_an_all_clear_it_did_not_measure():
    """The news gate and the halt badge used to read CLEAR and SYSTEM ACTIVE for a fleet that
    had published neither."""
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        src = path.read_text(encoding="utf-8")
        assert "NOT PUBLISHED" in src, (
            f"{path.name} does not publish an unpublished news gate")
        assert ('news_state = _published_flag(query_system_state("news_event_active", "")'
                in src), (
            f"{path.name} does not read the news gate through the publisher")
        assert "{'SYSTEM ACTIVE' if risk_halt == '0' else 'ENTRIES PAUSED (CLOSE ONLY)'}" not in src, (
            f"{path.name} still claims a halt state it may not have published")
        assert "{'EVENT VETO' if news_active=='1' else 'CLEAR'}" not in src, (
            f"{path.name} still claims a news gate it may not have published")


def test_g3_the_banner_publishes_an_unmeasured_leverage_cap_and_fear_greed_as_absences():
    """The banner used to publish "3x Dynamic" and "50/100" for a fleet that had published
    neither."""
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        src = path.read_text(encoding="utf-8")
        assert 'rec_lev_state = _published_state(query_system_state("recommended_leverage", "")' \
            in src, (
            f"{path.name} does not read the leverage cap through the publisher")
        assert 'fg_num = _optional_float(query_system_state("fear_greed_index", ""))' in src, (
            f"{path.name} does not read the Fear & Greed through the number parser")
        assert 'lev_text = f"{html.escape(rec_lev_state[\'text\'])}x Dynamic"' in src, (
            f"{path.name} does not publish the leverage cap through the publisher")
        assert 'fg_text = f"{fg_num:.0f}/100" if fg_num is not None else "not measured"' in src, (
            f"{path.name} does not publish the Fear & Greed as an absence when it is missing")


# ---------------------------------------------------------------------------
# G-4 — the number-shaped stand-ins in the two Streamlit dashboards
# ---------------------------------------------------------------------------
# What is proven below and what is not: the two dashboards are Streamlit scripts with module-level
# DB reads, HTTP fetches and chart rendering, so no test imports them. The publishers they call are
# pure, so `_published_number` and `_published_boolean` are compiled out of their own source and
# called for real — that decides every behaviour claim here. The panels themselves are checked as
# source text and as syntax trees: a stand-in reading that survives inside a panel is a reading the
# panel publishes. Before/after render parity on this box (every measured digit unchanged, only the
# `mlofi` -0.0 sign moving) was measured by hand against the read-only databases and is recorded in
# the G-4 report; the committed suite cannot boot Streamlit, so it pins the source instead.
#
# The convention is the one G-1 and G-3 landed: a figure that was never recorded publishes an
# absence — "not measured" plus the reason, a NOT MEASURED badge, neutral grey #94a3b8 — and never a
# number. A recorded zero is a fact and stays a number, and a measured -0.0 keeps its sign.

_G4_PUBLISHED = {}


def _g4_helper(name, path=None):
    """One of the G-4 publishers, extracted once per (file, name) and cached.

    Lazy on purpose, as `_g3_helper` is: a helper that does not exist yet must fail the test that
    asks for it, not the whole module at collection time.

    `path` is resolved when the helper is called, not when it is defined, so the pre-fix re-run
    can point the whole G-4 group at `git show HEAD:` extracts. A default argument would have
    frozen the live path into the signature and the re-run would have proved nothing.
    """
    target = DASHBOARD if path is None else path
    key = (str(target), name)
    if key not in _G4_PUBLISHED:
        _G4_PUBLISHED[key] = _dashboard_helper(name, target)
    return _G4_PUBLISHED[key]


# Every marker is ASCII-anchored so no emoji from either file enters a regex.
_G4_BLOCKS = {
    "macro": (r"^.*Macro & News Event Gating",
              (r"\n[ \t]*st\.markdown\(\s*[\"']#### .*Microstructure Veto Pulse",)),
    "ofi": (r"^.*Real-Time Order Flow Imbalance",
            (r"\n[ \t]*# .*Quantitative Microstructure & Sizing Telemetry",
             r"\n[ \t]*# Quantitative Regime Deck Gauges")),
    "micro": (r"^[ \t]*# .*Quantitative Microstructure & Sizing Telemetry",
              (r"\n[ \t]*# .*Bayesian Markov Chain",)),
    "radar": (r"^[ \t]*# .*Live Microstructure & Alpha Telemetry Banner",
              (r"\n[ \t]*fig_radar = build_position_candlestick_chart",)),
    "fleet": (r"^.*Fleet Multi-Level Microstructure & Toxicity Matrix",
              (r"\n[ \t]*# .*AI Order Flow Tape Whisperer",)),
    # G-5a: the cash-and-carry panel. `^.*BTC Cash-and-Carry Basis Oracle` matches TWO lines (the
    # section comment and the subheader), so `_section_block`'s exactly-once anchor assertion would
    # reject it; the subheader prefix makes it name one panel.
    "basis": (r'^[ \t]*st\.subheader\(".*BTC Cash-and-Carry Basis Oracle',
              (r'\n[ \t]*st\.markdown\("---"\)',)),
    # G-5b: the Bayesian Markov regime deck. Its start is the section comment (unique in the file),
    # and its end is the next section comment — which is also `"micro"`'s end pattern, so both
    # panels keep naming one region each. The heading text is deliberately NOT an anchor: the whole
    # point of B-1 is that the heading changes when the row goes stale.
    "markov": (r"^[ \t]*# .*Bayesian Markov Chain",
               (r"\n[ \t]*# .*Multi-Asset Fleet Microstructure",)),
    # G-5c: the fleet active-position deck. It is three loops over one frame, so all three are
    # registered — the aggregation strip, the per-position cards and the chart-tab builder. The
    # legacy twin has no chart-tab builder, so only the first two resolve there.
    "position_agg": (r"^# Active Positions Query across all 5 fleet sub-engines",
                     (r"\nmax_coin_slots = ",)),
    "position_cards": (r"^[ \t]*# Render visual metric cards for each live position",
                       (r'\n[ \t]*st\.markdown\(".*Detailed Active Position',)),
    "position_chart": (r"^[ \t]*# Build dynamic tabs: one tab for each coin currently open",
                       (r"\n[ \t]*# .*1\.3 Continuous AI In-Flight",)),
    # G-5c C-2/C-3: the hero KPI grid and the funding banner sit OUTSIDE those three loops and
    # publish the deck's own roll-ups, so a scan limited to the loops cannot reach them. Additive:
    # the three entries above keep naming exactly the regions they named before.
    "position_hero": (r"^# Clean distinct active coins",
                      (r"\n# P0: Global Margin Utilization Visual Progress Bar",)),
    # The legacy twin has no funding banner at all, so this one resolves for dashboard.py only.
    "position_strip": (r"^[ \t]*# Fleet Accrued Funding Summary Banner",
                       (r"\n[ \t]*# Render visual metric cards",)),
    # G-5c C-4: the hero card's funding delta and the margin bar's "Active Funding:" badge sit in
    # the gap between `position_hero`'s terminator and the funding banner, so the funding total had
    # no guard that could reach them. Additive: nothing above changes region or semantics.
    "position_funding_bar": (r"^# P0: Global Margin Utilization Visual Progress Bar",
                             (r"\n# .* Main Tabbed Navigation",)),
    # G-5c C-5: the risk tab's "Margin, Leverage & Allocation" column prints the read ceiling and
    # the cash buffer it implies. It sits in no block above, so the strip had no guard that could
    # reach it. Additive: nothing above changes region or semantics.
    "position_limits": (r'^[ \t]*st\.markdown\("#### .*Margin, Leverage & Allocation',
                        (r"\n[ \t]*with rcol2:",)),
    # C-5's briefing card is dashboard-only: the legacy twin has no executive-briefing tab, so this
    # anchor resolves for dashboard.py alone and no C-5 guard asks it of the twin.
    "position_briefing": (r"^[ \t]*# 4 Executive KPI Metric Cards",
                          (r'\n[ \t]*st\.markdown\("---"\)',)),
}

_G4_DASHBOARD_BLOCKS = ("macro", "ofi", "micro", "radar", "fleet")
_G4_LEGACY_BLOCKS = ("macro", "ofi")


def _section_block(path, start_pattern, end_patterns):
    """The source of one panel, so a stand-in reading inside it can be looked for.

    The extraction stops at the first terminator, and also at the first line that dedents back
    past the opening line — a panel is only the panel, and a fragment that ran on to the next
    section would let a guard scan code it was never meant to speak about. The dedented fragment
    is valid Python on its own, so the same text can be searched and parsed.
    """
    import textwrap

    src = path.read_text(encoding="utf-8")
    match = re.search(start_pattern, src, re.M)
    assert match is not None, f"{path.name} has no block starting {start_pattern!r}"
    # M-8: `re.search` stops at the FIRST marker, so a duplicated marker string would retarget
    # every guard that reads this panel without any of them noticing. Uniqueness is part of what
    # an anchor means; a second occurrence is a fork to resolve, not a region to guess inside.
    anchors = len(list(re.finditer(start_pattern, src, re.M)))
    assert anchors == 1, (
        f"{path.name} has {anchors} lines matching the {start_pattern!r} anchor, so this block "
        "no longer names one panel")
    lines = src[match.start():].split("\n")
    start_indent = len(lines[0]) - len(lines[0].lstrip())
    keep = []
    for index, line in enumerate(lines):
        if index and line.strip():
            if len(line) - len(line.lstrip()) < start_indent:
                break
            if any(re.search(pattern, "\n" + line) for pattern in end_patterns):
                break
        keep.append(line)
    block = textwrap.dedent("\n".join(keep))
    assert block.strip(), f"{path.name}'s {start_pattern!r} block extracted empty"
    return block


def _g4_block(name, path):
    start, ends = _G4_BLOCKS[name]
    return _section_block(path, start, ends)


def _g4_oracle_queries(path):
    """Every literal SELECT a dashboard runs against oracle_microstructure.

    Read out of the file's own syntax tree, so the test runs the query the shipped panel runs and
    not a transcription of it.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    queries = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "query_db" and len(node.args) >= 2):
            continue
        sql = node.args[1]
        if isinstance(sql, ast.Constant) and isinstance(sql.value, str) \
                and "oracle_microstructure" in sql.value:
            queries.append(sql.value)
    assert queries, f"{path.name} runs no literal query against oracle_microstructure"
    return queries


def _g4_shipped_oracle_schema():
    """The CREATE TABLE for oracle_microstructure that live_oracle.py actually ships."""
    import ast

    text = (REPO_ROOT / "live_oracle.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(text, filename="live_oracle.py")):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and "CREATE TABLE IF NOT EXISTS oracle_microstructure" in node.value:
            return node.value
    raise AssertionError("live_oracle.py does not create oracle_microstructure with a literal schema")


def _g4_parent_map(tree):
    import ast

    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


def _g4_enclosing_measured_guard(node, parents, source):
    import ast

    ancestor = parents.get(id(node))
    while ancestor is not None:
        if isinstance(ancestor, ast.If):
            test = ast.get_source_segment(source, ancestor.test) or ""
            if "measured" in test:
                return True
        ancestor = parents.get(id(ancestor))
    return False


_G4_DERIVED_NAMES = frozenset({
    "marker_pct", "imb_color", "bias_label", "micro_str", "vpin_color", "vpin_badge",
    "mlofi_color", "micro_color", "hawkes_color", "hawkes_badge", "kyle_color", "kyle_badge",
    "r_v_col", "r_v_lbl", "r_m_col", "r_h_col", "r_h_lbl", "r_kyle_col", "r_lvr_badge",
    "flow_posture", "h_stat", "tox_stat",
    # M-3: `v_stat` was deleted from dashboard.py as write-only dead code, so keeping it here
    # only made the crash guard scan for an assignment that can never be found. A derived name
    # belongs on this list while the source still derives it.
})

# The colours a *measured* reading wears. None of them may be worn by an absence.
_G4_MEASURED_COLOURS = ("#00ff66", "#ff003c", "#f59e0b", "#38bdf8", "#00e5ff")
_G4_ABSENT_COLOUR = "#94a3b8"

_G4_RETIRED_DASHBOARD = (
    "imb", "bid_wall", "ask_wall", "micro_p", "micro_bps", "mlofi_val", "vpin_val",
    "hawkes_spoof_val", "hawkes_rho_val", "kyle_lambda_val", "kyle_pctile_val",
    "lvr_tax_bps_val", "tox_halt_val", "hawkes_veto_val", "live_kelly_pct",
    "r_mlofi", "r_mp", "r_bps", "r_vpin", "r_h_spoof", "r_h_rho", "r_h_veto", "r_kyle",
    "r_kyle_pct", "r_lvr", "r_tox_halt",
    "s_mlofi", "s_mp", "s_bps", "s_vpin", "s_bbid", "s_bask", "s_h_spoof", "s_h_rho",
    "s_h_veto", "s_kyle_pct", "s_lvr_tax", "s_tox_halt",
    # G-5a: five stand-in readings the carry panel assigned and then never used. A stand-in that
    # feeds nothing is still a stand-in the next reader will trust, so they are deleted outright and
    # listed here so no line can quietly re-bind one.
    "b_spot", "b_perp", "b_upnl", "b_has_pos", "b_roundtrip_fee",
)
_G4_RETIRED_LEGACY = ("imb", "bid_wall", "ask_wall")


# Group 1 — `_published_number`: a reading that was taken, or an absence that names itself.
@pytest.mark.parametrize("value", [None, "", "   ", "n/a", "--", "None", "null", "nan",
                                   float("nan"), True, False])
def test_g4_published_number_reports_an_absence_for_an_unpublished_reading(value):
    """There is no number to show, so there is no number shown, and the reason says why."""
    out = _g4_helper("_published_number")(value, "no mlofi in microstructure telemetry")
    assert out == {"value": None, "measured": False,
                   "reason": "no mlofi in microstructure telemetry"}, out


def test_g4_published_number_keeps_a_measured_zero_as_a_reading():
    """A measured 0.0 is a fact. `float(0.0 or 0.5)` collapsed it into a stand-in."""
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        out = _g4_helper("_published_number", path)(0.0, "r")
        assert out == {"value": 0.0, "measured": True, "reason": ""}, (path.name, out)
        assert not (out["value"] is None or out["value"] is False), (path.name, out)


def test_g4_published_number_reads_a_recorded_number_and_a_numeric_string():
    for value, expected in ((0, 0.0), ("2.5", 2.5), (2.5, 2.5), ("  42  ", 42.0),
                            (-0.0, -0.0), ("-1.86", -1.86)):
        out = _g4_helper("_published_number")(value, "r")
        assert out["measured"] is True, (value, out)
        assert out["value"] == expected, (value, out)
        assert out["reason"] == "", (value, out)


def test_g4_published_number_keeps_the_sign_of_a_measured_negative_zero():
    """The one sanctioned digit change G-4 makes: mlofi is measured as -0.0 and used to render
    +0.000. A sign that a formatting choice erased is a reading that was overwritten."""
    import math

    out = _g4_helper("_published_number")(-0.0, "r")
    assert out["measured"] is True, out
    assert math.copysign(1.0, out["value"]) == -1.0, out
    assert f"{out['value']:+.3f}" == "-0.000", out
    assert f"{out['value']:+.2f}" == "-0.00", out


def test_g4_the_two_dashboards_publish_the_same_numeric_readings():
    cases = (0.0, -0.0, 0, 1, 2.5, "2.5", "", "n/a", None, True, False, "--", "1e3")
    main = _g4_helper("_published_number", DASHBOARD)
    legacy = _g4_helper("_published_number", LEGACY_DASHBOARD)
    for value in cases:
        assert main(value, "r") == legacy(value, "r"), (
            f"the two dashboards disagree about {value!r}: {main(value, 'r')} != "
            f"{legacy(value, 'r')}")


# Group 2 — `_published_boolean`: a flag the daemon never computed is NOT "measured false".
@pytest.mark.parametrize("value, expected", [
    (True, True), (False, False), ("true", True), ("1", True), ("yes", True), ("on", True),
    ("false", False), ("0", False), ("no", False), ("off", False), (1, True), (0, False),
])
def test_g4_published_boolean_reads_every_published_flag_spelling(value, expected):
    """A JSON `false` is a measurement: the daemon computed the flag and the answer was no."""
    out = _g4_helper("_published_boolean")(value, "no toxicity_halt in microstructure telemetry")
    assert out == {"value": expected, "measured": True, "reason": ""}, out


@pytest.mark.parametrize("value", [None, "", "   ", "maybe", "HALTED", "2", "1.0", {}, []])
def test_g4_published_boolean_reports_an_absence_for_a_flag_that_was_never_computed(value):
    """`bool(None)` is False, so the plain cast turned "never measured" into a measured no and
    painted a green badge for it. An unrecognised spelling is not evidence of either state."""
    out = _g4_helper("_published_boolean")(value, "no hawkes_veto in microstructure telemetry")
    assert out == {"value": None, "measured": False,
                   "reason": "no hawkes_veto in microstructure telemetry"}, out


def test_g4_published_boolean_is_a_dashboard_only_helper():
    """The legacy app renders no JSON boolean, so it gets no publisher for one. Pinning which
    file owns which helper stops the two apps drifting into different vocabularies."""
    assert "_published_boolean" in DASHBOARD.read_text(encoding="utf-8")
    assert "_published_boolean" not in LEGACY_DASHBOARD.read_text(encoding="utf-8")


# Group 3 — the duplicated copies are pinned to the same source, not merely to the same output.
def test_g4_the_two_dashboards_carry_byte_identical_copies_of_published_number():
    import ast

    sources = []
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        text = path.read_text(encoding="utf-8")
        node = next((n for n in ast.parse(text, filename=str(path)).body
                     if isinstance(n, ast.FunctionDef) and n.name == "_published_number"), None)
        assert node is not None, f"{path.name} has no module-level _published_number"
        sources.append(ast.get_source_segment(text, node))
    assert sources[0] == sources[1], (
        "_published_number differs between the two dashboards, so the same fleet would be shown "
        "two different readings")


# Group 4 — the legacy OFI deck's query is a query that can run.
def test_g4_the_microstructure_query_runs_against_the_shipped_oracle_schema(tmp_path):
    """`scripts/live_dashboard.py` selected `updated_at` from a table whose column is
    `timestamp`. The read-only app never raised into a visible panel — `query_db` swallows the
    error into an empty frame — so the deck silently published its stand-in zeros forever.

    The schema is the one live_oracle.py ships, lifted out of its source so this cannot drift,
    and the query is the one the panel runs, lifted out of the panel the same way.
    """
    import sqlite3

    db = tmp_path / "oracle_schema.db"
    with sqlite3.connect(str(db)) as conn:
        conn.execute(_g4_shipped_oracle_schema())
        conn.execute(
            "INSERT INTO oracle_microstructure (timestamp, symbol, liquidity_imbalance, "
            "vacuum_detected, bid_wall_price, bid_wall_size, ask_wall_price, ask_wall_size, "
            "signal_multiplier) VALUES ('2026-10-22T00:00:00Z', 'BTC/USDT', 0.634, 0, "
            "83865.70, 12.0, 83865.80, 9.0, 1.25)")
        for path in (DASHBOARD, LEGACY_DASHBOARD):
            for sql in _g4_oracle_queries(path):
                assert "updated_at" not in sql, (
                    f"{path.name} selects updated_at, which oracle_microstructure does not have: "
                    f"{sql}")
                conn.row_factory = sqlite3.Row
                row = conn.execute(sql).fetchone()
                assert row is not None, (
                    f"{path.name}'s query returns nothing against the shipped schema: {sql}")
                assert dict(row) == {
                    "symbol": "BTC/USDT", "liquidity_imbalance": 0.634,
                    "bid_wall_price": 83865.70, "ask_wall_price": 83865.80,
                    "timestamp": "2026-10-22T00:00:00Z"}, dict(row)


# Group 5 — the macro gating column publishes no stand-in.
def test_g4_the_macro_column_invents_no_regime_and_no_price():
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        block = _g4_block("macro", path)
        src = path.read_text(encoding="utf-8")
        for gone in ("get('macro_regime', 'NEUTRAL')", '"macro_regime", "NEUTRAL"',
                     "num(df_macro.iloc[0].get(", "float(df_macro.iloc[0].get("):
            assert gone not in block, (
                f"{path.name}'s macro column still invents a reading with {gone!r}")
        # The two shapes are the same defect in two different clothes: `num(x, 0.0)` in the main
        # app and `float(x or 0.0)` in the legacy one both print a number for a row that has none.
        # Only the *numeric* fallback is banned here: `str(.get("macro_regime") or "")` is the
        # correct shape for a text reading, because the empty string it falls back on is what
        # `_published_state` reports as an absence rather than dressing as a regime.
        assert not re.search(r"\.get\([^)]*\)\s*or\s+[\d.]", block), (
            f"{path.name}'s macro column still falls back on a numeric stand-in reading")
        assert "±30m" not in src, (
            f"{path.name} still claims a ±30m news window the code does not use")
        # I-1: pin the bare literal, not a quoted transcription of it. The pre-fix line was
        # `st.write(f"• **Hurst Exponent Filter:** Hurst >= 0.52 for Trend Breakouts")`, where the
        # character before `Hurst` is `*`, so an assertion carrying a leading `"` matched nothing in
        # any revision and stayed green if the hardcoded floor came back beside the config read.
        # The shipped line interpolates `{config.HURST_TREND_FLOOR}`, so it cannot contain this.
        assert "Hurst >= 0.52" not in src, (
            f"{path.name} still hardcodes the Hurst floor instead of reading config")
        assert "config.HURST_TREND_FLOOR" in block, (
            f"{path.name} does not source the Hurst floor from the config that owns it")
        assert "config.PRE_NEWS_MINUTES" in block and "config.POST_NEWS_MINUTES" in block, (
            f"{path.name} does not source the news window from the config that owns it")
        assert 'getattr(config' not in block, (
            f"{path.name} reads config through a getattr fallback, which is itself a stand-in")


def test_g4_the_macro_regime_has_three_states_and_never_collapses_them():
    """A real regime is a reading; UNKNOWN is the sentinel recording that it could not resolve
    one; a missing value is an absence. Dressed as NEUTRAL, all three read as an all-clear."""
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        block = _g4_block("macro", path)
        assert 'no macro telemetry row recorded' in block, (
            f"{path.name} does not name why the macro column is empty")
        assert 'UNKNOWN (macro telemetry unresolved)' in block, (
            f"{path.name} dresses an unresolved regime as a resolved one")
        assert 'regime_state = _published_state(raw_regime, no_row)' in block, (
            f"{path.name} does not read the macro regime through the publisher")
        # M-1's pin: the measured branch is the only line in this column that sends database text
        # to a markdown render path without unsafe_allow_html, so it has to escape first — the same
        # way the gate reason beside it does. An unescaped `**` or `[]` in a regime value would
        # render as formatting rather than as the reading.
        assert "Macro Regime:** {html.escape(regime_state['text'])}" in block, (
            f"{path.name} publishes the measured macro regime unescaped")
        assert "Macro Regime:** {regime_state['text']}" not in block, (
            f"{path.name} writes the raw regime text into a markdown render path")
        assert '#94a3b8' in block, f"{path.name} paints an unmeasured regime in a measured colour"


def test_g4_the_news_gate_reason_reaches_a_published_line():
    """G-3 banned this read because it was dead. G-4 puts it to work: the gate line now says
    which event the gate is looking at, so the key must reach a rendered string."""
    import ast

    for path in (DASHBOARD, LEGACY_DASHBOARD):
        block = _g4_block("macro", path)
        assert 'query_system_state("news_event_reason", "")' in block, (
            f"{path.name} gives the news reason a default or does not read it")
        tree = ast.parse(block, filename=str(path))
        published = [ast.get_source_segment(block, node) for node in ast.walk(tree)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                     and node.func.attr == "write"]
        assert any(seg and "gate_detail" in seg for seg in published), (
            f"{path.name} reads the news gate reason and publishes nowhere")
        assert '" — {html.escape(gate_reason)}"' in block, (
            f"{path.name} interpolates the news gate reason into unsafe_allow_html unescaped")
        assert 'gate_detail = ' in block and 'news_state["measured"] and gate_reason' in block, (
            f"{path.name} publishes the gate reason even when the gate state was never measured")


# Group 6 — no hardcoded leverage policy.
def test_g4_the_leverage_line_publishes_the_published_leverage():
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        src = path.read_text(encoding="utf-8")
        assert "3.0x CHOP / 5.0x TREND" not in src, (
            f"{path.name} still prints a leverage policy nobody published")
        assert 'lev_state = _published_number(query_system_state("recommended_leverage", "")' \
            in src, f"{path.name} does not read the leverage through the number publisher"


# Group 7 — the microstructure deck publishes no stand-in reading.
def test_g4_the_micro_deck_publishes_no_stand_in_readings():
    block = _g4_block("micro", DASHBOARD)
    numbers = _numeric_constants(block, filename="dashboard.py micro deck")
    assert 15.0 not in numbers, (
        "the micro deck still falls back on the fabricated 15% Kelly reading")
    assert "live_kelly_pct = 15.0" not in block, "the fabricated Kelly stand-in is back"
    assert not re.search(r"\.get\([^)]*\)\s*or\s+", block), (
        "the micro deck still invents a reading with a `.get(...) or` fallback")
    for gone in ("mlofi_val = 0.0", "micro_p = 0.0", "micro_bps = 0.0", "vpin_val = 0.0",
                 "hawkes_spoof_val = 0.0", "hawkes_rho_val = 0.0", "kyle_lambda_val = 0.0",
                 "kyle_pctile_val = 0.5", "lvr_tax_bps_val = 0.0"):
        assert gone not in block, f"the micro deck still initialises {gone}"
    assert "NOT MEASURED" in block, (
        "the micro deck has no badge for a reading that was never taken")


def test_g4_the_kelly_card_publishes_the_computed_fraction_or_an_absence():
    """The deck used to print 15.0% when RiskManager could not be built. A computed 0.0% is a
    measurement (the engine says there is no edge); an exception is not."""
    block = _g4_block("micro", DASHBOARD)
    assert 'kelly_state = {"value": None, "measured": False, "reason": "no kelly fraction published"}' \
        in block, "the Kelly card has no absence to fall back on"
    assert "kelly_state = _published_number(rm_dash.calculate_empirical_kelly_fraction() * 100.0" \
        in block, "the Kelly card does not publish the computed fraction"
    assert '"no kelly fraction published"' in block, "the Kelly absence names no reason"


# Group 8 — the radar banner and the fleet matrix publish no stand-in.
@pytest.mark.parametrize("block_name", ["radar", "fleet"])
def test_g4_the_symbol_panels_publish_no_stand_in(block_name):
    block = _g4_block(block_name, DASHBOARD)
    assert not re.search(r"\.get\([^)]*\)\s*or\s+", block), (
        f"the {block_name} panel still invents a reading with a `.get(...) or` fallback")
    assert "NOT MEASURED" in block, (
        f"the {block_name} panel has no verdict for a reading that was never taken")
    assert "microstructure telemetry published" in block, (
        f"the {block_name} panel does not name which telemetry is missing")


# Group 8b — the radar banner publishes each Hawkes reading on its own evidence (I-2).
@pytest.mark.parametrize("field,own_key,fmt", [
    ("r_h_spoof_text", "r_hawkes_spoof_st", ".3f"),
    ("r_h_rho_text", "r_hawkes_rho_st", ".2f"),
])
def test_g4_the_radar_banner_keys_each_hawkes_text_on_its_own_state(field, own_key, fmt):
    """A missing `hawkes_veto` may not make the banner disown the spoof index and ρ it measured.

    Both texts used to be assigned inside one trio gate on all three Hawkes keys, so with only
    `hawkes_veto` absent the banner rendered `Hawkes Spoof: not measured (ρ=not measured)` over two
    real readings — an absence that is a lie, the same defect class as a stand-in number pointed
    the other way. A bundled gate is legitimate for the combined verdict (`r_h_col` / `r_h_lbl`),
    which genuinely cannot be formed without all three keys; it is not legitimate for a field.

    Two checks, because the existing `"measured" in segment` heuristic is what let this through: a
    bundled gate also has `measured` in the segment. So the reading line must key on its OWN
    `measured` flag, and no assignment of a published field may sit under a gate that reads
    `hawkes_veto` at all. Both halves fail on the pre-fix block.
    """
    import ast

    block = _g4_block("radar", DASHBOARD)
    tree = ast.parse(block, filename="dashboard.py radar banner")
    parents = _g4_parent_map(tree)
    assigns = [node for node in ast.walk(tree)
               if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == field
                       for target in node.targets)]
    assert assigns, f"the radar banner no longer publishes {field} at all"
    for node in assigns:
        segment = ast.get_source_segment(block, node) or ""
        assert "hawkes_veto" not in segment, (
            f"the radar banner derives the published {field} from hawkes_veto: {segment}")
        ancestor = parents.get(id(node))
        while ancestor is not None:
            if isinstance(ancestor, ast.If):
                gate = ast.get_source_segment(block, ancestor.test) or ""
                assert "hawkes_veto" not in gate, (
                    f"the radar banner publishes {field} under a gate on hawkes_veto, so one "
                    f"missing veto hides a reading that was taken: {gate}")
            ancestor = parents.get(id(ancestor))
    keyed = re.compile(own_key + r"\[[^\]]*measured[^\]]*\]")
    readings = [ast.get_source_segment(block, node) or "" for node in assigns
                if keyed.search(ast.get_source_segment(block, node) or "")]
    assert readings, (
        f"the radar banner never keys {field} on {own_key}'s own measured flag; the reading is "
        "bundled with another field's state")
    for segment in readings:
        assert fmt in segment, (
            f"{field} lost its {fmt} rendering — the two Hawkes fields format differently: "
            f"{segment}")
        assert "not measured" in segment, (
            f"{field} has no absence for when its own state was never measured: {segment}")


# Group 9 — a measured zero percentile survives.
def test_g4_a_measured_zero_percentile_is_not_overwritten_by_its_own_stand_in():
    """`float(ms_json.get("kyle_lambda_pctile", 0.5) or 0.5)` reads a measured 0.0 as falsy and
    replaces it with 0.5 — the stand-in it was standing in for."""
    assert _g4_helper("_published_number")(0.0, "r")["value"] == 0.0
    src = DASHBOARD.read_text(encoding="utf-8")
    assert 'float(ms_json.get("kyle_lambda_pctile", 0.5) or 0.5)' not in src, (
        "the deck still overwrites a measured 0.0 percentile with 0.5")
    assert '_published_number(ms_json.get("kyle_lambda_pctile") if ms_ok else None' in src, (
        "the Kyle percentile is not read through the number publisher")


# Group 10 — no retired bare reading survives.
def test_g4_no_retired_bare_reading_survives():
    """G-4.0: the old float names are gone because every line that did arithmetic on them now
    reads a {"value", "measured"} dict instead. A name still bound means a line still doing
    arithmetic on a value that can be absent — which raises TypeError and kills the panel."""
    import ast

    for path, retired in ((DASHBOARD, _G4_RETIRED_DASHBOARD),
                          (LEGACY_DASHBOARD, _G4_RETIRED_LEGACY)):
        bound = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.Name):
                bound.add(node.id)
            elif isinstance(node, ast.arg):
                bound.add(node.arg)
        survivors = sorted(set(retired) & bound)
        assert not survivors, (
            f"{path.name} still binds the retired reading(s) {survivors}; a derived line is "
            "still treating an absent reading as a float")


# Group 11 — every derived verdict is guarded by `measured`.
def test_g4_every_derived_verdict_is_guarded_by_measured():
    """The crash guard and the false-all-clear guard in one.

    A verdict computed unconditionally from a reading that can be absent either raises TypeError
    (a crashed panel publishes nothing at all) or, if the author remembered `or 0`, publishes a
    verdict that was never measured.
    """
    import ast

    total = 0
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        parents = _g4_parent_map(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            else:
                continue
            for target in targets:
                if not (isinstance(target, ast.Name)
                        and target.id in _G4_DERIVED_NAMES):
                    continue
                total += 1
                segment = ast.get_source_segment(source, node) or ""
                assert "measured" in segment or _g4_enclosing_measured_guard(
                    node, parents, source), (
                    f"{path.name}:{node.lineno} derives {target.id} from a reading that can be "
                    f"absent, unguarded:\n{segment}")
    assert total >= 20, (
        f"only {total} derived verdicts were found; the guard is scanning an empty region and "
        "passing vacuously")


@pytest.mark.parametrize("unguarded", ["if micro_p > 0 else", 'else "Tracking..."',
                                       "if imb > 0.05 else", "if imb > 0.10 else"])
def test_g4_the_unguarded_verdict_shapes_are_gone(unguarded):
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        assert unguarded not in path.read_text(encoding="utf-8"), (
            f"{path.name} still derives a verdict from {unguarded!r}, which reads an absent "
            "reading as a number")


def test_g4_no_absence_is_painted_in_a_measured_colour():
    """Neutral grey is the only colour an absence may wear. Green, amber, red, cyan and blue are
    reserved for readings the fleet actually took.

    Scoped to the panels G-4 rewrote. The capital block in the legacy app has a legitimate
    `if not margin_measured:` / `elif …` ladder whose else-branch is the *measured* side, so a
    file-wide walk reads it backwards; that panel is G-2's and is pinned by its own tests.
    """
    import ast

    guards = 0
    ternaries = 0
    scope = [(name, DASHBOARD) for name in _G4_DASHBOARD_BLOCKS]
    scope += [(name, LEGACY_DASHBOARD) for name in _G4_LEGACY_BLOCKS]
    for block_name, path in scope:
        block = _g4_block(block_name, path)
        tree = ast.parse(block, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and node.orelse and "measured" in (
                    ast.get_source_segment(block, node.test) or ""):
                guards += 1
                for branch in node.orelse:
                    for inner in ast.walk(branch):
                        if not isinstance(inner, ast.Assign):
                            continue
                        for target in inner.targets:
                            if not (isinstance(target, ast.Name)
                                    and (target.id.endswith("_color")
                                         or target.id.endswith("_col"))):
                                continue
                            segment = ast.get_source_segment(block, inner) or ""
                            assert not any(colour in segment
                                           for colour in _G4_MEASURED_COLOURS), (
                                f"{path.name} {block_name} panel paints the absence "
                                f"{target.id} in a measured colour: {segment}")
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.IfExp) \
                    and isinstance(node.targets[0], ast.Name) \
                    and (node.targets[0].id.endswith("_color")
                         or node.targets[0].id.endswith("_col")):
                test = ast.get_source_segment(block, node.value.test) or ""
                if "measured" not in test:
                    continue
                ternaries += 1
                orelse = ast.get_source_segment(block, node.value.orelse) or ""
                assert _G4_ABSENT_COLOUR in orelse, (
                    f"{path.name} {block_name} panel gives {node.targets[0].id} no neutral grey "
                    f"for an absent reading: {orelse}")
    assert guards >= 8 and ternaries >= 10, (
        f"the colour guard inspected only {guards} measured-guards and {ternaries} verdict "
        "ternaries; it is scanning an empty region and passing vacuously")


# Group 12 — the OFI marker is not parked at the false centre.
def test_g4_the_ofi_marker_is_omitted_when_the_reading_is_absent():
    """The marker's 50% position IS the balanced assertion, drawn instead of written. Re-colouring
    it would still assert a centre the panel never measured, so the div is omitted."""
    import ast

    for path in (DASHBOARD, LEGACY_DASHBOARD):
        block = _g4_block("ofi", path)
        tree = ast.parse(block, filename=str(path))
        parents = _g4_parent_map(tree)
        found = 0
        for node in ast.walk(tree):
            segment = ast.get_source_segment(block, node)
            if not isinstance(segment, str):
                continue
            if "left: {marker_pct}%" not in segment and \
                    "background: #ffffff; border: 1px solid #000;" not in segment:
                continue
            # Only the construction that *emits* the div, not the giant f-string that contains it.
            if not isinstance(node, (ast.JoinedStr, ast.Assign)):
                continue
            found += 1
            assert _g4_enclosing_measured_guard(node, parents, block), (
                f"{path.name}'s OFI marker div is emitted unconditionally:\n{segment[:200]}")
        assert found, f"{path.name}'s OFI block no longer builds the marker div at all"
        assert "{marker_html}" in block, (
            f"{path.name}'s OFI deck does not interpolate the marker through a variable")
        assert re.search(r'marker_html\s*=\s*""', block), (
            f"{path.name}'s OFI deck has no empty marker for the absence case")


# The absence convention itself, one step further than the panels can reach on a healthy box.
def test_g4_the_absence_reasons_name_the_reading_that_is_missing():
    """On this box every key is present, so the panels print today's digits and the reasons never
    appear. They are still the difference between a blank and a blank that says why."""
    block = _g4_block("micro", DASHBOARD)
    assert "microstructure telemetry is not readable" in block
    assert '"no mlofi in microstructure telemetry"' in block
    assert '"no vpin in microstructure telemetry"' in block
    assert '"no kyle_lambda_pctile in microstructure telemetry"' in block
    ofi = _g4_block("ofi", DASHBOARD)
    assert "no oracle microstructure row recorded" in ofi
    assert "unknown symbol" in ofi, "the OFI deck still prints a blank symbol for a missing one"
    note = _g4_helper("_absence_note")("no vpin in microstructure telemetry")
    assert note.strip().startswith("<span style="), note
    assert "#94a3b8" in note, note
    assert "(no vpin in microstructure telemetry)</span>" in note, note


# ---------------------------------------------------------------------------
# G-5a — the cash-and-carry basis panel: a parked engine's economics as readings
# ---------------------------------------------------------------------------
# Measured at BASE against `file:trading_data.db?mode=ro`, `system_state`, key
# `basis_trader_telemetry`: the payload carries FIVE keys for an engine that is DISABLED, while the
# panel read twenty-three. Every field the payload does not carry came from a hardcoded stand-in, so
# the HUD published a $1,000 allocation, an 8 % hurdle, a 100 % liquidation buffer, a CROSSED margin
# mode and an "Active" margin guard for a book that does not exist.
#
# What is proven below: `_basis_readings` and the panel's own f-string are executed for real. The
# panel block is compiled out of dashboard.py and run against the live payload in a namespace built
# from dashboard.py's own top-level helpers, so a `{x['value']}` that formats None raises here rather
# than on the fleet — that is the crash-hazard claim, and it is a real one. `query_system_state` is
# `@st.cache_data`-wrapped and cannot be extracted, so the DB read itself is stubbed at its return
# value; the row is still seeded in a real `system_state` table and read back through sqlite, so the
# JSON the panel parses is the JSON a publisher would send. The clock is injected, so the age line
# is deterministic.

_G5_LIVE_PAYLOAD = {
    "status": "DISABLED",
    "state": "DISABLED",
    "reason": "Temporarily parked for account segregation",
    "net_delta": 0.0,
    "timestamp": "2026-09-17T22:45:48.677606+00:00",
}

# 2026-09-17T22:45:48Z published, this instant reading: 35 days.
_G5_FIX_NOW = datetime(2026, 10, 22, 20, 0, tzinfo=timezone.utc)

# The producer of the row above. `dashboard.py` has one reader and this file has two writers, so the
# fixtures below are built from its syntax tree rather than transcribed: a hand-written "active
# engine" payload is a claim about the producer, and the last three rounds of this branch were paid
# for by exactly that kind of claim.
BASIS_TRADER = REPO_ROOT / "archive" / "experimental_traders" / "basis_trader.py"


def _g5_producer_payload_keys(shape):
    """The keys one basis-trader writer actually publishes, read out of the producer's AST.

    `shape` is "active" — the heartbeat `publish_telemetry` enqueues into `system_state` — or
    "parked" — the payload `main()` writes when `config.BASIS_TRADER_ENABLED` is false. Each is a
    dict literal of constant string keys and the two are told apart by the keys themselves (only
    one carries the computed `net_delta_usd`), so a rename or a reordering moves this test instead
    of silently pointing it at the wrong writer.
    """
    import ast

    src = BASIS_TRADER.read_text(encoding="utf-8")
    dicts = []
    for node in ast.walk(ast.parse(src, filename=str(BASIS_TRADER))):
        if isinstance(node, ast.Dict) and all(isinstance(key, ast.Constant)
                                              and isinstance(key.value, str)
                                              for key in node.keys):
            dicts.append(frozenset(key.value for key in node.keys))
    if shape == "active":
        found = [keys for keys in dicts if "net_delta_usd" in keys]
    elif shape == "parked":
        found = [keys for keys in dicts
                 if "net_delta" in keys and "net_delta_usd" not in keys]
    else:
        raise AssertionError(f"no such basis trader shape: {shape!r}")
    assert len(found) == 1, (
        f"expected exactly one basis trader {shape} payload, found {len(found)}: {found}")
    return found[0]


# A running engine's book, at figures the panel formats distinctly, so a field that stopped being
# read shows up as a missing number rather than as another zero.
_G5_ENGINE_VALUES = {
    "engine": "basis_trader", "state": "ACTIVE_DELTA_NEUTRAL",
    "status": "ACTIVE_DELTA_NEUTRAL", "symbol": "BTC/USDT",
    "spot_price": 110000.0, "perp_price": 110200.0, "basis_spread_usd": -350.25,
    "basis_dislocation_pct": 0.42, "funding_rate_8h": 0.00012,
    "annualized_yield_pct": 11.25, "hurdle_rate_pct": 8.0, "unwind_hurdle_pct": 1.5,
    "capital_allocated_usd": 2500.0, "net_delta_usd": -0.75, "market_neutral": True,
    "has_active_position": True, "cumulative_funding_harvested": 1.8675,
    "unrealized_pnl_usd": 12.5, "holding_days": 3.5, "min_holding_days": 7.0,
    "roundtrip_fee_usd": 2.4, "fee_hurdle_target_usd": 3.0, "fee_amortized_pct": 118.4,
    "margin_mode": "isolated", "liquidation_price": 91000.0, "distance_to_liq_pct": 24.5,
    "min_liq_distance_pct": 20.0, "position": {"spot_qty": 0.0227},
    "timestamp": _G5_FIX_NOW.isoformat(),
}


def _g5_active_engine_payload():
    """One payload with every key the ACTIVE heartbeat publishes, and no key it does not.

    The key set comes from the producer, not from this file, which is the whole point: the active
    engine publishes `net_delta_usd` and there is no `net_delta` and no `reason` in its dict
    literal, so a fixture that said otherwise would be the same invention the panel was rebuilt to
    stop making. A key the producer adds later takes `0.0` and the panel's own coverage line
    reports it.
    """
    keys = _g5_producer_payload_keys("active")
    return {key: _G5_ENGINE_VALUES.get(key, 0.0) for key in sorted(keys)}


# The fields the panel used to invent and that the live five-key payload does not carry.
# `min_liq_distance_pct` joined this list when the guard started reading the engine's own published
# minimum instead of hardcoding a ladder (M-2): the parked path publishes no threshold, and a
# threshold of `0.0` from a running engine is a legitimate reading, not an absence.
_G5_ABSENT_IN_LIVE = ("capital_allocated_usd", "hurdle_rate_pct", "distance_to_liq_pct",
                      "margin_mode", "annualized_yield_pct", "unwind_hurdle_pct",
                      "funding_rate_8h", "cumulative_funding_harvested", "liquidation_price",
                      "fee_amortized_pct", "fee_hurdle_target_usd", "holding_days",
                      "min_holding_days", "basis_spread_usd", "basis_dislocation_pct",
                      "min_liq_distance_pct")

# The fields published through `_published_state`, which carry {"text"} instead of {"value"}.
_G5_TEXT_FIELDS = ("state", "reason", "margin_mode", "timestamp")

# Every payload field the panel publishes, so an "all fields measured" fixture can be built without
# transcribing the panel by hand.
_G5_ALL_FIELDS = _G5_ABSENT_IN_LIVE + ("state", "reason", "net_delta", "timestamp")

# (rendered text variable, the one state variable it may read). A field's own text must key on its
# own flag — G-4's radar banner is the precedent, and the one-line summary of its weakness is that a
# ternary naming a sibling's flag slips through, so this table is checked against a set.
_G5_FIELD_OWNERS = {
    "b_alloc_text": "b_alloc_st", "b_alloc_note": "b_alloc_st",
    "b_funding_text": "b_funding_st", "b_funding_note": "b_funding_st",
    "b_yield_text": "b_yield_st", "b_yield_note": "b_yield_st",
    "b_hurdle_text": "b_hurdle_st", "b_hurdle_note": "b_hurdle_st",
    "b_spread_text": "b_spread_st", "b_spread_note": "b_spread_st",
    "b_disloc_text": "b_disloc_st", "b_disloc_note": "b_disloc_st",
    "b_harvest_text": "b_harvest_st", "b_harvest_note": "b_harvest_st",
    "b_net_delta_text": "b_net_delta_st", "b_net_delta_note": "b_net_delta_st",
    "b_fee_amort_text": "b_fee_amort_st", "b_fee_amort_note": "b_fee_amort_st",
    "b_fee_target_text": "b_fee_target_st", "b_fee_target_note": "b_fee_target_st",
    "b_liq_price_text": "b_liq_price_st", "b_liq_price_note": "b_liq_price_st",
    "b_dist_liq_text": "b_dist_liq_st", "b_dist_liq_note": "b_dist_liq_st",
    "b_margin_mode_text": "b_margin_mode_st", "b_margin_mode_note": "b_margin_mode_st",
    "b_holding_text": "b_holding_st", "b_holding_note": "b_holding_st",
    "b_min_holding_text": "b_min_holding_st", "b_min_holding_note": "b_min_holding_st",
}

# Every verdict/colour the block derives, and so every one that must be guarded.
_G5_DERIVED_NAMES = frozenset({
    "status_color", "status_badge", "yield_color", "funding_color", "liq_color", "fee_color",
    "fee_bar_width", "margin_guard_color", "disloc_color", "harvest_color", "delta_color",
    "b_delta_posture", "b_time_text", "b_time_color", "b_reason_line", "b_coverage_line",
    "b_unmeasured_count", "b_is_stale", "b_age_days",
    # M-2: the threshold now in force, the label that names where it came from, and the verdict the
    # green rung rests on. M-7: the roster the coverage headline counts out of, and its length.
    "b_buffer_threshold_pct", "b_buffer_label", "b_buffer_guard_ok", "b_economics_states",
    "b_field_count",
})

# M-6: three verdicts talk about ALL the fields at once, so none of them can read one field's
# `measured` flag. They are exempt from the structural gate BY SHAPE, and the guard enforces the
# shape — each must read only `b_r` or another derived verdict and name no field-state variable, so
# the exemption can never be used to smuggle an ungated single-field verdict past the test.
_G5_CHAIN_VERDICTS = frozenset({"b_economics_states", "b_field_count", "b_coverage_line"})


def _g5_reads_measured_flag(text):
    """True when `text` reads a published reading's `measured` flag, rather than saying the word."""
    return re.search(r"\[[\"']measured[\"']\]", text) is not None


# ── M-5: a negative scan means CODE, not prose ──────────────────────────────────────────
# `_section_block` hands back the block verbatim, comments included, so a guard that greps it can
# be tripped by a one-line note quoting the stand-in it removed — the accident the G-5a implementer
# hit and documented — and can never be argued with. Both halves are pinned below: prose does not
# fail the guard, and the same text really in the code still does. Every G-5a block/helper scan
# reads `_g5_block_code()` or `_g5_code_only(...)` so this is structural, not discipline.
#
# Each case is paired with one line of code that genuinely contains the banned text, which is what
# makes the second half of the immunity test more than a restatement of the first.
_G5_STAND_IN_CASES = (
    ("or 8.0", "b_hurdle_text = raw or 8.0"),
    ("or 1.5", "b_unwind_text = raw or 1.5"),
    ("or 1000.0", "b_alloc_text = raw or 1000.0"),
    ("or 7", "b_min_holding_text = raw or 7"),
    ("or 2.40", "b_fee_text = raw or 2.40"),
    ("or 3.00", "b_target_text = raw or 3.00"),
    ("or 100.0", "b_liq_text = raw or 100.0"),
    ('"margin_mode", "crossed"', 'b_margin_mode_text = basis_data.get("margin_mode", "crossed")'),
)
_G5_PROSE_CLAIMS = (
    ("(Neutral)", 'b_delta_posture = "(Neutral)"'),
    ("Margin Mode Active", 'status_badge = "Margin Mode Active"'),
    ("IDLE (SCANNING YIELD & SPREAD)", 'status_badge = "IDLE (SCANNING YIELD & SPREAD)"'),
)


def _g5_code_only(src):
    """`src` with every `tokenize.COMMENT` and every docstring blanked out.

    Line count and the columns before each comment are preserved, so the result still parses at the
    same positions and `ast.get_source_segment` stays meaningful on it. A comment is prose: it
    cannot put a reading into the panel and it cannot take one out.

    Finding #4 added the docstring pass. Prose is prose wherever it lives, and a scan that a
    docstring can satisfy is a test that is true by construction: `assert "value" in reader_src and
    "updated_at" in reader_src` used to pass because the reader's own first docstring line says
    ``(value, updated_at)``, even if the body had selected no columns at all. Docstrings are blanked
    from the syntax tree (the first `Expr(Constant)` of a Module, ClassDef, FunctionDef or
    AsyncFunctionDef body) BEFORE the token pass, and blanked to spaces rather than deleted, so the
    line-and-column promise above survives. A fragment that will not parse keeps its docstrings and
    still fails or passes on the token pass exactly as it did: nothing silently becomes "code".
    """
    import ast
    import io
    import tokenize

    text = src if src.endswith("\n") else src + "\n"
    try:
        grid = [list(line) for line in text.splitlines(True)]
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef)):
                continue
            body = getattr(node, "body", None) or []
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                literal = body[0].value
                _g5_blank_region(grid, (literal.lineno, literal.col_offset),
                                 (literal.end_lineno, literal.end_col_offset))
        text = "".join("".join(row) for row in grid)
    except (SyntaxError, ValueError, AttributeError):
        pass

    lines = text.splitlines(True)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (SyntaxError, IndentationError, tokenize.TokenError) as exc:
        raise AssertionError(
            f"the scanned fragment is not valid Python ({exc}), so a comment-stripped scan of it "
            "would prove nothing about the shipped panel")
    for token in tokens:
        if token.type == tokenize.COMMENT:
            row, col = token.start[0], token.start[1]
            line = lines[row - 1]
            lines[row - 1] = line[:col] + ("\n" if line.endswith("\n") else "")
    stripped = "".join(lines)
    assert len(stripped.splitlines()) == len(text.splitlines()), (
        "comment stripping changed the line count, so scanned positions no longer match the block")
    return stripped


def _g5_blank_region(grid, start, end):
    """Blank the `[start, end)` source region of `grid` (a per-line character list) to spaces.

    Line breaks are never touched, so the caller's line count and every column before the region
    still line up with the fragment it came from.
    """
    (start_row, start_col), (end_row, end_col) = start, end
    for row in range(start_row, end_row + 1):
        line = grid[row - 1]
        from_col = start_col if row == start_row else 0
        to_col = end_col if row == end_row else len(line)
        for col in range(from_col, min(to_col, len(line))):
            if line[col] not in ("\n", "\r"):
                line[col] = " "


def _g5_banned_hits(src, banned):
    """The entries of `banned` that survive in `src` once its comments are stripped."""
    code = _g5_code_only(src)
    return [token for token in banned if token in code]


def _g5_block_code(path=None):
    """The carry panel's own source, comments removed: what every scan below actually reads."""
    return _g5_code_only(_g4_block("basis", DASHBOARD if path is None else path))


def _g5_helper_code(name="_basis_readings"):
    """One module-level helper's source, comments removed."""
    return _g5_code_only(_function_source(DASHBOARD, name))


def _g5_coverage_excluded():
    """The panel's own `_BASIS_COVERAGE_EXCLUDED`, read out of dashboard.py's syntax tree.

    The coverage headline counts the reader's keys minus this tuple, so the test has to take the
    same set from the same place or it is checking a transcription of the roster.
    """
    import ast

    src = DASHBOARD.read_text(encoding="utf-8")
    for node in ast.parse(src, filename=str(DASHBOARD)).body:
        targets = ([node.target] if isinstance(node, ast.AnnAssign)
                   else (node.targets if isinstance(node, ast.Assign) else []))
        if any(isinstance(t, ast.Name) and t.id == "_BASIS_COVERAGE_EXCLUDED" for t in targets):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError(
        "dashboard.py no longer defines _BASIS_COVERAGE_EXCLUDED, so the coverage headline has no "
        "stated rule about which entries are not economics fields")


class _G5RecordingColumn:
    """One `st.columns(...)` slot: a context manager that records into its owner's lists.

    Added for G-5b — the Markov deck lays its six KPI cards out with `st.columns(6)`, and the
    carry panel never needed it. Markdown emitted inside a column lands in the owner's own
    `markdown_calls`, so a render test reads the whole panel no matter which column spoke.
    """

    def __init__(self, owner):
        self._owner = owner

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def markdown(self, body, **kwargs):
        self._owner.markdown(body)

    def info(self, body, **kwargs):
        self._owner.info(body)

    def plotly_chart(self, fig, **kwargs):
        self._owner.plotly_chart(fig)

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            return None
        return _record


class _G5RecordingStreamlit:
    """Captures what the panel emits, so its f-string can be rendered without an app."""

    def __init__(self):
        self.markdown_calls = []
        self.infos = []
        self.charts = []

    def markdown(self, body, **kwargs):
        self.markdown_calls.append(str(body))

    def info(self, body, **kwargs):
        self.infos.append(str(body))

    def plotly_chart(self, fig, **kwargs):
        self.charts.append(fig)

    def columns(self, spec=3, **kwargs):
        count = spec if isinstance(spec, int) else len(spec)
        return [_G5RecordingColumn(self) for _ in range(int(count))]

    def cache_data(self, *args, **kwargs):
        """`@st.cache_data(...)` as the identity decorator, so a wrapped reader can be extracted.

        Added for G-5b fix #3: `query_system_state_row` is the only new SQL on the branch and was
        pinned by a text scan only, on the excuse that a `@st.cache_data`-wrapped function cannot be
        compiled out of the file. It can — the cache is a decorator, and this is the decorator, so
        what the test then calls is the shipped body, not a transcription of it. The 4s TTL and its
        keying are the real Streamlit's behaviour and stay unexercised; that is stated, not hidden.
        """
        def _identity(func):
            return func
        return _identity

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            return None
        return _record


def _g5_system_state_row(tmp_path, payload=_G5_LIVE_PAYLOAD, key="basis_trader_telemetry",
                         updated_at="2026-09-17T22:45:48.677606+00:00"):
    """Seed one real `system_state` row and hand back the JSON text sqlite stores for it.

    Written to a database in `tmp_path` — never to `trading_data.db`, which every test here opens
    read-only or not at all.

    G-5b added `key` and `updated_at`: the Markov deck's whole B-1 claim is about the row's own
    publish instant, so the fixture has to be able to set it rather than inherit the basis stamp.
    Re-seeding replaces the row instead of failing, because a G-5b test proves both halves of one
    claim (absent then measured) against the same tmp database.
    """
    import json as _json
    import sqlite3

    db = tmp_path / "telemetry_fixture.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS system_state (key TEXT PRIMARY KEY, value TEXT, "
                     "updated_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        raw = _json.dumps(payload)
        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                     (key, raw, updated_at))
        conn.commit()
        stored = conn.execute("SELECT value FROM system_state WHERE key = ?",
                              (key,)).fetchone()[0]
    finally:
        conn.close()
    return str(stored)


def _g5_panel_namespace(path, raw_value, now=None, extra_funcs=(), const_prefixes=("_BASIS_",),
                        extra_ns=None):
    """A namespace holding dashboard.py's own shipped helpers, without booting dashboard.py.

    Only stdlib imports, the `_BASIS_*` module constants and the pure renderers the block calls are
    compiled out of the file, in source order. `streamlit` is replaced by a recorder and
    `query_system_state` by the seeded payload text.

    G-5b added the keyword arguments rather than a second harness: the Markov deck needs its own
    pure reader, its own `_MARKOV_*` constants and a numpy/plotly pair the carry panel never
    touches, and every one of those is the same compile-out-of-the-file step. The defaults keep
    G-5a's call sites byte-for-byte as they were.
    """
    import ast

    wanted_funcs = {"_optional_float", "_published_number", "_published_boolean",
                    "_published_state", "_absence_note", "utc_to_sast", "_basis_readings"}
    wanted_funcs |= set(extra_funcs)
    safe_imports = {"json", "html", "pytz", "datetime", "typing", "time"}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    ns = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots = [(node.module or "").split(".")[0]]
        else:
            continue
        if roots and all(root in safe_imports for root in roots):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    for node in tree.body:
        targets = ([node.target] if isinstance(node, ast.AnnAssign)
                   else (node.targets if isinstance(node, ast.Assign) else []))
        if any(isinstance(t, ast.Name) and t.id.startswith(const_prefixes) for t in targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_funcs:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)

    fake = _G5RecordingStreamlit()
    ns["st"] = fake
    ns["query_system_state"] = lambda key, default="": (
        raw_value if key == "basis_trader_telemetry" else default)
    if now is not None and callable(ns.get("_basis_readings")):
        shipped = ns["_basis_readings"]
        ns["_basis_readings"] = lambda data, _s=shipped, _n=now: _s(data, now=_n)
    for name, value in (extra_ns or {}).items():
        ns[name] = value
    return ns, fake


def _g5_render_carry_panel(tmp_path, payload=None, path=None, raw=None):
    """Run the shipped carry-panel block for one payload and return the HTML it published.

    With a `tmp_path` the payload is seeded as a real `system_state` row and read back out of
    sqlite first, so the JSON the panel parses went through a storage round-trip; without one it is
    dumped straight. Either way the block itself is dashboard.py's own source.
    """
    import json as _json

    path = DASHBOARD if path is None else path
    if raw is None:
        body = _G5_LIVE_PAYLOAD if payload is None else payload
        raw = (_g5_system_state_row(tmp_path, body) if tmp_path is not None
               else _json.dumps(body))
    ns, fake = _g5_panel_namespace(path, raw, now=_G5_FIX_NOW)
    block = _g4_block("basis", path)
    exec(compile(block, f"{path.name} carry panel", "exec"), ns)
    return "\n".join(fake.markdown_calls), fake


def _g5_readings(payload, path=None):
    """`_basis_readings`, compiled out of the shipped file, called on one payload."""
    return _g4_helper("_basis_readings", path)(payload, now=_G5_FIX_NOW)


# Group 1 — the live payload, run through the shipped read path.
@pytest.mark.parametrize("field", _G5_ABSENT_IN_LIVE)
def test_g5a_the_live_payload_leaves_the_carry_economics_unmeasured(field):
    """A parked engine published no book, so the panel may publish none either.

    The five-key DISABLED payload is read exactly as the shipped helper reads it. `net_delta = 0.0`
    IS carried and must come back measured; everything with a stand-in default today must come back
    absent with a reason naming the field.
    """
    state = _g5_readings(dict(_G5_LIVE_PAYLOAD))[field]
    assert state["measured"] is False, (_G5_LIVE_PAYLOAD, field, state)
    if field in _G5_TEXT_FIELDS:
        assert "value" not in state, f"{field} is published as text but carries a value: {state}"
        assert state["text"] == "not measured", (field, state)
    else:
        assert state["value"] is None, f"{field} was absent but still carries a value: {state}"
    assert field in state["reason"], f"the absence for {field} does not name itself: {state}"


def _g5_positional_get_defaults(src):
    """Every two-argument `.get(key, default)` in `src`, read out of its syntax tree.

    A defaulted `.get()` cannot tell "the engine recorded nothing" from "the engine recorded X", so
    in a reader that is supposed to publish measurements-or-absences it publishes the default as if
    it were a reading. Truthiness is the same trap one step on: `a or b` drops a measured 0.0.
    """
    import ast

    hits = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "get" and len(node.args) >= 2:
            hits.append(ast.get_source_segment(src, node) or "")
    return hits


def test_g5a_the_live_payload_keeps_net_delta_a_measured_zero():
    """`net_delta = 0.0` is the one number in the live payload. It must stay a reading.

    This is the regression the round is judged on: an absence is not a zero and a zero is not an
    absence. The producer is one file with two writers and two key names for the same figure — the
    active heartbeat publishes `net_delta_usd` (`basis_trader.py:823`, computed at `:793`), the
    DISABLED parking path publishes `net_delta` (`:920`) — so the reader must name both, and it may
    not reach either through a default that answers for a key the engine never sent.
    """
    readings = _g5_readings(dict(_G5_LIVE_PAYLOAD))
    assert readings["net_delta"] == {"value": 0.0, "measured": True, "reason": ""}, readings["net_delta"]
    # I-2: both guards are scoped to the helper's own source. The whole-file ban this replaces
    # forbade `net_delta_usd`, which is the key the ACTIVE engine publishes, so it pinned the
    # stub-only read as the contract and made the correct fix fail.
    code = _g5_helper_code()
    assert '"net_delta_usd"' in code, (
        "the carry reader no longer names the net-delta key the active engine publishes")
    assert '"net_delta"' in code, (
        "the carry reader no longer names the net-delta key the parked engine publishes")
    assert _g5_positional_get_defaults(code) == [], (
        f"the carry reader defaults a `.get()` and so publishes a stand-in for a key it never "
        f"measured: {_g5_positional_get_defaults(code)}")


# I-1 — the dual-key read, stated as behaviour. Each case is one arm of the real two-writer
# contract; none of them is satisfiable by a reader that looks at only one key.
def test_g5a_the_active_engine_loses_nothing_when_its_delta_is_a_measured_zero():
    """`{"net_delta_usd": 0.0}` is what a running engine publishes for a flat twin book."""
    state = _g5_readings({"net_delta_usd": 0.0})["net_delta"]
    assert state == {"value": 0.0, "measured": True, "reason": ""}, state


def test_g5a_the_engine_key_wins_when_both_shapes_are_published():
    """Precedence is stated, not incidental: the computed reading beats the parked placeholder."""
    state = _g5_readings({"net_delta_usd": -0.75, "net_delta": 99.0})["net_delta"]
    assert state["measured"] is True, state
    assert state["value"] == -0.75, state


def test_g5a_the_parked_placeholder_is_read_when_the_engine_key_is_absent():
    state = _g5_readings({"net_delta": 99.0})["net_delta"]
    assert state["measured"] is True, state
    assert state["value"] == 99.0, state


def test_g5a_a_payload_with_neither_delta_key_reports_the_absence_in_its_own_words():
    state = _g5_readings({})["net_delta"]
    assert state == {"value": None, "measured": False,
                     "reason": "no net delta in basis trader telemetry"}, state


def test_g5a_a_measured_zero_under_the_engine_key_renders_as_a_reading(tmp_path):
    """The panel half of I-1: a live engine's measured zero must not print as an absence beside a
    badge that asserts delta-neutrality."""
    html, _fake = _g5_render_carry_panel(
        tmp_path, payload={"state": "ACTIVE_DELTA_NEUTRAL", "net_delta_usd": 0.0})
    assert "$+0.00" in html, html[:2500]
    assert "Neutral" in html, html[:2500]
    assert "posture not measured" not in html, "the engine's own delta lost its posture"
    assert "(no net delta in basis trader telemetry)" not in html, \
        "a measured net delta is still published as an absence"


@pytest.mark.parametrize("field,expected", [("state", "DISABLED"),
                                            ("reason", "Temporarily parked for account segregation")])
def test_g5a_the_live_payload_publishes_the_state_and_the_engines_own_reason(field, expected):
    state = _g5_readings(dict(_G5_LIVE_PAYLOAD))[field]
    assert state["measured"] is True, (field, state)
    assert state["text"] == expected, (field, state)


def test_g5a_the_age_of_the_live_telemetry_is_measured_from_its_own_timestamp():
    """35 days old is a fact about the payload, computed from the payload's timestamp."""
    readings = _g5_readings(dict(_G5_LIVE_PAYLOAD))
    assert readings["timestamp"]["measured"] is True, readings["timestamp"]
    assert readings["age_days"] is not None and 34.0 <= readings["age_days"] <= 36.0, \
        readings["age_days"]
    assert readings["stale"] is True, readings["age_days"]
    fresh = _g5_readings({**_G5_LIVE_PAYLOAD,
                          "timestamp": _G5_FIX_NOW.isoformat()})
    assert fresh["age_days"] <= 0.01 and fresh["stale"] is False, fresh["age_days"]
    broken = _g5_readings({**_G5_LIVE_PAYLOAD, "timestamp": "not a date"})
    assert broken["age_days"] is None and broken["stale"] is False, broken


def test_g5a_basis_readings_survives_a_payload_that_is_not_a_dict():
    """`json.loads` can hand back a list or a string; a panel that raises publishes nothing."""
    for junk in (None, [], "DISABLED", 7, True, {}):
        readings = _g5_readings(junk)
        assert readings["net_delta"]["measured"] is False, (junk, readings["net_delta"])
        assert readings["age_days"] is None and readings["stale"] is False, junk


# Group 2 — a measured zero is never an absence, for every field converted.
@pytest.mark.parametrize("field", sorted(_G5_ALL_FIELDS))
def test_g5a_a_measured_zero_stays_a_reading_in_the_carry_panel(field):
    """Every stand-in here was `... or 0.0` or `... or <default>`, which cannot tell a recorded
    zero from no record. Feed a real 0.0 to each field and it must come back as a reading."""
    state = _g5_readings({field: 0.0})[field]
    assert state["measured"] is True, (field, state)
    if field in _G5_TEXT_FIELDS:
        assert state["text"] == "0.0", (field, state)
    else:
        assert state["value"] == 0.0, (field, state)


def test_g5a_a_floor_of_measured_zeros_renders_no_absences_at_all(tmp_path):
    """With every field measured at zero, the panel must print zeros, not `not measured`."""
    payload = {field: 0.0 for field in _G5_ALL_FIELDS}
    payload["state"] = "ACTIVE_DELTA_NEUTRAL"
    payload["timestamp"] = _G5_FIX_NOW.isoformat()
    html, _fake = _g5_render_carry_panel(tmp_path, payload=payload)
    assert "not measured" not in html, html[:2500]
    assert "$0 USDT" in html, html
    assert "$+0.00" in html, html
    assert "+0.0000%" in html, html
    assert "$+0.0000 USD" in html, html
    assert "+0.0% Buffer" in html, html
    assert "panel scale: 125%" in html, html


# Group 3 — no numeric stand-in survives in this block.
@pytest.mark.parametrize("stand_in", [case[0] for case in _G5_STAND_IN_CASES])
def test_g5a_no_numeric_stand_in_survives_in_the_carry_panel(stand_in):
    block = _g5_block_code()
    assert stand_in not in block, f"the carry panel still invents a reading with {stand_in!r}"


@pytest.mark.parametrize("stand_in,code_line", _G5_STAND_IN_CASES + _G5_PROSE_CLAIMS)
def test_g5a_a_comment_neither_trips_nor_hides_a_negative_scan(stand_in, code_line):
    """M-5, both halves, or the guard is just weaker than it was.

    Prose about a stand-in is not a stand-in: a note that quotes `or 8.0` while explaining why it
    went must not fail a guard over correct code. And the same text really in the code must still
    fail it, even with a comment beside it — immunity to prose is not a licence to hide one.
    """
    prose = (f"def f():\n    # the panel used to publish {stand_in} here, and {stand_in} is "
             f"still what the note says\n    return 1\n")
    assert _g5_banned_hits(prose, [stand_in]) == [], (
        f"a comment quoting {stand_in!r} failed a negative scan against code that does not "
        "contain it")
    real = f"def f():\n    {code_line}\n    return 1\n"
    assert _g5_banned_hits(real, [stand_in]) == [stand_in], (
        f"{stand_in!r} is in the code and the stripped scan no longer sees it")
    hidden = f"def f():\n    {code_line}  # {stand_in}\n    return 1\n"
    assert _g5_banned_hits(hidden, [stand_in]) == [stand_in], (
        f"a comment beside {stand_in!r} in the code hid the stand-in from the scan")


def test_g5a_the_carry_panel_reads_through_the_publishers_and_the_scan_is_not_vacuous():
    """Positive side of the stand-in scan, plus the floor that makes the negative side mean
    something: a block that extracted the wrong region would be short and key-free.

    The payload keys are allowed to appear in exactly two places — the panel block and the pure
    `_basis_readings` helper it now calls. Anywhere else is a second reader of the same telemetry
    that this test is not looking at.
    """
    block = _g5_block_code()
    assert "_basis_readings(basis_data)" in block, (
        "the carry panel does not read its payload through the published-reading helper")
    assert block.count("_absence_note(") >= 12, (
        f"the carry panel explains only {block.count('_absence_note(')} absences")
    assert len(block.splitlines()) >= 150, (
        f"the extracted carry panel is {len(block.splitlines())} lines, so this is not the panel")
    helper = _g5_helper_code()
    assert helper, "dashboard.py has no _basis_readings to route the panel through"
    # M-5: every side of this subtraction is code-only, so a note that mentions a key cannot be
    # counted as a second reader of it — and a second reader cannot mention it in prose instead.
    src = _g5_code_only(DASHBOARD.read_text(encoding="utf-8"))
    for key in ("capital_allocated_usd", "hurdle_rate_pct", "unwind_hurdle_pct",
                "distance_to_liq_pct", "fee_hurdle_target_usd", "basis_dislocation_pct",
                "fee_amortized_pct", "liquidation_price"):
        outside = src.count(f'"{key}"') - block.count(f'"{key}"') - helper.count(f'"{key}"')
        assert outside == 0, (
            f"{key} is read {outside} time(s) outside the carry panel and its reader helper, so "
            "the block this test scanned is not the only place the figure is invented")


def test_g5a_the_dead_stand_in_reads_are_gone():
    """Five fields were assigned and never used. A stand-in feeding nothing is still a stand-in.

    I-2: the scope of this ban is the panel and its reader, not the file. All five are keys the
    ACTIVE heartbeat publishes (`basis_trader.py:814-815` for the two prices, `:825`
    `has_active_position`, `:827` `unrealized_pnl_usd`, `:830` `roundtrip_fee_usd`), so a whole-file
    ban would block a later round from showing a real figure and its message claimed those keys
    exist nowhere. What is banned here is exactly what was wrong: this panel reading a published key
    into a variable that renders nowhere. Same subtraction the key scan above already uses.
    """
    src = _g5_code_only(DASHBOARD.read_text(encoding="utf-8"))
    block = _g5_block_code()
    helper = _g5_helper_code()
    for key in ("unrealized_pnl_usd", "has_active_position", "spot_price", "perp_price",
                "roundtrip_fee_usd"):
        outside = src.count(f'"{key}"') - block.count(f'"{key}"') - helper.count(f'"{key}"')
        assert outside == 0, (
            f"{key} is read {outside} time(s) outside the carry panel and its reader helper — a "
            "dead stand-in read in this panel, or a second reader this test never scanned")


# Group 4 — an absence never wears a measured colour.
@pytest.mark.parametrize("signal_colour", ["#00ff66", "#ff9900", "#ff003c"])
def test_g5a_the_live_payload_prints_no_signal_colour_for_a_field_it_never_measured(signal_colour):
    """Against today's payload the only readings are state, reason and net delta, so the green
    all-clear, the amber warning and the red alarm all belong to numbers that do not exist. The
    Funding Harvested card used to be fixed `#00ff66` and the liquidation guard used to be bought
    green by a stand-in 100.0."""
    html, _fake = _g5_render_carry_panel(None)
    assert signal_colour not in html, (signal_colour, html[:2500])
    assert "#94a3b8" in html, html[:2500]


def test_g5a_the_funding_harvested_card_no_longer_hardcodes_the_all_clear():
    block = _g5_block_code()
    assert "color: #00ff66; font-size: 0.95rem" not in block, (
        "a card still wears a hardcoded green instead of a measured colour")
    assert "harvest_color" in block, "the Funding Harvested card has no state-driven colour"
    assert 'if b_harvest_st["measured"] else "#94a3b8"' in block, (
        "the Funding Harvested colour does not fall back to absence grey")


def test_g5a_the_thresholds_are_named_as_thresholds():
    """125% and 20% are the panel's own scale, not measurements — and they may not sit beside a
    stand-in reading so closely that they read as part of it."""
    block = _g5_block_code()
    assert "panel scale: {_BASIS_FEE_SCALE_PCT:.0f}%" in block, (
        "the 125% fee scale is not labelled as the panel's own threshold")
    assert "panel min-buffer threshold: {_BASIS_MIN_BUFFER_PCT:.0f}%" in block, (
        "the 20% min buffer is not labelled as the panel's own threshold")
    assert re.search(r"fee_bar_width\s*=\s*min\([^)]*b_fee_amort_st\[[^\]]*value[^\]]*\]",
                     block, re.S), (
        "the fee bar is not computed from the measured amortisation")
    html, _fake = _g5_render_carry_panel(None)
    assert "0.0% / 125%" not in html, html


# Group 5 — the three fabricated prose claims.
@pytest.mark.parametrize("claim", [case[0] for case in _G5_PROSE_CLAIMS])
def test_g5a_the_fabricated_prose_claims_are_gone(claim):
    block = _g5_block_code()
    assert claim not in block, f"the carry panel still asserts {claim!r} regardless of the payload"


def test_g5a_the_replacements_are_present_and_reachable():
    """Absence checks pass on a panel that deleted the card entirely, so the new wording is pinned
    in the block AND in the HTML the live payload actually renders."""
    block = _g5_block_code()
    for reachable in ("STATE NOT MEASURED", "b_delta_posture", "Engine-reported reason",
                      "Margin mode:", "STALE TELEMETRY"):
        assert reachable in block, f"the carry panel has no {reachable!r} to fall back on"
    html, _fake = _g5_render_carry_panel(None)
    assert "STATE PUBLISHED" not in html and "⏸️ PARKED / DISABLED" in html, html[:2000]
    assert "posture not measured" not in html, "the published delta lost its posture"
    assert "Engine-reported reason: <b style='color: #e2e8f0;'>" \
           "Temporarily parked for account segregation</b>" in html, html
    assert "35d old · ⚠️ STALE TELEMETRY" in html, html
    # M-7: the headline is now counted out of `_basis_readings`' own key set. It read "15 of 18"
    # while the tuple and the helper were hand-shadows of each other; the engine's published
    # minimum buffer joined the roster, so the honest figure for today's five-key payload is 16 of
    # 19. Pinned against the helper rather than against a literal, so it cannot go stale again.
    roster = [key for key in _g5_readings(dict(_G5_LIVE_PAYLOAD))
              if key not in _g5_coverage_excluded()]
    assert sorted(roster) == sorted(set(_G5_ALL_FIELDS) - {"timestamp"}), roster
    assert len(roster) == 19, roster
    assert f"16 of {len(roster)} economics fields" in html, html
    # And the state that was never published really does reach the absence badge.
    no_state = dict(_G5_LIVE_PAYLOAD)
    no_state.pop("state")
    html2, _fake2 = _g5_render_carry_panel(None, payload=no_state)
    assert "STATE NOT MEASURED" in html2, html2[:2000]
    assert "#94a3b8" in html2, html2[:2000]


def test_g5a_an_unrecognised_published_state_is_not_renamed():
    """A state the panel does not know was still measured, so it is shown as published — never
    quietly rewritten as IDLE, and never dressed in the cyan the stand-in used."""
    html, _fake = _g5_render_carry_panel(
        None, payload={**_G5_LIVE_PAYLOAD, "state": "REBALANCING_HALTED"})
    assert "STATE PUBLISHED, UNRECOGNISED: REBALANCING_HALTED" in html, html[:2000]


# Group 5b — M-1 to M-4: every number the panel shows says whose number it is.
def test_g5a_the_neutral_posture_names_the_threshold_it_was_judged_against():
    """M-1: the header comment claims the panel's own thresholds are labelled where they are shown.
    Two of them were; a bare `· Neutral` was the third, and it read as a property of the book rather
    than as a tolerance this HUD picked. The label is interpolated from the constant the verdict
    actually compares against, so the prose cannot drift away from the number.

    The parked payload's `net_delta: 0.0` stays a reading throughout: a recorded zero is a fact.
    """
    block = _g5_block_code()
    assert '"Neutral (±${_BASIS_NEUTRAL_TOL_USD:.2f} panel threshold)"' in block, (
        "the posture no longer names the panel threshold it was judged against")
    assert "(±$1.00" not in block, (
        "the tolerance is typed into the prose instead of read from the constant that enforces it")
    html, _fake = _g5_render_carry_panel(None)
    assert "Neutral (±$1.00 panel threshold)" in html, html[:2500]
    assert "$+0.00" in html, "the published zero stopped being a reading"
    html2, _fake2 = _g5_render_carry_panel(
        None, payload={**_G5_LIVE_PAYLOAD, "net_delta": 40.0})
    assert "Not neutral (±$1.00 panel threshold)" in html2, html2[:2500]


@pytest.mark.parametrize("published_min,expected_label", [
    (20.0, "engine min-buffer threshold: 20.0%"),
    (0.0, "engine min-buffer threshold: 0.0%"),
])
def test_g5a_the_buffer_ladder_is_driven_by_the_threshold_the_engine_publishes(
        published_min, expected_label):
    """M-2: the guard went green at a hardcoded 30 % while printing "threshold: 20 %", and the
    engine's own `min_liq_distance_pct` (`basis_trader.py:836`, the same figure it compares its
    distance against at `:315`) was never read. The published minimum is now the amber rung and the
    label names where it came from. A threshold of 0.0 is a threshold the engine enforces, so it
    stays a reading rather than being read as an absence."""
    html, _fake = _g5_render_carry_panel(
        None, payload={**_G5_LIVE_PAYLOAD, "distance_to_liq_pct": 24.5,
                       "min_liq_distance_pct": published_min})
    assert expected_label in html, html[:2500]


def test_g5a_a_book_under_the_published_minimum_is_not_printed_as_clear():
    """The threshold has to be operative, not decorative: 24.5 % of buffer against an engine that
    asks for 40 % is an alarm, and it cannot be bought by a panel constant."""
    html, _fake = _g5_render_carry_panel(
        None, payload={**_G5_LIVE_PAYLOAD, "distance_to_liq_pct": 24.5,
                       "min_liq_distance_pct": 40.0})
    assert "engine min-buffer threshold: 40.0%" in html, html[:2500]
    assert "#ff003c" in html, "a book 15 points under the engine's own minimum wears no alarm"


def test_g5a_the_guard_falls_back_to_the_panels_own_number_and_says_so():
    """Nothing published a threshold, so the panel's own stands in — and the label says which of the
    two the operator is looking at."""
    html, _fake = _g5_render_carry_panel(None)
    assert ("panel min-buffer threshold: 20% (panel's own — the engine published no threshold)"
            in html), html[:2500]


def test_g5a_the_buffer_ladder_has_no_bare_rungs():
    block = _g5_block_code()
    for bare in (">= 30.0", ">= 20.0"):
        assert bare not in block, f"the liquidation ladder still hardcodes {bare}"
    assert "_BASIS_SAFE_BUFFER_PCT" in block, "the green rung has no name to point at"
    assert '"min_liq_distance_pct"' in _g5_helper_code(), (
        "the engine's published minimum buffer is no longer read as a measurement")


def test_g5a_the_harvest_line_explains_every_absence_it_shows():
    """M-3: `Harvested: … · Target: … · Hold: … / …` rendered three published fields as a bare
    `not measured` while every other field on the panel carried its reason, so the branch's
    convention was 13-for-18 and the field-owner harness had no entry to defend them."""
    block = _g5_block_code()
    for note in ("b_fee_target_note", "b_holding_note", "b_min_holding_note"):
        assert f"{note} = " in block, f"the harvest line has no {note} any more"
        assert f"{{{note}}}" in block, f"{note} is built but never rendered"
    html, _fake = _g5_render_carry_panel(None)
    for key in ("fee_hurdle_target_usd", "holding_days", "min_holding_days"):
        assert f"(no {key} in basis trader telemetry)" in html, (key, html[:2500])


def test_g5a_the_hurdle_parenthetical_wears_the_absence_grey():
    """M-4: `(Hurdle: not measured …)` was painted the parked-DISABLED slate, so an absence wore a
    colour that means something else on this panel."""
    block = _g5_block_code()
    assert 'color:#94a3b8;">(Hurdle:' in block, "the hurdle parenthetical is not absence grey"
    assert 'color:#64748b;">(Hurdle:' not in block, (
        "the hurdle parenthetical still wears the parked slate")


def test_g5a_the_coverage_headline_counts_the_readers_own_fields():
    """M-7: the headline used to be built from a hand-kept 18-tuple that had to shadow
    `_basis_readings`. Add a field to the reader, forget it in the tuple, and the panel prints a
    number about itself that is no longer true — this branch's own defect class, in its own caption.

    The count for today's five-key payload moved from "15 of 18" to "16 of 19" when the engine's
    published minimum buffer joined the roster: one more field nobody published, counted honestly.
    """
    block = _g5_block_code()
    assert "b_r.items()" in block and "_BASIS_COVERAGE_EXCLUDED" in block, (
        "the coverage roster is not derived from the reader any more")
    assert "b_state_st, b_reason_st, b_margin_mode_st" not in block, (
        "the coverage roster is a hand-shadow of the reader again")
    excluded = set(_g5_coverage_excluded())
    assert {"age_days", "stale"} <= excluded, excluded
    assert excluded <= set(_g5_readings({})), excluded
    html, _fake = _g5_render_carry_panel(None)
    assert "16 of 19 economics fields" in html, html[:2500]


def test_g5a_a_running_but_scanning_engine_gets_an_honest_badge():
    """M-8: IDLE is the state the engine holds for most of its life (`basis_trader.py:155`, `:161`,
    `:410`, `:477`, `:503`, `:691`), and the panel shrugged at it as `STATE PUBLISHED, UNRECOGNISED`.
    It has its own entry now — and it is not green, because an engine with no twin position has not
    earned an all-clear."""
    html, _fake = _g5_render_carry_panel(None, payload={**_G5_LIVE_PAYLOAD, "state": "IDLE"})
    assert "IDLE — ENGINE RUNNING, NO TWIN POSITION OPEN" in html, html[:2000]
    assert "STATE PUBLISHED, UNRECOGNISED" not in html, html[:2000]
    assert "border-left: 3px solid #64748b" in html, "IDLE was promoted to a measured colour"


def test_g5a_an_unknown_state_still_shows_as_published():
    """The IDLE entry must not have become a catch-all: a state this panel has never heard of is
    still shown as published, never quietly renamed."""
    html, _fake = _g5_render_carry_panel(
        None, payload={**_G5_LIVE_PAYLOAD, "state": "SOMETHING_ELSE"})
    assert "STATE PUBLISHED, UNRECOGNISED: SOMETHING_ELSE" in html, html[:2000]


def test_g5a_the_panel_still_ignores_the_published_status_key():
    """`status` is a claim the engine makes about itself (`basis_trader.py:812`) that this panel has
    never rendered and does not start rendering now: the badge reads `state`, and one field gets one
    word rather than two opinions about the same engine."""
    assert '"status"' not in _g5_helper_code(), (
        "the reader started reading `status`, which renders nowhere on this panel")
    html, _fake = _g5_render_carry_panel(
        None, payload={**_G5_LIVE_PAYLOAD, "state": "IDLE", "status": "SCANNING_YIELD"})
    assert "SCANNING_YIELD" not in html, html[:2000]
    assert "IDLE — ENGINE RUNNING, NO TWIN POSITION OPEN" in html, html[:2000]


def test_g5a_the_disabled_engine_keeps_its_dimmed_colour_and_is_not_promoted():
    """`#64748b` is the parked colour; a stale row must not be painted the live cyan."""
    html, _fake = _g5_render_carry_panel(None)
    assert "border-left: 3px solid #64748b" in html, html[:2000]
    assert "IDLE" not in html, html[:2000]


# Group 6 — the AST guards: derived verdicts guarded, field values keyed on their own state.
def _function_source(path, name):
    """The source of one module-level function, read out of the file's own syntax tree."""
    import ast

    src = path.read_text(encoding="utf-8")
    for node in ast.parse(src, filename=str(path)).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node) or ""
    raise AssertionError(f"{path.name} has no module-level function named {name}")


def _g5_enclosing_measured_flag_guard(node, parents, block):
    """Like `_g4_enclosing_measured_guard`, but the gate has to READ a flag, not mention one.

    G-4's version accepts any `measured` in the enclosing `If` test, which a message can supply. For
    this panel the difference mattered: the coverage headline's own string ends "as not measured, not
    as readings", so the word in a message was gating a verdict.
    """
    import ast

    ancestor = parents.get(id(node))
    while ancestor is not None:
        if isinstance(ancestor, ast.If) and _g5_reads_measured_flag(
                ast.get_source_segment(block, ancestor.test) or ""):
            return True
        ancestor = parents.get(id(ancestor))
    return False


def test_g5a_every_derived_verdict_in_the_carry_panel_guards_on_measured():
    """A verdict computed from a reading that can be absent either raises TypeError (the panel then
    publishes NOTHING, which is worse than any stand-in) or publishes an all-clear nobody measured.

    Ancestor-walk over the enclosing `If` gates, the approach `test_g4_the_radar_banner_...` uses;
    its known limitation — a bundled gate also has `measured` in its test — is why the field values
    are checked separately and more strictly by `test_g5a_the_carry_panel_keys_each_field_on_its_own_state`.

    M-6: the evidence is now a flag read out of the comment-stripped block, so neither a note nor a
    message can gate a verdict. The three verdicts that count every field at once are exempt by
    shape, and the shape is asserted here rather than assumed in a list.
    """
    import ast

    block = _g5_block_code()
    tree = ast.parse(block, filename="dashboard.py carry panel")
    parents = _g4_parent_map(tree)
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if not (isinstance(target, ast.Name) and target.id in _G5_DERIVED_NAMES):
                continue
            seen.add(target.id)
            segment = ast.get_source_segment(block, node) or ""
            if _g5_reads_measured_flag(segment) or _g5_enclosing_measured_flag_guard(
                    node, parents, block):
                continue
            assert target.id in _G5_CHAIN_VERDICTS, (
                f"the carry panel derives {target.id} from a reading that can be absent: "
                f"{segment[:160]}")
            names_state = sorted(set(re.findall(r"\bb_[a-z_]*_st\b", segment)))
            assert not names_state, (
                f"{target.id} claims the chain exemption but names field state {names_state}, so it "
                "is a single-field verdict and has to be gated like every other one")
    missing = sorted(_G5_DERIVED_NAMES - seen)
    assert not missing, (
        f"the carry panel no longer derives {missing}, so this guard has nothing to check")


def test_g5a_a_prose_measured_gates_nothing_any_more():
    """M-6, proved rather than asserted: the coverage headline passes on its gate and its chain, not
    on the words it prints.

    The headline still says "not measured" — that is the honest sentence an operator reads — but it
    is no longer evidence of anything. The count underneath it is: it reads each field's own flag,
    and the `if` lets the line through only when something really went unpublished.
    """
    import ast

    block = _g5_block_code()
    tree = ast.parse(block, filename="dashboard.py carry panel")
    segments = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            segments.setdefault(node.targets[0].id, ast.get_source_segment(block, node) or "")
    headline = segments["b_coverage_line"]
    assert "not measured" in headline, "the honest headline message changed"
    assert not _g5_reads_measured_flag(headline), (
        "the coverage headline now reads a flag itself, so the chain exemption no longer describes "
        "it and the exemption list is stale")
    assert _g5_reads_measured_flag(segments["b_unmeasured_count"]), (
        "the count the headline is built from no longer reads each field's own flag, so the "
        "exemption would be hiding an unguarded verdict")
    assert "if b_unmeasured_count:" in block, (
        "the headline has no gate of its own, so it can print a coverage claim nobody measured")


@pytest.mark.parametrize("field,own_key", sorted(_G5_FIELD_OWNERS.items()))
def test_g5a_the_carry_panel_keys_each_field_on_its_own_state(field, own_key):
    """A missing hurdle may not make the panel disown the yield it measured, and a measured yield
    may not be published under the hurdle's flag.

    Stronger than the G-4 radar guard it is copied from: there, a ternary keyed on its own flag plus
    a sibling's still passed. Here the set of state variables a field's own line may name must be
    exactly one — its own.
    """
    import ast

    block = _g5_block_code()
    tree = ast.parse(block, filename="dashboard.py carry panel")
    parents = _g4_parent_map(tree)
    assigns = [node for node in ast.walk(tree)
               if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == field for t in node.targets)]
    assert assigns, f"the carry panel no longer publishes {field} at all"
    keyed = re.compile(own_key + r"\[[^\]]*measured[^\]]*\]")
    for node in assigns:
        segment = ast.get_source_segment(block, node) or ""
        named = set(re.findall(r"\bb_[a-z_]*_st\b", segment))
        assert named == {own_key}, (
            f"{field} is keyed on {sorted(named)}; a published field value may read only its own "
            f"state ({own_key}): {segment[:200]}")
        assert keyed.search(segment), (
            f"the carry panel never keys {field} on {own_key}'s own measured flag: {segment[:200]}")
        if field.endswith("_note"):
            # The parenthetical line: its whole job is the absence, so it must render the reason.
            assert "_absence_note(" in segment and "reason" in segment, (
                f"{field} publishes no reason for its own absence: {segment[:200]}")
        else:
            assert "not measured" in segment, (
                f"{field} has no absence for when its own state was never measured: "
                f"{segment[:200]}")
        assert not _g5_enclosing_gate_is_bundled(node, parents, block, own_key), (
            f"{field} is published under a gate on another field's state")


def _g5_enclosing_gate_is_bundled(node, parents, block, own_key, prefix="b"):
    """True when an enclosing `If` gate reads a state variable other than `own_key`.

    The G-4 radar guard (`test_g4_the_radar_banner_keys_each_hawkes_text_on_its_own_state`) only
    excluded one named sibling, so a ternary keyed on its own flag *plus* a sibling's still passed.
    This one asks whether any other field's state is in scope at all.

    `prefix` is the panel's own state-variable prefix — `b_` for the carry panel, `m_` for the
    Markov deck — so the deck reuses this gate instead of forking a second copy of it.
    """
    import ast

    state_pattern = r"\b%s_[a-z_]*_st\b" % prefix
    ancestor = parents.get(id(node))
    while ancestor is not None:
        if isinstance(ancestor, ast.If):
            gate = ast.get_source_segment(block, ancestor.test) or ""
            named = set(re.findall(state_pattern, gate))
            if named and named != {own_key}:
                return True
        ancestor = parents.get(id(ancestor))
    return False


def test_g5a_the_combined_yield_verdict_is_legitimately_bundled():
    """`yield_color` compares the yield against two thresholds, so it genuinely needs all three
    states — the one place in the panel where a bundled gate is the right answer, pinned so a later
    reader does not un-bundle it into `b_yield_st` alone and compare against a None hurdle."""
    import ast

    block = _g5_block_code()
    tree = ast.parse(block, filename="dashboard.py carry panel")
    parents = _g4_parent_map(tree)
    gates = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == "yield_color":
            ancestor = parents.get(id(node))
            while ancestor is not None:
                if isinstance(ancestor, ast.If):
                    gates.append(ast.get_source_segment(block, ancestor.test) or "")
                ancestor = parents.get(id(ancestor))
    assert any({"b_yield_st", "b_hurdle_st", "b_unwind_st"}
               <= set(re.findall(r"\bb_[a-z_]*_st\b", gate)) for gate in gates), (
        "the yield verdict is no longer gated on every threshold it compares against: "
        f"{gates}")
    assert re.search(r'yield_color\s*=\s*"#94a3b8"', block), (
        "the bundled yield verdict has no absence colour for when it cannot be formed")


# Group 7 — the panel renders. This is the claim the whole rewrite is checked against.
def test_g5a_the_live_payload_renders_parked_economics_instead_of_a_carry_book(tmp_path):
    """Nothing here re-asserts a string the panel happens to contain: the shipped f-string is
    executed against the live payload, so a `{x['value']}` formatting a None would raise here.

    The payload reaches the panel the way the fleet sends it: written into a `system_state` row in a
    fixture database and read back out of sqlite.
    """
    raw = _g5_system_state_row(tmp_path)
    import json as _json
    assert _json.loads(raw) == _G5_LIVE_PAYLOAD, raw
    html, fake = _g5_render_carry_panel(tmp_path, raw=raw)
    assert len(fake.markdown_calls) == 1, "the panel published an unexpected number of blocks"
    assert "None" not in html, html[:2500]
    assert "$1,000" not in html and "1,000.00" not in html, "an invented allocation is published"
    assert "(Hurdle: 8.0%)" not in html, "the invented 8% hurdle is published"
    assert "+100.0% Buffer" not in html, "the invented 100% liquidation buffer is published"
    assert "CROSSED" not in html, "an invented margin mode is published"
    assert "🛡️" not in html, "a guard the engine never armed is still claimed"
    assert "0.0% / 125%" not in html, "the fee line still reads as a measurement"
    assert "$+0.00" in html, "the one measured reading lost its digits"
    assert "not measured" in html and "(no capital_allocated_usd in basis trader telemetry)" in html
    assert "2026-09-18 00:45:48" in html, "the payload's own timestamp is not rendered"


def test_g5a_an_empty_payload_still_publishes_the_standing_by_message(tmp_path):
    """`if basis_data:` is truthy for a five-key dict, which is what let the stand-ins through. The
    genuinely-empty case must keep its own branch rather than rendering the deck."""
    _html, fake = _g5_render_carry_panel(tmp_path, payload={})
    assert _html == "", _html
    assert len(fake.infos) == 1 and "Cash-and-Carry Basis Trader standing by" in fake.infos[0], \
        fake.infos


def test_g5a_an_active_engine_still_publishes_its_real_book():
    """The other half of the crash hazard: a full book must still format, and the readings must
    stop claiming absences.

    I-2: this fixture is the producer's active payload, not a transcription of it — 29 keys taken
    out of the `ast.Dict` literal in `publish_telemetry` (`basis_trader.py:808-838`), which is why
    the delta below arrives as `net_delta_usd`. The hand-written version of this test omitted that
    key and invented `net_delta` and `reason`, and a reader wired to the parked stub alone passed it.
    """
    payload = _g5_active_engine_payload()
    active_keys = _g5_producer_payload_keys("active")
    assert set(payload) <= active_keys, (
        f"the 'active engine' fixture publishes {sorted(set(payload) - active_keys)}, which the "
        "active heartbeat does not")
    assert "net_delta_usd" in active_keys and "net_delta" not in active_keys, (
        "the active payload no longer carries the engine's computed delta, so this test is "
        "describing a different producer")
    html, _fake = _g5_render_carry_panel(None, payload=payload)
    # The live engine publishes no `reason` — only the parked path does (`basis_trader.py:919`) —
    # so exactly one card is entitled to say it measured nothing, and it must name itself. An
    # "all measured" claim here would be this test inventing a key the producer never writes.
    notes = re.findall(r"\(no ([a-z_]+) in basis trader telemetry\)", html)
    assert notes == ["reason"], (notes, html[:2500])
    assert "posture not measured" not in html, "the engine's own delta lost its posture"
    assert "$2,500 USDT" in html, html
    assert "+11.25%" in html and "(Hurdle: 8.0%)" in html, html
    assert "+0.0120%" in html, html
    assert "$+1.8675 USD" in html, html
    assert "$-0.75" in html, "the published net delta is not printed"
    assert "+24.5% Buffer" in html and "(Liq: $91,000.00)" in html, html
    assert "ISOLATED" in html, html
    assert "🟢 ACTIVE 1:1 DELTA-NEUTRAL TWIN" in html, html
    assert "3.5d / 7d" in html, html


def test_g5a_the_two_producer_shapes_are_the_two_fixtures_this_suite_runs():
    """The contract I-1 reads against, pinned to the producer rather than to a comment.

    One file, two writers: the heartbeat publishes 29 keys and the DISABLED parking path publishes
    five. If either literal changes shape, the panel's dual-key read and these fixtures have to be
    re-reasoned together, so this fails first rather than letting a reader quietly narrow again.
    """
    active_keys = _g5_producer_payload_keys("active")
    parked_keys = _g5_producer_payload_keys("parked")
    assert parked_keys == frozenset(_G5_LIVE_PAYLOAD), (
        f"the parked payload is no longer the five keys this suite seeds: {sorted(parked_keys)}")
    assert "net_delta" in parked_keys and "net_delta_usd" not in parked_keys
    assert sorted(active_keys) != sorted(parked_keys)
    readings = _g5_readings(_g5_active_engine_payload())
    assert readings["net_delta"] == {"value": -0.75, "measured": True, "reason": ""}, readings
    assert readings["reason"]["measured"] is False, readings["reason"]


def test_g5a_a_negative_harvest_is_not_printed_as_a_gain():
    """The old line hardcoded a `+` before the dollar figure, so a $-4.50 harvest read as a gain."""
    html, _fake = _g5_render_carry_panel(
        None, payload={**_G5_LIVE_PAYLOAD, "cumulative_funding_harvested": -4.5})
    assert "$-4.5000 USD" in html, html
    assert "+$-4.5000" not in html, html
    assert "#ff003c" in html, "a negative harvest still wears the all-clear green"


def test_g5a_the_legacy_dashboard_has_no_carry_panel_to_convert():
    """The scope claim in the brief, checked rather than assumed: `scripts/live_dashboard.py` never
    read this telemetry, so there is one file to fix and one block to guard."""
    legacy_src = LEGACY_DASHBOARD.read_text(encoding="utf-8")
    assert "basis_trader_telemetry" not in legacy_src
    assert "CASH-AND-CARRY" not in legacy_src
    with pytest.raises(Exception):
        _g4_block("basis", LEGACY_DASHBOARD)


# ---------------------------------------------------------------------------
# G-5b — the Bayesian Markov regime deck: a 35-hour-old row under "Live Forward Filter"
# ---------------------------------------------------------------------------
# Measured at BASE `df30aa3` against `file:trading_data.db?mode=ro`, `system_state`, key
# `markov_regime_telemetry`: `updated_at = '2026-10-21T14:21:29.139442+00:00'` and the payload
# carries exactly the ten keys `markov_regime.py::process_bars` publishes (`:398-413` short-frame
# path, `:445-460` filtered path). The deck read the value only — `query_system_state`
# (`dashboard.py:101-106`) is `SELECT value FROM system_state WHERE key = ?` and cannot see
# `updated_at` — grepped its whole window for `stale|STALE|updated_at|age` and found nothing, and
# printed the heading "(Live Forward Filter)". Every one of the fleet's four automated consumers
# (`super_bot.py:718-736`, `super_bot.py:909-924`, `inflight_shepherd.py:594-615`,
# `swing_trader.py:928-944`) refuses a row older than `config.REGIME_MAX_AGE_SEC` (`config.py:109`,
# default "900"). The HUD was the only consumer that never refused, and the only one a human reads.
#
# What is proven below: `_markov_regime_readings` is compiled out of dashboard.py and called for
# real, and the shipped deck block is exec'd against the seeded row in a namespace built out of
# dashboard.py's own helpers (the G-5a harness, extended — see `_g5_panel_namespace`). So the age
# path, the heading word, every absence and every colour is the code the fleet runs, not a
# paraphrase of it. The two DB readers stay stubbed at their return values because
# `@st.cache_data`-wrapped functions cannot be extracted; the row is still written into a real
# `system_state` table and read back out of sqlite first.
#
# The producer facts this suite reads against, all from `markov_regime.py`:
#   `MarkovRegimeEngine.__init__` (`:33-49`) is pure — uniform `pi`, the prior `P`, no dataset —
#   which is exactly why the deck's instantiate-and-run fallback published a prior as a measurement;
#   `get_regime_sizing_multiplier` (`:317-327`) bounds the multiplier to `[0.20, 1.0]`, so a
#   measured `0.0` is unreachable there and no test below claims otherwise; `stationary_distribution`
#   is a LIST of three rounded floats (`:409`, `:456`); the forecast is published as
#   `forecast_4step` (`:407`, `:454`) and the name `forecast_probabilities` never leaves
#   `forecast_k_steps` (`:360`).

MARKOV_REGIME = REPO_ROOT / "markov_regime.py"

_MARKOV_KEY = "markov_regime_telemetry"

# The live row, transcribed from the read-only database on 2026-10-23. Its KEY SET is not taken on
# trust: `test_g5b_the_fixture_keys_are_the_ten_the_producer_publishes` re-derives it from
# `markov_regime.py`'s syntax tree and fails this fixture if the two ever disagree.
_G5B_LIVE_PAYLOAD = {
    "dominant_regime": "CHOP",
    "state_probabilities": {"CHOP": 0.8878, "BULL_EXPANSION": 0.112, "BEAR_CASCADE": 0.0002},
    "expected_durations_bars": {"CHOP": 10.4, "BULL_EXPANSION": 16.5, "BEAR_CASCADE": 16.67},
    "breakdown_risk_4step": 0.6672,
    "forecast_4step": {"CHOP": 0.6374, "BULL_EXPANSION": 0.3328, "BEAR_CASCADE": 0.0297},
    "transition_matrix": [[0.9038461538461539, 0.08653846153846154, 0.009615384615384616],
                          [0.05757575757575757, 0.9393939393939394, 0.0030303030303030303],
                          [0.05, 0.01, 0.9399999999999998]],
    "stationary_distribution": [0.3703, 0.543, 0.0868],
    "is_tradable": True,
    "tradability_reason": "Tradable: E[D]=10.4b >= 3.5b",
    "sizing_multiplier": 0.64,
}

_G5B_LIVE_UPDATED_AT = "2026-10-21T14:21:29.139442+00:00"

# The live payload exactly as the row's `value` column stores it. Every reader-level test below
# calls `_markov_regime_readings` with this TEXT, because that is what the row reader hands back —
# the reader owns the `json.loads`, so a payload that is not JSON is the reader's problem to survive.
_MARKOV_JSON_TEXT = json.dumps(_G5B_LIVE_PAYLOAD)

# Fixed clock for the age path: 35h 00m after the live row was published, so the live row renders
# stale at a deterministic age (125999.86s -> "126000s old").
_G5B_FIX_NOW = datetime(2026, 10, 23, 1, 21, 29, tzinfo=timezone.utc)
_G5B_FRESH_UPDATED_AT = (_G5B_FIX_NOW - timedelta(seconds=60)).isoformat()

# The twenty-one figures the deck publishes. Every one of them is a state dict; the meta keys are
# not, and a test below pins that separation so the coverage headline cannot count a meta entry.
_G5B_PROB_FIELDS = ("prob_chop", "prob_bull", "prob_bear")
_G5B_DUR_FIELDS = ("dur_chop", "dur_bull", "dur_bear")
_G5B_FC_FIELDS = ("fc_chop", "fc_bull", "fc_bear")
_G5B_STAT_FIELDS = ("stat_chop", "stat_bull", "stat_bear")
_G5B_SCALAR_FIELDS = ("dominant_duration", "confidence", "breakdown_risk", "sizing_multiplier")
_G5B_TEXT_FIELDS = ("dominant_regime", "tradability_reason")
_G5B_FLAG_FIELDS = ("is_tradable",)
_G5B_VECTOR_FIELDS = ("equilibrium", "transition_matrix")
_G5B_FIELDS = (_G5B_TEXT_FIELDS + _G5B_PROB_FIELDS + _G5B_DUR_FIELDS + _G5B_FC_FIELDS
               + _G5B_STAT_FIELDS + _G5B_SCALAR_FIELDS + _G5B_FLAG_FIELDS + _G5B_VECTOR_FIELDS)
# `sizing_multiplier` is deliberately NOT in the measured-zero roster: the producer clamps it to
# [0.20, 1.0] at `markov_regime.py:327`, so a measured zero is not a reading it can publish.
_G5B_NUMBER_FIELDS = (_G5B_PROB_FIELDS + _G5B_DUR_FIELDS + _G5B_FC_FIELDS + _G5B_STAT_FIELDS
                      + ("breakdown_risk",))

# Payload key -> the readings the deck takes from it. Keyed by the producer's own key names, so the
# roster cannot drift away from `process_bars` without `test_g5b_the_fixture_keys_...` failing.
_G5B_PAYLOAD_FIELDS = {
    "dominant_regime": ("dominant_regime",),
    "state_probabilities": _G5B_PROB_FIELDS,
    "expected_durations_bars": _G5B_DUR_FIELDS,
    "forecast_4step": _G5B_FC_FIELDS,
    "stationary_distribution": _G5B_STAT_FIELDS,
    "breakdown_risk_4step": ("breakdown_risk",),
    "transition_matrix": ("transition_matrix",),
    "is_tradable": ("is_tradable",),
    "tradability_reason": ("tradability_reason",),
    "sizing_multiplier": ("sizing_multiplier",),
}

# The reason each figure prints when nothing was measured, pinned so the deck's parentheticals name
# the missing reading rather than the panel's own invention.
_G5B_FIELD_REASONS = {
    "dominant_regime": "no dominant_regime in markov regime telemetry",
    "prob_chop": "no CHOP in state_probabilities",
    "prob_bull": "no BULL_EXPANSION in state_probabilities",
    "prob_bear": "no BEAR_CASCADE in state_probabilities",
    "dur_chop": "no CHOP in expected_durations_bars",
    "dur_bull": "no BULL_EXPANSION in expected_durations_bars",
    "dur_bear": "no BEAR_CASCADE in expected_durations_bars",
    "fc_chop": "no CHOP in forecast_4step",
    "fc_bull": "no BULL_EXPANSION in forecast_4step",
    "fc_bear": "no BEAR_CASCADE in forecast_4step",
    "stat_chop": "no CHOP in stationary_distribution",
    "stat_bull": "no BULL_EXPANSION in stationary_distribution",
    "stat_bear": "no BEAR_CASCADE in stationary_distribution",
    "confidence": "no confidence: every state probability has to be measured first",
    "dominant_duration": ("no dominant duration: the dominant regime or its expected duration "
                          "was not measured"),
    "equilibrium": ("no equilibrium vector: stationary_distribution was not published as a "
                    "3-vector"),
    "breakdown_risk": "no breakdown_risk_4step in markov regime telemetry",
    "transition_matrix": "no 3x3 transition_matrix in markov regime telemetry",
    "is_tradable": "no is_tradable in markov regime telemetry",
    "tradability_reason": "no tradability_reason in markov regime telemetry",
    "sizing_multiplier": "no sizing_multiplier in markov regime telemetry",
}

# (rendered text variable, the one state variable it may read) — the G-5a idiom, one row per field
# the deck publishes as text. Finding #9: `m_gate_note` was the one per-field text with no owner, so
# a future edit could have keyed the gate's own absence note on another card's state unnoticed.
_G5B_FIELD_OWNERS = {
    "m_dom_text": "m_dom_st", "m_dom_note": "m_dom_st",
    "m_conf_text": "m_conf_st", "m_conf_note": "m_conf_st",
    "m_domdur_text": "m_domdur_st", "m_domdur_note": "m_domdur_st",
    "m_sizing_text": "m_sizing_st", "m_sizing_note": "m_sizing_st",
    "m_bd_text": "m_bd_st", "m_bd_note": "m_bd_st",
    "m_stat_text": "m_stat_st", "m_stat_note": "m_stat_st",
    "m_matrix_note": "m_matrix_st",
    "m_gate_note": "m_gate_st",
}

# Every verdict/colour the deck derives, and so every one that must be guarded. Finding #9 made this
# list exhaustive in BOTH directions — `test_g5b_every_name_the_deck_derives_is_rostered_or_explained`
# fails if the deck derives an `m_*` name that is not listed here, in `_G5B_FIELD_OWNERS`, in
# `_G5B_READER_ALIASES` or explained in `_G5B_UNROSTERED_SHAPES`, and fails if a listed name stops
# being derived. The four colours (`m_conf_color`, `m_domdur_color`, `m_stat_color`,
# `m_matrix_color`), `m_stamp_text` and the two cascade-card rungs (`m_bd_rungs`, `m_bd_alarm`) were
# all derived verdicts that no roster pinned.
_G5B_DERIVED_NAMES = frozenset(set(_G5B_FIELD_OWNERS) | {
    "m_dom_color", "m_conf_color", "m_domdur_color", "m_stat_color", "m_matrix_color",
    "m_bd_color", "m_bd_label", "m_sizing_color", "m_tradable_color",
    "m_tradable_badge", "m_reason_line", "m_time_text", "m_time_color", "m_heading",
    "m_coverage_line", "m_unmeasured_count", "m_field_count", "m_horizon_note", "m_matrix_html",
    "m_bars_html", "m_stamp_text",
    # Finding #9: two verdicts the deck publishes on the cascade card, both keyed on `m_bd_st` and
    # neither in a roster until now — exactly the "a future verdict could go unpinmarked" hole.
    "m_bd_rungs", "m_bd_alarm",
})

# The deck's own aliases of the reader's per-field states. They read exactly one key out of `m_r` and
# invent nothing; `test_g5b_the_reader_aliases_are_bare_lookups_into_the_reading_dict` pins that
# shape, because a line like `m_bd_st = m_r["breakdown_risk"] or {"measured": True, "value": 0.0}`
# would otherwise be a stand-in hiding inside a name the guards treat as a reading.
_G5B_READER_ALIASES = frozenset({
    "m_r", "m_dom_st", "m_conf_st", "m_domdur_st", "m_gate_st", "m_rsn_st", "m_sizing_st",
    "m_bd_st", "m_stat_st", "m_matrix_st",
})

# Names the deck assigns that are NOT single-field verdicts and so are deliberately absent from the
# rosters above. Finding #9 required the exclusion to be stated rather than implied, and the test
# below pins each entry's shape, so this comment cannot rot into a loophole: `m_series_specs` and
# `m_series_colors` are a label->reader-key tuple and a label->colour literal that read no
# measurement at all; `m_series_states` is the reader's own per-series aggregate (it folds the
# per-field `measured` flags, it does not invent one); `m_drawn_series` is that same aggregate
# filtered on `measured`, which is why an incomplete series cannot reach the plot;
# `m_horizon_absent` is a list of the reader's own absence reasons, consumed only by the chain
# verdict `m_horizon_note`. Each value is (why it is unrostered, substrings its assignment must
# contain, substrings it must not).
_G5B_UNROSTERED_SHAPES = {
    "m_series_specs": (
        "a label->reader-key tuple: it reads no measurement and paints no verdict",
        ('("Current π_t", "prob")', '"fc"', '"stat"'),
        ("measured", "m_r[", "_st[")),
    "m_series_colors": (
        "a label->colour literal: the colour is chosen per series, never per reading",
        ('"#00e5ff"', '"#a855f7"', '"#64748b"'),
        ("measured", "m_r[", "_st[")),
    "m_series_states": (
        "the reader's own per-series aggregate: it folds the per-field flags, invents none",
        ('"measured": all(', 'm_r["by_state"]'), ()),
    "m_drawn_series": (
        "that same aggregate filtered on measured, so an incomplete series never reaches the plot",
        ("m_series_states", 'one["measured"]'), ()),
    "m_horizon_absent": (
        "the reader's own absence reasons, consumed only by the chain verdict `m_horizon_note`",
        ('not m_r[one]["measured"]', '["reason"]'), ()),
}

# Derived verdicts that talk about the payload as a whole, or that are the reader's own decision
# rather than the block's: exempt from the per-field gate BY SHAPE, and the shape is asserted.
_G5B_CHAIN_VERDICTS = frozenset({"m_heading", "m_coverage_line", "m_unmeasured_count",
                                 "m_field_count", "m_horizon_note"})

# Provenance, measured (`g5b-roster-fidelity-check.py` against `94cf7fb:dashboard.py`): all 15 tokens
# occur verbatim in the pre-fix source, and THAT is the fidelity claim this roster carries. The
# `code_line`s are splice-sized for the reach test -- 8 verbatim, 7 reconstructions written against the
# reader's own variable names -- so they are synthetic fixtures, not evidence about the pre-fix file.
_G5B_STAND_IN_CASES = (
    ('"is_tradable", True', 'is_tradable = bool(m_tel.get("is_tradable", True))'),
    ('"sizing_multiplier", 1.0', 'sizing_mult = float(m_tel.get("sizing_multiplier", 1.0))'),
    ('"tradability_reason", "Tradable"', 'rsn = m_tel.get("tradability_reason", "Tradable")'),
    ('"dominant_regime", "CHOP"', 'dom_regime = m_tel.get("dominant_regime", "CHOP")'),
    ('"breakdown_risk_4step", 0.0', 'bd = float(m_tel.get("breakdown_risk_4step", 0.0))'),
    ('"stationary_distribution", {}', 'stat_dist = m_tel.get("stationary_distribution", {})'),
    ("or 0.0)", 'bd_risk = float(m_tel.get("breakdown_risk_4step") or 0.0)'),
    ("or 1.0)", 'sizing_mult = float(m_tel.get("sizing_multiplier") or 1.0)'),
    ("or 0.333", 'st_chop = float(raw_stat.get("CHOP") or 0.333)'),
    ('"CHOP": 0.60', 'm_tel = {"state_probabilities": {"CHOP": 0.60}}'),
    ("0.333, 0.333, 0.333", 'st_chop, st_bull, st_bear = 0.333, 0.333, 0.333'),
    ("default_P", 'raw_P = m_tel.get("transition_matrix", default_P.tolist())'),
    ("MarkovRegimeEngine", 'm_eng = MarkovRegimeEngine()'),
    ("from markov_regime import", 'from markov_regime import MarkovRegimeEngine'),
    ('"forecast_probabilities"', 'fc_probs = m_tel.get("forecast_probabilities", {})'),
)

_G5B_PROSE_CLAIMS = (
    ("(Live Forward Filter)", 'st.markdown("#### ... (Live Forward Filter)")'),
    ('"TRADABLE"', 'tradable_badge = "TRADABLE"'),
    # Finding #5: the caption the deck used to print under a measured matrix. It is prose in the
    # sense that a comment may quote it without failing the ban, but it is a CLAIM in the sense
    # that the only way to print it is a string literal in the code -- so the scan that kills it
    # reads code only, and this pair keeps the scan honest in both directions.
    ("estimated from labelled bars",
     'm_matrix_html = "<div>matrix the engine published (estimated from labelled bars)</div>"'),
)

_G5B_ABSENT_COLOUR = "#94a3b8"
_G5B_ALL_CLEAR_COLOUR = "#00ff66"


def _g5b_producer_returns():
    """The two `return` dict literals of `MarkovRegimeEngine.process_bars`, from the AST.

    Both paths — the `len(df) < 15` short-frame branch and the filtered branch — have to be found,
    because the contract this suite reads against is "exactly these keys, on both paths".
    """
    import ast

    tree = ast.parse(MARKOV_REGIME.read_text(encoding="utf-8"), filename=str(MARKOV_REGIME))
    engine = next((node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "MarkovRegimeEngine"), None)
    assert engine is not None, "markov_regime.py no longer defines MarkovRegimeEngine"
    process = next((node for node in engine.body if isinstance(node, ast.FunctionDef)
                    and node.name == "process_bars"), None)
    assert process is not None, "MarkovRegimeEngine no longer defines process_bars"
    literal_keys = []
    for node in ast.walk(process):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            keys = tuple(key.value for key in node.value.keys)
            assert all(isinstance(key, str) for key in keys), (
                "process_bars has a non-literal telemetry key, so this suite can no longer read "
                "the producer's contract off its syntax tree")
            literal_keys.append(keys)
    assert len(literal_keys) == 2, (
        f"process_bars publishes {len(literal_keys)} telemetry literals, expected the short-frame "
        "and filtered paths; the deck's contract needs re-reading")
    assert literal_keys[0] == literal_keys[1], (
        f"the two process_bars paths publish different key sets: {literal_keys}")
    return literal_keys[0]


def _g5b_producer_payload_keys():
    return frozenset(_g5b_producer_returns())


def _g5b_producer_states():
    """`MarkovRegimeEngine.STATES`, read out of the class body rather than typed in here."""
    import ast

    tree = ast.parse(MARKOV_REGIME.read_text(encoding="utf-8"), filename=str(MARKOV_REGIME))
    engine = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "MarkovRegimeEngine")
    for node in engine.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "STATES"
                                                for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("MarkovRegimeEngine no longer declares STATES, so the deck's roster of "
                         "regime names has nothing to be checked against")


def _g5b_sizing_bound_from_producer(src=None):
    """The `[low, high]` the producer clamps `sizing_multiplier` into, read out of its AST.

    `markov_regime.py:327` is `round(float(max(0.20, min(1.0, mult))), 2)`, so the bound is the one
    float ARGUMENT of the `max` call and the one float argument of the `min` call. The lower bound is
    what keeps this suite from writing the test the brief forbids: a deck that "measures" a 0.0
    sizing multiplier is describing a producer that cannot publish one.

    Finding #10: this used to take `min`/`max` over every float in the getter, which also swept up the
    posterior weights at `:326` (`1.0`, `0.60`, `0.20`) and the `2` in the rounding — it answered
    correctly only because those weights happen to sit inside `[0.20, 1.0]`. A producer that widened
    its weights or narrowed its clamp would have moved one number and hidden the other. `src` lets a
    test run the same reader over a getter whose weights fall outside the clamp.
    """
    import ast

    text = MARKOV_REGIME.read_text(encoding="utf-8") if src is None else src
    tree = ast.parse(text, filename=str(MARKOV_REGIME))
    getter = next((node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                   and node.name == "get_regime_sizing_multiplier"), None)
    assert getter is not None, "MarkovRegimeEngine no longer computes a sizing multiplier"
    clamp = {}
    for node in ast.walk(getter):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ("max", "min")):
            continue
        literals = [arg.value for arg in node.args
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, float)]
        assert len(literals) == 1, (
            f"the producer's {node.func.id}() clamp takes {literals}; this reader needs exactly one "
            "float argument per clamp call, or the deck's bound has to be re-derived")
        assert node.func.id not in clamp, (
            f"the producer clamps twice with {node.func.id}(), so sizing_multiplier has no single "
            "bound for the deck to quote")
        clamp[node.func.id] = literals[0]
    assert set(clamp) == {"max", "min"}, (
        f"get_regime_sizing_multiplier clamps with {sorted(clamp)}, expected max and min; the deck "
        "cannot quote a bound it cannot find")
    low, high = clamp["max"], clamp["min"]
    assert low <= high, (
        f"the producer's clamp is [{low}, {high}], which is empty; re-read the bound before testing "
        "what an absence may not be defaulted to")
    return low, high


def _g5b_readings(payload, updated_at=_G5B_LIVE_UPDATED_AT, now=_G5B_FIX_NOW,
                 max_age_sec=None, path=None):
    """`_markov_regime_readings`, compiled out of the shipped file, called on one row."""
    helper = _g4_helper("_markov_regime_readings", DASHBOARD if path is None else path)
    return helper(payload, updated_at, now=now, max_age_sec=max_age_sec)


_G5B_FIXTURE_DB_NAME = "telemetry_fixture.db"


def _g5b_extract_row_reader():
    """`query_system_state_row`, compiled out of dashboard.py with its decorator satisfied.

    Finding #3: this is the only new SQL on the branch, and every render test used to inject a lambda
    in its place, so a swapped SELECT list, a `row[0]` used twice, or an absent shape of
    `(None, None)` would have kept the whole focused file green while the deck computed an age from
    the regime JSON. The excuse the previous round offered — a `@st.cache_data`-wrapped function
    cannot be extracted — does not hold: the cache is a decorator, and `_G5RecordingStreamlit` now
    answers it with the identity function, so what runs below is the shipped body including its own
    `connect_readonly` (pulled in by the same dependency walk every other extracted helper uses).
    What stays unexercised is Streamlit's real 4s TTL cache, not the SELECT. Finding #6 made this the
    route every `tmp_path` render takes, not a special case a single test opts into.
    """
    return _dashboard_helper("query_system_state_row", DASHBOARD,
                             extra_ns={"st": _G5RecordingStreamlit()})


def _g5b_bind_reader(reader, db_path):
    """Point one extracted reader at a fixture database and hand back its fresh `FEED_FAILURES`.

    The namespace `_dashboard_helper` builds never gets `TRADING_DATA_DB` or `FEED_FAILURES` bound:
    the first is an `os.path.join(BASE_DIR, ...)` whose `BASE_DIR` the walk cannot resolve, and the
    second is an AnnAssign, which the walk does not compile at all. Both are set explicitly here so
    each test reads exactly one `tmp_path` database and can see what the reader recorded when it
    could not.
    """
    failures = {}
    reader.__globals__["TRADING_DATA_DB"] = str(db_path)
    reader.__globals__["FEED_FAILURES"] = failures
    return failures


def _g5b_block_code(path=None):
    """The Markov deck's own source, comments removed: what every scan below actually reads."""
    return _g5_code_only(_g4_block("markov", DASHBOARD if path is None else path))


def _g5b_helper_code(name="_markov_regime_readings"):
    """One module-level Markov helper's source, comments removed."""
    return _g5_code_only(_function_source(DASHBOARD, name))


def _g5b_deck_and_helper_code(helper=None):
    """The deck block AND the reader helper, comments and docstrings removed (finding #9).

    B-5 moved the payload reads out of the block and into `_markov_regime_readings`, so a scan of the
    block alone no longer sees the code that decides what the deck prints — which is how a stand-in
    could have come back inside the reader unnoticed. Every `_G5B_STAND_IN_CASES` guard reads this.
    `helper` replaces the helper half so a test can feed the same target a seeded reader and prove
    the widening is real rather than a concatenation that quietly drops it.
    """
    return _g5b_block_code() + "\n" + (_g5b_helper_code() if helper is None else helper)


def _g5b_helper_seeded_with(code_line):
    """The SHIPPED reader helper's own source with `code_line` spliced into the top of its body.

    This is the second half of the widened scan: a stand-in has to be caught where the reads actually
    live, so the seed goes into the real function text rather than into a stub that merely has the
    same name. The `def` line is asserted to be the first line, so the splice cannot silently drift
    into an outer scope if the signature is ever re-wrapped.
    """
    helper = _g5b_helper_code()
    lines = helper.splitlines(True)
    assert lines and lines[0].startswith("def _markov_regime_readings"), (
        f"the reader helper no longer starts with its own def line, so this seed proves nothing: "
        f"{lines[:1]}")
    return lines[0] + "    " + code_line.strip() + "\n" + "".join(lines[1:])


def _g5b_panel_namespace(path, raw_value, updated_at, now=None, config_bound=900.0):
    """The G-5a namespace, plus what only this deck needs: numpy, plotly and the row reader.

    `query_system_state` answers for BOTH telemetry keys so that the pre-fix deck — which read the
    value through `query_system_state` — renders the same row the post-fix deck reads through
    `query_system_state_row`. That is what makes the RED runs below evidence about the shipped
    panel rather than about a stub.
    """
    import plotly.graph_objects as go

    extra_funcs = ("_markov_regime_readings", "_regime_max_age_sec", "_stance_color", "num")
    ns, fake = _g5_panel_namespace(
        path, raw_value, now=None, extra_funcs=extra_funcs,
        const_prefixes=("_BASIS_", "_MARKOV_", "_REGIME_"),
        extra_ns={"np": np, "go": go,
                  "query_system_state": lambda key, default="": (
                      raw_value if key in (_MARKOV_KEY, "basis_trader_telemetry") else default),
                  "query_system_state_row": lambda key: (
                      (raw_value, updated_at) if key == _MARKOV_KEY else ("", ""))})

    class _Bound:
        REGIME_MAX_AGE_SEC = config_bound

    ns["config"] = _Bound()
    if now is not None and callable(ns.get("_markov_regime_readings")):
        shipped = ns["_markov_regime_readings"]
        # `max_age_sec` is forwarded, never swallowed: the whole point of finding #4 is that the
        # deck passes the fleet's bound through this signature, and a lambda that dropped the
        # keyword would pin the harness to the fallback and hide the bug again.
        ns["_markov_regime_readings"] = (
            lambda raw, published_at, max_age_sec=None, _s=shipped, _n=now:
            _s(raw, published_at, now=_n, max_age_sec=max_age_sec))
    return ns, fake


def _g5b_render_deck(tmp_path=None, payload=_G5B_LIVE_PAYLOAD, updated_at=_G5B_LIVE_UPDATED_AT,
                     raw=None, path=None, now=_G5B_FIX_NOW, config_bound=900.0,
                     use_shipped_reader=False, seed_updated_at=None):
    """Run the shipped Markov deck for one row and return everything it published.

    Which route the row takes, exactly (finding #6 rewrote this docstring after it had claimed a
    storage round-trip for the stamp that the harness never performed — only `value` went through
    sqlite while `updated_at` reached the deck as an argument of an injected lambda, which is the
    branch's own defect class inside the suite and is why finding #3 went unnoticed):

    * `payload=` dict with `tmp_path` — the row is WRITTEN to a `tmp_path` `system_state` by
      `_g5_system_state_row`, and the deck reads it back through the shipped
      `query_system_state_row` extracted from `dashboard.py` (fix #3). Both columns the deck parses
      and ages come out of the database, and the helper asserts the stored pair equals the pair the
      fixture wrote. `seed_updated_at` writes a stamp that disagrees with `updated_at`, so a harness
      that still injected the argument cannot pass as a round-trip.
    * `raw=`, or `payload=` str with no `tmp_path` — NO row is written, so there is nothing stored:
      the injected lambda answers with the text handed in. The sibling test further down that says
      `still_says_so` pins this exception is real rather than aspirational.
    * `use_shipped_reader=True` — forces the first route's reader even when the caller passed `raw=`,
      which is how `test_g5b_the_shipped_reader_hands_back_the_stored_stamp_the_deck_ages` proves the
      reader's own SELECT is what the deck aged.

    `updated_at` seeds the row in the first route; in the other two it is the injected stamp. The
    value-only `query_system_state` stand-in still answers for both Markov and basis keys, which is
    what lets a pre-fix `path=` render read the same row as the shipped one.
    """
    import json as _json

    path = DASHBOARD if path is None else path
    row_written = False
    if raw is None:
        if isinstance(payload, str):
            raw = payload
        elif tmp_path is not None and isinstance(payload, dict):
            raw = _g5_system_state_row(tmp_path, payload=payload, key=_MARKOV_KEY,
                                       updated_at=(updated_at if seed_updated_at is None
                                                   else seed_updated_at))
            row_written = True
        else:
            raw = _json.dumps(payload)
    ns, fake = _g5b_panel_namespace(path, raw, updated_at, now=now, config_bound=config_bound)
    if use_shipped_reader or row_written:
        assert tmp_path is not None, (
            "the shipped-reader route has to point dashboard.py's own SQL at a tmp database")
        reader = _g5b_extract_row_reader()
        _g5b_bind_reader(reader, tmp_path / _G5B_FIXTURE_DB_NAME)
        if row_written:
            stored_value, stored_stamp = reader(_MARKOV_KEY)
            assert stored_value == raw, (
                "finding #6: the value the deck reads from sqlite is not the value the fixture "
                "wrote into it")
            assert stored_stamp == (updated_at if seed_updated_at is None else seed_updated_at), (
                "finding #6: the stamp the deck reads from sqlite is not the stamp the fixture "
                "wrote into it")
        ns["query_system_state_row"] = reader
    block = _g4_block("markov", path)
    exec(compile(block, f"{path.name} markov deck", "exec"), ns)
    return "\n".join(fake.markdown_calls), fake, ns


def _g5b_heading(html):
    """The deck's `####` heading line, so a test can speak about the heading and nothing else."""
    lines = [one for one in html.splitlines() if one.strip().startswith("####")]
    assert lines, f"the deck published no heading:\n{html[:1500]}"
    return lines[0]


def _g5b_payload_without(*keys):
    """The live payload minus the named keys — the shape a partial publish or an older producer
    leaves behind, and the only way to ask the deck what it does when a reading is not there."""
    out = dict(_G5B_LIVE_PAYLOAD)
    for key in keys:
        out.pop(key, None)
    return out


# ── B-1: the row's own publish instant, and the word "Live" ──────────────────────────────
def test_g5b_a_stale_row_is_not_called_live(tmp_path):
    """The finding that leads the task: the live row is 35h old against a 900s bound, and the deck
    called it a live forward filter. This is the RED-against-`df30aa3` proof for B-1."""
    html, _fake, _ns = _g5b_render_deck(tmp_path)
    heading = _g5b_heading(html)
    assert "Live" not in heading, (
        f"a row 35 hours old still wears the live claim: {heading}")
    assert "(Live Forward Filter)" not in html, html[:1500]
    assert "STALE" in heading.upper(), f"the heading does not say what it is: {heading}"
    assert "s old" in html, "the deck never prints the row's age"
    assert "126000s old" in html, html[:1500]
    assert "STALE TELEMETRY" in html, html[:1500]
    assert "900" in html, "the deck does not name the bound the row blew through"


def test_g5b_a_fresh_row_is_still_called_a_live_forward_filter(tmp_path):
    """The other half, or the fix is just the word "Live" deleted. Inside the bound the claim is
    supported by the measurement and has to stay."""
    html, _fake, _ns = _g5b_render_deck(tmp_path, updated_at=_G5B_FRESH_UPDATED_AT)
    heading = _g5b_heading(html)
    assert "Live Forward Filter" in heading, heading
    assert "STALE" not in heading.upper(), heading
    assert "60s old" in html, html[:1500]


def test_g5b_an_absent_row_reads_differently_from_a_stale_one(tmp_path):
    """Absent and stale are different states and may not share a sentence: the fleet refuses a
    stale row and has nothing at all to refuse when the row was never written.

    Finding #8 extended this rather than replacing it: a row that EXISTS and is not a regime
    telemetry object (`'[1,2,3]'`, `'not json at all'`) blames the producer, while no row at all
    blames the scheduler, and until this fix all three malformed shapes plus the empty object
    printed the absent row's sentence, because `readings["row_present"] = bool(data)` cannot tell
    them apart. Still no green anywhere in the family.
    """
    stale_html = _g5b_render_deck(tmp_path)[0]
    absent_html = _g5b_render_deck(tmp_path, raw="", updated_at="")[0]
    assert "Live" not in _g5b_heading(absent_html), _g5b_heading(absent_html)
    assert "not measured" in absent_html, absent_html[:1500]
    assert "STALE TELEMETRY" not in absent_html, (
        "a row that was never published is labelled as one that went stale")
    assert "126000s old" not in absent_html
    assert "126000s old" in stale_html
    assert absent_html != stale_html, "absent and stale render identically"

    # Same family, with a usable stamp so the heading has to speak about the ROW and not about it.
    absent_row = _g5b_render_deck(tmp_path, raw="", updated_at=_G5B_FRESH_UPDATED_AT)[0]
    malformed_row = _g5b_render_deck(tmp_path, raw="[1,2,3]", updated_at=_G5B_FRESH_UPDATED_AT)[0]
    not_json_row = _g5b_render_deck(tmp_path, raw="not json at all",
                                    updated_at=_G5B_FRESH_UPDATED_AT)[0]
    empty_object_row = _g5b_render_deck(tmp_path, raw="{}", updated_at=_G5B_FRESH_UPDATED_AT)[0]

    assert "no row" in _g5b_heading(absent_row), _g5b_heading(absent_row)
    assert "not a regime telemetry object" not in _g5b_heading(absent_row), (
        "a row nobody wrote is being blamed on a producer that never ran")
    assert "not a regime telemetry object" in _g5b_heading(malformed_row), (
        _g5b_heading(malformed_row))
    assert _g5b_heading(not_json_row) == _g5b_heading(malformed_row), (
        "two rows that are not objects should read the same way")
    assert "not a regime telemetry object" not in _g5b_heading(empty_object_row), (
        _g5b_heading(empty_object_row))
    assert _g5b_heading(empty_object_row) != _g5b_heading(malformed_row), (
        "a published empty object and a row of garbage are one sentence")
    assert _g5b_heading(empty_object_row) != _g5b_heading(absent_row), (
        "the row that exists is read as the row that does not")
    for name, one in (("absent", absent_row), ("malformed", malformed_row),
                      ("not-json", not_json_row), ("empty object", empty_object_row)):
        assert "Live" not in _g5b_heading(one), (name, _g5b_heading(one))
        assert _G5B_ALL_CLEAR_COLOUR not in one, (name, one[:2000])
        assert _G5B_ABSENT_COLOUR in one, (name, one[:2000])


def test_g5b_the_age_is_measured_from_the_rows_own_updated_at():
    """`regime_classifier.py:1218-1221` writes the row with `updated_at =
    datetime.now(UTC).isoformat()` (`:1136`), so that stamp is the publish instant and the payload
    carries no timestamp of its own. Naive stamps and `Z` suffixes are both handled by the fleet's
    own parser (`inflight_shepherd.py:604-611`), and neither may be read as a live row."""
    readings = _g5b_readings(_MARKOV_JSON_TEXT)
    assert readings["age_sec"] is not None
    assert 125900 <= readings["age_sec"] <= 126100, readings["age_sec"]
    assert readings["stale"] is True and readings["age_known"] is True
    assert readings["published_at"]["measured"] is True
    zulu = _g5b_readings(_MARKOV_JSON_TEXT, updated_at="2026-10-21T14:21:29Z")
    assert zulu["age_known"] is True and abs(zulu["age_sec"] - 126000.0) < 2.0, zulu["age_sec"]
    naive = _g5b_readings(_MARKOV_JSON_TEXT, updated_at="2026-10-21 14:21:29")
    assert naive["age_known"] is True, "a naive stamp is the CURRENT_TIMESTAMP shape"
    assert abs(naive["age_sec"] - 126000.0) < 2.0, naive["age_sec"]
    junk = _g5b_readings(_MARKOV_JSON_TEXT, updated_at="syncing...")
    assert junk["age_sec"] is None and junk["age_known"] is False, junk["age_sec"]
    assert junk["stale"] is False
    assert junk["heading_is_live"] is False, "a row with no usable stamp may not claim to be live"
    missing = _g5b_readings(_MARKOV_JSON_TEXT, updated_at="")
    assert missing["published_at"]["measured"] is False, missing["published_at"]
    assert missing["heading_is_live"] is False


def test_g5b_the_bound_edges_are_the_fleet_semantics():
    """Consumers refuse on `age > max_age` (`inflight_shepherd.py:614`), so exactly at the bound is
    still fresh. The HUD may not be stricter or gentler than the gate that is actually enforced."""
    at_bound = _g5b_readings(_MARKOV_JSON_TEXT,
                             updated_at=(_G5B_FIX_NOW - timedelta(seconds=900)).isoformat())
    assert at_bound["age_sec"] <= 900.0 and at_bound["stale"] is False, at_bound["age_sec"]
    assert at_bound["heading_is_live"] is True
    over = _g5b_readings(_MARKOV_JSON_TEXT,
                         updated_at=(_G5B_FIX_NOW - timedelta(seconds=901)).isoformat())
    assert over["stale"] is True and over["heading_is_live"] is False
    assert over["max_age_sec"] == 900.0, over["max_age_sec"]


def test_g5b_the_bound_is_the_fleets_constant_and_not_a_second_number(tmp_path):
    """`config.py:109` is the bound; the four consumers all fall back to the literal 900.0. A panel
    with its own softer threshold would let a row through that the fleet is refusing."""
    for relpath in ("config.py", "super_bot.py", "inflight_shepherd.py", "swing_trader.py"):
        src = (REPO_ROOT / relpath).read_text(encoding="utf-8")
        assert "REGIME_MAX_AGE_SEC" in src, f"{relpath} no longer names the bound"

    # Finding #4: this is where `assert "REGIME_MAX_AGE_SEC" in _g5_code_only(helper_src)` used to
    # sit. Even with comments stripped it could be satisfied by the helper's docstring or by any other
    # piece of prose in it, so a helper that stopped consulting `config` entirely could have kept it
    # green. The pin is behavioural now: a bound the config tightens to 30s has to reach the deck, and
    # the deck has to refuse a 60-second row because of it. `config_bound` is injected as `config`
    # into the shipped namespace, so this runs `_regime_max_age_sec`'s own `getattr` path.
    tight_html, _tight_fake, tight_ns = _g5b_render_deck(
        tmp_path, updated_at=_G5B_FRESH_UPDATED_AT, config_bound=30.0)
    assert tight_ns["m_r"]["max_age_sec"] == 30.0, tight_ns["m_r"]["max_age_sec"]
    assert tight_ns["m_r"]["age_sec"] > 30.0, tight_ns["m_r"]["age_sec"]
    assert tight_ns["m_r"]["stale"] is True, (
        "config.REGIME_MAX_AGE_SEC = 30 was ignored and the deck kept a 60s row")
    assert "STALE" in _g5b_heading(tight_html), _g5b_heading(tight_html)
    assert "past 30s" in tight_html, (
        "the deck names a bound it did not read from config: " + tight_html[:1500])
    assert _G5B_ALL_CLEAR_COLOUR not in tight_html, (
        "a row the tightened bound refuses still wears the all-clear")

    wide_html, _wide_fake, wide_ns = _g5b_render_deck(
        tmp_path, updated_at=_G5B_FRESH_UPDATED_AT, config_bound=900.0)
    assert wide_ns["m_r"]["stale"] is False and wide_ns["m_r"]["max_age_sec"] == 900.0
    assert "Live Forward Filter" in _g5b_heading(wide_html), _g5b_heading(wide_html)
    assert _G5B_ALL_CLEAR_COLOUR in wide_html, (
        "the same 60s row inside the bound lost its measured all-clear")

    resolver = _g4_helper("_regime_max_age_sec")

    class _Stub:
        def __init__(self, value):
            self.REGIME_MAX_AGE_SEC = value

    original = resolver.__globals__.get("config")
    try:
        resolver.__globals__["config"] = _Stub(1800.0)
        assert resolver() == 1800.0, "a raised bound must be honoured, not overridden"
        for bad in ("abc", None, "", 0, -5, float("nan")):
            resolver.__globals__["config"] = _Stub(bad)
            assert resolver() == 900.0, (
                f"a missing/non-numeric REGIME_MAX_AGE_SEC ({bad!r}) must fall back to the "
                "consumers' 900.0")
        resolver.__globals__.pop("config", None)
        assert resolver() == 900.0, "a namespace without config must fall back to 900.0"
    finally:
        if original is not None:
            resolver.__globals__["config"] = original


def test_g5b_the_deck_reads_the_row_with_its_updated_at():
    """`query_system_state` (`dashboard.py:101-106`) returns `value` only, which is why B-1 needs a
    second reader. Its signature stays alone out of this round — it is called across the file."""
    block = _g5b_block_code()
    assert 'query_system_state_row("markov_regime_telemetry")' in block, (
        "the deck still reads the value without its publish instant")
    assert 'query_system_state("markov_regime_telemetry"' not in block, (
        "the deck still takes a value-only read of a row whose age is the finding")
    reader_src = _g5_code_only(_function_source(DASHBOARD, "query_system_state_row"))
    assert "SELECT value, updated_at FROM system_state WHERE key = ?" in reader_src, (
        "the new reader does not select the column the age path needs")
    # Finding #4: the word-membership line this replaced (`assert "value" in reader_src and
    # "updated_at" in reader_src`) was satisfied by the reader's own first docstring line and proved
    # nothing; the column pair is now pinned by executing the shipped SELECT against a real row, in
    # both orders, in `test_g5b_the_shipped_reader_returns_the_value_and_the_stamp_in_that_order`.


# ── Finding #3: the shipped row reader is EXECUTED, not described ─────────────────────────
def test_g5b_the_shipped_reader_returns_the_value_and_the_stamp_in_that_order(tmp_path):
    """`query_system_state_row` is B-1's input and the only new SQL on the branch. Until now every
    render test injected a lambda in its place, so nothing pinned which element of the tuple is the
    regime JSON and which is the publish stamp — swap the SELECT list, or return `row[0]` twice, and
    the deck would compute an age out of the JSON while all 625 focused cases stayed green.

    The shipped body runs here: extracted out of `dashboard.py` with its `@st.cache_data` answered by
    the identity decorator, pointed at a `tmp_path` database, through its own `connect_readonly`.
    """
    value_text = '{"is_tradable": true}'
    stamp = "2026-10-21T14:21:29.139442+00:00"
    seeded = _g5_system_state_row(tmp_path, payload={"is_tradable": True}, key=_MARKOV_KEY,
                                  updated_at=stamp)
    assert seeded == value_text, seeded
    db = tmp_path / _G5B_FIXTURE_DB_NAME
    assert db.exists(), "the fixture database the shipped reader must open was never written"

    reader = _g5b_extract_row_reader()
    failures = _g5b_bind_reader(reader, db)
    got = reader(_MARKOV_KEY)
    assert isinstance(got, tuple) and len(got) == 2, f"the row shape is not a 2-tuple: {got!r}"
    assert got[0] == value_text, f"element 0 is not `value`: {got!r}"
    assert got[1] == stamp, f"element 1 is not `updated_at`: {got!r}"
    assert got[0] != got[1], f"the reader returned one column twice: {got!r}"
    assert failures == {}, f"a clean read recorded a feed failure: {failures}"


def test_g5b_the_shipped_reader_hands_back_the_stored_stamp_the_deck_ages(tmp_path):
    """The round-trip the render helper's docstring used to claim for every test: with the shipped
    reader wired in, the stamp the deck prints its age from comes OUT OF THE DATABASE, not out of an
    injected lambda argument. Delete the reader's `updated_at` from its SELECT and this fails."""
    stamp = (_G5B_FIX_NOW - timedelta(seconds=126000)).isoformat()
    html, fake, ns = _g5b_render_deck(tmp_path, updated_at=stamp, use_shipped_reader=True)
    assert ns["query_system_state_row"](_MARKOV_KEY)[1] == stamp, "the deck is not reading this row"
    assert "126000s old" in html, html[:1500]
    assert "STALE" in _g5b_heading(html), _g5b_heading(html)
    assert "88.8%" in html and "0.64x" in html, html[:2000]
    assert fake.charts, "the deck published no chart off a row it read from sqlite"


def test_g5b_the_default_route_ages_the_deck_from_the_stored_stamp(tmp_path):
    """Finding #6: the render helper claimed its default route proved a stored stamp; it did not,
    because the deck's `updated_at` came from an injected lambda rather than from the row.

    This is that claim made true instead of reworded: the row in `tmp_path` is written with a stamp
    126000 s before the fixture clock while the `updated_at` ARGUMENT says 60 s. Only storage can
    produce the age the deck prints, so a harness that still injects the argument fails here — and
    the shipped `query_system_state_row` extracted out of `dashboard.py` has to be the callable the
    deck actually used, with its own `TRADING_DATA_DB` pointing at this test's database.
    """
    stored = (_G5B_FIX_NOW - timedelta(seconds=126000)).isoformat()
    injected = _G5B_FRESH_UPDATED_AT
    html, _fake, ns = _g5b_render_deck(tmp_path, updated_at=injected, seed_updated_at=stored)

    assert ns["m_r"]["age_sec"] == pytest.approx(126000.0), (
        "the deck aged " + str(ns["m_r"]["age_sec"]) + "s while the stored row says " + stored
        + " — the stamp came from the harness, not from sqlite")
    assert ns["m_r"]["published_at"]["text"] == stored, ns["m_r"]["published_at"]
    assert "126000s old" in html, html[:2000]
    assert "60s old" not in html, "the injected stamp was printed as if it were the row's"
    assert "STALE" in _g5b_heading(html), _g5b_heading(html)
    assert "Live" not in _g5b_heading(html), _g5b_heading(html)

    reader = ns["query_system_state_row"]
    assert getattr(reader, "__name__", "") == "query_system_state_row", (
        "the deck read the row through an injected stand-in, not through the shipped reader")
    assert reader.__globals__.get("TRADING_DATA_DB") == str(tmp_path / _G5B_FIXTURE_DB_NAME), (
        "the shipped reader was bound to some other database")
    value_text, stamp_back = reader(_MARKOV_KEY)
    assert stamp_back == stored, reader(_MARKOV_KEY)
    assert json.loads(value_text) == _G5B_LIVE_PAYLOAD, value_text[:200]


def test_g5b_a_route_that_writes_no_row_still_says_so(tmp_path):
    """The other half of finding #6, so the fix cannot become "the harness always reads a database".

    A caller that hands `raw=` text writes no `system_state` row at all: there is nothing stored for
    the deck to read, so the stamp stays an argument of the injected reader, and the render helper's
    docstring is allowed to say so. What this test pins is that the documented exception is real —
    the deck renders the row it was handed and the reader is the stand-in, not the shipped function.
    """
    html, _fake, ns = _g5b_render_deck(tmp_path, raw=json.dumps(_G5B_LIVE_PAYLOAD),
                                       updated_at=_G5B_FRESH_UPDATED_AT)
    reader = ns["query_system_state_row"]
    assert getattr(reader, "__name__", "") != "query_system_state_row", (
        "the documented injection-only route silently became a database read")
    assert reader(_MARKOV_KEY)[1] == _G5B_FRESH_UPDATED_AT, reader(_MARKOV_KEY)
    assert "60s old" in html, html[:1500]
    assert "Live Forward Filter" in _g5b_heading(html), _g5b_heading(html)


def test_g5b_the_shipped_reader_returns_two_empty_strings_for_an_absent_key(tmp_path):
    """An absent row is `("", "")` — not `(None, None)`, which the deck's `published_at` parse would
    read as a present-but-unusable stamp instead of nothing at all."""
    _g5_system_state_row(tmp_path, payload={"is_tradable": True}, key=_MARKOV_KEY,
                         updated_at=_G5B_LIVE_UPDATED_AT)
    reader = _g5b_extract_row_reader()
    failures = _g5b_bind_reader(reader, tmp_path / _G5B_FIXTURE_DB_NAME)
    got = reader("markov_regime_telemetry_nobody_writes")
    assert got == ("", ""), f"the absent shape drifted from the two empty strings: {got!r}"
    assert all(isinstance(one, str) for one in got), got
    assert failures == {}, failures


def test_g5b_the_shipped_reader_records_the_failure_it_cannot_read(tmp_path):
    """The `except` path, executed: a database with no `system_state` table must come back as the
    same `("", "")` absence AND announce itself in `FEED_FAILURES`, because a quiet absence is how
    the deck ended up printing a 35-hour-old row as a live forward filter."""
    import sqlite3

    db = tmp_path / "no_system_state_table.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("CREATE TABLE other_things (n INTEGER)")
        conn.commit()
    finally:
        conn.close()

    reader = _g5b_extract_row_reader()
    failures = _g5b_bind_reader(reader, db)
    assert reader(_MARKOV_KEY) == ("", ""), "a read that raised did not degrade to the absent shape"
    assert len(failures) == 1, f"the failed read recorded {len(failures)} feed failures: {failures}"
    feed_key = next(iter(failures))
    assert feed_key == f"read \u2192 {db.name}:{_MARKOV_KEY}", feed_key
    assert "OperationalError" in failures[feed_key], failures[feed_key]


def test_g5b_the_shipped_reader_opens_read_only_and_creates_no_database(tmp_path):
    """`connect_readonly` opens `file:...?mode=ro`, so a path with no database behind it is an
    absence, not a file the test just created. This is the one half of "the artefacts are untouched"
    a reader test can actually prove: the live `trading_data.db` is never the path under test."""
    missing = tmp_path / "was_never_created.db"
    reader = _g5b_extract_row_reader()
    failures = _g5b_bind_reader(reader, missing)
    assert reader(_MARKOV_KEY) == ("", ""), "a missing database did not degrade to an absence"
    assert not missing.exists(), "the reader CREATED a database, so its open is not read-only"
    assert len(failures) == 1, f"an unreadable database was not announced: {failures}"


# ── B-2: no model may be run to fill in a missing measurement ────────────────────────────
@pytest.mark.parametrize("banned,code_line", _G5B_STAND_IN_CASES)
def test_g5b_no_fallback_or_stand_in_survives_in_the_deck(banned, code_line):
    """Finding #9: the target is the deck block AND the reader helper.

    B-5 moved every payload read out of the block and into `_markov_regime_readings`, so scanning the
    block alone left the code that actually decides what the deck prints unswept: the gap was latent
    (the reviewer hand-checked all 15 tokens against the helper and none is present today), and a
    `m_tel.get("is_tradable", True)` written into the reader tomorrow would have kept this green.
    """
    hits = _g5_banned_hits(_g5b_deck_and_helper_code(), [banned])
    assert hits == [], f"the Markov deck still contains {banned!r}"


@pytest.mark.parametrize("banned,code_line", _G5B_STAND_IN_CASES)
def test_g5b_the_stand_in_scan_reaches_the_reader_helper(banned, code_line):
    """Both halves of the widening: the shipped deck is clean, and the same token seeded into the
    shipped reader helper's own body is caught by the target the guard above actually uses.

    The second assertion is the one that makes the first mean anything — a scan over a concatenation
    that quietly dropped the helper would look identical from the outside. It also records the defect
    it repaired: the block alone did not see a stand-in written into the reader.
    """
    seeded = _g5b_helper_seeded_with(code_line)
    assert banned not in _g5b_block_code(), (
        f"{banned!r} is in the deck block itself, so the block-only half below proves nothing")
    assert _g5_banned_hits(_g5b_block_code(), [banned]) == [], (
        f"the block-only scan already trips on {banned!r}; this is not the pre-fix gap being closed")
    assert _g5_banned_hits(_g5b_deck_and_helper_code(), [banned]) == [], (
        f"the shipped deck or its reader helper still contains {banned!r}")
    hits = _g5_banned_hits(_g5b_deck_and_helper_code(helper=seeded), [banned])
    assert hits == [banned], (
        f"a stand-in written into the reader helper's body is invisible to the deck-wide scan, so "
        f"the {banned!r} guard is still block-only")


@pytest.mark.parametrize("banned,code_line", _G5B_STAND_IN_CASES + _G5B_PROSE_CLAIMS)
def test_g5b_a_comment_neither_trips_nor_hides_a_markov_scan(banned, code_line):
    """Both halves of the prose immunity, on this deck's own tokens: prose about the deleted
    fallback must not fail the guard, and the same text really in the code must.

    Finding #4 added the docstring halves. A `#` comment was the only prose the scan knew how to
    ignore, so a docstring naming a banned token could trip a negative scan all by itself — and, the
    other way round, a scan that a docstring satisfies is a test that cannot fail. Immunity has to
    mean both kinds of prose, and neither of them may hide real code.
    """
    prose = (f"def f():\n    # the deck used to publish {banned} here, and {banned} is "
             f"still what the note says\n    return 1\n")
    assert _g5_banned_hits(prose, [banned]) == [], (
        f"a comment quoting {banned!r} failed a negative scan against code that does not "
        "contain it")
    documented = (f'def f():\n    """the deck used to publish {banned} here, and {banned} is '
                  f'still what the note says. """\n    return 1\n')
    assert _g5_banned_hits(documented, [banned]) == [], (
        f"a docstring quoting {banned!r} failed a negative scan against code that does not "
        "contain it")
    real = f"def f():\n    {code_line}\n    return 1\n"
    assert _g5_banned_hits(real, [banned]) == [banned], (
        f"{banned!r} is in the code and the stripped scan no longer sees it")
    hidden = f"def f():\n    {code_line}  # {banned}\n    return 1\n"
    assert _g5_banned_hits(hidden, [banned]) == [banned], (
        f"a comment beside {banned!r} in the code hid it from the scan")
    documented_and_real = (f'def f():\n    """{banned} in the prose. """\n'
                           f"    {code_line}\n    return 1\n")
    assert _g5_banned_hits(documented_and_real, [banned]) == [banned], (
        f"a docstring beside {banned!r} in the code hid it from the scan")


def test_g5b_the_matrix_caption_claims_publication_not_estimation(tmp_path):
    """Finding #5: `process_bars` has two publish paths and only one of them estimates.

    The short-frame path (`markov_regime.py:393-413`) emits `self.P.tolist()` (`:408`) — the prior
    from `:39-43`, the very `0.70/0.15/0.15` object B-4 deleted — and never calls
    `estimate_transition_matrix`, which runs only on the >=15-bar path (`:416`). Nothing in the row
    says which path wrote it, so a measured matrix cannot claim to have been estimated from labelled
    bars. The deck may say it published; it may not say how it came to be.

    Banned over the deck block AND the reader helper together, with docstrings stripped (fix #4),
    so neither a caption string nor a piece of prose about the old caption can satisfy it.
    """
    deck_code = _g5b_block_code() + "\n" + _g5b_helper_code()
    assert _g5_banned_hits(deck_code, ["estimated from labelled bars"]) == [], (
        "the deck still claims a matrix provenance the row cannot support")
    # What guards the caption's provenance claim is the code-layer ban above -- the shipped caption
    # is a string literal, so it is code -- plus the rendered-caption assertions below. It is not
    # guarded by a scan of comment prose: `_g5_code_only` blanks comments by design (finding #4), so
    # such a scan could never fire and an honest comment about the old caption must stay legal.

    html, _fake, ns = _g5b_render_deck(tmp_path, payload=_G5B_LIVE_PAYLOAD)
    assert ns["m_matrix_st"]["measured"] is True, "this case must be the measured-matrix branch"
    captions = [one for one in html.splitlines() if "Transition matrix" in one]
    assert captions, f"the deck published no matrix caption:\n{html[:1500]}"
    for one in captions:
        assert "estimated" not in one.lower(), one
    assert "not recoverable" in "\n".join(captions), (
        "the caption names neither the provenance nor its absence: " + "\n".join(captions))


def test_g5b_an_unmeasured_matrix_still_says_so(tmp_path):
    """The other half of the caption: with no matrix published the deck must not print the
    publication sentence at all, recoverable or otherwise."""
    html = _g5b_render_deck(tmp_path, payload={**_G5B_LIVE_PAYLOAD, "transition_matrix": None})[0]
    assert "not recoverable" not in html, "a caption about a published matrix printed without one"
    assert "Transition matrix not measured" in html, html[:1500]


def test_g5b_no_path_in_the_deck_can_construct_a_markov_regime_engine():
    """`MarkovRegimeEngine.__init__` (`markov_regime.py:33-49`) is pure — uniform `pi`, the prior
    `P`, no dataset — so every figure the fallback produced was the model's assumption about the
    world, not a measurement of this one, and it was printed under a live heading."""
    src = _g5_code_only(DASHBOARD.read_text(encoding="utf-8"))
    for token in ("MarkovRegimeEngine", "from markov_regime import", "markov_regime import"):
        assert token not in src, (
            f"dashboard.py can still reach {token!r}: the deck may not run a model to fill a gap, "
            "and a second importer would need the same age bound")
    block = _g5b_block_code()
    assert len(block.splitlines()) >= 120, (
        f"the extracted deck is {len(block.splitlines())} lines, so this is not the deck")


def test_g5b_an_absent_row_does_not_invent_a_deck(tmp_path):
    """The literals the fallback asserted, none of which any engine measured: `CHOP 0.60 /
    BULL 0.25 / BEAR 0.15`, durations 3.33, breakdown risk 0.35, uniform 0.333 equilibrium."""
    html = _g5b_render_deck(tmp_path, raw="", updated_at="")[0]
    for invented in ("60.0%", "3.3 Bars", "0.333", "35.0%", "33%/33%/33%", "25.0%", "15.0%"):
        assert invented not in html, f"the deck still prints the fabricated {invented}"
    assert "not measured" in html
    assert _G5B_ABSENT_COLOUR in html, "an absent deck is not dimmed to the absence colour"


def test_g5b_the_dashboard_carries_no_copy_of_the_engine_prior():
    """B-4: `default_P` was a byte-for-byte copy of `markov_regime.py:39-43`, which would silently
    diverge when the prior changed. `transition_matrix` is published on both producer paths, so the
    only row without it is a row that does not exist — and that renders one absent state."""
    src = _g5_code_only(DASHBOARD.read_text(encoding="utf-8"))
    assert "default_P" not in src, "the deck still names a stand-in transition matrix"
    assert "0.70, 0.15, 0.15" not in src, (
        "the engine's prior transition matrix is duplicated in the dashboard")
    producer_src = MARKOV_REGIME.read_text(encoding="utf-8")
    assert "0.70, 0.15, 0.15" in producer_src, (
        "markov_regime.py no longer carries the prior this ban is about, so the scan is vacuous")


# ── B-3: the key names and shapes the producer actually publishes ────────────────────────
def test_g5b_the_fixture_keys_are_the_ten_the_producer_publishes():
    """Producer-derived fixture, or the whole deck could be pinned to a name nobody writes."""
    published = _g5b_producer_payload_keys()
    assert len(published) == 10, sorted(published)
    assert frozenset(_G5B_LIVE_PAYLOAD) == published, (
        f"the deck's fixture drifted from process_bars: extra "
        f"{sorted(set(_G5B_LIVE_PAYLOAD) - published)}, missing "
        f"{sorted(published - set(_G5B_LIVE_PAYLOAD))}")
    assert "forecast_probabilities" not in published, (
        "the deck's dead read key is now a real one, so B-3 needs re-reading")
    assert "timestamp" not in published, (
        "the payload now carries its own timestamp, so the row's updated_at is not the only age")
    assert set(_G5B_PAYLOAD_FIELDS) == published, (
        "the payload-key to deck-field map no longer covers the producer's contract")


def test_g5b_the_regime_names_are_the_engines_own():
    states = _g5b_producer_states()
    assert states == ("CHOP", "BULL_EXPANSION", "BEAR_CASCADE"), states
    roster_src = _g5b_helper_code()
    for state in states:
        assert f'"{state}"' in roster_src, (
            f"the deck no longer reads the {state} probability the producer publishes")
    declared = re.findall(r'_MARKOV_STATE_KEYS = \((.*?)\n    \)', roster_src, re.S)
    assert declared, "the deck's regime roster is not a named table this test can read"
    pairs = re.findall(r'\("([a-z]+)", "([A-Z_]+)"\)', declared[0])
    assert [state for _, state in pairs] == list(states), (
        f"the deck's regime names {[s for _, s in pairs]} are not the engine's {states}")


def test_g5b_the_deck_reads_the_forecast_key_the_producer_publishes():
    """`:4779` read `forecast_probabilities`, a name that exists only inside `forecast_k_steps`
    (`markov_regime.py:360`) and never in the published payload; `process_bars` writes it as
    `forecast_4step` (`:407`, `:454`)."""
    block = _g5b_block_code()
    helper = _g5b_helper_code()
    assert '"forecast_4step"' in helper, "the reader does not name the published forecast key"
    for where, label in ((block, "the deck"), (helper, "the reader")):
        assert '"forecast_probabilities"' not in where, (
            f"{label} still reads forecast_probabilities, which no producer publishes")
    assert '"stationary_distribution"' in helper, "the reader stopped naming the equilibrium key"
    assert '"state_probabilities"' in helper and '"expected_durations_bars"' in helper


def test_g5b_a_dict_shaped_stationary_distribution_is_not_read_as_measured():
    """The producer publishes a LIST of three rounded floats (`markov_regime.py:409`, `:456`). Only
    the dict branch of the old parser could ever be reached — by the B-2 fallback that invented a
    dict — so a dict is a shape this deck may not present as a measurement."""
    as_list = _g5b_readings('{"stationary_distribution": [0.3703, 0.543, 0.0868]}')
    assert as_list["stat_chop"]["measured"] is True, as_list["stat_chop"]
    assert as_list["stat_chop"]["value"] == 0.3703, as_list["stat_chop"]
    assert as_list["equilibrium"]["measured"] is True
    assert as_list["equilibrium"]["value"] == [0.3703, 0.543, 0.0868], as_list["equilibrium"]
    as_dict = _g5b_readings('{"stationary_distribution": {"CHOP": 0.4, "BULL_EXPANSION": 0.4,'
                            ' "BEAR_CASCADE": 0.2}}')
    for field in _G5B_STAT_FIELDS:
        assert as_dict[field]["measured"] is False, (field, as_dict[field])
        assert "stationary_distribution" in as_dict[field]["reason"], as_dict[field]
    assert as_dict["equilibrium"]["measured"] is False
    for junk in (None, "", "[]", "[1, 2]", '"0.33"', "7", "{}", '[0.1]', '[[0.1, 0.2]]'):
        out = _g5b_readings(junk, updated_at=_G5B_LIVE_UPDATED_AT)
        if "stat_chop" in out:
            assert out["stat_chop"]["measured"] is False, (junk, out["stat_chop"])


# ── B-5: absence is not a zero, and a zero is not an absence ─────────────────────────────
@pytest.mark.parametrize("field", sorted(_G5B_FIELDS))
def test_g5b_an_empty_row_publishes_every_figure_as_its_own_absence(field):
    """Each reason names the missing reading, so the operator learns which key the engine did not
    write instead of being shown a number nobody took."""
    state = _g5b_readings("{}")[field]
    assert state["measured"] is False, (field, state)
    assert state["reason"] == _G5B_FIELD_REASONS[field], (field, state["reason"])
    if field in _G5B_VECTOR_FIELDS:
        assert state.get("rows") in (None, (), []), (field, state)
    elif field not in _G5B_TEXT_FIELDS and field != "is_tradable":
        assert state["value"] is None, (field, state)


@pytest.mark.parametrize("field", sorted(_G5B_NUMBER_FIELDS))
def test_g5b_a_measured_zero_probability_stays_a_reading(field):
    """A probability of 0.0 is reachable — the live row carries `BEAR_CASCADE: 0.0002` — and
    `float(raw or 0.0)` erases the difference between that and a missing key. `-0.0` is falsy too."""
    payload = {one_field: 0.0 for one_field in _G5B_NUMBER_FIELDS}
    state = _g5b_readings(_g5b_payload_for(payload))[field]
    assert state["measured"] is True, (field, state)
    assert state["value"] == 0.0, (field, state)
    signed = _g5b_readings(_g5b_payload_for({field: -0.0}))[field]
    assert signed["measured"] is True, (field, signed)


def _g5b_payload_for(overrides):
    """A payload carrying exactly the keys the deck reads these fields out of.

    Feeding `{"prob_chop": 0.0}` to the reader would be testing a key the producer never writes;
    the override map is keyed by DECK field and translated back to payload keys here.
    """
    import json as _json

    payload = {}
    for deck_field, value in overrides.items():
        payload_field = _G5B_FIELD_TO_PAYLOAD.get(deck_field)
        assert payload_field, f"{deck_field} is not a field this map can translate"
        key, state_name = payload_field
        if isinstance(state_name, int):
            # `stationary_distribution` is published as a LIST of three rounded floats
            # (`markov_regime.py:409`, `:456`), so the vector fields are addressed by index. The
            # generic branch below builds a dict, which is precisely the shape
            # `test_g5b_a_dict_shaped_stationary_distribution_is_not_read_as_measured` forbids: a
            # dict-built fixture would have made the zero-reading case pass by reading a shape no
            # producer publishes. Untouched slots are JSON nulls, i.e. absences, not zeros.
            slot = payload.setdefault(key, [None, None, None])
            slot[state_name] = value
        else:
            payload.setdefault(key, {})
            if state_name is None:
                payload[key] = value
            else:
                payload[key][state_name] = value
    return _json.dumps(payload)


_G5B_FIELD_TO_PAYLOAD = {
    "prob_chop": ("state_probabilities", "CHOP"),
    "prob_bull": ("state_probabilities", "BULL_EXPANSION"),
    "prob_bear": ("state_probabilities", "BEAR_CASCADE"),
    "dur_chop": ("expected_durations_bars", "CHOP"),
    "dur_bull": ("expected_durations_bars", "BULL_EXPANSION"),
    "dur_bear": ("expected_durations_bars", "BEAR_CASCADE"),
    "fc_chop": ("forecast_4step", "CHOP"),
    "fc_bull": ("forecast_4step", "BULL_EXPANSION"),
    "fc_bear": ("forecast_4step", "BEAR_CASCADE"),
    "stat_chop": ("stationary_distribution", 0),
    "stat_bull": ("stationary_distribution", 1),
    "stat_bear": ("stationary_distribution", 2),
    "breakdown_risk": ("breakdown_risk_4step", None),
    "sizing_multiplier": ("sizing_multiplier", None),
    "is_tradable": ("is_tradable", None),
    "dominant_regime": ("dominant_regime", None),
    "tradability_reason": ("tradability_reason", None),
    "transition_matrix": ("transition_matrix", None),
}


def test_g5b_the_live_payload_measures_every_figure_the_deck_prints():
    """The live ten-key row leaves the deck with nothing to disown: an absence shown against this
    payload is a reading the panel failed to take."""
    readings = _g5b_readings(_MARKOV_JSON_TEXT)
    unmeasured = sorted(field for field in readings["field_names"]
                        if not readings[field]["measured"])
    assert unmeasured == [], unmeasured
    assert readings["coverage"] == {"total": len(_G5B_FIELDS), "unmeasured": 0}, readings
    assert readings["dominant_regime"]["text"] == "CHOP"
    assert readings["prob_bear"]["value"] == 0.0002, readings["prob_bear"]
    assert readings["sizing_multiplier"]["value"] == 0.64
    assert readings["breakdown_risk"]["value"] == 0.6672
    assert readings["is_tradable"]["value"] is True
    assert readings["tradability_reason"]["text"] == "Tradable: E[D]=10.4b >= 3.5b"
    assert readings["confidence"]["value"] == 0.8878, readings["confidence"]
    assert readings["dominant_duration"]["value"] == 10.4, readings["dominant_duration"]


def test_g5b_a_partial_row_loses_only_the_fields_it_loses():
    """Per-field absence, not a bundled one: dropping the forecast must not disown the probabilities
    that were published, and must not leave the forecast at a stand-in zero."""
    readings = _g5b_readings(_json_text_of(_g5b_payload_without("forecast_4step",
                                                               "stationary_distribution")))
    for field in _G5B_FC_FIELDS + _G5B_STAT_FIELDS + ("equilibrium",):
        assert readings[field]["measured"] is False, (field, readings[field])
    for field in _G5B_PROB_FIELDS + ("breakdown_risk", "sizing_multiplier", "is_tradable"):
        assert readings[field]["measured"] is True, (field, readings[field])
    assert readings["coverage"]["unmeasured"] == 7, readings["coverage"]


def _json_text_of(payload):
    import json as _json

    return _json.dumps(payload)


def test_g5b_the_sizing_multiplier_is_never_defaulted_to_its_highest_value():
    """`super_bot.py:919-924` defaults a missing sizing multiplier to 0.50 defensive. The HUD
    defaulted it to 1.0, which is the largest figure the producer can emit
    (`markov_regime.py:327` bounds it to [0.20, 1.0]) — the riskiest possible stand-in."""
    low, high = _g5b_sizing_bound_from_producer()
    assert (low, high) == (0.2, 1.0), (
        f"the producer's sizing clamp moved to {(low, high)}; re-read the bound before writing a "
        "test about what an absence may not be defaulted to")
    html = _g5b_render_deck(payload=_g5b_payload_without("sizing_multiplier"))[0]
    assert "1.00x" not in html, "an absent sizing multiplier still prints as the maximum"
    assert "(no sizing_multiplier in markov regime telemetry)" in html, html[:2000]
    assert "0.64x" not in html, html[:2000]
    measured_html = _g5b_render_deck(payload=_G5B_LIVE_PAYLOAD)[0]
    assert "0.64x" in measured_html, measured_html[:2000]


# Finding #10 fixture: a producer whose posterior weights sit OUTSIDE its own clamp. It is a string,
# never written to disk and never imported; it exists so the extractor can be shown to read the
# clamp rather than every float in the getter.
_G5B_SIZING_CLAMP_FIXTURE = '''
class MarkovRegimeEngine:

    def get_regime_sizing_multiplier(self) -> float:
        """multiplier = pi[BULL]*1.0 + pi[CHOP]*2.50 + pi[BEAR]*-1.25, bounded [0.20, 1.0]."""
        p_chop = float(self.pi[0])
        p_bull = float(self.pi[1])
        p_bear = float(self.pi[2])
        mult = (p_bull * 1.0) + (p_chop * 2.50) + (p_bear * -1.25)
        return round(float(max(0.20, min(1.0, mult))), 2)
'''


def test_g5b_the_sizing_bound_is_read_from_the_clamp_not_from_every_float():
    """Finding #10: `min`/`max` over all the floats in the producer's getter is not its clamp.

    `markov_regime.py:326` weights the posterior with `1.0`, `0.60` and `0.20` and `:327` clamps the
    result with `max(0.20, min(1.0, mult))`. Reading every float answered correctly today only
    because each weight happens to fall inside `[0.20, 1.0]`; a producer that widened the weights or
    narrowed the clamp would have moved one number and hidden the other. The pair now comes off the
    arguments of the two call nodes, and the fixture pins the difference: for a getter whose weights
    are `2.50` and `-1.25` the old sweep returns `[0.20, 2.50]` while the clamp is still `[0.20,
    1.0]`.
    """
    import ast

    low, high = _g5b_sizing_bound_from_producer()
    assert (low, high) == (0.2, 1.0), (
        f"the producer's clamp reads {(low, high)}, not the [0.20, 1.0] at markov_regime.py:327")

    # The getter really does contain floats that are not the clamp, so a sweep and a clamp-reading
    # are different numbers on the shipped file, not only on the fixture.
    tree = ast.parse(MARKOV_REGIME.read_text(encoding="utf-8"), filename=str(MARKOV_REGIME))
    getter = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == "get_regime_sizing_multiplier")
    every_float = sorted({n.value for n in ast.walk(getter)
                          if isinstance(n, ast.Constant) and isinstance(n.value, float)})
    assert 0.60 in every_float, (
        "the pi weights this suite used to sweep are gone, so the fixture half has to be re-read")
    assert 0.60 not in (low, high), (
        f"the extracted bound {(low, high)} picked up the pi weight 0.60, so it is a sweep of every "
        "float and not the clamp")

    swept = sorted({n.value for n in ast.walk(ast.parse(_G5B_SIZING_CLAMP_FIXTURE))
                    if isinstance(n, ast.Constant) and isinstance(n.value, float)})
    assert (min(swept), max(swept)) != (0.2, 1.0), (
        "the fixture no longer separates a float sweep from the clamp, so it proves nothing")
    assert _g5b_sizing_bound_from_producer(_G5B_SIZING_CLAMP_FIXTURE) == (0.2, 1.0), (
        "the extractor read the fixture's weights instead of its clamp, so a producer-side bound "
        "change would move the wrong number")


def test_g5b_a_row_that_is_not_a_json_object_publishes_absences_and_never_raises():
    """`json.loads` hands back a list, a string or an int as readily as a dict, and a crashed panel
    publishes nothing at all — which the brief counts as worse than an honest absence."""
    for junk in (None, "", "[]", "[1, 2, 3]", '"CHOP"', "7", "not json at all", "{}"):
        html = _g5b_render_deck(payload=junk, updated_at=_G5B_LIVE_UPDATED_AT)[0]
        assert "not measured" in html, (junk, html[:1500])
        assert "Live" not in _g5b_heading(html), (junk, _g5b_heading(html))
        assert _G5B_ALL_CLEAR_COLOUR not in html, (junk, html[:1500])


# ── Finding #1 (Critical): the same claim, per PUBLISHED KEY ──────────────────────────────
# The battery above could never reach the crash, because every one of its payloads fails to parse as
# a dict, so `dominant_regime` was never measured and the reader never looked anything up under it.
# These rows ARE dicts and their fields ARE published — they just hold values no producer can write.
_G5B_JUNK_PER_KEY_VALUES = (7, "x", [], {}, None, ["CHOP"], {"a": 1}, "RANGE")

# Each card's colour variable and the one reading that may earn it. `_G5B_PAYLOAD_FIELDS` is pinned
# against `markov_regime.py`'s AST by `test_g5b_the_fixture_keys_are_the_ten_the_producer_publishes`,
# so this battery runs over the producer's real contract and not over a transcription of it.
_G5B_COLOUR_TO_FIELD = (
    ("m_dom_color", "dominant_regime"),
    ("m_conf_color", "confidence"),
    ("m_domdur_color", "dominant_duration"),
    ("m_sizing_color", "sizing_multiplier"),
    ("m_stat_color", "equilibrium"),
    ("m_matrix_color", "transition_matrix"),
    ("m_bd_color", "breakdown_risk"),
    ("m_tradable_color", "is_tradable"),
)


@pytest.mark.parametrize("payload_key,junk_value", [
    (one_key, one_junk)
    for one_key in sorted(_G5B_PAYLOAD_FIELDS)
    for one_junk in _G5B_JUNK_PER_KEY_VALUES
])
def test_g5b_a_junk_value_under_any_published_key_publishes_absences_and_never_raises(
        payload_key, junk_value):
    """Finding #1: one published key holding junk may not take the panel down.

    The reader looked the dominant regime's duration up with a subscript
    (`short_by_state[str(dom_st["text"]).strip()]`), so a row whose `dominant_regime` parses as a
    measured state string that names none of the engine's three states raised `KeyError` before any
    card was built — and the pre-fix code it replaced (`durations.get(dom_regime, 3.3)`) could not
    raise at all, so the fix traded a stand-in for a crash. Brief constraint 9: a crashed Streamlit
    panel publishes nothing, which is worse than an honest absence, and the `with tab_risk:` block
    aborting costs the other panels in the same run their feed too, with no `FEED_FAILURES` entry to
    say why.

    Both halves, for each of the producer's ten keys against each junk value: the reader returns and
    the shipped deck renders. The render then wears no green all-clear, no `TRADABLE`, no `Live`, and
    no measured colour on any field it did not measure — a junk value that really is a number (a
    `sizing_multiplier` of 7) keeps its own colour, because dimming a published reading is the
    defect `test_g5b_a_measured_cascade_risk_keeps_its_own_rung_and_thresholds` forbids.

    The stamp is the stale live one throughout, so the "no Live" half is a claim about the junk row
    and not about freshness; freshness has its own tests above and below.
    """
    import json as _json

    row = {payload_key: junk_value}
    text = _json.dumps(row)
    readings = _g5b_readings(text, updated_at=_G5B_LIVE_UPDATED_AT)
    assert readings["dominant_duration"]["measured"] is False, (text, readings["dominant_duration"])
    assert "dominant_regime" in readings["dominant_duration"]["reason"] or \
        "duration" in readings["dominant_duration"]["reason"], readings["dominant_duration"]
    assert readings["coverage"]["unmeasured"] >= 1, readings["coverage"]

    html, _fake, deck_ns = _g5b_render_deck(payload=row, updated_at=_G5B_LIVE_UPDATED_AT)
    assert "not measured" in html, (text, html[:1500])
    assert "Live" not in _g5b_heading(html), (text, _g5b_heading(html))
    assert "(Live Forward Filter)" not in html, text
    assert _G5B_ALL_CLEAR_COLOUR not in html, (text, html[:2000])
    assert "TRADABLE" not in html.replace("UNTRADEABLE", ""), (text, html[:2000])
    for colour_var, field in _G5B_COLOUR_TO_FIELD:
        if not readings[field]["measured"]:
            assert deck_ns[colour_var] == _G5B_ABSENT_COLOUR, (
                f"{field} was never measured from {text!r}, yet the deck paints its card with "
                f"{deck_ns[colour_var]!r}")


def test_g5b_an_unknown_dominant_regime_names_its_own_absence_reason():
    """Finding #1's other half: the three ways `dominant_duration` can fail must not collapse into
    one sentence. A row that never named a regime points at the producer's key set; a row that named
    a state the engine does not define points at the producer's vocabulary; a row that named a real
    state but published no duration for it points at the duration map. Only the second of those used
    to raise, and an absence the operator cannot tell the others apart from is no use at all."""
    unknown = _g5b_readings('{"dominant_regime": "RANGE", "state_probabilities": '
                            '{"CHOP": 0.4, "BULL_EXPANSION": 0.3, "BEAR_CASCADE": 0.3}}')
    assert unknown["dominant_regime"]["measured"] is True, unknown["dominant_regime"]
    assert unknown["dominant_duration"]["measured"] is False, unknown["dominant_duration"]
    assert "names none of the engine's states" in unknown["dominant_duration"]["reason"], \
        unknown["dominant_duration"]
    assert "dominant_duration" in unknown["unmeasured_fields"], unknown["unmeasured_fields"]
    assert "dominant_regime" not in unknown["unmeasured_fields"], unknown["unmeasured_fields"]

    no_regime = _g5b_readings("{}")
    assert no_regime["dominant_duration"]["reason"] == (
        _G5B_FIELD_REASONS["dominant_duration"]), no_regime["dominant_duration"]

    named_but_no_duration = _g5b_readings('{"dominant_regime": "CHOP"}')
    assert named_but_no_duration["dominant_duration"]["measured"] is False
    assert named_but_no_duration["dominant_duration"]["reason"] == (
        _G5B_FIELD_REASONS["dominant_duration"]), named_but_no_duration["dominant_duration"]

    measured = _g5b_readings(_MARKOV_JSON_TEXT)
    assert measured["dominant_duration"] == {"value": 10.4, "measured": True, "reason": ""}, \
        measured["dominant_duration"]

    html = _g5b_render_deck(payload='{"dominant_regime": "RANGE"}',
                            updated_at=_G5B_LIVE_UPDATED_AT)[0]
    assert "names none of the engine&#x27;s states" in html or \
        "names none of the engine's states" in html, html[:2000]
    assert "not measured" in html and _G5B_ALL_CLEAR_COLOUR not in html, html[:2000]


# ── B-6: the verdicts may not fail open ──────────────────────────────────────────────────
def test_g5b_a_payload_without_is_tradable_prints_no_all_clear(tmp_path):
    """The single most dangerous line in the panel: `bool(m_tel.get("is_tradable", True))` turned a
    missing key into the green badge, while `super_bot.py:733` logs a key-missing warning and
    skips. This is the RED-against-`df30aa3` proof for B-6."""
    html = _g5b_render_deck(tmp_path, payload=_g5b_payload_without("is_tradable"))[0]
    assert _G5B_ALL_CLEAR_COLOUR not in html, (
        f"an unmeasured tradability gate still wears the green all-clear: {html[:2000]}")
    assert "TRADABLE" not in html.replace("UNTRADEABLE", ""), html[:2000]
    assert "GATE NOT MEASURED" in html, html[:2000]
    assert _G5B_ABSENT_COLOUR in html, html[:2000]
    assert "(no is_tradable in markov regime telemetry)" in html, html[:2000]


def test_g5b_a_measured_untradable_gate_still_prints_the_red_verdict(tmp_path):
    """The fix may not be "always grey": a published false is a measurement and keeps its colour."""
    html = _g5b_render_deck(tmp_path, payload={**_G5B_LIVE_PAYLOAD, "is_tradable": False,
                                               "tradability_reason":
                                                   "Expected duration E[D]=2.0b < 3.5b threshold "
                                                   "(fee churn risk)"})[0]
    assert "UNTRADEABLE CHURN" in html, html[:2000]
    assert "#ff003c" in html, "a measured veto lost its alarm colour"
    assert "E[D]=2.0b &lt; 3.5b" in html or "E[D]=2.0b < 3.5b" in html, html[:2000]
    assert _G5B_ALL_CLEAR_COLOUR not in html, html[:2000]


def test_g5b_a_measured_tradable_gate_prints_the_engines_own_words(tmp_path):
    """A gate the engine measured true, on a row inside the fleet's age bound, keeps the all-clear.

    The render moved to a fresh stamp for finding #2: `_G5B_LIVE_UPDATED_AT` is ~35 h old and the
    deck now refuses to paint a stale row green, so this case — which used to run on the stale live
    stamp and assert the green — could only keep its claim by asking about a row the fleet still
    trades. Its name and every assertion are otherwise unchanged, and the stale half that replaces
    the old expectation is `test_g5b_a_stale_row_takes_the_green_off_the_tradability_gate`.
    """
    html = _g5b_render_deck(tmp_path, updated_at=_G5B_FRESH_UPDATED_AT)[0]
    assert "TRADABLE" in html, html[:2000]
    assert _G5B_ALL_CLEAR_COLOUR in html, "a measured all-clear was dimmed away"
    assert "Tradable: E[D]=10.4b &gt;= 3.5b" in html or "Tradable: E[D]=10.4b >= 3.5b" in html, \
        html[:2000]
    assert "(no tradability_reason in markov regime telemetry)" not in html


def test_g5b_a_stale_row_takes_the_green_off_the_tradability_gate(tmp_path):
    """Finding #2: brief B-1 requires the heading AND every verdict derived from the row to say the
    row is refused. The heading obeyed; the gate did not. Rendered with the transcribed live row and
    its real stamp, the deck printed `(STALE Forward Filter — row past the fleet age bound)` beside
    `[TRADABLE]` in `#00ff66` with `coverage {'total': 21, 'unmeasured': 0}` — a green all-clear
    claiming a currency the measurement does not have, on data all four automated consumers refuse
    (`inflight_shepherd.py:614`, `super_bot.py:733`, both on `age > max_age`).

    This card IS the deck's combined verdict, so the staleness bundling brief:252 allows for a
    verdict applies here: a true gate on a refused row goes to the stale amber and says so.
    """
    html, _fake, ns = _g5b_render_deck(tmp_path)
    assert ns["m_r"]["stale"] is True, ns["m_r"]["stale"]
    assert ns["m_gate_st"]["measured"] is True, "the gate must stay measured; only its verdict moves"
    assert ns["m_tradable_color"] != _G5B_ALL_CLEAR_COLOUR, (
        f"a row the fleet refuses still wears the all-clear: {ns['m_tradable_color']}")
    assert _G5B_ALL_CLEAR_COLOUR not in html, (
        f"the stale deck prints the green all-clear anyway: {html[:2500]}")
    assert ns["m_tradable_color"] == "#ffcc00", ns["m_tradable_color"]
    assert "TRADABLE (STALE ROW)" in html, html[:2500]
    assert "STALE" in _g5b_heading(html), _g5b_heading(html)


def test_g5b_a_row_without_a_publish_stamp_does_not_borrow_the_all_clear(tmp_path):
    """The other freshness failure: with no usable `updated_at` the row's age is unknown, not proven
    inside the bound, so the verdict it feeds may not wear the colour of a fresh one either."""
    html, _fake, ns = _g5b_render_deck(tmp_path, updated_at="")
    assert ns["m_r"]["age_known"] is False, ns["m_r"]["age_known"]
    assert ns["m_gate_st"]["measured"] is True, ns["m_gate_st"]
    assert _G5B_ALL_CLEAR_COLOUR not in html, html[:2500]
    assert ns["m_tradable_color"] == "#ffcc00", ns["m_tradable_color"]
    assert "TRADABLE (FRESHNESS NOT MEASURED)" in html, html[:2500]
    assert "TRADABLE (STALE ROW)" not in html, (
        "a row with no stamp at all is being called stale — those are different states")


def test_g5b_a_fresh_row_still_earns_the_green_tradable_all_clear(tmp_path):
    """The other half, or the fix is just "never print green". Inside the bound a measured true gate
    keeps `#00ff66` and the bare TRADABLE badge."""
    html, _fake, ns = _g5b_render_deck(tmp_path, updated_at=_G5B_FRESH_UPDATED_AT)
    assert ns["m_r"]["stale"] is False and ns["m_r"]["age_known"] is True
    assert ns["m_tradable_color"] == _G5B_ALL_CLEAR_COLOUR, ns["m_tradable_color"]
    assert ns["m_tradable_badge"] == "TRADABLE", ns["m_tradable_badge"]
    assert "STALE ROW" not in html and "FRESHNESS NOT MEASURED" not in html, html[:2500]
    assert "Live Forward Filter" in _g5b_heading(html), _g5b_heading(html)


def test_g5b_a_stale_row_leaves_a_measured_veto_and_an_unmeasured_gate_alone(tmp_path):
    """The two branches finding #2 must NOT touch: a published false keeps its alarm red, and an
    absent gate keeps `GATE NOT MEASURED` in the absence grey — on a fresh row as well as a stale
    one, so the freshness gate cannot become a way to dim a real reading."""
    veto = {**_G5B_LIVE_PAYLOAD, "is_tradable": False,
            "tradability_reason": "Expected duration E[D]=2.0b < 3.5b threshold (fee churn risk)"}
    for stamp in (_G5B_LIVE_UPDATED_AT, _G5B_FRESH_UPDATED_AT):
        html, _fake, ns = _g5b_render_deck(tmp_path, payload=veto, updated_at=stamp)
        assert ns["m_tradable_color"] == "#ff003c", (stamp, ns["m_tradable_color"])
        assert "UNTRADEABLE CHURN" in html, (stamp, html[:2000])
        assert "STALE ROW)" not in html, (stamp, html[:2000])
        assert _G5B_ALL_CLEAR_COLOUR not in html, html[:2000]
    for stamp in (_G5B_LIVE_UPDATED_AT, _G5B_FRESH_UPDATED_AT):
        html, _fake, ns = _g5b_render_deck(
            tmp_path, payload=_g5b_payload_without("is_tradable"), updated_at=stamp)
        assert ns["m_tradable_color"] == _G5B_ABSENT_COLOUR, (stamp, ns["m_tradable_color"])
        assert "GATE NOT MEASURED" in html, (stamp, html[:2000])
        assert _G5B_ALL_CLEAR_COLOUR not in html, html[:2000]
        assert "(no is_tradable in markov regime telemetry)" in html, (stamp, html[:2000])


def test_g5b_a_stamp_from_the_future_is_refused_not_clamped(tmp_path):
    """Finding #7, with the controller's ruling: a stamp ahead of the clock gets a refusal, NOT the
    fleet's `max(0.0, ...)` clamp.

    `inflight_shepherd.py:611` folds a negative age into zero before comparing, which is safe for a
    veto but would let this panel print "0s old" — a fabricated freshness claim, the exact defect
    class this branch exists to end. The deck is therefore DELIBERATELY STRICTER than the fleet on
    future stamps (see the report's close-out list): the age is not measured, the card wears the
    absence grey with a reason naming the cause, the heading loses "Live", and no negative age is
    ever printed as if it were an age.
    """
    ahead = (_G5B_FIX_NOW + timedelta(seconds=45)).isoformat()
    html, _fake, ns = _g5b_render_deck(tmp_path, updated_at=ahead)
    m_r = ns["m_r"]
    assert m_r["published_at"]["measured"] is True, (
        "the stamp itself IS measured — it is the age it should have yielded that is not")
    assert m_r["stamp_is_future"] is True, dict(m_r)
    assert m_r["age_sec"] is None and m_r["age_known"] is False, m_r["age_sec"]
    assert m_r["stale"] is False, "a stamp ahead of the clock is not a stale row either"
    assert m_r["heading_is_live"] is False, m_r["heading_is_live"]
    assert "Live" not in _g5b_heading(html), _g5b_heading(html)
    assert _G5B_ALL_CLEAR_COLOUR not in html, html[:2500]
    assert ns["m_time_color"] == _G5B_ABSENT_COLOUR, ns["m_time_color"]
    assert "s old" not in ns["m_time_text"], ns["m_time_text"]
    assert "future" in ns["m_time_text"], ns["m_time_text"]
    assert re.search(r"-\d+(\.\d+)?s\b", ns["m_time_text"]) is None, ns["m_time_text"]

    # The other half, or the fix is just "never print an age".
    fresh_html, _fresh_fake, fresh_ns = _g5b_render_deck(
        tmp_path, updated_at=_G5B_FRESH_UPDATED_AT)
    assert fresh_ns["m_r"]["stamp_is_future"] is False, fresh_ns["m_r"]["stamp_is_future"]
    assert fresh_ns["m_r"]["age_sec"] == pytest.approx(60.0), fresh_ns["m_r"]["age_sec"]
    assert "Live Forward Filter" in _g5b_heading(fresh_html), _g5b_heading(fresh_html)
    assert fresh_ns["m_time_color"] != _G5B_ABSENT_COLOUR, fresh_ns["m_time_text"]
    assert "s old" in fresh_ns["m_time_text"], fresh_ns["m_time_text"]

    # And the two unmeasurable-stamp branches stay distinguishable: a stamp that is not an instant
    # at all must not borrow the future stamp's reason.
    junk_html, _junk_fake, junk_ns = _g5b_render_deck(tmp_path, updated_at="not a timestamp")
    assert junk_ns["m_r"]["stamp_is_future"] is False, junk_ns["m_r"]["stamp_is_future"]
    assert junk_ns["m_r"]["age_known"] is False
    assert "future" not in junk_ns["m_time_text"], junk_ns["m_time_text"]
    assert junk_ns["m_time_color"] == _G5B_ABSENT_COLOUR, junk_ns["m_time_text"]


def test_g5b_an_absent_tradability_reason_supplies_no_excuse(tmp_path):
    """"Tradable" was the panel writing the engine's alibi for it."""
    html = _g5b_render_deck(tmp_path, payload=_g5b_payload_without("tradability_reason"))[0]
    assert "reason: not measured" in html, html[:2000]
    assert "(no tradability_reason in markov regime telemetry)" in html, html[:2000]
    assert ">Tradable<" not in html and "Tradable: " not in html, (
        "the panel still puts a tradability excuse in the engine's mouth")


def test_g5b_an_absent_breakdown_risk_is_not_a_green_stable_all_clear(tmp_path):
    """`:4778` defaulted the cascade risk to 0.0 and `:4863-4864` then painted that zero `#00ff66`
    and labelled it STABLE — an all-clear bought by the absence of a measurement."""
    html = _g5b_render_deck(tmp_path, payload=_g5b_payload_without("breakdown_risk_4step"))[0]
    assert "STABLE" not in html, html[:2000]
    assert "0.0%" not in html, "an absent cascade risk printed as a measured zero"
    assert _G5B_ABSENT_COLOUR in html
    assert "(no breakdown_risk_4step in markov regime telemetry)" in html, html[:2000]


def test_g5b_a_measured_cascade_risk_keeps_its_own_rung_and_thresholds(tmp_path):
    html = _g5b_render_deck(tmp_path)[0]
    assert "66.7%" in html, html[:2000]
    assert "HIGH" in html, "a 0.6672 cascade risk prints without its label"
    assert "#ff003c" in html, "the measured cascade risk lost its alarm colour"
    assert "panel rungs" in html, (
        "the deck judges the risk against thresholds it does not name")


# One row per KPI card: the payload key whose reading owns the card, the colour variable the deck
# paints that card with, and the colour that card earns when the live row IS measured.
_G5B_CARD_COLOUR_CASES = (
    ("dominant_regime", "m_dom_color", "#f59e0b"),
    ("expected_durations_bars", "m_domdur_color", "#00e5ff"),
    ("breakdown_risk_4step", "m_bd_color", "#ff003c"),
    ("sizing_multiplier", "m_sizing_color", "#38bdf8"),
    ("stationary_distribution", "m_stat_color", "#a855f7"),
    ("is_tradable", "m_tradable_color", "#00ff66"),
)


@pytest.mark.parametrize("dropped_key,colour_var,live_colour", _G5B_CARD_COLOUR_CASES)
def test_g5b_a_card_that_lost_its_reading_dims_while_its_neighbours_keep_theirs(
        tmp_path, dropped_key, colour_var, live_colour):
    """One reading per card, one colour per reading — proved in both directions per card.

    Rewritten from the inherited `the_live_row_prints_no_measured_colour_on_an_absent_field`, which
    dropped four keys and then demanded that NO card anywhere wear a measured colour. Two of the
    cards it scanned (the tradability gate's green and the cascade risk's red) keep their readings
    in that payload, so the test could only have passed by dimming measured readings — and
    `test_g5b_a_measured_cascade_risk_keeps_its_own_rung_and_thresholds` demands exactly the opposite
    of that card. Per card, both halves: absent -> grey, measured -> its own colour.
    """
    _html, _fake, absent_ns = _g5b_render_deck(tmp_path, payload=_g5b_payload_without(dropped_key))
    assert absent_ns[colour_var] == _G5B_ABSENT_COLOUR, (
        f"the card fed by {dropped_key!r} still wears {absent_ns[colour_var]!r} without its reading")
    # Finding #2 moved this half onto a fresh stamp: the tradability gate is the deck's combined
    # verdict and now also keys on the row's freshness, so the stale live stamp (`_G5B_LIVE_UPDATED_AT`,
    # ~35 h old) legitimately dims it even with a measured `true`. A row the fleet still trades is the
    # honest reference for "a measured reading keeps its own colour" — and `#00ff66` on the stale row
    # is exactly what `test_g5b_a_stale_row_takes_the_green_off_the_tradability_gate` now forbids.
    _html2, _fake2, live_ns = _g5b_render_deck(tmp_path, updated_at=_G5B_FRESH_UPDATED_AT)
    assert live_ns[colour_var] == live_colour, (
        f"the live row's {dropped_key!r} card lost its measured colour: {live_ns[colour_var]!r}")


def test_g5b_an_unpublished_row_wears_no_measured_colour_at_all(tmp_path):
    """The whole-deck half of the same rule: with nothing published there is no reading to colour,
    so no card and no verdict may wear green, red, cyan, blue or purple."""
    html = _g5b_render_deck(tmp_path, raw="", updated_at="")[0]
    for colour in (_G5B_ALL_CLEAR_COLOUR, "#ff003c", "#00e5ff", "#a855f7", "#38bdf8", "#f59e0b"):
        assert colour not in html, (colour, html[:2000])
    assert _G5B_ABSENT_COLOUR in html
    assert html.count("not measured") >= 6, html[:2000]


def test_g5b_the_transition_matrix_is_plotted_only_when_the_engine_published_one(tmp_path):
    """B-4's rendering half: `raw_P = m_tel.get("transition_matrix", default_P.tolist())` put a
    hardcoded prior into a heatmap titled as the matrix the engine estimated from the bars."""
    _html, absent_fake, _ns = _g5b_render_deck(tmp_path, raw="", updated_at="")
    assert not [one for one in absent_fake.charts
                if any(str(trace.type) == "heatmap" for trace in one.data)], (
        "an absent row still renders a transition-matrix heatmap")
    _html2, live_fake, _ns2 = _g5b_render_deck(tmp_path)
    heatmaps = [one for one in live_fake.charts
                if any(str(trace.type) == "heatmap" for trace in one.data)]
    assert len(heatmaps) == 1, len(live_fake.charts)
    z = [list(row) for row in heatmaps[0].data[0].z]
    assert z == _G5B_LIVE_PAYLOAD["transition_matrix"], "the plotted matrix is not the published one"
    assert [0.70, 0.15, 0.15] not in z, "the engine's prior reached the chart"


def test_g5b_a_bar_series_is_skipped_rather_than_plotted_at_zero(tmp_path):
    """A horizon with an unmeasured probability cannot be drawn: a bar at 0.0 would publish the
    absence as a measured zero, so the series is left out and the skip is named."""
    _html, fake, _ns = _g5b_render_deck(
        tmp_path, payload=_g5b_payload_without("stationary_distribution"))
    bars = [one for one in fake.charts if any(str(trace.type) == "bar" for trace in one.data)]
    assert len(bars) == 1, len(fake.charts)
    names = [trace.name for trace in bars[0].data]
    assert "Equilibrium π*" not in names, names
    assert "Current π_t" in names and "Forecast π_{t+4}" in names, names
    _html2, absent_fake, _ns2 = _g5b_render_deck(tmp_path, raw="", updated_at="")
    assert not [one for one in absent_fake.charts
                if any(str(trace.type) == "bar" for trace in one.data)]


def test_g5b_the_deck_publishes_the_live_readings_it_claims(tmp_path):
    """The integration claim: the shipped block, run against the live row seeded in a real
    `system_state` table, prints the row's own figures — and says the row is stale."""
    raw = _g5_system_state_row(tmp_path, payload=_G5B_LIVE_PAYLOAD, key=_MARKOV_KEY,
                               updated_at=_G5B_LIVE_UPDATED_AT)
    import json as _json
    assert _json.loads(raw) == _G5B_LIVE_PAYLOAD, raw
    html, fake, _ns = _g5b_render_deck(tmp_path, raw=raw)
    assert "None" not in html, html[:2000]
    assert "CHOP" in html and "88.8%" in html, html[:2000]
    assert "10.4 Bars" in html, html[:2000]
    assert "66.7%" in html and "0.64x" in html, html[:2000]
    assert "37%/54%/9%" in html, html[:2000]
    assert "126000s old" in html, html[:1500]
    assert fake.charts, "the deck published no chart for a fully measured row"


def test_g5b_the_coverage_line_counts_the_readers_own_fields():
    """Derived from the reader's own roster, never from a tuple kept by hand in the block."""
    block = _g5b_block_code()
    assert 'm_r["coverage"]' in block, "the coverage headline is not read from the reader"
    empty = _g5b_readings("{}")
    assert empty["field_names"] == tuple(sorted(_G5B_FIELDS)), empty["field_names"]
    assert empty["coverage"] == {"total": len(_G5B_FIELDS),
                                 "unmeasured": len(_G5B_FIELDS)}, empty["coverage"]
    html = _g5b_render_deck(payload="{}")[0]
    assert f"{len(_G5B_FIELDS)} of {len(_G5B_FIELDS)} published fields" in html, html[:2000]
    live_html = _g5b_render_deck(payload=_G5B_LIVE_PAYLOAD)[0]
    assert "published fields" not in live_html, "the live row is being reported as incomplete"


# ── structural guards: per-field gating, derived verdicts, one anchor per panel ──────────
@pytest.mark.parametrize("field,own_key", sorted(_G5B_FIELD_OWNERS.items()))
def test_g5b_the_deck_keys_each_field_on_its_own_state(field, own_key):
    """Same rule G-5a landed for the carry panel: a published value may read only its own flag."""
    import ast

    block = _g5b_block_code()
    tree = ast.parse(block, filename="dashboard.py markov deck")
    parents = _g4_parent_map(tree)
    assigns = [node for node in ast.walk(tree)
               if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == field for t in node.targets)]
    assert assigns, f"the deck no longer publishes {field} at all"
    keyed = re.compile(own_key + r"\[[^\]]*measured[^\]]*\]")
    for node in assigns:
        segment = ast.get_source_segment(block, node) or ""
        named = set(re.findall(r"\bm_[a-z_]*_st\b", segment))
        assert named == {own_key}, (
            f"{field} is keyed on {sorted(named)}; a published field value may read only its own "
            f"state ({own_key}): {segment[:200]}")
        assert keyed.search(segment), (
            f"the deck never keys {field} on {own_key}'s own measured flag: {segment[:200]}")
        if field.endswith("_note"):
            assert "_absence_note(" in segment and "reason" in segment, (
                f"{field} publishes no reason for its own absence: {segment[:200]}")
        else:
            assert "not measured" in segment, (
                f"{field} has no absence for when its own state was never measured: "
                f"{segment[:200]}")
        assert not _g5_enclosing_gate_is_bundled(node, parents, block, own_key, prefix="m"), (
            f"{field} is published under a gate on another field's state")


def test_g5b_every_derived_verdict_in_the_deck_guards_on_measured():
    """An unguarded verdict either raises (and the panel publishes nothing) or paints an absence in
    a measured colour. The exemptions are by shape, and the shape is asserted here."""
    import ast

    block = _g5b_block_code()
    tree = ast.parse(block, filename="dashboard.py markov deck")
    parents = _g4_parent_map(tree)
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if not (isinstance(target, ast.Name) and target.id in _G5B_DERIVED_NAMES):
                continue
            seen.add(target.id)
            segment = ast.get_source_segment(block, node) or ""
            if _g5_reads_measured_flag(segment) or _g5_enclosing_measured_flag_guard(
                    node, parents, block):
                continue
            assert target.id in _G5B_CHAIN_VERDICTS, (
                f"the deck derives {target.id} from a reading that can be absent: "
                f"{segment[:160]}")
            names_state = sorted(set(re.findall(r"\bm_[a-z_]*_st\b", segment)))
            assert not names_state, (
                f"{target.id} claims the chain exemption but names field state {names_state}, so "
                "it is a single-field verdict and has to be gated like every other one")
    missing = sorted(_G5B_DERIVED_NAMES - seen)
    assert not missing, (
        f"the deck no longer derives {missing}, so this guard has nothing to check")


def _g5b_deck_assignments():
    """Every `m_`-prefixed name the deck block assigns, mapped to the source of each assignment.

    Finding #9 needed the OTHER direction of the roster check: the guards below used to walk the
    roster and confirm each name still appears in the deck, which says nothing about a name the deck
    derives and the roster never heard of. This walks the deck.
    """
    import ast

    block = _g5b_block_code()
    tree = ast.parse(block, filename="dashboard.py markov deck")
    segments = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id.startswith("m_"):
                segments.setdefault(target.id, []).append(
                    ast.get_source_segment(block, node) or "")
    return segments


def test_g5b_every_name_the_deck_derives_is_rostered_or_explained():
    """Finding #9: a verdict added to the deck in the wrong place used to be invisible to every guard.

    Both directions are pinned. An `m_*` name the deck assigns and no roster knows is a derived
    verdict nobody guards, and a rostered name the deck no longer assigns is a claim about the deck
    that has stopped being true — either way the structural pin has to say so rather than pass.
    """
    assigned = set(_g5b_deck_assignments())
    accounted = (set(_G5B_FIELD_OWNERS) | set(_G5B_DERIVED_NAMES) | set(_G5B_READER_ALIASES)
                 | set(_G5B_UNROSTERED_SHAPES))
    unexplained = sorted(assigned - accounted)
    assert not unexplained, (
        f"the deck derives {unexplained} and no roster names them, so the per-field gate and the "
        "guarded-verdict sweep never look at them: add each to _G5B_FIELD_OWNERS (a single field's "
        "published text), _G5B_DERIVED_NAMES (a verdict or colour that must guard on measured), "
        "_G5B_READER_ALIASES (a bare lookup into the reader's dict) or _G5B_UNROSTERED_SHAPES "
        "(with the shape that justifies leaving it out)")
    gone = sorted(accounted - assigned)
    assert not gone, (
        f"the rosters still name {gone}, which the deck no longer assigns; a guard over a name that "
        "left the panel is a test that cannot fail")
    overlap = ((set(_G5B_UNROSTERED_SHAPES)
                & (set(_G5B_FIELD_OWNERS) | set(_G5B_DERIVED_NAMES)
                   | set(_G5B_READER_ALIASES)))
               | (set(_G5B_READER_ALIASES)
                  & (set(_G5B_FIELD_OWNERS) | set(_G5B_DERIVED_NAMES))))
    assert not overlap, (
        f"{sorted(overlap)} is both guarded and excused, which is neither")


@pytest.mark.parametrize("name", sorted(_G5B_UNROSTERED_SHAPES))
def test_g5b_each_unrostered_deck_name_is_excluded_for_the_stated_shape(name):
    """The exclusion list is only honest while each entry still has the shape it was excused for.

    Every reason in `_G5B_UNROSTERED_SHAPES` is pinned to text that has to be in the deck's own
    assignment (and, for the two literal maps, to text that must NOT be), so an entry cannot survive
    as a comment that stopped describing the code — and a name that starts reading a measurement
    cannot stay excused.
    """
    why, must, must_not = _G5B_UNROSTERED_SHAPES[name]
    assert name not in _G5B_FIELD_OWNERS and name not in _G5B_DERIVED_NAMES, (
        f"{name} is rostered as well as excused")
    segments = _g5b_deck_assignments().get(name) or []
    assert segments, (
        f"the deck no longer assigns {name}, so the roster comment excusing it as {why!r} is stale")
    joined = "\n".join(segments)
    for token in must:
        assert token in joined, (
            f"{name} is excused as {why!r}, but its assignment no longer contains {token!r}: "
            f"{joined[:220]}")
    for token in must_not:
        assert token not in joined, (
            f"{name} now reads {token!r}, so it is a verdict derived from a reading and belongs in "
            f"_G5B_DERIVED_NAMES, not in the unrostered list ({why})")


def test_g5b_the_reader_aliases_are_bare_lookups_into_the_reading_dict():
    """Finding #9: the roster that says "these names are the reader's own states" has to be true.

    An alias is allowed to skip the per-field gate precisely because it cannot invent a measurement.
    `m_bd_st = m_r["breakdown_risk"] or {"measured": True, "value": 0.0}` would keep the name and put
    the stand-in back, so each alias is pinned to one subscript of `m_r` — and `m_r` itself to the
    shipped reader call.
    """
    import re

    assignments = _g5b_deck_assignments()
    for name in sorted(_G5B_READER_ALIASES):
        segments = assignments.get(name) or []
        assert segments, f"{name} is rostered as a reader alias but the deck no longer assigns it"
        for segment in segments:
            if name == "m_r":
                shape = r"m_r = _markov_regime_readings\([\s\S]*\)"
                claim = "it is the shipped reader's own return value, nothing else"
            else:
                shape = r'%s = m_r\["[a-z_0-9]+"\]' % re.escape(name)
                claim = "it is one subscript of the reader's dict and folds no fallback"
            assert re.fullmatch(shape, segment), (
                f"{name} is rostered as a reader alias, but its code is {segment[:200]!r}; the "
                f"guard relies on {claim}")


def test_g5b_the_deck_explains_every_absence_it_shows(tmp_path):
    """`_absence_note` is the house parenthetical; a bare `not measured` tells the operator nothing
    about which reading went missing."""
    block = _g5b_block_code()
    assert block.count("_absence_note(") >= 7, (
        f"the deck explains only {block.count('_absence_note(')} absences")
    html = _g5b_render_deck(payload="{}")[0]
    for field in _G5B_NUMBER_FIELDS:
        assert _G5B_FIELD_REASONS[field] in html, (field, html[:3000])


def test_g5b_the_two_panel_anchors_still_name_one_panel_each():
    """The deck's start anchor is the section comment G-4's `"micro"` block already ends on, and
    `"markov"` ends on the next section comment. Both panels must keep extracting exactly one
    region, or every guard above is scanning an undefined window."""
    import ast

    micro = _g4_block("micro", DASHBOARD)
    markov = _g4_block("markov", DASHBOARD)
    assert "Bayesian Markov Chain" in markov.splitlines()[0], markov.splitlines()[0]
    assert "Multi-Asset Fleet Microstructure" not in markov, "the end anchor stopped cutting"
    assert "Bayesian Markov Chain" not in micro, "the micro block ran into the deck"
    assert "Markov" not in micro, (
        f"the microstructure block now covers the deck too: {micro.splitlines()[:3]}")
    ast.parse(markov, filename="dashboard.py markov deck")
    ast.parse(micro, filename="dashboard.py micro deck")


def test_g5b_the_legacy_dashboard_has_no_markov_deck_to_convert():
    """The brief's scope claim, checked rather than assumed: one file, one deck."""
    legacy_src = LEGACY_DASHBOARD.read_text(encoding="utf-8")
    assert _MARKOV_KEY not in legacy_src
    assert "Markov" not in legacy_src
    with pytest.raises(Exception):
        _g4_block("markov", LEGACY_DASHBOARD)


# ── B-7: the two carry-panel residues G-5a's re-review left, one line each ───────────────
# They are rendered claims, so they are proved on the shipped carry block, in the G-5a idiom, and
# each half is checked: the claim the panel now makes, and the claim it may not make again.

def test_g5b7_the_safe_clear_rung_is_named_on_both_threshold_branches():
    """An all-clear was earned by a 30% gate that lived only inside the comparison, while the header
    comment claimed every threshold the panel prints was labelled. It is printed now, and named as
    the panel's own, on the branch that actually shows the green."""
    html, _fake = _g5_render_carry_panel(None)
    assert "panel safe-clear rung: 30% (panel's own)" in html, html[:2500]
    measured_html, _fake2 = _g5_render_carry_panel(
        None, payload={**_G5_LIVE_PAYLOAD, "min_liq_distance_pct": 40.0})
    assert "engine min-buffer threshold: 40.0%" in measured_html, measured_html[:2500]
    assert "panel safe-clear rung: 30% (panel's own)" in measured_html, (
        "the rung is named only beside the panel's own fallback threshold, so the branch a measured "
        "book clears on is still the one that prints an unlabelled all-clear")


def test_g5b7_the_harvest_absence_is_explained_once():
    """`b_harvest_note` was interpolated in the harvest card and again in the harvest line, so one
    absent reading printed the same parenthetical twice side by side."""
    book = _g5_active_engine_payload()
    assert "cumulative_funding_harvested" in book, sorted(book)
    html, _fake = _g5_render_carry_panel(
        None, payload={k: v for k, v in book.items() if k != "cumulative_funding_harvested"})
    note = "(no cumulative_funding_harvested in basis trader telemetry)"
    assert note in html, html[:2500]
    assert html.count(note) == 1, (
        f"the harvest absence is printed {html.count(note)} times, not once")
    live_html, _fake2 = _g5_render_carry_panel(None, payload=book)
    assert f"{book['cumulative_funding_harvested']:+.4f}" in live_html, (
        "a measured harvest stopped printing its own figure")
    assert note not in live_html


# ===========================================================================
# G-5c — the fleet active-position deck (findings C-1 … C-6)
# ===========================================================================
# C-6 first: the deck renders five engine tables, every one of which is empty in the live
# database today, so no behavioural test here can run against `trading_data.db`. The fixture
# below recreates those tables in a tmp database from their REAL column lists (PRAGMA, never
# hand-typed), seeds rows into them, and rebuilds `df_open` with the five SELECTs the shipped
# deck runs. Every G-5c test then calls the shipped helpers for real.

_G5C_POSITION_TABLES = ("open_positions", "berserker_open_positions", "marl_live_open_positions",
                        "mean_reversion_open_positions", "swing_open_positions")
_G5C_TRADING_DATA_DB = REPO_ROOT / "trading_data.db"

# Every figure in this deck used to have a stand-in. Each pair is the banned text plus one line of
# code that genuinely contains it, which is what makes the second half of the prose-immunity test
# below more than a restatement of the first.
_G5C_STAND_IN_CASES = (
    ("1.0 - 0.035", "hard_sl = entry_p * (1.0 - 0.035)"),
    ("1.0 + 0.035", "hard_sl = entry_p * (1.0 + 0.035)"),
    ("(Exchange)", 'card = f\'Hard SL: ${hard_sl:,.4f} (Exchange)<br>\''),
    ("Hard SL (Bitget)", 'row = {"Hard SL (Bitget)": f"${hard_sl:,.4f}"}'),
)


def _g5c_schema_columns(table):
    """One position table's column list, read off the shipped schema with `PRAGMA table_info`.

    The fixture's columns come from the database the fleet writes, never typed into this file:
    C-1 rests on `swing_open_positions` having no `hard_sl` column while `open_positions` does,
    and a hand-written column list would only assert that the transcription agrees with itself.

    Opened `mode=ro`, and `tests/conftest.py` redirects even that to the read-only quarantine
    copy of the schema, so no G-5c test touches `trading_data.db` or its `-shm` sidecar.
    """
    import sqlite3

    uri = f"file:{_G5C_TRADING_DATA_DB.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    try:
        cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    finally:
        conn.close()
    assert cols, (f"{table} is not in the shipped trading_data.db schema, so the deck fixture "
                  "cannot be derived from it")
    return cols


def _g5c_seed_position_db(tmp_path, rows):
    """Create all five deck tables in a tmp DB with their real columns and seed `rows` into them.

    `rows` is a list of `{"table": name, **column_values}`. A column a row omits is stored as
    NULL, which is exactly how a writer that never set that field appears to the deck.
    """
    import sqlite3

    db = tmp_path / "positions_fixture.db"
    conn = sqlite3.connect(str(db))
    try:
        for table in _G5C_POSITION_TABLES:
            cols = _g5c_schema_columns(table)
            conn.execute(f'CREATE TABLE {table} ({", ".join(cols)})')
        for row in rows:
            payload = dict(row)
            table = payload.pop("table")
            cols = _g5c_schema_columns(table)
            unknown = sorted(set(payload) - set(cols))
            assert not unknown, (
                f"{table} has no column(s) {unknown}; the fixture may only seed columns the "
                "producer's own schema has")
            names = [c for c in cols if c in payload]
            conn.execute(
                f"INSERT INTO {table} ({', '.join(names)}) "
                f"VALUES ({', '.join('?' * len(names))})",
                [payload[c] for c in names])
        conn.commit()
    finally:
        conn.close()
    return db


def _g5c_deck_sql(path=DASHBOARD):
    """The five open-position SELECTs the deck runs, read out of the shipped source.

    Extracted from the assignment nodes rather than retyped, so the swing query's column list —
    the one that cannot and does not name `hard_sl` — is the file's own claim.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    sqls = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and re.fullmatch(r"df_open_[0-9]", t.id)
                   for t in node.targets):
            continue
        call = node.value
        assert isinstance(call, ast.Call) and getattr(call.func, "id", "") == "query_db", (
            "the deck's position frame is no longer built from literal query_db() calls, so this "
            "fixture no longer runs the shipped SQL")
        assert len(call.args) == 2, (
            "the deck's position query gained arguments this fixture ignores")
        sql = call.args[1]
        assert isinstance(sql, ast.Constant) and isinstance(sql.value, str), (
            "the deck's position query is no longer a literal string this fixture can run")
        sqls.append(sql.value)
    assert len(sqls) == 5, f"expected the five deck queries, found {len(sqls)}: {sqls}"
    return sqls


def _g5c_deck_frame(db_path, path=DASHBOARD):
    """`df_open` as the deck builds it: the shipped five queries, concat, keep-last de-dup."""
    import sqlite3

    conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True, timeout=5.0)
    try:
        frames = [pd.read_sql_query(sql, conn) for sql in _g5c_deck_sql(path)]
    finally:
        conn.close()
    kept = [frame for frame in frames if not frame.empty]
    if not kept:
        return pd.DataFrame()
    return pd.concat(kept, ignore_index=True).drop_duplicates(subset=["symbol"], keep="last")


def _g5c_block(part, path=DASHBOARD):
    return _g4_block(f"position_{part}", path)


def _g5c_helper_code(name, path=DASHBOARD):
    return _function_source(path, name)


def _g5c_deck_code(path=DASHBOARD):
    """Every deck block this finding touches, comments and docstrings removed."""
    parts = [_g5c_block("agg", path), _g5c_block("cards", path)]
    if path == DASHBOARD:
        parts.append(_g5c_block("chart", path))
    return "\n".join(parts)


def _g5c_hard_sl_code(path=DASHBOARD):
    """Deck blocks plus the helpers that decide what the hard-SL line prints (C-6)."""
    return "\n".join([_g5c_deck_code(path),
                      _g5c_helper_code("_position_hard_sl", path),
                      _g5c_helper_code("_position_deck_readings", path)])


# ── C-6: the testbed itself ──────────────────────────────────────────────────────────────

def test_g5c_the_fixture_columns_come_from_the_shipped_schema():
    """The producer contract C-1 rests on, pinned off PRAGMA rather than off this file.

    `swing_open_positions` — the table the live fleet writes (super_bot.py:1282,
    swing_trader.py:2116) — has no `hard_sl` column to read. `open_positions`, which the
    reconciliation recovery path writes (reconciliation_engine.py:830), does.
    """
    swing = _g5c_schema_columns("swing_open_positions")
    assert "hard_sl" not in swing, f"the swing table grew a hard_sl: {swing}"
    assert "leverage" in swing and "size" in swing, swing
    for table in ("open_positions", "berserker_open_positions", "marl_live_open_positions",
                  "mean_reversion_open_positions"):
        cols = _g5c_schema_columns(table)
        assert "hard_sl" in cols and "leverage" in cols, f"{table} lost a column: {cols}"


def test_g5c_the_deck_cannot_and_does_not_ask_a_swing_row_for_a_hard_sl():
    """The swing SELECT is the deck's own text; it names no `hard_sl`, so those rows cannot carry
    one even in principle."""
    swing_sql = [s for s in _g5c_deck_sql() if "swing_open_positions" in s]
    assert len(swing_sql) == 1, swing_sql
    assert "hard_sl" not in swing_sql[0], swing_sql[0]


def test_g5c_a_swing_row_reaches_the_deck_with_no_hard_sl_to_read(tmp_path):
    """The fabrication C-1 removed fired for EVERY position the live fleet can open, not for an
    edge case: `pd.concat` leaves a swing row's hard_sl as NaN and `num()` turns NaN into 0.0."""
    db = _g5c_seed_position_db(tmp_path, [{
        "table": "swing_open_positions", "symbol": "BTCUSDT", "side": "long",
        "entry_price": 100000.0, "size": 0.01, "leverage": 5, "stop_loss": 97000.0,
        "take_profit": 110000.0, "opened_at": "2026-10-22T10:00:00+00:00", "status": "OPEN"}])
    frame = _g5c_deck_frame(db)
    assert len(frame) == 1, "the seeded swing row never reached the deck's frame"
    row = next(r for _, r in frame.iterrows())
    read = _g4_helper("_position_hard_sl", DASHBOARD)(row)
    assert read == {"value": None, "measured": False,
                    "reason": "no hard SL recorded for this position"}, read
    assert _g4_helper("_deck_text", DASHBOARD)(read, "${:,.4f}") == "not measured"


@pytest.mark.parametrize("stored", [None, 0.0, -12.5, "syncing...", float("nan")],
                         ids=["null", "zero", "negative", "placeholder", "nan"])
def test_g5c_no_hard_sl_shape_renders_a_price(tmp_path, stored):
    """A table that HAS the column can still hold no stop: NULL, the column's 0.0 default, a
    negative, placeholder text or NaN. None of them may print as a price under any label."""
    db = _g5c_seed_position_db(tmp_path, [{
        "table": "open_positions", "symbol": "ETHUSDT", "side": "long",
        "entry_price": 4000.0, "qty": 1.0, "leverage": 5, "hard_sl": stored}])
    row = next(r for _, r in _g5c_deck_frame(db).iterrows())
    read = _g4_helper("_position_hard_sl", DASHBOARD)(row)
    assert read["measured"] is False, f"{stored!r} was published as a reading: {read}"
    assert read["value"] is None
    assert _g4_helper("_deck_text", DASHBOARD)(read, "${:,.4f}") == "not measured"


def test_g5c_a_recorded_hard_sl_stays_a_reading_unchanged(tmp_path):
    """The other half: a row that DOES carry a stop publishes exactly the stored number, with no
    absence note beside it."""
    db = _g5c_seed_position_db(tmp_path, [{
        "table": "open_positions", "symbol": "ETHUSDT", "side": "long",
        "entry_price": 4000.0, "qty": 1.0, "leverage": 5, "hard_sl": 3777.5}])
    row = next(r for _, r in _g5c_deck_frame(db).iterrows())
    read = _g4_helper("_position_hard_sl", DASHBOARD)(row)
    assert read == {"value": 3777.5, "measured": True, "reason": ""}, read
    assert _g4_helper("_deck_text", DASHBOARD)(read, "${:,.4f}") == "$3,777.5000"
    assert _g4_helper("_deck_note", DASHBOARD)(read) == ""


def test_g5c_a_mixed_book_keeps_the_recorded_stop_and_publishes_the_missing_one(tmp_path):
    """One swing row (no such column) plus one reconciled row (recorded stop) in the frame the
    deck builds: the recorded stop survives, the missing one becomes an absence, and neither
    borrows the other's value or gets an invented level."""
    db = _g5c_seed_position_db(tmp_path, [
        {"table": "swing_open_positions", "symbol": "BTCUSDT", "side": "long",
         "entry_price": 100000.0, "size": 0.01, "leverage": 5, "stop_loss": 97000.0},
        {"table": "mean_reversion_open_positions", "symbol": "ETHUSDT", "side": "short",
         "entry_price": 4000.0, "qty": 2.0, "leverage": 3, "hard_sl": 4120.0}])
    frame = _g5c_deck_frame(db)
    assert len(frame) == 2 and "hard_sl" in frame.columns, list(frame.columns)
    reads = {str(r["symbol"]): _g4_helper("_position_hard_sl", DASHBOARD)(r)
             for _, r in frame.iterrows()}
    assert reads["BTCUSDT"]["measured"] is False, reads["BTCUSDT"]
    assert reads["ETHUSDT"] == {"value": 4120.0, "measured": True, "reason": ""}, reads["ETHUSDT"]


def test_g5c_the_deck_builds_the_frame_the_fixture_builds():
    """The fixture's concat is the deck's own shape; if the deck's de-dup rule moves, this fails
    instead of quietly testing a different frame."""
    src = DASHBOARD.read_text(encoding="utf-8")
    assert "df_open = pd.concat(open_dfs, ignore_index=True)" in src
    assert "df_open = df_open.drop_duplicates(subset=['symbol'], keep='last')" in src
    assert "open_dfs = [d for d in [df_open_1, df_open_2, df_open_3, df_open_4, df_open_5]" in src


# ── C-1: a panel-invented stop published under the exchange's name ─────────────────────────

@pytest.mark.parametrize("banned,code_line", _G5C_STAND_IN_CASES)
def test_g5c_no_invented_stop_survives_in_the_deck(banned, code_line):
    """The ±3.5% stand-in and both authority labels are gone from every deck block, and from the
    helper that replaced the three copies — in dashboard.py and in its twin.

    Checked in two halves so a failure says which one moved: the panel that renders, then the
    helper that decides what it renders.
    """
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        hits = _g5_banned_hits(_g5c_deck_code(path), [banned])
        assert hits == [], f"{path.name}'s deck blocks still contain {banned!r}"
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        hits = _g5_banned_hits(_g5c_hard_sl_code(path), [banned])
        assert hits == [], f"{path.name}'s hard-SL helper still contains {banned!r}"


@pytest.mark.parametrize("banned,code_line", _G5C_STAND_IN_CASES)
def test_g5c_a_comment_neither_trips_nor_hides_a_position_deck_scan(banned, code_line):
    """Both halves of prose immunity on this deck's own tokens: prose about the deleted stand-in
    passes, and the same text really in the code fails."""
    prose = f"def f():\n    # the deck used to publish {banned} here\n    return 1\n"
    assert _g5_banned_hits(prose, [banned]) == [], (
        f"a comment quoting {banned!r} failed a negative scan against code without it")
    documented = f'def f():\n    """the deck used to publish {banned} here. """\n    return 1\n'
    assert _g5_banned_hits(documented, [banned]) == [], (
        f"a docstring quoting {banned!r} failed a negative scan against code without it")
    real = f"def f():\n    {code_line}\n    return 1\n"
    assert _g5_banned_hits(real, [banned]) == [banned], (
        f"{banned!r} is in the code and the stripped scan no longer sees it")
    hidden = f"def f():\n    {code_line}  # {banned}\n    return 1\n"
    assert _g5_banned_hits(hidden, [banned]) == [banned], (
        f"a comment beside {banned!r} in the code hid it from the scan")


def test_g5c_the_hard_sl_the_card_prints_is_the_reading_and_nothing_else():
    """Each render site reads the helper's output, so what the operator sees is the reading and
    the reason that came with it — never a number the loop built."""
    cards = _g5c_block("cards", DASHBOARD)
    assert "hard_sl = p_read[\"hard_sl\"]" in cards
    assert '_deck_text(hard_sl, "${:,.4f}")' in cards
    assert "_deck_note(hard_sl)" in cards
    legacy_cards = _g5c_block("cards", LEGACY_DASHBOARD)
    assert "hard_sl = p_read[\"hard_sl\"]" in legacy_cards
    assert '_deck_text(hard_sl, "${:,.4f}")' in legacy_cards


def test_g5c_the_chart_draws_no_hard_sl_it_cannot_measure():
    """An absent level reaches the chart as 0.0, which is the chart's own 'draw nothing' encoding
    (the same honesty its `est_liq: float = 0.0` default already had); a measured level is drawn
    unchanged."""
    measured = _g4_helper("_deck_reading", DASHBOARD)(3777.5)
    absent = _g4_helper("_deck_absence", DASHBOARD)("no hard SL recorded for this position")
    level = _g4_helper("_deck_chart_level", DASHBOARD)
    assert level(measured) == 3777.5
    assert level(absent) == 0.0
    block = _g5c_block("chart", DASHBOARD)
    assert block.count("hard_sl=_deck_chart_level(p[\"hard_sl\"])") == 2, (
        "both chart-tab call sites must hand the chart the reading, not the raw entry")


def test_g5c_the_twin_helpers_are_byte_identical_and_read_the_same_row():
    """scripts/live_dashboard.py cannot import dashboard.py, so the convention is a byte-identical
    twin. Pinned as text AND as behaviour on one row."""
    for name in ("_deck_absence", "_deck_reading", "_deck_is_absent", "_deck_text", "_deck_note",
                 "_deck_chart_level", "_position_hard_sl", "_position_deck_readings"):
        assert _g5c_helper_code(name, DASHBOARD) == _g5c_helper_code(name, LEGACY_DASHBOARD), name
    row = pd.Series({"symbol": "BTCUSDT", "side": "long", "entry_price": 100000.0})
    dashboard_read = _g4_helper("_position_hard_sl", DASHBOARD)(row)
    legacy_read = _g4_helper("_position_hard_sl", LEGACY_DASHBOARD)(row)
    assert dashboard_read == legacy_read == {
        "value": None, "measured": False, "reason": "no hard SL recorded for this position"}


# ── C-2: an assumed 10x/5x leverage became margin, liquidation, "% safety", ROI and the gauge ──

# Banned text plus one line that genuinely contains it, so the prose-immunity half below is a
# second proof rather than a restatement of the first.
_G5C_LEVERAGE_STAND_IN_CASES = (
    ('10.0 if "BTC"', 'lev = num(row.get("leverage"), 10.0 if "BTC" in s else 5.0)'),
    ('or (10.0 if "BTC"', 'lev = float(row.get("leverage") or (10.0 if "BTC" in s else 5.0))'),
    ('if lev > 0 else', 'margin = (qty * entry_p) / lev if lev > 0 else (qty * entry_p)'),
    ('margin > 0 else 0.0', 'unr_roi = (unr_pnl / margin * 100.0) if margin > 0 else 0.0'),
)

_G5C_C2_HELPERS = ("_position_leverage", "_position_margin", "_position_liquidation",
                   "_position_roi", "_position_liq_buffer", "_position_deck_readings",
                   "_fleet_position_rollup", "_deck_label", "_deck_value_or",
                   "_deck_account_balance", "_deck_margin_gauge", "_deck_margin_rung")

# ── C-3: an absent live price rendered flat-and-green with the entry price under "Mark" ────────

_G5C_MARK_STAND_IN_CASES = (
    ('live_prices.get(s, ep)', 'curr_p = num(live_prices.get(s, ep), ep)'),
    ('or entry_p)', 'curr_p = num(live_prices.get(sym_clean, entry_p) or entry_p)'),
    ('or s_entry)', 's_curr = num(live_prices.get(s_clean, s_entry) or s_entry)'),
    ('float(live_prices.get(', 'curr_p = float(live_prices.get(sym_clean, entry_p) or entry_p)'),
    ('"#00ff66" if unr_pnl >= 0', 'pnl_color = "#00ff66" if unr_pnl >= 0 else "#ff003c"'),
    ('"#00ff66" if p["unr_pnl"] >= 0', 'pnl_color = "#00ff66" if p["unr_pnl"] >= 0 else "#ff003c"'),
    ("'#00ff66' if total_unrealized_pnl>=0", "colors = {'#00ff66' if total_unrealized_pnl>=0 else '#ff003c'}"),
    ('"#00ff66" if true_portfolio_net >= 0',
     'true_port_color = "#00ff66" if true_portfolio_net >= 0 else "#ff003c"'),
)

_G5C_C3_HELPERS = ("_position_mark", "_position_unrealized_pnl", "_deck_pnl_color",
                   "_deck_signed_money", "_position_level_distance", "_position_target_label",
                   "_deck_sum", "_position_roi", "_position_liq_buffer",
                   "_position_deck_readings", "_fleet_position_rollup", "_deck_color",
                   "_deck_text", "_deck_note")


def _g5c_leverage_code(path=DASHBOARD):
    """Deck blocks plus every helper that decides what a leverage-shaped figure prints (C-2)."""
    return "\n".join([_g5c_deck_code(path)]
                     + [_g5c_helper_code(name, path) for name in _G5C_C2_HELPERS])


def _g5c_mark_code(path=DASHBOARD):
    """The C-3 surface: the three loops, the hero KPI, the funding banner and the mark helpers."""
    parts = [_g5c_deck_code(path), _g5c_block("hero", path)]
    if path == DASHBOARD:
        parts.append(_g5c_block("strip", path))
    parts += [_g5c_helper_code(name, path) for name in _G5C_C3_HELPERS]
    return "\n".join(parts)


def _g5c_seeded_row(tmp_path, table="open_positions", **columns):
    """One row off the C-6 testbed: the shipped schema, the shipped five SELECTs, the shipped concat."""
    payload = {"table": table, "symbol": "BTCUSDT", "side": "long",
               "entry_price": 100000.0, "hard_sl": None}
    if table == "swing_open_positions":
        payload["size"] = columns.pop("qty", 0.01)
        payload.pop("hard_sl")
    else:
        payload["qty"] = 0.01
    payload.update(columns)
    db = _g5c_seed_position_db(tmp_path, [payload])
    frame = _g5c_deck_frame(db)
    assert len(frame) == 1, list(frame.columns)
    return next(r for _, r in frame.iterrows())


@pytest.mark.parametrize("stored", [None, 0.0, -3.0, "syncing...", float("nan")],
                         ids=["null", "zero", "negative", "placeholder", "nan"])
def test_g5c_no_absent_leverage_shape_becomes_a_number(tmp_path, stored):
    """`leverage` IS a measured column for the tables the fleet writes; what the deck added was a
    default for the NULL/missing shape. Every unreadable shape is now an absence, including the
    measured 0.0 that the `or` used to overwrite with the assumed 10x."""
    row = _g5c_seeded_row(tmp_path, leverage=stored)
    read = _g4_helper("_position_leverage", DASHBOARD)(row)
    assert read["measured"] is False, f"{stored!r} was published as a leverage: {read}"
    assert read["value"] is None
    label = _g4_helper("_deck_label", DASHBOARD)(read, "leverage", "{:.0f}x")
    assert label == "leverage not measured", label
    assert "10x" not in label and "5x" not in label, label


def test_g5c_a_recorded_leverage_stays_a_reading_unchanged(tmp_path):
    """The other half of C-2: a row that carries its leverage publishes exactly it, unsigned by any
    assumption, with no absence note beside it."""
    row = _g5c_seeded_row(tmp_path, leverage=3)
    read = _g4_helper("_position_leverage", DASHBOARD)(row)
    assert read == {"value": 3.0, "measured": True, "reason": ""}, read
    assert _g4_helper("_deck_label", DASHBOARD)(read, "leverage", "{:.0f}x") == "3x"
    assert _g4_helper("_deck_note", DASHBOARD)(read) == ""


def test_g5c_a_row_with_no_leverage_column_at_all_is_an_absence():
    """A table the deck reads but no writer fills can reach the frame without the key at all
    (concat of differing column sets). That is the same absence, not a 5x."""
    read = _g4_helper("_position_leverage", DASHBOARD)(
        pd.Series({"symbol": "ETHUSDT", "side": "long", "entry_price": 4000.0}))
    assert read == {"value": None, "measured": False,
                    "reason": "no leverage recorded for this position"}, read


def test_g5c_everything_that_divides_by_an_assumed_leverage_is_withheld(tmp_path):
    """Per field, not bundled: margin, Est. Liq, the safety buffer and ROI each say which input
    they could not read. The recorded prices stay readings."""
    row = _g5c_seeded_row(tmp_path, leverage=None, stop_loss=97000.0, take_profit=110000.0)
    reads = _g4_helper("_position_deck_readings", DASHBOARD)(row, {"BTCUSDT": 105000.0}, {})
    # per field, each absence naming the input it could not read; the chain back to the leverage
    # runs margin -> "no margin without a measured leverage" and est_liq -> the same.
    for key, named in (("margin", "leverage"), ("est_liq", "leverage"),
                       ("unr_roi", "locked margin"), ("dist_liq", "liquidation estimate")):
        assert reads[key]["measured"] is False, (key, reads[key])
        assert named in reads[key]["reason"], (key, reads[key]["reason"])
    assert _g4_helper("_deck_text", DASHBOARD)(reads["margin"], "${:,.2f}") == "not measured"
    assert _g4_helper("_deck_text", DASHBOARD)(reads["unr_roi"], "{:+.2f}%") == "not measured"
    for key in ("entry_price", "qty", "mark", "unr_pnl", "stop_loss", "take_profit"):
        assert reads[key]["measured"] is True, (key, reads[key])


def _g5c_module_constant(name, path=DASHBOARD):
    """One module-level constant's own value, read off the shipped file rather than retyped."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{path.name} has no module-level constant {name}")


def test_g5c_a_measured_leverage_still_computes_the_whole_risk_line(tmp_path):
    """C-2 must not over-withhold: with a recorded leverage the same row publishes the margin, the
    panel's own liquidation model and the ROI built from it."""
    row = _g5c_seeded_row(tmp_path, leverage=5)
    reads = _g4_helper("_position_deck_readings", DASHBOARD)(row, {"BTCUSDT": 105000.0}, {})
    assert reads["margin"]["value"] == pytest.approx(200.0), reads["margin"]
    shipped_liq = _g4_helper("calculate_estimated_liquidation", DASHBOARD)(
        100000.0, 5.0, "long",
        mmr=_g5c_module_constant("_PANEL_LIQ_MODEL_MMR"))
    assert reads["est_liq"]["value"] == pytest.approx(shipped_liq), reads["est_liq"]
    assert reads["unr_pnl"]["value"] == pytest.approx(50.0), reads["unr_pnl"]
    assert reads["unr_roi"]["value"] == pytest.approx(25.0), reads["unr_roi"]
    buffer_ = _g4_helper("_position_liq_buffer", DASHBOARD)(
        reads["mark"], reads["est_liq"], "long")
    assert buffer_ == reads["dist_liq"], (buffer_, reads["dist_liq"])


def test_g5c_the_gauge_never_counts_a_margin_it_did_not_measure(tmp_path):
    """`total_margin_locked` feeds the utilization gauge, so one invented margin invalidates the
    whole total — and the gauge says how many rows are missing rather than summing the rest."""
    measured = _g4_helper("_position_deck_readings", DASHBOARD)(
        _g5c_seeded_row(tmp_path, leverage=5), {"BTCUSDT": 105000.0}, {})
    invented = _g4_helper("_position_deck_readings", DASHBOARD)(
        pd.Series({"symbol": "ETHUSDT", "side": "long", "entry_price": 4000.0, "qty": 2.0}),
        {}, {})
    readings = [measured, invented]
    rollup = _g4_helper("_fleet_position_rollup", DASHBOARD)(readings)
    assert _deck_absent(rollup["margin_locked"]), rollup
    assert rollup["margin_locked"]["reason"] == "1 of 2 positions report no margin", rollup
    assert rollup["margin_unmeasured"] == 1
    gauge = _g4_helper("_deck_margin_gauge", DASHBOARD)(
        rollup["margin_locked"], _g4_helper("_deck_account_balance", DASHBOARD)(10000.0),
        _g4_helper("_deck_reading", DASHBOARD)(70.0))
    assert _deck_absent(gauge["utilization"]), gauge
    rung = _g4_helper("_deck_margin_rung", DASHBOARD)(gauge["utilization"])
    assert rung == {"color": "#94a3b8", "status": "NOT MEASURED"}, rung
    assert rung["color"] != "#00ff66", "an absent total may not read as the all-clear"


def _deck_absent(read):
    return _g4_helper("_deck_is_absent", DASHBOARD)(read)


def test_g5c_a_fully_measured_book_still_totals_its_margin(tmp_path):
    """The sum survives when nothing is invented — the total is the two rows' own margins."""
    reads = [_g4_helper("_position_deck_readings", DASHBOARD)(
        _g5c_seeded_row(tmp_path, leverage=5), {"BTCUSDT": 105000.0}, {})]
    reads.append(_g4_helper("_position_deck_readings", DASHBOARD)(
        pd.Series({"symbol": "ETHUSDT", "side": "long", "entry_price": 4000.0, "qty": 2.0,
                   "leverage": 4.0}), {"ETHUSDT": 4100.0}, {}))
    rollup = _g4_helper("_fleet_position_rollup", DASHBOARD)(reads)
    assert rollup["margin_locked"]["measured"] is True, rollup
    assert rollup["margin_locked"]["value"] == pytest.approx(200.0 + 2000.0), rollup
    assert rollup["margin_unmeasured"] == 0, rollup


@pytest.mark.parametrize("banned,code_line", _G5C_LEVERAGE_STAND_IN_CASES)
def test_g5c_no_assumed_leverage_survives_in_the_deck(banned, code_line):
    """Neither the 10x/5x default, nor the implied-1x margin it fed, nor the ROI that printed 0.0
    for an unmeasured margin — in either file, in the panels or in the helpers behind them."""
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        hits = _g5_banned_hits(_g5c_deck_code(path), [banned])
        assert hits == [], f"{path.name}'s deck blocks still contain {banned!r}"
        hits = _g5_banned_hits(_g5c_leverage_code(path), [banned])
        assert hits == [], f"{path.name}'s leverage helpers still contain {banned!r}"


@pytest.mark.parametrize("banned,code_line", _G5C_LEVERAGE_STAND_IN_CASES)
def test_g5c_a_comment_neither_trips_nor_hides_a_leverage_scan(banned, code_line):
    """Both halves of prose immunity on C-2's own tokens: prose about the deleted default passes —
    `_position_leverage`'s docstring quotes it — and the same text really in code fails."""
    prose = f"def f():\n    # the deck used to publish {banned} here\n    return 1\n"
    assert _g5_banned_hits(prose, [banned]) == []
    documented = f'def f():\n    """the deck used to publish {banned} here. """\n    return 1\n'
    assert _g5_banned_hits(documented, [banned]) == []
    real = f"def f():\n    {code_line}\n    return 1\n"
    assert _g5_banned_hits(real, [banned]) == [banned], (
        f"{banned!r} is in the code and the stripped scan no longer sees it")
    hidden = f"def f():\n    {code_line}  # {banned}\n    return 1\n"
    assert _g5_banned_hits(hidden, [banned]) == [banned]


def test_g5c_the_card_names_the_missing_leverage_and_the_panel_model_its_liq():
    """What the operator sees: the badge and the table column carry the leverage label or its
    absence, and the liquidation figures are named as the panel's own model with its assumed MMR."""
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        cards = _g5c_block("cards", path)
        assert '_deck_label(lev, "leverage", "{:.0f}x")' in cards, path.name
        assert '"Leverage": _deck_label(lev, "leverage", "{:.0f}x")' in cards, path.name
        assert "assumed MMR" in cards, path.name
        assert '"Est. Liq Price (panel model)"' in cards, path.name
        assert '"Liq Safety Buffer (panel model)"' in cards, path.name
        assert '"Est. Liq Price": f"${est_liq:,.4f}"' not in cards, path.name


def test_g5c_the_chart_gets_no_liquidation_price_the_leverage_cannot_support():
    """An absent leverage reaches the chart as 0.0 — its own 'draw nothing' encoding — at both
    call sites, so the invented level never becomes an annotation."""
    block = _g5c_block("chart", DASHBOARD)
    assert block.count('est_liq=_deck_chart_level(p["est_liq"])') == 2, block.count(
        'est_liq=_deck_chart_level(p["est_liq"])')


def test_g5c_the_assumed_mmr_the_card_names_is_the_one_the_model_used():
    """Shape (b) only holds if the number on screen is the number in the arithmetic: the model call
    passes the same module constant the label prints, and both twins agree on it."""
    mmr = _g5c_module_constant("_PANEL_LIQ_MODEL_MMR", DASHBOARD)
    assert mmr == _g5c_module_constant("_PANEL_LIQ_MODEL_MMR", LEGACY_DASHBOARD), mmr
    assert mmr > 0.0, mmr
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        assert "mmr=_PANEL_LIQ_MODEL_MMR" in _g5c_helper_code("_position_liquidation", path), path
        assert "_PANEL_LIQ_MODEL_MMR*100:.1f}% assumed MMR" in _g5c_block("cards", path), path


# ── C-3: no live price → no mark, no PnL, no ROI, no distances, never the green all-clear ──────

def _g5c_live_price_row(tmp_path):
    """A BTC row with everything recorded except the live price, which no row can record."""
    db = _g5c_seed_position_db(tmp_path, [{
        "table": "open_positions", "symbol": "BTCUSDT", "side": "long",
        "entry_price": 100000.0, "qty": 0.01, "leverage": 5, "stop_loss": 97000.0,
        "take_profit": 110000.0}])
    return next(r for _, r in _g5c_deck_frame(db).iterrows())


@pytest.mark.parametrize("feed", [
    {},                                        # both fetches failed: the key is simply absent
    {"BTCUSDT": 0.0},                          # a zero is not a price
    {"BTCUSDT": -5.0},                         # a negative is not a price
    {"BTCUSDT": None},                         # a null quote
    {"BTCUSDT": "syncing..."},                 # placeholder text
    {"BTCUSDT": float("nan")},                 # NaN
], ids=["missing-key", "zero", "negative", "null", "placeholder", "nan"])
def test_g5c_an_unreadable_live_price_is_never_the_entry_price_wearing_the_mark(tmp_path, feed):
    """Contract item 8: a missing key is real information. The deck filled it with the entry price
    and printed that under "Mark" and under a column literally headed "Live Price"."""
    row = _g5c_live_price_row(tmp_path)
    entry_text = _g4_helper("_deck_text", DASHBOARD)(
        _g4_helper("_position_deck_readings", DASHBOARD)(row, feed, {})["entry_price"],
        "${:,.4f}")
    mark = _g4_helper("_position_mark", DASHBOARD)(feed, "BTCUSDT", "BTCUSDT")
    assert mark["measured"] is False, (feed, mark)
    rendered = _g4_helper("_deck_text", DASHBOARD)(mark, "${:,.4f}")
    assert rendered == "not measured", (feed, rendered)
    assert rendered != entry_text, "the absence still rendered the row's own entry price"
    assert _g4_helper("_deck_pnl_color", DASHBOARD)(mark) == "#94a3b8", feed
    assert _g4_helper("_deck_pnl_color", DASHBOARD)(mark) != "#00ff66", feed


def test_g5c_a_measured_flat_mark_is_a_reading_and_keeps_its_colour():
    """House convention: a recorded 0.0 is a measurement. A mark that genuinely equals the entry
    price prints +$0.0000 in the all-clear green — that is the case the absence rule must NOT eat."""
    mark = _g4_helper("_position_mark", DASHBOARD)({"BTCUSDT": 100000.0}, "BTCUSDT", "BTCUSDT")
    assert mark == {"value": 100000.0, "measured": True, "reason": ""}, mark
    pnl = _g4_helper("_position_unrealized_pnl", DASHBOARD)(
        mark, _g4_helper("_deck_reading", DASHBOARD)(100000.0),
        _g4_helper("_deck_reading", DASHBOARD)(0.01), "long")
    assert pnl == {"value": 0.0, "measured": True, "reason": ""}, pnl
    assert _g4_helper("_deck_pnl_color", DASHBOARD)(pnl) == "#00ff66"
    assert _g4_helper("_deck_signed_money", DASHBOARD)(pnl) == "+$0.0000"


def test_g5c_no_live_price_means_no_pnl_no_roi_no_distances_for_that_position(tmp_path):
    """Per field: the recorded columns survive, the price-dependent figures all become absences,
    and none of them is the 0.0 the pre-fix arithmetic produced from a stand-in mark."""
    row = _g5c_live_price_row(tmp_path)
    reads = _g4_helper("_position_deck_readings", DASHBOARD)(row, {}, {})
    for key in ("mark", "unr_pnl", "unr_roi", "dist_liq", "dist_tp", "dist_sl"):
        assert reads[key]["measured"] is False, (key, reads[key])
    assert reads["mark"]["reason"] == "no live price for this symbol in the current fetch"
    assert reads["unr_pnl"]["reason"] == "no unrealized PnL without a measured live price"
    for key in ("entry_price", "qty", "leverage", "margin", "stop_loss", "take_profit"):
        assert reads[key]["measured"] is True, (key, reads[key])
    assert _g4_helper("_deck_signed_money", DASHBOARD)(reads["unr_pnl"]) == "not measured"
    assert _g4_helper("_deck_pnl_color", DASHBOARD)(reads["unr_pnl"]) == "#94a3b8"
    assert _g4_helper("_deck_pnl_color", DASHBOARD)(reads["unr_pnl"]) != "#00ff66"
    assert _g4_helper("_position_target_label", DASHBOARD)(
        reads["dist_tp"], "long", "tp") == "not measured (tp)"
    # the feed's own key spellings are honoured: clean, raw, and the colon-suffixed ticker the
    # exchange returns ("BTCUSDT:USDT" must still find the deck's cleaned "BTCUSDT")
    for symbol, clean, feed in (
            ("BTCUSDT", "BTCUSDT", {"BTCUSDT": 105000.0}),
            ("BTC/USDT", "BTCUSDT", {"BTC/USDT": 105000.0}),
            ("BTCUSDT:USDT", "BTCUSDTUSDT", {"BTCUSDT:USDT": 105000.0}),
            ("BTC/USDT:USDT", "BTCUSDTUSDT", {"BTCUSDT": 105000.0})):
        assert _g4_helper("_position_mark", DASHBOARD)(feed, symbol, clean)["value"] \
            == 105000.0, (symbol, feed)


def test_g5c_a_short_position_is_marked_from_the_other_side():
    """The sign convention the three loops used to re-type: a falling mark is a gain on a short."""
    mark = _g4_helper("_position_mark", DASHBOARD)({"ETHUSDT": 3800.0}, "ETHUSDT", "ETHUSDT")
    entry = _g4_helper("_deck_reading", DASHBOARD)(4000.0)
    qty = _g4_helper("_deck_reading", DASHBOARD)(2.0)
    pnl = _g4_helper("_position_unrealized_pnl", DASHBOARD)(mark, entry, qty, "short")
    assert pnl["value"] == pytest.approx(400.0), pnl
    assert _g4_helper("_position_unrealized_pnl", DASHBOARD)(mark, entry, qty, "long")["value"] \
        == pytest.approx(-400.0)


def test_g5c_the_target_distances_match_what_the_deck_printed_before(tmp_path):
    """A measured mark keeps the wording and the numbers the panel had: +x% to TP on a long,
    -x% to TP and "% below SL" on a short."""
    row = _g5c_live_price_row(tmp_path)
    feeds = {"BTCUSDT": 105000.0}
    reads = _g4_helper("_position_deck_readings", DASHBOARD)(row, feeds, {})
    assert _g4_helper("_position_target_label", DASHBOARD)(
        reads["dist_tp"], "long", "tp") == "+4.76% to TP", reads["dist_tp"]
    assert _g4_helper("_position_target_label", DASHBOARD)(
        reads["dist_sl"], "long", "sl") == "7.62% above SL", reads["dist_sl"]
    short = _g4_helper("_position_deck_readings", DASHBOARD)(
        pd.Series({"symbol": "ETHUSDT", "side": "short", "entry_price": 4000.0, "qty": 2.0,
                   "leverage": 4.0, "stop_loss": 4200.0, "take_profit": 3600.0}),
        {"ETHUSDT": 4000.0}, {})
    assert _g4_helper("_position_target_label", DASHBOARD)(
        short["dist_tp"], "short", "tp") == "-10.00% to TP", short["dist_tp"]
    assert _g4_helper("_position_target_label", DASHBOARD)(
        short["dist_sl"], "short", "sl") == "5.00% below SL", short["dist_sl"]


def test_g5c_the_portfolio_pnl_is_absent_when_any_position_has_no_mark():
    """The roll-up needs its OWN measured-or-absent state: a sum over partly-invented addends is
    not a measurement, and the strip coloured exactly 0.0 green."""
    fed = _g4_helper("_position_deck_readings", DASHBOARD)(
        pd.Series({"symbol": "BTCUSDT", "side": "long", "entry_price": 100000.0, "qty": 0.01,
                   "leverage": 5.0}), {"BTCUSDT": 105000.0}, {})
    blind = _g4_helper("_position_deck_readings", DASHBOARD)(
        pd.Series({"symbol": "ETHUSDT", "side": "long", "entry_price": 4000.0, "qty": 2.0,
                   "leverage": 4.0}), {}, {})
    rollup = _g4_helper("_fleet_position_rollup", DASHBOARD)([fed, blind])
    assert _deck_absent(rollup["unrealized_pnl"]), rollup
    assert rollup["unrealized_pnl"]["reason"] == "1 of 2 positions report no mark PnL", rollup
    assert rollup["unr_pnl_unmeasured"] == 1
    assert _g4_helper("_deck_pnl_color", DASHBOARD)(rollup["unrealized_pnl"]) == "#94a3b8"
    net = _g4_helper("_deck_sum", DASHBOARD)(
        rollup["unrealized_pnl"], _g4_helper("_deck_reading", DASHBOARD)(1.5))
    assert _deck_absent(net) and "mark PnL" in net["reason"], net
    both = _g4_helper("_fleet_position_rollup", DASHBOARD)([fed, _g4_helper(
        "_position_deck_readings", DASHBOARD)(
        pd.Series({"symbol": "ETHUSDT", "side": "long", "entry_price": 4000.0, "qty": 2.0,
                   "leverage": 4.0}), {"ETHUSDT": 4100.0}, {})])
    assert both["unrealized_pnl"]["value"] == pytest.approx(50.0 + 200.0), both


@pytest.mark.parametrize("banned,code_line", _G5C_MARK_STAND_IN_CASES)
def test_g5c_no_stand_in_mark_survives_in_the_deck(banned, code_line):
    """No entry-price stand-in for a quote, and no green-at-zero comparison left inline — the
    colour decision now lives in `_deck_pnl_color`, which greys absences."""
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        hits = _g5_banned_hits(_g5c_deck_code(path), [banned])
        assert hits == [], f"{path.name}'s deck blocks still contain {banned!r}"
        hits = _g5_banned_hits(_g5c_mark_code(path), [banned])
        assert hits == [], f"{path.name}'s mark helpers still contain {banned!r}"


@pytest.mark.parametrize("banned,code_line", _G5C_MARK_STAND_IN_CASES)
def test_g5c_a_comment_neither_trips_nor_hides_a_mark_scan(banned, code_line):
    """Both halves on C-3's own tokens."""
    prose = f"def f():\n    # the deck used to publish {banned} here\n    return 1\n"
    assert _g5_banned_hits(prose, [banned]) == []
    documented = f'def f():\n    """the deck used to publish {banned} here. """\n    return 1\n'
    assert _g5_banned_hits(documented, [banned]) == []
    real = f"def f():\n    {code_line}\n    return 1\n"
    assert _g5_banned_hits(real, [banned]) == [banned], (
        f"{banned!r} is in the code and the stripped scan no longer sees it")
    hidden = f"def f():\n    {code_line}  # {banned}\n    return 1\n"
    assert _g5_banned_hits(hidden, [banned]) == [banned]


def test_g5c_the_mark_the_card_and_the_table_print_is_the_feed_or_nothing():
    """The render sites consume the reading: "Mark", the "Live Price" column, the PnL line and
    the PnL colour all come from the helpers, none of them from a substituted float."""
    cards = _g5c_block("cards", DASHBOARD)
    assert 'mark = p_read["mark"]' in cards
    assert '_deck_text(mark, "${:,.4f}")' in cards
    assert "_deck_note(mark)" in cards
    assert '"Live Price": _deck_text(mark, "${:,.4f}")' in cards
    assert '"Unrealized PnL": _deck_signed_money(unr_pnl)' in cards
    assert "pnl_color = _deck_pnl_color(unr_pnl)" in cards
    legacy_cards = _g5c_block("cards", LEGACY_DASHBOARD)
    assert '_deck_text(curr_p, "${:,.4f}")' in legacy_cards
    assert '"Live Price": _deck_text(curr_p, "${:,.4f}")' in legacy_cards
    assert "pnl_color = _deck_pnl_color(unr_pnl)" in legacy_cards


def test_g5c_the_strip_and_the_hero_card_render_the_rollup_not_a_sum():
    """The aggregate is published through the same measured-or-absent helpers as a per-position
    figure, so a partly-invented total cannot come back green."""
    strip = _g5c_block("strip", DASHBOARD)
    assert "_deck_pnl_color(total_unrealized_pnl)" in strip
    assert "_deck_signed_money(true_portfolio_net, 2)" in strip
    assert "_deck_sum(total_unrealized_pnl" in strip
    assert "_deck_note(true_portfolio_net)" in strip
    assert "true_port_color = _deck_pnl_color(true_portfolio_net)" in strip
    assert "color: {true_port_color};" in strip
    hero = _g5c_block("hero", DASHBOARD)
    assert "_deck_signed_money(total_unrealized_pnl, 2)" in hero
    assert "total_unrealized_pnl += " not in hero
    assert "_deck_signed_money(total_unrealized_pnl, 2)" in _g5c_block("hero", LEGACY_DASHBOARD)


def test_g5c_the_chart_draws_no_series_off_a_mark_it_does_not_have():
    """The chart's own `curr_p: float = 0.0` default falls back to the last candle close — a real
    exchange print — instead of the panel's entry-price stand-in, at both call sites."""
    block = _g5c_block("chart", DASHBOARD)
    assert block.count('curr_p=_deck_chart_level(p["curr"])') == 2
    assert 'curr_p=p["curr"],' not in block
    assert "Mark: `{_deck_text(p['curr'], '${:,.4f}')}`" in block


def test_g5c_the_twin_helpers_are_byte_identical_and_read_the_same_price():
    """The convention the branch carries for `scripts/live_dashboard.py`: same bytes, same answer
    on the same row, for the C-2 and C-3 helpers as well as the C-1 ones."""
    for name in _G5C_C2_HELPERS + _G5C_C3_HELPERS:
        assert _g5c_helper_code(name, DASHBOARD) == _g5c_helper_code(name, LEGACY_DASHBOARD), name
    row = pd.Series({"symbol": "BTCUSDT", "side": "long", "entry_price": 100000.0, "qty": 0.01,
                     "leverage": None})
    for name, args in (("_position_leverage", (row,)),
                       ("_position_deck_readings", (row, {}, {}))):
        dashboard_read = _g4_helper(name, DASHBOARD)(*args)
        legacy_read = _g4_helper(name, LEGACY_DASHBOARD)(*args)
        assert dashboard_read == legacy_read, (name, dashboard_read, legacy_read)


# ── C-4: absent funding rendered "$0.0000 (Neutral)" and was then ADDED to PnL ──────────────

# Banned text plus one line that genuinely contains it, so the prose-immunity half below is a
# second proof rather than a restatement of the first.
_G5C_FUNDING_STAND_IN_CASES = (
    ('num(f_info.get("total_funding"), 0.0)',
     'pos_funding = num(f_info.get("total_funding"), 0.0)'),
    ('total_live_funding += num(',
     'total_live_funding += num(f_info.get("total_funding"), 0.0)'),
    ('total_live_funding = 0.0', 'total_live_funding = 0.0'),
    ('(Neutral)', 'pos_funding_str = "$0.0000 (Neutral)"'),
    ('else "Neutral"',
     'fund_status = "Gained" if total_live_funding > 0 else ("Paid" if total_live_funding < 0 else "Neutral")'),
    ('_deck_reading(pos_funding)',
     'true_net_pnl = _deck_sum(unr_pnl, _deck_reading(pos_funding))'),
    ('_deck_reading(total_live_funding)',
     'true_portfolio_net = _deck_sum(total_unrealized_pnl, _deck_reading(total_live_funding))'),
    ('"#00ff66" if total_live_funding >= 0',
     'fund_badge_color = "#00ff66" if total_live_funding >= 0 else "#ffaa00"'),
)

# `_position_funding` is a dependency of the byte-identical `_position_deck_readings`, so the
# legacy twin carries the same bytes. The funding *renderers* below speak to panels the twin does
# not have (it has no funding banner, no per-position funding line and no fleet funding total), so
# they live only in dashboard.py and this suite only asks for them there.
_G5C_FUNDING_SHARED_HELPERS = ("_position_funding", "_position_deck_readings", "_deck_sum",
                                "_deck_signed_money", "_deck_text", "_deck_note", "_deck_color",
                                "_deck_pnl_color", "_deck_is_absent", "_deck_reading",
                                "_deck_absence")
_G5C_FUNDING_DASHBOARD_HELPERS = ("_deck_funding_word", "_deck_funding_color",
                                  "_deck_funding_text", "_deck_funding_total")


def _g5c_funding_code(path=DASHBOARD):
    """The C-4 surface: the aggregate, the hero card, the funding bar, the funding banner, the
    per-position cards, and every helper that decides what a funding figure prints."""
    parts = [_g5c_block("agg", path), _g5c_block("hero", path), _g5c_block("cards", path)]
    names = list(_G5C_FUNDING_SHARED_HELPERS)
    if path == DASHBOARD:
        parts += [_g5c_block("funding_bar", path), _g5c_block("strip", path)]
        names += list(_G5C_FUNDING_DASHBOARD_HELPERS)
    parts += [_g5c_helper_code(name, path) for name in names]
    return "\n".join(parts)


def _g5c_funding_entry(value):
    """One telemetry entry in the shape `fetch_live_position_funding` writes: the same dict under
    the raw symbol, the `/USDT` spelling and the clean form."""
    info = {"total_funding": value, "open_fee_total": 0.01, "close_fee_total": 0.0,
            "cur_realised_pnl": 0.0, "unrealised_pnl": 0.0}
    return {"BTCUSDT": info, "BTC/USDT": info, "BTC": info}


# Every shape that means "the funding map says nothing about this position". `live_position_funding`
# is `{}` in trading_data.db today (contract item 9), so the first one is the shape the fleet runs
# with for every symbol.
_G5C_UNMEASURED_FUNDING_MAPS = (
    ("map missing", None),
    ("empty map", {}),
    ("other symbol only", {"ETHUSDT": {"total_funding": 1.5}}),
    ("empty entry", {"BTCUSDT": {}}),
    ("no total_funding key", {"BTCUSDT": {"open_fee_total": 0.01}}),
    ("null funding", {"BTCUSDT": {"total_funding": None}}),
    ("placeholder funding", {"BTCUSDT": {"total_funding": "syncing..."}}),
    ("nan funding", {"BTCUSDT": {"total_funding": float("nan")}}),
    ("entry not a dict", {"BTCUSDT": "0.42"}),
)


@pytest.mark.parametrize(
    "funding_map", [m for _, m in _G5C_UNMEASURED_FUNDING_MAPS],
    ids=[label for label, _ in _G5C_UNMEASURED_FUNDING_MAPS])
def test_g5c_an_unmeasured_funding_is_an_absence_and_not_a_zero_verdict(funding_map):
    """C-4's headline: with `live_position_funding` = {} the deck printed `$0.0000 (Neutral)` for
    every symbol. Absence is not a verdict, so it renders the house absence and never the green."""
    read = _g4_helper("_position_funding", DASHBOARD)(funding_map, "BTC/USDT", "BTCUSDT")
    assert _deck_absent(read), read
    assert read["value"] is None and read["measured"] is False, read
    assert read["reason"], "an absence must carry the reason the deck renders"
    text = _g4_helper("_deck_funding_text", DASHBOARD)(read)
    assert text == "not measured", text
    assert "0.0000" not in text and "Neutral" not in text, text
    assert _g4_helper("_deck_funding_word", DASHBOARD)(read) == "not measured"
    colour = _g4_helper("_deck_funding_color", DASHBOARD)(read)
    assert colour == "#94a3b8", colour
    assert colour != "#00ff66", "an unmeasured funding may not wear the all-clear"
    note = _g4_helper("_deck_note", DASHBOARD)(read)
    assert note and "Neutral" not in note, note
    assert _g4_helper("_deck_signed_money", DASHBOARD)(read) == "not measured"


@pytest.mark.parametrize("stored", [0.0, 0, "0.0", -0.0],
                         ids=["zero", "int zero", "zero string", "negative zero"])
def test_g5c_a_recorded_zero_funding_is_a_reading_and_stays_green(stored):
    """The asymmetry the mark rule already pins, mirrored for funding: a recorded 0.0 IS a
    measurement — it says the position paid and received nothing — so it keeps its number, its
    "(Flat)" word, the all-clear green and no absence note."""
    read = _g4_helper("_position_funding", DASHBOARD)(_g5c_funding_entry(stored),
                                                      "BTC/USDT", "BTCUSDT")
    assert read == {"value": 0.0, "measured": True, "reason": ""}, read
    assert _g4_helper("_deck_funding_text", DASHBOARD)(read) == "+$0.0000 (Flat)"
    assert _g4_helper("_deck_funding_color", DASHBOARD)(read) == "#00ff66"
    assert _g4_helper("_deck_note", DASHBOARD)(read) == ""
    assert "not measured" not in _g4_helper("_deck_funding_text", DASHBOARD)(read)


def test_g5c_a_measured_funding_keeps_its_sign_its_word_and_its_colour():
    """Gained and Paid stay the two verdicts the exchange's own print supports — nothing about an
    unread symbol can reach either branch."""
    for value, word, colour in ((0.4210, "Gained", "#00ff66"), (-0.4210, "Paid", "#ffaa00")):
        read = _g4_helper("_position_funding", DASHBOARD)(_g5c_funding_entry(value),
                                                          "BTC/USDT", "BTCUSDT")
        assert read["measured"] is True, read
        assert _g4_helper("_deck_funding_text", DASHBOARD)(read) == \
            f"{_g4_helper('_deck_signed_money', DASHBOARD)(read)} ({word})", read
        assert _g4_helper("_deck_funding_color", DASHBOARD)(read) == colour, read


def test_g5c_the_funding_reading_survives_the_spellings_the_producer_writes():
    """`fetch_live_position_funding` keys one entry three ways; the deck looks the position up
    under the clean symbol and then the raw one, and both spellings are the same reading."""
    funding_map = _g5c_funding_entry(1.25)
    for symbol, clean in (("BTCUSDT", ""), ("BTC/USDT", "BTCUSDT"), ("BTC:USDT", "BTCUSDT"),
                          ("btc", "BTCUSDT")):
        read = _g4_helper("_position_funding", DASHBOARD)(funding_map, symbol, clean)
        assert read["measured"] is True, (symbol, clean, read)
        assert read["value"] == pytest.approx(1.25), (symbol, clean, read)


def test_g5c_true_net_pnl_refuses_to_publish_when_funding_was_never_measured():
    """C-4's arithmetic half. The panel wrapped an unmeasured funding in `_deck_reading(...)` and
    added it to PnL, dressing it as a measurement at the one call site that was otherwise correct.
    Decision: SUPPRESS, not label — see the comment at the card's `true_net_pnl` site."""
    helper = _g4_helper("_deck_sum", DASHBOARD)
    unr = _g4_helper("_deck_reading", DASHBOARD)(12.5)
    absent = _g4_helper("_position_funding", DASHBOARD)({}, "BTCUSDT", "BTCUSDT")
    net = helper(unr, absent)
    assert _deck_absent(net), net
    assert "funding" in net["reason"], net
    assert _g4_helper("_deck_pnl_color", DASHBOARD)(net) == "#94a3b8", net
    assert _g4_helper("_deck_signed_money", DASHBOARD)(net) == "not measured"
    measured = _g4_helper("_position_funding", DASHBOARD)(_g5c_funding_entry(2.5),
                                                          "BTCUSDT", "BTCUSDT")
    assert helper(unr, measured)["value"] == pytest.approx(15.0), helper(unr, measured)
    zero = _g4_helper("_position_funding", DASHBOARD)(_g5c_funding_entry(0.0),
                                                      "BTCUSDT", "BTCUSDT")
    flat = helper(unr, zero)
    assert flat["measured"] is True and flat["value"] == pytest.approx(12.5), flat
    # and the other addend still gates it: a measured funding cannot rescue an absent mark PnL
    blind = _g4_helper("_deck_absence", DASHBOARD)("no live price for this symbol in the current fetch")
    suppressed = helper(blind, measured)
    assert _deck_absent(suppressed) and "live price" in suppressed["reason"], suppressed


def test_g5c_the_fleet_funding_total_is_an_absence_when_no_symbol_reported_funding():
    """`total_live_funding` started at 0.0 and `+=`'d each row's stand-in, so an empty telemetry
    map summed to a printed `+$0.0000 (Neutral)`. The aggregate now has its own measured-or-absent
    state and says how many positions are missing it."""
    total = _g4_helper("_deck_funding_total", DASHBOARD)
    readings = _g4_helper("_position_funding", DASHBOARD)
    absent = readings({}, "BTCUSDT", "BTCUSDT")
    aggregate = total([absent, absent])
    assert _deck_absent(aggregate), aggregate
    assert aggregate["reason"] == "2 of 2 positions report no funding", aggregate
    assert "0.0000" not in _g4_helper("_deck_funding_text", DASHBOARD)(aggregate)
    assert _g4_helper("_deck_funding_color", DASHBOARD)(aggregate) == "#94a3b8"
    mixed = total([readings(_g5c_funding_entry(0.5), "BTCUSDT", "BTCUSDT"), absent])
    assert _deck_absent(mixed) and mixed["reason"] == "1 of 2 positions report no funding", mixed
    all_read = total([readings(_g5c_funding_entry(0.5), "BTCUSDT", "BTCUSDT"),
                      readings({"ETHUSDT": {"total_funding": -0.25}}, "ETHUSDT", "ETHUSDT"),
                      readings(_g5c_funding_entry(0.0), "BTCUSDT", "BTCUSDT")])
    assert all_read["measured"] is True and all_read["value"] == pytest.approx(0.25), all_read
    empty = total([])
    assert _deck_absent(empty) and "no open positions" in empty["reason"], empty


def test_g5c_a_seeded_position_carries_its_funding_reading_into_the_deck(tmp_path):
    """The reading reaches the deck through the shipped frame, the shipped SELECTs and the shipped
    `_position_deck_readings` — not through a float the panel defaulted."""
    row = _g5c_seeded_row(tmp_path, leverage=5)
    prices = {"BTCUSDT": 105000.0}
    funded = _g4_helper("_position_deck_readings", DASHBOARD)(
        row, prices, _g5c_funding_entry(0.75))
    assert funded["funding"] == {"value": 0.75, "measured": True, "reason": ""}, funded
    blind = _g4_helper("_position_deck_readings", DASHBOARD)(row, prices, {})
    assert _deck_absent(blind["funding"]), blind
    net = _g4_helper("_deck_sum", DASHBOARD)(blind["unr_pnl"], blind["funding"])
    assert _deck_absent(net) and "funding" in net["reason"], net
    rollup = _g4_helper("_deck_funding_total", DASHBOARD)([blind["funding"]])
    assert _deck_absent(rollup) and rollup["reason"] == "1 of 1 positions report no funding", rollup


@pytest.mark.parametrize("banned,code_line", _G5C_FUNDING_STAND_IN_CASES)
def test_g5c_no_zero_funding_stand_in_survives_in_the_deck(banned, code_line):
    """No `num(..., 0.0)` on the funding map, no `+=` into a float total, no `_deck_reading()`
    wrapper dressing an unmeasured funding, no green-at-zero funding colour — in the panels or in
    the helpers behind them, in either file."""
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        hits = _g5_banned_hits(_g5c_deck_code(path), [banned])
        assert hits == [], f"{path.name}'s deck blocks still contain {banned!r}"
        hits = _g5_banned_hits(_g5c_funding_code(path), [banned])
        assert hits == [], f"{path.name}'s funding surface still contains {banned!r}"


@pytest.mark.parametrize("banned,code_line", _G5C_FUNDING_STAND_IN_CASES)
def test_g5c_a_comment_neither_trips_nor_hides_a_funding_scan(banned, code_line):
    """Both halves on C-4's own tokens."""
    prose = f"def f():\n    # the deck used to publish {banned} here\n    return 1\n"
    assert _g5_banned_hits(prose, [banned]) == []
    documented = f'def f():\n    """the deck used to publish {banned} here. """\n    return 1\n'
    assert _g5_banned_hits(documented, [banned]) == []
    real = f"def f():\n    {code_line}\n    return 1\n"
    assert _g5_banned_hits(real, [banned]) == [banned], (
        f"{banned!r} is in the code and the stripped scan no longer sees it")
    hidden = f"def f():\n    {code_line}  # {banned}\n    return 1\n"
    assert _g5_banned_hits(hidden, [banned]) == [banned]


def test_g5c_the_card_and_the_table_print_the_funding_reading_or_nothing():
    """The render sites consume the reading: the card's badge, the "Funding (Live)" column and the
    True Net line all come from the helpers, none of them from a defaulted float."""
    cards = _g5c_block("cards", DASHBOARD)
    assert 'pos_funding = p_read["funding"]' in cards
    assert "pos_funding_str = _deck_funding_text(pos_funding)" in cards
    assert 'pos_fund_color = _deck_funding_color(pos_funding)' in cards
    assert '"Funding (Live)": pos_funding_str' in cards
    assert "💸 Live Funding: {pos_funding_badge}" in cards
    assert "true_net_pnl = _deck_sum(unr_pnl, pos_funding)" in cards
    assert "_deck_reading(pos_funding)" not in cards
    assert "pos_funding_badge" in cards and "_deck_note(pos_funding)" in cards


def test_g5c_the_hero_the_bar_and_the_banner_render_the_total_through_the_helpers():
    """The aggregate is published through the same measured-or-absent helpers as a per-position
    figure, so an unmeasured fleet funding cannot come back as `+$0.0000 (Neutral)` in green."""
    agg = _g5c_block("agg", DASHBOARD)
    assert "total_live_funding = _deck_funding_total(" in agg
    assert "total_live_funding += " not in agg
    hero = _g5c_block("hero", DASHBOARD)
    assert 'delta=f"Funding: {_deck_funding_text(total_live_funding)}"' in hero
    assert '_deck_is_absent(total_live_funding)' in hero
    bar = _g5c_block("funding_bar", DASHBOARD)
    assert "fund_badge_color = _deck_funding_color(total_live_funding)" in bar
    assert "Active Funding: <b style=\"color: {fund_badge_color};\">" \
           "{_deck_funding_text(total_live_funding)}</b>{_deck_note(total_live_funding)}" in bar
    strip = _g5c_block("strip", DASHBOARD)
    assert "total_fund_color = _deck_funding_color(total_live_funding)" in strip
    assert "_deck_sum(total_unrealized_pnl, total_live_funding)" in strip
    assert "_deck_note(true_portfolio_net)" in strip


def test_g5c_the_twin_carries_the_funding_reader_and_none_of_the_funding_panels():
    """`scripts/live_dashboard.py` has no funding deck (measured, not assumed), so it carries only
    the shared reader — byte-identical, answering the same — and the funding-only renderers stay
    where the panels that call them are."""
    assert _g5c_helper_code("_position_funding", DASHBOARD) == \
        _g5c_helper_code("_position_funding", LEGACY_DASHBOARD)
    for funding_map in (None, {}, _g5c_funding_entry(0.75), _g5c_funding_entry(0.0)):
        dashboard_read = _g4_helper("_position_funding", DASHBOARD)(funding_map, "BTC/USDT",
                                                                    "BTCUSDT")
        legacy_read = _g4_helper("_position_funding", LEGACY_DASHBOARD)(funding_map, "BTC/USDT",
                                                                        "BTCUSDT")
        assert dashboard_read == legacy_read, (funding_map, dashboard_read, legacy_read)
    row = pd.Series({"symbol": "BTCUSDT", "side": "long", "entry_price": 100000.0, "qty": 0.01,
                     "leverage": 5.0})
    for funding_map in ({}, _g5c_funding_entry(0.75)):
        assert _g4_helper("_position_deck_readings", DASHBOARD)(row, {}, funding_map) == \
            _g4_helper("_position_deck_readings", LEGACY_DASHBOARD)(row, {}, funding_map)
    legacy_src = _g5_code_only(LEGACY_DASHBOARD.read_text(encoding="utf-8"))
    for name in _G5C_FUNDING_DASHBOARD_HELPERS:
        assert f"def {name}(" not in legacy_src, (
            f"{name} was added to the twin, which has no panel that calls it")


# ── C-5: the deck states limits it did not read ────────────────────────────────────────────────

# The ceiling the gauge is measured against, the cash buffer it implies and the two colour rungs
# under the status word were all invented by the panel: `state_num(key, 70.0)` returns 70.0 when the
# row is absent, "(Ceiling: 70%)" and "(30% Cash Buffer)" were literals, and neither 45 nor 65 was
# ever rendered. Each pair is the banned text plus one line that genuinely contains it, so the
# prose-immunity half below is a second proof rather than a restatement of the first.
_G5C_CEILING_STAND_IN_CASES = (
    ('state_num("GLOBAL_MARGIN_LIMIT", 70.0)',
     'margin_limit_pct = state_num("GLOBAL_MARGIN_LIMIT", 70.0)'),
    ('query_system_state("GLOBAL_MARGIN_LIMIT", "70.0")',
     'margin_limit_pct = float(query_system_state("GLOBAL_MARGIN_LIMIT", "70.0"))'),
    ('(Ceiling: 70%)', 'delta = f"{b_margin_display} locked (Ceiling: 70%)"'),
    ('(70% Max)', 'label = "Margin Utilized (70% Max)"'),
    ('Ceiling (30% Cash Buffer)',
     'text = f"Margin Limit: {margin_limit_pct:.1f}% Ceiling (30% Cash Buffer)"'),
    ('_deck_reading(margin_limit_pct)',
     'gauge = _deck_margin_gauge(total, balance, _deck_reading(margin_limit_pct))'),
    ('margin_limit_pct:', 'text = f"{margin_limit_pct:.0f}%"'),
    ('value < 45.0', 'low_rung = value < 45.0'),
    ('value < 65.0', 'mid_rung = value < 65.0'),
)

# `_deck_ceiling_card` speaks to the executive-briefing card, which exists only in dashboard.py —
# the legacy twin has no briefing tab, so it carries no ceiling card either.
_G5C_CEILING_SHARED_HELPERS = ("_deck_margin_limit", "_deck_cash_buffer_pct",
                               "_deck_margin_limit_line", "_deck_margin_gauge",
                               "_deck_margin_rung", "_deck_margin_rung_label", "_deck_reading",
                               "_deck_absence", "_deck_is_absent", "_deck_text", "_deck_note",
                               "_deck_color", "_deck_value_or", "_optional_float")
_G5C_CEILING_DASHBOARD_HELPERS = ("_deck_ceiling_card",)


def _g5c_ceiling_code(path=DASHBOARD):
    """The C-5 surface: the limit read, the gauge, the bar, the briefing card, the risk strip and
    every helper that decides what a ceiling or a rung prints."""
    parts = [_g5c_block("agg", path), _g5c_block("hero", path),
             _g5c_block("funding_bar", path), _g5c_block("limits", path)]
    if path == DASHBOARD:
        parts.append(_g5c_block("briefing", path))
    names = list(_G5C_CEILING_SHARED_HELPERS)
    if path == DASHBOARD:
        names += list(_G5C_CEILING_DASHBOARD_HELPERS)
    parts += [_g5c_helper_code(name, path) for name in names]
    return "\n".join(parts)


def _g5c_gauge_limit_arg(path=DASHBOARD):
    """The third argument the shipped panel hands `_deck_margin_gauge`, unparsed from its own tree.

    Whether that argument is a reading or a defaulted float wrapped as one is the whole of C-5's
    gauge half, and a text match on a wrapped multi-line call is not a check of anything.
    """
    import ast

    calls = [node for node in ast.walk(ast.parse(_g5c_block("hero", path)))
             if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_deck_margin_gauge"]
    assert len(calls) == 1, (
        f"{path.name}'s hero block no longer calls the gauge exactly once: {len(calls)} calls")
    assert len(calls[0].args) == 3, ast.unparse(calls[0])
    return ast.unparse(calls[0].args[2])


_G5C_UNREAD_LIMIT_SHAPES = (
    ("absent row", ""),
    ("null", None),
    ("placeholder", "syncing..."),
    ("not a number", "aggressive"),
    ("nan", "nan"),
    ("zero percent", "0.0"),
    ("negative", "-5.0"),
    ("above the wallet", "101.5"),
)


@pytest.mark.parametrize("stored", [s for _, s in _G5C_UNREAD_LIMIT_SHAPES],
                         ids=[label for label, _ in _G5C_UNREAD_LIMIT_SHAPES])
def test_g5c_ceiling_an_unread_margin_limit_is_an_absence_not_a_silent_70(stored):
    """C-5's headline: `query_system_state` hands back its default when the row is missing, so an
    absent `GLOBAL_MARGIN_LIMIT` silently became 70.0 and every ceiling sentence downstream was a
    restatement of `config.py:237`'s in-tree default, not a reading."""
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        read = _g4_helper("_deck_margin_limit", path)(stored)
        assert _deck_absent(read), (path.name, stored, read)
        assert read["value"] is None and read["measured"] is False, read
        assert read["reason"], "an absence must carry the reason the deck renders"
        assert _g4_helper("_deck_text", path)(read, "{:.0f}%") == "not measured"
        colour = _g4_helper("_deck_color", path)(read, "#00ff66")
        assert colour == "#94a3b8", colour
        assert colour != "#00ff66", "an unread ceiling may not wear the all-clear"
        assert _g4_helper("_deck_note", path)(read) != ""
        line = _g4_helper("_deck_margin_limit_line", path)(
            read, _g4_helper("_deck_cash_buffer_pct", path)(read))
        assert "not measured" in line, line
        assert "70" not in line and "30%" not in line, line


@pytest.mark.parametrize("stored,ceiling,buffer",
                         [("70.0", "70.0", "30"), ("55.0", "55.0", "45"),
                          ("80.0", "80.0", "20"), ("42.5", "42.5", "58")],
                         ids=["as default", "lower", "higher", "fractional"])
def test_g5c_ceiling_the_cash_buffer_is_derived_from_the_limit_that_was_read(stored, ceiling,
                                                                            buffer):
    """"(30% Cash Buffer)" was only ever true while the limit was 70. The buffer is now 100% minus
    the read ceiling, and a limit of 70 still prints the sentence it printed before."""
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        limit = _g4_helper("_deck_margin_limit", path)(stored)
        assert limit == {"value": float(stored), "measured": True, "reason": ""}, limit
        read = _g4_helper("_deck_cash_buffer_pct", path)(limit)
        assert read["measured"] is True and read["value"] == pytest.approx(100.0 - float(stored))
        line = _g4_helper("_deck_margin_limit_line", path)(limit, read)
        assert line == f"{ceiling}% Ceiling ({buffer}% Cash Buffer)", line


def test_g5c_ceiling_the_limit_holds_its_own_shape():
    """A percentage ceiling is a usable-or-not gate, and the sentence keeps a stored 0.0 out of the
    gauge's denominator rather than printing 0% as if the fleet had configured a wall at zero."""
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        zero = _g4_helper("_deck_margin_limit", path)("0.0")
        assert _deck_absent(zero) and "0" not in _g4_helper("_deck_text", path)(zero), zero
        assert _g4_helper("_deck_cash_buffer_pct", path)(zero)["measured"] is False
        assert _g4_helper("_deck_margin_limit", path)("00.5")["measured"] is True


def test_g5c_ceiling_a_recorded_limit_still_drives_the_ratio_and_the_bar():
    """No over-withholding: with a stored ceiling the utilization, the bar width and the rung are
    the shipped arithmetic, and the bar is measured against the number on screen beside it."""
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        margin = _g4_helper("_deck_reading", path)(2000.0)
        balance = _g4_helper("_deck_account_balance", path)(10000.0)
        for stored, bar in (("70.0", 20.0 / 70.0 * 100.0), ("40.0", 50.0), ("25.0", 80.0)):
            gauge = _g4_helper("_deck_margin_gauge", path)(
                margin, balance, _g4_helper("_deck_margin_limit", path)(stored))
            assert gauge["utilization"]["measured"] is True, (stored, gauge)
            assert gauge["utilization"]["value"] == pytest.approx(20.0), (stored, gauge)
            assert gauge["bar_width"]["value"] == pytest.approx(bar), (stored, gauge)
            assert gauge["buffer"]["value"] == pytest.approx(8000.0), (stored, gauge)


def test_g5c_ceiling_an_absent_limit_greys_the_gauge_instead_of_earning_a_green_bar():
    """The gauge that depends on the ceiling renders the absence: no ratio, no bar, no verdict, and
    never the green all-clear for a wall nobody recorded."""
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        gauge = _g4_helper("_deck_margin_gauge", path)(
            _g4_helper("_deck_reading", path)(2000.0),
            _g4_helper("_deck_account_balance", path)(10000.0),
            _g4_helper("_deck_margin_limit", path)(""))
        for key in ("utilization", "bar_width", "buffer"):
            assert _deck_absent(gauge[key]), (key, gauge)
            assert "margin limit" in gauge[key]["reason"], (key, gauge[key])
        rung = _g4_helper("_deck_margin_rung", path)(gauge["utilization"])
        assert rung["color"] == "#94a3b8" and rung["status"] == "NOT MEASURED", rung
        assert _g4_helper("_deck_value_or", path)(gauge["bar_width"]) == 0.0, (
            "a gauge with no ceiling must draw nothing, not a full bar")
        assert _g4_helper("_deck_margin_rung_label", path)(gauge["utilization"]) == "", (
            "an unread utilization has no rung to name")


@pytest.mark.parametrize("util,colour,status",
                         [(0.0, "#00ff66", "SAFE BUFFER"), (44.9, "#00ff66", "SAFE BUFFER"),
                          (45.0, "#ff9900", "ELEVATED EXPOSURE"),
                          (64.9, "#ff9900", "ELEVATED EXPOSURE"),
                          (65.0, "#ff003c", "CRITICAL / NEARING CEILING")])
def test_g5c_ceiling_the_rung_that_coloured_the_bar_is_named_where_it_is_read(util, colour,
                                                                              status):
    """The status words were the only label, and 45/65 lived nowhere on screen. Each rung keeps its
    colour and its words exactly as they were and gains a name the operator can read."""
    safe = _g5c_module_constant("_DECK_MARGIN_SAFE_RUNG_PCT")
    elevated = _g5c_module_constant("_DECK_MARGIN_ELEVATED_RUNG_PCT")
    assert _g5c_module_constant("_DECK_MARGIN_SAFE_RUNG_PCT",
                                LEGACY_DASHBOARD) == safe, "the twin's rungs drifted"
    assert _g5c_module_constant("_DECK_MARGIN_ELEVATED_RUNG_PCT",
                                LEGACY_DASHBOARD) == elevated, "the twin's rungs drifted"
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        read = _g4_helper("_deck_reading", path)(util)
        assert _g4_helper("_deck_margin_rung", path)(read) == {"color": colour, "status": status}
        label = _g4_helper("_deck_margin_rung_label", path)(read)
        assert f"{safe:.0f}%" in label and f"{elevated:.0f}%" in label, label
        assert "(panel's own)" in label, label
        assert "SAFE BUFFER" not in label and status not in label, (
            f"the rung label must name thresholds, not repeat the verdict: {label}")


def test_g5c_ceiling_the_panel_reads_the_key_without_a_default_in_both_files():
    """The two reader shapes differ (`state_num` in dashboard.py, `float(query_system_state(...))`
    in the twin) and both are replaced by the same measured-or-absent read of the same key."""
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        hero = _g5c_block("hero", path)
        assert 'margin_limit = _deck_margin_limit(query_system_state("GLOBAL_MARGIN_LIMIT", ""))' \
            in hero, f"{path.name} no longer reads the ceiling with no default"
        assert "margin_cash_buffer_pct = _deck_cash_buffer_pct(margin_limit)" in hero
        assert "margin_limit_display = _deck_text(margin_limit, " in hero
        assert _g5c_gauge_limit_arg(path) == "margin_limit", (
            f"{path.name} feeds the gauge something other than the read ceiling")


def test_g5c_ceiling_the_bar_and_the_risk_strip_render_the_reading():
    """Both render sites consume the helpers, and the twin's strip answers with the same sentence as
    dashboard.py's — the residue was asymmetric, so each file is pinned on its own."""
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        bar = _g5c_block("funding_bar", path)
        assert "of {margin_ceiling_display}{_deck_note(margin_limit)}" in bar
        assert "({margin_status}{margin_rung_label})" in bar
        limits = _g5c_block("limits", path)
        assert "margin_limit_line = _deck_margin_limit_line(margin_limit, margin_cash_buffer_pct)" \
            in limits
        assert 'st.write(f"• **Global Margin Limit:** {margin_limit_line}", ' \
               "unsafe_allow_html=True)" in limits


def test_g5c_ceiling_the_briefing_card_prints_the_ceiling_it_read():
    """Behavioural half of defect (a): a stored 55 cannot render "Ceiling: 70%" two lines from the
    strip that says 55%, and an unread ceiling says so instead of naming the config default."""
    card = _g4_helper("_deck_ceiling_card", DASHBOARD)
    for stored, ceiling in (("70.0", "70"), ("55.0", "55"), ("80.0", "80")):
        out = card(_g4_helper("_deck_margin_limit", DASHBOARD)(stored))
        assert out == {"label": f"{ceiling}% Max", "delta": f"Ceiling: {ceiling}%"}, (stored, out)
    unread = card(_g4_helper("_deck_margin_limit", DASHBOARD)(""))
    assert "not measured" in unread["label"] and "not measured" in unread["delta"], unread
    assert "70" not in str(unread) and "30" not in str(unread), unread
    briefing = _g5c_block("briefing", DASHBOARD)
    assert "margin_card = _deck_ceiling_card(margin_limit)" in briefing
    assert 'Margin Utilized ({margin_card["label"]})' in briefing
    assert 'locked ({margin_card["delta"]})' in briefing


@pytest.mark.parametrize("banned,code_line", _G5C_CEILING_STAND_IN_CASES)
def test_g5c_no_invented_ceiling_or_unlabelled_rung_survives_in_the_deck(banned, code_line):
    """No defaulted ceiling read, no `_deck_reading()` wrapper dressing one, no hardcoded 70% card,
    no 30% buffer literal, no bare 45/65 rung comparison — in either file, panel or helper."""
    for path in (DASHBOARD, LEGACY_DASHBOARD):
        hits = _g5_banned_hits(_g5c_deck_code(path), [banned])
        assert hits == [], f"{path.name}'s deck blocks still contain {banned!r}"
        hits = _g5_banned_hits(_g5c_ceiling_code(path), [banned])
        assert hits == [], f"{path.name}'s ceiling surface still contains {banned!r}"


@pytest.mark.parametrize("banned,code_line", _G5C_CEILING_STAND_IN_CASES)
def test_g5c_a_comment_neither_trips_nor_hides_a_ceiling_scan(banned, code_line):
    """Both halves on C-5's own tokens."""
    prose = f"def f():\n    # the deck used to publish {banned} here\n    return 1\n"
    assert _g5_banned_hits(prose, [banned]) == []
    documented = f'def f():\n    """the deck used to publish {banned} here. """\n    return 1\n'
    assert _g5_banned_hits(documented, [banned]) == []
    real = f"def f():\n    {code_line}\n    return 1\n"
    assert _g5_banned_hits(real, [banned]) == [banned], (
        f"{banned!r} is in the code and the stripped scan no longer sees it")
    hidden = f"def f():\n    {code_line}  # {banned}\n    return 1\n"
    assert _g5_banned_hits(hidden, [banned]) == [banned]


def test_g5c_ceiling_the_twin_carries_the_shared_ceiling_readers_and_no_briefing_card():
    """`scripts/live_dashboard.py` cannot import dashboard.py, so the readers are duplicated and
    pinned byte-for-byte; the briefing card it does not have keeps its card helper out of it."""
    for name in _G5C_CEILING_SHARED_HELPERS:
        assert _g5c_helper_code(name, DASHBOARD) == _g5c_helper_code(name, LEGACY_DASHBOARD), name
    for stored in ("", "70.0", "55.0", "0.0", "syncing..."):
        assert _g4_helper("_deck_margin_limit", DASHBOARD)(stored) == \
            _g4_helper("_deck_margin_limit", LEGACY_DASHBOARD)(stored), stored
    assert _g4_helper("_deck_margin_rung_label", DASHBOARD)(
        _g4_helper("_deck_reading", DASHBOARD)(50.0)) == \
        _g4_helper("_deck_margin_rung_label", LEGACY_DASHBOARD)(
            _g4_helper("_deck_reading", LEGACY_DASHBOARD)(50.0))
    legacy_src = _g5_code_only(LEGACY_DASHBOARD.read_text(encoding="utf-8"))
    for name in _G5C_CEILING_DASHBOARD_HELPERS:
        assert f"def {name}(" not in legacy_src, (
            f"{name} was added to the twin, which has no panel that calls it")

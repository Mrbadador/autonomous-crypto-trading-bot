"""tests/test_measurement_fidelity.py — Task 5 of the 2026-10-22 audit remediation.

Pins the measurement machinery itself: a Sharpe ratio annualized by the trade frequency a run
actually produced (not by a constant), trailing stops that may only be tested against the bar they
are carried into, higher-timeframe values that may only be read once their bar has closed, indicator
warm-up windows that stay NaN instead of being back-filled, and datasets an arm asked for and did
not find being reported as missing instead of silently skipped.
"""

import re
from datetime import datetime, timezone
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

def _dashboard_helper(name, path=DASHBOARD):
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
    assert "margin_measured = live_bal_val is not None and live_bal_val > 0" in src
    assert 'margin_util_display = f"{margin_util_pct:.1f}%" if margin_measured else ' \
           '"n/a (no account balance)"' in src
    assert "n/a (no account balance)" in src


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

# The fields the panel used to invent and that the live five-key payload does not carry.
_G5_ABSENT_IN_LIVE = ("capital_allocated_usd", "hurdle_rate_pct", "distance_to_liq_pct",
                      "margin_mode", "annualized_yield_pct", "unwind_hurdle_pct",
                      "funding_rate_8h", "cumulative_funding_harvested", "liquidation_price",
                      "fee_amortized_pct", "fee_hurdle_target_usd", "holding_days",
                      "min_holding_days", "basis_spread_usd", "basis_dislocation_pct")

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
    "b_fee_target_text": "b_fee_target_st",
    "b_liq_price_text": "b_liq_price_st", "b_liq_price_note": "b_liq_price_st",
    "b_dist_liq_text": "b_dist_liq_st", "b_dist_liq_note": "b_dist_liq_st",
    "b_margin_mode_text": "b_margin_mode_st", "b_margin_mode_note": "b_margin_mode_st",
    "b_holding_text": "b_holding_st", "b_min_holding_text": "b_min_holding_st",
}

# Every verdict/colour the block derives, and so every one that must be guarded.
_G5_DERIVED_NAMES = frozenset({
    "status_color", "status_badge", "yield_color", "funding_color", "liq_color", "fee_color",
    "fee_bar_width", "margin_guard_color", "disloc_color", "harvest_color", "delta_color",
    "b_delta_posture", "b_time_text", "b_time_color", "b_reason_line", "b_coverage_line",
    "b_unmeasured_count", "b_is_stale", "b_age_days",
})


class _G5RecordingStreamlit:
    """Captures what the panel emits, so its f-string can be rendered without an app."""

    def __init__(self):
        self.markdown_calls = []
        self.infos = []

    def markdown(self, body, **kwargs):
        self.markdown_calls.append(str(body))

    def info(self, body, **kwargs):
        self.infos.append(str(body))

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            return None
        return _record


def _g5_system_state_row(tmp_path, payload=_G5_LIVE_PAYLOAD):
    """Seed one real `system_state` row and hand back the JSON text sqlite stores for it.

    Written to a database in `tmp_path` — never to `trading_data.db`, which every test here opens
    read-only or not at all.
    """
    import json as _json
    import sqlite3

    db = tmp_path / "telemetry_fixture.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("CREATE TABLE system_state (key TEXT PRIMARY KEY, value TEXT, "
                     "updated_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        raw = _json.dumps(payload)
        conn.execute("INSERT INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                     ("basis_trader_telemetry", raw, "2026-09-17T22:45:48.677606+00:00"))
        conn.commit()
        stored = conn.execute("SELECT value FROM system_state WHERE key = ?",
                              ("basis_trader_telemetry",)).fetchone()[0]
    finally:
        conn.close()
    return str(stored)


def _g5_panel_namespace(path, raw_value, now=None):
    """A namespace holding dashboard.py's own shipped helpers, without booting dashboard.py.

    Only stdlib imports, the `_BASIS_*` module constants and the pure renderers the block calls are
    compiled out of the file, in source order. `streamlit` is replaced by a recorder and
    `query_system_state` by the seeded payload text.
    """
    import ast

    wanted_funcs = {"_optional_float", "_published_number", "_published_boolean",
                    "_published_state", "_absence_note", "utc_to_sast", "_basis_readings"}
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
        if any(isinstance(t, ast.Name) and t.id.startswith("_BASIS_") for t in targets):
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


def test_g5a_the_live_payload_keeps_net_delta_a_measured_zero():
    """`net_delta = 0.0` is the one number in the live payload. It must stay a reading.

    This is the regression the round is judged on: an absence is not a zero and a zero is not an
    absence. The panel also used to read a USD-suffixed key the payload has never carried, so the
    reading it printed came from a default rather than from the engine.
    """
    readings = _g5_readings(dict(_G5_LIVE_PAYLOAD))
    assert readings["net_delta"] == {"value": 0.0, "measured": True, "reason": ""}, readings["net_delta"]
    src = _function_source(DASHBOARD, "_basis_readings")
    assert '"net_delta": _number("net_delta")' in src, (
        "the carry readings do not read the key the engine publishes")
    assert "net_delta_usd" not in DASHBOARD.read_text(encoding="utf-8"), (
        "the panel still reads the USD-suffixed delta key the payload never carried")


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
@pytest.mark.parametrize("stand_in", ["or 8.0", "or 1.5", "or 1000.0", "or 7", "or 2.40",
                                      "or 3.00", "or 100.0", '"margin_mode", "crossed"'])
def test_g5a_no_numeric_stand_in_survives_in_the_carry_panel(stand_in):
    block = _g4_block("basis", DASHBOARD)
    assert stand_in not in block, f"the carry panel still invents a reading with {stand_in!r}"


def test_g5a_the_carry_panel_reads_through_the_publishers_and_the_scan_is_not_vacuous():
    """Positive side of the stand-in scan, plus the floor that makes the negative side mean
    something: a block that extracted the wrong region would be short and key-free.

    The payload keys are allowed to appear in exactly two places — the panel block and the pure
    `_basis_readings` helper it now calls. Anywhere else is a second reader of the same telemetry
    that this test is not looking at.
    """
    block = _g4_block("basis", DASHBOARD)
    assert "_basis_readings(basis_data)" in block, (
        "the carry panel does not read its payload through the published-reading helper")
    assert block.count("_absence_note(") >= 12, (
        f"the carry panel explains only {block.count('_absence_note(')} absences")
    assert len(block.splitlines()) >= 150, (
        f"the extracted carry panel is {len(block.splitlines())} lines, so this is not the panel")
    helper = _function_source(DASHBOARD, "_basis_readings")
    assert helper, "dashboard.py has no _basis_readings to route the panel through"
    src = DASHBOARD.read_text(encoding="utf-8")
    for key in ("capital_allocated_usd", "hurdle_rate_pct", "unwind_hurdle_pct",
                "distance_to_liq_pct", "fee_hurdle_target_usd", "basis_dislocation_pct",
                "fee_amortized_pct", "liquidation_price"):
        outside = src.count(f'"{key}"') - block.count(f'"{key}"') - helper.count(f'"{key}"')
        assert outside == 0, (
            f"{key} is read {outside} time(s) outside the carry panel and its reader helper, so "
            "the block this test scanned is not the only place the figure is invented")


def test_g5a_the_dead_stand_in_reads_are_gone():
    """Five fields were assigned and never used. A stand-in feeding nothing is still a stand-in."""
    src = DASHBOARD.read_text(encoding="utf-8")
    for key in ("unrealized_pnl_usd", "has_active_position", "spot_price", "perp_price",
                "roundtrip_fee_usd"):
        assert f'"{key}"' not in src, f"the carry panel still reads {key}, which renders nowhere"


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
    block = _g4_block("basis", DASHBOARD)
    assert "color: #00ff66; font-size: 0.95rem" not in block, (
        "a card still wears a hardcoded green instead of a measured colour")
    assert "harvest_color" in block, "the Funding Harvested card has no state-driven colour"
    assert 'if b_harvest_st["measured"] else "#94a3b8"' in block, (
        "the Funding Harvested colour does not fall back to absence grey")


def test_g5a_the_thresholds_are_named_as_thresholds():
    """125% and 20% are the panel's own scale, not measurements — and they may not sit beside a
    stand-in reading so closely that they read as part of it."""
    block = _g4_block("basis", DASHBOARD)
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
@pytest.mark.parametrize("claim", ["(Neutral)", "Margin Mode Active",
                                   "IDLE (SCANNING YIELD & SPREAD)"])
def test_g5a_the_fabricated_prose_claims_are_gone(claim):
    block = _g4_block("basis", DASHBOARD)
    assert claim not in block, f"the carry panel still asserts {claim!r} regardless of the payload"


def test_g5a_the_replacements_are_present_and_reachable():
    """Absence checks pass on a panel that deleted the card entirely, so the new wording is pinned
    in the block AND in the HTML the live payload actually renders."""
    block = _g4_block("basis", DASHBOARD)
    for reachable in ("STATE NOT MEASURED", "b_delta_posture", "Engine-reported reason",
                      "Margin mode:", "STALE TELEMETRY"):
        assert reachable in block, f"the carry panel has no {reachable!r} to fall back on"
    html, _fake = _g5_render_carry_panel(None)
    assert "STATE PUBLISHED" not in html and "⏸️ PARKED / DISABLED" in html, html[:2000]
    assert "posture not measured" not in html, "the published delta lost its posture"
    assert "Engine-reported reason: <b style='color: #e2e8f0;'>" \
           "Temporarily parked for account segregation</b>" in html, html
    assert "35d old · ⚠️ STALE TELEMETRY" in html, html
    assert "15 of 18 economics fields" in html, html
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


def test_g5a_every_derived_verdict_in_the_carry_panel_guards_on_measured():
    """A verdict computed from a reading that can be absent either raises TypeError (the panel then
    publishes NOTHING, which is worse than any stand-in) or publishes an all-clear nobody measured.

    Ancestor-walk over the enclosing `If` gates, the approach `test_g4_the_radar_banner_...` uses;
    its known limitation — a bundled gate also has `measured` in its test — is why the field values
    are checked separately and more strictly by `test_g5a_the_carry_panel_keys_each_field_on_its_own_state`.
    """
    import ast

    block = _g4_block("basis", DASHBOARD)
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
            assert "measured" in segment or _g4_enclosing_measured_guard(node, parents, block), (
                f"the carry panel derives {target.id} from a reading that can be absent: "
                f"{segment[:160]}")
    missing = sorted(_G5_DERIVED_NAMES - seen)
    assert not missing, (
        f"the carry panel no longer derives {missing}, so this guard has nothing to check")


@pytest.mark.parametrize("field,own_key", sorted(_G5_FIELD_OWNERS.items()))
def test_g5a_the_carry_panel_keys_each_field_on_its_own_state(field, own_key):
    """A missing hurdle may not make the panel disown the yield it measured, and a measured yield
    may not be published under the hurdle's flag.

    Stronger than the G-4 radar guard it is copied from: there, a ternary keyed on its own flag plus
    a sibling's still passed. Here the set of state variables a field's own line may name must be
    exactly one — its own.
    """
    import ast

    block = _g4_block("basis", DASHBOARD)
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


def _g5_enclosing_gate_is_bundled(node, parents, block, own_key):
    """True when an enclosing `If` gate reads a state variable other than `own_key`.

    The G-4 radar guard (`test_g4_the_radar_banner_keys_each_hawkes_text_on_its_own_state`) only
    excluded one named sibling, so a ternary keyed on its own flag *plus* a sibling's still passed.
    This one asks whether any other field's state is in scope at all.
    """
    import ast

    ancestor = parents.get(id(node))
    while ancestor is not None:
        if isinstance(ancestor, ast.If):
            gate = ast.get_source_segment(block, ancestor.test) or ""
            named = set(re.findall(r"\bb_[a-z_]*_st\b", gate))
            if named and named != {own_key}:
                return True
        ancestor = parents.get(id(ancestor))
    return False


def test_g5a_the_combined_yield_verdict_is_legitimately_bundled():
    """`yield_color` compares the yield against two thresholds, so it genuinely needs all three
    states — the one place in the panel where a bundled gate is the right answer, pinned so a later
    reader does not un-bundle it into `b_yield_st` alone and compare against a None hurdle."""
    import ast

    block = _g4_block("basis", DASHBOARD)
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
    """The other half of the crash hazard: with every field present, the panel must show the
    readings and stop claiming absences — and a full book must still format."""
    payload = {
        "state": "ACTIVE_DELTA_NEUTRAL", "reason": "carry captured", "margin_mode": "isolated",
        "capital_allocated_usd": 2500.0, "hurdle_rate_pct": 8.0, "unwind_hurdle_pct": 1.5,
        "annualized_yield_pct": 11.25, "funding_rate_8h": 0.00012, "basis_spread_usd": -350.25,
        "basis_dislocation_pct": 0.42, "cumulative_funding_harvested": 1.8675, "net_delta": -0.75,
        "holding_days": 3.5, "min_holding_days": 7.0, "fee_hurdle_target_usd": 3.0,
        "fee_amortized_pct": 118.4, "liquidation_price": 91000.0, "distance_to_liq_pct": 24.5,
        "timestamp": _G5_FIX_NOW.isoformat(),
    }
    html, _fake = _g5_render_carry_panel(None, payload=payload)
    assert "not measured" not in html, html[:2500]
    assert "$2,500 USDT" in html, html
    assert "+11.25%" in html and "(Hurdle: 8.0%)" in html, html
    assert "+0.0120%" in html, html
    assert "$+1.8675 USD" in html, html
    assert "$-0.75" in html, "the published net delta is not printed"
    assert "+24.5% Buffer" in html and "(Liq: $91,000.00)" in html, html
    assert "ISOLATED" in html, html
    assert "🟢 ACTIVE 1:1 DELTA-NEUTRAL TWIN" in html, html
    assert "3.5d / 7d" in html, html


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


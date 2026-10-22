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

    `path` defaults to dashboard.py; the legacy scripts/live_dashboard.py carries its own copies of
    the same pure renderers (it is a standalone Streamlit script and cannot import dashboard.py),
    and they are extracted the same way.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    target = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
    assert target is not None, f"{path.name} has no module-level function named {name}"

    loaded = {n.id for n in ast.walk(target) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    ns = {}
    for node in [target] + [
        n for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in loaded for t in n.targets)
    ]:
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
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

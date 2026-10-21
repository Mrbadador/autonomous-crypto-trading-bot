"""Monte Carlo validation must report only what it measured from the recorded trade ledger."""

import json
import pathlib
import sqlite3

import pytest

import quant_validation
from quant_validation import (
    InsufficientValidationData,
    generate_full_validation_report,
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
    assert res["empirical_win_rate_pct"] == pytest.approx(66.7, abs=0.05)
    assert res["initial_equity_usdt"] == pytest.approx(545.24)


def test_roe_pct_is_the_return_series_not_pnl_sign(tmp_path, monkeypatch):
    """Every row is seeded so pnl and roe_pct disagree on sign; roe_pct must win.

    Seeded 5 roe-wins / 7 roe-losses rather than 4/8 because 4 wins against a 2.0 payoff is a
    zero-edge ledger, which the engine now refuses to simulate (see the `zero_edge` case of
    test_book_without_an_edge_raises_instead_of_reporting_a_positive_kelly).
    """
    monkeypatch.chdir(tmp_path)
    rows = [(i, 1.0, 100.0, 1.0, -5.0) for i in range(1, 8)]
    rows += [(i, -1.0, 100.0, 1.0, 10.0) for i in range(8, 13)]
    _seed_ledger("trading_data.db", rows)
    res = run_trade_sequence_monte_carlo(n_simulations=200, horizon_trades=12)
    assert res["historical_sample_size"] == 12
    assert res["empirical_win_rate_pct"] == pytest.approx(41.7, abs=0.05)
    assert res["empirical_payoff_ratio"] == pytest.approx(2.0, abs=0.01)


def test_source_contains_no_synthetic_return_fallback():
    src = pathlib.Path("quant_validation.py").read_text(encoding="utf-8")
    assert "0.036, -0.024, 0.042" not in src
    assert "if len(trade_pct_returns) < 10:" not in src


def test_source_contains_no_payoff_default_literals():
    """Payoff ratios must be measured, never defaulted (audit finding 2)."""
    src = pathlib.Path("quant_validation.py").read_text(encoding="utf-8")
    assert "else 0.035" not in src
    assert "else 0.020" not in src


def _mixed_ledger_rows(bad_roe=None):
    """11 measurable trades; optionally one extra row whose roe_pct is unparseable."""
    rows = [(i, 1.0, 100.0, 1.0, 10.0) for i in range(1, 9)]
    rows += [(i, -1.0, 100.0, 1.0, -5.0) for i in range(9, 12)]
    if bad_roe is not None:
        rows.append((12, 1.0, 100.0, 1.0, bad_roe))
    return rows


@pytest.mark.parametrize(
    ("roe_pct", "absent_kind"),
    [(2.5, "losing"), (-2.5, "winning")],
    ids=["all_green", "all_red"],
)
def test_one_sided_ledger_refuses_a_defaulted_payoff_ratio(tmp_path, monkeypatch, roe_pct, absent_kind):
    """A 10-trade all-green (or all-red) ledger has no payoff to measure; no literal may fill in."""
    monkeypatch.chdir(tmp_path)
    rows = [(i, 10.0 * (1.0 if roe_pct > 0 else -1.0), 100.0, 1.0, roe_pct) for i in range(1, 11)]
    _seed_ledger("trading_data.db", rows)
    with pytest.raises(InsufficientValidationData) as excinfo:
        run_trade_sequence_monte_carlo(n_simulations=50, horizon_trades=10)
    assert absent_kind in str(excinfo.value)


def test_unparseable_ledger_row_raises_insufficient_data_not_value_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed_ledger("trading_data.db", _mixed_ledger_rows(bad_roe="not_a_number"))
    with pytest.raises(InsufficientValidationData) as excinfo:
        run_trade_sequence_monte_carlo(n_simulations=50, horizon_trades=10)
    assert isinstance(excinfo.value.__cause__, (ValueError, TypeError))
    assert "not_a_number" in str(excinfo.value)


def _edge_ledger_rows(win_roe, n_wins, loss_roe, n_losses):
    """A ledger with both wins and losses so a payoff ratio and a Kelly fraction can be measured."""
    rows = [(i, 10.0, 100.0, 1.0, win_roe) for i in range(1, n_wins + 1)]
    rows += [
        (i, -10.0, 100.0, 1.0, loss_roe)
        for i in range(n_wins + 1, n_wins + n_losses + 1)
    ]
    return rows


def test_quarter_kelly_is_the_measured_value_not_the_8pct_floor(tmp_path, monkeypatch):
    """9 wins at +4.0% and 11 losses at -2.5% measure a 1/4 Kelly of 2.66%; the floor reported 8.0%."""
    monkeypatch.chdir(tmp_path)
    _seed_ledger("trading_data.db", _edge_ledger_rows(4.0, 9, -2.5, 11))

    res = run_trade_sequence_monte_carlo(n_simulations=200, horizon_trades=12)

    assert res["quarter_kelly_fraction_pct"] == pytest.approx(2.66, abs=0.01)
    assert res["quarter_kelly_fraction_pct"] != 8.0


def test_quarter_kelly_stays_capped_at_25pct_on_a_large_edge(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed_ledger("trading_data.db", _edge_ledger_rows(40.0, 19, -2.0, 1))

    res = run_trade_sequence_monte_carlo(n_simulations=200, horizon_trades=12)

    assert res["quarter_kelly_fraction_pct"] == pytest.approx(23.69, abs=0.01)
    assert res["quarter_kelly_fraction_pct"] <= 25.0


@pytest.mark.parametrize(
    ("win_roe", "n_wins", "loss_roe", "n_losses", "measured_kelly"),
    [
        (3.0, 8, -3.0, 12, "-0.2000"),
        (10.0, 4, -5.0, 8, "0.0000"),
    ],
    ids=["negative_edge", "zero_edge"],
)
def test_book_without_an_edge_raises_instead_of_reporting_a_positive_kelly(
    tmp_path, monkeypatch, win_roe, n_wins, loss_roe, n_losses, measured_kelly
):
    """A non-positive measured Kelly must raise naming that measurement, never return a sizing number."""
    monkeypatch.chdir(tmp_path)
    _seed_ledger("trading_data.db", _edge_ledger_rows(win_roe, n_wins, loss_roe, n_losses))

    with pytest.raises(InsufficientValidationData) as excinfo:
        run_trade_sequence_monte_carlo(n_simulations=200, horizon_trades=12)

    assert measured_kelly in str(excinfo.value)
    assert "no positive edge" in str(excinfo.value)


def test_source_contains_no_quarter_kelly_floor_or_default():
    src = pathlib.Path("quant_validation.py").read_text(encoding="utf-8")
    assert "else 0.14" not in src
    assert "max(0.08," not in src
    mc_body = src[src.index("def run_trade_sequence_monte_carlo"):]
    assert mc_body.index("full_kelly <= 0") < mc_body.index("rng.choice")


def _stub_the_other_validation_arms(monkeypatch):
    """Keeps report tests fast. The Monte Carlo arm is never stubbed — it is the code under test."""
    monkeypatch.setattr(
        quant_validation,
        "run_bootstrap_entry_significance",
        lambda *args, **kwargs: {"SCALP_SHORT": {"stub_arm": "bootstrap"}},
    )
    monkeypatch.setattr(
        quant_validation,
        "compute_finrl_rolling_sharpe_selector",
        lambda bootstrap_res, *args, **kwargs: {"ensemble_winner": "STUB", "strategy_weights": {}},
    )


def _assert_the_other_arms_survived(report):
    assert report["bootstrap_significance"] == {"SCALP_SHORT": {"stub_arm": "bootstrap"}}
    assert report["finrl_ensemble_selector"] == {"ensemble_winner": "STUB", "strategy_weights": {}}


def test_report_states_monte_carlo_unavailable_and_keeps_the_other_arms(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # empty directory: no trading_data.db to measure
    _stub_the_other_validation_arms(monkeypatch)

    report = generate_full_validation_report(force_refresh=True)

    mc = report["monte_carlo"]
    assert mc["status"] == "unavailable"
    assert mc["data_source"] == "none"
    assert isinstance(mc["reason"], str) and mc["reason"].strip()
    assert "n_simulations" not in mc and "equity_cone" not in mc
    _assert_the_other_arms_survived(report)

    cached = json.loads(
        pathlib.Path("cache", "quant_validation_report.json").read_text(encoding="utf-8")
    )
    assert cached["monte_carlo"]["status"] == "unavailable"


def test_report_states_monte_carlo_unavailable_for_an_unparseable_row(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed_ledger("trading_data.db", _mixed_ledger_rows(bad_roe="not_a_number"))
    _stub_the_other_validation_arms(monkeypatch)

    report = generate_full_validation_report(force_refresh=True)

    mc = report["monte_carlo"]
    assert mc["status"] == "unavailable"
    assert "row" in mc["reason"].lower()
    assert "not_a_number" in mc["reason"]
    assert "n_simulations" not in mc
    _assert_the_other_arms_survived(report)


def test_report_states_monte_carlo_ok_on_a_recorded_ledger(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed_ledger("trading_data.db", _mixed_ledger_rows())
    _stub_the_other_validation_arms(monkeypatch)

    report = generate_full_validation_report(force_refresh=True)

    mc = report["monte_carlo"]
    assert mc["status"] == "ok"
    assert mc["data_source"] == "swing_trades"
    assert mc["historical_sample_size"] == 11
    assert "reason" not in mc
    _assert_the_other_arms_survived(report)


def test_dashboard_gates_on_the_same_unavailable_sentinel():
    """dashboard.py renders its explicit warning only for this exact sentinel string."""
    src = pathlib.Path("dashboard.py").read_text(encoding="utf-8")
    assert 'mc_data.get("status") == "unavailable"' in src

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
    """Every row is seeded so pnl and roe_pct disagree on sign; roe_pct must win."""
    monkeypatch.chdir(tmp_path)
    rows = [(i, 1.0, 100.0, 1.0, -5.0) for i in range(1, 9)]
    rows += [(i, -1.0, 100.0, 1.0, 10.0) for i in range(9, 13)]
    _seed_ledger("trading_data.db", rows)
    res = run_trade_sequence_monte_carlo(n_simulations=200, horizon_trades=12)
    assert res["historical_sample_size"] == 12
    assert res["empirical_win_rate_pct"] == pytest.approx(33.3, abs=0.05)
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

"""Monte Carlo validation must report only what it measured from the recorded trade ledger."""

import sqlite3

import pytest

from quant_validation import (
    InsufficientValidationData,
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
    import pathlib
    src = pathlib.Path("quant_validation.py").read_text(encoding="utf-8")
    assert "0.036, -0.024, 0.042" not in src
    assert "if len(trade_pct_returns) < 10:" not in src

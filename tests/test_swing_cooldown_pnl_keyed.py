import sqlite3
from datetime import UTC, datetime, timedelta

import pytest


@pytest.mark.parametrize(
    "reason, pnl, should_cooldown",
    [
        ("STOP_LOSS", -1.25, True),
        ("AI_STRUCTURAL_INVALIDATION", -3.0, True),
        ("SCALP_SHORT_TIME_EXPIRY", -0.5, True),
        ("STOP_LOSS", 0.75, False),
        ("sl_hit", 0.10, False),
        ("breakeven_hit", 0.0, False),
        ("TAKE_PROFIT", 2.5, False),
        ("STOP_LOSS", None, True),
        ("TAKE_PROFIT", None, True),
    ],
)
def test_cooldown_predicate_is_pnl_keyed(reason, pnl, should_cooldown):
    from swing_trader import _cooldown_should_trigger

    assert _cooldown_should_trigger(reason, pnl) is should_cooldown


def _seed_last_closed_trade(pnl, exit_reason="STOP_LOSS", side="long", symbol="BTC/USDT"):
    """One closed trade in a real trading_data.db, exited 10 minutes ago (inside the 4h window)."""
    conn = sqlite3.connect("trading_data.db")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS swing_trades (
               id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
               symbol TEXT NOT NULL, side TEXT NOT NULL, pnl REAL, exit_reason TEXT)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS swing_cooldowns (
               symbol TEXT PRIMARY KEY, cooldown_until TEXT NOT NULL,
               reason TEXT NOT NULL, created_at TEXT NOT NULL)"""
    )
    conn.execute(
        "INSERT INTO swing_trades (timestamp, symbol, side, pnl, exit_reason) VALUES (?,?,?,?,?)",
        ((datetime.now(UTC) - timedelta(minutes=10)).isoformat(), symbol, side, pnl, exit_reason),
    )
    conn.commit()
    conn.close()


@pytest.mark.parametrize(
    ("pnl", "cooled_down"),
    [(None, True), (0.0, False)],
    ids=["unknown_pnl", "realised_zero_pnl"],
)
def test_ledger_pnl_column_reaches_the_predicate_as_unknown_not_zero(tmp_path, monkeypatch, pnl, cooled_down):
    """A NULL pnl column must arrive as None (cool down); a realised 0.0 must stay flat (no cooldown)."""
    from swing_trader import SwingArchitectTrader

    monkeypatch.chdir(tmp_path)
    _seed_last_closed_trade(pnl)

    trader = SwingArchitectTrader.__new__(SwingArchitectTrader)
    cooldown_until = trader._get_cooldown_sync("BTC/USDT", "LONG")

    assert (cooldown_until is not None) is cooled_down

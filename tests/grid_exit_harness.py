"""
Shared scaffolding for exit-engine tests (swing_trader.manage_active_positions).

Every position in this fleet is mirrored into `swing_open_positions` and managed by
SwingArchitectTrader, so exit tests need: a quarantined sqlite file holding the tables
the loop touches, a mocked UTA client with price/plan coverage, and a position fixture
whose `initial_risk_usd` matches the geometry the engine expects.
"""

import gc
import os
import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import tests.conftest  # noqa: F401 — enforce D-11 quarantine

_orig_sqlite_connect = sqlite3.connect

_LONG_NOTES = "Chop Maker Grid [Post-Only Limit] (TP: +1.4%)"
_SHORT_NOTES = "Quick Scalp Short [Post-Only Limit]"


def grid_pos(symbol="NEAR/USDT", entry=100.0, sl=98.85, tp=101.35, size=10.0,
             peak=100.75, opened_at=None, harvested=False, engine="GRID_MAKER",
             side="long", lev=3.0, risk_usd=None, status="ACTIVE"):
    now_iso = (opened_at or datetime.now(UTC)).isoformat()
    return {
        "symbol": symbol, "side": side, "entry_price": entry, "size": size,
        "stop_loss": sl, "take_profit": tp, "trailing_stop": sl, "peak_price": peak,
        "leverage": lev, "opened_at": now_iso, "updated_at": now_iso,
        "status": status, "r_multiple": 0.0,
        "initial_risk_usd": risk_usd if risk_usd is not None else abs(entry - 98.85) * size,
        "engine_type": engine,
        "notes": _LONG_NOTES if side == "long" else _SHORT_NOTES,
        "grid_partial_tp_harvested": 1 if harvested else 0,
    }


def wire_client(trader, ticker_price, ohlcv_close, plan_families=("loss", "profit"),
               coverage_error=None, close_result=None):
    """Mock the UTA client: live ticker, 4H candles, strategy-plan coverage and order calls."""
    trader._load_active_positions_sync = MagicMock()
    trader.uta_client = MagicMock()
    trader.uta_client._normalize_symbol = lambda s: s.replace("/", "")
    trader.uta_client.exchange = MagicMock()
    trader.uta_client.exchange.fetch_ticker = AsyncMock(
        return_value={"last": ticker_price, "close": ticker_price})
    trader.uta_client.exchange.fetch_ohlcv = AsyncMock(
        return_value=[[1000, 100.0, 100.5, 99.0, ohlcv_close, 100.0] for _ in range(50)])
    trader.uta_client.exchange.fetch_positions = AsyncMock(return_value=[])
    trader.uta_client.modify_strategy_order = AsyncMock(return_value=True)
    trader.uta_client.close_position_order = AsyncMock(
        return_value=close_result or {"id": "partial_fill_1", "qty": 5.0})
    if coverage_error is not None:
        trader.uta_client.get_resting_plan_families = AsyncMock(side_effect=coverage_error)
    else:
        trader.uta_client.get_resting_plan_families = AsyncMock(return_value=set(plan_families))


def aged(minutes: int = 90):
    """Positions younger than 60 minutes are muted by the Rule #173/175 breathing-room gate."""
    return datetime.now(UTC) - timedelta(minutes=minutes)


class SwingExitHarnessCase(unittest.TestCase):
    """Provides a quarantined trading_data.db with the tables the exit loop reads/writes."""

    @classmethod
    def setUpClass(cls):
        cls._prev_cwd = os.getcwd()
        cls.temp_dir = tempfile.TemporaryDirectory()
        os.chdir(cls.temp_dir.name)
        cls.db_path = os.path.join(cls.temp_dir.name, "trading_data.db")
        conn = _orig_sqlite_connect(cls.db_path)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS swing_open_positions ("
                "symbol TEXT PRIMARY KEY, side TEXT, entry_price REAL, size REAL, "
                "stop_loss REAL, take_profit REAL, trailing_stop REAL, peak_price REAL, "
                "leverage REAL, opened_at TEXT, updated_at TEXT, status TEXT, "
                "r_multiple REAL, initial_risk_usd REAL, notes TEXT, engine_type TEXT, "
                "grid_partial_tp_harvested INTEGER DEFAULT 0)")
            conn.execute("CREATE TABLE IF NOT EXISTS system_state (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
            conn.execute("CREATE TABLE IF NOT EXISTS asset_locks (symbol TEXT PRIMARY KEY, locked_by TEXT, heartbeat TEXT)")
            conn.commit()
        finally:
            conn.close()

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._prev_cwd)
        gc.collect()
        try:
            cls.temp_dir.cleanup()
        except Exception:
            pass

    def setUp(self):
        self.db_conn_ctx = patch("swing_trader.sqlite3.connect",
                                 side_effect=lambda *a, **kw: _orig_sqlite_connect(self.db_path))
        self.db_conn_ctx.start()
        self.addCleanup(self.db_conn_ctx.stop)

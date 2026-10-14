"""
tests/test_reconciler_ledger_guard.py

Functional regression tests for the P2 ledger-loss guard in
reconciliation_engine.audit_live_exchange_positions().

Invariant under test: a local open-position row may only be DELETEd once the closed-trade
ledger row is confirmed written. If the ledger INSERT raises, the open row must survive so the
next pass retries it instead of losing the trade.

Safety: throwaway temp DB + fake BitgetUTAClient injected via sys.modules.
No live DB, no network, no orders, no email. Table DDL mirrors trading_data.db as of
2026-10-13 (captured from sqlite_master).

TASK 6 fix coverage: the finalize site used to issue a generic INSERT (type/qty/balance/action)
that production swing_trades does not have, so every swing row raised and deferred forever. It
now routes swing_trades through the swing column set, and its recent_close dedup probe is
windowed to 15 minutes instead of matching ANY historical row for the symbol.
"""

import asyncio
import os
import sqlite3
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import reconciliation_engine as rec  # noqa: E402

SNIPER = "TESTA/USDT"
SWING = "TESTB/USDT"

# --- production DDL snapshot (trading_data.db, 2026-10-13) ---
DDL = [
    """CREATE TABLE "system_state" (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""",
    """CREATE TABLE swing_cooldowns (
                        symbol TEXT PRIMARY KEY,
                        cooldown_until TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )""",
    """CREATE TABLE open_positions (
                    symbol TEXT PRIMARY KEY, side TEXT, entry_price REAL, qty REAL,
                    take_profit REAL, stop_loss REAL, peak_price REAL, entry_time TEXT
                , is_closing INTEGER DEFAULT 0, partial_tp_hit INTEGER DEFAULT 0, action_confidence REAL, comments TEXT, leverage REAL, hard_sl REAL DEFAULT 0.0, sl_breach_time REAL DEFAULT 0.0)""",
    """CREATE TABLE berserker_open_positions (
                    symbol TEXT PRIMARY KEY, side TEXT, entry_price REAL, qty REAL,
                    take_profit REAL, stop_loss REAL, peak_price REAL, entry_time TEXT
                , is_closing INTEGER DEFAULT 0, partial_tp_hit INTEGER DEFAULT 0, action_confidence REAL, comments TEXT, leverage REAL, hard_sl REAL DEFAULT 0.0, sl_breach_time REAL DEFAULT 0.0)""",
    """CREATE TABLE marl_live_open_positions (
            symbol TEXT PRIMARY KEY,
            side TEXT,
            entry_price REAL,
            qty REAL,
            take_profit REAL,
            stop_loss REAL,
            peak_price REAL,
            entry_time TEXT,
            is_closing INTEGER DEFAULT 0,
            partial_tp_hit INTEGER DEFAULT 0,
            action_confidence REAL,
            comments TEXT,
            leverage REAL
        , hard_sl REAL DEFAULT 0.0, sl_breach_time REAL DEFAULT 0.0)""",
    """CREATE TABLE trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pos_id INTEGER,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    type TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL,
                    exit_price REAL,
                    qty REAL,
                    pnl REAL,
                    balance REAL,
                    regime TEXT,
                    phase TEXT,
                    quality REAL,
                    sl_mode TEXT,
                    tp_mode TEXT,
                    trail_mode TEXT,
                    mfe REAL,
                    notes TEXT
                , mae REAL, exposure_time_mins REAL, efficiency_score REAL, fear_index INTEGER, news_sentiment REAL, status TEXT DEFAULT 'closed', ai_rationale TEXT, slippage_bps REAL DEFAULT 0.0, action TEXT)""",
    """CREATE TABLE berserker_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, symbol TEXT, side TEXT, type TEXT,
                    entry_price REAL, exit_price REAL, qty REAL, pnl REAL,
                    balance REAL, regime TEXT, phase TEXT, quality REAL, notes TEXT
                , status TEXT DEFAULT 'closed', ai_rationale TEXT, slippage_bps REAL DEFAULT 0.0, mfe REAL, mae REAL, exposure_time_mins REAL, efficiency_score REAL, fear_index INTEGER, news_sentiment REAL, action TEXT)""",
    """CREATE TABLE marl_live_trades (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT, symbol TEXT, side TEXT, type TEXT,
                            entry_price REAL, exit_price REAL, qty REAL, pnl REAL,
                            balance REAL, regime TEXT, phase TEXT, quality REAL, notes TEXT,
                            status TEXT DEFAULT 'closed',
                            ai_rationale TEXT,
                            slippage_bps REAL DEFAULT 0.0,
                            action TEXT
                        )""",
    # swing_open_positions / swing_trades: production has NO type/qty/balance/action columns.
    """CREATE TABLE swing_open_positions (
                    symbol TEXT, side TEXT, entry_price REAL, size REAL, stop_loss REAL,
                    take_profit REAL, trailing_stop REAL, peak_price REAL, leverage REAL,
                    opened_at TEXT, updated_at TEXT, status TEXT, r_multiple REAL,
                    initial_risk_usd REAL, notes TEXT, intended_entry_price REAL,
                    actual_fill_price REAL, slippage_bps REAL, fee_entry_usd REAL,
                    execution_type TEXT, gates_snapshot TEXT
                )""",
    """CREATE TABLE swing_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, symbol TEXT,
                    side TEXT, entry_price REAL, exit_price REAL, size REAL, pnl REAL,
                    roe_pct REAL, holding_hours REAL, exit_reason TEXT, notes TEXT,
                    intended_entry_price REAL, actual_fill_price REAL, slippage_bps REAL,
                    fee_paid_usd REAL, fee_maker_usd REAL, fee_taker_usd REAL,
                    funding_usd REAL, funding_source TEXT, gates_snapshot TEXT, status TEXT
                )""",
]

OPEN_INSERTS = [
    ("INSERT INTO open_positions (symbol, side, entry_price, qty, take_profit, stop_loss, entry_time, leverage) "
     "VALUES (?, 'long', 100.0, 1.0, 104.0, 99.0, datetime('now'), 3.0)", (SNIPER,)),
    ("INSERT INTO swing_open_positions (symbol, side, entry_price, size, stop_loss, take_profit, leverage, opened_at, status) "
     "VALUES (?, 'long', 100.0, 1.0, 99.0, 104.0, 3.0, datetime('now'), 'OPEN')", (SWING,)),
]


class _FakeExchange:
    def __getattr__(self, name):
        async def _any(*args, **kwargs):
            return {"data": {}}
        return _any


class _FakeUTA:
    def __init__(self, *args, **kwargs):
        self.exchange = _FakeExchange()

    async def fetch_balance(self, *args, **kwargs):
        return 340.0

    async def fetch_positions(self, *args, **kwargs):
        return []          # exchange reports nothing -> every local row looks closed on-exchange

    async def get_recent_fills(self, *args, **kwargs):
        return []

    async def cancel_order(self, *args, **kwargs):
        return {}

    async def close_position_order(self, *args, **kwargs):
        return {}

    async def close(self, *args, **kwargs):
        return None


class ReconcilerLedgerGuardTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="recon_guard_")
        self.db_path = os.path.join(self.tmpdir, "trading_data.db")

        self._orig = {
            "DATA_DB": rec.DATA_DB,
            "LEDGER_DB": rec.LEDGER_DB,
            "MODE": getattr(config, "MODE", None),
            "PAPER_TRADE": getattr(config, "PAPER_TRADE", None),
            "EXCHANGE_ID": getattr(config, "EXCHANGE_ID", None),
            "fetch_live_price": rec.fetch_live_price,
        }
        rec.DATA_DB = self.db_path
        rec.LEDGER_DB = os.path.join(self.tmpdir, "trading_ledger.db")
        config.MODE = "live"
        config.PAPER_TRADE = False
        config.EXCHANGE_ID = "bitget"

        # hermetic: no REST, no SMTP
        rec.fetch_live_price = lambda symbol: None
        self._orig_modules = {}
        for name in ("bitget_uta_client", "email_dispatcher"):
            self._orig_modules[name] = sys.modules.get(name)
            sys.modules[name] = types.ModuleType(name)
        sys.modules["bitget_uta_client"].BitgetUTAClient = _FakeUTA
        sys.modules["email_dispatcher"].send_trade_log_async = lambda *a, **k: asyncio.sleep(0)
        sys.modules["email_dispatcher"].send_trade_log = lambda *a, **k: None

        self._build_db()

    def tearDown(self):
        for name, mod in self._orig_modules.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
        rec.DATA_DB = self._orig["DATA_DB"]
        rec.LEDGER_DB = self._orig["LEDGER_DB"]
        config.MODE = self._orig["MODE"]
        config.PAPER_TRADE = self._orig["PAPER_TRADE"]
        config.EXCHANGE_ID = self._orig["EXCHANGE_ID"]
        rec.fetch_live_price = self._orig["fetch_live_price"]

    def _build_db(self):
        with sqlite3.connect(self.db_path) as conn:
            for ddl in DDL:
                conn.execute(ddl)
            for sql, params in OPEN_INSERTS:
                conn.execute(sql, params)
            conn.commit()

    def _table_counts(self):
        with sqlite3.connect(self.db_path) as conn:
            out = {}
            for t in ("open_positions", "swing_open_positions", "trades", "swing_trades"):
                try:
                    out[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                except sqlite3.Error:
                    out[t] = -1
            return out

    async def test_confirmed_ledger_write_allows_prune(self):
        """Sniper row: the generic ledger INSERT is schema-compatible, so write + prune both happen."""
        await rec.audit_live_exchange_positions()
        counts = self._table_counts()
        self.assertEqual(counts["trades"], 1, "expected one reconciled row in the trades ledger")
        self.assertEqual(counts["open_positions"], 0, "open row must be pruned once its ledger row is confirmed")

    async def test_failed_ledger_write_retains_open_row(self):
        """Core P2 guard: with the ledger table gone the INSERT raises, so the open row survives."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DROP TABLE trades")
            conn.commit()

        await rec.audit_live_exchange_positions()

        counts = self._table_counts()
        self.assertEqual(counts["open_positions"], 1, "P2 guard: open row must survive a failed ledger INSERT")

    async def test_swing_finalize_writes_ledger_and_prunes(self):
        """TASK 6: swing rows must be recorded through the swing_trades column set, not deferred.

        Inverted from the previous characterization test, which asserted this INSERT always
        raised against production DDL.
        """
        await rec.audit_live_exchange_positions()

        counts = self._table_counts()
        self.assertEqual(counts["swing_trades"], 1, "finalize site must write a swing_trades ledger row")
        self.assertEqual(counts["swing_open_positions"], 0, "swing open row prunes only after that row is confirmed")

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT side, size, roe_pct, holding_hours, exit_reason, fee_paid_usd, funding_usd, funding_source, status "
                "FROM swing_trades"
            ).fetchone()
        side, size, roe, hours, reason, fee, funding, source, status = row
        self.assertEqual(side, "long")
        self.assertAlmostEqual(size, 1.0)
        self.assertIsNotNone(roe)
        self.assertGreater(hours, 0.0)
        self.assertIn(reason, ("tp_hit", "breakeven_hit", "sl_hit"))
        self.assertGreaterEqual(fee, 0.0)
        self.assertIsNotNone(funding)
        self.assertEqual(source, "reconciler")
        self.assertEqual(status, "CLOSED")

    async def test_stale_ledger_history_does_not_suppress_new_ledger_row(self):
        """TASK 6: the old recent_close probe matched ANY historical row, letting a months-old
        trade mark ledger_write_ok=True and prune the current position without recording it."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO swing_trades (timestamp, symbol, side, entry_price, exit_price, size, pnl, exit_reason, status) "
                "VALUES (datetime('now', '-40 days'), ?, 'long', 100.0, 97.0, 1.0, -3.0, 'sl_hit', 'CLOSED')",
                (SWING,),
            )
            conn.commit()

        await rec.audit_live_exchange_positions()

        counts = self._table_counts()
        self.assertEqual(counts["swing_trades"], 2, "a 40-day-old ledger row must not suppress today's write")
        self.assertEqual(counts["swing_open_positions"], 0)

    async def test_recent_engine_close_suppresses_duplicate_ledger_row(self):
        """The windowed probe must still dedup a close the engine already recorded (< 15 min ago)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO swing_trades (timestamp, symbol, side, entry_price, exit_price, size, pnl, exit_reason, status) "
                "VALUES (datetime('now', '-3 minutes'), ?, 'long', 100.0, 97.0, 1.0, -3.0, 'sl_hit', 'CLOSED')",
                (SWING,),
            )
            conn.commit()

        await rec.audit_live_exchange_positions()

        counts = self._table_counts()
        self.assertEqual(counts["swing_trades"], 1, "engine-recorded close must not be doubled")
        self.assertEqual(counts["swing_open_positions"], 0, "position still resolves as closed; prune is allowed")


if __name__ == "__main__":
    unittest.main()

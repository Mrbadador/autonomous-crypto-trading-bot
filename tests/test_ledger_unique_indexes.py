import os
import shutil
import sqlite3
import tempfile
import unittest
from contextlib import closing

import db_manager


class TestLedgerUniqueIndexes(unittest.TestCase):
    """The unique ledger indexes must be created by schema code, not by a
    one-off SQL command against a database file."""

    def setUp(self):
        # Best-effort cleanup: the write queue keeps a handle on the database
        # file, so Windows refuses to remove it while the process is alive.
        self._tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self._tmp_dir, "ledger.db")

    def tearDown(self):
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def _indexes(self, table):
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?", (table,)
            ).fetchall()
        return {r[0] for r in rows}

    def test_trades_unique_index_is_created_by_schema_code(self):
        db_manager.DatabaseManager(db_path=self.db_path)
        self.assertIn("idx_trades_unique", self._indexes("trades"))

    def test_swing_trades_unique_index_is_created_by_schema_code(self):
        db_manager.DatabaseManager(db_path=self.db_path)
        self.assertIn("idx_swing_trades_unique", self._indexes("swing_trades"))

    def test_duplicate_trade_insert_is_rejected(self):
        db_manager.DatabaseManager(db_path=self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO trades (timestamp, symbol, type, side, entry_price, qty)"
                " VALUES ('2026-11-04T00:00:00', 'BTC/USDT', 'grid', 'long', 100.0, 1.0)"
            )
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO trades (timestamp, symbol, type, side, entry_price, qty)"
                    " VALUES ('2026-11-04T00:00:00', 'BTC/USDT', 'grid', 'long', 100.0, 1.0)"
                )

    def test_duplicate_swing_trade_insert_is_rejected(self):
        db_manager.DatabaseManager(db_path=self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO swing_trades (timestamp, symbol, side, entry_price, size)"
                " VALUES ('2026-11-04T00:00:00', 'BTC/USDT', 'long', 100.0, 1.0)"
            )
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO swing_trades (timestamp, symbol, side, entry_price, size)"
                    " VALUES ('2026-11-04T00:00:00', 'BTC/USDT', 'long', 100.0, 1.0)"
                )

    def test_distinct_trades_on_the_same_symbol_are_allowed(self):
        """The uniqueness constraint must not reject genuinely different trades."""
        db_manager.DatabaseManager(db_path=self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO trades (timestamp, symbol, type, side, entry_price, qty)"
                " VALUES ('2026-11-04T00:00:00', 'BTC/USDT', 'grid', 'long', 100.0, 1.0)"
            )
            conn.execute(
                "INSERT INTO trades (timestamp, symbol, type, side, entry_price, qty)"
                " VALUES ('2026-11-04T00:00:01', 'BTC/USDT', 'grid', 'long', 100.0, 1.0)"
            )
            conn.execute(
                "INSERT INTO trades (timestamp, symbol, type, side, entry_price, qty)"
                " VALUES ('2026-11-04T00:00:00', 'BTC/USDT', 'grid', 'short', 100.0, 1.0)"
            )
            conn.execute(
                "INSERT INTO trades (timestamp, symbol, type, side, entry_price, qty)"
                " VALUES ('2026-11-04T00:00:00', 'BTC/USDT', 'grid', 'long', 100.0, 2.0)"
            )
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0], 4
            )

    def test_schema_init_tolerates_a_ledger_that_still_has_duplicates(self):
        """A ledger that predates the constraint cannot be indexed; startup must
        not fail, it must report the problem instead."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE trades (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       timestamp TEXT NOT NULL,
                       symbol TEXT NOT NULL,
                       type TEXT NOT NULL,
                       side TEXT NOT NULL,
                       entry_price REAL,
                       exit_price REAL,
                       qty REAL)"""
            )
            conn.execute(
                "INSERT INTO trades (timestamp, symbol, type, side, entry_price, qty)"
                " VALUES ('2026-11-04T00:00:00', 'BTC/USDT', 'grid', 'long', 100.0, 1.0)"
            )
            conn.execute(
                "INSERT INTO trades (timestamp, symbol, type, side, entry_price, qty)"
                " VALUES ('2026-11-04T00:00:00', 'BTC/USDT', 'grid', 'long', 100.0, 1.0)"
            )
            conn.commit()

        db_manager.DatabaseManager(db_path=self.db_path)

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT COUNT(*) FROM trades"
            ).fetchone()[0]
        self.assertEqual(rows, 2, "startup must not delete rows from the ledger")

    def test_write_queue_drops_a_duplicate_without_losing_the_other_ops(self):
        """A duplicate that the unique index rejects must not stall the writer.

        The duplicate is dropped (the row already exists), the other ops in the
        same batch are still written, and the flush must complete.
        """
        db = db_manager.DatabaseManager(db_path=self.db_path)
        with closing(sqlite3.connect(self.db_path, timeout=5.0)) as conn:
            conn.execute(
                "INSERT INTO trades (timestamp, symbol, type, side, entry_price, qty)"
                " VALUES ('2026-11-04T00:00:00', 'BTC/USDT', 'grid', 'long', 100.0, 1.0)"
            )
            conn.commit()

        db.write_queue.enqueue_write(
            "trades",
            "INSERT INTO trades (timestamp, symbol, type, side, entry_price, qty)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            ("2026-11-04T00:00:00", "BTC/USDT", "grid", "long", 100.0, 1.0),
        )
        db.write_queue.enqueue_write(
            "trades",
            "INSERT INTO trades (timestamp, symbol, type, side, entry_price, qty)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            ("2026-11-04T00:00:01", "ETH/USDT", "grid", "long", 200.0, 1.0),
        )

        self.assertTrue(
            db.write_queue.flush_sync(timeout=10.0),
            "flush must not stall because one op was rejected as a duplicate",
        )

        with closing(sqlite3.connect(self.db_path, timeout=5.0)) as conn:
            rows = conn.execute(
                "SELECT symbol, COUNT(*) FROM trades GROUP BY symbol ORDER BY symbol"
            ).fetchall()
        self.assertEqual(rows, [("BTC/USDT", 1), ("ETH/USDT", 1)])
        self.assertTrue(db.write_queue._worker_thread.is_alive())

import contextlib
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from super_bot import acquire_asset_lock


class TestAcquireAssetLock(unittest.TestCase):
    """Regression: acquiring a symbol lock must never raise, and must not let a
    self-owned orphan lock block a legitimate entry."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmp.name, "locks.db")
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS asset_locks (
                       symbol TEXT PRIMARY KEY,
                       locked_by TEXT NOT NULL,
                       locked_at TEXT NOT NULL,
                       reason TEXT,
                       heartbeat TEXT NOT NULL)"""
            )
            conn.commit()
        self.now = datetime.now(timezone.utc).isoformat()
        self.old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()

    def tearDown(self):
        self._tmp.cleanup()

    def _seed(self, symbol, owner, heartbeat):
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "INSERT INTO asset_locks (symbol, locked_by, locked_at, reason, heartbeat)"
                " VALUES (?, ?, ?, ?, ?)",
                (symbol, owner, heartbeat, "seed", heartbeat),
            )
            conn.commit()

    def test_renews_self_owned_lock_without_raising(self):
        # This is the exact state that produced
        # "sqlite3.IntegrityError: UNIQUE constraint failed: asset_locks.symbol"
        self._seed("BTC/USDT", "super_bot_grid", self.now)
        self.assertTrue(
            acquire_asset_lock(self.db_path, "BTC/USDT", "super_bot_grid", "grid", self.now)
        )

    def test_takes_free_symbol(self):
        self.assertTrue(
            acquire_asset_lock(self.db_path, "ETH/USDT", "super_bot_grid", "grid", self.now)
        )

    def test_refuses_symbol_held_by_other_engine(self):
        self._seed("SOL/USDT", "swing_architect", self.now)
        self.assertFalse(
            acquire_asset_lock(self.db_path, "SOL/USDT", "super_bot_grid", "grid", self.now)
        )

    def test_takes_symbol_whose_other_lock_is_stale(self):
        self._seed("SOL/USDT", "swing_architect", self.old)
        self.assertTrue(
            acquire_asset_lock(self.db_path, "SOL/USDT", "super_bot_grid", "grid", self.now)
        )

    def test_lock_is_released_for_next_owner_after_release(self):
        self.assertTrue(
            acquire_asset_lock(self.db_path, "BTC/USDT", "super_bot_grid", "grid", self.now)
        )
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "DELETE FROM asset_locks WHERE symbol = ? AND locked_by = ?",
                ("BTC/USDT", "super_bot_grid"),
            )
            conn.commit()
        self.assertTrue(
            acquire_asset_lock(self.db_path, "BTC/USDT", "super_bot_scalp_short", "short", self.now)
        )

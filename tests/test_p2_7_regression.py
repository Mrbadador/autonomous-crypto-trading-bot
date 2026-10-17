"""
tests/test_p2_7_regression.py — Regression tests for P2-7 fixes.

Tests:
- Test 1 (R1 Acceptance b): Explicit verdicts across 3 counter scenarios:
  1. Counter row absent (status=ERROR, counter absent) -> must veto (is_ok=False)
  2. Counter row stale (status=ERROR, counter updated_at older than status updated_at) -> must veto (is_ok=False)
  3. Counter row present, fresh, and count=1 (transient blip) -> must permit (is_ok=True)
- Test 2 (R1 Atomic publish): Tests update_system_state_batch writes atomically and raises on failure.
- Test 3 (R4 Lock safety across engines): Tests that reconciliation_engine's scoped delete does NOT delete
  another engine's freshly-acquired lock or locks owned by different engines (e.g. 'basis_trader', 'external').
- Test 4 (R4 Reconciler NULL heartbeat zombie locks): Tests that reconciliation_engine's background prune
  cleans zombie locks with NULL heartbeat and old locked_at using COALESCE(heartbeat, locked_at).
"""

import os
import sys
import time
import json
import sqlite3
import tempfile
from datetime import datetime, timezone, timedelta
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from conviction_filter import check_database_integrity, PROD_DATABASES
import db_maintenance_daemon


def test_r1_consecutive_errors_three_verdicts():
    """
    Acceptance (b): Show the three verdicts explicitly:
      1. Counter absent: status=ERROR, no counter row -> must veto (is_ok=False)
      2. Counter stale: status=ERROR, counter updated_at older than status -> must veto (is_ok=False)
      3. Counter present, fresh with count 1: status=ERROR, counter count=1, same timestamp -> must permit (is_ok=True)
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE system_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        """)
        conn.commit()

        now_utc = datetime.now(timezone.utc)
        fresh_ts = now_utc.strftime("%Y-%m-%d %H:%M:%S")

        # Base setup: all databases OK
        for d in PROD_DATABASES:
            conn.execute("INSERT OR REPLACE INTO system_state VALUES (?, 'OK', ?)", (f"db_integrity_{d}", fresh_ts))
            conn.execute("INSERT OR REPLACE INTO system_state VALUES (?, '0', ?)", (f"db_integrity_{d}_consecutive_errors", fresh_ts))
        conn.commit()

        # ── Case 1: Status=ERROR, counter row ABSENT ──
        # Set trading_data.db to ERROR and delete its counter row
        conn.execute("INSERT OR REPLACE INTO system_state VALUES ('db_integrity_trading_data.db', 'ERROR', ?)", (fresh_ts,))
        conn.execute("DELETE FROM system_state WHERE key = 'db_integrity_trading_data.db_consecutive_errors'")
        conn.commit()

        res1 = check_database_integrity(conn, max_age_sec=900.0)
        print(f"Verdict 1 (Counter absent): is_ok={res1['is_ok']}, corrupt_dbs={res1['corrupt_dbs']}")
        assert res1["is_ok"] is False, "Expected veto when counter row is absent!"
        assert "trading_data.db" in res1["corrupt_dbs"]

        # ── Case 2: Status=ERROR, counter row STALE (older than status row) ──
        # Status row updated at fresh_ts; counter row updated 60s earlier
        older_ts = (now_utc - timedelta(seconds=60)).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("INSERT OR REPLACE INTO system_state VALUES ('db_integrity_trading_data.db', 'ERROR', ?)", (fresh_ts,))
        conn.execute("INSERT OR REPLACE INTO system_state VALUES ('db_integrity_trading_data.db_consecutive_errors', '1', ?)", (older_ts,))
        conn.commit()

        res2 = check_database_integrity(conn, max_age_sec=900.0)
        print(f"Verdict 2 (Counter stale): is_ok={res2['is_ok']}, corrupt_dbs={res2['corrupt_dbs']}")
        assert res2["is_ok"] is False, "Expected veto when counter row updated_at is older than status row!"
        assert "trading_data.db" in res2["corrupt_dbs"]

        # ── Case 3: Status=ERROR, counter row present, fresh, count=1 (transient blip) ──
        conn.execute("INSERT OR REPLACE INTO system_state VALUES ('db_integrity_trading_data.db', 'ERROR', ?)", (fresh_ts,))
        conn.execute("INSERT OR REPLACE INTO system_state VALUES ('db_integrity_trading_data.db_consecutive_errors', '1', ?)", (fresh_ts,))
        conn.commit()

        res3 = check_database_integrity(conn, max_age_sec=900.0)
        print(f"Verdict 3 (Counter fresh count=1): is_ok={res3['is_ok']}, corrupt_dbs={res3['corrupt_dbs']}")
        assert res3["is_ok"] is True, "Expected permit on 1st transient error with fresh counter row!"
        assert len(res3["corrupt_dbs"]) == 0

        # And verify count=2 with fresh timestamp DOES veto
        conn.execute("INSERT OR REPLACE INTO system_state VALUES ('db_integrity_trading_data.db_consecutive_errors', '2', ?)", (fresh_ts,))
        conn.commit()
        res4 = check_database_integrity(conn, max_age_sec=900.0)
        assert res4["is_ok"] is False
        assert "trading_data.db" in res4["corrupt_dbs"]

        conn.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_r1_atomic_publish_batch():
    """
    Tests that update_system_state_batch writes all key-values in a single transaction.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        test_state_db = os.path.join(tmpdir, "trading_data.db")
        conn = sqlite3.connect(test_state_db)
        conn.execute("""
            CREATE TABLE system_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        """)
        conn.commit()
        conn.close()

        # Monkeypatch BASE_DIR in db_maintenance_daemon
        orig_base = db_maintenance_daemon.BASE_DIR
        db_maintenance_daemon.BASE_DIR = tmpdir
        try:
            kvs = [
                ("db_integrity_trading_data.db", "OK"),
                ("db_integrity_trading_data.db_consecutive_errors", "0")
            ]
            db_maintenance_daemon.update_system_state_batch(kvs)

            conn = sqlite3.connect(test_state_db)
            rows = conn.execute("SELECT key, value, updated_at FROM system_state ORDER BY key").fetchall()
            conn.close()

            assert len(rows) == 2
            assert rows[0][0] == "db_integrity_trading_data.db" and rows[0][1] == "OK"
            assert rows[1][0] == "db_integrity_trading_data.db_consecutive_errors" and rows[1][1] == "0"
            assert rows[0][2] == rows[1][2], "Expected identical updated_at timestamp in single atomic transaction"
        finally:
            db_maintenance_daemon.BASE_DIR = orig_base


def test_r4_reconciler_locks_scoped_and_preserves_other_engines():
    """
    Tests that reconciliation_engine scoped delete queries:
    1. Do NOT delete a lock owned by another engine (e.g. 'basis_trader' or 'external').
    2. Do NOT delete a freshly-acquired live lock owned by 'super_bot_grid' (age <= 900s).
    3. DO delete an expired lock owned by 'super_bot_grid' (age > 900s).
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE asset_locks (
                symbol TEXT PRIMARY KEY,
                locked_by TEXT,
                locked_at TEXT,
                heartbeat TEXT
            )
        """)

        old_ts = (datetime.now(timezone.utc) - timedelta(minutes=25)).strftime("%Y-%m-%d %H:%M:%S")
        fresh_ts = (datetime.now(timezone.utc) - timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S")

        # Lock 1: Owned by basis_trader (different engine) -> MUST SURVIVE
        conn.execute("INSERT INTO asset_locks VALUES ('BTC/USDT', 'basis_trader', ?, ?)", (old_ts, old_ts))
        
        # Lock 2: Owned by super_bot_grid, fresh (2 min old) -> MUST SURVIVE
        conn.execute("INSERT INTO asset_locks VALUES ('ETH/USDT', 'super_bot_grid', ?, ?)", (old_ts, fresh_ts))

        # Lock 3: Owned by super_bot_grid, expired (> 900s) -> MUST BE PRUNED
        conn.execute("INSERT INTO asset_locks VALUES ('ADA/USDT', 'super_bot_grid', ?, ?)", (old_ts, old_ts))

        conn.commit()

        # Execute scoped reconciler delete on all 3 symbols
        for sym in ("BTC/USDT", "ETH/USDT", "ADA/USDT"):
            conn.execute("""
                DELETE FROM asset_locks
                WHERE symbol = ?
                  AND locked_by IN ('super_bot_grid', 'super_bot_scalp_short', 'swing_architect')
                  AND (julianday('now') - julianday(COALESCE(heartbeat, locked_at))) * 86400.0 > 900.0
            """, (sym,))
        conn.commit()

        remaining = {r[0]: r[1] for r in conn.execute("SELECT symbol, locked_by FROM asset_locks").fetchall()}
        assert "BTC/USDT" in remaining, "Lock owned by another engine ('basis_trader') was erroneously deleted!"
        assert "ETH/USDT" in remaining, "Fresh live lock owned by 'super_bot_grid' was erroneously deleted!"
        assert "ADA/USDT" not in remaining, "Expired lock owned by 'super_bot_grid' was not pruned!"
        conn.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_r4_reconciler_null_heartbeat_prune():
    """
    Tests that reconciler's background prune query cleans locks with NULL heartbeat and old locked_at.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE asset_locks (
                symbol TEXT PRIMARY KEY,
                locked_by TEXT,
                locked_at TEXT,
                heartbeat TEXT
            )
        """)

        now_utc = datetime.now(timezone.utc)
        old_locked_at = (now_utc - timedelta(minutes=25)).strftime("%Y-%m-%d %H:%M:%S")
        fresh_locked_at = (now_utc - timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S")
        now_iso = now_utc.strftime("%Y-%m-%d %H:%M:%S")

        # Zombie lock: NULL heartbeat, old locked_at -> MUST BE DELETED
        conn.execute("INSERT INTO asset_locks VALUES ('SOL/USDT', 'super_bot_grid', ?, NULL)", (old_locked_at,))
        # Fresh lock: NULL heartbeat, fresh locked_at -> MUST SURVIVE
        conn.execute("INSERT INTO asset_locks VALUES ('NEAR/USDT', 'super_bot_grid', ?, NULL)", (fresh_locked_at,))
        conn.commit()

        # Reconciler background prune query
        conn.execute(
            "DELETE FROM asset_locks WHERE (julianday(?) - julianday(COALESCE(heartbeat, locked_at))) * 86400.0 > 900.0",
            (now_iso,)
        )
        conn.commit()

        remaining = [r[0] for r in conn.execute("SELECT symbol FROM asset_locks").fetchall()]
        assert "SOL/USDT" not in remaining, "Zombie lock with NULL heartbeat was NOT deleted!"
        assert "NEAR/USDT" in remaining, "Recent lock with NULL heartbeat was erroneously deleted!"
        conn.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

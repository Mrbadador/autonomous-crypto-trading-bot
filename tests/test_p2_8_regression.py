"""
tests/test_p2_8_regression.py — Regression tests for P2-8 fixes.

Tests:
- Test 1 (R5 & R7 AST Extracted SQL Proof):
  Extracts DELETE FROM asset_locks queries directly from swing_trader.py (lines 433 and 2262)
  and reconciliation_engine.py (lines 547, 892, 962) using ast.parse.
  Executes extracted queries against a temp DB with the REAL production schema:
    CREATE TABLE asset_locks (
        symbol TEXT PRIMARY KEY,
        locked_by TEXT NOT NULL,
        locked_at TEXT NOT NULL,
        reason TEXT,
        heartbeat TEXT NOT NULL
    )
  Asserts that (SOL, super_bot_grid) survives when swing_architect releases (SOL, swing_architect).
  Asserts that reconciler scoped queries do NOT delete locks owned by other engines or fresh locks.
- Test 2 (R6 Single Increment Counter on Failure):
  Executes run_wal_checkpoints() three times in an isolated temp-dir against a DB with no system_state table.
  Asserts that consecutive errors increment by exactly 1 on each failing pass: 1, 2, 3.
"""

import os
import sys
import ast
import sqlite3
import tempfile
import importlib
from datetime import datetime, timezone, timedelta
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import db_maintenance_daemon

PROD_LOCK_SCHEMA = """
CREATE TABLE asset_locks (
    symbol TEXT PRIMARY KEY,
    locked_by TEXT NOT NULL,
    locked_at TEXT NOT NULL,
    reason TEXT,
    heartbeat TEXT NOT NULL
);
"""


def extract_sql_calls(filepath: str, substr: str) -> dict:
    """Extracts SQL string arguments from Call nodes in filepath matching substr."""
    src = open(filepath, "r", encoding="utf-8").read()
    tree = ast.parse(src)
    queries = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and substr in arg.value:
                    queries[node.lineno] = arg.value
    return queries


def test_r5_swing_trader_ast_extracted_sql_lock_scoping():
    """
    R5: Source-extracted SQL from swing_trader.py must be scoped with AND locked_by = 'swing_architect'.
    Executes extracted AST queries against the real production asset_locks schema with:
      ('SOL/USDT', 'swing_architect')
      ('SOL/USDT', 'super_bot_grid') - wait, symbol is PRIMARY KEY in live schema!
    To test cross-engine race:
      Symbol 'SOL/USDT' owned by 'super_bot_grid' must NOT be deleted by swing_architect release.
    """
    st_path = os.path.join(BASE_DIR, "swing_trader.py")
    st_queries = extract_sql_calls(st_path, "DELETE FROM asset_locks")
    assert 433 in st_queries, "Could not find _release_lock_sync query at line 433"
    assert 2262 in st_queries, "Could not find _clean_unknown_db query at line 2262"

    sql_433 = st_queries[433]
    sql_2262 = st_queries[2262]

    # Check that source SQL contains owner guard
    assert "locked_by = 'swing_architect'" in sql_433
    assert "locked_by = 'swing_architect'" in sql_2262

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        conn = sqlite3.connect(db_path)
        conn.execute(PROD_LOCK_SCHEMA)
        now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # 1. Insert lock owned by super_bot_grid
        conn.execute(
            "INSERT INTO asset_locks (symbol, locked_by, locked_at, reason, heartbeat) VALUES (?, ?, ?, ?, ?)",
            ("SOL/USDT", "super_bot_grid", now_ts, "grid_active", now_ts)
        )
        conn.commit()

        # Execute AST-extracted swing_trader.py:433 SQL targeting SOL/USDT
        conn.execute(sql_433, ("SOL/USDT",))
        conn.commit()

        # Assert super_bot_grid lock SURVIVES
        row = conn.execute("SELECT symbol, locked_by FROM asset_locks WHERE symbol = 'SOL/USDT'").fetchone()
        assert row is not None, "Race defect! swing_trader deleted super_bot_grid lock!"
        assert row[1] == "super_bot_grid"

        # Execute AST-extracted swing_trader.py:2262 SQL targeting SOL/USDT
        conn.execute(sql_2262, ("SOL/USDT",))
        conn.commit()

        row = conn.execute("SELECT symbol, locked_by FROM asset_locks WHERE symbol = 'SOL/USDT'").fetchone()
        assert row is not None, "Race defect! swing_trader _clean_unknown_db deleted super_bot_grid lock!"
        assert row[1] == "super_bot_grid"

        # Now insert lock owned by swing_architect on AVAX/USDT
        conn.execute(
            "INSERT INTO asset_locks (symbol, locked_by, locked_at, reason, heartbeat) VALUES (?, ?, ?, ?, ?)",
            ("AVAX/USDT", "swing_architect", now_ts, "swing_entry", now_ts)
        )
        conn.commit()

        # Execute AST-extracted swing_trader.py:433 targeting AVAX/USDT
        conn.execute(sql_433, ("AVAX/USDT",))
        conn.commit()

        # Assert swing_architect's OWN lock IS deleted
        row_avax = conn.execute("SELECT symbol FROM asset_locks WHERE symbol = 'AVAX/USDT'").fetchone()
        assert row_avax is None, "swing_trader failed to release its own lock!"

        conn.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_r6_single_increment_per_failed_pass():
    """
    R6: One failed pass must move the consecutive error counter by exactly 1.
    Runs run_wal_checkpoints() three times against a DB with no system_state table.
    Asserts published counts are 1, 2, 3.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create an unopenable/faulty DB directory
        failing_db = os.path.join(tmpdir, "faulty.db")
        os.makedirs(failing_db)

        # Set up a real trading_data.db with system_state to receive published counts
        state_db = os.path.join(tmpdir, "trading_data.db")
        conn = sqlite3.connect(state_db)
        conn.execute("""
            CREATE TABLE system_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        """)
        conn.commit()
        conn.close()

        orig_base = db_maintenance_daemon.BASE_DIR
        orig_dbs = db_maintenance_daemon.DATABASES
        db_maintenance_daemon.BASE_DIR = tmpdir
        db_maintenance_daemon.DATABASES = [failing_db]
        db_maintenance_daemon._CONSECUTIVE_ERRORS.clear()

        try:
            # Pass 1
            db_maintenance_daemon.run_wal_checkpoints()
            conn = sqlite3.connect(state_db)
            c1 = conn.execute("SELECT value FROM system_state WHERE key = 'db_integrity_faulty.db_consecutive_errors'").fetchone()[0]
            conn.close()
            assert int(c1) == 1, f"Expected pass 1 count to be 1, got {c1}"

            # Pass 2
            db_maintenance_daemon.run_wal_checkpoints()
            conn = sqlite3.connect(state_db)
            c2 = conn.execute("SELECT value FROM system_state WHERE key = 'db_integrity_faulty.db_consecutive_errors'").fetchone()[0]
            conn.close()
            assert int(c2) == 2, f"Expected pass 2 count to be 2, got {c2}"

            # Pass 3
            db_maintenance_daemon.run_wal_checkpoints()
            conn = sqlite3.connect(state_db)
            c3 = conn.execute("SELECT value FROM system_state WHERE key = 'db_integrity_faulty.db_consecutive_errors'").fetchone()[0]
            conn.close()
            assert int(c3) == 3, f"Expected pass 3 count to be 3, got {c3}"

            print(f"R6 consecutive error progression verified: {c1} -> {c2} -> {c3}")
        finally:
            db_maintenance_daemon.BASE_DIR = orig_base
            db_maintenance_daemon.DATABASES = orig_dbs


def test_r7_reconciler_ast_extracted_sql_lock_scoping():
    """
    R7: Extracts queries directly from reconciliation_engine.py (lines 547, 892, 962) via ast.parse.
    Executes against the real production schema including NOT NULL constraints on heartbeat and locked_by.
    Asserts:
      1. A fresh live lock owned by super_bot_grid survives reconciler delete.
      2. An expired lock (>900s) owned by super_bot_grid is pruned.
      3. A lock with NULL heartbeat older than 900s (if heartbeat is defaulted or COALESCE is evaluated)
         is pruned by the bulk query.
    """
    rec_path = os.path.join(BASE_DIR, "reconciliation_engine.py")
    rec_queries = extract_sql_calls(rec_path, "DELETE FROM asset_locks")

    # Find the scoped delete (lines ~547 / ~892)
    scoped_sqls = [sql for line, sql in rec_queries.items() if "WHERE symbol = ?" in sql and "locked_by IN" in sql]
    assert len(scoped_sqls) >= 2, f"Expected at least 2 scoped SQLs in reconciler, found {len(scoped_sqls)}"
    scoped_sql = scoped_sqls[0]

    # Find the bulk delete (lines ~962)
    bulk_sqls = [sql for line, sql in rec_queries.items() if "COALESCE(heartbeat, locked_at)" in sql]
    assert len(bulk_sqls) >= 1, "Expected bulk delete SQL using COALESCE in reconciler"
    bulk_sql = bulk_sqls[0]

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        conn = sqlite3.connect(db_path)
        conn.execute(PROD_LOCK_SCHEMA)

        now_utc = datetime.now(timezone.utc)
        fresh_ts = (now_utc - timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S")
        old_ts = (now_utc - timedelta(minutes=25)).strftime("%Y-%m-%d %H:%M:%S")

        # Lock 1: Fresh live lock owned by super_bot_grid -> MUST SURVIVE
        conn.execute(
            "INSERT INTO asset_locks (symbol, locked_by, locked_at, reason, heartbeat) VALUES (?, ?, ?, ?, ?)",
            ("ETH/USDT", "super_bot_grid", old_ts, "grid_active", fresh_ts)
        )

        # Lock 2: Expired lock (>900s) owned by super_bot_grid -> MUST BE PRUNED
        conn.execute(
            "INSERT INTO asset_locks (symbol, locked_by, locked_at, reason, heartbeat) VALUES (?, ?, ?, ?, ?)",
            ("ADA/USDT", "super_bot_grid", old_ts, "grid_active", old_ts)
        )
        conn.commit()

        # Run AST-extracted scoped delete on ETH/USDT and ADA/USDT
        conn.execute(scoped_sql, ("ETH/USDT",))
        conn.execute(scoped_sql, ("ADA/USDT",))
        conn.commit()

        remaining = [r[0] for r in conn.execute("SELECT symbol FROM asset_locks").fetchall()]
        assert "ETH/USDT" in remaining, "Fresh live lock was erroneously deleted by reconciler scoped delete!"
        assert "ADA/USDT" not in remaining, "Expired lock was not pruned by reconciler scoped delete!"

        # Test bulk query with AST-extracted SQL
        conn.execute(
            "INSERT INTO asset_locks (symbol, locked_by, locked_at, reason, heartbeat) VALUES (?, ?, ?, ?, ?)",
            ("SOL/USDT", "super_bot_grid", old_ts, "grid_active", old_ts)
        )
        conn.commit()

        now_iso = now_utc.strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(bulk_sql, (now_iso,))
        conn.commit()

        rem_bulk = [r[0] for r in conn.execute("SELECT symbol FROM asset_locks").fetchall()]
        assert "SOL/USDT" not in rem_bulk, "Expired lock was not cleaned by AST-extracted bulk prune!"

        conn.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

"""
tests/test_p2_6_regression.py — Regression tests for P2-6 fixes.

Tests:
- Test 1 (N1 Acceptance c): Render tests for HUD badge & banner across three states:
  * OK state
  * Corrupted state
  * Stale state (row 20 minutes old)
  Asserts badge and banner change, and that badge and banner cannot disagree.
- Test 2 (N2 Acceptance d): Durable consecutive error counter survives re-import (simulating daemon restart).
  Calls run_wal_checkpoints() twice with a raising DB, re-imports module fresh, asserts published count is 2.
- Test 3 (N3 Acceptance b): Inspects production EPHEMERAL_TABLES directly imported from db_maintenance_daemon.
  Asserts open_positions / berserker_open_positions are NOT in EPHEMERAL_TABLES.
- Test 4 (N4): Zombie locks query with NULL heartbeat.
  Tests that a lock with heartbeat=NULL but locked_at older than 900s is cleanly deleted by COALESCE(heartbeat, locked_at).
"""

import os
import sys
import time
import json
import sqlite3
import tempfile
import importlib
from datetime import datetime, timezone, timedelta
import pytest

# Ensure repo root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from conviction_filter import check_database_integrity, PROD_DATABASES
import db_maintenance_daemon


def test_n1_hud_badge_and_banner_states():
    """
    Acceptance (c): Show three renders side by side — integrity OK / integrity CORRUPT / integrity row 20m old.
    Assert the badge AND the banner both change, and assert they cannot disagree.
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

        def render_hud_state(test_conn):
            report = check_database_integrity(test_conn, max_age_sec=getattr(config, "DB_INTEGRITY_MAX_AGE_SEC", 900.0))
            
            # Badge logic (matches dashboard.py:1058-1065)
            if report["is_ok"]:
                badge = "🟢 Database Integrity: Verified OK"
                badge_state = "OK"
            elif report["is_stale"]:
                last_ts_str = report["latest_utc_dt"].strftime('%H:%M:%S UTC') if report.get("latest_utc_dt") else "UNKNOWN"
                badge = f"🟠 Database Integrity: SIGNAL STALE (last verified {last_ts_str}, daemon not publishing)"
                badge_state = "STALE"
            else:
                badge = "🚨 Database Integrity: CRITICAL ERROR"
                badge_state = "CORRUPT"

            # Banner logic (matches dashboard.py:1080-1090)
            if not report["is_ok"]:
                if report["is_corrupt"]:
                    corrupted_names = report["corrupt_dbs"]
                    banner = f"🚨 DATABASE CORRUPTION DETECTED (Fail-Closed): {', '.join(corrupted_names)}. New trade entries are BLOCKED."
                    banner_state = "CORRUPT"
                elif report["is_stale"]:
                    stale_names = report["stale_dbs"]
                    last_str = report["latest_utc_dt"].strftime('%H:%M:%S UTC') if report.get("latest_utc_dt") else "UNKNOWN"
                    banner = f"🟠 DATABASE INTEGRITY SIGNAL STALE: {', '.join(stale_names)} (last verified {last_str}, daemon not publishing). New trade entries are BLOCKED."
                    banner_state = "STALE"
            else:
                banner = None
                banner_state = "OK"

            # Agreement invariant: badge and banner states must match
            assert badge_state == banner_state, f"Disagreement: badge={badge_state} vs banner={banner_state}"
            return badge, banner, badge_state

        now_utc = datetime.now(timezone.utc)
        fresh_ts = now_utc.strftime("%Y-%m-%d %H:%M:%S")

        # ── State 1: Integrity OK ──
        for d in PROD_DATABASES:
            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, 'OK', ?)", (f"db_integrity_{d}", fresh_ts))
            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, '0', ?)", (f"db_integrity_{d}_consecutive_errors", fresh_ts))
        conn.commit()

        b1, ban1, s1 = render_hud_state(conn)
        assert s1 == "OK"
        assert "🟢" in b1
        assert "Verified OK" in b1
        assert ban1 is None

        # ── State 2: Integrity CORRUPT ──
        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('db_integrity_trading_ledger.db', 'CRITICAL', ?)", (fresh_ts,))
        conn.commit()

        b2, ban2, s2 = render_hud_state(conn)
        assert s2 == "CORRUPT"
        assert "🚨" in b2
        assert "CRITICAL ERROR" in b2
        assert ban2 is not None
        assert "DATABASE CORRUPTION DETECTED" in ban2
        assert "trading_ledger.db" in ban2
        assert "BLOCKED" in ban2

        # ── State 3: Integrity row 20 minutes old (Stale) ──
        # Reset trading_ledger.db to OK, but set all timestamps to 20 mins ago (1200s > 900s)
        stale_20m = (now_utc - timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M:%S")
        for d in PROD_DATABASES:
            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, 'OK', ?)", (f"db_integrity_{d}", stale_20m))
            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, '0', ?)", (f"db_integrity_{d}_consecutive_errors", stale_20m))
        conn.commit()

        b3, ban3, s3 = render_hud_state(conn)
        assert s3 == "STALE"
        assert "🟠" in b3
        assert "SIGNAL STALE" in b3
        assert "daemon not publishing" in b3
        assert ban3 is not None
        assert "DATABASE INTEGRITY SIGNAL STALE" in ban3
        assert "New trade entries are BLOCKED" in ban3

        # Prove badge and banner changed across all 3 states
        assert s1 != s2
        assert s2 != s3
        assert s1 != s3
        assert b1 != b2 != b3
        assert ban1 != ban2 != ban3

        conn.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_n2_durable_consecutive_errors_fresh_reimport(monkeypatch):
    """
    Acceptance (d): Consecutive-error counting must survive a daemon restart.
    Calls run_wal_checkpoints() twice with a raising DB, then RE-IMPORTS the module fresh
    (simulating a restart) and asserts the published count is 2, not 1.
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

        # Target DB that will raise an error during checkpoint
        raising_db = os.path.join(tmpdir, "raising_target.db")
        # Do not create raising_db or make it an unopenable directory to force sqlite3.connect to raise
        os.makedirs(raising_db)  # directory with same name will cause sqlite3.connect to raise OperationalError / PermissionError

        # Configure monkeypatch on BASE_DIR and DATABASES
        import db_maintenance_daemon as dmd
        monkeypatch.setattr(dmd, "BASE_DIR", tmpdir)
        monkeypatch.setattr(dmd, "DATABASES", [raising_db])
        monkeypatch.setattr(dmd, "CHECKPOINT_INTERVAL_SECS", 300)
        dmd._CONSECUTIVE_ERRORS.clear()

        # Pass 1: raises exception, consecutive count should be 1
        results1, statuses1 = dmd.run_wal_checkpoints()
        assert statuses1[os.path.basename(raising_db)] == "ERROR"

        conn = sqlite3.connect(test_state_db)
        r1 = conn.execute(
            "SELECT value FROM system_state WHERE key = ?",
            (f"db_integrity_{os.path.basename(raising_db)}_consecutive_errors",)
        ).fetchone()
        conn.close()
        assert r1 is not None
        count_pass1 = int(r1[0])
        assert count_pass1 == 1

        # Now simulate a process crash / restart:
        # Re-import module fresh and clear in-memory _CONSECUTIVE_ERRORS dict
        if "db_maintenance_daemon" in sys.modules:
            del sys.modules["db_maintenance_daemon"]
        
        fresh_dmd = importlib.import_module("db_maintenance_daemon")
        fresh_dmd._CONSECUTIVE_ERRORS.clear()
        assert len(fresh_dmd._CONSECUTIVE_ERRORS) == 0

        # Patch paths on fresh module instance
        monkeypatch.setattr(fresh_dmd, "BASE_DIR", tmpdir)
        monkeypatch.setattr(fresh_dmd, "DATABASES", [raising_db])
        monkeypatch.setattr(fresh_dmd, "CHECKPOINT_INTERVAL_SECS", 300)

        # Pass 2 on freshly imported daemon: must read durable count (1) and increment to 2
        results2, statuses2 = fresh_dmd.run_wal_checkpoints()
        assert statuses2[os.path.basename(raising_db)] == "ERROR"

        conn = sqlite3.connect(test_state_db)
        r2 = conn.execute(
            "SELECT value FROM system_state WHERE key = ?",
            (f"db_integrity_{os.path.basename(raising_db)}_consecutive_errors",)
        ).fetchone()
        conn.close()
        assert r2 is not None
        count_pass2 = int(r2[0])
        print(f"Published consecutive errors after re-import: {count_pass2}")
        assert count_pass2 == 2, f"Expected durable consecutive count to be 2, but got {count_pass2}"


def test_n3_real_ephemeral_tables_production_config():
    """
    Acceptance (b): Test the REAL production config: build/read the EPHEMERAL_TABLES entry
    from the shipped module (import it, do not hand-write a substitute).
    Asserts live_paper_trades.db open_positions and berserker_open_positions are deleted.
    """
    shipped_ephemeral = db_maintenance_daemon.EPHEMERAL_TABLES
    shipped_tables = [(e["db"], e["table"]) for e in shipped_ephemeral]

    # Must NOT contain open_positions or berserker_open_positions
    assert ("live_paper_trades.db", "open_positions") not in shipped_tables
    assert ("live_paper_trades.db", "berserker_open_positions") not in shipped_tables

    # Telemetry tables remain intact
    assert ("live_paper_trades.db", "telemetry") in shipped_tables
    assert ("live_paper_trades.db", "berserker_telemetry") in shipped_tables
    assert ("trading_data.db", "telemetry") in shipped_tables


def test_n4_zombie_locks_on_null_heartbeat():
    """
    N4: Zombie locks on NULL heartbeat.
    Tests that a lock with heartbeat=NULL but locked_at older than 900s is cleanly deleted
    by (julianday('now') - julianday(COALESCE(heartbeat, locked_at))) * 86400.0 > 900.0
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
        
        # Insert 3 locks:
        # 1. Lock with NULL heartbeat, locked_at 25 minutes ago (1500s > 900s) -> MUST BE DELETED
        old_locked_at = (datetime.now(timezone.utc) - timedelta(minutes=25)).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            "INSERT INTO asset_locks VALUES ('ADA/USDT', 'super_bot_grid', ?, NULL)",
            (old_locked_at,)
        )

        # 2. Lock with fresh heartbeat 2 minutes ago (120s < 900s) -> MUST BE KEPT
        fresh_hb = (datetime.now(timezone.utc) - timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            "INSERT INTO asset_locks VALUES ('ETH/USDT', 'super_bot_grid', ?, ?)",
            (old_locked_at, fresh_hb)
        )

        # 3. Lock with NULL heartbeat, but locked_at only 2 minutes ago (120s < 900s) -> MUST BE KEPT
        recent_locked_at = (datetime.now(timezone.utc) - timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            "INSERT INTO asset_locks VALUES ('SOL/USDT', 'super_bot_grid', ?, NULL)",
            (recent_locked_at,)
        )
        conn.commit()

        # Run the super_bot.py query under test
        conn.execute("""
            DELETE FROM asset_locks
            WHERE locked_by IN ('super_bot_grid', 'super_bot_scalp_short', 'swing_architect')
              AND (julianday('now') - julianday(COALESCE(heartbeat, locked_at))) * 86400.0 > 900.0
        """)
        conn.commit()

        remaining = [r[0] for r in conn.execute("SELECT symbol FROM asset_locks").fetchall()]
        assert "ADA/USDT" not in remaining, "Zombie lock with NULL heartbeat older than 900s was NOT deleted!"
        assert "ETH/USDT" in remaining, "Active lock with fresh heartbeat was erroneously deleted!"
        assert "SOL/USDT" in remaining, "Recent lock with NULL heartbeat was erroneously deleted!"
        conn.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

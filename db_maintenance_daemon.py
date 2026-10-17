"""
db_maintenance_daemon.py — Sovereign Zero-Maintenance Database Daemon

Key Capabilities:
1. Automated periodic WAL checkpoints (PASSIVE and TRUNCATE) to keep SQLite WAL journals minimal.
2. Ephemeral data pruning: Trims high-frequency tick logs, telemetry, and debug tables older than 7 days.
3. Strict Ledger Preservation: Never deletes closed trades (trades, berserker_trades, marl_live_trades, mean_reversion_trades).
4. SQLite Query Plan Optimization: Runs `PRAGMA optimize` to rebuild indexes and maintain sub-millisecond query execution.
5. Temp file cleanup: Removes stale temporary process files (*.tmp, *.npy) from the root workspace.
6. Publishes health, per-database integrity verdicts (db_integrity_<name>) and storage
   metrics to system_state for live Streamlit visibility and downstream fail-closed gating.
"""

import os
import sys
import time
import json
import glob
import sqlite3
import logging
import contextlib
from datetime import datetime, UTC, timedelta
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [DB_MAINTENANCE] %(message)s"
)
log = logging.getLogger("db_maintenance")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASES = [
    os.path.join(BASE_DIR, "trading_data.db"),
    os.path.join(BASE_DIR, "trading_ledger.db"),
    os.path.join(BASE_DIR, "live_paper_trades.db"),
    os.path.join(BASE_DIR, "test_ledger.db")
]

CHECKPOINT_INTERVAL_SECS = 300  # Run maintenance every 5 minutes
RETENTION_DAYS = 7              # Retain ephemeral telemetry for 7 days

_WAL_MODE_WARNED = set()

def _ensure_wal(conn, db_name: str, busy_timeout_ms: int) -> str:
    """Set busy_timeout first, then attempt WAL, and report the mode actually granted.

    PRAGMA journal_mode does not raise when conversion fails (locked DB or network
    share) - it returns the mode still in effect, so the result has to be inspected.
    """
    conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms};")
    row = conn.execute("PRAGMA journal_mode=WAL;").fetchone()
    actual = str(row[0]).lower() if row and row[0] is not None else "unknown"
    if actual != "wal" and db_name not in _WAL_MODE_WARNED:
        _WAL_MODE_WARNED.add(db_name)
        log.warning(
            f"[{db_name}] journal_mode is '{actual}', not 'wal' - WAL conversion failed. "
            "Readers will contend with writers."
        )
    return actual

EPHEMERAL_TABLES = [
    {"db": "trading_ledger.db", "table": "market_metrics", "time_col": "timestamp"},
    {"db": "trading_ledger.db", "table": "oracle_microstructure", "time_col": "timestamp"},
    {"db": "trading_ledger.db", "table": "macro_market_context", "time_col": "timestamp"},
    {"db": "trading_data.db", "table": "ghost_trades", "time_col": "timestamp"},
    {"db": "trading_data.db", "table": "telemetry", "time_col": "timestamp"},
]

def update_system_state(key: str, value: str):
    """Safely writes a key-value record to the system_state table."""
    try:
        data_db = os.path.join(BASE_DIR, "trading_data.db")
        with contextlib.closing(sqlite3.connect(data_db, timeout=10.0)) as conn:
            with conn:
                _ensure_wal(conn, "trading_data.db", 10000)
                conn.execute(
                    "INSERT INTO system_state (key, value, updated_at) VALUES (?, ?, datetime('now')) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                    (key, value)
                )
    except Exception as e:
        # A silently dropped health write is itself a fault: downstream guards gate on
        # freshness of these keys, so callers must be able to see the publish failure.
        log.warning(f"Failed to update system_state ({key}): {e}")

def run_wal_checkpoints():
    """Runs non-blocking WAL checkpoints and PRAGMA quick_check across all databases.

    Returns (results, statuses). Integrity verdicts are published to system_state as
    db_integrity_<dbname> so engines and the HUD can fail closed instead of relying on
    a human reading logs.
    """
    results = {}
    statuses = {}
    for db_path in DATABASES:
        if not os.path.exists(db_path):
            continue
        db_name = os.path.basename(db_path)
        try:
            with contextlib.closing(sqlite3.connect(db_path, timeout=30.0)) as conn:
                _ensure_wal(conn, db_name, 30000)
                qc = conn.execute("PRAGMA quick_check(1);").fetchone()
                qc_status = str(qc[0]) if qc else "unknown"
                integrity_ok = qc_status.lower() == "ok"
                res = conn.execute("PRAGMA wal_checkpoint(PASSIVE);").fetchone()
                conn.execute("PRAGMA optimize;")

            if not integrity_ok:
                log.critical(f"[{db_name}] 🚨 DATABASE INTEGRITY QUICK_CHECK FAILED: {qc_status}")
            update_system_state(f"db_integrity_{db_name}", "OK" if integrity_ok else "CRITICAL")

            busy = int(res[0]) if res and res[0] is not None else -1
            log_frames = int(res[1]) if res and res[1] is not None else -1
            checkpointed = int(res[2]) if res and res[2] is not None else -1
            # quick_check returns a multi-line dump on failure; keep the published string
            # single-line so it stays readable inside the system_state JSON blob.
            qc_brief = " ".join(qc_status.split())[:120]

            if not integrity_ok:
                status = "CORRUPT"
            elif busy == 1:
                # PASSIVE checkpoint returned busy: WAL was not drained this pass.
                status = "BUSY"
            elif log_frames != checkpointed:
                status = "DEGRADED"
            else:
                status = "OK"
            statuses[db_name] = status
            results[db_name] = (
                f"WAL Checkpoint {status} (busy={busy}, log={log_frames}, "
                f"checkpointed={checkpointed}, integrity={qc_brief})"
            )
            log.info(f"[{db_name}] {results[db_name]}")
        except Exception as e:
            # An exception in this path is worse than a failed result: the DB was not
            # verified at all. Route it to the same machine-readable CRITICAL signal.
            statuses[db_name] = "ERROR"
            results[db_name] = f"WAL Checkpoint ERROR (integrity unverified): {e}"
            log.error(f"[{db_name}] 🚨 Checkpoint/integrity pass raised before verification: {e}")
            update_system_state(f"db_integrity_{db_name}", "ERROR")
    return results, statuses

def prune_ephemeral_records():
    """Prunes high-frequency market tick logs and telemetry older than RETENTION_DAYS."""
    pruned_summary = {}
    cutoff_dt = datetime.now(UTC) - timedelta(days=RETENTION_DAYS)
    cutoff_space = cutoff_dt.strftime("%Y-%m-%d %H:%M:%S")
    cutoff_iso = cutoff_dt.strftime("%Y-%m-%dT%H:%M:%S")
    
    for item in EPHEMERAL_TABLES:
        db_path = os.path.join(BASE_DIR, item["db"])
        table = item["table"]
        time_col = item["time_col"]
        
        if not os.path.exists(db_path):
            continue
            
        try:
            with contextlib.closing(sqlite3.connect(db_path, timeout=10.0)) as conn:
                _ensure_wal(conn, item["db"], 10000)
                # Check if table exists
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
                )
                if not cursor.fetchone():
                    continue
                    
                # Delete rows older than cutoff (handling both standard space and ISO-T string formats)
                with conn:
                    res = conn.execute(
                        f"DELETE FROM {table} WHERE {time_col} < ? OR ({time_col} LIKE '%T%' AND {time_col} < ?)",
                        (cutoff_space, cutoff_iso)
                    )
                    deleted = res.rowcount
                if deleted > 0:
                    pruned_summary[f"{item['db']}.{table}"] = deleted
                    log.info(f"[{item['db']}] Pruned {deleted} rows older than {RETENTION_DAYS}d from {table}")
        except Exception as e:
            log.warning(f"Prune skip for {item['db']}.{table}: {e}")
            
    return pruned_summary

def cleanup_stale_temp_files():
    """Removes orphan temporary files generated by processes and rotates old logs."""
    cleaned = 0
    patterns = ["*.tmp", "*.npy.tmp", "temp_*.json", "tmp*.json"]
    for pat in patterns:
        for fpath in glob.glob(os.path.join(BASE_DIR, pat)):
            try:
                # Delete temp files older than 1 hour
                if (time.time() - os.path.getmtime(fpath)) > 3600:
                    os.remove(fpath)
                    cleaned += 1
            except Exception:
                pass
                
    # Also prune logs older than 7 days in logs/
    log_dir = os.path.join(BASE_DIR, "logs")
    if os.path.exists(log_dir):
        for lpath in glob.glob(os.path.join(log_dir, "*.log*")):
            try:
                # If older than 7 days and not the active main/launcher log
                if (time.time() - os.path.getmtime(lpath)) > (7 * 86400):
                    if not any(active in os.path.basename(lpath) for active in ["main.log", "launcher.log"]):
                        os.remove(lpath)
                        cleaned += 1
            except Exception:
                pass

    if cleaned > 0:
        log.info(f"[CLEANUP] Removed {cleaned} stale temporary and old log files from workspace.")
    return cleaned

def get_storage_stats():
    """Calculates total disk footprint of databases and logs."""
    stats = {}
    total_bytes = 0
    for db_path in DATABASES:
        if os.path.exists(db_path):
            sz = os.path.getsize(db_path)
            total_bytes += sz
            stats[os.path.basename(db_path)] = f"{sz / (1024 * 1024):.2f} MB"
            
            # Check WAL file size
            wal_path = db_path + "-wal"
            if os.path.exists(wal_path):
                wal_sz = os.path.getsize(wal_path)
                stats[os.path.basename(wal_path)] = f"{wal_sz / (1024 * 1024):.2f} MB"
                total_bytes += wal_sz

    stats["total_storage"] = f"{total_bytes / (1024 * 1024):.2f} MB"
    return stats

def main():
    log.info("=" * 60)
    log.info("     SOVEREIGN DATABASE MAINTENANCE DAEMON ACTIVE")
    log.info(f"     Prune Retention: {RETENTION_DAYS} Days | Interval: {CHECKPOINT_INTERVAL_SECS}s")
    log.info("=" * 60)

    while True:
        try:
            log.info("Starting scheduled database maintenance pass...")
            
            # 1. Non-blocking WAL Checkpoints
            cp_res, cp_status = run_wal_checkpoints()
            
            # 2. Prune old records
            prune_res = prune_ephemeral_records()
            
            # 3. Clean temp files
            cleaned_temp = cleanup_stale_temp_files()
            
            # 4. Storage sizes
            storage_stats = get_storage_stats()
            
            # 5. Emit health metrics (status reflects the worst per-DB verdict, never a
            # hardcoded healthy)
            if any(s in ("CORRUPT", "ERROR") for s in cp_status.values()):
                overall_status = "CRITICAL"
            elif any(s in ("BUSY", "DEGRADED") for s in cp_status.values()):
                overall_status = "DEGRADED"
            else:
                overall_status = "HEALTHY"
            telemetry = {
                "last_run": datetime.now(UTC).isoformat(),
                "status": overall_status,
                "storage": storage_stats,
                "checkpoints": cp_res,
                "integrity": cp_status,
                "pruned_tables": prune_res,
                "temp_files_cleaned": cleaned_temp
            }
            update_system_state("db_maintenance_stats", json.dumps(telemetry))
            log.info(f"Maintenance pass complete. Status: {overall_status}. Storage: {storage_stats['total_storage']}. Sleeping {CHECKPOINT_INTERVAL_SECS}s.")
            
        except Exception as e:
            log.error(f"Maintenance pass error: {e}")
            
        time.sleep(CHECKPOINT_INTERVAL_SECS)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Database maintenance daemon terminated.")

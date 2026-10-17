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
from datetime import datetime, timezone, UTC, timedelta
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

import config

CHECKPOINT_INTERVAL_SECS = getattr(config, "DB_INTEGRITY_CHECKPOINT_INTERVAL_SECS", 300)  # Run maintenance every 5 minutes
ERROR_COUNTER_WINDOW_SECS = float(CHECKPOINT_INTERVAL_SECS + 60.0)  # 360s window to cover 304-305s inter-pass interval plus margin
RETENTION_DAYS = 7              # Retain ephemeral telemetry for 7 days

_WAL_MODE_WARNED = set()
_CONSECUTIVE_ERRORS = {}

def get_durable_consecutive_errors(db_name: str, window_sec: float = None) -> int:
    """
    Reads previously published db_integrity_<db>_consecutive_errors from system_state.
    If the row is within window_sec (default ERROR_COUNTER_WINDOW_SECS = CHECKPOINT_INTERVAL_SECS + 60.0), returns its
    integer count; otherwise returns 0 (stale or absent, starting a fresh run).

    NOTE (R2): get_durable_consecutive_errors reads trading_data.db, so when trading_data.db
    is the locked DB the counter resets to 1 every pass and never escalates; the escalation
    path is inert for that DB. Only the 900s staleness backstop blocks entries when trading_data.db
    is locked/unreachable.
    """
    if window_sec is None:
        window_sec = ERROR_COUNTER_WINDOW_SECS
    data_db = os.path.join(BASE_DIR, "trading_data.db")
    if not os.path.exists(data_db):
        return 0
    try:
        with contextlib.closing(sqlite3.connect(data_db, timeout=5.0)) as conn:
            _ensure_wal(conn, "trading_data.db", 5000)
            key = f"db_integrity_{db_name}_consecutive_errors"
            row = conn.execute(
                "SELECT value, updated_at FROM system_state WHERE key = ?", (key,)
            ).fetchone()
            if not row or not row[0] or not row[1]:
                return 0
            val_str, u_at = str(row[0]).strip(), str(row[1]).strip()
            
            clean_ts = u_at.replace("Z", "+00:00")
            if "T" not in clean_ts and " " in clean_ts:
                clean_ts = clean_ts.replace(" ", "T")
            parsed_dt = datetime.fromisoformat(clean_ts)
            if parsed_dt.tzinfo is None:
                parsed_dt = parsed_dt.replace(tzinfo=UTC)
            
            age_sec = max(0.0, (datetime.now(UTC) - parsed_dt).total_seconds())
            if age_sec <= window_sec:
                return int(val_str)
            else:
                return 0
    except Exception as e:
        log.warning(f"Failed to read durable consecutive errors for {db_name}: {e}")
        return 0

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
    {"db": "live_paper_trades.db", "table": "telemetry", "time_col": "timestamp"},
    {"db": "live_paper_trades.db", "table": "berserker_telemetry", "time_col": "timestamp"},
]

def update_system_state(key: str, value: str):
    """Safely writes a key-value record to the system_state table."""
    update_system_state_batch([(key, value)])

def update_system_state_batch(kvs: list) -> bool:
    """
    Atomically writes a batch of (key, value) pairs to the system_state table in a single transaction.
    Returns True on success. Raises an exception on failure so callers know the write failed.
    """
    if not kvs:
        return True
    data_db = os.path.join(BASE_DIR, "trading_data.db")
    with contextlib.closing(sqlite3.connect(data_db, timeout=10.0)) as conn:
        with conn:
            _ensure_wal(conn, "trading_data.db", 10000)
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            conn.executemany(
                "INSERT INTO system_state (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                [(k, v, now_str) for (k, v) in kvs]
            )
    return True

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
            
            # Atomic publish of status and counter in ONE transaction
            status_val = "OK" if integrity_ok else "CRITICAL"
            update_system_state_batch([
                (f"db_integrity_{db_name}", status_val),
                (f"db_integrity_{db_name}_consecutive_errors", "0")
            ])
            _CONSECUTIVE_ERRORS[db_name] = 0

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
            # An exception in this path means checkpoint/check raised (e.g. transient DB lock)
            # OR publishing to system_state failed.
            # Increment durable consecutive error counter and publish to system_state in ONE atomic transaction.
            prior_consec = _CONSECUTIVE_ERRORS.get(db_name)
            if prior_consec is None:
                prior_consec = get_durable_consecutive_errors(db_name, ERROR_COUNTER_WINDOW_SECS)
            curr_consec = prior_consec + 1
            _CONSECUTIVE_ERRORS[db_name] = curr_consec

            statuses[db_name] = "ERROR"
            results[db_name] = f"WAL Checkpoint ERROR (consecutive={curr_consec}, unverified): {e}"
            log.error(f"[{db_name}] 🚨 Checkpoint/integrity pass raised (consecutive={curr_consec}): {e}")
            try:
                update_system_state_batch([
                    (f"db_integrity_{db_name}", "ERROR"),
                    (f"db_integrity_{db_name}_consecutive_errors", str(curr_consec))
                ])
            except Exception as batch_err:
                log.warning(f"[{db_name}] Failed to atomically publish ERROR status and consecutive errors: {batch_err}")
    return results, statuses

def prune_ephemeral_records():
    """Prunes high-frequency market tick logs and telemetry older than RETENTION_DAYS."""
    pruned_summary = {}
    
    for item in EPHEMERAL_TABLES:
        db_path = os.path.join(BASE_DIR, item["db"])
        table = item["table"]
        time_col = item["time_col"]
        retention = item.get("retention_days", RETENTION_DAYS)
        extra_where = item.get("extra_where", None)
        
        cutoff_dt = datetime.now(UTC) - timedelta(days=retention)
        cutoff_space = cutoff_dt.strftime("%Y-%m-%d %H:%M:%S")
        cutoff_iso = cutoff_dt.strftime("%Y-%m-%dT%H:%M:%S")
        
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

                # Inspect schema to see if status column exists
                cols = [c[1].lower() for c in conn.execute(f"PRAGMA table_info({table})").fetchall()]
                where_clauses = [f"({time_col} < ? OR ({time_col} LIKE '%T%' AND {time_col} < ?))"]
                params = [cutoff_space, cutoff_iso]

                if "status" in cols:
                    # Table has a status column: NEVER delete ACTIVE or PENDING_CLOSE rows
                    where_clauses.append("UPPER(COALESCE(status, '')) NOT IN ('ACTIVE', 'PENDING_CLOSE')")
                    if extra_where:
                        where_clauses.append(f"({extra_where})")
                else:
                    # If this is an open positions table without a status column, protect live rows
                    if "open_position" in table.lower():
                        log.info(f"[{item['db']}] Table {table} has no status column; skipping prune to protect live positions.")
                        continue
                    if extra_where and "status" not in extra_where.lower():
                        where_clauses.append(f"({extra_where})")

                where_sql = " AND ".join(where_clauses)
                query = f"DELETE FROM {table} WHERE {where_sql}"
                with conn:
                    res = conn.execute(query, tuple(params))
                    deleted = res.rowcount
                if deleted > 0:
                    pruned_summary[f"{item['db']}.{table}"] = deleted
                    log.info(f"[{item['db']}] Pruned {deleted} rows older than {retention}d from {table}")
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

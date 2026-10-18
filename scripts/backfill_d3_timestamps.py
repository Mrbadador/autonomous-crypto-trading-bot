"""
scripts/backfill_d3_timestamps.py
Part of FIX P2-9 (D-3).

Normalizes legacy ISO-formatted timestamps ('2026-10-17T12:54:23.926740+00:00') in
swing_trades into canonical SQLite UTC space format ('YYYY-MM-DD HH:MM:SS').

Defaults to --dry-run. Requires --apply to make changes.
When --apply is specified:
  1. Creates backup table 'swing_trades_backup_d3' if not exists.
  2. Updates all rows matching timestamp LIKE '%T%'.
  3. Verifies that 0 rows remain with 'T'.
"""

import argparse
import os
import sqlite3
import sys


def backfill_timestamps(db_path: str, table_name: str = "swing_trades", apply_changes: bool = False) -> int:
    if not os.path.exists(db_path):
        print(f"[ERROR] Database file not found: {db_path}")
        return 1

    print(f"Connecting to {db_path} (table: {table_name})...")
    conn = sqlite3.connect(db_path, timeout=10.0)
    try:
        conn.execute("PRAGMA busy_timeout = 10000;")

        # Check if table exists
        has_table = conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name = ?", (table_name,)
        ).fetchone()[0]
        if not has_table:
            print(f"[ERROR] Table '{table_name}' does not exist.")
            return 1

        # Find ISO rows
        iso_rows = conn.execute(
            f"SELECT rowid, timestamp FROM {table_name} WHERE timestamp LIKE '%T%' ORDER BY rowid"
        ).fetchall()
        print(f"Found {len(iso_rows)} rows in {table_name} with ISO 'T' timestamps.")

        if not iso_rows:
            print(f"No rows need normalization. {table_name} is already canonical!")
            return 0

        print("\nSample before -> after transformations:")
        for r_id, r_ts in iso_rows[:5]:
            norm_ts = str(r_ts)[:19].replace("T", " ")
            print(f"  Row {r_id}: '{r_ts}' -> '{norm_ts}'")

        if not apply_changes:
            print("\n[DRY RUN] No changes applied. Run with --apply to execute migration.")
            return 0

        # Apply mode: backup then update
        backup_table = f"{table_name}_backup_d3"
        print(f"\n[APPLY] Creating backup table '{backup_table}'...")
        conn.execute(f"CREATE TABLE IF NOT EXISTS {backup_table} AS SELECT * FROM {table_name}")
        backup_count = conn.execute(f"SELECT count(*) FROM {backup_table}").fetchone()[0]
        print(f"[APPLY] Backup table verified with {backup_count} rows.")

        print(f"[APPLY] Normalizing {len(iso_rows)} timestamps in {table_name}...")
        cur = conn.execute(f"""
            UPDATE {table_name}
            SET timestamp = replace(substr(timestamp, 1, 19), 'T', ' ')
            WHERE timestamp LIKE '%T%'
        """)
        updated_count = cur.rowcount
        conn.commit()
        print(f"[APPLY] Successfully updated {updated_count} rows.")

        # Post-verification
        rem_count = conn.execute(
            f"SELECT count(*) FROM {table_name} WHERE timestamp LIKE '%T%'"
        ).fetchone()[0]
        print(f"[VERIFY] Remaining rows with 'T' in timestamp: {rem_count}")
        if rem_count == 0:
            print(f"[SUCCESS] All {table_name} timestamps are now in canonical SQLite space format!")
            return 0
        else:
            print(f"[WARNING] {rem_count} rows still have 'T'!")
            return 1
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill ISO timestamps in trade tables to canonical space format.")
    parser.add_argument("--db", default="trading_data.db", help="Path to SQLite database (default: trading_data.db)")
    parser.add_argument("--table", default="swing_trades", help="Table name to backfill (default: swing_trades)")
    parser.add_argument("--apply", action="store_true", default=False, help="Apply changes to database (default is dry-run)")
    args = parser.parse_args()

    sys.exit(backfill_timestamps(args.db, table_name=args.table, apply_changes=args.apply))

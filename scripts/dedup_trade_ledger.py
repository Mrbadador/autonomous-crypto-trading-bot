"""Deduplicate the trade ledger and install the unique indexes.

Destructive: it DELETES duplicate trade rows. A timestamped backup of the
database is written first unless --no-backup is given.

Usage:
    python scripts/dedup_trade_ledger.py --db trading_data.db
    python scripts/dedup_trade_ledger.py --db trading_data.db --dry-run
"""
import argparse
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone

# The uniqueness keys the schema enforces. Duplicate rows are exact repeats on
# every one of these columns.
DUP_KEYS = {
    "trades": ("symbol", "side", "entry_price", "timestamp", "qty"),
    "swing_trades": ("symbol", "side", "entry_price", "timestamp", "size"),
}

UNIQUE_INDEXES = (
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_unique"
        " ON trades (symbol, side, entry_price, timestamp, qty)"
    ),
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_swing_trades_unique"
        " ON swing_trades (symbol, side, entry_price, timestamp, size)"
    ),
)


def dedup_table(conn, table):
    """Removes duplicate rows from `table`, keeping the lowest id of each group.

    Returns (rows_before, rows_after).
    """
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    if not cols or "id" not in cols:
        print(f"{table}: table not present, skipped")
        return 0, 0
    keys = [c for c in DUP_KEYS[table] if c in cols]
    if len(keys) != len(DUP_KEYS[table]):
        print(f"{table}: missing columns {set(DUP_KEYS[table]) - set(cols)}, skipped")
        return 0, 0

    before = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    key_sql = ", ".join(keys)
    conn.execute(
        f"""DELETE FROM {table} WHERE id NOT IN (
                SELECT MIN(id) FROM {table} GROUP BY {key_sql}
            )"""
    )
    after = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return before, after


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, help="path to the SQLite ledger")
    ap.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    ap.add_argument("--no-backup", action="store_true", help="skip the backup copy")
    args = ap.parse_args()

    db_path = os.path.abspath(args.db)
    if not os.path.exists(db_path):
        print(f"ERROR: {db_path} does not exist")
        return 1

    if args.dry_run:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            for table in DUP_KEYS:
                before, after = dedup_table(conn, table)
                print(f"{table}: {before} -> {after} rows (dry run, not committed)")
        finally:
            conn.close()
        print("dry run: no changes made")
        return 0

    if not args.no_backup:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup = f"{db_path}.bak_{stamp}"
        shutil.copy2(db_path, backup)
        print(f"backup written: {backup}")

    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        for table in DUP_KEYS:
            before, after = dedup_table(conn, table)
            conn.commit()
            print(f"{table}: {before} -> {after} rows")
        for stmt in UNIQUE_INDEXES:
            conn.execute(stmt)
        conn.commit()
        print("unique indexes created")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

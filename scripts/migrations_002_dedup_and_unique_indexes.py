"""
scripts/migrations_002_dedup_and_unique_indexes.py
=================================================
Deduplicates rows in `trades` and `swing_trades` based on
(symbol, entry_price, timestamp) and enforces UNIQUE indexes
so duplicate closed trades can never re-accumulate.
"""

import os
import sys
import sqlite3
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("migration_002")

def run_migration(db_path: str = "trading_data.db"):
    db_path = os.path.abspath(db_path)
    if not os.path.exists(db_path):
        log.warning(f"Target DB {db_path} does not exist. Skipping.")
        return

    log.info(f"Running Migration 002 on {db_path}...")
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")

    with conn:
        # 1. Deduplicate trades table
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
        if cursor.fetchone():
            count_pre = conn.execute("SELECT count(*) FROM trades").fetchone()[0]
            conn.execute("""
                DELETE FROM trades
                WHERE rowid NOT IN (
                    SELECT MIN(rowid)
                    FROM trades
                    GROUP BY symbol, entry_price, timestamp
                )
            """)
            count_post = conn.execute("SELECT count(*) FROM trades").fetchone()[0]
            log.info(f"trades table deduplicated: {count_pre} -> {count_post} rows (removed {count_pre - count_post})")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_unique ON trades (symbol, entry_price, timestamp)")
            log.info("Created unique index idx_trades_unique on trades(symbol, entry_price, timestamp)")

        # 2. Deduplicate swing_trades table
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='swing_trades'")
        if cursor.fetchone():
            count_pre = conn.execute("SELECT count(*) FROM swing_trades").fetchone()[0]
            conn.execute("""
                DELETE FROM swing_trades
                WHERE rowid NOT IN (
                    SELECT MIN(rowid)
                    FROM swing_trades
                    GROUP BY symbol, entry_price, timestamp
                )
            """)
            count_post = conn.execute("SELECT count(*) FROM swing_trades").fetchone()[0]
            log.info(f"swing_trades table deduplicated: {count_pre} -> {count_post} rows (removed {count_pre - count_post})")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_swing_trades_unique ON swing_trades (symbol, entry_price, timestamp)")
            log.info("Created unique index idx_swing_trades_unique on swing_trades(symbol, entry_price, timestamp)")

    conn.close()
    log.info("Migration 002 completed successfully.")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "trading_data.db"
    run_migration(target)

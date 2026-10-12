import sqlite3
import os
import sys

DB_PATH = "trading_data.db"

def run_migration():
    print(f"Applying migration to {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH, timeout=15.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=15000;")

    with conn:
        # 1. Create asset_locks table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS asset_locks (
                symbol TEXT PRIMARY KEY,
                locked_by TEXT NOT NULL,
                locked_at TEXT NOT NULL,
                reason TEXT,
                heartbeat TEXT NOT NULL
            )
        """)
        print("Created/verified table: asset_locks")

        # 2. Create swing_open_positions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS swing_open_positions (
                symbol TEXT PRIMARY KEY,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                size REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                trailing_stop REAL,
                peak_price REAL,
                leverage REAL DEFAULT 3.0,
                opened_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT DEFAULT 'ACTIVE',
                r_multiple REAL DEFAULT 0.0,
                initial_risk_usd REAL,
                notes TEXT
            )
        """)
        print("Created/verified table: swing_open_positions")

        # 3. Create swing_trades (history) table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS swing_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                size REAL NOT NULL,
                pnl REAL NOT NULL,
                roe_pct REAL NOT NULL,
                holding_hours REAL,
                exit_reason TEXT,
                notes TEXT
            )
        """)
        print("Created/verified table: swing_trades")

    # Verify tables
    tables = [t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print("All tables in DB:", [t for t in tables if 'swing' in t or 'lock' in t])

    # Check MARL position to ensure 100% untouched
    marl_pos = conn.execute("SELECT symbol, side, entry_price, qty FROM marl_live_open_positions").fetchall()
    print("MARL live open positions (Verification):", marl_pos)

    conn.close()
    print("Migration completed successfully.")

if __name__ == "__main__":
    run_migration()

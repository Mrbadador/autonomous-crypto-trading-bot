import sqlite3
import os

SOURCE_DB = "live_paper_trades.db"
DEST_DB = "trading_data.db"

def migrate():
    if not os.path.exists(SOURCE_DB):
        print(f"Source {SOURCE_DB} not found. Skipping migration.")
        return

    print(f"Starting migration from {SOURCE_DB} to {DEST_DB}...")
    
    src_conn = sqlite3.connect(SOURCE_DB)
    dest_conn = sqlite3.connect(DEST_DB)
    
    # 1. Standard Tables
    tables_to_migrate = [
        "open_positions", 
        "berserker_open_positions",
        "berserker_trades",
        "system_state",
        "telemetry",
        "berserker_telemetry"
    ]
    
    for table in tables_to_migrate:
        try:
            src_cursor = src_conn.execute(f"SELECT * FROM {table}")
            rows = src_cursor.fetchall()
            if not rows: continue
            
            cursor = src_conn.execute(f"PRAGMA table_info({table})")
            cols = [c[1] for c in cursor.fetchall()]
            col_str = ", ".join(cols)
            placeholders = ", ".join(["?"] * len(cols))
            
            cmd = "REPLACE" if table == "system_state" else "INSERT OR IGNORE"
            dest_conn.executemany(f"{cmd} INTO {table} ({col_str}) VALUES ({placeholders})", rows)
            dest_conn.commit()
            print(f"    Migrated {len(rows)} rows into {table}.")
        except Exception as e:
            print(f"    Skipped {table}: {e}")
            
    # 2. Special handling for 'trades' table due to schema divergence
    print("  Migrating trades (Special Mapping)...")
    try:
        src_cursor = src_conn.execute("SELECT * FROM trades")
        rows = src_cursor.fetchall()
        src_cols = [c[1] for c in src_conn.execute("PRAGMA table_info(trades)").fetchall()]
        
        for row in rows:
            data = dict(zip(src_cols, row))
            # Map columns from LivePaperExchange schema to Centralized Dashboard schema
            mapped_data = (
                data.get('timestamp'),
                data.get('symbol'),
                data.get('action'), # mapped to 'type'
                data.get('side'),
                data.get('entry_price'),
                data.get('exit_price'),
                data.get('qty'),
                data.get('pnl'),
                data.get('balance_after'), # mapped to 'balance'
                data.get('regime'),
                data.get('phase'),
                data.get('quality'),
                data.get('notes'),
                'closed' # default status
            )
            dest_conn.execute("""
                INSERT OR IGNORE INTO trades 
                (timestamp, symbol, type, side, entry_price, exit_price, qty, pnl, balance, regime, phase, quality, notes, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, mapped_data)
        dest_conn.commit()
        print(f"    Migrated {len(rows)} rows into trades.")
    except Exception as e:
        print(f"    Failed to migrate trades: {e}")
            
    src_conn.close()
    dest_conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()

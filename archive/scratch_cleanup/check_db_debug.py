import sqlite3
import os

db_path = "trading_data.db"
if not os.path.exists(db_path):
    print(f"Database {db_path} not found.")
else:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall()]
    print(f"Tables: {tables}")
    
    for table in tables:
        if "trades" in table or "positions" in table:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"Table {table}: {count} rows")
            if count > 0:
                # Check for OPEN trades
                try:
                    cursor.execute(f"PRAGMA table_info({table})")
                    cols = [c[1] for c in cursor.fetchall()]
                    if "status" in cols:
                        cursor.execute(f"SELECT symbol, status FROM {table} WHERE status != 'closed'")
                        open_trades = cursor.fetchall()
                        print(f"  Open/Active trades in {table}: {open_trades}")
                except:
                    pass
    conn.close()

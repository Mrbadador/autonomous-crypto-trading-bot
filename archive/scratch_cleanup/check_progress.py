import sqlite3
import os

db_path = 'trading_data.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    count = conn.execute('SELECT COUNT(*) FROM trades').fetchone()[0]
    cycle_count = conn.execute('SELECT COUNT(*) FROM cycles').fetchone()[0]
    print(f"Trades: {count}")
    print(f"Cycles: {cycle_count}")
    conn.close()
else:
    print("DB not found")

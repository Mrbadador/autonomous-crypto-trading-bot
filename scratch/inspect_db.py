import sqlite3
import os

db_path = 'live_paper_trades.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables in live_paper_trades.db:")
    for table in tables:
        print(f" - {table[0]}")
    conn.close()
else:
    print(f"{db_path} does not exist.")

db_path = 'trading_data.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("\nTables in trading_data.db:")
    for table in tables:
        print(f" - {table[0]}")
    conn.close()
else:
    print(f"\n{db_path} does not exist.")

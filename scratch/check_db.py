import sqlite3
import os

DB_PATH = "trading_ledger.db"

if not os.path.exists(DB_PATH):
    print(f"Database {DB_PATH} not found.")
else:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check oracle_microstructure schema
    cursor.execute("PRAGMA table_info(oracle_microstructure);")
    columns = cursor.fetchall()
    print("Columns in oracle_microstructure:")
    for col in columns:
        print(col)
    
    # Check latest data
    cursor.execute("SELECT * FROM oracle_microstructure ORDER BY timestamp DESC LIMIT 1;")
    row = cursor.fetchone()
    print("\nLatest row in oracle_microstructure:")
    print(row)
    
    conn.close()

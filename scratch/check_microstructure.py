import sqlite3
try:
    conn = sqlite3.connect("trading_ledger.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM oracle_microstructure LIMIT 5;")
    rows = cursor.fetchall()
    print(f"Found {len(rows)} rows in oracle_microstructure.")
    for row in rows:
        print(row)
    conn.close()
except Exception as e:
    print(f"Error: {e}")

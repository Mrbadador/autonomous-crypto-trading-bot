import sqlite3
db = "trading_data.db"
conn = sqlite3.connect(db)
cur = conn.cursor()
print("--- legacy_trades ---")
try:
    res = cur.execute("PRAGMA table_info(legacy_trades)").fetchall()
    for r in res: print(r)
except Exception as e: print(e)
conn.close()

import sqlite3
import os

db = "trading_data.db"
if not os.path.exists(db):
    print("DB not found")
    exit(1)

conn = sqlite3.connect(db)
cur = conn.cursor()

tables = ["system_state", "trades", "berserker_trades"]
for t in tables:
    print(f"--- {t} ---")
    try:
        res = cur.execute(f"PRAGMA table_info({t})").fetchall()
        for r in res:
            print(r)
    except Exception as e:
        print(e)

conn.close()

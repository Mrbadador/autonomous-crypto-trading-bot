# scratch/inspect_db_rows.py
import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dbs = ['trading_data.db', 'trading_ledger.db']

for db in dbs:
    p = os.path.join(BASE_DIR, db)
    if not os.path.exists(p):
        print(f"{db} does not exist.")
        continue
    
    print(f"=== {db} ===")
    conn = sqlite3.connect(p)
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for t in tables:
            try:
                cnt = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                print(f"  {t}: {cnt}")
            except Exception as e:
                print(f"  {t}: Error {e}")
    finally:
        conn.close()

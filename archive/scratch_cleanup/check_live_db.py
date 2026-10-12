import sqlite3
db = "live_paper_trades.db"
conn = sqlite3.connect(db)
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [r[0] for r in cursor.fetchall()]
print(f"Tables in {db}: {tables}")
for table in tables:
    if "trades" in table or "positions" in table:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        print(f"Table {table}: {cursor.fetchone()[0]} rows")
conn.close()

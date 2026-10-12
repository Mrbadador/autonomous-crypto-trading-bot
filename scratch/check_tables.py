import sqlite3
conn = sqlite3.connect("trading_data.db")
c = conn.cursor()
tables = [row[0] for row in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
for t in tables:
    cols = [col[1] for col in c.execute(f"PRAGMA table_info({t})").fetchall()]
    print(f"Table: {t}")
    print(f"Columns: {cols}\n")
conn.close()

import sqlite3
conn = sqlite3.connect('trading_ledger.db')
cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print(f"Tables in DB: {tables}")
for table in tables:
    print(f"\nSchema for table {table}:")
    cursor = conn.execute(f"PRAGMA table_info({table})")
    for col in cursor.fetchall():
        print(col)
conn.close()

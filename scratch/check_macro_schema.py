import sqlite3
conn = sqlite3.connect("trading_ledger.db")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print(f"Tables: {tables}")
for t in tables:
    t_name = t[0]
    if "macro" in t_name.lower():
        cursor.execute(f"PRAGMA table_info({t_name})")
        print(f"\nTable Info for {t_name}:")
        for row in cursor.fetchall():
            print(row)
        print("Sample Row:")
        cursor.execute(f"SELECT * FROM {t_name} ORDER BY id DESC LIMIT 1")
        print(cursor.fetchone())
conn.close()

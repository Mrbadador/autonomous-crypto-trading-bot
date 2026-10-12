import sqlite3
conn = sqlite3.connect("trading_data.db")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print(f"Tables in trading_data.db: {tables}")
for t in tables:
    t_name = t[0]
    if "position" in t_name.lower():
        cursor.execute(f"PRAGMA table_info({t_name})")
        print(f"\nTable Info for {t_name}:")
        for row in cursor.fetchall():
            print(row)
        print("Sample Row:")
        cursor.execute(f"SELECT * FROM {t_name} LIMIT 1")
        print(cursor.fetchone())
conn.close()

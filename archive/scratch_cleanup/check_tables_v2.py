import sqlite3
conn = sqlite3.connect("trading_data.db")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print(f"Tables: {[r[0] for r in cursor.fetchall()]}")
conn.close()

import sqlite3
import json

conn = sqlite3.connect('trading_data.db')
row = conn.execute("SELECT value FROM system_state WHERE key = 'ALLOWED_SYMBOLS'").fetchone()
print("ALLOWED_SYMBOLS:", row[0] if row else "None")

rows = conn.execute("SELECT key, value FROM system_state WHERE key LIKE '%symbol%' OR key LIKE '%fleet%' OR key LIKE '%pause%'").fetchall()
for r in rows:
    print(r)
conn.close()

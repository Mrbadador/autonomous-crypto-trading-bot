import sqlite3
conn = sqlite3.connect("trading_data.db")
cursor = conn.cursor()
cursor.execute("SELECT id, timestamp, symbol, side, status FROM trades ORDER BY id DESC LIMIT 5")
rows = cursor.fetchall()
for r in rows:
    print(r)
conn.close()

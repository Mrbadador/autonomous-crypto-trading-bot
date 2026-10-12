import sqlite3
conn = sqlite3.connect('trading_data.db')
cur = conn.cursor()
print("Samples from trades/legacy_trades:")
res = cur.execute("SELECT notes, type, regime FROM trades LIMIT 5").fetchall()
for r in res: print("TRADE:", r)
res = cur.execute("SELECT notes, type, regime FROM legacy_trades LIMIT 5").fetchall()
for r in res: print("LEGACY:", r)
conn.close()

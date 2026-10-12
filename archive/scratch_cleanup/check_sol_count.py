import sqlite3
conn = sqlite3.connect("trading_data.db")
cursor = conn.cursor()
cursor.execute("SELECT count(*) FROM trades WHERE symbol LIKE '%SOL%'")
print(f"SOL trades count in 'trades': {cursor.fetchone()[0]}")
cursor.execute("SELECT count(*) FROM berserker_trades WHERE symbol LIKE '%SOL%'")
print(f"SOL trades count in 'berserker_trades': {cursor.fetchone()[0]}")
conn.close()

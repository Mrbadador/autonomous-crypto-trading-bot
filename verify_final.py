import sqlite3
conn = sqlite3.connect("trading_data.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM berserker_open_positions")
print(f"Open Berserker Positions: {cursor.fetchall()}")
conn.close()

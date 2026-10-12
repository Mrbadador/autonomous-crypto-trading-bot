import sqlite3
conn=sqlite3.connect('trading_data.db')
try:
 print('open_positions:', conn.execute('SELECT * FROM open_positions').fetchall())
except Exception as e: print(e)

try:
 rows=conn.execute('SELECT key, value FROM system_state WHERE key LIKE '%risk%' OR key LIKE '%kill%' OR key LIKE '%balance%'').fetchall()
 for r in rows: print(r)
except Exception as e: print(e)

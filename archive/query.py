import sqlite3
conn=sqlite3.connect('trading_data.db')
conn.row_factory=sqlite3.Row
rows=conn.execute('SELECT symbol,side,pnl,action,notes,timestamp FROM trades WHERE status=''closed'' ORDER BY id DESC LIMIT 5').fetchall()
for r in rows:
    print(dict(r))

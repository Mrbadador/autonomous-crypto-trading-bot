import sqlite3

conn = sqlite3.connect('trading_data.db')
conn.row_factory = sqlite3.Row

print("=== Trades by Action ===")
rows = conn.execute('''
    SELECT action, COUNT(1) as cnt, 
           SUM(CAST(pnl AS REAL)) as total_pnl, 
           AVG(CAST(pnl AS REAL)) as avg_pnl, 
           SUM(CASE WHEN CAST(pnl AS REAL) > 0 THEN 1 ELSE 0 END)*100.0/COUNT(1) as wr 
    FROM trades 
    GROUP BY action
''').fetchall()
for r in rows:
    print(dict(r))

print("\n=== Swing Trades ===")
s_rows = conn.execute('''
    SELECT COUNT(1) as cnt,
           SUM(CAST(pnl AS REAL)) as total_pnl,
           AVG(CAST(pnl AS REAL)) as avg_pnl,
           SUM(CASE WHEN CAST(pnl AS REAL) > 0 THEN 1 ELSE 0 END)*100.0/COUNT(1) as wr
    FROM swing_trades
''').fetchall()
for r in s_rows:
    print(dict(r))

print("\n=== Trades by Symbol (Top 10) ===")
sym_rows = conn.execute('''
    SELECT symbol, COUNT(1) as cnt,
           SUM(CAST(pnl AS REAL)) as total_pnl,
           SUM(CASE WHEN CAST(pnl AS REAL) > 0 THEN 1 ELSE 0 END)*100.0/COUNT(1) as wr
    FROM trades
    GROUP BY symbol
    ORDER BY total_pnl DESC
    LIMIT 10
''').fetchall()
for r in sym_rows:
    print(dict(r))

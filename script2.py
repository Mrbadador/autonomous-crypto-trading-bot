import sqlite3
with sqlite3.connect('trading_ledger.db') as conn:
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    print('=== ALL TRADE LESSONS ===')
    lessons = c.execute('SELECT bot_type, COUNT(*) as n, AVG(pnl) as avg_pnl, SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) as wins FROM trade_lessons GROUP BY bot_type').fetchall()
    for l in lessons: print(dict(l))
    print('\n=== RECENT LESSONS ===')
    recent = c.execute('SELECT * FROM trade_lessons ORDER BY id DESC LIMIT 10').fetchall()
    for r in recent: print(f'[{r["timestamp"]}] {r["bot_type"]} PnL=${r["pnl"]:+.2f}: {r["lesson"][:100]}')

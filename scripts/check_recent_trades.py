import sqlite3

print("=== RECENT CLOSED SWING TRADES (Last 10) ===")
c = sqlite3.connect('trading_data.db')
c.row_factory = sqlite3.Row
trades = c.execute('SELECT id, timestamp, symbol, side, entry_price, exit_price, size, pnl, roe_pct, holding_hours, exit_reason, notes FROM swing_trades ORDER BY id DESC LIMIT 10').fetchall()
for t in trades:
    print(f"#{t['id']} | {t['timestamp'][:19]} | {t['symbol']:10} | {t['side']:5} | Entry: {t['entry_price']} | Exit: {t['exit_price']} | PnL: ${t['pnl']:+.4f} ({t['roe_pct']:+.2f}%) | Reason: {t['exit_reason']}")
c.close()

print("\n=== RECENT TRADE LESSONS (trading_ledger.db) ===")
c2 = sqlite3.connect('trading_ledger.db')
c2.row_factory = sqlite3.Row
lessons = c2.execute('SELECT id, trade_id, bot_type, symbol, pnl, lesson, timestamp FROM trade_lessons ORDER BY id DESC LIMIT 8').fetchall()
for l in lessons:
    print(f"Lesson #{l['id']} (Trade {l['trade_id']} {l['symbol']}, PnL: ${l['pnl']:+.2f}): {l['lesson']}")

print("\n=== LATEST LEARNED RULES ADDED ===")
rules = c2.execute('SELECT rule_id, category, rule_text, confidence, created_at FROM learned_rules ORDER BY rule_id DESC LIMIT 8').fetchall()
for r in rules:
    print(f"Rule #{r['rule_id']} [{r['category']}] (Conf: {r['confidence']:.0%}): {r['rule_text']}")
c2.close()

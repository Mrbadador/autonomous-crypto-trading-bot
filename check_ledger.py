import sqlite3

conn = sqlite3.connect("trading_ledger.db")
c = conn.cursor()

print("--- trade_lessons counts by bot_type ---")
for r in c.execute("SELECT bot_type, count(*) FROM trade_lessons GROUP BY bot_type").fetchall():
    print(r)

print("\n--- swing lessons in trade_lessons ---")
for r in c.execute("SELECT trade_id, symbol, pnl, lesson, timestamp FROM trade_lessons WHERE bot_type = 'swing'").fetchall():
    print(f"Trade #{r[0]} | {r[1]} | PnL: ${r[2]:+.2f} | Lesson: {r[3][:120]}... | Time: {r[4]}")

print("\n--- learned_rules summary ---")
for r in c.execute("SELECT rule_id, category, rule_text, confidence FROM learned_rules ORDER BY rule_id DESC LIMIT 15").fetchall():
    print(f"Rule #{r[0]} [{r[1]}] (Conf: {r[3]:.0%}): {r[2]}")

conn.close()

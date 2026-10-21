import sqlite3

conn = sqlite3.connect("trading_data.db")
conn.row_factory = sqlite3.Row
print("--- SCHEMA ---")
for r in conn.execute("SELECT sql FROM sqlite_master WHERE name='swing_trades'").fetchall():
    print(r[0])
print("\n--- RECENT TRADES ---")
for r in conn.execute("SELECT symbol, side, entry_price, exit_price, pnl, fee_paid_usd, fee_maker_usd, fee_taker_usd, exit_reason, notes, timestamp FROM swing_trades ORDER BY id DESC LIMIT 5").fetchall():
    print(dict(r))

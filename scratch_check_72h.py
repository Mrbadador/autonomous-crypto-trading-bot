import sqlite3
conn = sqlite3.connect('trading_data.db')
conn.row_factory = sqlite3.Row

print("=== Trades in Last 72h (Sept 23-26) ===")
rows = conn.execute("SELECT timestamp, symbol, side, pnl, action FROM trades WHERE timestamp >= '2026-09-23' ORDER BY timestamp DESC").fetchall()
print(f"Total trades: {len(rows)}")
wins = [r for r in rows if float(r['pnl'] or 0) > 0]
losses = [r for r in rows if float(r['pnl'] or 0) < 0]
net_pnl = sum(float(r['pnl'] or 0) for r in rows)
print(f"Wins: {len(wins)}, Losses: {len(losses)}, Net PnL: ${net_pnl:+.4f}")
for r in rows[:25]:
    print(f"{r['timestamp'][:16]} | {r['symbol']:10} | {r['side'] or 'N/A':5} | ${float(r['pnl'] or 0):+7.4f} | {r['action']}")

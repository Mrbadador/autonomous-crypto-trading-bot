import sqlite3
import json
import os

conn = sqlite3.connect('trading_data.db')
conn.row_factory = sqlite3.Row

tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print('=== TABLES IN TRADING_DATA.DB ===')
print(tables)

if 'positions' in tables:
    print('=== OPEN POSITIONS ===')
    open_pos = conn.execute("SELECT * FROM positions WHERE status='open'").fetchall()
    print([dict(p) for p in open_pos] if open_pos else 'No open positions.')


print('\n=== SWING TRADES TODAY (2026-10-07) ===')
today_swings = conn.execute("SELECT * FROM swing_trades WHERE timestamp LIKE '2026-10-07%' ORDER BY id DESC").fetchall()
if today_swings:
    for t in today_swings:
        print(dict(t))
else:
    print('No swing trades closed today.')

print('\n=== TRADES TABLE TODAY (2026-10-07) ===')
today_trades = conn.execute("SELECT * FROM trades WHERE timestamp LIKE '2026-10-07%' ORDER BY id DESC").fetchall()
if today_trades:
    for t in today_trades:
        print(dict(t))
else:
    print('No trades in trades table today.')

print('\n=== RECENT CLOSED SWING TRADES (LAST 5) ===')
recent_swings = conn.execute("SELECT id, timestamp, symbol, side, entry_price, exit_price, pnl, roe_pct, exit_reason FROM swing_trades ORDER BY id DESC LIMIT 5").fetchall()
for t in recent_swings:
    print(dict(t))

print('\n=== SYSTEM STATE SNAPSHOT ===')
state = dict(conn.execute("SELECT key, value FROM system_state WHERE key IN ('balance', 'wallet_balance', 'equity', 'current_drawdown', 'hwm_balance', 'macro_regime', 'market_regime', 'active_symbols', 'fleet_universe')").fetchall())
for k, v in state.items():
    print(f'{k}: {v}')

if os.path.exists('microstructure_state.json'):
    with open('microstructure_state.json') as f:
        data = json.load(f)
        print('\n=== MICROSTRUCTURE STATE (Sample 4 assets) ===')
        for k in list(data.keys())[:4]:
            print(k, data[k])

conn.close()

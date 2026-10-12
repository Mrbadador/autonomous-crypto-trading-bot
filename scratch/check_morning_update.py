import sqlite3
import json
import time
import urllib.request

conn = sqlite3.connect('trading_data.db')
conn.row_factory = sqlite3.Row

print("=== RECENT TRADES (LAST 10) ===")
rows = conn.execute("SELECT id, timestamp, symbol, side, entry_price, exit_price, size, pnl, roe_pct, exit_reason FROM swing_trades ORDER BY id DESC LIMIT 10").fetchall()
for r in rows:
    print(dict(r))

print("\n=== TRADES CLOSED TODAY (2026-10-08) ===")
today_trades = conn.execute("SELECT * FROM swing_trades WHERE timestamp LIKE '2026-10-08%'").fetchall()
print(f"Count: {len(today_trades)}")
for r in today_trades:
    print(dict(r))

print("\n=== OPEN POSITIONS ===")
open_swings = conn.execute("SELECT * FROM swing_open_positions").fetchall()
print(f"Swing Open Positions: {len(open_swings)}")
for r in open_swings:
    print(dict(r))

open_pos = conn.execute("SELECT * FROM open_positions").fetchall()
print(f"Standard Open Positions: {len(open_pos)}")
for r in open_pos:
    print(dict(r))

print("\n=== SYSTEM STATE ===")
state = dict(conn.execute("SELECT key, value FROM system_state WHERE key IN ('balance', 'equity', 'hwm_balance', 'current_drawdown', 'macro_regime', 'market_regime', 'fleet_universe')").fetchall())
for k, v in state.items():
    print(f"{k}: {v}")

conn.close()

# Check Streamlit
try:
    resp = urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5)
    print(f"\nStreamlit Health: HTTP {resp.status}")
except Exception as e:
    print(f"\nStreamlit Health Error: {e}")

# Check Microstructure
try:
    with open('microstructure_state.json') as f:
        mstate = json.load(f)
        age = time.time() - mstate.get('timestamp', 0)
        sym_count = len(mstate.get('symbols', {}))
        print(f"Microstructure State: {sym_count} symbols, age={age:.1f}s")
except Exception as e:
    print(f"Microstructure State Error: {e}")

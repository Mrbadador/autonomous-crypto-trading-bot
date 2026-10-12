import sqlite3
import json
import os
import time

print("=== 1. MICROSTRUCTURE STATE (OFI DAEMON) ===")
if os.path.exists("microstructure_state.json"):
    mtime = os.path.getmtime("microstructure_state.json")
    print(f"File updated: {time.time() - mtime:.1f}s ago")
    try:
        with open("microstructure_state.json", "r") as f:
            data = json.load(f)
        symbols_data = data.get("symbols", {})
        for sym, d in sorted(symbols_data.items()):
            print(f"{sym:10} | MLOFI: {d.get('mlofi', 0):+9.4f} | VPIN: {d.get('vpin', 0):.4f} | Micro-Price: {d.get('micro_price', 0):10.4f} ({d.get('micro_price_bps', 0):+5.2f} bps)")
    except Exception as e:
        print(f"Error reading microstructure: {e}")
else:
    print("microstructure_state.json NOT FOUND!")

print("\n=== 2. DATABASE TABLES ===")
for db_name in ["trading_data.db", "trading_ledger.db"]:
    if os.path.exists(db_name):
        conn = sqlite3.connect(db_name)
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        print(f"{db_name} tables: {tables}")
        conn.close()

print("\n=== 3. ACTIVE / OPEN POSITIONS ===")
conn = sqlite3.connect("trading_data.db")
conn.row_factory = sqlite3.Row
# Check swing_trades or trades where status is open
try:
    swing_open = conn.execute("SELECT * FROM swing_trades WHERE exit_time IS NULL OR exit_reason IS NULL").fetchall()
    print(f"Open swing trades: {len(swing_open)}")
    for s in swing_open:
        print(dict(s))
except Exception as e:
    print(f"Error checking swing_trades: {e}")

try:
    trades_open = conn.execute("SELECT * FROM trades WHERE status='open'").fetchall()
    print(f"Open live trades: {len(trades_open)}")
    for t in trades_open:
        print(dict(t))
except Exception as e:
    print(f"Error checking trades: {e}")

print("\n=== 4. RECENT CLOSED SWING TRADES (LAST 5) ===")
try:
    recent_swings = conn.execute("SELECT id, symbol, side, entry_price, exit_price, pnl, exit_reason, exit_time FROM swing_trades ORDER BY id DESC LIMIT 5").fetchall()
    for s in recent_swings:
        print(dict(s))
except Exception as e:
    print(f"Error checking recent swing trades: {e}")

print("\n=== 5. SYSTEM STATE FLAGS ===")
flags = conn.execute("""
    SELECT key, value, updated_at 
    FROM system_state 
    WHERE key IN ('bot_status', 'market_regime', 'wallet_balance', 'peak_balance', 'current_drawdown', 'fleet_relative_strength', 'executive_briefing', 'last_scan_time')
""").fetchall()
for f in flags:
    val = str(f['value'])
    if len(val) > 120:
        val = val[:120] + "..."
    print(f"{f['key']}: {val} (at {f['updated_at']})")

conn.close()

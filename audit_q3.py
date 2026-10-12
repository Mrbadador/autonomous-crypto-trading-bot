import sqlite3
import sys
import os
import time

# Force UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

print("=== SWING TRADES SCHEMA ===")
conn = sqlite3.connect("trading_data.db")
try:
    cols = conn.execute("PRAGMA table_info(swing_trades)").fetchall()
    for c in cols:
        print(c)
    rows = conn.execute("SELECT * FROM swing_trades ORDER BY rowid DESC LIMIT 10").fetchall()
    for r in rows:
        print(r)
except Exception as e:
    print(f"swing_trades error: {e}")

print("\n=== MARL_LIVE_TRADES RECENT ===")
try:
    conn.row_factory = sqlite3.Row
    rows2 = conn.execute("SELECT timestamp, symbol, side, type, pnl, status, notes FROM marl_live_trades ORDER BY timestamp DESC LIMIT 15").fetchall()
    for r in rows2:
        print(dict(r))
except Exception as e:
    print(f"marl_live_trades error: {e}")

print("\n=== SYSTEM_STATE KEY SEARCH: training, gate, halt, drawdown ===")
conn.row_factory = None
search_terms = ["training", "gate", "halt", "drawdown", "hwm", "peak", "session"]
for term in search_terms:
    rows3 = conn.execute("SELECT key, SUBSTR(value,1,80), updated_at FROM system_state WHERE key LIKE ?", (f"%{term}%",)).fetchall()
    for r in rows3:
        print(r)

conn.close()

print("\n=== LEARNED RULES SCHEMA ===")
conn2 = sqlite3.connect("trading_ledger.db")
try:
    cols2 = conn2.execute("PRAGMA table_info(learned_rules)").fetchall()
    for c in cols2:
        print(c)
    cnt = conn2.execute("SELECT COUNT(*) FROM learned_rules").fetchone()
    print(f"Total count: {cnt[0]}")
    rows4 = conn2.execute("SELECT * FROM learned_rules ORDER BY confidence DESC LIMIT 5").fetchall()
    for r in rows4:
        # Encode to ascii ignoring unknown to avoid cp1252 crash
        safe = str(r).encode("ascii","replace").decode("ascii")
        print(safe)
except Exception as e:
    print(f"learned_rules error: {e}")
conn2.close()

print("\n=== POLICY CHECKPOINT FILES ===")
for fname in ["strategist_weights_v28.pth","strategist_weights_v27.pth","strategist_weights_v26.pth","policy_model.pth"]:
    if os.path.exists(fname):
        mtime = os.path.getmtime(fname)
        print(f"{fname}: mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))}, size={os.path.getsize(fname)} bytes")
    else:
        print(f"{fname}: NOT FOUND")

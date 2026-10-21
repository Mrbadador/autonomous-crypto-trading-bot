import sqlite3

conn = sqlite3.connect("file:trading_data.db?mode=ro", uri=True)
conn.row_factory = sqlite3.Row
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print("Tables in trading_data.db:", tables)
if "swing_open_positions" in tables:
    swing_positions = conn.execute("SELECT * FROM swing_open_positions WHERE status='open'").fetchall()
    print(f"swing_open_positions ({len(swing_positions)}): {[dict(r) for r in swing_positions]}")

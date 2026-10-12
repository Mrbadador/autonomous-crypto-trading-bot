import sqlite3
import datetime

conn = sqlite3.connect('trading_data.db', timeout=15.0)
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
updates = [
    ('GLOBAL_MARGIN_LIMIT', '70.0'),
    ('MAX_PORTFOLIO_MARGIN_PCT', '0.70'),
    ('MAX_MARGIN_PER_TRADE_PCT', '0.20'),
    ('SWING_BALANCE_PCT', '0.175'),
    ('SWING_TRADE_ALLOCATION_USDT', '26.5'),
    ('GRID_ALLOCATION_USDT', '26.5')
]
for k, v in updates:
    conn.execute('''
        INSERT INTO system_state (key, value, updated_at) 
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
    ''', (k, v, now))
conn.commit()
print("Successfully updated system_state with 70% wallet utilization parameters:")

rows = conn.execute("""
    SELECT key, value, updated_at 
    FROM system_state 
    WHERE key IN ('GLOBAL_MARGIN_LIMIT', 'MAX_PORTFOLIO_MARGIN_PCT', 'MAX_MARGIN_PER_TRADE_PCT', 'SWING_BALANCE_PCT', 'SWING_TRADE_ALLOCATION_USDT', 'GRID_ALLOCATION_USDT')
""").fetchall()
for r in rows:
    print(f"  {r[0]}: {r[1]} ({r[2]})")
conn.close()

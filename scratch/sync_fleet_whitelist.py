import sqlite3
import json

fleet_12 = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT',
    'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'NEAR/USDT',
    'SUI/USDT', 'INJ/USDT', 'APT/USDT'
]

conn = sqlite3.connect('trading_data.db')
conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ALLOWED_SYMBOLS', ?, datetime('now'))", (json.dumps(fleet_12),))
conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('SYMBOLS', ?, datetime('now'))", (','.join(fleet_12),))
conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('asset_scout_fleet_size', ?, datetime('now'))", (str(len(fleet_12)),))
conn.commit()

row = conn.execute("SELECT value FROM system_state WHERE key = 'ALLOWED_SYMBOLS'").fetchone()
print("Updated ALLOWED_SYMBOLS in system_state:", row[0])
conn.close()

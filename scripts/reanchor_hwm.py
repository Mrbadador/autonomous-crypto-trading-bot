import sqlite3
with sqlite3.connect('trading_data.db') as conn:
    conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('global_peak_balance', '335.00', datetime('now'))")
    conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('hwm_balance', '335.00', datetime('now'))")
    conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('RISK_HALT', '0', datetime('now'))")
    conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('AUTO_RISK_HALTED', '0', datetime('now'))")
    conn.commit()
    print('Reset global_peak_balance and cleared RISK_HALT successfully')

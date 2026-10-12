import sqlite3

def seed_bot_parameters():
    conn = sqlite3.connect("trading_data.db")
    with conn:
        defaults = [
            ("sniper", "min_nn_confidence", 0.85),
            ("sniper", "vol_gate_mult", 1.5),
            ("berserker", "min_nn_confidence", 0.80),
            ("berserker", "vol_gate_mult", 1.2),
            ("berserker", "veto_threshold", 65.0)
        ]
        
        for bot_name, param_name, param_value in defaults:
            conn.execute("""
                INSERT OR IGNORE INTO bot_parameters (bot_name, param_name, param_value, last_updated)
                VALUES (?, ?, ?, datetime('now'))
            """, (bot_name, param_name, param_value))
            
    conn.close()
    print("Successfully seeded Sniper and Berserker parameters.")

if __name__ == "__main__":
    seed_bot_parameters()

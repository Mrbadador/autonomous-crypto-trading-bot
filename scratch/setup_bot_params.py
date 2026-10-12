import sqlite3

def setup_bot_parameters():
    conn = sqlite3.connect("trading_data.db")
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bot_parameters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_name TEXT,
                param_name TEXT,
                param_value REAL,
                last_updated TEXT,
                UNIQUE(bot_name, param_name)
            )
        """)
        
        # Seed Mean Reversion default parameters
        defaults = [
            ("mean_reversion", "rsi_buy", 30.0),
            ("mean_reversion", "rsi_sell", 70.0),
            ("mean_reversion", "take_profit_pct", 1.01),
            ("mean_reversion", "stop_loss_pct", 0.99)
        ]
        
        for bot_name, param_name, param_value in defaults:
            conn.execute("""
                INSERT OR IGNORE INTO bot_parameters (bot_name, param_name, param_value, last_updated)
                VALUES (?, ?, ?, datetime('now'))
            """, (bot_name, param_name, param_value))
            
    conn.close()
    print("Successfully initialized bot_parameters table and seeded default values.")

if __name__ == "__main__":
    setup_bot_parameters()

import sqlite3

def setup_mean_reversion_tables():
    with sqlite3.connect("trading_data.db") as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        c = conn.cursor()
        
        # 1. Trades table
        c.execute("""
            CREATE TABLE IF NOT EXISTS mean_reversion_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, symbol TEXT, side TEXT, type TEXT,
                entry_price REAL, exit_price REAL, qty REAL, pnl REAL,
                balance REAL, regime TEXT, phase TEXT, quality REAL, notes TEXT,
                status TEXT DEFAULT 'closed',
                ai_rationale TEXT,
                slippage_bps REAL DEFAULT 0.0,
                action TEXT
            )
        """)
        
        # 2. Open Positions table
        c.execute("""
            CREATE TABLE IF NOT EXISTS mean_reversion_open_positions (
                row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT, side TEXT, entry_price REAL, qty REAL, 
                take_profit REAL, stop_loss REAL, peak_price REAL, entry_time TEXT, 
                is_closing INTEGER DEFAULT 0, partial_tp_hit INTEGER DEFAULT 0, 
                action_confidence REAL, comments TEXT, leverage REAL
            )
        """)
        
        # 3. Telemetry table
        c.execute("""
            CREATE TABLE IF NOT EXISTS mean_reversion_telemetry (
                symbol TEXT PRIMARY KEY, timestamp TEXT, reasoning TEXT,
                price REAL, ema_ref REAL, vol_pct REAL, action TEXT, quality REAL
            )
        """)
        
        # 4. Starting Balance in system_state
        c.execute("""
            INSERT OR IGNORE INTO system_state (key, value) 
            VALUES ('balance_mean_reversion', '500.00')
        """)
        
        # Ensure it's exactly 500.00 if it exists but we want to reset it for paper trading
        c.execute("""
            UPDATE system_state SET value = '500.00' WHERE key = 'balance_mean_reversion' AND value IS NULL
        """)
        
        conn.commit()
        print("Successfully created mean reversion tables and set starting balance.")

if __name__ == "__main__":
    setup_mean_reversion_tables()

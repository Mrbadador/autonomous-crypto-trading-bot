import sqlite3
import asyncio
from post_mortem_agent import analyze_trades

def inject_fake_trade():
    conn = sqlite3.connect("trading_data.db")
    with conn:
        # Create table if not exists (in case it wasn't yet created by mean_reversion_trader)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mean_reversion_trades (
                timestamp TEXT,
                symbol TEXT,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                qty REAL,
                pnl REAL,
                balance REAL,
                status TEXT
            )
        """)
        
        # Inject a fake "bad" trade where stop loss triggered
        conn.execute("""
            INSERT INTO mean_reversion_trades 
            (timestamp, symbol, side, entry_price, exit_price, qty, pnl, balance, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("2026-07-01T12:00:00Z", "BTC/USDT", "LONG", 60000.0, 59400.0, 0.001, -0.6, 500.0, "closed"))
    conn.close()
    print("Injected fake BAD trade.")

async def main():
    inject_fake_trade()
    print("Running analyze_trades()...")
    await analyze_trades()
    
    # Check if parameters were updated
    conn = sqlite3.connect("trading_data.db")
    cur = conn.execute("SELECT param_name, param_value FROM bot_parameters WHERE bot_name = 'mean_reversion'")
    print("Updated Bot Parameters:")
    for row in cur.fetchall():
        print(row)
    conn.close()

if __name__ == "__main__":
    asyncio.run(main())

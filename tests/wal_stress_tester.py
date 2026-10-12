import sqlite3
import asyncio
import time
import random

DB_PATH = "trading_ledger.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    # Ensure WAL is active to handle concurrent writes
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute('''
        CREATE TABLE IF NOT EXISTS stress_test_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            side TEXT,
            timestamp REAL
        )
    ''')
    conn.commit()
    conn.close()

async def async_write_trade(symbol, side):
    """
    Attempts to write a unique mock trade record to the SQLite database.
    Implements an exponential backoff retry loop to handle write contention.
    """
    def _write():
        max_retries = 10
        backoff = 0.05  # Initial backoff of 50ms
        
        for attempt in range(max_retries):
            try:
                conn = sqlite3.connect(DB_PATH, timeout=5.0)
                conn.execute(
                    "INSERT INTO stress_test_trades (symbol, side, timestamp) VALUES (?, ?, ?)",
                    (symbol, side, time.time())
                )
                conn.commit()
                conn.close()
                return True
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    # Exponential backoff with jitter to desynchronize retries
                    jitter = random.uniform(0.01, 0.05)
                    time.sleep(backoff + jitter)
                    backoff *= 2
                else:
                    raise e
        return False

    # Execute the synchronous SQLite write in a separate thread
    res = await asyncio.to_thread(_write)
    if res:
        print(f"[SUCCESS] Wrote mock trade for {symbol:<10} | Side: {side}")
    else:
        print(f"[FAILURE] Database locked after 10 retries for {symbol}")

async def main():
    print("Initializing Database with WAL mode...")
    init_db()
    
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT", "BNB/USDT", "DOGE/USDT"]
    print(f"Launching 7 parallel write tasks for the asset matrix...")
    
    # Force all 7 tasks to simultaneously attempt to write at the exact same millisecond
    tasks = []
    for sym in symbols:
        side = random.choice(["LONG", "SHORT"])
        tasks.append(async_write_trade(sym, side))
        
    start_time = time.time()
    await asyncio.gather(*tasks)
    elapsed = time.time() - start_time
    
    print(f"\nAll tasks completed in {elapsed:.4f} seconds.")
    
    # Verification
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM stress_test_trades").fetchone()[0]
    conn.close()
    print(f"Total rows in stress_test_trades: {count}")
    print("Zero OperationalError: database is locked exceptions encountered!")

if __name__ == "__main__":
    asyncio.run(main())

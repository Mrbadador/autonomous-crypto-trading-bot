import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import ccxt
import pandas as pd
from datetime import datetime, UTC

def download_squadron_data(symbol, start_date="2019-01-01"):
    print(f"--- [ Squadron Data Harvester: {symbol} ] ---")
    
    # Initialize exchange (Binance for full history)
    exchange = ccxt.binance({"enableRateLimit": True})
    
    since = exchange.parse8601(f"{start_date}T00:00:00Z")
    all_candles = []
    
    # Check if file exists to avoid re-downloading if interrupted
    safe_name = symbol.replace("/", "_").lower()
    filepath = f"historical_{safe_name}.csv"
    if os.path.exists(filepath):
        print(f"File {filepath} already exists. Skipping.")
        return

    print(f"Fetching from {start_date}...")
    while since < exchange.milliseconds():
        try:
            # 1H candles
            candles = exchange.fetch_ohlcv(symbol, "1h", since=since, limit=1000)
            if not candles:
                break
            
            all_candles.extend(candles)
            since = candles[-1][0] + 1
            
            if len(all_candles) % 5000 == 0:
                print(f"  Progress: {len(all_candles)} candles...")
            
            if len(candles) < 200: # Reached present
                break
                
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(1) # Backoff
            
    if not all_candles:
        print(f"❌ No data for {symbol}.")
        return

    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df.to_csv(filepath, index=False)
    print(f"SUCCESS: {len(df)} candles saved to {filepath}")

if __name__ == "__main__":
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT"]
    for s in symbols:
        download_squadron_data(s)

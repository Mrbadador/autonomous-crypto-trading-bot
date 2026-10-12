import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import ccxt
import pandas as pd
from datetime import datetime, UTC, timedelta

def download_historical_data(symbol, timeframe, days=30):
    print(f"--- [ Historical Data Harvester ] ---")
    print(f"Symbol: {symbol} | Timeframe: {timeframe} | Depth: {days} days")
    
    # Ensure data directory exists
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    # Initialize exchange (Public V5)
    exchange = ccxt.bybit({
        "enableRateLimit": True,
        "options": {"defaultType": "future"}
    })
    
    since = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
    all_candles = []
    
    print("Fetching sequential buckets...")
    while since < exchange.milliseconds():
        try:
            # Bybit limit is 200/1000 depending on endpoint
            limit = 200 if timeframe == '1h' else 1000
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not candles:
                break
            
            all_candles.extend(candles)
            since = candles[-1][0] + 1 # Advance 'since' to the next ms
            print(f"  Fetched {len(all_candles)} candles so far...")
            
            if len(candles) < limit:
                break # Reached the present
                
        except Exception as e:
            print(f"  Error during fetch: {e}")
            break
            
    if not all_candles:
        print("❌ No data fetched. Check symbol and connectivity.")
        return

    # Convert to CSV
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    clean_sym = symbol.replace("/", "")
    filename = f"{clean_sym}_{timeframe}.csv"
    filepath = os.path.join(config.DATA_DIR, filename)
    
    df.to_csv(filepath, index=False)
    print(f"SUCCESS: {len(df)} candles saved to {filepath}")

if __name__ == "__main__":
    # Default to config symbols
    syms = config.SYMBOLS
    tf = config.TIMEFRAME
    
    for s in syms:
        download_historical_data(s, tf, days=15) # 15 days of 1h data is ~360 candles

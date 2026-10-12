import ccxt
import pandas as pd
from datetime import datetime, UTC
import os

def download_historical_data(symbol="BTC/USDT", timeframe="1h", limit=2000):
    print(f"Connecting to Binance to fetch {limit} candles for {symbol} ({timeframe})...")
    exchange = ccxt.binance()
    
    try:
        # Fetch OHLCV
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp to ISO format
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Save to CSV
        file_path = "historical_1h.csv"
        df.to_csv(file_path, index=False)
        print(f"SUCCESS: Data saved to {file_path}")
        print(f"Range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        
    except Exception as e:
        print(f"ERROR: Failed to download data: {e}")

if __name__ == "__main__":
    download_historical_data()

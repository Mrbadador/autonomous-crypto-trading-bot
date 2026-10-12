"""
download_swing_history.py — Downloads full 4H and 1D historical candle data from Binance.
"""
import os
import time
import ccxt
import pandas as pd
from datetime import datetime, timezone

symbols = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'LINK/USDT', 
    'AVAX/USDT', 'XRP/USDT', 'ADA/USDT', 'BNB/USDT', 'NEAR/USDT'
]

ex = ccxt.binance({'enableRateLimit': True})
os.makedirs('data/historical', exist_ok=True)

# 2.5 years of historical data: starting Jan 1, 2024
start_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

for sym in symbols:
    clean = sym.replace('/', '')
    file_4h = f'data/historical/{clean}_4h_full.csv'
    file_1d = f'data/historical/{clean}_1d_full.csv'
    
    # 1. Fetch 1D full
    all_1d = []
    since = start_ts
    while True:
        try:
            batch = ex.fetch_ohlcv(sym, '1d', since=since, limit=1000)
            if not batch:
                break
            all_1d.extend(batch)
            if len(batch) < 1000:
                break
            since = batch[-1][0] + (24 * 3600 * 1000)
            time.sleep(0.05)
        except Exception as e:
            print(f"Error fetching 1D {sym}: {e}")
            break
            
    if all_1d:
        df_1d = pd.DataFrame(all_1d, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_1d.drop_duplicates(subset=['timestamp'], inplace=True)
        df_1d.sort_values('timestamp', inplace=True)
        df_1d.to_csv(file_1d, index=False)
        print(f"{sym} 1D: {len(df_1d)} days saved.")

    # 2. Fetch 4H full
    all_4h = []
    since = start_ts
    while True:
        try:
            batch = ex.fetch_ohlcv(sym, '4h', since=since, limit=1000)
            if not batch:
                break
            all_4h.extend(batch)
            if len(batch) < 1000:
                break
            since = batch[-1][0] + (4 * 3600 * 1000)
            time.sleep(0.05)
        except Exception as e:
            print(f"Error fetching 4H {sym}: {e}")
            break
            
    if all_4h:
        df_4h = pd.DataFrame(all_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_4h.drop_duplicates(subset=['timestamp'], inplace=True)
        df_4h.sort_values('timestamp', inplace=True)
        df_4h.to_csv(file_4h, index=False)
        t_start = pd.to_datetime(df_4h['timestamp'].iloc[0], unit='ms')
        t_end = pd.to_datetime(df_4h['timestamp'].iloc[-1], unit='ms')
        print(f"{sym} 4H: {len(df_4h)} candles saved ({t_start} -> {t_end}).")

print("\nAll historical data downloaded successfully.")

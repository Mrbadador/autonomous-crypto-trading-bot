import ccxt
import pandas as pd
import os
from datetime import datetime, UTC

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
START_DATE = "2019-01-01T00:00:00Z"
END_DATE = "2024-12-31T23:59:59Z"
OUTPUT_FILE = "historical_1h.csv"

def download():
    print(f"============================================================")
    print(f"   MEGA DATA DOWNLOADER: {SYMBOL} {TIMEFRAME}")
    print(f"   Range: {START_DATE} -> {END_DATE}")
    print(f"============================================================")
    
    ex = ccxt.binance({"enableRateLimit": True})
    start_ts = int(datetime.strptime(START_DATE, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp() * 1000)
    end_ts = int(datetime.strptime(END_DATE, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp() * 1000)
    
    all_ohlcv = []
    current_ts = start_ts
    ms_per_tf = 3600000 # 1h
    
    try:
        while current_ts < end_ts:
            ohlcv = ex.fetch_ohlcv(SYMBOL, TIMEFRAME, since=current_ts, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            current_ts = ohlcv[-1][0] + ms_per_tf
            
            readable_ts = datetime.fromtimestamp(current_ts/1000, UTC).strftime('%Y-%m-%d %H:%M')
            print(f"  [PROGRESS] Fetched up to {readable_ts} ({len(all_ohlcv)} candles total)")
            
            if len(all_ohlcv) % 5000 == 0:
                # Save partial to avoid loss
                df_partial = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df_partial["timestamp"] = pd.to_datetime(df_partial["timestamp"], unit='ms', utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                df_partial.to_csv(OUTPUT_FILE, index=False)

        df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        # Format timestamp to ISO for the bot
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms', utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        df = df[pd.to_datetime(df["timestamp"]) <= pd.to_datetime(END_DATE)]
        
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"============================================================")
        print(f"   SUCCESS: Saved {len(df)} candles to {OUTPUT_FILE}")
        print(f"============================================================")
    except Exception as e:
        print(f"!!! FATAL DOWNLOAD ERROR: {e}")

if __name__ == "__main__":
    download()

import ccxt
import pandas as pd
import os
from datetime import datetime, UTC

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT"]
TIMEFRAMES = ["1h", "4h"]
START_DATE = "2023-04-01T00:00:00Z"
END_DATE = "2023-07-31T23:59:59Z"
DATA_DIR = "data/historical"

def download():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    ex = ccxt.binance({"enableRateLimit": True})
    start_ts = int(datetime.strptime(START_DATE, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp() * 1000)
    end_ts = int(datetime.strptime(END_DATE, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp() * 1000)
    
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            print(f"Downloading {symbol} {tf}...")
            all_ohlcv = []
            current_ts = start_ts
            
            # Ms per candle
            ms_per_tf = {"1h": 3600000, "4h": 14400000}[tf]
            
            while current_ts < end_ts:
                try:
                    ohlcv = ex.fetch_ohlcv(symbol, tf, since=current_ts, limit=1000)
                    if not ohlcv:
                        break
                    all_ohlcv.extend(ohlcv)
                    current_ts = ohlcv[-1][0] + ms_per_tf
                    print(f"  Fetched up to {datetime.fromtimestamp(current_ts/1000, UTC)}")
                except Exception as e:
                    print(f" Error fetching {symbol} {tf}: {e}")
                    break
                
            df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df = df[df["timestamp"] <= end_ts]
            
            clean_sym = symbol.replace("/", "")
            filename = f"{clean_sym}_{tf}.csv"
            filepath = os.path.join(DATA_DIR, filename)
            df.to_csv(filepath, index=False)
            print(f"Saved {len(df)} candles to {filepath}")

if __name__ == "__main__":
    download()

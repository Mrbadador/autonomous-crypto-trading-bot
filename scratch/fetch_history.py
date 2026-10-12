import requests
import pandas as pd
import time

def fetch_sol_history(symbol="SOLUSDT", interval="5m", limit=1000, total_candles=10000):
    print(f"[*] Downloading real {symbol} data via Binance Ingestion Pipeline...")
    url = "https://api.binance.com/api/v3/klines"
    
    all_data = []
    end_time = None
    
    iterations = total_candles // limit
    
    for i in range(iterations):
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if end_time:
            params["endTime"] = end_time - 1
            
        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            
            if not data or len(data) == 0:
                break
                
            all_data = data + all_data
            # Update endTime to oldest candle's timestamp in batch for reverse pagination
            end_time = data[0][0]
            
            print(f"    -> Captured batch {i+1}/{iterations} ({len(all_data)} candles so far)")
            time.sleep(0.5) # polite rate limit
            
        except Exception as e:
            print(f"[!] Request error: {e}")
            break
            
    if not all_data:
        print("[!] Fetch failed completely.")
        return
        
    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    
    # Cast numeric keys
    cols = ["open", "high", "low", "close", "volume"]
    df[cols] = df[cols].astype(float)
    
    # Clean index
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    out_file = "sol_history.csv"
    df.to_csv(out_file, index=False)
    print(f"[+] Successfully saved {len(df)} real SOL/USDT rows to '{out_file}'.")

if __name__ == "__main__":
    fetch_sol_history()

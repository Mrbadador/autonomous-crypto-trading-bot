import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import ccxt

def check_public_data():
    print("--- [ Bybit Public V5 Connectivity Check ] ---")
    print("Mode: Hardened Virtual Core (No API Keys)")
    
    try:
        # Initialize as a public client (No Keys)
        exchange = ccxt.bybit({
            "enableRateLimit": True,
            "options": {"defaultType": "future"}
        })
        
        print("Checking Public Ticker (BTC/USDT)...")
        ticker = exchange.fetch_ticker("BTC/USDT")
        print(f"Connection: ✅ SUCCESS")
        print(f"Live BTC Price: ${ticker.get('last', 'Unknown')}")
        
        print("\nChecking Public OHLCV (1h)...")
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe='1h', limit=5)
        if ohlcv:
            print(f"OHLCV Data: ✅ SUCCESS ({len(ohlcv)} candles fetched)")
        
    except Exception as e:
        print(f"Connection: ❌ FAILED")
        print(f"Error: {e}")

if __name__ == "__main__":
    check_public_data()

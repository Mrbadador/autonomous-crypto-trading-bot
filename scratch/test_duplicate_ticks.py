import asyncio
import sys
import os

# Add codebase to path
sys.path.append(r"c:\Users\user\.gemini\antigravity\scratch\llama_trading_bot")

from live_trader import LiveTrader

async def run_test():
    print("[TEST] Initializing LiveTrader for duplicate tick check...")
    trader = LiveTrader()
    
    # Setup mock dataframe for indicator execution without CCXT call
    import pandas as pd
    import numpy as np
    dummy_data = {
        "open": [50000.0]*250,
        "high": [50100.0]*250,
        "low": [49900.0]*250,
        "close": [50050.0]*250,
        "volume": [10.0]*250
    }
    dummy_df = pd.DataFrame(dummy_data, index=pd.date_range("2026-05-19 00:00:00", periods=250, freq="15min", tz="UTC"))
    from indicators import compute_indicators
    trader.dfs["BTC/USDT"] = compute_indicators(dummy_df)
    
    # 1. Dispatch first candle close tick
    kline_tick = {
        't': 1716160000000, # Unix ms timestamp
        'o': '50000.0',
        'h': '51000.0',
        'l': '49500.0',
        'c': '50500.0',
        'v': '12.5',
        'x': True
    }
    
    print("[TEST] Sending first close tick at timestamp 1716160000000...")
    # Wrap in try-except since handle_candle_close triggers full decision matrix which might fail on missing keys
    try:
        await trader.handle_candle_close("BTC/USDT", kline_tick)
    except Exception as e:
        print(f"[TEST] Decision Matrix ran but caught expected logic error during real trade dispatch: {e}")
        
    assert trader.last_candle_t["BTC/USDT"] == 1716160000000, "Failed to record last candle close timestamp!"
    
    # 2. Dispatch duplicate tick
    print("[TEST] Sending duplicate close tick at timestamp 1716160000000...")
    import logging
    # Capture log outputs or inspect last_candle_t
    trader.last_candle_t["BTC/USDT"] = 1716160000000
    
    # If the duplicate check works, it should return immediately before executing computation or decision matrix
    # Let's corrupt dfs["BTC/USDT"] to None; if it doesn't return, it will crash immediately with AttributeError
    trader.dfs["BTC/USDT"] = None
    try:
        await trader.handle_candle_close("BTC/USDT", kline_tick)
        print("[TEST] Successfully rejected duplicate tick without executing decision matrix!")
    except Exception as e:
        assert False, f"Duplicate check did NOT block execution! Crashed with: {e}"
        
    print("\n[SUCCESS] Concurrency Duplicate Kline close guard validation PASSED!")

if __name__ == "__main__":
    asyncio.run(run_test())

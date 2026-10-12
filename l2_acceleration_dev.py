import os
import json
import time
import asyncio
from collections import deque

async def run_l2_analytics():
    print("Initializing L2 Microstructure Acceleration Dev...")
    ofi_history = deque(maxlen=60) # keep history for velocity calc
    last_timestamp = 0
    
    while True:
        try:
            if os.path.exists("microstructure_state.json"):
                with open("microstructure_state.json", "r") as f:
                    data = json.load(f)
                
                ts = data.get("timestamp", 0)
                if ts > last_timestamp:
                    last_timestamp = ts
                    raw_ofi = data.get("net_ofi_score", 0.0)
                    best_bid = data.get("best_bid", 50000.0) # Safe fallbacks if not yet streaming
                    best_ask = data.get("best_ask", 50001.0)
                    
                    # Store for rolling 5s window
                    ofi_history.append((ts, raw_ofi))
                    
                    # Calculate OFI Velocity (5-second window)
                    cutoff = ts - 5.0
                    past_ofi = raw_ofi
                    for hist_ts, hist_ofi in ofi_history:
                        if hist_ts >= cutoff:
                            past_ofi = hist_ofi
                            break
                            
                    ofi_velocity = raw_ofi - past_ofi
                    
                    # Bid-Ask Spread Compression
                    spread = abs(best_ask - best_bid)
                    mid_price = (best_bid + best_ask) / 2.0
                    spread_compression = spread / mid_price if mid_price > 0 else 0.0
                    
                    # Determine momentum
                    momentum = "ACCELERATING" if abs(ofi_velocity) > 10.0 else "DECAYING"
                    
                    print(f"[{ts:.2f}] OFI: {raw_ofi:7.2f} | Vel: {ofi_velocity:7.2f} ({momentum}) | Spread: {spread_compression:.6f}")
                    
        except Exception as e:
            print(f"Error reading state: {e}")
            
        await asyncio.sleep(1.0)

if __name__ == "__main__":
    try:
        asyncio.run(run_l2_analytics())
    except KeyboardInterrupt:
        print("Shutting down L2 analytics.")

import asyncio
import websockets
import json
import time
from collections import deque
from datetime import datetime
import sys

# Binance Futures Partial Book Depth Stream for BTCUSDT (Top 5 levels every 100ms)
WS_URL = "wss://fstream.binance.com/ws/btcusdt@depth5@100ms"

class OFISensor:
    def __init__(self, window_seconds=60):
        self.window_seconds = window_seconds
        
        # Previous state trackers
        self.prev_bid_p = 0.0
        self.prev_bid_q = 0.0
        self.prev_ask_p = 0.0
        self.prev_ask_q = 0.0
        
        # Rolling window storage: tuples of (timestamp, ofi_value)
        self.ofi_history = deque()
        self.last_print_time = time.time()
        
    def process_tick(self, best_bid_p, best_bid_q, best_ask_p, best_ask_q):
        """
        Implements the Order Flow Imbalance (OFI) mathematical model.
        Tracks actual volume changes at the Best Bid and Best Ask.
        """
        # Calculate Bid Delta (Volume Added/Removed at best bid)
        if best_bid_p > self.prev_bid_p:
            # Price moved up: new best bid, all quantity is "added" volume
            delta_w = best_bid_q
        elif best_bid_p == self.prev_bid_p:
            # Price stable: change in quantity
            delta_w = best_bid_q - self.prev_bid_q
        else:
            # Price moved down: old best bid was completely removed
            delta_w = -self.prev_bid_q
            
        # Calculate Ask Delta (Volume Added/Removed at best ask)
        if best_ask_p < self.prev_ask_p:
            # Price moved down: new best ask, all quantity is "added" volume
            delta_v = best_ask_q
        elif best_ask_p == self.prev_ask_p:
            # Price stable: change in quantity
            delta_v = best_ask_q - self.prev_ask_q
        else:
            # Price moved up: old best ask was completely removed
            delta_v = -self.prev_ask_q
            
        # Calculate Tick OFI
        ofi = delta_w - delta_v
        
        # Update previous state
        self.prev_bid_p = best_bid_p
        self.prev_bid_q = best_bid_q
        self.prev_ask_p = best_ask_p
        self.prev_ask_q = best_ask_q
        
        # Add to rolling history
        now = time.time()
        self.ofi_history.append((now, ofi))
        
        # Prune old values from the 60-second rolling window
        while self.ofi_history and now - self.ofi_history[0][0] > self.window_seconds:
            self.ofi_history.popleft()
            
        # Print output every 5 seconds
        if now - self.last_print_time >= 5.0:
            self.last_print_time = now
            self.print_dashboard(best_bid_p, best_ask_p)

    def print_dashboard(self, best_bid_p, best_ask_p):
        """Generates clean terminal output with visual pressure indicators."""
        # Calculate Net OFI over the current window
        net_ofi = sum(val for _, val in self.ofi_history)
        
        # Visual Indicator Logic
        if net_ofi > 5.0:
            indicator = "[+++] (STRONG BUY PRESSURE)"
        elif net_ofi > 0.0:
            indicator = "[+] (BULLISH)"
        elif net_ofi < -5.0:
            indicator = "[---] (STRONG SELL PRESSURE)"
        elif net_ofi < 0.0:
            indicator = "[-] (BEARISH)"
        else:
            indicator = "[=] (NEUTRAL)"
            
        # Terminal Output
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] BTCUSDT L2 | Bid: {best_bid_p:.2f} | Ask: {best_ask_p:.2f} | 60s Net OFI Score: {net_ofi:>10.4f} {indicator}")


async def stream_level2_data():
    sensor = OFISensor(window_seconds=60)
    print("=========================================================")
    print("   *** INITIALIZING L2 MARKET MICROSTRUCTURE SENSOR ***")
    print("=========================================================")
    print(f"Connecting to Binance Futures: {WS_URL}")
    
    while True:
        try:
            async with websockets.connect(WS_URL) as ws:
                print("[OK] Connection Established. Listening for L2 Orderbook Data...\n")
                
                async for message in ws:
                    data = json.loads(message)
                    
                    # Ensure valid payload with Bids and Asks
                    if "b" in data and len(data["b"]) > 0 and "a" in data and len(data["a"]) > 0:
                        best_bid_p = float(data["b"][0][0])
                        best_bid_q = float(data["b"][0][1])
                        
                        best_ask_p = float(data["a"][0][0])
                        best_ask_q = float(data["a"][0][1])
                        
                        # Only process if we have initialized the previous state
                        if sensor.prev_bid_p == 0.0:
                            sensor.prev_bid_p = best_bid_p
                            sensor.prev_bid_q = best_bid_q
                            sensor.prev_ask_p = best_ask_p
                            sensor.prev_ask_q = best_ask_q
                            continue
                            
                        # Pass to the OFI Math Engine
                        sensor.process_tick(best_bid_p, best_bid_q, best_ask_p, best_ask_q)
                        
        except Exception as e:
            print(f"\n[WARN] WebSocket Disconnected: {e}. Reconnecting in 3 seconds...")
            await asyncio.sleep(3)

if __name__ == "__main__":
    try:
        asyncio.run(stream_level2_data())
    except KeyboardInterrupt:
        print("\n[STOP] OFI Sensor Terminated by User.")
        sys.exit(0)

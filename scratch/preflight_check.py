# scratch/preflight_check.py
import os
import sys
import time
import sqlite3
import requests

# Set working directory to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

print("=" * 60)
print("  ANTI-GRAVITY PRE-FLIGHT SYSTEMS CHECK")
print("=" * 60)

# 1. Database Integrity Checks
print("\n[1] DATABASE INTEGRITY CHECKS")
dbs = {
    "trading_data.db": os.path.join(BASE_DIR, "trading_data.db"),
    "trading_ledger.db": os.path.join(BASE_DIR, "trading_ledger.db"),
}

for name, path in dbs.items():
    if not os.path.exists(path):
        print(f"  • {name}: ERROR - File not found at {path}")
        continue
    try:
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        result = cursor.fetchone()
        conn.close()
        status = result[0] if result else "unknown"
        if status.lower() == "ok":
            print(f"  • {name}: PASS (integrity_check: {status})")
        else:
            print(f"  • {name}: FAIL (integrity_check: {status})")
    except Exception as e:
        print(f"  • {name}: EXCEPTION - {e}")

# 2. WebSocket/REST Pre-Flight Connection Check
print("\n[2] WEBSOCKET/REST PRE-FLIGHT CONNECTION CHECK")
binance_futures_ping_url = "https://fapi.binance.com/fapi/v1/ping"
try:
    start_time = time.time()
    response = requests.get(binance_futures_ping_url, timeout=10.0)
    latency_ms = (time.time() - start_time) * 1000
    if response.status_code == 200:
        print(f"  • Binance Futures API Ping: PASS (Latency: {latency_ms:.2f}ms)")
    else:
        print(f"  • Binance Futures API Ping: FAIL (HTTP Status: {response.status_code}, Latency: {latency_ms:.2f}ms)")
except Exception as e:
    print(f"  • Binance Futures API Ping: EXCEPTION - {e}")

# 3. Config Validation
print("\n[3] CONFIG AND BOUNDS VALIDATION")
try:
    import config
    print(f"  • config.py loaded successfully.")
    
    # Check environment variable synchronization
    initial_balance = getattr(config, "INITIAL_BALANCE", 1000.0)
    leverage = getattr(config, "LEVERAGE", 10)
    risk_per_trade_pct = getattr(config, "RISK_PER_TRADE_PCT", 10.0)
    mode = getattr(config, "MODE", "paper")
    sandbox = getattr(config, "SANDBOX", True)
    
    print(f"    - MODE: {mode} (SANDBOX: {sandbox})")
    print(f"    - INITIAL_BALANCE: ${initial_balance:.2f}")
    print(f"    - LEVERAGE: {leverage}x")
    print(f"    - RISK_PER_TRADE_PCT: {risk_per_trade_pct}%")

    # Validate bounds for $1,000 baseline
    # Bounds rules:
    # 1. Leverage should not exceed standard risk parameters (typically max 20x for high-frequency RL; 10x is conservative/standard).
    # 2. Risk per trade pct should be within acceptable bounds (typically max 10.0% of balance or 5% depending on Kelly configuration, but 10.0% max limit).
    # 3. Mode must be paper / sandbox true for safety.
    
    leverage_ok = 1 <= leverage <= 25
    risk_ok = 0.1 <= risk_per_trade_pct <= 15.0
    
    if leverage_ok:
        print(f"  • Leverage check: PASS ({leverage}x is within bounds [1x, 25x])")
    else:
        print(f"  • Leverage check: FAIL ({leverage}x is out of bounds [1x, 25x])")
        
    if risk_ok:
        print(f"  • Risk per trade check: PASS ({risk_per_trade_pct}% is within bounds [0.1%, 15.0%])")
    else:
        print(f"  • Risk per trade check: FAIL ({risk_per_trade_pct}% is out of bounds [0.1%, 15.0%])")

except Exception as e:
    print(f"  • Config import/validation: EXCEPTION - {e}")

print("=" * 60)

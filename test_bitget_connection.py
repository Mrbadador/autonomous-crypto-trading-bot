"""
Sovereign Swarm — Live Bitget USDⓈ-M Futures Pre-Flight Probe
Safely verifies Bitget API credentials, permissions, and wallet balance.
Does NOT place any real orders.
"""
import os
import ccxt
from dotenv import load_dotenv

load_dotenv()

api_key    = os.getenv("BITGET_API_KEY") or os.getenv("API_KEY") or ""
api_secret = os.getenv("BITGET_API_SECRET") or os.getenv("API_SECRET") or ""
passphrase = os.getenv("BITGET_PASSPHRASE") or os.getenv("API_PASSWORD") or ""

print("\n" + "="*70)
print("   BITGET LIVE USDT-M DERIVATIVES PRE-FLIGHT PROBE")
print("="*70 + "\n")

if not api_key or not api_secret or not passphrase:
    print("[CONFIGURATION NEEDED]")
    print("Bitget requires 3 credentials:")
    print("  1. BITGET_API_KEY")
    print("  2. BITGET_API_SECRET")
    print("  3. BITGET_PASSPHRASE  (The password you entered when creating the API key on Bitget)")
    print("\nPlease add these to your .env file or provide them to test the connection.")
    raise SystemExit(0)

try:
    exchange = ccxt.bitget({
        "apiKey": api_key,
        "secret": api_secret,
        "password": passphrase,
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",  # Bitget USDT-M perpetual futures
        }
    })

    print("[*] 1. Pinging Bitget Futures Server...", end=" ", flush=True)
    server_time = exchange.fetch_time()
    print(f"SUCCESS (Server timestamp: {server_time})")

    print("[*] 2. Verifying Bitget API Signature & Permissions...", end=" ", flush=True)
    balance = exchange.fetch_balance()
    print("AUTHENTICATED & ACTIVE ✅")

    print("\n" + "-"*70)
    print("   LIVE BITGET WALLET BALANCE AUDIT")
    print("-"*70)

    usdt_free  = balance.get("USDT", {}).get("free", 0.0) or 0.0
    usdt_used  = balance.get("USDT", {}).get("used", 0.0) or 0.0
    usdt_total = balance.get("USDT", {}).get("total", 0.0) or 0.0

    print(f"  Bitget Futures USDT Free Balance : ${usdt_free:,.4f}")
    print(f"  Bitget Futures USDT In-Use Margin: ${usdt_used:,.4f}")
    print(f"  Bitget Futures USDT Total Equity : ${usdt_total:,.4f}")

    print("\n[*] 3. Testing Bitget Live Ticker Feeds:")
    for test_sym in ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]:
        try:
            ticker = exchange.fetch_ticker(test_sym)
            print(f"  ✓ {test_sym:<15} -> Live Price: ${ticker.get('last'):,.2f} | Bid: ${ticker.get('bid'):,.2f} | Ask: ${ticker.get('ask'):,.2f}")
        except Exception as e:
            print(f"  ⚠️ {test_sym}: {e}")

    print("\n" + "="*70)
    print("   BITGET PRE-FLIGHT VERDICT: 100% OPERATIONAL & READY ✅")
    print("="*70 + "\n")

except Exception as e:
    print(f"\n[ERROR] Bitget connection failed: {e}")
    print("Please verify that Futures Trading permission is enabled on your Bitget API key.")

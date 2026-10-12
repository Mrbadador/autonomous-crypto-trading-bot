import os
import ccxt
from dotenv import load_dotenv

load_dotenv()

api_key = (os.getenv("BINANCE_API_KEY") or os.getenv("API_KEY") or "").strip()
secret = (os.getenv("BINANCE_API_SECRET") or os.getenv("API_SECRET") or "").strip()

print("\n" + "="*70)
print("   API KEY COMPATIBILITY DIAGNOSTIC")
print("="*70)

print(f"API Key: {api_key[:6]}...{api_key[-4:]} (Length: {len(api_key)})")

# Test 1: Binance Live Spot
print("\n[TEST 1] Binance Live Spot (api.binance.com)...", end=" ")
try:
    spot = ccxt.binance({"apiKey": api_key, "secret": secret, "enableRateLimit": True})
    b_spot = spot.fetch_balance()
    print("SUCCESS ✅")
    print(f"  Spot USDT: {b_spot.get('USDT', {})}")
except Exception as e:
    print(f"FAILED: {e}")

# Test 2: Binance Live USD-M Futures
print("\n[TEST 2] Binance Live USDⓈ-M Futures (fapi.binance.com)...", end=" ")
try:
    fut = ccxt.binanceusdm({"apiKey": api_key, "secret": secret, "enableRateLimit": True})
    b_fut = fut.fetch_balance()
    print("SUCCESS ✅")
    print(f"  Futures USDT: {b_fut.get('USDT', {})}")
except Exception as e:
    print(f"FAILED: {e}")

# Test 3: Binance Testnet USD-M Futures
print("\n[TEST 3] Binance Testnet Futures (testnet.binancefuture.com)...", end=" ")
try:
    testnet = ccxt.binanceusdm({"apiKey": api_key, "secret": secret, "enableRateLimit": True})
    testnet.set_sandbox_mode(True)
    b_test = testnet.fetch_balance()
    print("SUCCESS ✅")
    print(f"  Testnet USDT: {b_test.get('USDT', {})}")
except Exception as e:
    print(f"FAILED: {e}")

# Test 4: Binance COIN-M Futures
print("\n[TEST 4] Binance COIN-M Futures (dapi.binance.com)...", end=" ")
try:
    dapi = ccxt.binancecoinm({"apiKey": api_key, "secret": secret, "enableRateLimit": True})
    b_dapi = dapi.fetch_balance()
    print("SUCCESS ✅")
    print(f"  COIN-M Balance: {b_dapi.get('info', {})}")
except Exception as e:
    print(f"FAILED: {e}")

print("="*70 + "\n")

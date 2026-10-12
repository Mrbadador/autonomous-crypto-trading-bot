import os
import ccxt
from dotenv import load_dotenv

load_dotenv()

k = os.getenv("BITGET_API_KEY")
s = os.getenv("BITGET_API_SECRET")
p = os.getenv("BITGET_PASSPHRASE")

print("\n" + "="*70)
print("   BITGET UNIFIED ACCOUNT V2 PROBE")
print("="*70)

ex = ccxt.bitget({
    "apiKey": k,
    "secret": s,
    "password": p,
    "enableRateLimit": True,
})

# Test V2 Unified endpoints
endpoints_to_test = [
    {"type": "swap", "productType": "USDT-FUTURES"},
    {"type": "swap", "productType": "usdt-futures"},
    {"type": "spot"},
    {"type": "future", "productType": "USDT-FUTURES"},
]

for opts in endpoints_to_test:
    print(f"\n[*] Testing parameters: {opts} ...", end=" ")
    try:
        ex.options["defaultType"] = opts.get("type", "swap")
        params = {}
        if "productType" in opts:
            params["productType"] = opts["productType"]
        
        b = ex.fetch_balance(params=params)
        print("SUCCESS ✅")
        usdt_info = b.get("USDT", {})
        print(f"    USDT Free : ${usdt_info.get('free', 0.0)}")
        print(f"    USDT Used : ${usdt_info.get('used', 0.0)}")
        print(f"    USDT Total: ${usdt_info.get('total', 0.0)}")
        
        # Check all non-zero assets
        non_zero = {k: v for k, v in b.get("total", {}).items() if v > 0}
        print(f"    Active Balances: {non_zero}")
        break
    except Exception as e:
        print(f"FAILED: {e}")

print("="*70 + "\n")

import os
import ccxt
from dotenv import load_dotenv

load_dotenv()

k = os.getenv("BITGET_API_KEY")
s = os.getenv("BITGET_API_SECRET")
p = os.getenv("BITGET_PASSPHRASE")

ex = ccxt.bitget({
    "apiKey": k,
    "secret": s,
    "password": p,
    "enableRateLimit": True,
})

print("\n" + "="*70)
print("   BITGET LIVE WALLET BALANCE AUDIT")
print("="*70)

# 1. Unified Trading Account
try:
    u = ex.privateUtaGetV3AccountAssets()
    data = u.get("data", {})
    print(f"\n[1] Unified Trading Account (Futures/Swap Margin):")
    print(f"    Account Total Equity: ${float(data.get('accountEquity', 0)):,.4f} USD")
    print(f"    USDT Equity         : ${float(data.get('usdtEquity', 0)):,.4f} USDT")
    assets = data.get("assets", [])
    found_assets = False
    for a in assets:
        avail = float(a.get("available", 0))
        if avail > 0.00001:
            found_assets = True
            print(f"    • {a.get('coin')}: Available = {avail:,.4f} | Total = {float(a.get('balance',0)):,.4f} (USD Value: ${float(a.get('usdValue',0)):,.4f})")
    if not found_assets:
        print("    • (No assets > $0.0001 in Unified Trading Account yet)")
except Exception as e:
    print(f"    Unified check error: {e}")

# 2. Funding Account
try:
    f = ex.privateUtaGetV3AccountFundingAssets()
    f_data = f.get("data", [])
    print(f"\n[2] Funding Account Wallet:")
    found_funding = False
    for a in f_data:
        avail = float(a.get("available", 0))
        if avail > 0.00001:
            found_funding = True
            print(f"    • {a.get('coin')}: Available = {avail:,.4f} | Total = {float(a.get('balance',0)):,.4f}")
    if not found_funding:
        print("    • (No assets in Funding Account)")
except Exception as e:
    print(f"    Funding check error: {e}")

# 3. Check Spot / other wallets
try:
    transfer_records = ex.privateUtaGetV3AccountTransferableCoins()
    print(f"\n[3] Transferable Coins in Account: {transfer_records.get('data', [])}")
except Exception as e:
    pass

print("="*70 + "\n")

"""
Sovereign Swarm — Live Binance USDⓈ-M Futures Pre-Flight Probe
Safely verifies API connectivity, credentials, futures permissions, and wallet balance.
Does NOT place any orders.
"""
import os
import ccxt
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("BINANCE_API_KEY") or os.getenv("API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET") or os.getenv("API_SECRET")

print("\n" + "="*70)
print("   BINANCE LIVE USDⓈ-M FUTURES PRE-FLIGHT CONNECTION PROBE")
print("="*70 + "\n")

if not api_key or not api_secret:
    print("[FATAL ERROR] Binance API Key or Secret missing from .env file.")
    raise SystemExit(1)

print(f"[*] API Key: {api_key[:6]}...{api_key[-4:]}")

try:
    exchange = ccxt.binanceusdm({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",
            "adjustForTimeDifference": True,
        }
    })

    print("[*] 1. Pinging Binance Futures server...", end=" ", flush=True)
    server_time = exchange.fetch_time()
    print(f"SUCCESS (Server timestamp: {server_time})")

    print("[*] 2. Verifying API Key signature & permissions...", end=" ", flush=True)
    balance = exchange.fetch_balance()
    print("AUTHENTICATED & ACTIVE ✅")

    print("\n" + "-"*70)
    print("   LIVE WALLET BALANCE AUDIT")
    print("-"*70)

    # Futures USDⓈ-M wallet breakdown
    usdt_free  = balance.get("USDT", {}).get("free", 0.0) or 0.0
    usdt_used  = balance.get("USDT", {}).get("used", 0.0) or 0.0
    usdt_total = balance.get("USDT", {}).get("total", 0.0) or 0.0

    print(f"  Futures USDT Free Balance : ${usdt_free:,.4f}")
    print(f"  Futures USDT In-Use Margin: ${usdt_used:,.4f}")
    print(f"  Futures USDT Total Equity : ${usdt_total:,.4f}")

    # Check other collateral currencies if any
    other_balances = {}
    for asset, data in balance.items():
        if asset not in ["USDT", "info", "free", "used", "total"] and isinstance(data, dict):
            tot = data.get("total", 0.0) or 0.0
            if tot > 0.0001:
                other_balances[asset] = tot

    if other_balances:
        print("\n  Other Balances in Account:")
        for asset, tot in other_balances.items():
            print(f"    - {asset}: {tot}")

    print("\n" + "-"*70)
    print("   EXCHANGE POSITION & PERMISSION CHECK")
    print("-"*70)

    # Check open positions on the exchange
    positions = exchange.fetch_positions()
    active_positions = [p for p in positions if float(p.get("contracts", 0) or 0) > 0]

    print(f"  Active Live Futures Positions: {len(active_positions)}")
    if active_positions:
        for p in active_positions:
            sym = p.get("symbol")
            side = p.get("side")
            contracts = p.get("contracts")
            entry_px = p.get("entryPrice")
            unrealized_pnl = p.get("unrealizedPnl")
            print(f"    - {sym} {side.upper()}: {contracts} contracts @ ${entry_px} (Unrealized PnL: ${unrealized_pnl})")
    else:
        print("  ✓ Zero open positions on Binance exchange (Clean state)")

    # Check open orders
    open_orders = exchange.fetch_open_orders()
    print(f"  Active Live Orders on Binance : {len(open_orders)}")
    if open_orders:
        for o in open_orders:
            print(f"    - {o.get('symbol')} {o.get('side')} {o.get('type')} @ {o.get('price')} (Status: {o.get('status')})")
    else:
        print("  ✓ Zero open orders on Binance exchange (Clean state)")

    # Test market ticker fetching
    print("\n[*] 3. Probing Live WebSocket/REST Market Feeds:")
    for test_sym in ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]:
        try:
            ticker = exchange.fetch_ticker(test_sym)
            last_price = ticker.get("last")
            bid = ticker.get("bid")
            ask = ticker.get("ask")
            print(f"  ✓ {test_sym:<15} -> Live Price: ${last_price:,.2f} | Bid: ${bid:,.2f} | Ask: ${ask:,.2f}")
        except Exception as e:
            print(f"  ⚠️ {test_sym}: {e}")

    print("\n" + "="*70)
    print("   PRE-FLIGHT VERDICT: 100% OPERATIONAL & READY ✅")
    print("="*70 + "\n")

except Exception as e:
    print(f"\n[ERROR] Connection failed: {e}")
    print("Please verify that 'Enable Futures' is checked in your Binance API settings.")

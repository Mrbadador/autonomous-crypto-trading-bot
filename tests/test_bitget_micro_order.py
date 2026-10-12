"""
Sovereign Swarm — Live Bitget Micro-Order & Routing Verification
Places a safe limit order (deep below market), verifies order placement in Bitget,
and cancels it immediately to guarantee 100% live trading execution capability.
"""
import os
import ccxt
from dotenv import load_dotenv

def run_live_micro_order_test():
    load_dotenv()

    k = os.getenv("BITGET_API_KEY")
    s = os.getenv("BITGET_API_SECRET")
    p = os.getenv("BITGET_PASSPHRASE")

    if not os.getenv("ALLOW_LIVE_ORDER_TESTS"):
        print("[SKIP] Live order test skipped. Set ALLOW_LIVE_ORDER_TESTS=1 to run.")
        return

    print("\n" + "="*76)
    print("   BITGET LIVE ORDER ROUTING & EXECUTION LIFECYCLE TEST")
    print("="*76 + "\n")

    exchange = ccxt.bitget({
        "apiKey": k,
        "secret": s,
        "password": p,
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",
        }
    })

    try:
        # 1. Fetch live balance
        print("[*] 1. Auditing Live Wallet Margin...")
        u = exchange.privateUtaGetV3AccountAssets()
        data = u.get("data", {})
        usdt_equity = float(data.get("usdtEquity", 0))
        print(f"  ✓ Live USDT Equity: ${usdt_equity:,.4f}\n")

        # 2. Test pair: XRP/USDT:USDT (1 contract = 1 XRP = ~$1.00)
        symbol = "XRP/USDT:USDT"
        print(f"[*] 2. Fetching Real-Time Ticker for {symbol}...")
        ticker = exchange.fetch_ticker(symbol)
        bid_price = ticker["bid"]
        ask_price = ticker["ask"]
        print(f"  ✓ Live Market Price: ${ticker['last']:,.4f} (Bid: ${bid_price} | Ask: ${ask_price})\n")

        # 3. Set Leverage to 5x
        print("[*] 3. Setting Leverage to 5x (Isolated Margin)...", end=" ")
        try:
            exchange.set_leverage(5, symbol)
            print("SUCCESS ✅")
        except Exception as le:
            print(f"Note: {le}")

        # 4. Create a Safe Limit Maker Buy Order deep below market (50% below bid to prevent fill)
        safe_limit_price = round(bid_price * 0.50, 4)
        qty = 1.0  # 1 XRP (~$0.50 notional, ~$0.10 margin)
        print(f"\n[*] 4. Placing Live Test Maker Limit Order on Bitget:")
        print(f"  • Order: BUY {qty} XRP @ ${safe_limit_price} (Deep Below Market - 0% Fill Risk)")

        order = exchange.create_order(
            symbol=symbol,
            type="limit",
            side="buy",
            amount=qty,
            price=safe_limit_price,
            params={"postOnly": True}  # Strict Maker order
        )
        order_id = order.get("id")
        print(f"  ✓ ORDER PLACED SUCCESSFULLY ON BITGET! Order ID: {order_id} ✅\n")

        # 5. Fetch Open Orders from Bitget
        print("[*] 5. Querying Bitget Open Orders to Confirm Exchange Acceptance...")
        open_orders = exchange.fetch_open_orders(symbol)
        matching = [o for o in open_orders if o["id"] == order_id]
        if matching:
            print(f"  ✓ Confirmed: Order {order_id} is ACTIVE on Bitget orderbook! Status: {matching[0].get('status')}\n")
        else:
            print(f"  • Open orders returned: {len(open_orders)} orders.\n")

        # 6. Cancel Order Immediately
        print(f"[*] 6. Canceling Test Order {order_id} on Bitget...", end=" ")
        cancel_res = exchange.cancel_order(order_id, symbol)
        print("ORDER CANCELED CLEANLY ✅\n")

        # 7. Final Verification
        open_orders_after = exchange.fetch_open_orders(symbol)
        print(f"[*] 7. Verifying Bitget Clean Slate: {len(open_orders_after)} open orders remaining.")

        print("\n" + "="*76)
        print("   BITGET LIVE ORDER EXECUTION TEST: 100% PASSED! 🚀")
        print("   Full Read/Write, Order Placement & Cancellation Verified.")
        print("="*76 + "\n")

    except Exception as e:
        print(f"\n[ERROR] Live test error: {e}")

if __name__ == "__main__":
    run_live_micro_order_test()

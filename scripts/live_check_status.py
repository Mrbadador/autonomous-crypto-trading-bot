import asyncio
import sqlite3
from bitget_uta_client import BitgetUTAClient

async def check():
    client = BitgetUTAClient()
    bal = await client.fetch_balance()
    pos = await client.fetch_positions()
    print(f"=== BITGET UTA LIVE ===")
    print(f"Cash Balance: ${bal:.2f} USDT")
    print(f"Active Positions: {len(pos)}")
    tot_unr = 0.0
    for p in pos:
        sym = p.get("symbol")
        side = p.get("posSide") or p.get("holdSide") or p.get("side")
        size = float(p.get("total", 0.0) or 0.0)
        entry = float(p.get("avgPrice", 0.0) or 0.0)
        mark = float(p.get("markPrice", 0.0) or 0.0)
        unr = float(p.get("unrealisedPnl", 0.0) or 0.0)
        roe = float(p.get("profitRate", 0.0) or 0.0) * 100.0
        lev = p.get("leverage")
        tot_unr += unr
        print(f"  {sym} {str(side).upper()} | Size: {size} | Lev: {lev}x | Entry: {entry} | Mark: {mark} | Unr PnL: ${unr:+.2f} ({roe:+.2f}% ROE)")
    
    tot_equity = bal + tot_unr
    print(f"Total Portfolio Equity: ${tot_equity:.2f} USDT")
    net_profit = tot_equity - 137.00
    roi = (net_profit / 137.00) * 100.0
    print(f"Net Profit vs $137.00 Deposit: ${net_profit:+.2f} ({roi:+.2f}%)")

    # Check strategy / stop orders
    for sym in ["NEARUSDT", "SOLUSDT"]:
        try:
            res = await client.exchange.private_uta_get_v3_trade_unfilled_strategy_orders({
                "category": "USDT-FUTURES",
                "symbol": sym
            })
            orders = res.get("data", [])
            print(f"\n=== Pending Strategy Orders for {sym} ({len(orders)}) ===")
            for o in orders:
                print(f"  Type: {o.get('type')} | StopLoss: {o.get('stopLoss')} | TakeProfit: {o.get('takeProfit')} | Status: {o.get('status')}")
        except Exception as e:
            print(f"Error checking strategy orders for {sym}: {e}")

    # Check closed trades in last 6 hours
    conn_data = sqlite3.connect("trading_data.db")
    c = conn_data.cursor()
    print("\n=== Recent Closed Trades (trading_data.db) ===")
    rows = c.execute("SELECT symbol, side, pnl, exit_reason, timestamp FROM swing_trades ORDER BY id DESC LIMIT 5").fetchall()
    for r in rows:
        print(f"  [SWING] {r[0]} {r[1].upper()} | PnL: ${r[2]:+.2f} | Reason: {r[3]} | Time: {r[4]}")
    
    rows_trades = c.execute("SELECT symbol, side, pnl, action, timestamp FROM trades ORDER BY id DESC LIMIT 5").fetchall()
    for r in rows_trades:
        print(f"  [INTRADAY] {r[0]} {r[1].upper()} | PnL: ${float(r[2] or 0):+.2f} | Action: {r[3]} | Time: {r[4]}")
    conn_data.close()

asyncio.run(check())

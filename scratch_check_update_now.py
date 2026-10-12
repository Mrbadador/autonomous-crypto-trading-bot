import asyncio
import os
import json
import sqlite3
import sys
from dotenv import load_dotenv

load_dotenv(".env")
from bitget_uta_client import BitgetUTAClient

async def main():
    client = BitgetUTAClient()
    try:
        bal = await client.fetch_balance()
        pos = await client.fetch_positions()
        active_pos = [p for p in pos if float(p.get("total", 0.0) or p.get("contracts", 0.0) or 0.0) > 0]
        unfilled = await client.exchange.private_uta_get_v3_trade_unfilled_orders({"category": "USDT-FUTURES"})
        unfilled_list = unfilled.get("data", {}).get("list", []) if isinstance(unfilled.get("data"), dict) else []

        print("=== 1. BITGET LIVE STATUS ===")
        print(f"USDT Balance: ${bal:.2f}")
        print(f"Active Positions: {len(active_pos)}")
        for p in active_pos:
            print(f"  {p.get('symbol')}: side={p.get('posSide')}, qty={p.get('total')}, entry={p.get('avgPrice')}, mark={p.get('markPrice')}, uPnL={p.get('unrealisedPnl')}")
        print(f"Unfilled Orders: {len(unfilled_list)}")
        for u in unfilled_list:
            print(f"  {u.get('symbol')}: id={u.get('orderId')}, side={u.get('side')}, price={u.get('price')}")
    finally:
        await client.close()

    print("\n=== 2. LOCAL SQLITE OPEN POSITIONS ===")
    with sqlite3.connect("trading_data.db") as conn:
        conn.row_factory = sqlite3.Row
        for tbl in ["swing_open_positions", "open_positions"]:
            rows = conn.execute(f"SELECT * FROM {tbl}").fetchall()
            print(f"{tbl}: {len(rows)} records")
            for r in rows:
                print(" ", dict(r))

        print("\n=== 3. TRADES CLOSED RECENTLY ===")
        rows2 = conn.execute("SELECT id, timestamp, symbol, side, entry_price, exit_price, pnl, exit_reason FROM swing_trades ORDER BY id DESC LIMIT 5").fetchall()
        print(f"Latest 5 closed trades:")
        for r in rows2:
            print(" ", dict(r))

if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import json
from bitget_uta_client import BitgetUTAClient

async def check():
    c = BitgetUTAClient()
    bal = await c.fetch_balance()
    print(f"UTA Cash Balance: {bal}")
    
    pos = await c.fetch_positions()
    active = [p for p in pos if float(p.get('total', 0) or 0) > 0]
    print(f"Active positions ({len(active)}):")
    for p in active:
        sym = p.get('symbol')
        side = p.get('posSide')
        total = p.get('total')
        entry = p.get('avgPrice')
        mark = p.get('markPrice')
        upnl = p.get('unrealisedPnl')
        profit_rate = float(p.get('profitRate', 0) or 0) * 100.0
        lev = p.get('leverage')
        liq = p.get('liquidationPrice')
        be = p.get('breakEvenPrice')
        print(f"  {sym} - Side: {side}, Size: {total}, Lev: {lev}x, Entry: {entry}, Mark: {mark}, uPnL: +${upnl} USDT, ROE: +{profit_rate:.2f}%, Liq: {liq}, BE: {be}")

    await c.exchange.close()

if __name__ == "__main__":
    asyncio.run(check())

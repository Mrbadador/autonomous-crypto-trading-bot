import asyncio
import sqlite3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bitget_uta_client import BitgetUTAClient

async def get_update():
    client = BitgetUTAClient()
    try:
        bal = await client.fetch_balance()
        pos = await client.fetch_positions()
        print('=== BITGET UTA LIVE ===')
        print(f'Wallet Balance: ${bal:,.2f} USDT')
        active_exchange_pos = [p for p in pos if float(p.get('total', 0) or p.get('available', 0) or 0) > 0]
        print(f'Exchange Active Positions: {len(active_exchange_pos)}')
        for p in active_exchange_pos:
            sym = p.get('symbol')
            side = p.get('posSide')
            sz = p.get('total')
            mk = p.get('markPrice')
            upl = p.get('unrealizedPL', 0)
            print(f'  {sym} {side}: Size={sz} Mark={mk} UPL={upl}')
    except Exception as e:
        print('Bitget API error:', e)
    finally:
        await client.close()

    print('\n=== DATABASE POSITIONS & LOCKS ===')
    with sqlite3.connect('trading_data.db') as conn:
        print('MARL Open:', conn.execute('SELECT * FROM marl_live_open_positions').fetchall())
        print('Sniper Open:', conn.execute('SELECT * FROM open_positions').fetchall())
        print('Berserker Open:', conn.execute('SELECT * FROM berserker_open_positions').fetchall())
        print('Swing Open:', conn.execute('SELECT * FROM swing_open_positions').fetchall())
        print('Asset Locks:', conn.execute('SELECT * FROM asset_locks').fetchall())
        
        print('\n=== RECENT CLOSED TRADES (LAST 3) ===')
        trades = conn.execute('SELECT timestamp, symbol, side, type, entry_price, exit_price, pnl FROM marl_live_trades ORDER BY id DESC LIMIT 3').fetchall()
        for t in trades:
            print('  MARL:', t)
            
        print('\n=== TELEMETRY / FLAGS ===')
        for k in ['DISABLED_SIGNAL_Sniper-Strategy', 'DISABLED_SIGNAL_Berserker-Strategy', 'macro_regime', 'ai_macro_posture', 'SWING_TRADER_ENABLED']:
            r = conn.execute('SELECT value FROM system_state WHERE key = ?', (k,)).fetchone()
            print(f'  {k}: {r[0] if r else None}')

if __name__ == '__main__':
    asyncio.run(get_update())

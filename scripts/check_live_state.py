import asyncio
import sqlite3
from bitget_uta_client import BitgetUTAClient

async def inspect():
    client = BitgetUTAClient()
    try:
        equity = await client.fetch_balance()
        raw_positions = await client.fetch_positions()
    finally:
        await client.close()

    bitget_pos = {p.get('symbol'): p for p in raw_positions}

    conn = sqlite3.connect('trading_data.db')
    conn.row_factory = sqlite3.Row
    db_positions = conn.execute('SELECT * FROM swing_open_positions').fetchall()

    regime_row = conn.execute("SELECT value FROM system_state WHERE key = 'regime'").fetchone()
    current_regime = regime_row[0] if regime_row else "CHOP"

    adx_row = conn.execute("SELECT value FROM system_state WHERE key = 'adx'").fetchone()
    current_adx = float(adx_row[0]) if adx_row and adx_row[0] else 0.0

    hurst_row = conn.execute("SELECT value FROM system_state WHERE key = 'hurst'").fetchone()
    current_hurst = float(hurst_row[0]) if hurst_row and hurst_row[0] else 0.0

    print("=================================================================")
    print("           SUPER BOT LIVE FLEET STATUS & PORTFOLIO AUDIT         ")
    print("=================================================================")
    print(f"Total Bitget Account Equity: ${equity:.2f} USDT")
    print(f"Live Market Regime:         {current_regime} (ADX: {current_adx:.1f}, Hurst: {current_hurst:.2f})")
    print(f"Active Portfolio Locks:     {len(db_positions)}/4 Concurrent Slots Filled")
    print("=================================================================\n")

    total_upnl = 0.0
    for p in db_positions:
        sym = p['symbol']
        clean = sym.replace('/', '')
        b_pos = bitget_pos.get(clean, {})
        mark = float(b_pos.get('markPrice', p['peak_price']) or p['peak_price'])
        entry = float(p['entry_price'])
        sl = float(p['stop_loss'])
        tp = float(p['take_profit'])
        peak = float(p['peak_price'])
        upnl = float(b_pos.get('unrealisedPnl', 0.0) or 0.0)
        roe = float(b_pos.get('profitRate', 0.0) or 0.0) * 100
        size = float(p['size'])
        dist_tp = ((tp - mark) / mark) * 100
        dist_sl = ((mark - sl) / mark) * 100
        total_upnl += upnl

        print(f"[{sym}] - {p['notes']}")
        print(f"  * Size:           {size:.4f} units ({p['leverage']}x Leverage)")
        print(f"  * Entry Price:    ${entry:,.4f}")
        print(f"  * Current Mark:   ${mark:,.4f} (Peak: ${peak:,.4f})")
        print(f"  * Unrealized PnL: ${upnl:+,.4f} USDT ({roe:+.2f}% ROE)")
        print(f"  * Stop Loss:      ${sl:,.4f} (Safety Distance: {dist_sl:+.2f}%)")
        print(f"  * Take Profit:    ${tp:,.4f} (Distance to TP:  {dist_tp:+.2f}%)")
        print(f"  * R-Multiple:     {p['r_multiple']:+.2f}R\n")

    print(f"--> Combined Open Unrealized PnL: ${total_upnl:+,.4f} USDT")

    c2 = sqlite3.connect('trading_ledger.db')
    rule_count = c2.execute("SELECT COUNT(*) FROM learned_rules").fetchone()[0]
    print(f"--> AI Continuous Learning: {rule_count} Dynamic Rules Injected into Boardroom")
    c2.close()
    conn.close()

if __name__ == "__main__":
    asyncio.run(inspect())

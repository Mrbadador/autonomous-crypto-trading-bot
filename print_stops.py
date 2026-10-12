import sqlite3, json, os

live_prices = {}
if os.path.exists('microstructure_state.json'):
    try:
        with open('microstructure_state.json', 'r', encoding='utf-8') as f:
            for s, d in json.load(f).get('symbols', {}).items():
                mid = (d.get('best_bid', 0.0) + d.get('best_ask', 0.0)) / 2.0
                live_prices[s.replace('USDT', '/USDT')] = mid
                live_prices[s] = mid
    except Exception:
        pass

conn = sqlite3.connect('file:trading_data.db?mode=ro', uri=True, timeout=5.0)
conn.row_factory = sqlite3.Row
print('=== CURRENT OPEN POSITIONS & STOPS ===')
for tbl, name in [('open_positions', 'Sniper'), ('marl_live_open_positions', 'MARL'), ('berserker_open_positions', 'Berserker')]:
    try:
        rows = conn.execute('SELECT symbol, side, entry_price, qty, stop_loss, take_profit, hard_sl FROM ' + tbl).fetchall()
        for r in rows:
            d = dict(r)
            sym = d['symbol']
            side = str(d['side']).upper()
            ep = float(d['entry_price'] or 0.0)
            sl = float(d['stop_loss'] or 0.0)
            tp = float(d['take_profit'] or 0.0)
            hsl = float(d['hard_sl'] or 0.0)
            mp = live_prices.get(sym, live_prices.get(sym.replace('/', ''), ep))
            pnl_pct = ((mp - ep) if side == 'LONG' else (ep - mp)) / ep * 100.0 if ep > 0 else 0.0
            print(f'[{name:<9}] {sym:<10} {side:<5} | Entry: ${ep:<10.4f} | Mark: ${mp:<10.4f} ({pnl_pct:+.2f}%) | SL: ${sl:<10.4f} | TP: ${tp:<10.4f} | HardSL: ${hsl:<10.4f}')
    except Exception:
        pass
conn.close()

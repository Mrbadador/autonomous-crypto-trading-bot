# audit_wednesday.py - Automated Wednesday Capital Scaling Scorecard Runner
import sqlite3
import pandas as pd
import json
import os
from datetime import datetime

def run_scorecard():
    # 1. Ingest real-time mark prices from microstructure state
    live_prices = {}
    if os.path.exists("microstructure_state.json"):
        try:
            with open("microstructure_state.json", "r", encoding="utf-8") as f:
                for s, d in json.load(f).get("symbols", {}).items():
                    mid = (d.get("best_bid", 0.0) + d.get("best_ask", 0.0)) / 2.0
                    live_prices[s.replace("USDT", "/USDT")] = mid
                    live_prices[s] = mid
        except Exception:
            pass

    db_path = 'file:trading_data.db?mode=ro'
    conn = sqlite3.connect(db_path, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    
    print('===============================================================')
    print('       FLEET CAPITAL SCALING FORENSIC SCORECARD')
    print(f'       Audit Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} (Local)')
    print('===============================================================\n')
    
    # 2. Closed Trades Evaluation
    query = """
        SELECT id, symbol, side, entry_price, exit_price, pnl, timestamp, notes 
        FROM trades 
        WHERE timestamp >= '2026-09-12T10:30:00' AND pnl IS NOT NULL
        ORDER BY timestamp ASC
    """
    df = pd.read_sql_query(query, conn)
    
    total_trades = len(df)
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] < 0]
    breakeven = df[df['pnl'] == 0]
    gross_profit = float(wins['pnl'].sum()) if not wins.empty else 0.0
    gross_loss = float(abs(losses['pnl'].sum())) if not losses.empty else 0.0
    net_pnl = float(df['pnl'].sum()) if not df.empty else 0.0
    win_rate = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.9 if gross_profit > 0 else 0.0)
    
    print('[1. CLOSED TRADES SUMMARY (SINCE RELOAD)]')
    print(f'  Total Closed Trades (N) : {total_trades}')
    print(f'  Win Rate                : {win_rate:.1f}% ({len(wins)}W / {len(losses)}L / {len(breakeven)}BE)')
    print(f'  Gross Profit            : +${gross_profit:.4f}')
    print(f'  Gross Loss              : -${gross_loss:.4f}')
    print(f'  Net Realized PnL        : ${net_pnl:+.4f}')
    print(f'  Profit Factor (PF)      : {profit_factor:.2f}\n')
    
    # 3. Active Positions across all fleet tables
    print('[2. CURRENT OPEN POSITIONS ACROSS ALL FLEET TABLES]')
    has_pos = False
    for tbl, fleet_name in [('open_positions', 'Sniper'), ('marl_live_open_positions', 'MARL'), ('berserker_open_positions', 'Berserker')]:
        try:
            open_pos = conn.execute(f'SELECT symbol, side, entry_price, qty, stop_loss, take_profit, hard_sl FROM {tbl}').fetchall()
            for r in open_pos:
                has_pos = True
                d = dict(r)
                sym = d['symbol']
                side = str(d['side']).upper()
                ep = float(d['entry_price'] or 0.0)
                sl = float(d['stop_loss'] or 0.0)
                tp = float(d['take_profit'] or 0.0)
                hsl = float(d['hard_sl'] or 0.0)
                mp = live_prices.get(sym, live_prices.get(sym.replace('/', ''), ep))
                upnl = ((mp - ep) if side == 'LONG' else (ep - mp)) / ep * 100.0 if ep > 0 else 0.0
                print(f"  * [{fleet_name:<9}] {sym:<10} {side:<5} | Entry: ${ep:<10.4f} | Mark: ${mp:<10.4f} ({upnl:+.2f}%) | SL: ${sl:<10.4f} | TP: ${tp:<10.4f} | HardSL: ${hsl:<10.4f}")
        except Exception:
            pass
    if not has_pos:
        print('  No open positions active across any fleet table.\n')
        
    # 4. Decision Gating
    print('\n[3. CAPITAL SCALING VERDICT]')
    if total_trades >= 20 and profit_factor >= 1.30 and net_pnl > 0:
        print('  >>> [GREEN LIGHT] All statistical criteria satisfied. Account is approved for capital scaling to $150-$250.')
    elif total_trades < 20 and net_pnl >= 0:
        print(f'  >>> [YELLOW LIGHT] Capital preserved (${net_pnl:+.4f}), but trade sample is small (N={total_trades} < 20). Recommended: Hold at $60.')
    else:
        print('  >>> [RED LIGHT] Drawdown or negative expectancy observed. VETO capital addition until post-mortem review.')
    print('===============================================================')
    conn.close()

if __name__ == '__main__':
    run_scorecard()

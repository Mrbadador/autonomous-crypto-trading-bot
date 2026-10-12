import sqlite3, datetime, json, math

def safe_div(a,b): return a/b if b else 0

with sqlite3.connect('trading_data.db') as conn:
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    print('=== SNIPER / LIVE TRADER PERFORMANCE ===')
    trades = c.execute('SELECT * FROM trades WHERE status="closed" ORDER BY id ASC').fetchall()
    if trades:
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        print(f'Total trades: {len(trades)}')
        print(f'Win rate: {safe_div(len(wins),len(trades))*100:.1f}%')
        print(f'Total PnL: ${sum(pnls):+.2f}')
        print(f'Avg win: ${safe_div(sum(wins),max(len(wins),1)):+.2f}')
        print(f'Avg loss: ${safe_div(sum(losses),max(len(losses),1)):+.2f}')
        print(f'Profit factor: {safe_div(sum(wins),abs(sum(losses))):.2f}')
        print(f'Max single win: ${max(pnls):+.2f}')
        print(f'Max single loss: ${min(pnls):+.2f}')
        print(f'Starting balance: ${trades[0]["balance"]:.2f}')
        print(f'Ending balance: ${trades[-1]["balance"]:.2f}')
        # Max drawdown
        peak = trades[0]['balance']
        max_dd = 0
        for t in trades:
            if t['balance'] > peak: peak = t['balance']
            dd = (peak - t['balance'])/peak*100
            if dd > max_dd: max_dd = dd
        print(f'Max drawdown: {max_dd:.1f}%')
    
    print('\n=== MARL LIVE TRADER PERFORMANCE ===')
    marl = c.execute('SELECT * FROM marl_live_trades ORDER BY id ASC').fetchall()
    if marl:
        pnls = [t['pnl'] for t in marl]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        print(f'Total MARL trades: {len(marl)}')
        print(f'Win rate: {safe_div(len(wins),len(marl))*100:.1f}%')
        print(f'Total PnL: ${sum(pnls):+.2f}')
        print(f'Profit factor: {safe_div(sum(wins),abs(sum(losses)) if losses else 1):.2f}')
    
    print('\n=== MEAN REVERSION PERFORMANCE ===')
    mr = c.execute('SELECT * FROM mean_reversion_trades ORDER BY id ASC').fetchall()
    if mr:
        pnls = [t['pnl'] for t in mr]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        print(f'Total MR trades: {len(mr)}')
        print(f'Win rate: {safe_div(len(wins),len(mr))*100:.1f}%')
        print(f'Total PnL: ${sum(pnls):+.2f}')
        print(f'Profit factor: {safe_div(sum(wins),abs(sum(losses)) if losses else 1):.2f}')
    
    print('\n=== BALANCES ===')
    for r in c.execute("SELECT key,value,updated_at FROM system_state WHERE key LIKE 'balance%'"):
        print(f'{r[0]}: ${float(r[1]):,.2f} (updated: {r[2]})')
    
    print('\n=== CURRENT OPEN POSITIONS ===')
    for tbl in ['marl_live_open_positions','mean_reversion_open_positions','open_positions']:
        try:
            rows = c.execute(f'SELECT * FROM {tbl}').fetchall()
            print(f'{tbl}: {len(rows)} positions')
            for r in rows: print(' ', dict(r))
        except: pass

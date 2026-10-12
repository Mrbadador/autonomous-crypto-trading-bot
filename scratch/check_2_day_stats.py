import sqlite3
import datetime

def check_stats():
    # 1. Balances
    with sqlite3.connect('trading_data.db') as conn:
        c = conn.cursor()
        
        # Balances
        balances = {}
        for agent in ['sniper', 'berserker', 'marl']:
            try:
                c.execute(f"SELECT value FROM system_state WHERE key = 'balance_{agent}'")
                row = c.fetchone()
                balances[agent] = float(row[0]) if row else 0.0
            except Exception as e:
                balances[agent] = f"Error: {e}"
        
        # PnL & Win Rate
        stats = {}
        for table, agent in [('trades', 'sniper'), ('berserker_trades', 'berserker'), ('marl_live_trades', 'marl')]:
            try:
                c.execute(f"SELECT COUNT(*), SUM(pnl) FROM {table}")
                row = c.fetchone()
                trades = int(row[0] or 0)
                pnl = float(row[1] or 0.0)
                
                c.execute(f"SELECT COUNT(*) FROM {table} WHERE pnl > 0")
                wins = int(c.fetchone()[0] or 0)
                winrate = (wins / trades * 100) if trades > 0 else 0.0
                
                stats[agent] = {'trades': trades, 'pnl': pnl, 'winrate': winrate}
            except Exception as e:
                stats[agent] = f"Error: {e}"
                
        # Open Positions
        open_pos = {}
        for table, agent in [('open_positions', 'sniper'), ('berserker_open_positions', 'berserker'), ('marl_live_open_positions', 'marl')]:
            try:
                c.execute(f"SELECT COUNT(*), SUM(entry_price * qty) FROM {table}")
                row = c.fetchone()
                open_pos[agent] = {'count': int(row[0] or 0), 'exposure': float(row[1] or 0.0)}
            except Exception as e:
                open_pos[agent] = f"Error: {e}"
                
    # 2. Sentiments & Regimes
    with sqlite3.connect('trading_ledger.db') as conn2:
        c2 = conn2.cursor()
        try:
            c2.execute("SELECT fear_greed_index, timestamp FROM market_context ORDER BY id DESC LIMIT 1")
            row = c2.fetchone()
            fear_greed = row if row else "No data"
        except Exception as e:
            fear_greed = f"Error: {e}"
            
        try:
            c2.execute("SELECT spy_health, timestamp FROM macro_logs ORDER BY id DESC LIMIT 1")
            row = c2.fetchone()
            spy = row if row else "No data"
        except Exception as e:
            spy = f"Error: {e}"

    print("--- BALANCES ---")
    for k,v in balances.items(): print(f"{k}: {v}")
    
    print("\n--- STATS ---")
    for k,v in stats.items(): print(f"{k}: {v}")
    
    print("\n--- OPEN POSITIONS ---")
    for k,v in open_pos.items(): print(f"{k}: {v}")
    
    print("\n--- MACRO ---")
    print(f"Fear & Greed: {fear_greed}")
    print(f"SPY Health: {spy}")

if __name__ == '__main__':
    check_stats()

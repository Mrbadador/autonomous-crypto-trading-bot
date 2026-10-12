import sqlite3
import pandas as pd
from datetime import datetime, timedelta

def analyze_trades():
    db_path = 'trading_data.db'
    ledger_path = 'trading_ledger.db'
    
    print("=== 24-HOUR DARK PATROL REPORT ===")
    
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql("SELECT * FROM marl_live_trades", conn)
        
        print(f"Total entries recorded: {len(df)}")
        
        closed = df[df['status'] == 'closed']
        open_trades = df[df['status'] != 'closed']
        
        print(f"Closed positions: {len(closed)}")
        print(f"Active/Open positions: {len(open_trades)}")
        
        if not closed.empty:
            pnl_sum = closed['pnl'].sum()
            win_count = len(closed[closed['pnl'] > 0])
            loss_count = len(closed[closed['pnl'] <= 0])
            win_rate = win_count / len(closed) * 100
            
            print(f"Cumulative PnL: {pnl_sum:.4f}")
            print(f"Win Rate: {win_rate:.1f}% ({win_count}W / {loss_count}L)")
        
        # Checking Vetoes in tradesync
        try:
            vetos = pd.read_sql("SELECT COUNT(*) as count FROM tradesync WHERE source LIKE 'VETO_SKIPPED:%' AND timestamp > datetime('now', '-1 day')", conn)
            print(f"Disaster Vetoes Triggered: {vetos.iloc[0]['count']}")
        except:
            pass

        conn.close()
        
    except Exception as e:
        print(f"Error reading trading_data.db: {e}")

if __name__ == '__main__':
    analyze_trades()

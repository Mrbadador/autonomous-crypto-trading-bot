import sqlite3
import pandas as pd

try:
    conn = sqlite3.connect("trading_data.db")
    # Load all trades
    df = pd.read_sql_query("SELECT pos_id, symbol, side, type, pnl, notes, timestamp FROM trades", conn)
    
    if len(df) == 0:
        print("No trades found.")
    else:
        # Group by pos_id to get total pnl per position
        # Filter for close events (sl_hit, tp_hit, partial_tp, exit_ai, reduce_ai)
        # Note: 'open' events have pnl 0.0 or NULL
        df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0)
        
        pos_stats = df.groupby('pos_id').agg({
            'symbol': 'first',
            'side': 'first',
            'pnl': 'sum'
        }).reset_index()
        
        print(pos_stats)
        
        print("\nSummary Stats (Position Level):")
        print(f"Total Positions: {len(pos_stats)}")
        print(f"Total PnL: {pos_stats['pnl'].sum():.2f}")
        
        wins = pos_stats[pos_stats['pnl'] > 0]
        losses = pos_stats[pos_stats['pnl'] < 0]
        
        print(f"Wins: {len(wins)} | Losses: {len(losses)}")
        if (len(wins) + len(losses)) > 0:
            print(f"Win Rate: {(len(wins) / (len(wins) + len(losses))) * 100:.1f}%")
            print(f"Avg Win: {wins['pnl'].mean():.2f}")
            print(f"Avg Loss: {losses['pnl'].mean():.2f}")
            if len(losses) > 0 and abs(losses['pnl'].mean()) > 0:
                print(f"Profit Factor (R:R): {abs(wins['pnl'].mean() / losses['pnl'].mean()):.2f}:1")

    conn.close()
except Exception as e:
    print(f"Error: {e}")

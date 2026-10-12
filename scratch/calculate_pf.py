import pandas as pd
import sys

def calculate_pf(csv_file):
    df = pd.read_csv(csv_file)
    # Assume the last session is what we want, but since we append, we can't easily tell.
    # However, we can look at the last N trades or filter by the current year if we knew the dates.
    # Let's just calculate overall PF from the file.
    
    # Filter for trades with PnL
    df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce')
    trades = df.dropna(subset=['pnl'])
    
    gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
    
    if gross_loss > 0:
        pf = gross_profit / gross_loss
    elif gross_profit > 0:
        pf = float('inf')
    else:
        pf = 0.0

    win_rate = (len(trades[trades['pnl'] > 0]) / len(trades)) * 100 if len(trades) > 0 else 0
    
    print(f"Total Trades: {len(trades)}")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Win Rate: {win_rate:.2f}%")
    
    return pf, win_rate
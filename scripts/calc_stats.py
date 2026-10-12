import pandas as pd

df = pd.read_csv('final_performance_report.csv')
closed_trades = df[df['pnl'] != 0].copy()
wins = closed_trades[closed_trades['pnl'] > 0]
losses = closed_trades[closed_trades['pnl'] < 0]

win_rate = len(wins) / len(closed_trades) if len(closed_trades) > 0 else 0
gross_profit = wins['pnl'].sum()
gross_loss = abs(losses['pnl'].sum())
profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

print(f"Win Rate: {win_rate*100:.2f}%")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Total Trades (Segments): {len(closed_trades)}")
print(f"Gross Profit: ${gross_profit:.2f}")
print(f"Gross Loss: ${gross_loss:.2f}")
print(f"Net PnL: ${gross_profit - gross_loss:.2f}")

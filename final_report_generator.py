import pandas as pd
import numpy as np

def calculate_drawdown(balances):
    balances = np.array(balances)
    peak = np.maximum.accumulate(balances)
    drawdown = (balances - peak) / peak
    return drawdown.min() * 100

df = pd.read_csv('trade_history.csv')

if df.empty:
    print("No trades found.")
    exit()

# Filter for closed trades (including partials)
closed_trades = df[df['pnl'] != 0].copy()

# Portfolio Stats
initial_balance = 100000.0
final_balance = df['balance'].iloc[-1]
total_pnl = df['pnl'].sum()
max_dd = calculate_drawdown(df['balance'])

print(f"--- PORTFOLIO SUMMARY ---")
print(f"Initial Balance: ${initial_balance:,.2f}")
print(f"Final Balance:   ${final_balance:,.2f}")
print(f"Net PnL (Sum):   ${total_pnl:,.2f}")
print(f"Max Drawdown:    {max_dd:.2f}%")
print(f"Total Segments:  {len(closed_trades)}")

# Per Symbol Stats
print(f"\n--- SYMBOL PERFORMANCE ---")
print(f"{'SYMBOL':<10} | {'WIN%':<8} | {'PF':<6} | {'NET PNL':<12}")
print("-" * 45)

symbols = df['symbol'].unique()
for sym in symbols:
    sym_df = df[df['symbol'] == sym]
    sym_closed = sym_df[sym_df['pnl'] != 0]
    
    if len(sym_closed) == 0: continue
    
    wins = len(sym_closed[sym_closed['pnl'] > 0])
    wr = (wins / len(sym_closed)) * 100
    
    gross_profits = sym_closed[sym_closed['pnl'] > 0]['pnl'].sum()
    gross_losses = abs(sym_closed[sym_closed['pnl'] < 0]['pnl'].sum())
    pf = (gross_profits / gross_losses) if gross_losses > 0 else 99.9
    
    net_pnl = sym_df['pnl'].sum()
    print(f"{sym:<10} | {wr:>6.1f}% | {pf:>5.2f} | ${net_pnl:>9.2f}")

# Reconciliation
expected_bal = initial_balance + total_pnl
print(f"\nRECONCILIATION: {'MATCH' if abs(expected_bal - final_balance) < 1.0 else 'DISCREPANCY'}")
if abs(expected_bal - final_balance) >= 1.0:
    print(f"Expected: {expected_bal:.2f}, Actual: {final_balance:.2f}, Diff: {final_balance - expected_bal:.2f}")

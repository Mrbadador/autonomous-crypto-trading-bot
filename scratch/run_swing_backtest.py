import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backtest_swing_architect import SwingArchitectBacktester

tester = SwingArchitectBacktester(
    symbols=['SOL/USDT', 'NEAR/USDT', 'AVAX/USDT', 'ADA/USDT', 'XRP/USDT'],
    initial_balance=500.0,
    allocation_usd=85.0,
    standard_leverage=3.0,
    high_quality_leverage=3.0,
    allow_shorts=False,
    t1_roe=30.0,
    t1_ratio=0.30,
    t2_roe=60.0,
    t2_ratio=0.60,
    be_threshold_r=1.5
)
res = tester.run()

print("=" * 65)
print("     4H SWING ARCHITECT (LONGS ONLY) BACKTEST RESULTS")
print("=" * 65)
print(f" Initial Balance:       ${res['initial_balance']:,.2f}")
print(f" Final Balance:         ${res['final_balance']:,.2f}")
print(f" Net Profit:            ${res['net_profit']:+,.2f} ({res['net_profit']/res['initial_balance']*100:+.2f}%)")
print(f" Win Rate:              {res['win_rate']:.1f}%")
print(f" Profit Factor:         {res['profit_factor']:.2f}")
print(f" Win/Loss Ratio:        {res['win_loss_ratio']:.2f}x")
print(f" Max Drawdown:          {res['max_drawdown_pct']:.2f}%")
print(f" Total Trades:          {res['total_trades']}")
print(f" Total Fees Paid:       ${res['total_fees_paid']:,.2f}")
print(f" Avg Holding Time:      {res['avg_holding_hours']:.1f} hours")
print("=" * 65)

if "df_trades" in res and not res["df_trades"].empty:
    df_t = res["df_trades"]
    print("\nPerformance by Symbol:")
    by_sym = df_t.groupby("symbol").agg(
        n=("net_pnl", "count"),
        wins=("net_pnl", lambda x: (x > 0).sum()),
        pnl=("net_pnl", "sum"),
        fees=("total_fees", "sum")
    )
    by_sym["wr"] = (by_sym["wins"] / by_sym["n"]) * 100
    for sym, r in by_sym.iterrows():
        print(f"  {sym:10} | Trades: {int(r['n']):3d} | WR: {r['wr']:5.1f}% | Net PnL: ${r['pnl']:+8.2f} | Fees: ${r['fees']:6.2f}")

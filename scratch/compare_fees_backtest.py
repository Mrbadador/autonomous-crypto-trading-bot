import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backtest_swing_architect import SwingArchitectBacktester

print("=" * 75)
print(" COMPARATIVE BACKTEST: TAKER VS MAKER FEES + EXITS ON 4H LONGS (2021-2026)")
print("=" * 75)

# 1. TAKER BASELINE (0.06% fee)
t_base = SwingArchitectBacktester(
    symbols=['SOL/USDT', 'NEAR/USDT', 'AVAX/USDT', 'ADA/USDT', 'XRP/USDT'],
    initial_balance=500.0,
    allocation_usd=85.0,
    standard_leverage=3.0,
    high_quality_leverage=3.0,
    allow_shorts=False,
    fee_rate=0.0006,  # 0.06% Taker
    slippage_rate=0.0002,
    t1_roe=30.0,
    t1_ratio=0.30,
    t2_roe=60.0,
    t2_ratio=0.60,
    be_threshold_r=2.0
)
res_base = t_base.run()

# 2. MAKER POST-ONLY ENTRY (0.02% fee) + TIGHTER BREAKEVEN (1.2R)
t_maker = SwingArchitectBacktester(
    symbols=['SOL/USDT', 'NEAR/USDT', 'AVAX/USDT', 'ADA/USDT', 'XRP/USDT'],
    initial_balance=500.0,
    allocation_usd=85.0,
    standard_leverage=3.0,
    high_quality_leverage=3.0,
    allow_shorts=False,
    fee_rate=0.0002,  # 0.02% Maker Limit Entry
    slippage_rate=0.0,
    t1_roe=30.0,
    t1_ratio=0.30,
    t2_roe=60.0,
    t2_ratio=0.60,
    be_threshold_r=1.2
)
res_maker = t_maker.run()

print(f"{'Metric':<25} | {'Taker Baseline':<20} | {'Maker + Optimized Exits':<20}")
print("-" * 75)
print(f"{'Final Balance':<25} | ${res_base['final_balance']:<19,.2f} | ${res_maker['final_balance']:<19,.2f}")
print(f"{'Net Profit ($)':<25} | ${res_base['net_profit']:<+19,.2f} | ${res_maker['net_profit']:<+19,.2f}")
print(f"{'Net Return (%)':<25} | {res_base['net_profit']/500*100:<+19.1f}% | {res_maker['net_profit']/500*100:<+19.1f}%")
print(f"{'Win Rate':<25} | {res_base['win_rate']:<19.1f}% | {res_maker['win_rate']:<19.1f}%")
print(f"{'Profit Factor':<25} | {res_base['profit_factor']:<19.2f} | {res_maker['profit_factor']:<19.2f}")
print(f"{'Total Fees Paid':<25} | ${res_base['total_fees_paid']:<19,.2f} | ${res_maker['total_fees_paid']:<19,.2f}")
print(f"{'Total Trades':<25} | {res_base['total_trades']:<19d} | {res_maker['total_trades']:<19d}")
print("=" * 75)

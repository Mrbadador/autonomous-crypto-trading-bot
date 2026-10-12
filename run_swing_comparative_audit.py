"""
run_swing_comparative_audit.py — Runs parametric backtests to identify what needs work in the Swing Architect.
"""
import sys
import pandas as pd
from backtest_swing_architect import SwingArchitectBacktester

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

scenarios = [
    {
        "name": "1. Live Baseline (Current Code)",
        "params": {"use_shifted_cloud": False, "allow_shorts": True, "t1_roe": 15.0, "t2_roe": 30.0, "atr_sl_mult": 2.5, "runner_tp_mult": 4.0}
    },
    {
        "name": "2. Classical Shifted Kumo (26-bar forward)",
        "params": {"use_shifted_cloud": True, "allow_shorts": True, "t1_roe": 15.0, "t2_roe": 30.0, "atr_sl_mult": 2.5, "runner_tp_mult": 4.0}
    },
    {
        "name": "3. Long Only (Disable Shorts)",
        "params": {"use_shifted_cloud": False, "allow_shorts": False, "t1_roe": 15.0, "t2_roe": 30.0, "atr_sl_mult": 2.5, "runner_tp_mult": 4.0}
    },
    {
        "name": "4. Pure Kijun Trail (No ROE Profit Locks)",
        "params": {"use_shifted_cloud": False, "allow_shorts": True, "t1_roe": 0.0, "t2_roe": 0.0, "atr_sl_mult": 2.5, "runner_tp_mult": 4.0}
    },
    {
        "name": "5. Wider Profit Locks (20% / 40% ROE)",
        "params": {"use_shifted_cloud": False, "allow_shorts": True, "t1_roe": 20.0, "t2_roe": 40.0, "atr_sl_mult": 2.5, "runner_tp_mult": 4.0}
    },
    {
        "name": "6. Tight SL (2.0x ATR)",
        "params": {"use_shifted_cloud": False, "allow_shorts": True, "t1_roe": 15.0, "t2_roe": 30.0, "atr_sl_mult": 2.0, "runner_tp_mult": 4.0}
    },
    {
        "name": "7. Wide SL (3.0x ATR)",
        "params": {"use_shifted_cloud": False, "allow_shorts": True, "t1_roe": 15.0, "t2_roe": 30.0, "atr_sl_mult": 3.0, "runner_tp_mult": 4.0}
    },
    {
        "name": "8. Max Locks = 3 (Concentrated fleet beta)",
        "params": {"use_shifted_cloud": False, "allow_shorts": True, "t1_roe": 15.0, "t2_roe": 30.0, "atr_sl_mult": 2.5, "runner_tp_mult": 4.0, "max_locks": 3}
    }
]

results = []
for sc in scenarios:
    print(f"Running: {sc['name']}...")
    tester = SwingArchitectBacktester(**sc["params"])
    res = tester.run()
    results.append({
        "Configuration": sc["name"],
        "Return %": f"{res['total_return_pct']:+.1f}%",
        "Net Profit": f"${res['net_profit']:+.2f}",
        "Trades": res["total_trades"],
        "Win Rate": f"{res['win_rate']:.1f}%",
        "Profit Factor": f"{res['profit_factor']:.2f}",
        "Win/Loss": f"{res['win_loss_ratio']:.2f}x",
        "Max DD": f"{res['max_drawdown_pct']:.1f}%",
        "Sharpe": f"{res['sharpe_ratio']:.2f}",
        "Sortino": f"{res['sortino_ratio']:.2f}",
        "Fees Paid": f"${res['total_fees_paid']:.2f}"
    })

df_res = pd.DataFrame(results)
print("\n" + "=" * 90)
print("             THE SWING ARCHITECT — PARAMETRIC COMPARATIVE AUDIT MATRIX")
print("=" * 90)
print(df_res.to_string(index=False))
print("=" * 90)

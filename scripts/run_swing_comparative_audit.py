"""
run_swing_comparative_audit.py — Runs parametric backtests to identify what needs work in the Swing Architect.

Run it by hand from the repo root: `python scripts/run_swing_comparative_audit.py`. The backtester
replays the local history in data/historical/, so the repo root is also the working directory.
That invocation puts <repo>/scripts on sys.path, not the repo root, so this module puts the repo
root there itself before it imports the backtester that lives in backtests/.
"""
import os
import sys
import pandas as pd

# The backtester lives in backtests/ (a namespace package: it has no __init__.py), so the repo root
# has to be on sys.path before it can be imported. `python scripts/run_swing_comparative_audit.py`
# puts <repo>/scripts on sys.path[0] instead, so this module puts the repo root there itself --
# the same spelling the repo's other hand-run scripts use (scripts/quant_validation.py,
# scripts/run_grid_ablation.py, scripts/reconcile_d4_corrupt_exits.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtests.backtest_swing_architect import (
    SwingArchitectBacktester,
    format_profit_factor,
    format_win_loss_ratio,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Every cell of the matrix is a measurement or an explicit absence. A run that closed no trade has
# no rate, no ratio and no denominator to divide by, and the row says so instead of printing a
# number that was never measured (the profit-factor cell used to print a hardcoded 99.00).
NO_DATA = "n/a (no historical data)"
NO_CLOSED_TRADES = "n/a (no closed trades, nothing was measured)"

scenarios = [
    {
        # The unshifted arm reads the Ichimoku cloud on the bar that computed it. The live engine
        # does not use it: it always projects the cloud 26 bars forward (swing_trader.py:642-643,
        # quant_validation.py:89-90). It is kept as the A/B comparison against that treatment, so
        # it is named for what it is and not as a "live baseline" it never was.
        "name": "1. Unshifted Cloud Comparison Arm (not the live engine)",
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


def _cell(value, spec, absence):
    """Render a measured value in `spec`, or the absence that says why there is none."""
    return absence if value is None else spec.format(value)


def arm_row(name, res):
    """Build one row of the comparative audit matrix from a backtest result.

    `res` is the dictionary `SwingArchitectBacktester.run()` returns. Every cell is either a
    measurement or an explicit absence naming its cause: a run that produced no history, and a run
    that closed no trade, are told apart from a run that measured a value.
    """
    if res.get("error"):
        # The run produced no data at all, so nothing was measured on any axis.
        return {
            "Configuration": name,
            "Return %": NO_DATA,
            "Net Profit": NO_DATA,
            "Trades": 0,
            "Win Rate": NO_DATA,
            "Profit Factor": NO_DATA,
            "Win/Loss": NO_DATA,
            "Max DD": NO_DATA,
            "Sharpe": NO_DATA,
            "Sortino": NO_DATA,
            "Fees Paid": NO_DATA,
        }

    n_trades = res.get("total_trades", 0)
    loss_count = res.get("loss_count", 0)
    return {
        "Configuration": name,
        "Return %": _cell(res.get("total_return_pct"), "{:+.1f}%", NO_CLOSED_TRADES),
        "Net Profit": _cell(res.get("net_profit"), "${:+.2f}", NO_CLOSED_TRADES),
        "Trades": n_trades,
        "Win Rate": _cell(res.get("win_rate"), "{:.1f}%", NO_CLOSED_TRADES),
        # The two ratios have no denominator when the run had no losing trade: the backtester
        # publishes None there and the renderers say which absence it is.
        "Profit Factor": format_profit_factor(res.get("profit_factor"), n_trades, loss_count),
        "Win/Loss": format_win_loss_ratio(res.get("win_loss_ratio"), n_trades, loss_count),
        "Max DD": _cell(res.get("max_drawdown_pct"), "{:.1f}%", NO_CLOSED_TRADES),
        "Sharpe": _cell(res.get("sharpe_ratio"), "{:.2f}", NO_CLOSED_TRADES),
        "Sortino": _cell(res.get("sortino_ratio"), "{:.2f}", NO_CLOSED_TRADES),
        "Fees Paid": _cell(res.get("total_fees_paid"), "${:.2f}", NO_CLOSED_TRADES),
    }


def main():
    results = []
    for sc in scenarios:
        print(f"Running: {sc['name']}...")
        tester = SwingArchitectBacktester(**sc["params"])
        res = tester.run()
        results.append(arm_row(sc["name"], res))

    df_res = pd.DataFrame(results)
    print("\n" + "=" * 90)
    print("             THE SWING ARCHITECT — PARAMETRIC COMPARATIVE AUDIT MATRIX")
    print("=" * 90)
    print(df_res.to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    main()

import sys
import os
import asyncio
import logging
import csv
from itertools import product

# Add parent directory to path to allow importing bot modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from exchange_manager import ExchangeManager
from state_manager import EngineState
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "archive"))
from bot_v10_backtest import run_cycle

# Disable most logging for speed but keep WARNING+
logging.getLogger().setLevel(logging.WARNING)

async def run_backtest(partial_tp, trail_act, start_date, end_date, sl_halt_count):
    """Runs a full replay backtest with specific hyperparameters over a given date window."""
    config.MODE                           = "replay"
    config.IS_REPLAY                      = True
    config.BYPASS_AI_IN_REPLAY            = True
    config.SYMBOLS                        = ["BTC/USDT"]
    config.PARTIAL_TP_ATR_MULTIPLE        = partial_tp
    config.TRAIL_ACTIVATION_ATR_MULTIPLE  = trail_act
    config.REPLAY_START_DATE              = start_date
    config.REPLAY_END_DATE                = end_date
    # Loosen halt count so bot can recover from stop-hunt wiggles in chop
    config.CONSECUTIVE_SL_HALT_COUNT      = sl_halt_count

    em    = ExchangeManager()
    state = EngineState(sim_instance=em.sim)
    em.init_replay(config.SYMBOLS, config.TIMEFRAME)

    for b in em.replay.buffers.values():
        b.cursor = config.OHLCV_LIMIT   # Skip warmup

    test_symbol = "BTC/USDT"
    while True:
        await run_cycle(test_symbol, em, state)
        state.update_metrics()
        can_continue = em.replay.advance_all(config.TIMEFRAME)
        if not can_continue.get(test_symbol, False):
            break

    trades      = state.journal.trades
    wins        = [t for t in trades if t.pnl > 0]
    losses      = [t for t in trades if t.pnl <= 0]
    win_rate    = len(wins) / len(trades) if trades else 0
    total_wins  = sum(t.pnl for t in wins)
    total_loss  = abs(sum(t.pnl for t in losses))
    pf          = (total_wins / total_loss) if total_loss > 0 else (99.9 if total_wins > 0 else 0)
    net_pnl     = em.balance.total - config.INITIAL_BALANCE
    max_dd      = state.max_drawdown
    rf          = (net_pnl / max_dd) if max_dd > 0 else (net_pnl if net_pnl > 0 else 0)

    return {
        "PARTIAL_TP_ATR_MULTIPLE":       partial_tp,
        "TRAIL_ACTIVATION_ATR_MULTIPLE": trail_act,
        "Net PnL":        net_pnl,
        "pnl_pct":        ((em.balance.total / config.INITIAL_BALANCE) - 1) * 100,
        "drawdown":       max_dd,
        "win_rate":       win_rate,
        "profit_factor":  pf,
        "recovery_factor": rf,
        "trade_count":    len(trades),
    }


async def main():
    # ---------------------------------------------------------------------------
    # Chop-Focused Diagnostic Optimization — July 2023 BTC/USDT
    # ---------------------------------------------------------------------------
    OOS_START   = "2023-07-01"
    OOS_END     = "2023-07-31"
    SL_HALT     = 5          # Loosened from 3 to recover through stop-hunt wiggles

    # Tighter search grid suited for a sideways / range-bound regime
    tp_range    = [0.8, 1.0, 1.2, 1.5]
    trail_range = [0.1, 0.2, 0.3]

    print("=" * 65)
    print("  CHOP DIAGNOSTIC OPTIMIZATION -- July 2023 BTC/USDT")
    print("=" * 65)
    print(f"  Date Window      : {OOS_START} -> {OOS_END}")
    print(f"  TP range         : {tp_range}")
    print(f"  Trail range      : {trail_range}")
    print(f"  SL Halt Count    : {SL_HALT} (loosened for stop-hunt recovery)")
    print(f"  Macro Gate       : V6.1 -- flat EMA slope => CHOP (allow both sides)")
    print("=" * 65 + "\n")

    csv_file    = "chop_optimization_results.csv"
    results     = []
    combinations = list(product(tp_range, trail_range))
    total       = len(combinations)

    for i, (tp, trail) in enumerate(combinations):
        print(f"  [{i+1:2d}/{total}] Testing: TP={tp:.1f} | Trail={trail:.1f} ...", end=" ", flush=True)
        m = await run_backtest(tp, trail, OOS_START, OOS_END, SL_HALT)
        results.append(m)
        pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] < 99 else "INF"
        print(f"PnL={m['Net PnL']:+.2f}  PF={pf_str}  Trades={m['trade_count']}")

        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=m.keys())
            writer.writeheader()
            writer.writerows(results)

    # ---------------------------------------------------------------------------
    # Scoring: for a chop regime, Profit Factor is the primary quality signal
    # ---------------------------------------------------------------------------
    MIN_TRADES = 5
    valid = [r for r in results if r["trade_count"] >= MIN_TRADES]

    def chop_score(r):
        return (r["profit_factor"] * 20) + (r["recovery_factor"] * 5) + (r["win_rate"] * 10)

    ranked = sorted(valid, key=chop_score, reverse=True)

    print("\n" + "=" * 65)
    print("  CHOP MASTER SETTINGS -- RANKED BY PROFIT FACTOR")
    print("=" * 65)
    if not ranked:
        print(f"  No configs met the minimum {MIN_TRADES} trade threshold.")
    else:
        print(f"  {'Rank':<5} {'TP':>5} {'Trail':>7} {'PF':>7} {'RF':>7} {'PnL%':>8} {'WR':>8} {'Trades':>7}")
        print("  " + "-" * 55)
        for rank, r in enumerate(ranked[:8], start=1):
            pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] < 99 else " INF "
            print(f"  {rank:<5} {r['PARTIAL_TP_ATR_MULTIPLE']:>5.1f} "
                  f"{r['TRAIL_ACTIVATION_ATR_MULTIPLE']:>7.1f} "
                  f"{pf_str:>7} "
                  f"{r['recovery_factor']:>7.2f} "
                  f"{r['pnl_pct']:>7.2f}% "
                  f"{r['win_rate']*100:>7.1f}% "
                  f"{r['trade_count']:>7}")

        best = ranked[0]
        print(f"\n  [CHOP MASTER SETTING]")
        print(f"  TP={best['PARTIAL_TP_ATR_MULTIPLE']} | Trail={best['TRAIL_ACTIVATION_ATR_MULTIPLE']}")
        print(f"  Profit Factor : {best['profit_factor']:.2f}")
        print(f"  Net PnL       : ${best['Net PnL']:+.2f}  ({best['pnl_pct']:+.2f}%)")
        print(f"  Win Rate      : {best['win_rate']*100:.1f}%")
        print(f"  Trades        : {best['trade_count']}")

    print(f"\n  Full results -> {csv_file}\n")


if __name__ == "__main__":
    asyncio.run(main())

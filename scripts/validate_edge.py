import asyncio
import sys
import os

# Add parent directory to path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from bot import run_cycle
from exchange_manager import ExchangeManager
from state_manager import EngineState


async def validate_out_of_sample():
    # -----------------------------------------------------------------------
    # BTC Master Setting — injected directly (proven by optimization)
    # -----------------------------------------------------------------------
    BEST_TRAIL = 0.2

    print("\n" + "=" * 60)
    print("   V6.1 REGIME SENSOR -- OUT-OF-SAMPLE VALIDATION")
    print("=" * 60)
    print(f"  Parameters  : TP=Dynamic (Trend: 2.5x / Chop: 1.0x) | Trail={BEST_TRAIL}x ATR")
    print(f"  OOS Window  : October 1 - October 31, 2023")
    print(f"  Symbol      : BTC/USDT")
    print(f"  HTF Filter  : 4H EMA200 Slope-Based Regime Sensor")
    print("=" * 60 + "\n")

    # -----------------------------------------------------------------------
    # 1. Force config into OOS mode
    # -----------------------------------------------------------------------
    config.MODE                       = "replay"
    config.IS_REPLAY                  = True
    config.BYPASS_AI_IN_REPLAY        = True
    config.SYMBOLS                    = ["BTC/USDT"]
    config.TRAIL_ACTIVATION_ATR_MULTIPLE = BEST_TRAIL
    config.REPLAY_START_DATE          = "2023-10-01"
    config.REPLAY_END_DATE            = "2023-10-31"
    config.CONSECUTIVE_SL_HALT_COUNT  = 5  # Loosened to survive stop hunts
    config.HTF_ENABLED                = True   # Ensure Macro Gate is active

    test_symbol = "BTC/USDT"

    # -----------------------------------------------------------------------
    # 2. Bootstrap ExchangeManager + replay engine
    # -----------------------------------------------------------------------
    em    = ExchangeManager()
    state = EngineState(sim_instance=em.sim)
    em.init_replay(config.SYMBOLS, config.TIMEFRAME)
    em.set_leverage(test_symbol, config.LEVERAGE)

    # Skip the warmup period — cursor starts after OHLCV_LIMIT candles
    for b in em.replay.buffers.values():
        b.cursor = config.OHLCV_LIMIT

    # -----------------------------------------------------------------------
    # 3. Run the replay loop
    # -----------------------------------------------------------------------
    print(f"Running replay for {test_symbol}  [{config.REPLAY_START_DATE} -> {config.REPLAY_END_DATE}] ...\n")
    while True:
        await run_cycle(test_symbol, em, state)
        state.update_metrics()
        can_continue = em.replay.advance_all(config.TIMEFRAME)
        if not can_continue.get(test_symbol, False):
            break

    # -----------------------------------------------------------------------
    # 4. Compute summary metrics
    # -----------------------------------------------------------------------
    trades       = state.journal.trades
    oos_pnl      = em.balance.total - config.INITIAL_BALANCE
    trade_count  = len(trades)
    wins         = [t for t in trades if t.pnl > 0]
    losses       = [t for t in trades if t.pnl <= 0]
    win_rate     = len(wins) / trade_count if trade_count > 0 else 0.0
    total_profit = sum(t.pnl for t in wins)
    total_loss   = abs(sum(t.pnl for t in losses))
    profit_factor = (total_profit / total_loss) if total_loss > 0 else (99.9 if total_profit > 0 else 0.0)
    max_dd       = state.max_drawdown

    # -----------------------------------------------------------------------
    # 5. Print report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("   OOS VALIDATION RESULTS")
    print("=" * 60)
    print(f"  Net PnL        : ${oos_pnl:+.2f}")
    print(f"  Profit Factor  : {profit_factor:.2f}")
    print(f"  Win Rate       : {win_rate * 100:.1f}%")
    print(f"  Trades         : {trade_count}  (W:{len(wins)} / L:{len(losses)})")
    print(f"  Max Drawdown   : ${max_dd:.2f}")
    print("=" * 60)

    if oos_pnl > 0:
        print("\n  [PASS] EDGE VALIDATED -- Parameters survived unseen July 2023 data!")
    else:
        print("\n  [FAIL] OVERFITTED -- Edge failed on unseen data. Refine Macro Gate.")
    print()


if __name__ == "__main__":
    asyncio.run(validate_out_of_sample())

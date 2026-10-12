"""
Quick diagnostic: runs a single-symbol replay for 50 candles and dumps all state.
"""
import sys, os, asyncio, logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING)

import config
# CRITICAL: set IS_REPLAY BEFORE any other module uses it
config.MODE = "replay"
config.IS_REPLAY = True
config.BYPASS_AI_IN_REPLAY = True
config.REPLAY_DELAY = 0
config.SYMBOLS = ["BTC/USDT"]

# Enable bot-level INFO so we see decision output
logging.getLogger("bot").setLevel(logging.INFO)

from exchange_manager import ExchangeManager
from state_manager import EngineState
from bot import run_cycle

async def main():
    em = ExchangeManager()
    state = EngineState(sim_instance=em.sim)

    em.init_replay(config.SYMBOLS, config.TIMEFRAME)
    # Warm up with 500 candles (gives 4H EMA200 ~125 bars)
    for b in em.replay.buffers.values():
        b.cursor = 500

    sym = "BTC/USDT"
    n = 0
    max_cycles = 200

    while n < max_cycles:
        await run_cycle(sym, em, state)
        state.update_metrics()
        ok = em.replay.advance_all(config.TIMEFRAME)
        if not ok.get(sym, False):
            break
        n += 1

    print(f"\n=== DIAGNOSTIC RESULTS after {n} cycles ===")
    print(f"Balance : {em.balance.total:.4f}")
    print(f"Journal : {len(state.journal.trades)} entries")
    print(f"Halted  : {state.is_halted}")
    print(f"ObsOnly : {state.is_observe_only}")

    # Show last 5 journal entries
    for t in state.journal.trades[-5:]:
        print(f"  [{t.final_action}] side={t.side} pnl={t.pnl:.4f}")

    # Check trade_history.csv
    import csv
    csv_rows = 0
    try:
        with open("trade_history.csv") as f:
            csv_rows = sum(1 for _ in csv.reader(f)) - 1
    except FileNotFoundError:
        pass
    print(f"CSV trade rows : {csv_rows}")

asyncio.run(main())

import asyncio
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from live_oracle import LiveOracle

async def test_run():
    print("Starting Oracle test run...")
    oracle = LiveOracle(symbol="BTC/USDT", db_path="trading_ledger.db")
    # Set sys.argv to include --once
    sys.argv.append("--once")
    try:
        await oracle.main_update_loop()
        print("Oracle test run successful.")
    except Exception as e:
        print(f"Oracle test run failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_run())

import json
import time
import asyncio
from executor_agent import ExecutorAgent

async def run_test():
    # Mock microstructure state with VETO trigger (-140.0 OFI)
    state = {
        "timestamp": time.time(),
        "net_ofi_score": -140.0
    }
    with open("microstructure_state.json", "w") as f:
        json.dump(state, f)

    agent = ExecutorAgent()

    print("\n--- Testing LONG VETO ---")
    res = await agent.execute_trade({"action_name": "LONG", "confidence": 0.8}, 50000.0)
    print(f"Trade Execution Result: {res}")

if __name__ == "__main__":
    asyncio.run(run_test())

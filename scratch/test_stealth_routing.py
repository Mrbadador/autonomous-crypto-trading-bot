import asyncio
import sys
import os

# Add workspace to sys path so we can import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from executor_agent import ExecutorAgent

async def test_stealth_routing():
    print("======================================================================")
    print("Testing Stealth Routing Engine for LONG and SHORT Orders")
    print("======================================================================\n")

    # Instantiate ExecutorAgent
    executor = ExecutorAgent(risk_per_trade=0.02, min_confidence=0.50)

    # Inject LivePaperExchange for local execute_order testing
    from live_paper_exchange import LivePaperExchange
    paper_ex = LivePaperExchange(initial_balance=1000.0, account_name="marl_live")
    executor.exchange = paper_ex

    # Mock get_latest_microstructure to return non-vetoable permissive data
    async def mock_get_latest_microstructure(symbol):
        return {
            "liquidity_imbalance": 0.0,
            "vacuum_detected": 0,
            "signal_multiplier": 1.0,
            "stale": False
        }
    executor.get_latest_microstructure = mock_get_latest_microstructure

    # 1. Test LONG Order
    print("[TEST] 1. Executing LONG order...")
    long_decision = {
        "action_name": "LONG",
        "confidence": 0.80,
        "account_balance": 10000.00
    }
    success_long = await executor.execute_trade(long_decision, 65000.00, symbol="BTC/USDT")
    print(f"LONG Trade success: {success_long}\n")

    # 2. Test SHORT Order
    print("[TEST] 2. Executing SHORT order...")
    short_decision = {
        "action_name": "SHORT",
        "confidence": 0.80,
        "account_balance": 10000.00
    }
    success_short = await executor.execute_trade(short_decision, 65000.00, symbol="BTC/USDT")
    print(f"SHORT Trade success: {success_short}\n")

    # Wait for fire-and-forget tasks to execute/log
    print("Waiting for TradeSync background logging task...")
    await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(test_stealth_routing())

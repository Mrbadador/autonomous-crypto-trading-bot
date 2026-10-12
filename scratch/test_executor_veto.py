import asyncio
import logging
import sys
import os
sys.path.append(os.getcwd())
from executor_agent import ExecutorAgent

logging.basicConfig(level=logging.INFO)

async def test_veto():
    executor = ExecutorAgent(risk_per_trade=0.02, min_confidence=0.1) # low threshold for test
    
    # Mock decision
    decision = {"action_name": "SHORT", "confidence": 0.8}
    current_price = 78200.0
    
    print("\n--- Testing Veto for SHORT (with negative imbalance in DB, should pass) ---")
    # Our DB check showed imbalance ~ -0.64 (Ask resistance). 
    # Veto for SHORT happens if imbalance > 0.65 (Bid support).
    # So this should PASS.
    success = await executor.execute_trade(decision, current_price, symbol="BTC/USDT")
    print(f"Trade success: {success}")

    print("\n--- Testing Veto for LONG (with negative imbalance, should VETO) ---")
    # Veto for LONG happens if imbalance < -0.65.
    # Our DB has -0.648... let's see if we can find one with < -0.65 or just mock it.
    decision_long = {"action_name": "LONG", "confidence": 0.8}
    success_long = await executor.execute_trade(decision_long, current_price, symbol="BTC/USDT")
    print(f"Trade success: {success_long}")

if __name__ == "__main__":
    asyncio.run(test_veto())

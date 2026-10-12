# scratch/test_full_swarm_execution.py
import os
import sys
import time

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from swarm_orchestrator import SwarmOrchestrator
import numpy as np

import asyncio

async def test_full_flow():
    print("=== STARTING END-TO-END SWARM EXECUTOR SHAKEDOWN ===")
    
    # 1. Instantiate Swarm Orchestrator
    print("[1] Instantiating Central Swarm Orchestrator...")
    orchestrator = SwarmOrchestrator()
    
    # 2. Fetch state tensor from Analyst to get genuine real-world market data shapes
    print("\n[2] Generating live Analyst state vector...")
    state_tensor = await orchestrator.analyst.build_state_tensor()
    print(f"--> Real-world State shape: {state_tensor.shape}")
    
    # 3. Construct a custom, high-conviction decision to bypass thresholds
    high_conviction_decision = {
        "action_id": 1,
        "action_name": "LONG",
        "confidence": 0.9875  # Extremely high confidence to trigger executor
    }
    current_btc_price = 68500.00
    
    print(f"\n[3] Injecting Synthetic High-Conviction Order...")
    print(f"--> Action: {high_conviction_decision['action_name']} | Confidence: {high_conviction_decision['confidence']:.2%}")
    print(f"--> Entry Reference: ${current_btc_price:,.2f}")
    
    # 4. Manually dispatch to executor with real state tensor
    print("\n[4] Executing downstream pipeline...")
    success = await orchestrator.executor.execute_trade(
        strategy_decision=high_conviction_decision,
        current_price=current_btc_price,
        market_state=state_tensor,
        symbol="BTC/USDT"
    )
    
    if success:
        print("\n=== SHAKEDOWN VERIFIED: ALMGREN-CHRISS STEALTH INTEGRATION OPERATIONAL ===")
    else:
        print("\n--> Verification Failed: Executor rejected high confidence trade.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_full_flow())

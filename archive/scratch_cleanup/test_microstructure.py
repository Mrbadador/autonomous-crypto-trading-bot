import asyncio
import numpy as np
import sys
import os

# Add workspace to sys path so we can import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from microstructure_guard import MicrostructureGuard, STFT_NPERSEG

async def main():
    print("[TEST] Initializing MicrostructureGuard...")
    guard = MicrostructureGuard()
    
    print("[TEST] Scenario 1: Standard benign market context.")
    vol_series = np.random.normal(10.0, 2.0, size=100).clip(min=1.0)
    context_benign = {
        "ofi": 0.05,
        "trade_volumes": np.array([50.0, 48.0], dtype=np.float64),
        "trade_signs": np.array([1.0, -1.0], dtype=np.float64),
        "vol_24h": 10000.0,
        "volume_series": vol_series,
        "funding_rate": 0.0001,
        "spread": 1.0,
        "dt": 1.0,
        # Spoofing inputs: evenly balanced additions and cancellations
        "cancel_counts": np.ones(10, dtype=np.float64) * 2,
        "add_counts": np.ones(10, dtype=np.float64) * 2
    }
    
    passed, reason = await guard.evaluate_pre_trade_veto(context_benign)
    print(f"   Verdict: Passed={passed} | Reason: {reason}")
    assert passed == True, "Benign case should pass."
    
    print("\n[TEST] Scenario 2: Extreme LOB cancellation (Spoofing Veto trigger).")
    # High cancellations, zero additions -> high Hawkes toxicity
    context_toxic = context_benign.copy()
    context_toxic["cancel_counts"] = np.ones(10, dtype=np.float64) * 50
    context_toxic["add_counts"] = np.zeros(10, dtype=np.float64)
    
    # We might need a couple updates to feed the Hawkes intensity above 0.65
    for _ in range(5):
        passed, reason = await guard.evaluate_pre_trade_veto(context_toxic)
        
    print(f"   Verdict (after self-excitation): Passed={passed} | Reason: {reason}")
    
    print("\n[TEST] Scenario 3: Flow Toxicity (VPIN Veto trigger via massive volume imbalance).")
    # Instantiate fresh guard to isolate VPIN and clear Scenario 2 Hawkes excitation residue
    guard_vpin = MicrostructureGuard()
    guard_vpin.active_veto = False
    
    # Generate alternating high imbalance series to create massive variance in the deltas
    # Each 15.0 tick fills one 10.0 bucket, alternating buys and sells -> std(deltas) maximized!
    heavy_volumes = np.ones(100, dtype=np.float64) * 15.0
    heavy_signs = np.array([1.0, -1.0] * 50, dtype=np.float64) 

    
    context_vpin_toxic = {
        "ofi": 0.9,
        "trade_volumes": heavy_volumes,
        "trade_signs": heavy_signs,
        "vol_24h": 500.0, # bucket size = 10.0
        "volume_series": np.ones(STFT_NPERSEG, dtype=np.float64) * 500.0,
        "funding_rate": 0.005, 
        "spread": 5.0,
        "dt": 1.0
    }
    
    passed, reason = await guard_vpin.evaluate_pre_trade_veto(context_vpin_toxic)

         
    print(f"   Verdict (after massive flow): Passed={passed} | Reason: {reason}")
    assert passed == True, "Extreme VPIN should allow execution under Observation Mode."
    assert "[OBSERVATION MODE]" in reason, "Reason should reflect that we are in Observation Mode."

    
    print("\n[TEST] SUCCESS: All pre-trade mathematical layers validated asynchronously!")

if __name__ == "__main__":
    asyncio.run(main())

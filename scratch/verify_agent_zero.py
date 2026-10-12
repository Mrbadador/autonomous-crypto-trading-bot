# scratch/verify_agent_zero.py
import os
import sys
import numpy as np
import torch

# Set sys.path to include parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.adversarial_simulator import Generator, Discriminator, KillZoneDetector, perturb_lob_fast
from agent_zero_trainer import LOBStateSynthesizer

def verify():
    print("=== STARTING ADVERSARIAL SYSTEM HARNESS VERIFICATION ===")
    
    # 1. Test Synthesizer dimensions
    print("[1] Testing LOBStateSynthesizer...")
    synthesizer = LOBStateSynthesizer()
    
    # Mock row from SQLite
    mock_row = (1, "2026-05-14T23:00:00+00:00", 35.0, 1.2, 0.4, 80000.0, 65.0, 0.55)
    
    try:
        states = synthesizer.synthesize(mock_row, num_samples=5)
        print(f"--> Synthesized batch shape: {states.shape} (Expected: (5, 37))")
        assert states.shape == (5, 37), f"Synthesizer shape mismatch: {states.shape}"
        
        # Basic assertions on physical bounds
        for s in states:
            assert s[0] < s[1], "Best Bid must be less than Best Ask!"
            assert abs(s[0] + s[1])/2.0 - 80000.0 < 200.0, "Mid point far from baseline VWAP!"
        print("--> Baseline physical plausibility check: PASSED")
    except Exception as e:
        print(f"--> FAILED Synthesizer test: {e}")
        raise

    # 2. Test Neural Dimensions
    print("[2] Testing PyTorch Tensor Flows...")
    generator = Generator(z_dim=32, regime_dim=3, action_dim=24)
    discriminator = Discriminator(state_dim=37)
    kill_detector = KillZoneDetector(state_dim=37, action_dim=24)
    
    z = torch.randn(5, 32)
    regime = torch.zeros(5, 3)
    regime[:, 0] = 1.0 # Low Vol label
    
    try:
        continuous, wash = generator(z, regime)
        print(f"--> Generator output: continuous={continuous.shape}, wash={wash.shape}")
        assert continuous.shape == (5, 24), "Gen continuous head shape mismatch!"
        assert wash.shape == (5, 1), "Gen discrete head shape mismatch!"
        
        state_tensor = torch.tensor(states, dtype=torch.float32)
        realism, vulnerability = discriminator(state_tensor)
        print(f"--> Discriminator outputs: realism={realism.shape}, vulnerability={vulnerability.shape}")
        assert realism.shape == (5, 1), "Realism shape mismatch!"
        
        kill_prob = kill_detector(state_tensor, continuous)
        print(f"--> KillZoneDetector output: {kill_prob.shape}")
        assert kill_prob.shape == (5, 1), "KillZone shape mismatch!"
        
        print("--> Neural Tensor dimension checks: ALL PASSED")
    except Exception as e:
        print(f"--> FAILED Neural test: {e}")
        raise

    # 3. Test typed Numba function
    print("[3] Testing compiled Numba speed path...")
    np_base = states[0]
    np_perturb = continuous[0].detach().numpy()
    
    try:
        np_toxic = perturb_lob_fast(np_base, np_perturb, epsilon_max=0.05)
        print(f"--> Original mid-price: {(np_base[0]+np_base[1])/2.0:.2f}")
        print(f"--> Toxic mid-price:    {(np_toxic[0]+np_toxic[1])/2.0:.2f}")
        
        assert np_toxic.shape == (37,), "Numba output length mismatch!"
        
        # Test that volume logic processed
        for i in range(10):
            vol_idx = 2 + i * 2 + 1
            assert np_toxic[vol_idx] >= 0.0, f"Negative volume generated at idx {vol_idx}!"
            
        print("--> Numba Nopython execution: PASSED SUCCESSFULLY")
    except Exception as e:
        print(f"--> FAILED Numba verification: {e}")
        raise
        
    print("=== HARNESS VERIFICATION 100% COMPLETED ===")

if __name__ == "__main__":
    verify()

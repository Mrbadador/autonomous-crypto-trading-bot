# scratch/verify_overlord.py
import os
import sys
import torch

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.overlord import OverlordMetaController, RewardComposer, OptionNetwork
from swarm_orchestrator import SwarmOrchestrator

def verify():
    print("=== STARTING OVERLORD SYSTEM VERIFICATION ===")
    
    # 1. Verify neural models instantiate
    print("[1] Testing Overlord Network architecture instantiation...")
    state_dim = 58
    num_options = 4
    num_sub_swarms = 4
    hidden_dim = 64
    
    try:
        overlord = OverlordMetaController(
            state_dim=state_dim, 
            num_options=num_options, 
            num_sub_swarms=num_sub_swarms, 
            hidden_dim=hidden_dim
        )
        print("--> OverlordMetaController successfully built.")
    except Exception as e:
        print(f"--> FAILED building OverlordMetaController: {e}")
        raise

    # 2. Verify simplex projection and actions
    print("[2] Testing forward pass and simplex validation...")
    mock_state = torch.randn(1, state_dim)
    try:
        option_idx, log_prob = overlord.select_option(mock_state, epsilon=0.0)
        print(f"--> Selected Option ID: {option_idx}")
        
        meta_action = overlord.get_meta_action(mock_state, option_idx)
        weights = meta_action['reward_weights']
        weight_sum = weights.sum(dim=-1).item()
        
        print(f"--> Output weights: {weights.detach().numpy()}")
        print(f"--> Weights Sum: {weight_sum:.6f} (Target: 1.0)")
        assert abs(weight_sum - 1.0) < 1e-5, f"Simplex projection violated: Sum is {weight_sum}"
        print("--> Simplex constraints preserved flawlessly.")
    except Exception as e:
        print(f"--> FAILED forward pass verification: {e}")
        raise

    # 3. Verify Reward Composer
    print("[3] Testing RewardComposer template configuration...")
    composer = RewardComposer()
    trending_weights = composer.regime_template('trending')
    low_vol_weights = composer.regime_template('low_vol')
    
    print(f"--> Trending weights template: {trending_weights.numpy()}")
    print(f"--> Neutral/LowVol weights template: {low_vol_weights.numpy()}")
    assert abs(trending_weights.sum().item() - 1.0) < 1e-5, "Template sum error"
    print("--> Preset templates verified correctly.")

    # 4. Verify Execution Pipeline linkage
    print("[4] Testing swarm pipeline propagation to Executor agent...")
    orchestrator = SwarmOrchestrator()
    
    mock_btc_price = 65000.00
    # Override decision pipeline or mock execution
    print("--> Executing custom run_cycle pass...")
    try:
        orchestrator.run_cycle(mock_btc_price, reward_weights=low_vol_weights)
        
        # Read back stored weights from executor
        assert hasattr(orchestrator.executor, 'reward_weights'), "Executor missed reward_weights property!"
        stored = orchestrator.executor.reward_weights
        print(f"--> Readback from Executor: {stored.numpy()}")
        assert torch.equal(stored, low_vol_weights), "Stored weights do not match input!"
        print("--> Pipeline propagation fully functional!")
    except Exception as e:
        print(f"--> FAILED pipeline test: {e}")
        raise

    print("=== OVERLORD VERIFICATION 100% SUCCESSFUL ===")

if __name__ == "__main__":
    verify()

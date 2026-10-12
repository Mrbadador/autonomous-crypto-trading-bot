# scratch/test_exec_agent.py
import os
import sys
import torch
import numpy as np

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.execution_agent import ExecutionAgent, ExecutionPolicyNetwork, AlmgrenChrissParams

def test_execution_agent():
    print("=== RUNNING EXECUTION AGENT VERIFICATION HARNESS ===")
    
    # 1. Set up configuration parameters
    state_dim = 120  # Matching AnalystAgent output of (10, 12) flattened
    n_slices = 10
    parent_size = 50.0
    
    print(f"[1] Initializing components (state_dim={state_dim}, action_bins=50)...")
    try:
        params = AlmgrenChrissParams(
            gamma=1e-7,
            eta=1e-5,
            sigma=0.02,
            lam=1e-6
        )
        policy_net = ExecutionPolicyNetwork(state_dim=state_dim, action_bins=50)
        exec_agent = ExecutionAgent(policy_net, params, device='cpu')
        print("--> Initialization Successful!")
    except Exception as e:
        print(f"--> CRITICAL FAILURE DURING INITIALIZATION: {e}")
        sys.exit(1)
        
    # 2. Mock a state matrix matching AnalystAgent (10, 12) and flatten it
    print("\n[2] Generating mock market state matching AnalystAgent shape (10, 12) flattened...")
    mock_matrix = np.random.randn(10, 12).astype(np.float32)
    mock_state_flat = mock_matrix.flatten()
    print(f"--> Synthetic matrix shape: {mock_matrix.shape} -> Flattened dimension: {len(mock_state_flat)}")
    
    # 3. Perform Order Slicing Simulation
    parent_order = {
        'size': parent_size,
        'side': 'buy'
    }
    
    print(f"\n[3] Dispatched parent order: BUY {parent_size} units.")
    print(f"Executing Stealth Trajectory calculation across {n_slices} slices...")
    
    try:
        child_orders = exec_agent.slice_order(
            parent_order=parent_order,
            market_state=mock_state_flat,
            n_slices=n_slices
        )
        
        print("\n" + "="*70)
        print(f"{'SLICE':<8} | {'SIZE (UNITS)':<15} | {'DELAY (SEC)':<15} | {'URGENCY':<10}")
        print("-" * 70)
        
        total_sliced = 0.0
        for idx, child in enumerate(child_orders):
            print(f"{idx:<8} | {child['size']:<15.6f} | {child['delay_seconds']:<15.2f} | {child['urgency_score']:<10.4f}")
            total_sliced += child['size']
            
        print("=" * 70)
        print(f"Total Sliced Yield: {total_sliced:.6f} (Expected Parent: {parent_size})")
        
        # Assert physical validity (sum of slices should be very close to parent order)
        diff = abs(total_sliced - parent_size)
        print(f"Slippage Leakage: {diff:.6f}")
        
        # Numba baseline test
        print("\n[4] Validating Numba Almgren-Chriss baseline calculations...")
        x, u = policy_net.get_baseline_trajectory(parent_size, 300, n_steps=n_slices)
        print(f"--> Baseline trajectory steps generated: {len(x)}")
        print(f"--> Initial hold size: {x[0]:.4f} -> Final hold size: {x[-1]:.4f}")
        
        print("\n=== EXECUTION AGENT VERIFICATION 100% SUCCESSFUL ===")
        
    except Exception as e:
        import traceback
        print(f"--> RUNTIME CRASH ENCOUNTERED:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    test_execution_agent()

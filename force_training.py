# force_training.py — V15 AI Policy Optimization Force Runner

import os
import json
import sys
import numpy as np

# Terminal Quant Style Banner
print("=" * 65)
print("      V15 AI COGNITIVE AGENT: FORCE OPTIMIZATION RUN (BRAIN BAKE)  ")
print("=" * 65)

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[CRITICAL] PyTorch is not installed. Force training requires PyTorch module.")
    sys.exit(1)

from strategist_agent import LSTMPolicyNetwork

# 1. Initialize policy model and load weights
model_path = "policy_model.pth"
policy = LSTMPolicyNetwork(input_size=12, hidden_size=64, output_dim=3)

if os.path.exists(model_path):
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        policy.load_state_dict(state_dict)
        print(f"[INIT] Loaded active weights from '{model_path}' successfully.")
    except Exception as e:
        print(f"[WARNING] Failed to load '{model_path}', initializing fresh network. Details: {e}")
else:
    print(f"[INIT] Model file '{model_path}' not found on disk. Initializing fresh network weights.")

# 2. Load Memories from experience_replay.json
replay_path = "experience_replay.json"
memories = []
if os.path.exists(replay_path):
    try:
        with open(replay_path, "r") as f:
            memories = json.load(f)
        print(f"[DATA] Successfully loaded {len(memories)} experiences from '{replay_path}'.")
    except Exception as e:
        print(f"[ERROR] Failed to read '{replay_path}': {e}")
else:
    print(f"[WARNING] '{replay_path}' not found. Generating mock training experiences.")
    # Generate some mock experience if empty
    memories = [
        {"state": [0.5, 0.2, 0.1, -0.1, 0.3, 0.4, -0.2], "action_id": 1, "reward": 1.5},
        {"state": [-0.3, 0.8, -0.5, 0.6, -0.2, -0.3, 0.1], "action_id": 2, "reward": -1.0},
        {"state": [0.1, -0.2, 0.3, -0.4, 0.2, 0.0, 0.0], "action_id": 0, "reward": 0.5},
    ]

if not memories:
    print("[ERROR] No memories available for training. Exiting.")
    sys.exit(1)

# Prepare training tensors with (10, 12) tiling
states_list = []
actions_list = []
rewards_list = []

for idx, mem in enumerate(memories):
    raw_state = mem.get("state", [0.0] * 12)
    # Force/pad state to exactly 12 dimensions
    if len(raw_state) < 12:
        padded_state = raw_state + [0.0] * (12 - len(raw_state))
    else:
        padded_state = raw_state[:12]
        
    tiled_state = np.tile(padded_state, (10, 1))
    states_list.append(tiled_state)
    actions_list.append(mem.get("action_id", 0))
    rewards_list.append(mem.get("reward", 0.0))

states_tensor = torch.tensor(np.array(states_list, dtype=np.float32), dtype=torch.float32)
actions_tensor = torch.tensor(actions_list, dtype=torch.long)
rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32)

# 3. Setup optimizer and loss calculations
optimizer = optim.Adam(policy.parameters(), lr=0.01)
epochs = 100
starting_loss = None
ending_loss = None

policy.train()
print("\n>>> STARTING NEURAL BRAIN BAKE (100 EPOCHS)...")

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    
    # Forward pass to get action logits and param logits
    logits, param_logits = policy(states_tensor)
    
    # Calculate policy loss: -log_prob(action) * reward
    log_probs = F.log_softmax(logits, dim=1)
    action_log_probs = log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
    
    # Minimize negative policy reward (maximize reward)
    action_loss = -(action_log_probs * rewards_tensor).mean()
    
    # Param loss (continuous head)
    param_loss = -torch.tanh(param_logits).mean(dim=-1) * rewards_tensor
    param_loss = param_loss.mean()
    
    loss = action_loss + (param_loss * 0.25)
    
    loss.backward()
    optimizer.step()
    
    if epoch == 1:
        starting_loss = loss.item()
    if epoch == epochs:
        ending_loss = loss.item()
        
    if epoch % 20 == 0:
        print(f"    Epoch {epoch:3d}/100 | Policy Loss: {loss.item():.6f}")

# 4. Save optimized weights back to policy_model.pth
try:
    tmp_path = model_path + ".tmp"
    torch.save(policy.state_dict(), tmp_path)
    import os
    os.replace(tmp_path, model_path)
    print(f"\n[SAVE] Saved optimized policy weights to '{model_path}' successfully.")
except Exception as e:
    print(f"[CRITICAL] Failed to write updated weights to '{model_path}': {e}")
    sys.exit(1)

# 5. Terminal Output: Summary
print("\n" + "=" * 65)
print("               OPTIMIZATION RUN METRICS SUMMARY")
print("=" * 65)
print(f"  • Memories Loaded     : {len(memories)}")
print(f"  • Observation Tensors : Shape {list(states_tensor.shape)}")
print(f"  • Starting Loss       : {starting_loss:.6f}")
print(f"  • Ending Loss         : {ending_loss:.6f}")
print(f"  • Net Loss Delta      : {(starting_loss - ending_loss):+.6f}")
print("  • Neural Weight Status: OVERWRITTEN & OPTIMIZED (V15 BRAIN RE-ORIENTED)")
print("=" * 65)
print("[SYSTEM] Force Training execution complete. Sentinel HUD will update weights on next reload.\n")

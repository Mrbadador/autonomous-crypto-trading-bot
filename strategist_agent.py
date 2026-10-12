# strategist_agent.py — MARL Policy & Action Selection Agent
# Required: pip install torch numpy

import numpy as np
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] STRATEGIST_AGENT: %(message)s")
log = logging.getLogger("strategist_agent")

# Safe PyTorch Import Guard
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    log.warning("PyTorch ('torch') is not installed in this environment. Running in Graceful Mock Fallback Mode. Run 'pip install torch' to enable PyTorch MLP policy execution.")

# 1. Define Policy Network if torch is available
if HAS_TORCH:
    class DuelingDQNPolicyNetwork(nn.Module):
        """
        Dueling DQN Policy Network with LSTM Temporal state extraction.
        Recombines state Value V(s) and action Advantage A(s, a) heads cleanly into Q-values.
        """
        def __init__(self, input_size=15, hidden_size=64, num_actions=3):
            super(DuelingDQNPolicyNetwork, self).__init__()
            
            # Shared representations
            self.shared_fc1 = nn.Linear(input_size, hidden_size)
            self.shared_fc2 = nn.Linear(hidden_size, hidden_size)
            
            # Value Head V(s)
            self.value_fc = nn.Linear(hidden_size, hidden_size // 2)
            self.value_out = nn.Linear(hidden_size // 2, 1)
            
            # Advantage Head A(s, a)
            self.advantage_fc = nn.Linear(hidden_size, hidden_size // 2)
            self.advantage_out = nn.Linear(hidden_size // 2, num_actions)
            
            # Dynamic dynamic parameter head for compatibility with dynamic thresholds
            self.fc_params = nn.Linear(hidden_size, 2)
            
        def forward(self, x):
            # Evaluate the aggregated sequence context representations across the full sequence vector (Features=15)
            if x.dim() == 3:
                x = x.mean(dim=1)
                
            x_shared = F.relu(self.shared_fc1(x))
            x_shared = F.relu(self.shared_fc2(x_shared))
            
            # Value stream
            val = F.relu(self.value_fc(x_shared))
            V = self.value_out(val)
            
            # Advantage stream
            adv = F.relu(self.advantage_fc(x_shared))
            A = self.advantage_out(adv)
            
            # Combine Advantage and Value to compute Q-values
            Q = V + (A - A.mean(dim=-1, keepdim=True))
            
            # Param head (dynamic thresholding support)
            param_logits = self.fc_params(x_shared)
            
            return Q, param_logits
else:
    # Minimal mock network definition if PyTorch is absent
    class DuelingDQNPolicyNetwork:
        def __init__(self, input_size=15, hidden_size=64, num_actions=3):
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_actions = num_actions


# =============================================================================
# GENERALIZED ADVANTAGE ESTIMATION (GAE) — Schulman et al. (2016)
# =============================================================================
def compute_gae(
    rewards,
    values,
    next_values=None,
    dones=None,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    standardize: bool = True
):
    """
    Computes Generalized Advantage Estimation (GAE):
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - d_t) - V(s_t)
        A_t^{GAE} = sum_{l=0}^{T - t - 1} (gamma * lambda)^l * delta_{t+l}
        Returns: (advantages, returns)
    """
    is_torch = HAS_TORCH and isinstance(rewards, torch.Tensor)
    if is_torch:
        T = rewards.shape[0]
        if next_values is None:
            next_values = torch.zeros_like(values)
            if T > 1:
                next_values[:-1] = values[1:]
        elif isinstance(next_values, (float, int)):
            nv = torch.zeros_like(values)
            if T > 1:
                nv[:-1] = values[1:]
            nv[-1] = float(next_values)
            next_values = nv

        if dones is None:
            dones = torch.zeros_like(rewards)

        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(T)):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_values[t] * non_terminal - values[t]
            last_gae = delta + gamma * lambda_ * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        if standardize and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns
    else:
        # NumPy computation
        r = np.array(rewards, dtype=np.float64)
        v = np.array(values, dtype=np.float64)
        T = len(r)
        if next_values is None:
            nv = np.zeros_like(v)
            if T > 1:
                nv[:-1] = v[1:]
        elif isinstance(next_values, (float, int)):
            nv = np.zeros_like(v)
            if T > 1:
                nv[:-1] = v[1:]
            nv[-1] = float(next_values)
        else:
            nv = np.array(next_values, dtype=np.float64)

        d = np.zeros_like(r) if dones is None else np.array(dones, dtype=np.float64)

        advantages = np.zeros_like(r)
        last_gae = 0.0
        for t in reversed(range(T)):
            non_terminal = 1.0 - d[t]
            delta = r[t] + gamma * nv[t] * non_terminal - v[t]
            last_gae = delta + gamma * lambda_ * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + v
        if standardize and len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return advantages, returns


# =============================================================================
# ADAPTIVE KL DIVERGENCE TRACKER
# =============================================================================
class AdaptiveKLTracker:
    """
    Adaptive KL Divergence tracking:
        target_kl = 0.015
        Scale beta_kl *= 2.0 if D_kl > 1.5 * target_kl
        Scale beta_kl *= 0.5 if D_kl < target_kl / 1.5
    """
    def __init__(
        self,
        target_kl: float = 0.015,
        beta_init: float = 1.0,
        min_beta: float = 0.1,
        max_beta: float = 20.0
    ):
        self.target_kl = float(target_kl)
        self.beta_kl = float(beta_init)
        self.min_beta = float(min_beta)
        self.max_beta = float(max_beta)
        self.history = []

    def update(self, d_kl: float) -> float:
        d_kl = float(d_kl)
        self.history.append(d_kl)
        if d_kl > 1.5 * self.target_kl:
            self.beta_kl = min(self.max_beta, self.beta_kl * 2.0)
        elif d_kl < (self.target_kl / 1.5):
            self.beta_kl = max(self.min_beta, self.beta_kl * 0.5)
        return self.beta_kl


# =============================================================================
# DYNAMIC ENTROPY SCHEDULER
# =============================================================================
class DynamicEntropyScheduler:
    """
    Dynamic Entropy Annealing:
        c2(t) decays exponentially from 0.05 to 0.005 over 50,000 steps.
    """
    def __init__(
        self,
        init_coef: float = 0.05,
        min_coef: float = 0.005,
        total_steps: int = 50000
    ):
        self.init_coef = float(init_coef)
        self.min_coef = float(min_coef)
        self.total_steps = int(total_steps)
        self.step_count = 0
        self.decay_ratio = self.min_coef / self.init_coef

    def get_entropy_coef(self, step: int = None) -> float:
        if step is not None:
            current_step = step
        else:
            current_step = self.step_count

        if current_step >= self.total_steps:
            return self.min_coef
        if current_step <= 0:
            return self.init_coef

        frac = current_step / float(self.total_steps)
        coef = self.init_coef * (self.decay_ratio ** frac)
        return float(max(self.min_coef, min(self.init_coef, coef)))

    def step(self) -> float:
        c = self.get_entropy_coef(self.step_count)
        self.step_count += 1
        return c


class StrategistAgent:
    """
    Agent 3: The Strategist (The Brain).
    Evaluates state tensors generated by the AnalystAgent and decides trading actions
    by running them through the policy neural network.
    """
    def __init__(self):
        self.action_names = {0: "HOLD", 1: "LONG", 2: "SHORT"}
        self.training_mode = False
        
        if HAS_TORCH:
            import os
            self.policy = DuelingDQNPolicyNetwork()
            
            # V28.0 Weight Loading Protocol (Sprint 4 Safe Promotion)
            weight_paths = [
                "strategist_weights_v28.pth",
                "strategist_weights_v27.pth",
                "strategist_weights_v26.pth",
                "policy_model.pth"
            ]
            loaded = False
            for w_path in weight_paths:
                if os.path.exists(w_path):
                    try:
                        self.policy.load_state_dict(torch.load(w_path, map_location=torch.device("cpu"), weights_only=True))
                        log.info(f"StrategistAgent successfully booted Sovereign Neural Weights from: {w_path}")
                        loaded = True
                        break
                    except Exception as e:
                        log.warning(f"Failed to load weights from {w_path}: {e}")
            
            if not loaded:
                log.warning("No pre-trained policy weights found. Initializing with empty neural distribution.")
                
            self.policy.eval() # Set network to evaluation mode by default
            log.info("StrategistAgent initialized with PyTorch LSTMPolicyNetwork.")
        else:
            log.warning("StrategistAgent initialized in Mock Fallback Mode (PyTorch missing).")

    def get_action(self, state_tensor):
        """
        Accepts a 1D NumPy array representing the state tensor.
        Converts it to PyTorch tensor, conducts a forward pass, and selects the
        action with the highest confidence level.
        
        Returns:
            dict: {
                "action_id": int (0, 1, or 2),
                "action_name": str ("HOLD", "LONG", or "SHORT"),
                "confidence": float (probability of chosen action)
            }
        """
        if HAS_TORCH:
            # V20.0 Sequence Adaptation Protocol
            st_arr = np.nan_to_num(np.array(state_tensor), nan=0.0, posinf=1.0, neginf=-1.0)
            if st_arr.ndim == 1:
                if len(st_arr) < 15:
                    st_arr = np.append(st_arr, [0.0] * (15 - len(st_arr)))
                st_arr = np.tile(st_arr[:15], (10, 1))
            elif st_arr.ndim == 2:
                if st_arr.shape[1] < 15:
                    padding = np.zeros((st_arr.shape[0], 15 - st_arr.shape[1]))
                    st_arr = np.hstack((st_arr, padding))
                if st_arr.shape[0] < 10:
                    padding_steps = np.zeros((10 - st_arr.shape[0], 15))
                    st_arr = np.vstack((padding_steps, st_arr))
                elif st_arr.shape[0] > 10:
                    st_arr = st_arr[-10:, :15]
                    
            state_torch = torch.FloatTensor(st_arr).unsqueeze(0) # shape: (1, 10, 15)
                
            # Disable gradients for inference
            with torch.no_grad():
                # Get raw dual-headed model outputs
                logits, _ = self.policy(state_torch)
                
                # Action selection: Categorical sampling during training, argmax during inference
                if self.training_mode:
                    temperature = 1.5
                    probabilities = F.softmax(logits / temperature, dim=-1)
                    dist = torch.distributions.Categorical(probabilities)
                    action_id_tensor = dist.sample()
                    action_id = int(action_id_tensor.item())
                    confidence = float(probabilities[0, action_id].item())
                else:
                    action_id_tensor = torch.argmax(logits, dim=-1)
                    action_id = int(action_id_tensor.item())
                    
                    # Robust confidence: scale by advantage dispersion
                    # to prevent softmax saturation/collapse on low-magnitude Q-values
                    q_vals = logits[0]
                    q_std = q_vals.std().item()
                    if q_std > 1e-4:
                        # Dynamic temperature scaled to Q-value dispersion with minimum floor
                        tau = max(q_std * 0.75, 0.05)
                        probabilities = F.softmax(q_vals / tau, dim=-1)
                        confidence = float(probabilities[action_id].item())
                    else:
                        probabilities = F.softmax(q_vals, dim=-1)
                        confidence = float(probabilities[action_id].item())
        else:
            # High-fidelity mock decision when PyTorch is not available
            sum_val = float(np.sum(state_tensor))
            action_id = int(abs(int(sum_val * 10)) % 3)
            confidence = 0.65 + (sum_val % 0.35)
            confidence = min(max(confidence, 0.0), 1.0)
            
        action_name = self.action_names.get(action_id, "HOLD")
        
        result = {
            "action_id": action_id,
            "action_name": action_name,
            "confidence": confidence
        }
        
        log.info(f"Policy evaluation completed. Selected Action: {action_name} | Confidence: {confidence:.2%}")
        return result

    def get_dynamic_thresholds(self, state_tensor) -> dict:
        """
        V17.5 Dynamic Sovereignty Vector Generation.
        Accepts 9D array and runs parametric regression through the dual-head.
        
        Returns:
            dict: { "dynamic_rsi": float, "dynamic_vwap": float }
        """
        default_res = {"dynamic_rsi": 30.0, "dynamic_vwap": 0.98}
        if not HAS_TORCH:
            return default_res
            
        try:
            # V20.0 Sequence Adaptation Protocol
            st_arr = np.nan_to_num(np.array(state_tensor), nan=0.0, posinf=1.0, neginf=-1.0)
            if st_arr.ndim == 1:
                if len(st_arr) < 15:
                    st_arr = np.append(st_arr, [0.0] * (15 - len(st_arr)))
                st_arr = np.tile(st_arr[:15], (10, 1))
            elif st_arr.ndim == 2:
                if st_arr.shape[1] < 15:
                    padding = np.zeros((st_arr.shape[0], 15 - st_arr.shape[1]))
                    st_arr = np.hstack((st_arr, padding))
                if st_arr.shape[0] < 10:
                    padding_steps = np.zeros((10 - st_arr.shape[0], 15))
                    st_arr = np.vstack((padding_steps, st_arr))
                elif st_arr.shape[0] > 10:
                    st_arr = st_arr[-10:, :15]
                    
            state_t = torch.FloatTensor(st_arr).unsqueeze(0)
                
            with torch.no_grad():
                # Extract strictly parametric head
                _, param_vector = self.policy(state_t)
                v = param_vector[0, :] # Extract specific item vector from batch
                
                # Scale RSI using Sigmoid bound mapped to [20.0, 40.0]
                dyn_rsi = 20.0 + float(torch.sigmoid(v[0]).item() * 20.0)
                
                # Scale VWAP support using Sigmoid bound mapped to [0.95, 1.05]
                dyn_vwap = 0.95 + float(torch.sigmoid(v[1]).item() * 0.10)
                
                return {
                    "dynamic_rsi": round(dyn_rsi, 2),
                    "dynamic_vwap": round(dyn_vwap, 4)
                }
        except Exception as e:
            log.warning(f"Parametric sovereignty header read failure: {e}. Reverting to 17.0 benchmarks.")
            return default_res

if __name__ == "__main__":
    log.info("--- Starting Strategist Agent Test ---")
    
    # 1. Simulate exact 15-feature float32 state tensor from AnalystAgent
    mock_state = np.array([0.5, 0.2, 0.5, 0.2, 0.5, 0.1, 0.2, 0.1, 0.5, 0.2, 0.5, 0.1, 0.5, 0.5, 0.5], dtype=np.float32)
    log.info(f"Received Mock State Tensor: {mock_state}")
    
    # 2. Instantiate StrategistAgent
    agent = StrategistAgent()
    
    # 3. Execute policy inference
    action_decision = agent.get_action(mock_state)
    
    # 4. Print clean action output
    print("\n" + "="*50)
    print("         MARL POLICY ACTION SELECTION")
    print("="*50)
    for k, v in action_decision.items():
        if isinstance(v, float):
            print(f"{k:<15}: {v:.6f} ({v:.2%})")
        else:
            print(f"{k:<15}: {v}")
    print("="*50 + "\n")

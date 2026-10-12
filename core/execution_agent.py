import torch
import torch.nn as nn
import numpy as np
from numba import jit

class AlmgrenChrissParams:
    """Market impact parameters (calibrated per asset)."""
    def __init__(self, gamma=1e-7, eta=1e-5, sigma=0.02, lam=1e-6):
        self.gamma = gamma   # Permanent impact ($/share^2)
        self.eta = eta       # Temporary impact ($·s/share^2)
        self.sigma = sigma   # Volatility ($/share/√s)
        self.lam = lam       # Risk aversion (1/$^2)
        self.kappa = np.sqrt(lam * sigma**2 / eta)

@jit(nopython=True)
def optimal_trajectory(X, T, kappa, n_steps=100):
    """Numba-optimized Almgren-Chriss optimal trajectory."""
    dt = T / n_steps
    x = np.zeros(n_steps)
    u = np.zeros(n_steps)
    
    for i in range(n_steps):
        t = i * dt
        if kappa * T > 1e-6:
            x[i] = X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)
        else:
            x[i] = X * (1 - t / T)
        if i > 0:
            u[i] = (x[i-1] - x[i]) / dt
    
    u[0] = u[1] if n_steps > 1 else X / T
    return x, u

class ExecutionCostCalculator:
    """Vectorized cost computation for RL reward shaping."""
    def __init__(self, params: AlmgrenChrissParams):
        self.params = params
        
    def compute_cost(self, trajectory, dt):
        x = trajectory[:, :-1]
        dx = trajectory[:, 1:] - trajectory[:, :-1]
        u = -dx / dt
        
        permanent = self.params.gamma * u * x * dt
        temporary = self.params.eta * u**2 * dt
        risk = self.params.lam * self.params.sigma**2 * x**2 * dt
        
        return (permanent + temporary + risk).sum(dim=1)

class ExecutionPolicyNetwork(nn.Module):
    """Deep RL policy for optimal execution slicing."""
    def __init__(self, state_dim=58, hidden_dim=256, action_bins=50):
        super().__init__()
        self.action_bins = action_bins
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_bins)
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.baseline_kappa = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, state, remaining_position, time_remaining, market_vol):
        x_norm = remaining_position / 100.0
        t_norm = time_remaining / 3600.0
        urgency = torch.abs(x_norm) / (t_norm + 1e-6)
        
        exec_state = torch.cat([
            state, x_norm.unsqueeze(-1), t_norm.unsqueeze(-1), urgency.unsqueeze(-1)
        ], dim=-1)
        
        features = self.state_encoder(exec_state)
        advantage = self.advantage(features)
        value = self.value(features)
        
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))
    
    def get_baseline_trajectory(self, X, T, n_steps=100):
        kappa = torch.abs(self.baseline_kappa).item()
        x, u = optimal_trajectory(X, T, kappa, n_steps)
        return torch.tensor(x), torch.tensor(u)

class ExecutionAgent:
    """RL Agent that slices parent orders into child orders."""
    def __init__(self, policy_net, params, device='cpu'):
        self.policy = policy_net.to(device)
        self.params = params
        self.device = device
        self.cost_calc = ExecutionCostCalculator(params)
        
    def slice_order(self, parent_order, market_state, n_slices=10):
        X = parent_order['size']
        T = 300  # 5 minute default horizon
        side = 1 if parent_order['side'] == 'buy' else -1
        
        x_baseline, u_baseline = self.policy.get_baseline_trajectory(abs(X), T, n_steps=n_slices)
        child_orders = []
        remaining = X
        
        for i in range(n_slices):
            t_rem = T * (1 - i / n_slices)
            state_tensor = torch.tensor(market_state, device=self.device).float()
            
            q_vals = self.policy(
                state_tensor.unsqueeze(0),
                torch.tensor([remaining], device=self.device).float(),
                torch.tensor([t_rem], device=self.device).float(),
                torch.tensor([self.params.sigma], device=self.device).float()
            )
            
            baseline_rate = u_baseline[i].item() if i < len(u_baseline) else remaining / t_rem
            action_idx = q_vals.argmax().item()
            action_rate = self._idx_to_rate(action_idx, baseline_rate)
            
            slice_size = action_rate * (T / n_slices) * side
            slice_size = np.clip(slice_size, -abs(remaining), abs(remaining))
            
            child_orders.append({
                'size': slice_size,
                'delay_seconds': i * (T / n_slices),
                'urgency_score': abs(slice_size) / (abs(remaining) + 1e-6)
            })
            remaining -= slice_size
            
        return child_orders
    
    def _idx_to_rate(self, idx, baseline, max_deviation=0.3):
        rates = np.linspace(baseline * (1 - max_deviation), baseline * (1 + max_deviation), self.policy.action_bins)
        return rates[idx]

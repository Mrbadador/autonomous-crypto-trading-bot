# core/overlord.py — V28.9 Hierarchical RL Overlord Meta-Controller & Reward Composer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Dict, Tuple

class OptionNetwork(nn.Module):
    """
    Represents a single option: initiation, intra-option policy, termination.
    In our case, options correspond to "regime-specific reward templates".
    """
    def __init__(
        self,
        state_dim: int = 58,
        action_dim: int = 11,
        option_id: int = 0
    ):
        super().__init__()
        self.option_id = option_id
        
        # Initiation network: is this option available?
        self.initiation = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Intra-option policy: what meta-action to take
        self.policy_mean = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.policy_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Termination network: should we switch options?
        self.termination = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns all option components."""
        return {
            'initiation': self.initiation(state),
            'policy_mean': self.policy_mean(state),
            'policy_std': torch.exp(self.policy_logstd).expand_as(self.policy_mean(state)),
            'termination': self.termination(state)
        }


class OverlordMetaController(nn.Module):
    """
    Hierarchical Meta-Controller managing multiple Berserker sub-swarms.
    Implements Options Framework with continuous action spaces.
    """
    def __init__(
        self,
        state_dim: int = 58,
        num_options: int = 4,  # HighVol, LowVol, Trending, MeanRev
        num_sub_swarms: int = 4,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_options = num_options
        self.num_sub_swarms = num_sub_swarms
        
        # Shared state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Option set: each option is a reward-weight template + sub-swarm selector
        self.options = nn.ModuleList([
            OptionNetwork(state_dim, num_sub_swarms + 5, i)  # +5 for reward weights
            for i in range(num_options)
        ])
        
        # Meta-policy: which option to select
        self.meta_policy = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_options)
        )
        
        # Value function for advantage estimation
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Reward weight projector: ensures simplex constraint
        self.reward_projector = nn.Sequential(
            nn.Linear(5, 5),
            nn.Softmax(dim=-1)
        )
    
    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Encode raw state into latent representation."""
        return self.state_encoder(state)
    
    def select_option(
        self,
        state: torch.Tensor,
        epsilon: float = 0.1
    ) -> Tuple[int, torch.Tensor]:
        """
        Epsilon-greedy option selection with availability masking.
        Returns: (option_idx, log_prob)
        """
        encoded = self.encode_state(state)
        logits = self.meta_policy(encoded)
        
        # Mask unavailable options
        availabilities = torch.stack([
            opt(state)['initiation'].squeeze(-1) for opt in self.options
        ], dim=-1)
        
        masked_logits = logits + torch.log(availabilities + 1e-8)
        probs = F.softmax(masked_logits, dim=-1)
        
        dist = Categorical(probs)
        if np.random.random() < epsilon:
            option_idx = torch.randint(0, self.num_options, (1,)).item()
        else:
            option_idx = dist.sample().item()
        
        return option_idx, dist.log_prob(torch.tensor(option_idx))
    
    def get_meta_action(
        self,
        state: torch.Tensor,
        option_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Execute intra-option policy.
        Returns: {
            'sub_swarm': (batch, num_sub_swarms) softmax,
            'reward_weights': (batch, 5) simplex,
            'termination_prob': (batch, 1)
        }
        """
        option_out = self.options[option_idx](state)
        
        # Split output: first num_sub_swarms for sub-swarm selection
        # remaining 5 for reward weights
        action_mean = option_out['policy_mean']
        
        sub_swarm_logits = action_mean[:, :self.num_sub_swarms]
        raw_weights = action_mean[:, self.num_sub_swarms:self.num_sub_swarms + 5]
        
        # Apply constraints
        sub_swarm = F.softmax(sub_swarm_logits, dim=-1)
        reward_weights = self.reward_projector(raw_weights)
        
        return {
            'sub_swarm': sub_swarm,
            'reward_weights': reward_weights,
            'termination_prob': option_out['termination'],
            'initiation_prob': option_out['initiation'],
            'option_idx': torch.tensor(option_idx)
        }
    
    def compute_advantage(
        self,
        state: torch.Tensor,
        option_idx: int,
        rewards: torch.Tensor,
        next_state: torch.Tensor,
        dones: torch.Tensor = None,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        standardize: bool = True
    ) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation (GAE) for meta-policy update.
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - d_t) - V(s_t)
        A_t = delta_t + (gamma * lambda) * (1 - d_t) * A_{t+1}
        """
        current_value = self.value_head(self.encode_state(state)).squeeze(-1)
        next_value = self.value_head(self.encode_state(next_state)).squeeze(-1)
        
        if dones is None:
            dones = torch.zeros_like(rewards)
            
        # If input is 1D tensor (single step or trajectory)
        if rewards.dim() == 1:
            T = rewards.shape[0]
            advantages = torch.zeros_like(rewards)
            last_gae = 0.0
            for t in reversed(range(T)):
                non_terminal = 1.0 - dones[t]
                delta = rewards[t] + gamma * next_value[t] * non_terminal - current_value[t]
                last_gae = delta + gamma * lambda_ * non_terminal * last_gae
                advantages[t] = last_gae
        else:
            delta = rewards + gamma * next_value * (1.0 - dones) - current_value
            advantages = delta
            
        if standardize and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        return advantages
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full forward pass for training."""
        encoded = self.encode_state(state)
        
        # Meta-policy
        meta_logits = self.meta_policy(encoded)
        meta_probs = F.softmax(meta_logits, dim=-1)
        
        # Value
        value = self.value_head(encoded)
        
        return {
            'meta_logits': meta_logits,
            'meta_probs': meta_probs,
            'value': value,
            'encoded_state': encoded
        }


# =============================================================================
# POTENTIAL-BASED REWARD SHAPING (PBRS) — POLICY INVARIANCE (Ng et al., 1999)
# =============================================================================
class PotentialBasedRewardShaper:
    """
    Potential-Based Reward Shaping (PBRS) mathematically guaranteeing that
    the optimal policy pi* under shaped reward is invariant to the original MDP:
        Phi(s) = w1 * Sortino(s) - w2 * (Q_inv / MaxMargin)^2 + w3 * (side * Z_ofi(s))
        F(s, a, s') = gamma * Phi(s') - Phi(s)
        R_shaped = R_env + F(s, a, s')
    """
    def __init__(
        self,
        w1: float = 1.0,
        w2: float = 0.5,
        w3: float = 0.25,
        gamma: float = 0.99
    ):
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.w3 = float(w3)
        self.gamma = float(gamma)

    @staticmethod
    def _normalize_side(side) -> float:
        if isinstance(side, (int, float)):
            if side == 1 or side > 0.5:
                return 1.0
            elif side == 2 or side < -0.5:
                return -1.0
            return 0.0
        s = str(side).lower()
        if s in ("long", "buy"):
            return 1.0
        elif s in ("short", "sell"):
            return -1.0
        return 0.0

    def compute_potential(
        self,
        sortino: float = 0.0,
        inventory_qty: float = 0.0,
        max_margin: float = 1000.0,
        side: float = 0.0,
        z_ofi: float = 0.0
    ) -> float:
        """
        Computes scalar state potential Phi(s).
        """
        side_val = self._normalize_side(side)
        safe_margin = max(float(max_margin), 1e-4)
        inv_ratio = abs(float(inventory_qty)) / safe_margin
        
        # Phi(s) = w1 * Sortino - w2 * (Q_inv/MaxMargin)^2 + w3 * (side * Z_ofi)
        phi = (
            self.w1 * float(sortino)
            - self.w2 * (inv_ratio ** 2)
            + self.w3 * (side_val * float(z_ofi))
        )
        return float(phi)

    def compute_potential_tensor(
        self,
        sortino: torch.Tensor,
        inventory_ratio: torch.Tensor,
        side_signed: torch.Tensor,
        z_ofi: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized computation of state potential Phi(s) for batches of tensors.
        """
        return (
            self.w1 * sortino
            - self.w2 * torch.pow(inventory_ratio, 2)
            + self.w3 * (side_signed * z_ofi)
        )

    def shape_reward(
        self,
        r_env: float,
        phi_s: float,
        phi_next: float,
        gamma: float = None
    ) -> float:
        """
        F(s, a, s') = gamma * Phi(s') - Phi(s)
        R_shaped = R_env + F(s, a, s')
        """
        g = self.gamma if gamma is None else float(gamma)
        f_shaping = g * float(phi_next) - float(phi_s)
        return float(r_env + f_shaping)

    def batch_shape_rewards(
        self,
        r_env_batch: torch.Tensor,
        phi_s: torch.Tensor,
        phi_next: torch.Tensor,
        gamma: float = None
    ) -> torch.Tensor:
        """
        Vectorized reward shaping for PyTorch batches.
        """
        g = self.gamma if gamma is None else float(gamma)
        return r_env_batch + (g * phi_next - phi_s)


class RewardComposer:
    """
    Composes sub-agent rewards based on Overlord's weight vector.
    Ensures smooth transitions between reward regimes and incorporates PBRS.
    """
    def __init__(
        self,
        base_rewards: list = ['pnl', 'drawdown', 'sharpe', 'turnover', 'regime'],
        w1: float = 1.0,
        w2: float = 0.5,
        w3: float = 0.25,
        gamma: float = 0.99
    ):
        self.base_rewards = base_rewards
        self.reward_dim = len(base_rewards)
        self.pbrs = PotentialBasedRewardShaper(w1=w1, w2=w2, w3=w3, gamma=gamma)
    
    def compose(
        self,
        raw_rewards: torch.Tensor,  # (batch, reward_dim)
        weights: torch.Tensor       # (batch, reward_dim) from Overlord
    ) -> torch.Tensor:
        """
        Weighted combination with stability constraints.
        """
        # Ensure minimum weight for each component (prevents collapse)
        weights = torch.clamp(weights, min=0.1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        # Smooth composition
        composed = (weights * raw_rewards).sum(dim=-1)
        
        return composed

    def compose_shaped(
        self,
        raw_rewards: torch.Tensor,
        weights: torch.Tensor,
        phi_s: torch.Tensor,
        phi_next: torch.Tensor,
        gamma: float = None
    ) -> torch.Tensor:
        """
        Applies both Overlord weighted composition and PBRS potential differences.
        """
        base_composed = self.compose(raw_rewards, weights)
        return self.pbrs.batch_shape_rewards(base_composed, phi_s, phi_next, gamma=gamma)
    
    def regime_template(self, regime: str) -> torch.Tensor:
        """Get preset reward template for regime."""
        templates = {
            'high_vol': torch.tensor([0.20, 0.40, 0.20, 0.10, 0.10]),
            'low_vol': torch.tensor([0.35, 0.15, 0.30, 0.15, 0.05]),
            'trending': torch.tensor([0.40, 0.15, 0.25, 0.10, 0.10]),
            'mean_rev': torch.tensor([0.25, 0.20, 0.30, 0.15, 0.10])
        }
        return templates.get(regime, templates['low_vol'])


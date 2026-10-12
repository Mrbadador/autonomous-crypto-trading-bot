# core/adversarial_simulator.py — V28.9 Agent-Zero Adversarial LOB Simulator
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numba import jit, float64
from typing import Tuple, Dict

class Generator(nn.Module):
    """
    Conditional Generator for Toxic LOB Scenarios.
    Input: (z_dim + regime_dim) -> Output: LOB perturbation vector
    """
    def __init__(
        self,
        z_dim: int = 32,
        regime_dim: int = 3,
        state_dim: int = 37,
        action_dim: int = 24,
        hidden_dims: list = [256, 512, 256]
    ):
        super().__init__()
        self.z_dim = z_dim
        self.regime_dim = regime_dim
        
        input_dim = z_dim + regime_dim
        
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            ])
            prev = h
        
        self.backbone = nn.Sequential(*layers)
        
        # Head outputs continuous perturbations
        self.continuous_head = nn.Linear(prev, action_dim)
        # Separate head for wash-trade probability
        self.discrete_head = nn.Linear(prev, 1)
        
        # Initialize with small weights for stability
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        z: torch.Tensor,
        regime: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (batch, z_dim) latent noise
            regime: (batch, regime_dim) one-hot regime label
        Returns:
            continuous: (batch, action_dim) bounded perturbations
            wash_prob: (batch, 1) probability of wash-trade injection
        """
        x = torch.cat([z, regime], dim=-1)
        features = self.backbone(x)
        
        continuous = torch.tanh(self.continuous_head(features))  # [-1, 1]
        wash_prob = torch.sigmoid(self.discrete_head(features))
        
        return continuous, wash_prob
    
    def sample_perturbation(
        self,
        batch_size: int,
        regime_idx: int,
        device: str = 'cpu'
    ) -> Dict[str, torch.Tensor]:
        """Convenience method for sampling with regime specification."""
        z = torch.randn(batch_size, self.z_dim, device=device)
        regime = F.one_hot(
            torch.tensor(regime_idx, device=device),
            num_classes=self.regime_dim
        ).float().unsqueeze(0).expand(batch_size, -1)
        
        continuous, wash = self.forward(z, regime)
        
        # Scale to action bounds
        return {
            'delta_mid': continuous[:, 0] * 0.02,
            'delta_spread': continuous[:, 1] * 0.5,
            'delta_depth': continuous[:, 2:22] * 0.3,
            'alpha_corr': continuous[:, 22] * 0.3,
            'wash_prob': wash.squeeze(-1)
        }


class Discriminator(nn.Module):
    """
    Discriminator: Distinguishes real LOB states from Agent-Zero perturbed states.
    Also provides Berserker vulnerability score.
    """
    def __init__(
        self,
        state_dim: int = 37,
        hidden_dims: list = [256, 512, 256]
    ):
        super().__init__()
        
        layers = []
        prev = state_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.LeakyReLU(0.2)
            ])
            prev = h
        
        self.backbone = nn.Sequential(*layers)
        self.realism_head = nn.Linear(prev, 1)  # Real vs Fake
        self.vulnerability_head = nn.Linear(prev, 1)  # Kill zone probability
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (batch, state_dim) LOB state vector
        Returns:
            realism: (batch, 1) probability state is real
            vulnerability: (batch, 1) estimated Berserker vulnerability
        """
        features = self.backbone(state)
        realism = torch.sigmoid(self.realism_head(features))
        vulnerability = torch.sigmoid(self.vulnerability_head(features))
        return realism, vulnerability


class KillZoneDetector(nn.Module):
    """
    Identifies Berserker vulnerability regions in state-action space.
    Trains a classifier on (state, action) -> catastrophic outcome.
    """
    def __init__(self, state_dim: int = 37, action_dim: int = 24):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predict_vulnerability(state, action)
    
    def predict_vulnerability(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Predicts probability of entering the drawdown kill zone.
        """
        x = torch.cat([state, action], dim=-1)
        return self.classifier(x)


@jit(nopython=True, cache=True)
def perturb_lob_fast(
    lob_state: np.ndarray,  # (37,) flattened
    perturbation: np.ndarray,  # (24,) from generator
    epsilon_max: float = 0.05
) -> np.ndarray:
    """
    Vectorized LOB perturbation with strict bounds.
    No pandas. Pure numpy + numba.
    """
    perturbed = lob_state.copy()
    
    # Unpack perturbation safely using native scalar min/max for Numba typing
    p0 = perturbation[0]
    delta_mid = max(min(p0, 0.02), -0.02)
    
    p1 = perturbation[1]
    delta_spread = max(min(p1, 0.5), -0.5)
    
    p22 = perturbation[22]
    alpha_corr = max(min(p22, 0.3), -0.3)
    
    # Apply mid-price shift
    mid = (perturbed[0] + perturbed[1]) / 2.0
    mid_shift = mid * delta_mid
    perturbed[0] += mid_shift  # Best bid
    perturbed[1] += mid_shift  # Best ask
    
    # Apply spread multiplier
    spread = perturbed[1] - perturbed[0]
    new_spread = spread * (1.0 + delta_spread)
    half_spread = new_spread / 2.0
    center = mid + mid_shift
    perturbed[0] = center - half_spread
    perturbed[1] = center + half_spread
    
    # Apply depth perturbations (levels 2-10 per side)
    # Indices 2-21: 10 levels of (price, volume) per side
    for i in range(10):
        vol_idx = 2 + i * 2 + 1  # Volume index
        
        # Grab depth perturbation element-wise and clip
        d_i = perturbation[2 + i]
        d_clipped = max(min(d_i, 0.3), -0.3)
        
        # We multiply current volume by (1.0 + d_clipped)
        perturbed[vol_idx] *= (1.0 + d_clipped)
        # Clip to ensure non-negative
        if perturbed[vol_idx] < 0.0:
            perturbed[vol_idx] = 0.0
            
    # Correlation injection (affects recent trades correlation)
    # Indices 22-36: trade history and features
    # Simple linear correlation injection on trade prices
    if len(perturbed) > 25:
        trade_prices = perturbed[22:25]  # Last 3 trade prices
        # std calculation
        mean_val = 0.0
        for p in trade_prices:
            mean_val += p
        mean_val /= len(trade_prices)
        
        var_val = 0.0
        for p in trade_prices:
            var_val += (p - mean_val)**2
        var_val /= len(trade_prices)
        std_val = var_val**0.5
        
        noise = np.random.randn(len(trade_prices)) * alpha_corr * std_val
        perturbed[22:25] += noise
    
    return perturbed


def generate_wash_trade(
    mid: float,
    spread: float,
    volume: float = 1.0
) -> np.ndarray:
    """Generate synthetic wash trade that executes on both sides."""
    half_spread = spread / 2.0
    return np.array([
        mid - half_spread, volume, 1.0,  # Buy at bid
        mid + half_spread, volume, -1.0   # Sell at ask
    ])

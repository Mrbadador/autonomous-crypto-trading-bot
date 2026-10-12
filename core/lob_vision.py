import torch
import torch.nn as nn
import torch.nn.functional as F

class LOBFieldEncoder(nn.Module):
    """Converts 1D LOB arrays into a 4-Channel 2D spatial-temporal field."""
    def __init__(self, num_price_levels=25, time_steps=20):
        super().__init__()
        self.N = num_price_levels 
        self.T = time_steps        
        self.register_buffer('lob_history', torch.zeros(1, self.T, 100))
        
    def forward(self, lob_flat):
        # Shift history and append new state
        self.lob_history = torch.cat([self.lob_history[:, 1:], lob_flat.unsqueeze(1)], dim=1)
        batch, T, _ = self.lob_history.shape
        
        bid_prices, bid_vols = self.lob_history[:, :, 0:25], self.lob_history[:, :, 25:50]
        ask_prices, ask_vols = self.lob_history[:, :, 50:75], self.lob_history[:, :, 75:100]
        mid = (bid_prices[:, :, 0:1] + ask_prices[:, :, 0:1]) / 2
        
        # Build Field: Channels = [Bid Vol, Ask Vol, Distance to Mid, Imbalance]
        field = torch.zeros(batch, 4, T, self.N, device=self.lob_history.device)
        field[:, 0] = bid_vols
        field[:, 1] = ask_vols
        field[:, 2] = torch.abs(bid_prices - mid)
        field[:, 3] = (bid_vols - ask_vols) / (bid_vols + ask_vols + 1e-8)
        
        return field

class ViTPatchEmbed(nn.Module):
    """Volume-weighted patch embedding for the LOB visual field."""
    def __init__(self, patch_size=(4, 5), in_chans=4, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x):
        B = x.shape[0]
        patches = self.proj(x)
        
        # Calculate volume weighting per patch
        x_reshaped = x.unfold(2, self.patch_size[0], self.patch_size[0]).unfold(3, self.patch_size[1], self.patch_size[1])
        volumes = x_reshaped[:, 0:2].sum(dim=(1, 4, 5))
        weights = F.softmax(self.alpha * volumes.reshape(B, -1), dim=-1)
        
        patches = patches.flatten(2).transpose(1, 2)
        return patches, weights

class LOBVisionTransformer(nn.Module):
    """Full Vision Transformer extracting 64-dim visual embeddings from the LOB."""
    def __init__(self, field_shape=(4, 20, 25), patch_size=(4, 5), embed_dim=64, depth=4, num_heads=4):
        super().__init__()
        self.field_encoder = LOBFieldEncoder()
        self.patch_embed = ViTPatchEmbed(patch_size, 4, embed_dim)
        
        num_patches = (field_shape[1] // patch_size[0]) * (field_shape[2] // patch_size[1])
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
    def forward(self, lob_flat):
        field = self.field_encoder(lob_flat)
        patches, vol_weights = self.patch_embed(field)
        patches = patches + self.pos_embed * vol_weights.unsqueeze(-1)
        
        vit_embedding = self.transformer(patches)
        # Global average pooling weighted by volume
        vit_pooled = (vit_embedding * vol_weights.unsqueeze(-1)).sum(dim=1)
        return vit_pooled

class OverlordStateTensorV30(nn.Module):
    """The Master Brain: Fuses all V30.0 modules into a single Policy vector."""
    def __init__(self):
        super().__init__()
        # Dims: Legacy(58) + ViT(64) + Exec(3) + GNN(64) + Flow(4) = 193
        self.state_compressor = nn.Sequential(
            nn.Linear(193, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128)
        )
        self.policy_head = nn.Linear(128, 4)  # Options: [Sniper, Berserker, Execution, GNN-Scout]
        
    def forward(self, legacy_state, vit_emb, exec_state, gnn_emb, flow_state):
        full_state = torch.cat([legacy_state, vit_emb, exec_state, gnn_emb, flow_state], dim=-1)
        compressed = self.state_compressor(full_state)
        policy_probs = F.softmax(self.policy_head(compressed), dim=-1)
        return compressed, policy_probs

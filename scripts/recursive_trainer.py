import os
import sys
import logging
import sqlite3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

# Add base directory to sys path to import local modules
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Attempt importing local model
try:
    from strategist_agent import LSTMPolicyNetwork
except ImportError:
    # Redefine minimal class stub for dependency safety if import fails, but prefer real import
    raise

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] recursive_trainer: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/recursive_trainer.log")
    ]
)
log = logging.getLogger("recursive_trainer")

DB_PATH = os.path.join(BASE_DIR, "trading_data.db")
SOURCE_MODEL = os.path.join(BASE_DIR, "policy_model.pth")
TARGET_MODEL = os.path.join(BASE_DIR, "strategist_weights_v26.pth")

def get_feature_maps():
    regime_map = {
        'TRENDING': 1.0,
        'RANGING': 0.0,
        'CHOP': -1.0,
        'CHAOTIC': -1.0,
        'NEUTRAL': 0.0
    }
    phase_map = {
        'ACCUMULATING': 0.1,
        'STARTING': 0.3,
        'EARLY': 0.5,
        'MATURE': 0.8,
        'LATE': 1.0
    }
    vol_map = {
        'TRENDING': 0.5,
        'CHOP': 1.0,
        'RANGING': 0.2,
        'CHAOTIC': 1.0
    }
    return regime_map, phase_map, vol_map

def run_shadow_training():
    log.info("=== Starting V26.1 Shadow Learner Training ===")
    
    # 1. Load Data
    if not os.path.exists(DB_PATH):
        log.error(f"Database not found: {DB_PATH}")
        return
        
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT side, regime, phase, quality, pnl FROM legacy_trades WHERE pnl IS NOT NULL", conn)
        conn.close()
    except Exception as e:
        log.error(f"Data fetch failed: {e}")
        return
        
    if len(df) == 0:
        log.warning("No data rows found to train on. Aborting.")
        return
        
    log.info(f"Fetched {len(df)} samples for historical replay optimization.")
    
    # 2. Initialize Neural Net
    device = torch.device("cpu")
    policy = LSTMPolicyNetwork().to(device)
    
    if os.path.exists(SOURCE_MODEL):
        try:
            policy.load_state_dict(torch.load(SOURCE_MODEL, map_location=device, weights_only=True))
            log.info(f"Successfully seeded network architecture with {SOURCE_MODEL}")
        except Exception as e:
            log.warning(f"Seeding error: {e}. Proceeding from randomized distribution.")
    else:
        log.info("No source model found. Starting training from zero.")
        
    policy.train()
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    
    # Pre-map tables
    regime_m, phase_m, vol_m = get_feature_maps()
    
    total_loss = 0.0
    processed = 0
    
    log.info("Cycling Epoch: Running Policy Gradient Forward/Backward Passes...")
    
    # 3. Training Loop over historical rows
    for idx, row in df.iterrows():
        # A. Build Input Feature Vector (12-D)
        # Columns based on analyst_agent: [fg, vol, trend, ema, vol_m, hurst, adx, rsi, vwap, sen, spy, dxy]
        regime_str = str(row['regime']).upper() if pd.notna(row['regime']) else 'NEUTRAL'
        phase_str = str(row['phase']).upper() if pd.notna(row['phase']) else ''
        
        val_regime = regime_m.get(regime_str, 0.0)
        val_vol = vol_m.get(regime_str, 0.5)
        val_trend = phase_m.get(phase_str, 0.5)
        
        # Zero padding the remaining inputs as we lack distinct history for them in this table
        # Replicating (1, 10, 13) tiling architecture
        base_vector = np.array([
            0.0,          # Fear Greed (Index 0)
            val_vol,      # Volatility State (Index 1)
            val_trend,    # Trend Strength (Index 2)
            0.0,          # EMA distance
            0.5,          # Vol Multiplier
            val_regime,   # Hurst proxy
            0.5,          # ADX placeholder
            0.0,          # RSI placeholder
            0.0,          # VWAP distance
            0.0,          # Sentiment (Index 9)
            0.0,          # SPY
            0.0,          # DXY
            0.0           # OFI
        ], dtype=np.float32)
        
        # Tile 10 times -> (10, 13)
        tiled = np.tile(base_vector, (10, 1))
        state_tensor = torch.FloatTensor(tiled).unsqueeze(0).to(device) # (1, 10, 13)
        
        # B. Map Taken Action and Reward
        side = str(row['side']).lower()
        action_id = 1 if 'long' in side else 2 # Action ID mapping from core logic
        
        # Reward scaling (prevent extreme gradients, clip extreme wins/losses)
        pnl = float(row['pnl'])
        reward = np.clip(pnl / 10.0, -5.0, 5.0) # Normalized reward scale
        
        # C. Optimize Step
        optimizer.zero_grad()
        
        logits, param_logits = policy(state_tensor)
        
        # Step 1: Action Policy Log-Prob
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        log_prob = torch.log(probs[action_id] + 1e-8)
        action_loss = -log_prob * reward
        
        # Step 2: Parameter Tanh Loss
        param_loss = -torch.tanh(param_logits).mean() * reward
        
        # Step 3: Total Composite Loss (standard scale factor 0.25)
        loss = action_loss + (param_loss * 0.25)
        
        loss.backward()
        optimizer.step()
        
        total_loss += float(loss.item())
        processed += 1
        
        if processed % 500 == 0:
            log.info(f"Checkpoint: Cycle count {processed}/{len(df)}. Running avg loss: {total_loss/processed:.6f}")
            
    log.info("Optimization Epoch complete.")
    log.info(f"Aggregated Policy Gradient Loss Mean: {total_loss/max(1, processed):.6f}")
    
    # 4. Save weights
    log.info(f"Flushing Sovereign V26.1 weights to disk: {TARGET_MODEL}")
    tmp_path = TARGET_MODEL + ".tmp"
    torch.save(policy.state_dict(), tmp_path)
    import os
    os.replace(tmp_path, TARGET_MODEL)
    log.info("Shadow Training Sequence Successful.")

if __name__ == "__main__":
    run_shadow_training()

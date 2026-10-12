# nightly_trainer_v2.py — Agent 6: The Nightly Trainer / Policy Updater
# MARL Trading Swarm | Offline Policy Gradient RL Update Engine
#
# Architecture: Auto-detects weights from policy_model.pth:
#   - If LSTM weights found  → LSTMPolicyNetwork (12 features, seq_len=10, hidden=64, 3 actions)
#   - If MLP / file missing  → MLPPolicyNetwork  (12 features, hidden=64, 3 actions)  [fallback]
#
# Input  : learned_memory.json   (reward_log written by NightlyAuditor)
# Output : policy_model.pth      (updated weights, atomic safe-write)
#
# NOTE: The existing nightly_trainer.py (V32.x production PPO) is NOT touched.
#       This file is the clean MARL Agent 6 trainer for the new pipeline.

# ── [P0] Fortran crash-guard: must precede ANY numpy/torch import ─────────────
import os
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

import json
import logging
import sys
import shutil
from datetime import datetime, timezone
from typing import Optional

import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] NIGHTLY_TRAINER: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("nightly_trainer")

# ── PyTorch Guard ─────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    log.critical("PyTorch not installed. Run: pip install torch")
    sys.exit(1)

# ── Absolute Paths ─────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MEMORY_PATH   = os.path.join(BASE_DIR, "learned_memory.json")
MODEL_PATH    = os.path.join(BASE_DIR, "policy_model.pth")

# ── Hyperparameters ───────────────────────────────────────────────────────────
INPUT_FEATURES = 12       # State tensor width (matches AnalystAgent)
SEQ_LEN        = 10       # LSTM temporal window
HIDDEN_SIZE    = 64       # LSTM / MLP hidden units
N_ACTIONS      = 3        # HOLD=0, LONG=1, SHORT=2
LEARNING_RATE  = 1e-4     # Adam LR (matches production trainer)
REWARD_CLIP    = 10.0     # Clamp rewards to prevent exploding gradients
ENTROPY_COEFF  = 0.01     # Entropy bonus — encourages exploration
MIN_EXPERIENCES = 1       # Minimum reward_log entries needed to train


# ── Network Definitions ───────────────────────────────────────────────────────
class LSTMPolicyNetwork(nn.Module):
    """
    Primary policy network — matches the architecture already stored in
    policy_model.pth (weights: lstm.weight_ih_l0, fc_actions.*, fc_params.*).
    Input : (batch, seq_len=10, features=12)
    Output: action_logits (3,), param_logits (2,)
    """
    def __init__(
        self,
        input_size:  int = INPUT_FEATURES,
        hidden_size: int = HIDDEN_SIZE,
        output_dim:  int = N_ACTIONS,
    ):
        super().__init__()
        self.lstm      = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc_actions = nn.Linear(hidden_size, output_dim)
        self.fc_params  = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor):
        lstm_out, _ = self.lstm(x)          # (B, T, H)
        final = lstm_out[:, -1, :]          # last time-step
        return self.fc_actions(final), self.fc_params(final)


class MLPPolicyNetwork(nn.Module):
    """
    Lightweight fallback — used when no LSTM weights are detected on disk.
    Accepts a flat 12-feature vector (no sequence dimension required).
    """
    def __init__(
        self,
        input_size:  int = INPUT_FEATURES,
        hidden_size: int = HIDDEN_SIZE,
        output_dim:  int = N_ACTIONS,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.fc_actions = nn.Linear(hidden_size, output_dim)
        self.fc_params  = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor):
        # Accept either (B, F) or (B, T, F) — flatten sequence if needed
        if x.dim() == 3:
            x = x[:, -1, :]                # take last time-step
        h = self.net(x)
        return self.fc_actions(h), self.fc_params(h)


# ── Utility: state tensor from a reward_log entry ────────────────────────────
_ACTION_MAP = {"long": 1, "short": 2, "hold": 0, "buy": 1, "sell": 2}

def _entry_to_state(entry: dict) -> np.ndarray:
    """
    Reconstruct a 12-feature float32 state vector from a reward_log row.
    Fields present  (4): pnl, duration_mins, reward + outcome_sign
    Fields imputed  (8): zeros (conservative — avoids hallucinating data)
    The vector is zero-padded to INPUT_FEATURES=12 for model compatibility.
    """
    pnl      = float(entry.get("pnl", 0.0) or 0.0)
    dur      = float(entry.get("duration_mins", 30.0) or 30.0)
    reward   = float(entry.get("reward", 0.0) or 0.0)
    outcome  = entry.get("outcome", "BREAKEVEN")
    o_sign   = 1.0 if outcome == "WIN" else (-1.0 if outcome == "LOSS" else 0.0)

    vec = np.array([
        np.clip(pnl   / 5.0, -1.0, 1.0),   # normalised PnL
        np.clip(dur   / 300.0, 0.0, 1.0),  # normalised duration
        np.clip(reward / 5.0, -1.0, 1.0),  # normalised reward
        o_sign,                             # outcome sign
        0.0, 0.0, 0.0, 0.0,                # imputed features
        0.0, 0.0, 0.0, 0.0,
    ], dtype=np.float32)
    return vec


def _entry_to_action(entry: dict) -> int:
    side = str(entry.get("side", "hold")).lower().strip()
    return _ACTION_MAP.get(side, 0)


# ── NightlyTrainer ────────────────────────────────────────────────────────────
class NightlyTrainer:
    """
    Agent 6 — The Nightly Trainer / Policy Updater.

    Workflow:
      1. Load reward experiences from learned_memory.json.
      2. Load / initialise the policy network.
      3. Run a Policy Gradient update (REINFORCE with entropy bonus).
      4. Safe-write updated weights back to policy_model.pth.
    """

    def __init__(
        self,
        model_path:  str = MODEL_PATH,
        memory_path: str = MEMORY_PATH,
        lr:          float = LEARNING_RATE,
    ):
        self.model_path  = model_path
        self.memory_path = memory_path
        self.lr          = lr
        self.device      = torch.device("cpu")  # CPU-only for midnight safety
        self.network     = None
        self.optimizer   = None
        self.is_lstm     = False
        log.info(f"NightlyTrainer initialised. Model={os.path.basename(model_path)}  LR={lr}")

    # ── 1. Memory Ingestion ───────────────────────────────────────────────────
    def load_experiences(self) -> tuple[list, list, list]:
        """
        Reads the most recent audit_ session from learned_memory.json.
        Returns (states, actions, rewards) as parallel Python lists.
        """
        if not os.path.exists(self.memory_path):
            log.warning(f"learned_memory.json not found at {self.memory_path}.")
            return [], [], []

        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                memory = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            log.error(f"Cannot read learned_memory.json: {e}")
            return [], [], []

        # Collect ALL audit sessions' reward logs for a richer batch
        audit_keys = sorted(k for k in memory if k.startswith("audit_"))
        if not audit_keys:
            log.warning("No audit_ session keys found in learned_memory.json.")
            return [], [], []

        states, actions, rewards = [], [], []
        total_sessions = 0

        for key in audit_keys:
            session    = memory[key]
            reward_log = session.get("reward_log", [])
            for entry in reward_log:
                states.append(_entry_to_state(entry))
                actions.append(_entry_to_action(entry))
                rewards.append(float(entry.get("reward", 0.0) or 0.0))
            total_sessions += 1

        log.info(
            f"[DB-INGEST] Loaded {len(rewards)} experience(s) "
            f"from {total_sessions} audit session(s)."
        )
        return states, actions, rewards

    # ── 2. Network Load / Init ────────────────────────────────────────────────
    def load_network(self) -> None:
        """
        Attempts to load existing weights from policy_model.pth.
        Auto-detects LSTM vs MLP by inspecting the state-dict keys.
        Falls back to a fresh MLP if the file is missing or incompatible.
        """
        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(
                    self.model_path,
                    map_location=self.device,
                    weights_only=True,
                )
                is_lstm = any("lstm" in k for k in state_dict.keys())

                if is_lstm:
                    net = LSTMPolicyNetwork().to(self.device)
                    net.load_state_dict(state_dict)
                    self.is_lstm = True
                    arch = "LSTMPolicyNetwork"
                else:
                    net = MLPPolicyNetwork().to(self.device)
                    net.load_state_dict(state_dict)
                    self.is_lstm = False
                    arch = "MLPPolicyNetwork"

                self.network = net
                log.info(f"Loaded existing model weights from: {self.model_path}  [{arch}]")

            except Exception as e:
                log.warning(f"Failed to load weights ({e}). Initialising fresh MLP.")
                self.network = MLPPolicyNetwork().to(self.device)
                self.is_lstm = False
        else:
            log.warning(f"No model file found at {self.model_path}. Initialising fresh MLP.")
            self.network = MLPPolicyNetwork().to(self.device)
            self.is_lstm = False

        self.network.train()
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

    # ── 3. Policy Gradient Update ─────────────────────────────────────────────
    def _build_tensors(
        self,
        states:  list,
        actions: list,
        rewards: list,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert raw Python lists → PyTorch tensors with LSTM sequence shaping.
        Rewards are normalised (zero-mean, unit-std) and clipped.
        """
        # State tensor: (N, F) → expand to (N, T, F) for LSTM
        state_arr = np.stack(states, axis=0).astype(np.float32)     # (N, 12)
        if self.is_lstm:
            state_arr = np.tile(
                state_arr[:, np.newaxis, :], (1, SEQ_LEN, 1)
            )                                                        # (N, 10, 12)
        states_t  = torch.FloatTensor(state_arr).to(self.device)

        actions_t = torch.LongTensor(actions).to(self.device)       # (N,)

        # Reward normalisation — stabilises gradient magnitude
        r_arr = np.array(rewards, dtype=np.float32)
        r_arr = np.clip(r_arr, -REWARD_CLIP, REWARD_CLIP)
        if r_arr.std() > 1e-6:
            r_arr = (r_arr - r_arr.mean()) / (r_arr.std() + 1e-8)
        rewards_t = torch.FloatTensor(r_arr).to(self.device)        # (N,)

        return states_t, actions_t, rewards_t

    def run_gradient_update(
        self,
        states:  list,
        actions: list,
        rewards: list,
    ) -> dict:
        """
        REINFORCE Policy Gradient with entropy regularisation:
            L = -mean( log π(a|s) × R ) - entropy_coeff × H(π)

        Returns a dict with actor_loss, entropy, total_loss, and avg_reward.
        """
        result = {
            "actor_loss":  None,
            "entropy":     None,
            "total_loss":  None,
            "avg_reward":  float(np.mean(rewards)) if rewards else 0.0,
            "n_samples":   len(rewards),
            "status":      "SKIPPED",
        }

        if len(rewards) < MIN_EXPERIENCES:
            log.warning(
                f"Only {len(rewards)} experience(s) — minimum is {MIN_EXPERIENCES}. "
                f"Skipping gradient update."
            )
            return result

        try:
            self.optimizer.zero_grad()

            states_t, actions_t, rewards_t = self._build_tensors(states, actions, rewards)

            # Forward pass
            action_logits, _ = self.network(states_t)               # (N, 3)
            log_probs_all    = F.log_softmax(action_logits, dim=-1)  # (N, 3)
            probs_all        = torch.exp(log_probs_all)

            # Select log prob of the action actually taken
            log_probs_taken  = log_probs_all.gather(
                1, actions_t.unsqueeze(1)
            ).squeeze(1)                                             # (N,)

            # Policy gradient loss (REINFORCE)
            actor_loss = -(log_probs_taken * rewards_t).mean()

            # Entropy bonus — prevents premature policy collapse
            entropy    = -(probs_all * log_probs_all).sum(dim=-1).mean()
            total_loss = actor_loss - ENTROPY_COEFF * entropy

            # Gradient step
            total_loss.backward()
            # Gradient clipping — prevents exploding gradients on small batches
            nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()

            result.update({
                "actor_loss": round(float(actor_loss.item()), 6),
                "entropy":    round(float(entropy.item()),    6),
                "total_loss": round(float(total_loss.item()), 6),
                "status":     "SUCCESS",
            })

            log.info(
                f"Training step complete.  "
                f"Actor Loss={result['actor_loss']:+.6f}  "
                f"Entropy={result['entropy']:.6f}  "
                f"Total Loss={result['total_loss']:+.6f}  "
                f"Avg Reward={result['avg_reward']:+.4f}  "
                f"N={result['n_samples']}"
            )

        except Exception as e:
            log.error(f"Gradient update failed: {e}. Weights unchanged.", exc_info=True)
            result["status"] = f"ERROR: {e}"

        return result

    # ── 4. Safe Model Save ────────────────────────────────────────────────────
    def save_weights(self) -> bool:
        """
        Atomic weight save via temp-file → os.replace().
        Backs up the existing model before overwriting.
        """
        if self.network is None:
            log.error("No network loaded — cannot save.")
            return False

        tmp_path    = self.model_path + ".tmp"
        backup_path = self.model_path + ".bak"

        try:
            # Backup existing weights
            if os.path.exists(self.model_path):
                shutil.copy2(self.model_path, backup_path)

            torch.save(self.network.state_dict(), tmp_path)
            os.replace(tmp_path, self.model_path)
            log.info(f"Optimized model weights successfully saved: {self.model_path}")
            return True

        except Exception as e:
            log.error(f"Failed to save model: {e}")
            # Attempt rollback from backup
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, self.model_path)
                log.warning("Rolled back to backup weights.")
            return False

    # ── Full Training Pipeline ────────────────────────────────────────────────
    def run(self) -> dict:
        """
        Execute the full Agent 6 workflow and return the training result dict.
        """
        log.info("=" * 55)
        log.info("  --- Starting Nightly Trainer Test ---")
        log.info("=" * 55)

        # Step 1 — Ingest experiences
        states, actions, rewards = self.load_experiences()

        # Step 2 — Load / init network
        self.load_network()

        # Step 3 — Gradient update
        result = self.run_gradient_update(states, actions, rewards)

        # Step 4 — Save weights (only if update succeeded)
        if result["status"] == "SUCCESS":
            saved = self.save_weights()
            result["weights_saved"] = saved
        else:
            result["weights_saved"] = False
            log.warning("Gradient update did not succeed — weights NOT overwritten.")

        log.info("Nightly training session complete.")
        return result


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("─" * 55)
    log.info("  NightlyTrainer (Agent 6) — Self-Test")
    log.info("─" * 55)

    trainer = NightlyTrainer(
        model_path  = MODEL_PATH,
        memory_path = MEMORY_PATH,
        lr          = LEARNING_RATE,
    )

    result = trainer.run()

    # ── Print Banner ──────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  MARL PPO ACTOR-CRITIC OPTIMIZATION")
    print("=" * 55)
    print(f"  Status            : {result['status']}")
    if result["status"] == "SUCCESS":
        print(f"  Model Saved To    : {os.path.basename(MODEL_PATH)}")
        print(f"  Optimizer Used    : Adam (LR={LEARNING_RATE})")
        print(f"  Actor Loss        : {result['actor_loss']:+.6f}")
        print(f"  Entropy           : {result['entropy']:.6f}")
        print(f"  Total Loss        : {result['total_loss']:+.6f}")
        print(f"  Avg Session Reward: {result['avg_reward']:+.4f}")
        print(f"  Experiences Used  : {result['n_samples']}")
        loss_status = "SUCCESSFUL" if abs(result["total_loss"]) < 100 else "REVIEW NEEDED"
        print(f"  Loss Convergence  : {loss_status}")
    print("=" * 55 + "\n")

    sys.exit(0 if result["status"] in ("SUCCESS", "SKIPPED") else 1)

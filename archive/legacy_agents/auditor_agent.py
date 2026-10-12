# auditor_agent.py — MARL Experience Replay & Reward Auditor Agent

import json
import os
import numpy as np
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] AUDITOR_AGENT: %(message)s")
log = logging.getLogger("auditor_agent")

class AuditorAgent:
    """
    Agent 5: The Auditor (The Teacher).
    Calculates reinforcement learning reward feedback scores and persists experiences
    into a JSON Replay Buffer for model training and policy gradient updates.
    """
    def __init__(self, memory_file="experience_replay.json"):
        self.memory_file = memory_file
        self.memory = []
        self._load_memory()
        log.info(f"AuditorAgent initialized. Replay File: {self.memory_file} | Loaded Experiences: {len(self.memory)}")

    def _load_memory(self):
        """Attempts to load pre-existing experiences from disk to maintain state across cycles."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.memory = data
                    else:
                        log.warning("Memory file format is invalid. Starting with a fresh buffer.")
            except Exception as e:
                log.warning(f"Could not load memory file: {e}. Starting fresh.")

    def calculate_reward(self, pnl_percentage, max_drawdown):
        """
        Computes the RL reward metric:
        - Winning trades yield a base reward of +1.0.
        - Losing trades yield a base reward of -1.0.
        - Drawdown Penalty: Subtracts (max_drawdown * 0.5) to penalize excessive risk.
        
        Returns:
            float: Calculated reward value.
        """
        # Determine base reward
        if pnl_percentage > 0:
            base_reward = 1.0
        elif pnl_percentage < 0:
            base_reward = -1.0
        else:
            base_reward = 0.0
            
        # Drawdown penalty (penalize larger drawdowns)
        penalty = max_drawdown * 0.5
        reward = base_reward - penalty
        
        log.info(f"Reward calculated: {reward:.4f} (Base: {base_reward:+.1f} | Drawdown Penalty: -{penalty:.4f})")
        return float(reward)

    def record_experience(self, state_tensor, action_id, reward, next_state_tensor=None):
        """
        Packages state-action transitions into JSON-serializable experience dictionary
        and saves it to the local buffer list.
        """
        # Convert numpy array state_tensor to a list for JSON compatibility
        state_list = state_tensor.tolist() if isinstance(state_tensor, np.ndarray) else list(state_tensor)
        
        # Convert next_state_tensor if provided
        next_state_list = None
        if next_state_tensor is not None:
            next_state_list = next_state_tensor.tolist() if isinstance(next_state_tensor, np.ndarray) else list(next_state_tensor)
            
        experience = {
            "state": state_list,
            "action_id": int(action_id),
            "reward": float(reward),
            "next_state": next_state_list
        }
        
        self.memory.append(experience)
        log.info(f"Recorded experience. Action: {action_id} | Reward: {reward:.4f}")

    def save_memory(self):
        """Writes the accumulated experience replay memory to disk as a JSON file."""
        try:
            with open(self.memory_file, "w") as f:
                json.dump(self.memory, f, indent=2)
            log.info(f"Experience Replay buffer successfully persisted to disk: {self.memory_file}")
            return True
        except Exception as e:
            log.error(f"Failed to persist Experience Replay buffer: {e}")
            return False

if __name__ == "__main__":
    log.info("--- Starting Auditor Agent Test ---")
    
    # 1. Instantiate the AuditorAgent
    auditor = AuditorAgent(memory_file="experience_replay.json")
    
    # 2. Simulate a winning LONG trade: PnL = +4.0%, Max Drawdown = 1.0%
    pnl = 0.04
    drawdown = 0.01
    log.info(f"Simulating trade results: PnL = {pnl:+.1%} | Max Drawdown = {drawdown:.1%}")
    
    # 3. Calculate reward
    reward_score = auditor.calculate_reward(pnl, drawdown)
    
    # 4. Simulate a state tensor and action taken (1 for LONG)
    mock_state = np.array([0.5, 0.2, 0.5, 0.2, 0.5], dtype=np.float32)
    mock_next_state = np.array([0.6, 0.1, 0.4, 0.3, 0.6], dtype=np.float32)
    chosen_action = 1
    
    # 5. Record experience and save to JSON "brain"
    auditor.record_experience(mock_state, chosen_action, reward_score, next_state_tensor=mock_next_state)
    auditor.save_memory()
    
    # 6. Display formatted confirmation
    print("\n" + "=" * 50)
    print("         MARL EXPERIENCE AUDIT COMPLETED")
    print("=" * 50)
    print(f"  Calculated Reward : {reward_score:+.4f}")
    print(f"  Buffer Location   : {auditor.memory_file}")
    print(f"  Total Buffer Size : {len(auditor.memory)} experiences")
    print("=" * 50 + "\n")

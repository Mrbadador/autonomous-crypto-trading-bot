import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQNPolicyNetwork(nn.Module):
    def __init__(self, input_size=13, hidden_size=64, num_actions=3):
        super(DuelingDQNPolicyNetwork, self).__init__()
        
        # Shared linear representation
        self.shared_fc1 = nn.Linear(input_size, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Value Head V(s) - outputs a scalar representing the safety/quality of the market state
        self.value_fc = nn.Linear(hidden_size, hidden_size // 2)
        self.value_out = nn.Linear(hidden_size // 2, 1)
        
        # Advantage Head A(s, a) - outputs relative advantage of taking actions (HOLD, LONG, SHORT)
        self.advantage_fc = nn.Linear(hidden_size, hidden_size // 2)
        self.advantage_out = nn.Linear(hidden_size // 2, num_actions)
        
    def forward(self, x):
        # Shared representations
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))
        
        # Value stream
        val = F.relu(self.value_fc(x))
        V = self.value_out(val)
        
        # Advantage stream
        adv = F.relu(self.advantage_fc(x))
        A = self.advantage_out(adv)
        
        # Combine natives inside the forward pass using the standard stable equation:
        # Q(s, a) = V(s) + (A(s, a) - 1/|A| * sum(A(s, a')))
        Q = V + (A - A.mean(dim=-1, keepdim=True))
        
        return Q

if __name__ == "__main__":
    print("--- Dueling DQN Sandbox Initialization ---")
    model = DuelingDQNPolicyNetwork(input_size=13)
    
    # Mock simulation block passing a dummy (1, 13) tensor
    print("Generating mock state tensor with 13 dimensions...")
    mock_state = torch.randn(1, 13)
    print(f"Input Shape: {mock_state.shape}")
    
    print("Running forward pass...")
    try:
        q_values = model(mock_state)
        print(f"Output Q-Values Shape: {q_values.shape}")
        print(f"Output Q-Values: {q_values.detach().numpy()}")
        print("Success! Dimensional propagation verified.")
    except Exception as e:
        print(f"Forward pass failed: {e}")

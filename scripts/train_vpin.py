import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# =============================================================================
# 1. Neural Network Architecture (Fully Compatible with microstructure_guard.py)
# =============================================================================
class FlowClassifier(nn.Module):
    """Input(7) -> Hidden(32) -> Hidden(16) -> Output(2). [Noise, Informed]"""
    def __init__(self, input_dim: int = 7, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# =============================================================================
# 2. Synthetic Data Generation (10,000 Rows)
# =============================================================================
def generate_synthetic_order_flow(n_samples: int = 10000) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates high-fidelity synthetic order flow features modeling both Normal
    and Toxic flow regimes matching the expected 7 inputs.
    
    Inputs (7):
    - vpin_adj: Adjusted VPIN score [0.0 to 1.0]
    - stft_low: Low-frequency volume velocity [0.0 to 100.0]
    - stft_mid: Mid-frequency volume velocity [0.0 to 100.0]
    - stft_high: High-frequency volume velocity [0.0 to 100.0]
    - funding_rate: Funding rate [-0.01 to 0.01]
    - ofi: Order Flow Imbalance [-500.0 to 500.0]
    - spread_bps: Bid-Ask spread [1.0 to 50.0]
    """
    np.random.seed(42)
    half_samples = n_samples // 2
    
    # ------------------ Class 0: Normal / Noise Flow ------------------
    # Low VPIN, low volume features, stable funding, balanced OFI, tight spread
    vpin_0 = np.random.uniform(0.05, 0.40, half_samples)
    stft_low_0 = np.random.exponential(5.0, half_samples)
    stft_mid_0 = np.random.exponential(3.0, half_samples)
    stft_high_0 = np.random.exponential(1.5, half_samples)
    funding_0 = np.random.normal(0.0, 0.001, half_samples)
    ofi_0 = np.random.normal(0.0, 30.0, half_samples)  # Balanced OFI
    spread_0 = np.random.uniform(1.0, 8.0, half_samples)  # Tight spread
    
    X_0 = np.column_stack([vpin_0, stft_low_0, stft_mid_0, stft_high_0, funding_0, ofi_0, spread_0])
    y_0 = np.zeros(half_samples, dtype=np.int64)
    
    # ------------------ Class 1: Toxic / Informed Flow ------------------
    # High VPIN, huge volume spikes, absolute skewed funding, massive OFI skew, wide spread
    vpin_1 = np.random.uniform(0.65, 0.95, half_samples)
    stft_low_1 = np.random.exponential(30.0, half_samples) + 20.0
    stft_mid_1 = np.random.exponential(25.0, half_samples) + 15.0
    stft_high_1 = np.random.exponential(15.0, half_samples) + 10.0
    funding_1 = np.random.choice([-1.0, 1.0], half_samples) * np.random.uniform(0.003, 0.01, half_samples)
    
    # Highly skewed OFI (deep negative or highly positive)
    ofi_skew = np.random.choice([-1.0, 1.0], half_samples) * (np.random.exponential(150.0, half_samples) + 120.0)
    ofi_1 = np.clip(ofi_skew, -500.0, 500.0)
    spread_1 = np.random.uniform(12.0, 45.0, half_samples)  # Wide spread
    
    X_1 = np.column_stack([vpin_1, stft_low_1, stft_mid_1, stft_high_1, funding_1, ofi_1, spread_1])
    y_1 = np.ones(half_samples, dtype=np.int64)
    
    # Combine & Shuffle
    X = np.vstack([X_0, X_1])
    y = np.concatenate([y_0, y_1])
    
    shuffled_indices = np.random.permutation(n_samples)
    return X[shuffled_indices], y[shuffled_indices]

# =============================================================================
# 3. Training Script Execution
# =============================================================================
def train():
    print("=" * 60)
    print("      VPIN FLOW TOXICITY CLASSIFIER TRAINING SEQUENCE")
    print("=" * 60)
    
    # Generate data
    print("Generating 10,000 synthetic order flow samples...")
    X, y = generate_synthetic_order_flow(10000)
    
    # Convert to PyTorch Tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)  # Class indices
    
    # Train-test split (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X_tensor[:split_idx], X_tensor[split_idx:]
    y_train, y_test = y_tensor[:split_idx], y_tensor[split_idx:]
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Model instantiation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute target device: {device}")
    
    model = FlowClassifier(input_dim=7, hidden=32).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    
    # Training Loop
    epochs = 80
    print(f"Beginning training loop for {epochs} epochs...")
    model.train()
    
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            
        epoch_loss = running_loss / len(X_train)
        
        if epoch == 1 or epoch % 10 == 0:
            # Calculate validation accuracy
            model.eval()
            with torch.no_grad():
                test_x, test_y = X_test.to(device), y_test.to(device)
                test_outputs = model(test_x)
                _, preds = torch.max(test_outputs, dim=1)
                correct = (preds == test_y).sum().item()
                val_acc = correct / len(y_test)
            model.train()
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_loss:.5f} | Val Accuracy: {val_acc:.2%}")
            
    # Final Model Evaluation
    model.eval()
    with torch.no_grad():
        test_x, test_y = X_test.to(device), y_test.to(device)
        test_outputs = model(test_x)
        probs = torch.softmax(test_outputs, dim=1).cpu().numpy()
        _, preds = torch.max(test_outputs, dim=1)
        correct = (preds == test_y).sum().item()
        final_acc = correct / len(y_test)
        
    print("\n" + "-"*50)
    print(f"Final Model Validation Accuracy: {final_acc:.2%}")
    print("-"*50)
    
    # Export Weights (Safety: export to models/flow_classifier_synthetic.pth to never overwrite production weights)
    os.makedirs("models", exist_ok=True)
    export_path = os.path.join("models", "flow_classifier_synthetic.pth")
    print(f"Saving synthetic model state_dict to {os.path.abspath(export_path)}...")
    tmp_path = export_path + ".tmp"
    torch.save(model.state_dict(), tmp_path)
    os.replace(tmp_path, export_path)
    print(f"SUCCESS: Synthetic VPIN flow classifier weights exported to {export_path} (production weights protected).")
    print("=" * 60)

if __name__ == "__main__":
    train()

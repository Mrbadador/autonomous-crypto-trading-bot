import torch
import os
import sys

def perform_surgery(input_path, output_path, layer_name="lstm.weight_ih_l0"):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        sys.exit(1)
        
    print(f"Loading weights from {input_path}...")
    try:
        state_dict = torch.load(input_path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        sys.exit(1)
        
    if layer_name not in state_dict:
        print(f"Error: {layer_name} not found in state dict.")
        sys.exit(1)
        
    w = state_dict[layer_name]
    print(f"Original {layer_name} shape: {w.shape}")
    
    # Check if it's already 13
    if w.shape[1] == 13:
        print(f"Weights already have input dimension 13. Aborting surgery to prevent data corruption.")
        sys.exit(0)
        
    if w.shape[1] != 12:
        print(f"Expected input dimension 12, found {w.shape[1]}. Aborting surgery.")
        sys.exit(1)
        
    # Append a column of zeros
    new_col = torch.zeros(w.shape[0], 1, device=w.device, dtype=w.dtype)
    new_w = torch.cat([w, new_col], dim=1)
    
    state_dict[layer_name] = new_w
    print(f"New {layer_name} shape: {new_w.shape}")
    
    print(f"Saving modified weights to {output_path}...")
    torch.save(state_dict, output_path)
    print("Surgery complete. 13th dimension successfully initialized with zeros.")

if __name__ == "__main__":
    perform_surgery("strategist_weights_v26.pth", "strategist_weights_v27.pth")

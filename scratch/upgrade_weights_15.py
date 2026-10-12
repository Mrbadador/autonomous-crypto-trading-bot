# scratch/upgrade_weights_15.py
import torch
import os

def upgrade_weights(input_path):
    if not os.path.exists(input_path):
        print(f"Skipping {input_path} (not found).")
        return
    print(f"Loading {input_path}...")
    try:
        # Load weights
        state_dict = torch.load(input_path, map_location="cpu", weights_only=True)
        layer = "shared_fc1.weight"
        if layer not in state_dict:
            print(f"Layer '{layer}' not found in {input_path} state dict. Weight surgery skipped.")
            return
            
        w = state_dict[layer]
        print(f"Current weight shape for {layer}: {w.shape}")
        
        if w.shape[1] == 15:
            print(f"Weights in {input_path} already have input dimension 15. Skipping surgery.")
            return
        elif w.shape[1] == 13:
            # Append 2 columns of zeros
            new_cols = torch.zeros(w.shape[0], 2, device=w.device, dtype=w.dtype)
            new_w = torch.cat([w, new_cols], dim=1)
            state_dict[layer] = new_w
            
            # Save the modified file
            torch.save(state_dict, input_path)
            print(f"Successfully upgraded {input_path} to 15 features. New shape: {new_w.shape}")
        else:
            print(f"Unexpected weight dimension {w.shape[1]} in {input_path}. Expecting 13. Skipped.")
    except Exception as e:
        print(f"Error during surgery on {input_path}: {e}")

if __name__ == "__main__":
    upgrade_weights("strategist_weights_v27.pth")
    upgrade_weights("strategist_weights_v26.pth")
    upgrade_weights("policy_model.pth")

import os
import json
import time
import numpy as np

def run_audit(duration_seconds=3600):
    print(f"Starting L2 Telemetry Audit for {duration_seconds} seconds...")
    start_time = time.time()
    ofi_scores = []
    
    last_timestamp = 0
    state_file = "microstructure_state.json"
    
    while time.time() - start_time < duration_seconds:
        if os.path.exists(state_file):
            try:
                with open(state_file, "r") as f:
                    data = json.load(f)
                
                ts = data.get("timestamp", 0)
                if ts != last_timestamp:
                    ofi_scores.append(data.get("net_ofi_score", 0.0))
                    last_timestamp = ts
            except Exception:
                pass
                
        time.sleep(1)
        
    print("Audit collection complete. Calculating statistics...")
    if not ofi_scores:
        print("No OFI data collected.")
        return
        
    raw_array = np.array(ofi_scores)
    mean_ofi = np.mean(raw_array)
    std_ofi = np.std(raw_array)
    min_ofi = np.min(raw_array)
    max_ofi = np.max(raw_array)
    
    # Calculate normalization distribution
    norm_array = np.tanh(raw_array / 50.0)
    mean_norm = np.mean(norm_array)
    std_norm = np.std(norm_array)
    
    drops_below_veto = int(np.sum(raw_array < -5.0))
    
    results = {
        "samples_collected": len(raw_array),
        "raw_mean": float(mean_ofi),
        "raw_std": float(std_ofi),
        "raw_min": float(min_ofi),
        "raw_max": float(max_ofi),
        "norm_mean": float(mean_norm),
        "norm_std": float(std_norm),
        "norm_min": float(np.min(norm_array)),
        "norm_max": float(np.max(norm_array)),
        "drops_below_veto": drops_below_veto,
        "veto_trigger_rate_pct": float(drops_below_veto / len(raw_array) * 100) if len(raw_array) > 0 else 0.0
    }
    
    with open("ofi_audit_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to ofi_audit_results.json: {results}")

if __name__ == "__main__":
    run_audit(3600)

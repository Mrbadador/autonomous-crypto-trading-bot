import json
import os

DASHBOARD_FILE = "grafana_dashboard.json"

def update_dashboard():
    with open(DASHBOARD_FILE, "r", encoding="utf-8") as f:
        dashboard = json.load(f)

    # We want to add queries for 'mean_reversion' to panels that plot agent metrics.
    # We will look for panels that have a target with expr containing 'agent="sniper"' 
    # and duplicate that target for 'mean_reversion'
    
    modified = False
    
    # Check if mean_reversion is already in the panels
    # To avoid double adding
    already_added = False
    for panel in dashboard.get("panels", []):
        for target in panel.get("targets", []):
            if 'agent="mean_reversion"' in target.get("expr", ""):
                already_added = True
                break
                
    if already_added:
        print("Mean reversion already in dashboard.")
        return
        
    for panel in dashboard.get("panels", []):
        new_targets = []
        has_sniper = False
        
        for target in panel.get("targets", []):
            new_targets.append(target)
            if 'agent="sniper"' in target.get("expr", ""):
                has_sniper = True
                # Create a new target for mean_reversion based on the sniper one
                new_target = dict(target)
                new_target["expr"] = new_target["expr"].replace('agent="sniper"', 'agent="mean_reversion"')
                
                # Assign a unique refId (like 'D' if A,B,C are used)
                # We'll just generate one
                existing_refIds = [t.get("refId", "") for t in panel.get("targets", [])]
                new_refId = "MR"
                new_target["refId"] = new_refId
                
                # Assign a legendFormat
                if "legendFormat" in new_target:
                    new_target["legendFormat"] = new_target["legendFormat"].replace("Sniper", "Mean Reversion")
                    
                new_targets.append(new_target)
                modified = True
                
        if has_sniper:
            panel["targets"] = new_targets

    if modified:
        with open(DASHBOARD_FILE, "w", encoding="utf-8") as f:
            json.dump(dashboard, f, indent=2)
        print("Successfully updated Grafana dashboard with Mean Reversion metrics.")
    else:
        print("No changes made. Couldn't find sniper panels to copy.")

if __name__ == "__main__":
    update_dashboard()

import os
import json

directories = [
    r"C:\Users\user\Downloads",
    r"C:\Users\user\Desktop",
    r"C:\Users\user\Desktop\Workspaces"
]

targets = ["Almgren", "optimal_trajectory", "ExecutionPolicyNetwork"]

for d in directories:
    if not os.path.exists(d): continue
    print(f"Scanning {d}...")
    for root, dirs, files in os.walk(d):
        # skip clips and big game repo folders to be fast
        if "Clips" in root or "REPO" in root or "_CommonRedist" in root:
            continue
        for f in files:
            if not f.endswith('.json') and not f.endswith('.txt') and not f.endswith('.md'):
                continue
            path = os.path.join(root, f)
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                for t in targets:
                    if t in content:
                        print(f"MATCH FOUND: '{t}' in {path}!")
                        # print context around it
                        pos = content.find(t)
                        print(content[max(0, pos-200):min(len(content), pos+1000)])
                        print("-" * 60)
            except Exception as e:
                pass
print("Scan complete.")

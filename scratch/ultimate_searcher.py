import os

root_dir = r"C:\Users\user\.gemini\antigravity"
targets = [b"Almgren", b"optimal_trajectory", b"ExecutionCostCalculator", b"ExecutionPolicyNetwork"]

results = []

print(f"Scanning everything under {root_dir}...")
for root, dirs, files in os.walk(root_dir):
    if ".venv" in root or "__pycache__" in root or ".git" in root:
        continue
    for f in files:
        path = os.path.join(root, f)
        # Skip extremely large files to be safe (> 100MB)
        try:
            sz = os.path.getsize(path)
            if sz > 100 * 1024 * 1024:
                continue
            with open(path, 'rb') as file:
                data = file.read()
            for t in targets:
                if t in data:
                    print(f"!!! MATCH FOUND IN FILE !!! -> {path} (matched '{t.decode()}')")
                    results.append((path, t))
        except Exception:
            pass

print("=" * 60)
print(f"Search complete. Found {len(results)} total matches.")
for r in results:
    print(f"Matched {r[1].decode()} in {r[0]}")

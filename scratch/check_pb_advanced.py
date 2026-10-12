import os
import math

pb_path = r"C:\Users\user\.gemini\antigravity\conversations\2decef6b-c9e1-4fba-bb8a-2e336dd93461.pb"
print(f"Checking {pb_path}...")
stat = os.stat(pb_path)
import datetime
print("Modified time:", datetime.datetime.fromtimestamp(stat.st_mtime))

with open(pb_path, "rb") as f:
    data = f.read()

# Check entropy
freq = {}
for b in data:
    freq[b] = freq.get(b, 0) + 1
entropy = 0.0
for count in freq.values():
    p = count / len(data)
    entropy -= p * math.log2(p)
print(f"Shannon Entropy of file: {entropy:.4f} bits/byte (Compressed/encrypted is >7.5)")

# Searches
targets = ["Almgren", "optimal_trajectory", "ExecutionCostCalculator", "ExecutionPolicyNetwork"]

for t in targets:
    found = False
    for enc in ['utf-8', 'utf-16-le', 'utf-16-be', 'ascii']:
        b_t = t.encode(enc, errors='ignore')
        pos = data.find(b_t)
        if pos != -1:
            print(f"FOUND '{t}' via {enc} at {pos}!")
            found = True
            break
    if not found:
        print(f"'{t}' NOT found in any common encoding.")

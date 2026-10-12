import os

pb_path = r"C:\Users\user\.gemini\antigravity\conversations\2decef6b-c9e1-4fba-bb8a-2e336dd93461.pb"

print(f"Reading {pb_path}...")
with open(pb_path, "rb") as f:
    data = f.read()

print(f"Read {len(data)} bytes.")

# Search for common tokens
targets = [b"AlmgrenChrissParams", b"optimal_trajectory", b"ExecutionCostCalculator", b"ExecutionPolicyNetwork"]

for target in targets:
    pos = data.find(target)
    if pos != -1:
        print(f"Found '{target.decode()}' at position {pos}!")
        # Print 1000 bytes before and after
        start = max(0, pos - 50)
        end = min(len(data), pos + 10000) # we want the whole code block
        chunk = data[start:end]
        # Replace non-printable chars to safely print
        readable = "".join(chr(b) if 32 <= b <= 126 or b == 10 or b == 13 else "." for b in chunk)
        print("--- BEGIN EXTRACT ---")
        print(readable)
        print("--- END EXTRACT ---")
        
        # Write output to a text file so we can see it fully
        out_path = f"scratch/extracted_{target.decode()}.txt"
        with open(out_path, "wb") as out:
            out.write(chunk)
        print(f"Saved binary chunk to {out_path}")
    else:
        print(f"'{target.decode()}' not found in file.")

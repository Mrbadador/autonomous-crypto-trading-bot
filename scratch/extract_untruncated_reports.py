import os
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

parent_convo = "33384146-062f-45b6-a975-af41efadf64b"
msg_dir = f"C:/Users/user/.gemini/antigravity/brain/{parent_convo}/.system_generated/messages"

files = [
    ("3a6e830d-7f14-421f-acdd-2caa4ceff016.json", "domain_4_5_untruncated.txt"),
    ("c02acfc1-d4e7-4cee-a529-657b9807d877.json", "domain_2_3_untruncated.txt")
]

for filename, out_name in files:
    full_path = os.path.join(msg_dir, filename)
    out_path = f"c:/Users/user/.gemini/antigravity/scratch/{out_name}"
    
    if not os.path.exists(full_path):
        print(f"File {full_path} does not exist.")
        continue
        
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            content = data.get("content", "")
            
        with open(out_path, "w", encoding="utf-8") as out:
            out.write(content)
            
        print(f"Extracted {filename} -> {out_name} (length: {len(content)} characters)")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

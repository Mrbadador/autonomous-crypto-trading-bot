import os
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

parent_convo = "33384146-062f-45b6-a975-af41efadf64b"
path = f"C:/Users/user/.gemini/antigravity/brain/{parent_convo}/.system_generated/logs/transcript.jsonl"

subagents = [
    ("5a61b884-96ca-48d1-add8-879dcd2eb53a", "Domain 2+3"),
    ("2f8d2895-bdec-4885-aff2-77be30814bd8", "Domain 4+5")
]

if not os.path.exists(path):
    print(f"Parent transcript not found at {path}")
    sys.exit(1)

with open(path, "r", encoding="utf-8", errors="ignore") as f:
    for line_idx, line in enumerate(f):
        try:
            data = json.loads(line)
            # Look for messages containing subagent IDs or replies
            content = data.get("content", "")
            step_idx = data.get("step_index", line_idx)
            
            # Check if this step is a message from a subagent or references it
            for sa_id, sa_name in subagents:
                if sa_id in str(data):
                    # Print summary of the step where the ID appears
                    print(f"Step {step_idx}: Found reference to {sa_name} ({sa_id}). Type: {data.get('type')}, Source: {data.get('source')}, Content len: {len(content)}")
                    
                    # If this step contains a message from the subagent, let's print/save it
                    if data.get("type") == "SUBAGENT_MESSAGE" or "message" in str(data.get("type")).lower():
                        print(f"--> Found message content of len {len(content)}")
                        # Save it
                        out_file = f"c:/Users/user/.gemini/antigravity/scratch/{sa_name.replace(' ', '_').replace('+', '_')}_incoming.txt"
                        with open(out_file, "w", encoding="utf-8") as out:
                            out.write(content)
                        print(f"--> Saved to {out_file}")
                        
                    # Let's inspect tool_calls or tool_outputs
                    if "tool_calls" in data:
                        for tc in data["tool_calls"]:
                            if sa_id in str(tc):
                                print(f"--> Found in tool_call args/name: {tc.get('name') or tc.get('tool_name')}")
                    if "tool_output" in data or "output" in data:
                        out_val = data.get("tool_output") or data.get("output", "")
                        if sa_id in str(out_val):
                            print(f"--> Found in tool output. Length: {len(str(out_val))}")
        except Exception as e:
            pass

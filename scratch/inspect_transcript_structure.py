import os
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

parent_convo = "33384146-062f-45b6-a975-af41efadf64b"
path = f"C:/Users/user/.gemini/antigravity/brain/{parent_convo}/.system_generated/logs/transcript.jsonl"

subagents = ["5a61b884-96ca-48d1-add8-879dcd2eb53a", "2f8d2895-bdec-4885-aff2-77be30814bd8"]

with open(path, "r", encoding="utf-8", errors="ignore") as f:
    for line_idx, line in enumerate(f):
        try:
            data = json.loads(line)
            step_idx = data.get("step_index", line_idx)
            # Find steps that are messages or system notifications about subagent execution
            for sa_id in subagents:
                if sa_id in str(data):
                    # If the step type is related to subagent completing or message
                    tp = data.get("type", "")
                    if "subagent" in tp.lower() or "message" in tp.lower() or "notify" in tp.lower() or "planner" in tp.lower():
                        # Let's print the step structure (keys and snippet)
                        print(f"Step {step_idx} | Type: {tp} | Keys: {list(data.keys())}")
                        if "content" in data and data["content"]:
                            print(f"  Content length: {len(data['content'])}")
                            print(f"  Content snippet: {data['content'][:200]}...")
                        # Print any tool outputs
                        if "output" in data and data["output"]:
                            print(f"  Output length: {len(str(data['output']))}")
                            print(f"  Output snippet: {str(data['output'])[:200]}...")
                        if "tool_output" in data and data["tool_output"]:
                            print(f"  Tool Output length: {len(str(data['tool_output']))}")
                            print(f"  Tool Output snippet: {str(data['tool_output'])[:200]}...")
        except Exception as e:
            pass

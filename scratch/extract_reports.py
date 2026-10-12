import os
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def extract_from_transcript(convo_id, output_path):
    path = f"C:/Users/user/.gemini/antigravity/brain/{convo_id}/.system_generated/logs/transcript.jsonl"
    if not os.path.exists(path):
        print(f"Path not found: {path}")
        return
    
    print(f"Reading {path}...")
    messages = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                # Check for tool_calls or message content
                if data.get("type") in ["PLANNER_RESPONSE", "MODEL_RESPONSE"]:
                    content = data.get("content", "")
                    if content:
                        messages.append(content)
                # Also check tool outputs or messages
                if "tool_calls" in data:
                    for tc in data["tool_calls"]:
                        if tc.get("name") == "send_message" or tc.get("tool_name") == "send_message":
                            msg = tc.get("args", {}).get("Message") or tc.get("arguments", {}).get("Message")
                            if msg:
                                messages.append(msg)
            except Exception as e:
                pass
                
    # Find messages containing audit details
    audit_msgs = []
    for msg in messages:
        if "AUDIT" in msg or "P0" in msg or "P1" in msg or "P2" in msg:
            audit_msgs.append(msg)
            
    if audit_msgs:
        # Get the latest/longest one
        longest = max(audit_msgs, key=len)
        with open(output_path, "w", encoding="utf-8") as out:
            out.write(longest)
        print(f"Extracted report to {output_path} (length: {len(longest)} characters)")
    else:
        print("No audit messages found.")

print("Extracting Domain 2+3...")
extract_from_transcript("5a61b884-96ca-48d1-add8-879dcd2eb53a", "c:/Users/user/.gemini/antigravity/scratch/domain_2_3_report.txt")

print("Extracting Domain 4+5...")
extract_from_transcript("2f8d2895-bdec-4885-aff2-77be30814bd8", "c:/Users/user/.gemini/antigravity/scratch/domain_4_5_report.txt")

import os
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def dump_all_messages(convo_id, output_path):
    path = f"C:/Users/user/.gemini/antigravity/brain/{convo_id}/.system_generated/logs/transcript.jsonl"
    if not os.path.exists(path):
        print(f"Path not found: {path}")
        return
    
    print(f"\n==========================================")
    print(f"CONVO: {convo_id}")
    print(f"==========================================")
    
    messages = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            try:
                data = json.loads(line)
                # Look for tool calls or response content
                if data.get("source") == "MODEL" or data.get("type") in ["PLANNER_RESPONSE", "MODEL_RESPONSE"]:
                    content = data.get("content", "")
                    if content and len(content) > 500:
                        print(f"Step {data.get('step_index', line_idx)} Type {data.get('type')}: Length {len(content)}")
                        messages.append((data.get('step_index', line_idx), "MODEL_CONTENT", content))
                
                # Check tool calls for send_message
                tool_calls = data.get("tool_calls", [])
                for tc in tool_calls:
                    name = tc.get("name") or tc.get("tool_name")
                    if name == "send_message":
                        args = tc.get("args") or tc.get("arguments")
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except:
                                pass
                        msg = args.get("Message") if isinstance(args, dict) else str(args)
                        if msg:
                            print(f"Step {data.get('step_index', line_idx)} send_message: Length {len(msg)}")
                            messages.append((data.get('step_index', line_idx), "SEND_MESSAGE", msg))
            except Exception as e:
                pass

    if messages:
        # Sort messages by step index
        messages.sort(key=lambda x: x[0])
        with open(output_path, "w", encoding="utf-8") as out:
            for step, msg_type, msg in messages:
                out.write(f"\n\n==========================================\n")
                out.write(f"STEP {step} | TYPE: {msg_type}\n")
                out.write(f"==========================================\n")
                out.write(msg)
        print(f"Dumped {len(messages)} messages to {output_path}")
    else:
        print("No messages found.")

dump_all_messages("5a61b884-96ca-48d1-add8-879dcd2eb53a", "c:/Users/user/.gemini/antigravity/scratch/domain_2_3_full.txt")
dump_all_messages("2f8d2895-bdec-4885-aff2-77be30814bd8", "c:/Users/user/.gemini/antigravity/scratch/domain_4_5_full.txt")

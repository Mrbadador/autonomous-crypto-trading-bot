import time
import os
import sys

LOG_FILE = "logs/testnet_session.log"

def print_highlighted(line):
    line = (line.strip() if line is not None else "")
    if "HEARTBEAT" in line:
        print(f"\033[92m{line}\033[0m") # Green
    elif "AI Conviction" in line:
        print(f"\033[96m{line}\033[0m") # Cyan
    else:
        print(line)

def tail_f(filepath):
    print(f"Tailing {filepath}...")
    if not os.path.exists(filepath):
        print(f"File {filepath} not found yet. Waiting...")
        while not os.path.exists(filepath):
            time.sleep(1)
            
    with open(filepath, 'r', encoding='utf-8') as f:
        # Get last 20 lines
        f.seek(0, 2)
        size = f.tell()
        lines = []
        block = -1024
        while size > 0 and len(lines) <= 20:
            if (size + block) > 0:
                f.seek(size + block)
                lines = f.readlines()
            else:
                f.seek(0)
                lines = f.readlines()
                break
            block *= 2
            
        for line in lines[-20:]:
            print_highlighted(line)
            
        # Tail -f
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            print_highlighted(line)

if __name__ == "__main__":
    try:
        tail_f(LOG_FILE)
    except KeyboardInterrupt:
        sys.exit(0)

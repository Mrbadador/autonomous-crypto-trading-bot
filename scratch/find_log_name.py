with open("berserker_trader.py", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if "self.log_name" in line:
            print(f"L{i+1}: {line.strip()}")

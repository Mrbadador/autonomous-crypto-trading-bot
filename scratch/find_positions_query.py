with open("dashboard.py", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if "SELECT" in line.upper() and "FROM" in line.upper() and "POSITION" in line.upper():
            print(f"L{i+1}: {line.strip()}")

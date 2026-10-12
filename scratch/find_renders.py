with open("dashboard.py", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if "render_" in line and "(" in line and "def " not in line:
            print(f"L{i+1}: {line.strip()}")

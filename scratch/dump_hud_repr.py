with open("dashboard.py", "r", encoding="utf-8") as f:
    lines = f.readlines()
    print(f"Line 322: {repr(lines[321])}")
    print(f"Line 323: {repr(lines[322])}")

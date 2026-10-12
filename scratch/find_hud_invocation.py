with open("dashboard.py", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if "render_macro_sentinel_hud()" in line:
            print(f"L{i+1}: {line}")

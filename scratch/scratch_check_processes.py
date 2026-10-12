import psutil

print("=== RUNNING PYTHON PROCESSES ===")
for p in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        cmd = " ".join(p.info['cmdline'] or [])
        if "python" in p.info['name'].lower() and any(x in cmd for x in ['super_bot', 'start_system', 'swing_trader', 'dashboard', 'reconciliation']):
            print(f"PID {p.info['pid']}: {cmd}")
    except Exception:
        pass

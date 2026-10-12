import sqlite3
import os
import subprocess
from datetime import datetime, timezone

def run_checks():
    print("=== POST-DEPLOYMENT VERIFICATION & SYSTEM HEALTH CHECK ===")
    
    # 1. Schema Check
    db_path = "trading_data.db"
    if not os.path.exists(db_path):
        print(f"[SCHEMA] FAIL: {db_path} not found.")
        return
        
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("PRAGMA table_info(ghost_trades)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}
    has_tp = "paper_tp" in columns
    has_sl = "paper_sl" in columns
    print(f"[SCHEMA] Columns in ghost_trades: {list(columns.keys())}")
    if has_tp and has_sl:
        print("[SCHEMA] status: GREEN (paper_tp and paper_sl columns successfully added)")
    else:
        print("[SCHEMA] status: RED (missing paper_tp/paper_sl)")

    # 2. Ghost Activity
    cursor = conn.execute(
        "SELECT id, timestamp, symbol, signal, entry_price, paper_sl, paper_tp, exit_price, pnl, outcome "
        "FROM ghost_trades ORDER BY id DESC LIMIT 5"
    )
    rows = cursor.fetchall()
    total_ghosts = conn.execute("SELECT COUNT(*) FROM ghost_trades").fetchone()[0]
    print(f"\n[GHOSTS] Total Ghost Trades in DB: {total_ghosts}")
    if rows:
        print("[GHOSTS] Recent Ghost Trades:")
        for r in rows:
            print(f"  ID={r[0]} | TS={r[1]} | Sym={r[2]} | Sig={r[3]} | Entry={r[4]} | SL={r[5]} | TP={r[6]} | Exit={r[7]} | PnL={r[8]} | Outcome={r[9]}")
    else:
        print("[GHOSTS] No ghost trades logged in the database yet.")

    # Check for NULLs in today's runs
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    null_count = conn.execute(
        "SELECT COUNT(*) FROM ghost_trades WHERE (paper_sl IS NULL OR paper_tp IS NULL) AND timestamp LIKE ?",
        (f"{today_str}%",)
    ).fetchone()[0]
    print(f"[GHOSTS] Today's ghost entries with NULL paper_sl/paper_tp: {null_count}")

    # Check system state sniper kill-switch flag
    switch_val = conn.execute(
        "SELECT value FROM system_state WHERE key = 'DISABLED_SIGNAL_Sniper-Strategy'"
    ).fetchone()
    switch_status = switch_val[0] if switch_val else "UNKNOWN"
    print(f"[SYSTEM_STATE] Sniper Strategy Kill-switch status: {switch_status} (1 = Disabled, 0 = Enabled)")
    conn.close()

    # 3. Process Status (Daemon Status)
    print("\n[DAEMONS] Checking running Python processes...")
    try:
        # On Windows, we use tasklist
        proc = subprocess.run("tasklist /FI \"IMAGENAME eq python.exe\"", shell=True, capture_output=True, text=True)
        print(proc.stdout)
    except Exception as e:
        print(f"[DAEMONS] Process check failed: {e}")

    # Check manager log and sniper log for exceptions in the last 15 minutes
    print("\n[LOGS] Scanning logs for recent exceptions...")
    log_files = ["logs/sniper.log", "logs/main.log", "logs/alpha_auditor.log"]
    found_errors = False
    for log_file in log_files:
        if os.path.exists(log_file):
            print(f"  Scanning {log_file} (last 50 lines):")
            with open(log_file, "r", errors="ignore") as f:
                lines = f.readlines()[-50:]
            for idx, line in enumerate(lines):
                if any(x in line for x in ["Exception", "Traceback", "Error", "NameError", "AttributeError"]):
                    # Suppress minor known warnings/errors if any, but print exceptions
                    print(f"    Line {idx+1}: {line.strip()}")
                    found_errors = True
        else:
            print(f"  {log_file} does not exist.")
            
    if not found_errors:
        print("  [LOGS] No errors or exceptions found in recent logs. Status: GREEN")
    else:
        print("  [LOGS] Some warnings/errors detected in logs.")

if __name__ == "__main__":
    run_checks()

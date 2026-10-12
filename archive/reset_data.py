import os
import sqlite3
import shutil

FILES_TO_DELETE = [
    "trading_data.db",
    "trades.csv",
    "state.json",
    "bot.log",
    "backtest.log",
    "startup.log",
    "trade_history.csv"
]

DIRS_TO_CLEAN = [
    "logs"
]

def reset():
    print("============================================================")
    print("   RESETTING LLAMA TRADING ENGINE DATA")
    print("============================================================")
    
    # Delete Files
    for f in FILES_TO_DELETE:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"[✓] Deleted {f}")
            except Exception as e:
                print(f"[!] Failed to delete {f}: {e}")
    
    # Clean Log Directories
    for d in DIRS_TO_CLEAN:
        if os.path.exists(d):
            try:
                shutil.rmtree(d)
                os.makedirs(d)
                print(f"[✓] Cleaned {d}/ directory")
            except Exception as e:
                print(f"[!] Failed to clean {d}: {e}")

    print("============================================================")
    print("   CLEAN SLATE ACHIEVED. YOU ARE READY TO RUN.")
    print("============================================================")

if __name__ == "__main__":
    reset()

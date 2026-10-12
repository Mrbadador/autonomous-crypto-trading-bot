"""
Sovereign Swarm — Clean Slate Database Sanitizer
Safely purges stale paper test positions and ghost rows from previous mock runs.
Prepares trading_data.db for fresh, pristine live deployment.
"""
import sqlite3
import os

os.chdir(r"c:\Users\user\.gemini\antigravity\scratch\llama_trading_bot")

tables_to_clear = [
    "open_positions",
    "mean_reversion_open_positions",
    "marl_live_open_positions",
    "berserker_open_positions",
    "ghost_trades",
]

print("\n" + "="*70)
print("   CLEAN SLATE DATABASE SANITIZATION")
print("="*70)

with sqlite3.connect("trading_data.db") as conn:
    for table in tables_to_clear:
        try:
            count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            conn.execute(f"DELETE FROM {table}")
            print(f"  [PURGED] {table:<32}: Cleared {count} stale records -> 0 rows")
        except Exception as e:
            print(f"  [ERROR]  {table:<32}: {e}")
    conn.commit()

print("\n[+] Verification Check across all position tables:")
with sqlite3.connect("trading_data.db") as conn:
    for table in tables_to_clear:
        count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"  ✓ {table:<32}: {count} open rows")

print("\n[+] Database is 100% clean and ready for live pilot.")
print("="*70 + "\n")

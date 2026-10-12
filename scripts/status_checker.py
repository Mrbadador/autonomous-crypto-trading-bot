# status_checker.py
import os
import sys
import sqlite3
import logging
from datetime import datetime, UTC

# Suppress debug logs
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DATA = os.path.join(BASE_DIR, "trading_data.db")
DB_LEDGER = os.path.join(BASE_DIR, "trading_ledger.db")
LOG_FILE = os.path.join(BASE_DIR, "launcher.log") # Launcher log holds current pulse details

def get_read_only_conn(db_path):
    """Establishes a URI-based read-only connection to SQLite to prevent locking active transactions."""
    if not os.path.exists(db_path):
        return None
    db_uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(db_uri, uri=True, timeout=5.0)

def main():
    print("=" * 60)
    print("  ANTI-GRAVITY SWARM STATUS CHECKER")
    print(f"  Current Time (UTC): {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. Active Positions
    print("\n[ACTIVE MARL POSITIONS]")
    conn_data = get_read_only_conn(DB_DATA)
    if conn_data:
        try:
            cursor = conn_data.execute("SELECT symbol, side, entry_price, qty FROM marl_live_open_positions")
            rows = cursor.fetchall()
            if not rows:
                print("  No active positions found.")
            else:
                for row in rows:
                    print(f"  • {row[0]}: {row[1]} | Entry: ${row[2]:.4f} | Qty: {row[3]:.4f}")
        except Exception as e:
            print(f"  Error reading active positions: {e}")
        finally:
            conn_data.close()
    else:
        print("  trading_data.db not found.")

    # 2. Fleet PnL from trade_lessons
    print("\n[FLEET OVERVIEW]")
    conn_ledger = get_read_only_conn(DB_LEDGER)
    if conn_ledger:
        try:
            cursor = conn_ledger.execute("SELECT SUM(pnl) FROM trade_lessons")
            res = cursor.fetchone()
            pnl_val = float(res[0]) if res and res[0] is not None else 0.0
            
            cursor_cnt = conn_ledger.execute("SELECT COUNT(*) FROM trade_lessons")
            total_trades = cursor_cnt.fetchone()[0]
            
            print(f"  Total Trades Registered: {total_trades}")
            print(f"  Cumulative Fleet PnL   : ${pnl_val:+.2f}")
        except Exception as e:
            print(f"  Error reading fleet PnL: {e}")
        finally:
            conn_ledger.close()
    else:
        print("  trading_ledger.db not found.")

    # 3. System Pulse Check
    print("\n[DAEMON SYSTEM PULSE]")
    if os.path.exists(LOG_FILE):
        try:
            # Read last 3 lines
            with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                last_lines = [line.strip() for line in lines[-3:] if line.strip()]
            for line in last_lines:
                print(f"  {line}")
        except Exception as e:
            print(f"  Error checking system pulse: {e}")
    else:
        print("  System log file not found.")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()

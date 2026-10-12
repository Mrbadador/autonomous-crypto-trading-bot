import sqlite3
import os

def check_db():
    db_path = "trading_data.db"
    if not os.path.exists(db_path):
        print(f"Database {db_path} not found.")
        return
        
    conn = sqlite3.connect(db_path)
    try:
        # Check balance
        bal_row = conn.execute("SELECT key, value FROM system_state WHERE key LIKE 'balance%'").fetchall()
        print("Balances in system_state:")
        for r in bal_row:
            print(f"  {r[0]}: {r[1]}")
            
        # Check active positions
        open_pos = conn.execute("SELECT * FROM open_positions").fetchall()
        print(f"\nOpen positions (Total: {len(open_pos)}):")
        for p in open_pos:
            print(f"  {p}")
            
        # Check ghost trades pending
        pending_ghosts = conn.execute("SELECT COUNT(*) FROM ghost_trades WHERE outcome IS NULL").fetchone()[0]
        print(f"\nPending/Open ghost trades: {pending_ghosts}")
        
        # Check Sniper kill-switch
        disabled_sniper = conn.execute("SELECT value FROM system_state WHERE key = 'DISABLED_SIGNAL_Sniper-Strategy'").fetchone()
        print(f"DISABLED_SIGNAL_Sniper-Strategy: {disabled_sniper[0] if disabled_sniper else 'NOT SET'}")
        
    except Exception as e:
        print(f"Error querying database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_db()

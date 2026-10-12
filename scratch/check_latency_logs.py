import sqlite3
import os
import json

def get_latency_logs():
    db_path = "trading_ledger.db"
    if not os.path.exists(db_path):
        print(f"Database {db_path} not found.")
        return
        
    conn = sqlite3.connect(db_path)
    try:
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='latency_logs'").fetchone()
        if not tables:
            print("Table 'latency_logs' does not exist in the database.")
            return
            
        count = conn.execute("SELECT COUNT(*) FROM latency_logs").fetchone()[0]
        print(f"Total latency logs in database: {count}")
        
        rows = conn.execute(
            "SELECT timestamp, symbol, total_ms, segments FROM latency_logs "
            "ORDER BY timestamp DESC LIMIT 5"
        ).fetchall()
        
        for r in rows:
            print(f"\nTimestamp: {r[0]} | Symbol: {r[1]} | Total: {r[2]}ms")
            try:
                segs = json.loads(r[3])
                for k, v in segs.items():
                    print(f"  - {k}: {v}ms")
            except Exception:
                print(f"  - Raw segments: {r[3]}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    get_latency_logs()

# scratch/verify_oracle.py
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sqlite3
from live_oracle import LiveOracle

def verify():
    print("=== STARTING LIVE ORACLE VERIFICATION SHAKEDOWN ===")
    db_path = "scratch/test_verification.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    oracle = LiveOracle(db_path=db_path)
    print("[1] DB successfully initialized.")
    
    print("[2] Performing multi-asset correlation scan...")
    oracle._fetch_and_process_data()
    print("Correlation scan completed.")
    
    print("[3] Running engineer_features() with Entropy Engine...")
    fng, vol, trend, vwap, rsi, readiness = oracle.engineer_features()
    print(f"Features Extracted:")
    print(f"  F&G: {fng}")
    print(f"  Vol State: {vol}")
    print(f"  Trend Str: {trend}")
    print(f"  VWAP: {vwap}")
    print(f"  RSI: {rsi}")
    print(f"  Readiness Score: {readiness}")
    
    print("[4] Persisting to DB...")
    success = oracle.commit_to_database(fng, vol, trend, vwap, rsi, readiness)
    print(f"Persist successful? {success}")
    
    print("[5] Reading database contents back...")
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT * FROM market_context LIMIT 1")
    row = cursor.fetchone()
    columns = [col[0] for col in cursor.description]
    conn.close()
    
    print("Database Row Snapshot:")
    for col, val in zip(columns, row):
        print(f"  {col}: {val}")
        
    assert 'signal_readiness' in columns, "MISSING COLUMN IN DB SCHEMA"
    assert row is not None, "NO DATA STORED IN DB"
    print("=== VERIFICATION SUCCESSFUL ===")

if __name__ == "__main__":
    verify()

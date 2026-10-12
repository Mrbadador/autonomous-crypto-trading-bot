import sqlite3
import os

db_files = ["trading_data.db", "trading_ledger.db"]
suspicious_keywords = ["key", "secret", "pass", "token", "api", "pwd", "credential"]

def scan_db(db_path):
    if not os.path.exists(db_path):
        print(f"Database {db_path} does not exist.")
        return
    
    print(f"\n--- Scanning Database: {db_path} ---")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in c.fetchall()]
    
    for table in tables:
        # Check table columns
        c.execute(f"PRAGMA table_info({table})")
        columns = c.fetchall()
        col_names = [col[1] for col in columns]
        
        # Check if any column name is suspicious
        susp_cols = [col for col in col_names if any(k in col.lower() for k in suspicious_keywords)]
        if susp_cols:
            print(f"Table '{table}' has suspicious columns: {susp_cols}")
            
        # Sample rows to inspect content
        try:
            c.execute(f"SELECT * FROM {table} LIMIT 100")
            rows = c.fetchall()
            for r_idx, row in enumerate(rows):
                for c_idx, val in enumerate(row):
                    if isinstance(val, str):
                        # check if string value looks like a raw secret (e.g. Binance API keys are 64 characters, secrets are 64, etc.)
                        val_lower = val.lower()
                        # Avoid matching standard hashes or IDs that aren't secrets
                        # Check for API key prefix or long random-looking strings
                        if len(val) >= 32 and any(char.isdigit() for char in val) and any(char.isalpha() for char in val):
                            # Check if the string actually contains BINANCE, GROQ, API, etc. or looks like a secret
                            if any(k in val_lower for k in suspicious_keywords) or ("ai_trader" in val_lower) or (len(val) == 64 and not val.isalnum()):
                                print(f"ALERT: Suspicious long string in Table '{table}', Col '{col_names[c_idx]}', Row {r_idx}: {val[:10]}...")
        except Exception as e:
            print(f"Error scanning table '{table}': {e}")
            
    conn.close()

if __name__ == "__main__":
    for db in db_files:
        scan_db(db)

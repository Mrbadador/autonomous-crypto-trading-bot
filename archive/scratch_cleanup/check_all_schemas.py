import sqlite3

def dump_schema(db_file):
    print(f"=== Schema for {db_file} ===")
    try:
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cur.fetchall()
        for t in tables:
            table_name = t[0]
            print(f"\n--- Table: {table_name} ---")
            cur.execute(f"PRAGMA table_info({table_name});")
            columns = cur.fetchall()
            for col in columns:
                print(f"  {col[1]} ({col[2]})")
        conn.close()
    except Exception as e:
        print(f"Error reading {db_file}: {e}")

dump_schema("trading_data.db")
dump_schema("trading_ledger.db")

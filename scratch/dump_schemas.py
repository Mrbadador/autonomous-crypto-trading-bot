import sqlite3

def dump_schema(db_name):
    print(f"\n--- SCHEMA FOR {db_name} ---")
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table in tables:
        t_name = table[0]
        print(f"\nTable: {t_name}")
        cursor.execute(f"PRAGMA table_info({t_name})")
        columns = cursor.fetchall()
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
    conn.close()

dump_schema("trading_data.db")
dump_schema("trading_ledger.db")

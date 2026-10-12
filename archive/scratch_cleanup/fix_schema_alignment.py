import sqlite3

db = "trading_data.db"
conn = sqlite3.connect(db)
cursor = conn.cursor()

tables = ["berserker_trades", "trades"] # trades already has them, but berserker_trades might not

for table in tables:
    try:
        cursor.execute(f"PRAGMA table_info({table})")
        cols = [c[1] for c in cursor.fetchall()]
        
        if "action" in cols and "type" not in cols:
            print(f"Renaming action to type in {table}...")
            cursor.execute(f"ALTER TABLE {table} RENAME COLUMN action TO type")
            
        if "balance_after" in cols and "balance" not in cols:
            print(f"Renaming balance_after to balance in {table}...")
            cursor.execute(f"ALTER TABLE {table} RENAME COLUMN balance_after TO balance")
            
        if "status" not in cols:
            print(f"Adding status to {table}...")
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN status TEXT DEFAULT 'closed'")
            
    except Exception as e:
        print(f"Error updating {table}: {e}")

conn.commit()
conn.close()
print("Schema alignment complete.")

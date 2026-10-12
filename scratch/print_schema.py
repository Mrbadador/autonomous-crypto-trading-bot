import sqlite3

def print_schema(db_name):
    print(f"\nSchema for {db_name}:")
    with sqlite3.connect(db_name) as conn:
        c = conn.cursor()
        c.execute("SELECT sql FROM sqlite_master WHERE type='table'")
        for row in c.fetchall():
            if row[0]: print(row[0])

if __name__ == "__main__":
    print_schema("trading_data.db")

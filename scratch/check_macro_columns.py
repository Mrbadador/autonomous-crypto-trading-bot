import sqlite3
try:
    conn = sqlite3.connect("trading_ledger.db")
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(macro_market_context);")
    cols = cur.fetchall()
    for col in cols:
        print(f"Col: {col[1]}")
    conn.close()
except Exception as e:
    print(f"Error: {e}")

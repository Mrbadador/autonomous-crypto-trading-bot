import sqlite3
for db in ["live_paper_trades.db", "trading_data.db"]:
    conn = sqlite3.connect(db)
    cursor = conn.execute("PRAGMA table_info(trades)")
    cols = [c[1] for c in cursor.fetchall()]
    print(f"Columns in {db} (trades): {cols}")
    conn.close()

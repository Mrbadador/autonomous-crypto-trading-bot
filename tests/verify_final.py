import os
import sqlite3
import tempfile
import tests.conftest  # noqa: F401 — enforce D-11 quarantine

tmp_db = os.path.join(tempfile.mkdtemp(prefix="verify_final_"), "trading_data.db")
conn = sqlite3.connect(tmp_db)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS berserker_open_positions (symbol TEXT PRIMARY KEY)")
cursor.execute("SELECT * FROM berserker_open_positions")
print(f"Open Berserker Positions: {cursor.fetchall()}")
conn.close()

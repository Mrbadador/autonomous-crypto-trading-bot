import urllib.request
import time
import json
import sqlite3

try:
    resp = urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5)
    st_status = f"HTTP {resp.status}"
except Exception as e:
    st_status = f"Error: {e}"
print(f"Streamlit Health: {st_status}")

conn = sqlite3.connect('trading_data.db')
conn.row_factory = sqlite3.Row
bal = conn.execute("SELECT value FROM system_state WHERE key='balance'").fetchone()
open_pos = conn.execute("SELECT * FROM swing_open_positions").fetchall()
print(f"Database Balance: ${float(bal[0]) if bal else 0:.2f}")
print(f"Open Positions: {len(open_pos)}")
conn.close()

try:
    with open('microstructure_state.json') as f:
        mstate = json.load(f)
        age = time.time() - mstate.get('timestamp', 0)
        sym_count = len(mstate.get('symbols', {}))
        print(f"Microstructure JSON: {sym_count} symbols, age={age:.1f}s")
except Exception as e:
    print(f"Microstructure JSON Error: {e}")

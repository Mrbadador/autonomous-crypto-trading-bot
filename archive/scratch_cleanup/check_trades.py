import sqlite3
import pandas as pd

conn = sqlite3.connect('trading_data.db')
try:
    df = pd.read_sql_query("SELECT * FROM trades LIMIT 20", conn)
    print(df.to_string())
except Exception as e:
    print(f"Error: {e}")
finally:
    conn.close()

import sqlite3
import os
import re

db_path = r"c:\Users\user\.gemini\antigravity\scratch\llama_trading_bot\trading_data.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get schemas
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='trades'")
trades_schema = cursor.fetchone()[0]

cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='berserker_trades'")
berserker_schema = cursor.fetchone()[0]

# Create legacy schemas
legacy_trades_schema = re.sub(r'CREATE TABLE\s+(["\']?)trades\1', r'CREATE TABLE IF NOT EXISTS legacy_trades', trades_schema, flags=re.IGNORECASE)
legacy_berserker_schema = re.sub(r'CREATE TABLE\s+(["\']?)berserker_trades\1', r'CREATE TABLE IF NOT EXISTS legacy_berserker_trades', berserker_schema, flags=re.IGNORECASE)

cursor.execute(legacy_trades_schema)
cursor.execute(legacy_berserker_schema)

# Count before
cursor.execute("SELECT COUNT(*) FROM trades")
t_count = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM berserker_trades")
b_count = cursor.fetchone()[0]
total_moved = t_count + b_count

# Move data
cursor.execute("INSERT INTO legacy_trades SELECT * FROM trades")
cursor.execute("INSERT INTO legacy_berserker_trades SELECT * FROM berserker_trades")

# Clear tables
cursor.execute("DELETE FROM trades")
cursor.execute("DELETE FROM berserker_trades")

conn.commit()
conn.close()

print(f"[ARCHIVE-PIVOT] {total_moved:,} trades moved to Legacy Vault. Active tables are now empty.")

import sqlite3

db_path = "trading_data.db"
print(f"Connecting to {db_path}...")
conn = sqlite3.connect(db_path)
c = conn.cursor()

# 1. Add ai_rationale columns
try:
    print("Adding ai_rationale to trades...")
    c.execute("ALTER TABLE trades ADD COLUMN ai_rationale TEXT")
    print("Added successfully.")
except sqlite3.OperationalError as e:
    print(f"Skipped (maybe exists): {e}")

try:
    print("Adding ai_rationale to berserker_trades...")
    c.execute("ALTER TABLE berserker_trades ADD COLUMN ai_rationale TEXT")
    print("Added successfully.")
except sqlite3.OperationalError as e:
    print(f"Skipped (maybe exists): {e}")

# 2. System State Migration for DEFAULT CURRENT_TIMESTAMP
print("Migrating system_state table...")
try:
    c.execute("BEGIN IMMEDIATE")
    c.execute("DROP TABLE IF EXISTS system_state_new")
    c.execute("""
        CREATE TABLE system_state_new (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("INSERT INTO system_state_new (key, value, updated_at) SELECT key, value, updated_at FROM system_state")
    c.execute("DROP TABLE system_state")
    c.execute("ALTER TABLE system_state_new RENAME TO system_state")
    conn.commit()
    print("System state migrated successfully.")
except Exception as e:
    conn.rollback()
    print(f"System state migration failed: {e}")

conn.close()
print("Migration phase complete.")

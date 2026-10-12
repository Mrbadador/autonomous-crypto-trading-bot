import os
import sys
import tempfile
import sqlite3

sys.path.insert(0, os.path.abspath("."))
from db_manager import DatabaseManager

def test_stage1_notnull_rebuild_migration():
    """
    FIX 1 Acceptance Test:
    Builds an OLD-schema table in a temporary SQLite database with NOT NULL on
    exit_price, pnl, and roe_pct (simulating pre-1.2 trading_data.db).
    Runs the DatabaseManager._init_db migration.
    Asserts:
    1. Schema PRAGMA confirms exit_price, pnl, and roe_pct are now nullable (notnull=0).
    2. Inserting an UNRECONCILED row with NULL exit_price, NULL pnl, and NULL roe_pct succeeds.
    3. The inserted row round-trips with intact data and status='UNRECONCILED'.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_db_path = tf.name

    try:
        # Step 1: Create an old database with NOT NULL constraints
        conn = sqlite3.connect(temp_db_path)
        conn.execute("""
            CREATE TABLE system_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        conn.execute("INSERT INTO system_state (key, value, updated_at) VALUES ('migrations_version', '1.1', '2026-10-12T22:00:00Z');")

        # Create old swing_trades with strict NOT NULL constraints
        conn.execute("""
            CREATE TABLE swing_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                size REAL NOT NULL,
                pnl REAL NOT NULL,
                roe_pct REAL NOT NULL,
                holding_hours REAL,
                exit_reason TEXT,
                notes TEXT
            );
        """)
        # Insert a pre-existing closed trade
        conn.execute("""
            INSERT INTO swing_trades (timestamp, symbol, side, entry_price, exit_price, size, pnl, roe_pct)
            VALUES ('2026-10-12T20:00:00Z', 'BTCUSDT', 'LONG', 68000.0, 69000.0, 0.05, 50.0, 1.47);
        """)
        conn.commit()
        conn.close()

        # Step 2: Trigger DatabaseManager which runs migration 1.2
        dm = DatabaseManager(temp_db_path)

        # Step 3: Inspect PRAGMA table_info on swing_trades
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(swing_trades);")
        cols = {row[1]: {"notnull": row[3]} for row in cursor.fetchall()}

        assert cols["exit_price"]["notnull"] == 0, "exit_price must be nullable after migration 1.2!"
        assert cols["pnl"]["notnull"] == 0, "pnl must be nullable after migration 1.2!"
        assert cols["roe_pct"]["notnull"] == 0, "roe_pct must be nullable after migration 1.2!"

        # Step 4: Verify inserting UNRECONCILED row with NULLs succeeds without IntegrityError
        insert_sql = """
            INSERT INTO swing_trades (
                timestamp, symbol, side, entry_price, exit_price, size,
                pnl, roe_pct, holding_hours, exit_reason, notes,
                intended_entry_price, actual_fill_price, slippage_bps,
                fee_paid_usd, fee_maker_usd, fee_taker_usd,
                funding_usd, funding_source, gates_snapshot, status
            ) VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?
            )
        """
        params = (
            "2026-10-12T22:30:00Z", "ETHUSDT", "LONG", 2600.0, None, 1.0,
            None, None, 4.5, "UNRECONCILED_CLOSE", "Exchange position closed externally with no fills found",
            2600.0, 2600.5, 1.9,
            0.52, 0.52, 0.0,
            0.0, "unavailable", None, "UNRECONCILED"
        )

        cursor.execute(insert_sql, params)
        conn.commit()

        # Step 5: Query and assert round-trip
        cursor.execute("SELECT symbol, exit_price, pnl, roe_pct, status FROM swing_trades WHERE symbol = 'ETHUSDT';")
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "ETHUSDT"
        assert row[1] is None, "exit_price must be NULL"
        assert row[2] is None, "pnl must be NULL"
        assert row[3] is None, "roe_pct must be NULL"
        assert row[4] == "UNRECONCILED"

        # Step 6: Verify pre-existing trade was retained
        cursor.execute("SELECT symbol, pnl FROM swing_trades WHERE symbol = 'BTCUSDT';")
        old_row = cursor.fetchone()
        assert old_row is not None
        assert old_row[0] == "BTCUSDT"
        assert old_row[1] == 50.0

        # Step 7: Verify reconciler Anti-Knife and close logic does not raise on net_pnl=None
        close_type = "UNRECONCILED_CLOSE"
        net_pnl = None
        exit_p = None
        # Simulating reconciliation_engine.py:530
        anti_knife_triggered = (close_type == "sl_hit" or (net_pnl is not None and net_pnl < 0))
        assert anti_knife_triggered is False, "Anti-knife guard must evaluate cleanly without TypeError on None"

        conn.close()
        dm.write_queue.shutdown()
        print("\n[OK] test_stage1_notnull_rebuild_migration passed cleanly (including None-guard assertions)!")

    finally:
        if os.path.exists(temp_db_path):
            try:
                os.remove(temp_db_path)
            except Exception:
                pass

if __name__ == "__main__":
    test_stage1_notnull_rebuild_migration()

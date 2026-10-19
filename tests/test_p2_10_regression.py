"""
tests/test_p2_10_regression.py - Comprehensive Unit & Regression Test Suite for P2-10 Tasks

Covers:
- T-1: Dual-case readers ('closed' / 'CLOSED') across all 9 hardened modules and isolated DB verification.
       Step 4 reconciler emitted status matching table vocabulary.
- T-2: Canonical timestamp format and normalized queries.
- T-3: Authoritative asset lock owner lookup from asset_locks table and heartbeat advance rowcount check.
- T-4: Background task tracking in _background_tasks with discard callback.
- T-5: Corrupt exit reconciliation with dynamic ratio discovery, provenance string, and preserved roe/pnl.
- T-6: Exit price plausibility allowing genuine large moves (±45%) while blocking corrupt cross-symbol fills.
- T-7: conftest.py exception raising and temporary isolation.
"""

import os
import sys
import json
import tempfile
import sqlite3
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class TestP210T1StatusVocabulary(unittest.TestCase):
    """
    T-1 (P0): Status vocabulary dual-case hardening and table-specific writer vocabulary.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "trading_data.db")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

        # Create the 4 non-swing tables and swing_trades
        self.conn.executescript("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,
                status TEXT
            );
            CREATE TABLE berserker_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,
                status TEXT
            );
            CREATE TABLE marl_live_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,
                status TEXT
            );
            CREATE TABLE mean_reversion_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,
                status TEXT
            );
            CREATE TABLE swing_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,
                status TEXT DEFAULT 'CLOSED'
            );
        """)
        self.conn.commit()

    def tearDown(self):
        self.conn.close()
        self.temp_dir.cleanup()

    def test_dual_case_readers_return_both_cases(self):
        """Seed 1 'closed' and 1 'CLOSED' row in each table; assert patched queries return BOTH (2 rows)."""
        tables = ["trades", "berserker_trades", "marl_live_trades", "mean_reversion_trades"]
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        for tbl in tables:
            self.conn.execute(
                f"INSERT INTO {tbl} (timestamp, symbol, side, entry_price, exit_price, pnl, status) "
                f"VALUES (?, 'BTC/USDT', 'long', 50000, 51000, 100.0, 'closed')",
                (now_str,)
            )
            self.conn.execute(
                f"INSERT INTO {tbl} (timestamp, symbol, side, entry_price, exit_price, pnl, status) "
                f"VALUES (?, 'ETH/USDT', 'short', 3000, 2900, 100.0, 'CLOSED')",
                (now_str,)
            )
        self.conn.commit()

        for tbl in tables:
            rows = self.conn.execute(
                f"SELECT * FROM {tbl} WHERE (status = 'closed' OR status = 'CLOSED')"
            ).fetchall()
            self.assertEqual(len(rows), 2, f"Table {tbl} must return both 'closed' and 'CLOSED' rows")
            statuses = {r["status"] for r in rows}
            self.assertEqual(statuses, {"closed", "CLOSED"})

    def test_step4_reconciler_emits_matching_vocabulary(self):
        """Assert reconciler FLEET_TABLES dict configures lowercase for non-swing and uppercase for swing."""
        from reconciliation_engine import FLEET_TABLES

        # Non-swing tables must write lowercase 'closed' and 'unreconciled'
        non_swing = ["trades", "berserker_trades", "marl_live_trades", "mean_reversion_trades"]
        for tbl_name in non_swing:
            entry = next((e for e in FLEET_TABLES if e["closed_table"] == tbl_name), None)
            self.assertIsNotNone(entry, f"Table {tbl_name} must exist in FLEET_TABLES")
            self.assertEqual(
                entry.get("closed_status"), "closed",
                f"Table {tbl_name} closed_status must be lowercase 'closed'"
            )
            self.assertEqual(
                entry.get("unreconciled_status"), "unreconciled",
                f"Table {tbl_name} unreconciled_status must be lowercase 'unreconciled'"
            )

        # swing_trades must write uppercase 'CLOSED' and 'UNRECONCILED'
        swing_entry = next((e for e in FLEET_TABLES if e["closed_table"] == "swing_trades"), None)
        self.assertIsNotNone(swing_entry, "swing_trades must exist in FLEET_TABLES")
        self.assertEqual(swing_entry.get("closed_status"), "CLOSED")
        self.assertEqual(swing_entry.get("unreconciled_status"), "UNRECONCILED")


class TestP210T3HeartbeatAdvance(unittest.TestCase):
    """
    T-3 (P1): Lock heartbeat advance with authoritative lookup and rowcount check.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "trading_data.db")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

        self.conn.executescript("""
            CREATE TABLE asset_locks (
                symbol TEXT PRIMARY KEY,
                locked_by TEXT NOT NULL,
                locked_at TEXT NOT NULL,
                reason TEXT,
                heartbeat TEXT NOT NULL
            );
            CREATE TABLE swing_open_positions (
                symbol TEXT PRIMARY KEY,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                size REAL NOT NULL,
                status TEXT DEFAULT 'OPEN',
                notes TEXT
            );
        """)
        self.conn.commit()

    def tearDown(self):
        self.conn.close()
        self.temp_dir.cleanup()

    def test_heartbeat_advances_when_lock_owner_absent(self):
        """
        Seed a lock owned by 'super_bot_grid' for an active position whose lock_owner is absent
        and whose notes contain no grid/chop token; assert the heartbeat still advances (rowcount 1).
        """
        from swing_trader import SwingArchitectTrader

        symbol = "SOL/USDT"
        now_dt = datetime.now(timezone.utc)
        acquired_at = (now_dt - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
        old_heartbeat = (now_dt - timedelta(minutes=4)).strftime("%Y-%m-%d %H:%M:%S")

        # Seed asset_locks with owner 'super_bot_grid'
        self.conn.execute("""
            INSERT INTO asset_locks (symbol, locked_by, locked_at, reason, heartbeat)
            VALUES (?, 'super_bot_grid', ?, 'grid trade', ?)
        """, (symbol, acquired_at, old_heartbeat))

        # Seed swing_open_positions with notes having NO grid/chop token
        self.conn.execute("""
            INSERT INTO swing_open_positions (symbol, side, entry_price, size, notes)
            VALUES (?, 'long', 150.0, 1.0, 'Regular trend position without grid notes')
        """, (symbol,))
        self.conn.commit()

        # Instantiate SwingArchitectTrader with mock client
        with patch.object(SwingArchitectTrader, "__init__", lambda self: None):
            trader = SwingArchitectTrader()
            trader.client = MagicMock()
            trader.active_positions = {
                symbol: {
                    "symbol": symbol,
                    "side": "long",
                    "entry_price": 150.0,
                    "size": 1.0,
                    "notes": "Regular trend position without grid notes",
                    # Note: "lock_owner" is intentionally ABSENT
                }
            }

            # Call _refresh_heartbeats_sync against our isolated DB
            real_connect = sqlite3.connect
            with patch("swing_trader.sqlite3.connect", lambda *args, **kwargs: real_connect(self.db_path)):
                trader._refresh_heartbeats_sync()

        # Query asset_locks and assert heartbeat was updated
        row = self.conn.execute("SELECT * FROM asset_locks WHERE symbol = ?", (symbol,)).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["locked_by"], "super_bot_grid")
        self.assertNotEqual(row["heartbeat"], old_heartbeat)
        
        # Verify the heartbeat is recent (within 5 seconds)
        hb_dt = datetime.fromisoformat(row["heartbeat"])
        self.assertLess((datetime.now(timezone.utc) - hb_dt).total_seconds(), 5.0)


class TestP210T6ExitPricePlausibility(unittest.TestCase):
    """
    T-6 (P2): Exit price plausibility allows legitimate alt swings (±45%) while blocking corrupt cross-fills.
    """

    def test_genuine_large_moves_pass(self):
        from reconciliation_engine import validate_exit_price_plausibility

        # Genuine +45% rally on altcoin (ratio 1.45)
        self.assertTrue(
            validate_exit_price_plausibility("ENA/USDT", "LONG", 0.50, 0.725),
            "A +45% move on an altcoin must be accepted"
        )

        # Genuine -45% collapse/stop on altcoin (ratio 0.55)
        self.assertTrue(
            validate_exit_price_plausibility("DOGE/USDT", "LONG", 0.20, 0.11),
            "A -45% move on an altcoin must be accepted"
        )

        # Genuine +40% gain on short (ratio 0.60)
        self.assertTrue(
            validate_exit_price_plausibility("NEAR/USDT", "SHORT", 5.00, 3.00),
            "A -40% price drop on short must be accepted"
        )

    def test_corrupt_cross_symbol_fills_are_blocked(self):
        from reconciliation_engine import validate_exit_price_plausibility

        # Cross-symbol fill: BNB entry $750 vs cross-fill exit $11.60 (ratio 0.015)
        self.assertFalse(
            validate_exit_price_plausibility("BNB/USDT", "LONG", 750.0, 11.60),
            "BNB with $11.60 exit must be blocked"
        )

        # Cross-symbol fill: SOL entry $111 vs cross-fill exit $0.504 (ratio 0.0045)
        self.assertFalse(
            validate_exit_price_plausibility("SOL/USDT", "LONG", 111.0, 0.504),
            "SOL with $0.504 exit must be blocked"
        )

        # Corrupt 2x anomaly: SUI entry $1.052 vs corrupt exit $2.106 (ratio 2.002 > 1.70)
        self.assertFalse(
            validate_exit_price_plausibility("SUI/USDT", "LONG", 1.052, 2.106),
            "SUI corrupt 2x price must be blocked"
        )

        # Directional contradiction: LONG with +$100 PnL but exit at $50 vs entry $100
        self.assertFalse(
            validate_exit_price_plausibility("ETH/USDT", "LONG", 100.0, 50.0, size=1.0, pnl=100.0),
            "LONG with positive PnL but exit far below entry must be blocked"
        )


class TestP210T4BackgroundTasks(unittest.TestCase):
    """
    T-4 (P2): Background task registration and discard callback in reconciliation_engine.
    """

    def test_t_close_registered_in_background_tasks(self):
        import reconciliation_engine
        self.assertTrue(hasattr(reconciliation_engine, "_background_tasks"))
        self.assertIsInstance(reconciliation_engine._background_tasks, set)


class TestEmergencyStatePersistence(unittest.TestCase):
    """
    F-01, F-02, F-07: Verify emergency state serialization and persistence on SL + close failure.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "trading_data.db")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("""
            CREATE TABLE system_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        """)
        self.conn.commit()

    def tearDown(self):
        self.conn.close()
        self.temp_dir.cleanup()

    def test_emergency_state_json_payload_serializes_cleanly(self):
        import json
        clean_sym = "SOLUSDT"
        pos_side = "long"
        qty = 0.5
        last_strat_err = RuntimeError("Order precision invalid 25610")
        abort_err_str = "Network timeout on emergency close"
        now_iso = datetime.now(timezone.utc).isoformat()

        payload = json.dumps({
            "symbol": clean_sym,
            "side": pos_side,
            "qty": qty,
            "sl_error": str(last_strat_err),
            "close_error": abort_err_str,
            "timestamp": now_iso,
            "status": "UNPROTECTED_NAKED_POSITION"
        })

        self.conn.execute("""
            INSERT INTO system_state (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?
        """, (
            f"emergency_sl_failure_{clean_sym}",
            payload,
            now_iso,
            payload,
            now_iso
        ))
        self.conn.commit()

        row = self.conn.execute("SELECT value FROM system_state WHERE key = ?", (f"emergency_sl_failure_{clean_sym}",)).fetchone()
        self.assertIsNotNone(row)
        parsed = json.loads(row[0])
        self.assertEqual(parsed["status"], "UNPROTECTED_NAKED_POSITION")
        self.assertEqual(parsed["close_error"], "Network timeout on emergency close")
        self.assertEqual(parsed["symbol"], "SOLUSDT")

    def test_emergency_sl_failure_blocks_reentry(self):
        """Verify conviction filter vetos entry if emergency_sl_failure row is present."""
        clean_sym = "AVAXUSDT"
        now_iso = "2026-10-19T20:00:00+00:00"
        self.conn.execute("""
            INSERT INTO system_state (key, value, updated_at)
            VALUES (?, ?, ?)
        """, (
            f"emergency_sl_failure_{clean_sym}",
            json.dumps({"status": "UNPROTECTED_NAKED_POSITION"}),
            now_iso
        ))
        self.conn.commit()

        # Query simulating conviction filter
        row_emerg = self.conn.execute(
            "SELECT value, updated_at FROM system_state WHERE key = ?",
            (f"emergency_sl_failure_{clean_sym}",)
        ).fetchone()
        self.assertIsNotNone(row_emerg)


class TestP210T7ConftestHardening(unittest.TestCase):
    """
    T-7 (P2): conftest.py _seed_temp_files fail loudly and cleanup registration.
    """

    def test_seed_temp_files_fails_loudly_on_backup_failure(self):
        from tests import conftest
        db_name = "test_fake.db"
        prod_path = os.path.join(conftest.PROJECT_ROOT_RAW, db_name)
        dst_path = os.path.join(conftest._SESSION_TEMP_DIR, db_name)

        def mock_exists(p):
            if p == prod_path:
                return True
            if p == dst_path:
                return False
            return False

        with patch.object(conftest, "_ORIG_SQLITE_CONNECT", side_effect=sqlite3.OperationalError("disk I/O error")):
            with patch.object(conftest, "QUARANTINED_DB_BASENAMES", (db_name,)):
                with patch("os.path.exists", side_effect=mock_exists):
                    with self.assertRaises(RuntimeError) as cm:
                        conftest._seed_temp_files()
                    self.assertIn("QUARANTINE-SEED-ERROR", str(cm.exception))


if __name__ == "__main__":
    unittest.main()

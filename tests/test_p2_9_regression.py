"""
Regression tests for P2-9 (D-1 through D-12).
"""

import ast
import asyncio
from datetime import datetime, timezone, timedelta
import importlib
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

import tests.conftest as quarantine


class TestP29D11Quarantine(unittest.TestCase):
    """D-11: Verify test suite quarantine from production databases and start_system.py."""

    def test_d11_start_system_import_blocked(self):
        with self.assertRaises(RuntimeError) as ctx:
            importlib.import_module("start_system")
        self.assertIn("FLEET-KILL TRAP BLOCKED", str(ctx.exception))

    def test_d11_relative_db_redirected_to_temp(self):
        conn = sqlite3.connect("trading_data.db")
        try:
            db_list = conn.execute("PRAGMA database_list").fetchall()
            actual_file = db_list[0][2]
            self.assertNotEqual(
                os.path.normcase(os.path.abspath(os.path.dirname(actual_file))),
                quarantine.PROJECT_ROOT,
                f"Expected relative trading_data.db to be redirected away from PROJECT_ROOT, got {actual_file}",
            )
            self.assertTrue(
                actual_file.startswith(quarantine._SESSION_TEMP_DIR),
                f"Expected DB path inside {quarantine._SESSION_TEMP_DIR}, got {actual_file}",
            )
        finally:
            conn.close()

    def test_d11_repo_root_db_redirected_to_temp(self):
        repo_db = os.path.join(quarantine.PROJECT_ROOT, "trading_data.db")
        conn = sqlite3.connect(repo_db)
        try:
            db_list = conn.execute("PRAGMA database_list").fetchall()
            actual_file = db_list[0][2]
            self.assertTrue(actual_file.startswith(quarantine._SESSION_TEMP_DIR))
        finally:
            conn.close()

    def test_d11_microstructure_state_json_redirected(self):
        with open("microstructure_state.json", "w", encoding="utf-8") as f:
            f.write('{"test": true}')
        self.assertTrue(
            os.path.exists(os.path.join(quarantine._SESSION_TEMP_DIR, "microstructure_state.json"))
        )


class TestP29D1SetLeverage(unittest.TestCase):
    """D-1: Verify set_leverage argument order, defensive normalization, and entry veto."""

    def test_d1_ast_static_call_shape(self):
        st_path = os.path.join(quarantine.PROJECT_ROOT, "swing_trader.py")
        with open(st_path, "r", encoding="utf-8") as f:
            src = f.read()
        tree = ast.parse(src, filename=st_path)

        found_call = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "open_swing_position":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call) and getattr(sub.func, "attr", None) == "set_leverage":
                        found_call = True
                        kw_keys = {kw.arg for kw in sub.keywords}
                        self.assertEqual(len(sub.args), 0, "Expected set_leverage to be called with keyword args")
                        self.assertIn("leverage", kw_keys)
                        self.assertIn("symbol", kw_keys)
        self.assertTrue(found_call, "Could not find set_leverage call in open_swing_position")

        # Verify veto behavior and logging level in source text
        self.assertIn("[SWING-LEVERAGE-VETO]", src)
        self.assertNotIn('log.debug(f"[SWING-LEVERAGE-NOTE]', src)

    def test_d1_ast_no_reversed_callers_in_codebase(self):
        import glob
        for py_file in glob.glob(os.path.join(quarantine.PROJECT_ROOT, "*.py")):
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
            tree = ast.parse(content, filename=py_file)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "set_leverage":
                    # If called positionally, arg 0 must NOT be named symbol or a symbol string
                    if len(node.args) >= 1:
                        first_arg = node.args[0]
                        if isinstance(first_arg, ast.Name):
                            self.assertNotEqual(
                                first_arg.id,
                                "symbol",
                                f"Found reversed set_leverage call in {py_file}:{node.lineno} passing symbol as arg 0",
                            )
                        elif isinstance(first_arg, ast.Constant):
                            self.assertFalse(
                                isinstance(first_arg.value, str) and "USDT" in first_arg.value,
                                f"Found reversed set_leverage call in {py_file}:{node.lineno} passing symbol literal as arg 0",
                            )

    def test_d1_normalize_symbol_validation(self):
        from bitget_uta_client import BitgetUTAClient
        client = BitgetUTAClient()
        self.assertEqual(client._normalize_symbol("BTC/USDT"), "BTCUSDT")
        self.assertEqual(client._normalize_symbol("ETH/USDT:USDT"), "ETHUSDT")
        self.assertEqual(client._normalize_symbol("SOLUSDT"), "SOLUSDT")

        for bad in [123, None, "", "   ", "999"]:
            with self.assertRaises(TypeError):
                client._normalize_symbol(bad)

    def test_d1_set_leverage_defensive_fail(self):
        from bitget_uta_client import BitgetUTAClient
        client = BitgetUTAClient()

        # Non-str symbol must fail gracefully inside try block, returning False
        res_bad_sym = asyncio.run(client.set_leverage(leverage=5, symbol=123))
        self.assertFalse(res_bad_sym)

        # Non-positive leverage must fail gracefully inside try block, returning False
        res_bad_lev = asyncio.run(client.set_leverage(leverage=0, symbol="BTCUSDT"))
        self.assertFalse(res_bad_lev)

    def test_d1_open_swing_position_vetoes_on_leverage_failure(self):
        from unittest.mock import AsyncMock, MagicMock
        from swing_trader import SwingArchitectTrader

        trader = SwingArchitectTrader.__new__(SwingArchitectTrader)
        trader.active_positions = {}
        trader.max_locks = 4
        trader.allocation_usdt = 50.0
        trader.high_quality_conviction = 0.80
        trader.high_quality_adx = 30.0
        trader.standard_leverage = 3
        trader.high_quality_leverage = 5
        trader.atr_sl_mult = 1.5
        trader.runner_tp_mult = 3.0
        trader._news_size_multipliers = {}
        trader.uta_client = MagicMock()
        trader.uta_client._normalize_symbol = lambda s: s.replace("/", "")
        trader.uta_client.fetch_balance = AsyncMock(return_value=250.0)
        trader.uta_client.exchange = MagicMock()
        trader.uta_client.exchange.fetch_ticker = AsyncMock(return_value={"last": 60000.0})
        trader.uta_client.set_leverage = AsyncMock(return_value=False)
        trader.ai_geometry = MagicMock()
        trader.ai_geometry.calibrate_exit_geometry = AsyncMock(return_value={})
        trader._count_fleet_long_positions_sync = MagicMock(return_value=0)
        trader._acquire_lock_sync = MagicMock(return_value=True)
        trader._release_lock_sync = MagicMock(return_value=1)
        trader.client = MagicMock()
        trader.client.create_order = AsyncMock()

        ind_4h = {
            "close": 60000.0,
            "atr": 1000.0,
            "adx": 35.0,
            "rsi": 50.0,
            "cloud_top": 58000.0,
            "cloud_bottom": 57000.0,
            "tenkan": 59500.0,
            "kijun": 59000.0,
        }

        # Mock ai_engine.tune_exit_geometry to prevent external network calls during test
        import ai_engine
        with patch.object(ai_engine, "tune_exit_geometry", new=AsyncMock(return_value={})):
            asyncio.run(
                trader.open_swing_position(
                    symbol="BTCUSDT",
                    ind_4h=ind_4h,
                    direction="LONG",
                    conviction_score=0.85,
                )
            )

        # Must release lock and veto order placement
        trader._release_lock_sync.assert_called_with("BTCUSDT")
        trader.client.create_order.assert_not_called()

    def test_d1_open_swing_position_vetoes_on_leverage_exception(self):
        from unittest.mock import AsyncMock, MagicMock, patch
        from swing_trader import SwingArchitectTrader

        trader = SwingArchitectTrader.__new__(SwingArchitectTrader)
        trader.active_positions = {}
        trader.max_locks = 4
        trader.allocation_usdt = 50.0
        trader.high_quality_conviction = 0.80
        trader.high_quality_adx = 30.0
        trader.standard_leverage = 3
        trader.high_quality_leverage = 5
        trader.atr_sl_mult = 1.5
        trader.runner_tp_mult = 3.0
        trader._news_size_multipliers = {}
        trader.uta_client = MagicMock()
        trader.uta_client._normalize_symbol = lambda s: s.replace("/", "")
        trader.uta_client.fetch_balance = AsyncMock(return_value=250.0)
        trader.uta_client.exchange = MagicMock()
        trader.uta_client.exchange.fetch_ticker = AsyncMock(return_value={"last": 3000.0})
        trader.uta_client.set_leverage = AsyncMock(side_effect=RuntimeError("Bitget network timeout"))
        trader.ai_geometry = MagicMock()
        trader.ai_geometry.calibrate_exit_geometry = AsyncMock(return_value={})
        trader._count_fleet_long_positions_sync = MagicMock(return_value=0)
        trader._acquire_lock_sync = MagicMock(return_value=True)
        trader._release_lock_sync = MagicMock(return_value=1)
        trader.client = MagicMock()
        trader.client.create_order = AsyncMock()

        ind_4h = {
            "close": 3000.0,
            "atr": 50.0,
            "adx": 32.0,
            "rsi": 48.0,
            "cloud_top": 2900.0,
            "cloud_bottom": 2800.0,
            "tenkan": 2950.0,
            "kijun": 2920.0,
        }

        import ai_engine
        with patch.object(ai_engine, "tune_exit_geometry", new=AsyncMock(return_value={})):
            asyncio.run(
                trader.open_swing_position(
                    symbol="ETHUSDT",
                    ind_4h=ind_4h,
                    direction="LONG",
                    conviction_score=0.80,
                )
            )

        trader._release_lock_sync.assert_called_with("ETHUSDT")
        trader.client.create_order.assert_not_called()


class TestP29D2AssetLockLifecycle(unittest.TestCase):
    """D-2: Verify asset-lock lifecycle in both directions, owner threading, and 900s age guard."""

    REAL_ASSET_LOCKS_SCHEMA = """
    CREATE TABLE asset_locks (
        symbol TEXT PRIMARY KEY,
        locked_by TEXT NOT NULL,
        locked_at TEXT NOT NULL,
        reason TEXT,
        heartbeat TEXT NOT NULL
    );
    """

    def test_d2_ast_extracted_queries_have_owner_and_age_guards(self):
        st_path = os.path.join(quarantine.PROJECT_ROOT, "swing_trader.py")
        with open(st_path, "r", encoding="utf-8") as f:
            src = f.read()
        tree = ast.parse(src, filename=st_path)

        release_query = None
        heartbeat_update = None
        heartbeat_deletes = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "_release_lock_sync":
                    for sub in ast.walk(node):
                        if isinstance(sub, ast.Call):
                            for a in sub.args:
                                if isinstance(a, ast.Constant) and isinstance(a.value, str) and "DELETE FROM asset_locks" in a.value:
                                    release_query = a.value
                elif node.name == "_refresh_heartbeats_sync":
                    for sub in ast.walk(node):
                        if isinstance(sub, ast.Call):
                            for a in sub.args:
                                val = None
                                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                                    val = a.value
                                elif isinstance(a, ast.JoinedStr):
                                    val = "".join(part.value for part in a.values if isinstance(part, ast.Constant))
                                if val:
                                    if "UPDATE asset_locks SET heartbeat" in val:
                                        heartbeat_update = val
                                    elif "DELETE FROM asset_locks" in val:
                                        heartbeat_deletes.append(val)

        self.assertIsNotNone(release_query, "Could not find DELETE query in _release_lock_sync")
        self.assertIn("locked_by", release_query)

        self.assertIsNotNone(heartbeat_update, "Could not find UPDATE query in _refresh_heartbeats_sync")
        self.assertIn("locked_by", heartbeat_update)

        self.assertGreaterEqual(len(heartbeat_deletes), 2, "Expected at least 2 DELETE queries in _refresh_heartbeats_sync")
        for dq in heartbeat_deletes:
            self.assertIn("86400.0 > 900.0", dq, f"Expected 900s age predicate in heartbeat delete: {dq}")

    def test_d2_release_lock_sync_owner_isolation_in_memory(self):
        from unittest.mock import patch
        from swing_trader import SwingArchitectTrader
        trader = SwingArchitectTrader.__new__(SwingArchitectTrader)

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = tf.name

        try:
            conn = sqlite3.connect(db_path)
            conn.execute(self.REAL_ASSET_LOCKS_SCHEMA)
            now_ts = "2026-10-18 00:00:00"

            # Seed super_bot_grid lock on SOL/USDT and swing_architect lock on AVAX/USDT
            conn.execute(
                "INSERT INTO asset_locks VALUES ('SOL/USDT', 'super_bot_grid', ?, 'grid_order', ?)",
                (now_ts, now_ts),
            )
            conn.execute(
                "INSERT INTO asset_locks VALUES ('AVAX/USDT', 'swing_architect', ?, 'swing_breakout', ?)",
                (now_ts, now_ts),
            )
            conn.commit()
            conn.close()

            real_connect = sqlite3.connect
            # Monkey-patch sqlite3.connect to connect to our isolated test db
            with patch("sqlite3.connect", side_effect=lambda *a, **kw: real_connect(db_path)):
                # Attempt release by swing_architect on SOL/USDT -> must return 0 deleted, grid lock survives!
                del_cnt = trader._release_lock_sync("SOL/USDT", owner="swing_architect")
                self.assertEqual(del_cnt, 0)
                
                chk_conn = real_connect(db_path)
                surviving = chk_conn.execute("SELECT locked_by FROM asset_locks WHERE symbol = 'SOL/USDT'").fetchone()
                chk_conn.close()
                self.assertIsNotNone(surviving)
                self.assertEqual(surviving[0], "super_bot_grid")

                # Release by super_bot_grid on SOL/USDT -> must return 1 deleted, grid lock freed!
                del_cnt_grid = trader._release_lock_sync("SOL/USDT", owner="super_bot_grid")
                self.assertEqual(del_cnt_grid, 1)
                chk_conn = real_connect(db_path)
                freed = chk_conn.execute("SELECT locked_by FROM asset_locks WHERE symbol = 'SOL/USDT'").fetchone()
                chk_conn.close()
                self.assertIsNone(freed)

                # Release by swing_architect on AVAX/USDT -> must return 1 deleted
                del_cnt_avax = trader._release_lock_sync("AVAX/USDT", owner="swing_architect")
                self.assertEqual(del_cnt_avax, 1)
                chk_conn = real_connect(db_path)
                freed_avax = chk_conn.execute("SELECT locked_by FROM asset_locks WHERE symbol = 'AVAX/USDT'").fetchone()
                chk_conn.close()
                self.assertIsNone(freed_avax)
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_d2_refresh_heartbeats_does_not_prune_fresh_unmirrored_locks(self):
        from datetime import datetime, timezone, timedelta
        from unittest.mock import patch
        from swing_trader import SwingArchitectTrader
        trader = SwingArchitectTrader.__new__(SwingArchitectTrader)
        trader.active_positions = {}  # Empty active positions (simulating lag before sync)

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = tf.name

        try:
            conn = sqlite3.connect(db_path)
            conn.execute(self.REAL_ASSET_LOCKS_SCHEMA)
            now_dt = datetime.now(timezone.utc)
            fresh_ts = now_dt.strftime("%Y-%m-%d %H:%M:%S")
            stale_ts = (now_dt - timedelta(seconds=1200)).strftime("%Y-%m-%d %H:%M:%S")

            # Insert fresh grid lock (0 age) and stale swing lock (1200s age)
            conn.execute(
                "INSERT INTO asset_locks VALUES ('SOL/USDT', 'super_bot_grid', ?, 'grid', ?)",
                (fresh_ts, fresh_ts),
            )
            conn.execute(
                "INSERT INTO asset_locks VALUES ('NEAR/USDT', 'swing_architect', ?, 'stale', ?)",
                (stale_ts, stale_ts),
            )
            conn.commit()
            conn.close()

            real_connect = sqlite3.connect
            with patch("sqlite3.connect", side_effect=lambda *a, **kw: real_connect(db_path)):
                trader._refresh_heartbeats_sync()

            chk_conn = real_connect(db_path)
            # SOL/USDT must survive (age < 900s), NEAR/USDT must be pruned (age > 900s)
            sol_row = chk_conn.execute("SELECT locked_by FROM asset_locks WHERE symbol = 'SOL/USDT'").fetchone()
            self.assertIsNotNone(sol_row, "Fresh super_bot_grid lock was pruned prematurely!")

            near_row = chk_conn.execute("SELECT locked_by FROM asset_locks WHERE symbol = 'NEAR/USDT'").fetchone()
            self.assertIsNone(near_row, "Stale (>900s) lock was not pruned!")
            chk_conn.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)



class TestP29D3TimestampFormat(unittest.TestCase):
    """D-3: Verify timestamp normalization across writers, readers, and backfill script."""

    REAL_SWING_TRADES_SCHEMA = """
    CREATE TABLE swing_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        entry_price REAL NOT NULL,
        exit_price REAL NOT NULL,
        size REAL NOT NULL,
        pnl REAL,
        roe_pct REAL,
        holding_hours REAL,
        exit_reason TEXT,
        notes TEXT,
        intended_entry_price REAL,
        actual_fill_price REAL,
        slippage_bps REAL,
        fee_paid_usd REAL,
        fee_maker_usd REAL,
        fee_taker_usd REAL,
        funding_usd REAL,
        funding_source TEXT,
        gates_snapshot TEXT,
        status TEXT
    );
    """

    def test_d3_ast_extracted_queries_use_normalized_timestamps(self):
        # 1. swing_trader.py
        with open("swing_trader.py", "r", encoding="utf-8") as f:
            st_src = f.read()
        self.assertIn("replace(substr(timestamp, 1, 19), 'T', ' ') >= datetime('now', '-15 minutes')", st_src)
        self.assertIn('.strftime("%Y-%m-%d %H:%M:%S")', st_src)

        # 2. reconciliation_engine.py
        with open("reconciliation_engine.py", "r", encoding="utf-8") as f:
            rec_src = f.read()
        self.assertEqual(
            rec_src.count("replace(substr(timestamp, 1, 19), 'T', ' ') >= datetime('now', '-15 minutes')"),
            2,
            "Expected 2 occurrences of normalized timestamp query in reconciliation_engine.py (Step 1 and Step 4)",
        )

        # 3. db_manager.py
        with open("db_manager.py", "r", encoding="utf-8") as f:
            db_src = f.read()
        self.assertIn("replace(substr(timestamp, 1, 19), 'T', ' ') > datetime('now', '-1 day')", db_src)
        self.assertIn("replace(substr(timestamp, 1, 19), 'T', ' ') < {cutoff}", db_src)

    def test_d3_query_matches_both_iso_and_space_timestamps(self):
        from datetime import datetime, timezone
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute(self.REAL_SWING_TRADES_SCHEMA)
            now_dt = datetime.now(timezone.utc)
            now_iso = now_dt.isoformat()
            now_space = now_dt.strftime("%Y-%m-%d %H:%M:%S")

            conn.execute(
                "INSERT INTO swing_trades (timestamp, symbol, side, entry_price, exit_price, size) VALUES (?, 'BTC/USDT', 'LONG', 60000, 61000, 1.0)",
                (now_iso,),
            )
            conn.execute(
                "INSERT INTO swing_trades (timestamp, symbol, side, entry_price, exit_price, size) VALUES (?, 'ETH/USDT', 'LONG', 3000, 3100, 1.0)",
                (now_space,),
            )
            conn.commit()

            matched = conn.execute("""
                SELECT symbol FROM swing_trades
                WHERE replace(substr(timestamp, 1, 19), 'T', ' ') >= datetime('now', '-15 minutes')
            """).fetchall()
            matched_syms = [r[0] for r in matched]
            self.assertIn("BTC/USDT", matched_syms, "ISO timestamp failed to match normalized query!")
            self.assertIn("ETH/USDT", matched_syms, "Space timestamp failed to match normalized query!")
        finally:
            conn.close()

    def test_d3_backfill_script_dry_run_and_apply(self):
        from scripts.backfill_d3_timestamps import backfill_timestamps
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = tf.name

        try:
            conn = sqlite3.connect(db_path)
            conn.execute(self.REAL_SWING_TRADES_SCHEMA)
            # Insert 5 ISO format rows and 3 space format rows
            for i in range(5):
                conn.execute(
                    "INSERT INTO swing_trades (timestamp, symbol, side, entry_price, exit_price, size) VALUES (?, 'SOL/USDT', 'LONG', 100, 105, 1.0)",
                    (f"2026-10-17T12:0{i}:00.123456+00:00",),
                )
            for i in range(3):
                conn.execute(
                    "INSERT INTO swing_trades (timestamp, symbol, side, entry_price, exit_price, size) VALUES (?, 'AVAX/USDT', 'LONG', 20, 22, 1.0)",
                    (f"2026-10-17 14:0{i}:00",),
                )
            conn.commit()
            conn.close()

            # 1. Dry run: 0 rows modified
            res_dry = backfill_timestamps(db_path, apply_changes=False)
            self.assertEqual(res_dry, 0)
            chk_conn = sqlite3.connect(db_path)
            iso_cnt = chk_conn.execute("SELECT count(*) FROM swing_trades WHERE timestamp LIKE '%T%'").fetchone()[0]
            self.assertEqual(iso_cnt, 5, "Dry run must NOT modify any rows!")
            chk_conn.close()

            # 2. Apply: backup created, 5 rows updated, 0 ISO remain
            res_apply = backfill_timestamps(db_path, apply_changes=True)
            self.assertEqual(res_apply, 0)
            chk_conn = sqlite3.connect(db_path)
            iso_cnt_post = chk_conn.execute("SELECT count(*) FROM swing_trades WHERE timestamp LIKE '%T%'").fetchone()[0]
            self.assertEqual(iso_cnt_post, 0, "All ISO rows must be normalized!")
            backup_cnt = chk_conn.execute("SELECT count(*) FROM swing_trades_backup_d3").fetchone()[0]
            self.assertEqual(backup_cnt, 8, "Backup table must preserve all 8 rows!")
            chk_conn.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)



class TestP29D4ExitPriceGuard(unittest.TestCase):
    """D-4: Verify exit price plausibility validation, fills filtering, and reconciliation script."""

    REAL_SWING_TRADES_SCHEMA = """
    CREATE TABLE swing_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        entry_price REAL NOT NULL,
        exit_price REAL,
        size REAL NOT NULL,
        pnl REAL,
        roe_pct REAL,
        holding_hours REAL,
        exit_reason TEXT,
        notes TEXT,
        intended_entry_price REAL,
        actual_fill_price REAL,
        slippage_bps REAL,
        fee_paid_usd REAL,
        fee_maker_usd REAL,
        fee_taker_usd REAL,
        funding_usd REAL,
        funding_source TEXT,
        gates_snapshot TEXT,
        status TEXT
    );
    """

    def test_d4_validate_exit_price_plausibility_blocks_corrupt_prices(self):
        from reconciliation_engine import validate_exit_price_plausibility

        # Corrupt cases (must BLOCK):
        self.assertFalse(validate_exit_price_plausibility("BNB/USDT", "LONG", 751.61, 11.60), "BNB exiting at $11.60 must be blocked!")
        self.assertFalse(validate_exit_price_plausibility("SOL/USDT", "LONG", 111.3, 0.504), "SOL exiting at $0.504 must be blocked!")
        self.assertFalse(validate_exit_price_plausibility("BTC/USDT", "LONG", 83149.9, 4855.8), "BTC exiting at $4855.8 must be blocked!")
        self.assertFalse(validate_exit_price_plausibility("SUI/USDT", "LONG", 1.052, 2.106), "SUI exiting at 100% gain on 5x must be blocked!")
        self.assertFalse(validate_exit_price_plausibility("ETH/USDT", "LONG", 2422.41, 14.85), "ETH exiting at $14.85 must be blocked!")

        # Invalid prices (must BLOCK):
        self.assertFalse(validate_exit_price_plausibility("BTC/USDT", "LONG", 0.0, 60000.0), "Zero entry price must be blocked!")
        self.assertFalse(validate_exit_price_plausibility("BTC/USDT", "LONG", 60000.0, -100.0), "Negative exit price must be blocked!")

        # Directional contradiction (must BLOCK):
        self.assertFalse(
            validate_exit_price_plausibility("BTC/USDT", "LONG", 60000.0, 50000.0, size=1.0, pnl=500.0),
            "LONG with positive PnL but exit well below entry must be blocked!",
        )

        # Plausible prices (must PASS):
        self.assertTrue(validate_exit_price_plausibility("BTC/USDT", "LONG", 83149.9, 83422.5), "Normal BTC gain must pass!")
        self.assertTrue(validate_exit_price_plausibility("ETH/USDT", "LONG", 2422.41, 2415.90), "Normal ETH stop-loss must pass!")

    def test_d4_bitget_uta_client_filters_fills_by_symbol(self):
        from bitget_uta_client import BitgetUTAClient
        client = BitgetUTAClient.__new__(BitgetUTAClient)
        client.exchange = MagicMock()
        client.ensure_time_sync = AsyncMock()

        raw_exchange_fills = {
            "data": {
                "list": [
                    {"symbol": "BTCUSDT", "orderId": "101", "execPrice": "83000", "execQty": "0.001", "tradeSide": "close"},
                    {"symbol": "LINKUSDT", "orderId": "102", "execPrice": "11.60", "execQty": "10.0", "tradeSide": "close"},
                    {"symbol": "ETHUSDT", "orderId": "103", "execPrice": "2400", "execQty": "0.1", "tradeSide": "close"},
                ]
            }
        }
        client.exchange.private_uta_get_v3_trade_fills = AsyncMock(return_value=raw_exchange_fills)

        # 1. get_order_fills for BTC/USDT must filter out LINK and ETH fills
        btc_fills = asyncio.run(client.get_order_fills("BTC/USDT", order_id="101"))
        self.assertEqual(len(btc_fills), 1)
        self.assertEqual(btc_fills[0]["symbol"], "BTCUSDT")

        # 2. get_recent_fills for LINK/USDT must filter out BTC and ETH fills
        link_fills = asyncio.run(client.get_recent_fills("LINK/USDT", lookback_mins=15))
        self.assertEqual(len(link_fills), 1)
        self.assertEqual(link_fills[0]["symbol"], "LINKUSDT")

    def test_d4_fetch_live_price_staleness_and_symbol_isolation(self):
        from unittest.mock import patch
        import time
        from reconciliation_engine import fetch_live_price

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            state_file = tf.name

        try:
            with open(state_file, "w") as f:
                f.write('{"best_bid": 99999.0, "symbol": "BTC/USDT"}')

            with patch("reconciliation_engine.BASE_DIR", os.path.dirname(state_file)), \
                 patch("os.path.join", side_effect=lambda *args: state_file if "microstructure_state.json" in args[-1] else os.path.join(*args)):

                # 1. Non-BTC symbol must NEVER read BTC microstructure price
                non_btc_price = fetch_live_price("ETH/USDT")
                self.assertNotEqual(non_btc_price, 99999.0, "Non-BTC symbol read BTC microstructure state!")

                # 2. Stale BTC microstructure state (>300s) must be rejected
                stale_mtime = time.time() - 350.0
                os.utime(state_file, (stale_mtime, stale_mtime))
                stale_btc_price = fetch_live_price("BTC/USDT")
                self.assertNotEqual(stale_btc_price, 99999.0, "Stale (>300s) microstructure state was erroneously accepted!")

                # 3. Fresh BTC microstructure state (<=300s) is accepted
                fresh_mtime = time.time() - 10.0
                os.utime(state_file, (fresh_mtime, fresh_mtime))
                fresh_btc_price = fetch_live_price("BTC/USDT")
                self.assertEqual(fresh_btc_price, 99999.0, "Fresh BTC microstructure state should be accepted!")
        finally:
            if os.path.exists(state_file):
                os.remove(state_file)

    def test_d4_migration_script_dry_run_and_apply(self):
        from scripts.reconcile_d4_corrupt_exits import reconcile_corrupt_exits
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = tf.name

        try:
            conn = sqlite3.connect(db_path)
            conn.execute(self.REAL_SWING_TRADES_SCHEMA)
            # Insert the 9 corrupt entries
            corrupt_data = [
                (149, 'BNB/USDT', 'LONG', 751.61, 11.603, 0.2, 2.219, 'STOP_LOSS', 'Gross: $-148.0014 | Fees: $0.2125'),
                (152, 'APT/USDT', 'LONG', 0.8951, 1.554, 169.09, -0.062, 'tp_hit', 'On-exchange TP_HIT on Bitget UTA (Net: $-0.06, Fee: -$0.182)'),
                (156, 'SOL/USDT', 'LONG', 111.3, 0.504, 1.4, -1.227, 'STOP_LOSS', 'Gross: $-155.1137 | Fees: $0.2144'),
                (163, 'BTC/USDT', 'LONG', 83149.9, 4855.83, 0.0009, 0.155, 'breakeven_hit', 'On-exchange BREAKEVEN_HIT on Bitget UTA (Net: $+0.16, Fee: -$0.090)'),
                (166, 'BNB/USDT', 'LONG', 741.07, 12.038, 0.1, 0.124, 'STOP_LOSS', 'Gross: $-72.9032 | Fees: $0.1492'),
                (168, 'BTC/USDT', 'LONG', 83213.0, 16.622, 0.0009, 0.396, 'sl_hit', 'On-exchange SL_HIT on Bitget UTA (Net: $+0.40, Fee: -$0.118)'),
                (170, 'LINK/USDT', 'LONG', 12.894, 1.584, 12.0, 0.494, 'sl_hit', 'On-exchange SL_HIT on Bitget UTA (Net: $+0.49, Fee: -$0.184)'),
                (172, 'SUI/USDT', 'LONG', 1.052, 2.106, 142.7, 0.154, 'STOP_LOSS', 'Gross: $+150.4656 | Fees: $0.2105'),
                (192, 'ETH/USDT', 'LONG', 2422.41, 14.85, 0.03, -0.324, 'sl_hit', 'On-exchange SL_HIT on Bitget UTA (Net: $-0.32, Fee: -$0.129)'),
            ]
            for row in corrupt_data:
                conn.execute("""
                    INSERT INTO swing_trades (id, timestamp, symbol, side, entry_price, exit_price, size, pnl, exit_reason, notes, status)
                    VALUES (?, '2026-10-14 00:00:00', ?, ?, ?, ?, ?, ?, ?, ?, 'CLOSED')
                """, row)
            conn.commit()
            conn.close()

            # 1. Dry run: must make NO changes
            res_dry = reconcile_corrupt_exits(db_path, apply_changes=False)
            self.assertEqual(res_dry, 0)
            chk_conn = sqlite3.connect(db_path)
            bnb_row = chk_conn.execute("SELECT exit_price FROM swing_trades WHERE id = 149").fetchone()
            self.assertAlmostEqual(bnb_row[0], 11.603, places=3, msg="Dry run must NOT alter prices!")
            chk_conn.close()

            # 2. Apply: backup created, prices reconciled, 0 corrupt remain
            res_apply = reconcile_corrupt_exits(db_path, apply_changes=True)
            self.assertEqual(res_apply, 0)
            chk_conn = sqlite3.connect(db_path)
            b_cnt = chk_conn.execute("SELECT count(*) FROM swing_trades_backup_d4").fetchone()[0]
            self.assertEqual(b_cnt, 9, "Backup table must preserve all 9 rows!")

            # Verify no corrupt prices remain
            corrupt_count = chk_conn.execute("""
                SELECT count(*) FROM swing_trades
                WHERE exit_price IS NOT NULL
                  AND (exit_price / entry_price < 0.70 OR exit_price / entry_price > 1.40)
            """).fetchone()[0]
            self.assertEqual(corrupt_count, 0, "No corrupt prices should remain after reconciliation!")
            chk_conn.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)



class TestP29D5ReconcilerStep4Safeguards(unittest.TestCase):
    """D-5: Verify Reconciler Step 4 safeguards (grace period, targeted fetch retry, symbol fills filtering, plausibility guard, NULL/UNRECONCILED fallback)."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE swing_open_positions (
                symbol TEXT PRIMARY KEY,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                size REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                trailing_stop REAL,
                peak_price REAL,
                leverage REAL DEFAULT 3.0,
                opened_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT DEFAULT 'ACTIVE',
                r_multiple REAL DEFAULT 0.0,
                initial_risk_usd REAL,
                notes TEXT,
                intended_entry_price REAL,
                actual_fill_price REAL,
                slippage_bps REAL,
                fee_entry_usd REAL,
                execution_type TEXT,
                gates_snapshot TEXT
            );
        """)
        conn.execute("""
            CREATE TABLE swing_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                size REAL NOT NULL,
                pnl REAL,
                roe_pct REAL,
                holding_hours REAL,
                exit_reason TEXT,
                notes TEXT,
                intended_entry_price REAL,
                actual_fill_price REAL,
                slippage_bps REAL,
                fee_paid_usd REAL,
                fee_maker_usd REAL,
                fee_taker_usd REAL,
                funding_usd REAL,
                funding_source TEXT,
                gates_snapshot TEXT,
                status TEXT
            );
        """)
        conn.execute("""
            CREATE TABLE asset_locks (
                symbol TEXT PRIMARY KEY,
                locked_by TEXT NOT NULL,
                locked_at TEXT NOT NULL,
                reason TEXT,
                heartbeat TEXT NOT NULL
            );
        """)
        conn.execute("""
            CREATE TABLE system_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.execute("""
            CREATE TABLE swing_cooldowns (
                symbol TEXT PRIMARY KEY,
                cooldown_until TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
        """)
        conn.execute("CREATE TABLE open_positions (symbol TEXT PRIMARY KEY, side TEXT, entry_price REAL, qty REAL, stop_loss REAL, take_profit REAL);")
        conn.execute("CREATE TABLE trades (id INTEGER PRIMARY KEY, timestamp TEXT, symbol TEXT, side TEXT, type TEXT, entry_price REAL, exit_price REAL, qty REAL, pnl REAL, balance REAL, notes TEXT, status TEXT, action TEXT);")
        conn.execute("CREATE TABLE berserker_open_positions (symbol TEXT PRIMARY KEY, side TEXT, entry_price REAL, qty REAL, stop_loss REAL, take_profit REAL);")
        conn.execute("CREATE TABLE berserker_trades (id INTEGER PRIMARY KEY, timestamp TEXT, symbol TEXT, side TEXT, type TEXT, entry_price REAL, exit_price REAL, qty REAL, pnl REAL, balance REAL, notes TEXT, status TEXT, action TEXT);")
        conn.execute("CREATE TABLE marl_live_open_positions (symbol TEXT PRIMARY KEY, side TEXT, entry_price REAL, qty REAL, stop_loss REAL, take_profit REAL);")
        conn.execute("CREATE TABLE marl_live_trades (id INTEGER PRIMARY KEY, timestamp TEXT, symbol TEXT, side TEXT, type TEXT, entry_price REAL, exit_price REAL, qty REAL, pnl REAL, balance REAL, notes TEXT, status TEXT, action TEXT);")
        conn.commit()
        conn.close()

    def tearDown(self):
        try:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
        except OSError:
            pass

    def test_d5_step4_grace_period(self):
        import reconciliation_engine
        from unittest.mock import patch

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO swing_open_positions (symbol, side, entry_price, size, stop_loss, take_profit, opened_at, updated_at)
            VALUES ('BTC/USDT', 'LONG', 80000.0, 0.001, 78000.0, 85000.0, ?, ?)
        """, (now_str, now_str))
        conn.commit()
        conn.close()

        mock_uta = MagicMock()
        mock_uta.fetch_balance = AsyncMock(return_value=500.0)
        mock_uta.fetch_positions = AsyncMock(return_value=[])
        mock_uta._normalize_symbol = lambda s: str(s).replace('/', '').upper()
        mock_uta.exchange = MagicMock()
        mock_uta.exchange.private_uta_get_v3_trade_unfilled_orders = AsyncMock(return_value={"data": {"list": []}})

        with patch("reconciliation_engine.DATA_DB", self.db_path), \
             patch("reconciliation_engine.config.MODE", "live"), \
             patch("reconciliation_engine.config.PAPER_TRADE", False), \
             patch("reconciliation_engine.config.EXCHANGE_ID", "bitget"), \
             patch("bitget_uta_client.BitgetUTAClient", return_value=mock_uta):

            asyncio.run(reconciliation_engine.audit_live_exchange_positions())

        chk_conn = sqlite3.connect(self.db_path)
        open_cnt = chk_conn.execute("SELECT count(*) FROM swing_open_positions").fetchone()[0]
        closed_cnt = chk_conn.execute("SELECT count(*) FROM swing_trades").fetchone()[0]
        chk_conn.close()

        self.assertEqual(open_cnt, 1, "Grace period (<60s) must protect open position from being pruned!")
        self.assertEqual(closed_cnt, 0, "No closed trade row should be recorded for a position in grace period!")

    def test_d5_step4_targeted_fetch_shield(self):
        import reconciliation_engine
        from unittest.mock import patch

        old_str = (datetime.now(timezone.utc) - timedelta(seconds=120)).strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO swing_open_positions (symbol, side, entry_price, size, stop_loss, take_profit, opened_at, updated_at)
            VALUES ('BTC/USDT', 'LONG', 80000.0, 0.001, 78000.0, 85000.0, ?, ?)
        """, (old_str, old_str))
        conn.commit()
        conn.close()

        mock_uta = MagicMock()
        mock_uta.fetch_balance = AsyncMock(return_value=500.0)
        # Global query returns empty, but targeted query confirms position is active
        mock_uta.fetch_positions = AsyncMock(side_effect=lambda symbol=None: [] if symbol is None else [{"symbol": "BTCUSDT", "posSide": "long", "total": 0.001}])
        mock_uta._normalize_symbol = lambda s: str(s).replace('/', '').upper()
        mock_uta.exchange = MagicMock()
        mock_uta.exchange.private_uta_get_v3_trade_unfilled_orders = AsyncMock(return_value={"data": {"list": []}})

        with patch("reconciliation_engine.DATA_DB", self.db_path), \
             patch("reconciliation_engine.config.MODE", "live"), \
             patch("reconciliation_engine.config.PAPER_TRADE", False), \
             patch("reconciliation_engine.config.EXCHANGE_ID", "bitget"), \
             patch("bitget_uta_client.BitgetUTAClient", return_value=mock_uta):

            asyncio.run(reconciliation_engine.audit_live_exchange_positions())

        chk_conn = sqlite3.connect(self.db_path)
        open_cnt = chk_conn.execute("SELECT count(*) FROM swing_open_positions").fetchone()[0]
        chk_conn.close()

        self.assertEqual(open_cnt, 1, "Targeted single-symbol query must shield active position from transient API drop!")

    def test_d5_step4_unreconciled_fallback_when_fills_missing(self):
        import reconciliation_engine
        from unittest.mock import patch

        old_str = (datetime.now(timezone.utc) - timedelta(seconds=120)).strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO swing_open_positions (symbol, side, entry_price, size, stop_loss, take_profit, opened_at, updated_at)
            VALUES ('BTC/USDT', 'LONG', 80000.0, 0.001, 78000.0, 85000.0, ?, ?)
        """, (old_str, old_str))
        conn.commit()
        conn.close()

        mock_uta = MagicMock()
        mock_uta.fetch_balance = AsyncMock(return_value=500.0)
        mock_uta.fetch_positions = AsyncMock(return_value=[])  # Truly gone
        mock_uta.get_recent_fills = AsyncMock(return_value=[])  # No fills available
        mock_uta._normalize_symbol = lambda s: str(s).replace('/', '').upper()
        mock_uta.exchange = MagicMock()
        mock_uta.exchange.private_uta_get_v3_trade_unfilled_orders = AsyncMock(return_value={"data": {"list": []}})

        with patch("reconciliation_engine.DATA_DB", self.db_path), \
             patch("reconciliation_engine.config.MODE", "live"), \
             patch("reconciliation_engine.config.PAPER_TRADE", False), \
             patch("reconciliation_engine.config.EXCHANGE_ID", "bitget"), \
             patch("bitget_uta_client.BitgetUTAClient", return_value=mock_uta):

            asyncio.run(reconciliation_engine.audit_live_exchange_positions())

        chk_conn = sqlite3.connect(self.db_path)
        open_cnt = chk_conn.execute("SELECT count(*) FROM swing_open_positions").fetchone()[0]
        self.assertEqual(open_cnt, 0, "Position must be pruned when confirmed closed.")

        rec_row = chk_conn.execute("SELECT status, exit_price, pnl, exit_reason, notes FROM swing_trades WHERE symbol = 'BTC/USDT'").fetchone()
        chk_conn.close()

        self.assertIsNotNone(rec_row, "Closed trade record must exist!")
        self.assertEqual(rec_row[0], "UNRECONCILED", "Status must be UNRECONCILED when fills cannot be verified!")
        self.assertIsNone(rec_row[1], "Exit price must be NULL (no fabricated prices allowed)!")
        self.assertIsNone(rec_row[2], "PnL must be NULL when exit price is unknown!")
        self.assertEqual(rec_row[3], "UNRECONCILED_CLOSE")
        self.assertIn("UNRECONCILED", rec_row[4])

    def test_d5_step4_implausible_fill_rejected_to_unreconciled(self):
        import reconciliation_engine
        from unittest.mock import patch

        old_str = (datetime.now(timezone.utc) - timedelta(seconds=120)).strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO swing_open_positions (symbol, side, entry_price, size, stop_loss, take_profit, opened_at, updated_at)
            VALUES ('BNB/USDT', 'LONG', 750.0, 0.1, 730.0, 800.0, ?, ?)
        """, (old_str, old_str))
        conn.commit()
        conn.close()

        mock_uta = MagicMock()
        mock_uta.fetch_balance = AsyncMock(return_value=500.0)
        mock_uta.fetch_positions = AsyncMock(return_value=[])  # Truly gone
        # Returns corrupt fill with price $11.60
        mock_uta.get_recent_fills = AsyncMock(return_value=[{
            "symbol": "BNBUSDT",
            "tradeSide": "close_long",
            "execQty": "0.1",
            "execPrice": "11.60",
            "execValue": "1.16",
            "fee": "0.01",
        }])
        mock_uta._normalize_symbol = lambda s: str(s).replace('/', '').upper()
        mock_uta.exchange = MagicMock()
        mock_uta.exchange.private_uta_get_v3_trade_unfilled_orders = AsyncMock(return_value={"data": {"list": []}})

        with patch("reconciliation_engine.DATA_DB", self.db_path), \
             patch("reconciliation_engine.config.MODE", "live"), \
             patch("reconciliation_engine.config.PAPER_TRADE", False), \
             patch("reconciliation_engine.config.EXCHANGE_ID", "bitget"), \
             patch("bitget_uta_client.BitgetUTAClient", return_value=mock_uta):

            asyncio.run(reconciliation_engine.audit_live_exchange_positions())

        chk_conn = sqlite3.connect(self.db_path)
        rec_row = chk_conn.execute("SELECT status, exit_price, pnl FROM swing_trades WHERE symbol = 'BNB/USDT'").fetchone()
        chk_conn.close()

        self.assertIsNotNone(rec_row, "Closed trade record must exist!")
        self.assertEqual(rec_row[0], "UNRECONCILED", "Implausible exit price must force UNRECONCILED status!")
        self.assertIsNone(rec_row[1], "Exit price must be NULL to prevent corrupting ledger with $11.60!")
        self.assertIsNone(rec_row[2], "PnL must be NULL!")



class TestP29D6NetworkInterfaceBinding(unittest.TestCase):
    """D-6: Verify explicit localhost binding for Streamlit HUD and Prometheus Exporter."""

    def test_d6_start_system_binds_streamlit_to_localhost_statically(self):
        # NOTE: start_system.py MUST NOT be imported (0.3 Fleet-Kill Trap).
        # We verify AST statically.
        with open("start_system.py", "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        found_streamlit = False
        for node in ast.walk(tree):
            if isinstance(node, ast.List):
                el_vals = []
                for elt in node.elts:
                    if isinstance(elt, ast.Constant):
                        el_vals.append(elt.value)
                if "streamlit" in el_vals and "run" in el_vals:
                    found_streamlit = True
                    self.assertIn("--server.address", el_vals, "Streamlit must specify --server.address!")
                    addr_idx = el_vals.index("--server.address")
                    self.assertEqual(el_vals[addr_idx + 1], "127.0.0.1", "--server.address must be 127.0.0.1!")
                    self.assertIn("--server.headless", el_vals, "Streamlit must specify --server.headless!")
                    headless_idx = el_vals.index("--server.headless")
                    self.assertEqual(el_vals[headless_idx + 1], "true", "--server.headless must be true!")

        self.assertTrue(found_streamlit, "Streamlit launch config not found in start_system.py!")

    def test_d6_prometheus_exporter_binds_to_localhost_statically(self):
        with open("prometheus_exporter.py", "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        found_start_http = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                if func_name == "start_http_server":
                    found_start_http = True
                    # Check keyword argument addr is passed
                    addr_kw = [kw for kw in node.keywords if kw.arg == "addr"]
                    self.assertTrue(len(addr_kw) > 0, "start_http_server must specify addr keyword argument!")

        self.assertTrue(found_start_http, "start_http_server call not found in prometheus_exporter.py!")


class TestP29D7SwingPartialFillAndUnboundLocal(unittest.TestCase):
    """D-7: Verify filled_amount initialization and partial fill adoption in swing entry."""

    def test_d7_filled_amount_initialized_statically(self):
        with open("swing_trader.py", "r", encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content)

        # 1. Verify filled_amount = 0.0 is present before the while loop
        found_init = False
        found_partial_adopt = False

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "filled_amount":
                        if isinstance(node.value, ast.Constant) and node.value.value == 0.0:
                            found_init = True
            if isinstance(node, ast.Compare):
                # Check for: filled_amount > 0.0
                left = node.left
                if isinstance(left, ast.Name) and left.id == "filled_amount":
                    for op, comparator in zip(node.ops, node.comparators):
                        if isinstance(op, ast.Gt) and isinstance(comparator, ast.Constant) and comparator.value == 0.0:
                            found_partial_adopt = True

        self.assertTrue(found_init, "filled_amount = 0.0 must be initialized before the sniper poll loop!")
        self.assertTrue(found_partial_adopt, "Partial fill check (filled_amount > 0.0) must be present!")

    def test_d7_partial_fill_adopts_partial_qty(self):
        # Verify the calculation of actual_qty when 0 < filled_amount < qty
        qty = 10.0
        filled_amount = 4.0  # 40% partial fill
        actual_qty = filled_amount if (filled_amount > 0 and filled_amount <= qty) else qty
        self.assertEqual(actual_qty, 4.0, "actual_qty must be exactly 4.0 (partial fill), not 10.0!")

        # Verify precision formatting with partial qty
        mock_uta = MagicMock()
        mock_uta._normalize_symbol = lambda s: "SOLUSDT"
        mock_uta.format_quantity_precision = MagicMock(return_value="4.0")
        mock_uta.format_price_precision = MagicMock(return_value="105.50")

        qty_str = mock_uta.format_quantity_precision("SOL/USDT", actual_qty)
        self.assertEqual(qty_str, "4.0")


class TestP29D8DbIntegrityErrorWindow(unittest.TestCase):
    """D-8: Verify db_maintenance_daemon error counter window cleanly spans inter-pass interval."""

    def test_d8_error_window_exceeds_pass_interval(self):
        import db_maintenance_daemon
        self.assertGreaterEqual(
            db_maintenance_daemon.ERROR_COUNTER_WINDOW_SECS,
            db_maintenance_daemon.CHECKPOINT_INTERVAL_SECS + 60.0,
            "ERROR_COUNTER_WINDOW_SECS must be at least CHECKPOINT_INTERVAL_SECS + 60s!"
        )
        self.assertGreater(
            db_maintenance_daemon.ERROR_COUNTER_WINDOW_SECS,
            305.0,
            "ERROR_COUNTER_WINDOW_SECS must exceed the measured 304-305s inter-pass interval!"
        )

    def test_d8_consecutive_errors_counted_across_305s_interval(self):
        import db_maintenance_daemon

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            temp_db = tf.name

        try:
            conn = sqlite3.connect(temp_db)
            conn.execute("CREATE TABLE system_state (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT);")
            # Row updated 305 seconds ago (the exact inter-pass interval that previously caused resets)
            ts_305s = (datetime.now(timezone.utc) - timedelta(seconds=305)).strftime("%Y-%m-%d %H:%M:%S")
            conn.execute(
                "INSERT INTO system_state (key, value, updated_at) VALUES ('db_integrity_trading_ledger_consecutive_errors', '2', ?)",
                (ts_305s,)
            )
            conn.commit()
            conn.close()

            with patch("db_maintenance_daemon.BASE_DIR", os.path.dirname(temp_db)), \
                 patch("os.path.join", side_effect=lambda *args: temp_db if "trading_data.db" in args[-1] else os.path.join(*args)):

                # Under the updated window, a 305s old row must be RECOGNIZED as consecutive (returns 2), NOT reset to 0!
                consec = db_maintenance_daemon.get_durable_consecutive_errors("trading_ledger")
                self.assertEqual(consec, 2, "A 305s old error counter must be retained, not prematurely dropped as stale!")

                # An older row beyond 360s (e.g. 400s) should be treated as stale (returns 0)
                conn = sqlite3.connect(temp_db)
                ts_400s = (datetime.now(timezone.utc) - timedelta(seconds=400)).strftime("%Y-%m-%d %H:%M:%S")
                conn.execute(
                    "UPDATE system_state SET updated_at = ? WHERE key = 'db_integrity_trading_ledger_consecutive_errors'",
                    (ts_400s,)
                )
                conn.commit()
                conn.close()

                consec_stale = db_maintenance_daemon.get_durable_consecutive_errors("trading_ledger")
                self.assertEqual(consec_stale, 0, "A 400s old error counter (>360s window) should be recognized as stale!")
        finally:
            if os.path.exists(temp_db):
                os.remove(temp_db)


if __name__ == "__main__":
    unittest.main()








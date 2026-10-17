"""
Regression tests for P2-9 (D-1 through D-12).
"""

import ast
import asyncio
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


if __name__ == "__main__":
    unittest.main()

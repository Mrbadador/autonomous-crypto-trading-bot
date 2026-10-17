"""
Regression tests for P2-9 (D-1 through D-12).
"""

import ast
import importlib
import os
import sqlite3
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()

import ast
import importlib.util
import os
import unittest

import db_manager

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKTEST_PATH = os.path.join(REPO_ROOT, "backtests", "run_backtest.py")
PROD_DB = os.path.join(REPO_ROOT, "trading_data.db")


def _load_module(name):
    spec = importlib.util.spec_from_file_location(name, BACKTEST_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBacktestDbIsolation(unittest.TestCase):
    """The replay must never open the production ledger for write, and must
    never delete it."""

    @classmethod
    def setUpClass(cls):
        with open(BACKTEST_PATH, "r", encoding="utf-8") as f:
            cls.src = f.read()
        cls.tree = ast.parse(cls.src)
        cls.module = _load_module("run_backtest_isolation_under_test")

    def test_database_manager_is_constructed_with_an_explicit_path(self):
        calls = [
            n for n in ast.walk(self.tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "DatabaseManager"
        ]
        self.assertTrue(calls, "expected a DatabaseManager(...) call")
        for call in calls:
            self.assertTrue(
                call.args or call.keywords,
                "DatabaseManager() with no db_path resolves to the production "
                "trading_data.db and wipes the live ledger",
            )

    def test_production_ledger_name_is_never_named_in_the_module(self):
        self.assertNotIn(
            "trading_data.db",
            self.src,
            "the replay module must never name the production ledger, because "
            "it deletes and rewrites every path it names",
        )

    def test_sim_db_path_is_scoped_to_the_backtests_directory(self):
        sim_db = self.module.resolve_sim_db_path()
        self.assertTrue(
            os.path.abspath(sim_db).startswith(os.path.abspath(self.module.SIM_DIR) + os.sep),
            f"sim db {sim_db} must live under {self.module.SIM_DIR}",
        )
        self.assertEqual(os.path.basename(sim_db), "backtest_sim.db")
        self.assertNotEqual(os.path.abspath(sim_db), os.path.abspath(PROD_DB))

    def test_production_ledger_is_not_in_the_purge_list(self):
        for path in self.module.sim_artifact_paths():
            self.assertNotEqual(
                os.path.abspath(path), os.path.abspath(PROD_DB),
                "the purge list must never include the production ledger",
            )

    def test_database_manager_default_resolves_to_the_production_ledger(self):
        """Premise check: DatabaseManager() with no argument resolves to the
        repo-root production ledger, so the replay must always pass db_path."""
        default_path = os.path.join(
            os.path.dirname(os.path.abspath(db_manager.__file__)), "trading_data.db"
        )
        self.assertEqual(os.path.abspath(default_path), os.path.abspath(PROD_DB))
        self.assertNotEqual(
            os.path.abspath(self.module.resolve_sim_db_path()),
            os.path.abspath(default_path),
        )

    def test_simulation_workspace_is_outside_the_repository_root(self):
        workspace = self.module.simulation_workspace()
        self.assertTrue(
            os.path.abspath(workspace).startswith(os.path.abspath(self.module.SIM_DIR) + os.sep),
            f"workspace {workspace} must live under {self.module.SIM_DIR}",
        )

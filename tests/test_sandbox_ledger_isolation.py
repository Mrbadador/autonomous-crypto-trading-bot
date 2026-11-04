import importlib.util
import os
import shutil
import sqlite3
import tempfile
import unittest
from unittest import mock

import db_manager

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_DIR = os.path.join(REPO_ROOT, "archive", "legacy_agents")
PROD_DB = os.path.join(REPO_ROOT, "trading_data.db")
SANDBOX_ENV = "TRADING_DATA_DB_PATH"


def _load_state_manager():
    if LEGACY_DIR not in sys_path():
        import sys
        sys.path.insert(0, LEGACY_DIR)
    spec = importlib.util.spec_from_file_location(
        "state_manager_under_test", os.path.join(LEGACY_DIR, "state_manager.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sys_path():
    import sys
    return sys.path


class TestSandboxLedgerIsolation(unittest.TestCase):
    """The replay must never reach the production ledger through an indirect
    path. A CWD sandbox is not enough: DatabaseManager() resolves its default
    path from its own module directory, so any component that constructs one
    without a path opens the live ledger no matter where the process runs."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="sandbox_ledger_")
        self.sandbox_db = os.path.join(self.tmp, "sim.db")
        self.cwd = os.getcwd()
        os.chdir(self.tmp)
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        os.environ[SANDBOX_ENV] = self.sandbox_db
        self.addCleanup(self.env.stop)
        self.addCleanup(os.chdir, self.cwd)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_default_database_manager_honors_the_sandbox_override(self):
        """A default-constructed manager must land on the sandbox ledger."""
        os.environ[SANDBOX_ENV] = self.sandbox_db
        manager = db_manager.DatabaseManager()
        self.assertEqual(
            os.path.abspath(manager.db_path),
            os.path.abspath(self.sandbox_db),
            "DatabaseManager() ignored the sandbox ledger override and opened "
            "the production trading_data.db",
        )
        self.assertNotEqual(os.path.abspath(manager.db_path), os.path.abspath(PROD_DB))

    def test_engine_state_never_constructs_a_manager_without_a_path(self):
        """EngineState owns a DatabaseManager; it must ask for the sandbox one."""
        sm = _load_state_manager()
        recorded = {}

        class _Recorder:
            def __init__(self, db_path=None):
                recorded["db_path"] = db_path

            def _get_conn(self):
                raise AssertionError("no connection expected in this test")

        with mock.patch.object(sm, "DatabaseManager", _Recorder):
            sm.EngineState(sim_instance=None)

        self.assertEqual(
            recorded.get("db_path"),
            self.sandbox_db,
            "EngineState constructed a DatabaseManager without an explicit "
            "ledger path, so it opens the production ledger during a replay",
        )

    def test_state_manager_reads_the_sandbox_ledger(self):
        """The archived state manager must read its balances from the sandbox."""
        sm = _load_state_manager()
        conn = sqlite3.connect(self.sandbox_db)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS system_state (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO system_state (key, value, updated_at) "
            "VALUES ('user_total_deposit', '4242.5', '2026-11-04T00:00:00+00:00')"
        )
        conn.commit()
        conn.close()

        state = sm.EngineState(sim_instance=None)
        self.assertAlmostEqual(
            state.start_balance,
            4242.5,
            places=2,
            msg="EngineState did not read the sandbox ledger; it read the "
                "production ledger or an empty file instead",
        )

    def test_state_manager_names_no_ledger_path_literally(self):
        """The literal production name must not be hardcoded in the connector."""
        with open(os.path.join(LEGACY_DIR, "state_manager.py"), "r", encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn(
            'sqlite3.connect("trading_data.db"',
            src,
            "state_manager connects to the production ledger by literal name; "
            "it must resolve the sandbox ledger instead",
        )

    def test_replay_publishes_the_sandbox_ledger_path(self):
        """The replay must point every default-constructed manager at the sandbox."""
        spec = importlib.util.spec_from_file_location(
            "run_backtest_sandbox_under_test",
            os.path.join(REPO_ROOT, "backtests", "run_backtest.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        module.apply_sandbox_ledger_path()
        self.assertEqual(
            os.path.abspath(os.environ[SANDBOX_ENV]),
            os.path.abspath(module.resolve_sim_db_path()),
            "the replay did not publish the sandbox ledger path",
        )

    def test_replay_only_cycles_the_symbols_it_loaded(self):
        """A symbol with no historical file must not be simulated."""
        spec = importlib.util.spec_from_file_location(
            "run_backtest_symbols_under_test",
            os.path.join(REPO_ROOT, "backtests", "run_backtest.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        em = module.MockExchangeManager(symbols=["BTC/USDT", "AVAX/USDT"])
        self.assertIn("BTC/USDT", em.loaded_symbols)
        self.assertNotIn(
            "AVAX/USDT",
            em.loaded_symbols,
            "a symbol with no historical file was loaded and will raise KeyError "
            "on every cycle",
        )

        with open(os.path.join(REPO_ROOT, "backtests", "run_backtest.py"), "r", encoding="utf-8") as f:
            src = f.read()
        self.assertIn(
            "for symbol in mock_em.loaded_symbols",
            src,
            "the replay still cycles every configured symbol, including the "
            "ones it skipped for lack of data",
        )


if __name__ == "__main__":
    unittest.main()

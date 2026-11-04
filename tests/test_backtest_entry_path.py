import asyncio
import importlib.util
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKTEST_PATH = os.path.join(REPO_ROOT, "backtests", "run_backtest.py")
PROD_DB = os.path.join(REPO_ROOT, "trading_data.db")


def _load_run_backtest():
    for extra in (os.path.join(REPO_ROOT, "archive"),
                  os.path.join(REPO_ROOT, "archive", "legacy_agents")):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    spec = importlib.util.spec_from_file_location("rb_entry_under_test", BACKTEST_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBacktestEntryPath(unittest.TestCase):
    """The replay must be able to open a simulated position.

    A replay that reports zero trades because every entry raised is
    indistinguishable from a strategy with no edge, so the entry path is
    exercised here with a stubbed LLM verdict.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bt_entry_")
        self.cwd = os.getcwd()
        os.chdir(self.tmp)
        self.addCleanup(os.chdir, self.cwd)
        self.addCleanup(shutil.rmtree, self.tmp, True)

        self.rb = _load_run_backtest()
        import ai_engine
        self.ai_engine = ai_engine

    async def _drive_one_entry(self):
        """Run the replay's own cycle until it opens a position."""
        import config
        from bot_v10_backtest import run_cycle

        self.rb.apply_sandbox_ledger_path()
        em = self.rb.MockExchangeManager(symbols=["ETH/USDT"], initial_balance=config.INITIAL_BALANCE)
        state = self.rb.EngineState(sim_instance=em)

        async def _verdict(self, *args, **kwargs):
            return {
                "action": "open_long",
                "entry_quality": 70.0,
                "notes": "stubbed verdict",
                "size_mult": 1.0,
                "sl_mult": 1.5,
            }

        with mock.patch.object(self.ai_engine.LLMClient, "get_risk_params", _verdict):
            for candle in em.timeline[800:3000]:
                em.update_mock_data(candle)
                for sym in em.loaded_symbols:
                    await run_cycle(sym, em, state, candle_time=candle)
                    if em.positions:
                        return em, state
        return em, state

    def test_a_gated_entry_opens_a_simulated_position(self):
        em, state = asyncio.run(self._drive_one_entry())
        self.assertTrue(
            em.positions,
            "the replay opened no position: every entry that passed the gates "
            "failed before the position was created",
        )

    def test_calculate_exits_signature_matches_its_callers(self):
        """Every caller of calculate_exits must unpack all three values."""
        import ast

        from archive.legacy_agents import core_engine  # noqa: F401
        sys.path.insert(0, os.path.join(REPO_ROOT, "archive", "legacy_agents"))
        from core_engine import calculate_exits

        self.assertEqual(
            len(calculate_exits(100.0, "long", 1.0)),
            3,
            "calculate_exits no longer returns (soft_sl, tp, hard_sl)",
        )

        targets = []
        for rel in ("archive/bot_v10_backtest.py",
                    "archive/experimental_traders/live_trader.py",
                    "archive/live_trader_backup_pre_audit_fix.py"):
            path = os.path.join(REPO_ROOT, rel)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Tuple)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "calculate_exits"
                ):
                    targets.append((rel, len(node.targets[0].elts)))

        self.assertTrue(targets, "expected at least one calculate_exits caller")
        for rel, count in targets:
            self.assertEqual(
                count,
                3,
                f"{rel} unpacks calculate_exits into {count} values but the "
                "function returns 3 (soft_sl, tp, hard_sl)",
            )

    def test_the_replay_never_touches_the_production_ledger(self):
        """Running the replay must not modify the production ledger."""
        before = os.path.getsize(PROD_DB)
        mtime_before = os.path.getmtime(PROD_DB)
        asyncio.run(self._drive_one_entry())
        self.assertEqual(
            os.path.getsize(PROD_DB),
            before,
            "the replay changed the size of the production ledger",
        )
        self.assertEqual(
            os.path.getmtime(PROD_DB),
            mtime_before,
            "the replay wrote to the production ledger",
        )


if __name__ == "__main__":
    unittest.main()

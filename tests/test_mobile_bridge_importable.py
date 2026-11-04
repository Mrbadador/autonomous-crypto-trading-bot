import importlib.util
import os
import unittest
from unittest import mock


class TestMobileBridgeImportable(unittest.TestCase):
    """archive/legacy_agents/mobile_web_bridge.py must import without raising,
    and must not ship a hardcoded password default."""

    PATH = os.path.join("archive", "legacy_agents", "mobile_web_bridge.py")

    def _load(self):
        spec = importlib.util.spec_from_file_location("mwb_under_test", self.PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_module_imports_cleanly_without_a_configured_password(self):
        """The module must import on a host that has no APP_PASSWORD set.

        The workspace .env supplies APP_PASSWORD, so the environment has to be
        cleared explicitly or the failure path is never executed.
        """
        env = {k: v for k, v in os.environ.items() if k not in ("APP_PASSWORD", "SESSION_SECRET_KEY")}
        with mock.patch.dict(os.environ, env, clear=True):
            module = self._load()
        self.assertTrue(module.APP_PASSWORD)
        self.assertNotEqual(module.APP_PASSWORD, "SovereignSwarm2026!")

    def test_module_imports_cleanly_with_a_configured_password(self):
        with mock.patch.dict(os.environ, {"APP_PASSWORD": "configured-secret"}):
            module = self._load()
        self.assertEqual(module.APP_PASSWORD, "configured-secret")

    def test_no_hardcoded_password_literal(self):
        with open(self.PATH, "r", encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("SovereignSwarm2026", src)

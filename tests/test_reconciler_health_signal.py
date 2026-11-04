import asyncio
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import reconciliation_engine as recon


class TestReconcilerHealthSignal(unittest.TestCase):
    """Both audit paths must publish a real 'status' key, otherwise the
    heartbeat telemetry is permanently DEGRADED."""

    def test_paper_audit_reports_healthy_on_a_clean_ledger(self):
        with mock.patch.object(recon, "FLEET_TABLES", []):
            summary = recon.audit_paper_positions()
        self.assertEqual(summary.get("status"), "HEALTHY")
        self.assertIn("positions_active", summary)
        self.assertIn("desyncs_healed", summary)

    def test_paper_audit_reports_degraded_when_a_table_cannot_be_audited(self):
        with tempfile.TemporaryDirectory() as tmp:
            # A directory that does not exist: sqlite cannot even open the file.
            unreadable = os.path.join(tmp, "missing_dir", "unreadable.db")
            with mock.patch.object(recon, "DATA_DB", unreadable):
                summary = recon.audit_paper_positions()
        self.assertEqual(summary.get("status"), "DEGRADED")

    def test_live_audit_reports_healthy_in_paper_mode(self):
        with mock.patch.object(recon.config, "MODE", "paper"), \
                mock.patch.object(recon.config, "PAPER_TRADE", True):
            result = asyncio.run(recon.audit_live_exchange_positions())
        self.assertEqual(result.get("status"), "HEALTHY")
        self.assertEqual(result.get("mode"), "paper_active")

    def test_live_audit_reports_degraded_on_error(self):
        class _Boom:
            async def fetch_balance(self):
                raise RuntimeError("exchange unreachable")

            async def close(self):
                return None

        fake_module = types.ModuleType("bitget_uta_client")
        fake_module.BitgetUTAClient = _Boom
        with mock.patch.object(recon.config, "MODE", "live"), \
                mock.patch.object(recon.config, "EXCHANGE_ID", "bitget"), \
                mock.patch.dict(sys.modules, {"bitget_uta_client": fake_module}):
            result = asyncio.run(recon.audit_live_exchange_positions())
        self.assertEqual(result.get("status"), "DEGRADED")
        self.assertEqual(result.get("mode"), "live_error")

    def test_health_helper_requires_a_real_status(self):
        self.assertTrue(recon._is_healthy({"status": "HEALTHY"}, None))
        self.assertFalse(recon._is_healthy({"status": "DEGRADED"}, None))
        self.assertFalse(recon._is_healthy({}, None))
        self.assertFalse(recon._is_healthy({"status": "HEALTHY"}, {"status": "DEGRADED"}))
        self.assertTrue(recon._is_healthy({"status": "HEALTHY"}, {"status": "HEALTHY"}))

    def test_heartbeat_telemetry_reflects_the_audit_status(self):
        healthy = recon.build_heartbeat_telemetry(
            {"status": "HEALTHY", "positions_active": 0, "desyncs_healed": 0},
            {"status": "HEALTHY"},
        )
        self.assertEqual(healthy["status"], "HEALTHY")

        degraded = recon.build_heartbeat_telemetry(
            {"status": "DEGRADED", "positions_active": 0, "desyncs_healed": 0},
            {"status": "HEALTHY"},
        )
        self.assertEqual(degraded["status"], "DEGRADED")

    def test_heartbeat_log_line_is_not_hardcoded(self):
        src = open(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reconciliation_engine.py"),
            "r", encoding="utf-8",
        ).read()
        self.assertNotIn("Status: OK", src)

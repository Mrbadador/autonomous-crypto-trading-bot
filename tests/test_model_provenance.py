"""
tests/test_model_provenance.py — FIX 2 unit verification for served_model_id provenance

Covers:
  1. ensure_boardroom_provenance_schema: legacy 9-column table gains the nullable
     served_model_id column; pre-existing rows are untouched (NULL, never backfilled).
  2. Round-trip: insert a fake debate row WITH served_model_id, read it back.
  3. ai_engine.red_team_deliberation provenance keys are present in both the served
     path (model string) and fail-closed path (None) — mocked HTTP, no live calls.
"""

import os
import sys
import sqlite3
import tempfile
import asyncio
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db_manager import ensure_boardroom_provenance_schema

LEGACY_SCHEMA = """
CREATE TABLE boardroom_debates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    side TEXT,
    execute INTEGER,
    conviction_score REAL,
    red_team_objection TEXT,
    alpha_thesis TEXT,
    final_rationale TEXT,
    timestamp TEXT
)
"""


class TestProvenanceMigration(unittest.TestCase):
    def setUp(self):
        self.fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(self.fd)
        with sqlite3.connect(self.path) as c:
            c.executescript(LEGACY_SCHEMA)
            c.execute("INSERT INTO boardroom_debates (symbol, side, execute, conviction_score, red_team_objection, alpha_thesis, final_rationale, timestamp) VALUES ('BTCUSDT','LONG',0,0.0,'o','a','r','2026-10-01T00:00:00+00:00')")

    def tearDown(self):
        import gc
        gc.collect()
        try:
            sqlite3.connect(self.path).close()
        except Exception:
            pass
        for suf in ("", "-wal", "-shm"):
            p = self.path + suf
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass

    def test_migration_is_additive_and_idempotent(self):
        self.assertTrue(ensure_boardroom_provenance_schema(self.path))
        c = sqlite3.connect(self.path)
        cols = [r[1] for r in c.execute("PRAGMA table_info(boardroom_debates)")]
        c.close()
        self.assertIn("served_model_id", cols)
        # idempotent second run
        self.assertTrue(ensure_boardroom_provenance_schema(self.path))
        c = sqlite3.connect(self.path)
        cols2 = [r[1] for r in c.execute("PRAGMA table_info(boardroom_debates)")]
        c.close()
        self.assertEqual(cols, cols2)

    def test_existing_rows_untouched_and_null_provenance(self):
        ensure_boardroom_provenance_schema(self.path)
        with sqlite3.connect(self.path) as c:
            n = c.execute("SELECT COUNT(*) FROM boardroom_debates").fetchone()[0]
            row = c.execute("SELECT symbol, served_model_id FROM boardroom_debates WHERE id=1").fetchone()
        self.assertEqual(n, 1)
        self.assertEqual(row[0], "BTCUSDT")
        self.assertIsNone(row[1])

    def test_insert_and_read_back_with_provenance(self):
        ensure_boardroom_provenance_schema(self.path)
        with sqlite3.connect(self.path) as c:
            c.execute(
                "INSERT INTO boardroom_debates (symbol, side, execute, conviction_score, red_team_objection, alpha_thesis, final_rationale, timestamp, served_model_id) VALUES (?,?,?,?,?,?,?,?,?)",
                ("ETHUSDT", "SHORT", 1, 0.82, "obj", "alpha", "rat", "2026-10-13T00:00:00+00:00", "grok-4-fast"),
            )
            got = c.execute("SELECT served_model_id FROM boardroom_debates ORDER BY id DESC LIMIT 1").fetchone()[0]
        self.assertEqual(got, "grok-4-fast")


class TestEngineCapture(unittest.TestCase):
    """red_team_deliberation must report the answering model; no behavior change."""

    def _run(self, fake_resp):
        import ai_engine

        async def fake_groq(session, payload, timeout_sec=30.0):
            return fake_resp

        with mock.patch.object(ai_engine, "_query_groq_cloud_with_rotation", fake_groq), \
             mock.patch.object(ai_engine, "_query_gemini_cloud", lambda session, prompt, timeout_sec=15.0: asyncio.sleep(0, result=None)), \
             mock.patch.object(ai_engine, "get_llm", lambda name: mock.MagicMock(session=None)):
            return asyncio.run(ai_engine.red_team_deliberation("BTCUSDT", "long", {"adx": 20}))

    def test_served_path_reports_model(self):
        resp = {
            "model_used": "openai/gpt-oss-20b",
            "choices": [{"message": {"content": '{"execute": true, "conviction": 80, "bear_objection": "b", "bull_thesis": "t", "rationale": "r"}'}}],
        }
        out = self._run(resp)
        self.assertEqual(out.get("served_model_id"), "openai/gpt-oss-20b")
        self.assertTrue(out["execute"])

    def test_fail_closed_reports_none(self):
        out = self._run(None)
        self.assertFalse(out["execute"])
        self.assertIn("served_model_id", out)
        self.assertIsNone(out["served_model_id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

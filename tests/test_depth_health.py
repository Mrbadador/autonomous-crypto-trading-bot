#!/usr/bin/env python3
"""
tests/test_depth_health.py — FIX 3 unit tests for the depth staleness probe.

Tests depth_recorder.read_depth_health() in isolation (a pure, read-only probe).
Intentionally does NOT import start_system, whose import-time atexit hook would
checkpoint the live databases.
"""

import os
import sys
import time
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import depth_recorder  # noqa: E402


def _make_partition(dir_path, date_str, timestamps_ms):
    """Builds a daily depth partition with the real books15 schema and given snapshot ts."""
    db_path = os.path.join(dir_path, f"depth_{date_str}.db")
    depth_recorder.init_depth_db(db_path)
    rows = []
    for ts in timestamps_ms:
        for lvl in range(15):
            rows.append((ts, "BTCUSDT", "BID", lvl, 100.0 + lvl, 1.0, 1))
        for lvl in range(15):
            rows.append((ts, "BTCUSDT", "ASK", lvl, 101.0 + lvl, 1.0, 1))
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO books15_snapshots (timestamp_ms, symbol, side, level, price, size, seq) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    return db_path


class TestDepthHealth(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="depth_health_")
        self.now = time.time()
        self.date_str = datetime.fromtimestamp(self.now, tz=timezone.utc).strftime("%Y-%m-%d")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fresh(self):
        _make_partition(self.tmp, self.date_str, [int((self.now - 60) * 1000)])
        res = depth_recorder.read_depth_health(now_ts=self.now, depth_dir=self.tmp, date_str=self.date_str)
        self.assertEqual(res["status"], "FRESH")
        self.assertLessEqual(res["age_sec"], 900.0)
        self.assertEqual(res["snapshots"], 1)

    def test_stale_warn(self):
        # 20 minutes old: >15min but <=60min
        _make_partition(self.tmp, self.date_str, [int((self.now - 20 * 60) * 1000)])
        res = depth_recorder.read_depth_health(now_ts=self.now, depth_dir=self.tmp, date_str=self.date_str)
        self.assertEqual(res["status"], "STALE_WARN")

    def test_stale_error(self):
        # 90 minutes old: >60min
        _make_partition(self.tmp, self.date_str, [int((self.now - 90 * 60) * 1000)])
        res = depth_recorder.read_depth_health(now_ts=self.now, depth_dir=self.tmp, date_str=self.date_str)
        self.assertEqual(res["status"], "STALE_ERROR")

    def test_boundary_exactly_15min_is_fresh(self):
        _make_partition(self.tmp, self.date_str, [int((self.now - 899) * 1000)])
        res = depth_recorder.read_depth_health(now_ts=self.now, depth_dir=self.tmp, date_str=self.date_str)
        self.assertEqual(res["status"], "FRESH")

    def test_empty_partition(self):
        _make_partition(self.tmp, self.date_str, [])  # table created, no rows
        res = depth_recorder.read_depth_health(now_ts=self.now, depth_dir=self.tmp, date_str=self.date_str)
        self.assertEqual(res["status"], "EMPTY")

    def test_missing_partition(self):
        res = depth_recorder.read_depth_health(now_ts=self.now, depth_dir=self.tmp, date_str="1970-01-01")
        self.assertEqual(res["status"], "MISSING")

    def test_probe_is_read_only(self):
        db_path = _make_partition(self.tmp, self.date_str, [int((self.now - 30) * 1000)])
        before = os.path.getmtime(db_path)
        depth_recorder.read_depth_health(now_ts=self.now, depth_dir=self.tmp, date_str=self.date_str)
        after = os.path.getmtime(db_path)
        self.assertEqual(before, after, "read_depth_health must not modify the partition file")

    def test_snapshot_counting(self):
        _make_partition(self.tmp, self.date_str, [int((self.now - 30) * 1000), int((self.now - 20) * 1000)])
        res = depth_recorder.read_depth_health(now_ts=self.now, depth_dir=self.tmp, date_str=self.date_str)
        self.assertEqual(res["snapshots"], 2)
        self.assertEqual(res["rows"], 60)  # 2 snapshots x 30 levels


if __name__ == "__main__":
    unittest.main(verbosity=2)

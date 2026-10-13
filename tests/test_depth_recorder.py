"""
tests/test_depth_recorder.py — Verification of D1 Depth Recorder Bootstrap and Schema
"""

import os
import unittest
import sqlite3
from depth_recorder import DepthRecorder, get_daily_db_path, init_depth_db


class TestDepthRecorder(unittest.TestCase):
    def test_rest_bootstrap_and_schema(self):
        recorder = DepthRecorder(symbols=["BTCUSDT"])
        # 1. Fetch REST snapshot
        rows = recorder.fetch_rest_bootstrap_snapshot("BTCUSDT")
        self.assertGreaterEqual(len(rows), 20, "Should fetch at least 20 level rows (bids + asks)")

        # Verify tuple structure
        sample = rows[0]
        self.assertEqual(len(sample), 7)
        ts_ms, sym, side, lvl, price, size, seq = sample
        self.assertIsInstance(ts_ms, int)
        self.assertEqual(sym, "BTCUSDT")
        self.assertIn(side, ["BID", "ASK"])
        self.assertGreater(price, 0.0)
        self.assertGreater(size, 0.0)

        # 2. Flush to test database
        recorder.row_buffer.extend(rows)
        recorder.flush_buffer_sync()
        self.assertGreater(recorder.total_rows_written, 0)

        # 3. Query table in daily DB
        db_path = get_daily_db_path(ts_ms)
        self.assertTrue(os.path.exists(db_path))
        with sqlite3.connect(db_path) as conn:
            cnt = conn.execute("SELECT COUNT(*) FROM books15_snapshots WHERE symbol='BTCUSDT'").fetchone()[0]
            self.assertGreaterEqual(cnt, len(rows))


if __name__ == "__main__":
    unittest.main()

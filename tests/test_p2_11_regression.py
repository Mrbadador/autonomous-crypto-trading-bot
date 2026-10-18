"""P2-11 regression: sentinel headline veto context, and mixed-format timestamp readers."""
import os
import sqlite3
import sys
import tempfile
import unittest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class TestSentinelHeadlineVeto(unittest.TestCase):
    def setUp(self):
        from sentinel_agent import SentimentSentinel
        self.s = SentimentSentinel()

    def test_individual_victim_scam_story_is_warning_not_veto(self):
        h = ["Scammer gets scammed in Ledger spy-chip exploit, loses almost $1 million in bitcoin"]
        verdict, _ = self.s._audit_ticker_headlines("BTC", h)
        self.assertEqual(verdict, "WARNING")

    def test_real_protocol_exploit_still_vetoes(self):
        h = ["Solana network exploited, validators halt as attacker drains funds"]
        verdict, _ = self.s._audit_ticker_headlines("SOL", h)
        self.assertEqual(verdict, "VETO")

    def test_exchange_hack_still_vetoes(self):
        h = ["Binance hacked: withdrawals suspended after $400M outflow"]
        verdict, _ = self.s._audit_ticker_headlines("BNB", h)
        self.assertEqual(verdict, "VETO")


class TestMixedTimestampReaders(unittest.TestCase):
    def test_normalized_cutoff_counts_iso_and_space_rows(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db")
            c = sqlite3.connect(db)
            c.execute("CREATE TABLE trades (timestamp TEXT, pnl REAL, status TEXT)")
            c.executemany("INSERT INTO trades VALUES (?,?,?)", [
                ("2026-10-17T12:00:00.123+00:00", 1.0, "closed"),
                ("2026-10-17 13:00:00", 2.0, "closed"),
                ("2026-10-10 13:00:00", 9.0, "closed"),
            ])
            cutoff = "2026-10-17 00:00:00"
            rows = c.execute(
                "SELECT pnl FROM trades WHERE (status = 'closed' OR status = 'CLOSED') "
                "AND replace(substr(timestamp, 1, 19), 'T', ' ') >= ?", (cutoff,)).fetchall()
            self.assertEqual(sorted(r[0] for r in rows), [1.0, 2.0])
            c.close()


if __name__ == "__main__":
    unittest.main()

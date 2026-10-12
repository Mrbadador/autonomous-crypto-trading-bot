import unittest
import asyncio
from unittest.mock import patch
from microstructure_guard import MicrostructureGuard, MicrostructureStateCache

class TestVPINGate(unittest.TestCase):
    def test_vpin_gate_blocks_toxic_flow(self):
        guard = MicrostructureGuard()
        guard.active_veto = True
        
        # Mock cached state to simulate toxic VPIN = 0.85
        with patch.object(MicrostructureStateCache, "get_state", return_value={"vpin": 0.85, "cancel_counts": [0]*10, "add_counts": [0]*10}):
            passed, reason = asyncio.run(guard.evaluate_pre_trade_veto({"symbol": "BTCUSDT", "side": "long"}))
            self.assertFalse(passed)
            self.assertIn("VPIN Toxicity Veto", reason)

    def test_vpin_gate_allows_benign_flow(self):
        guard = MicrostructureGuard()
        guard.active_veto = True
        
        # Mock cached state with benign VPIN = 0.35 and balanced order book
        with patch.object(MicrostructureStateCache, "get_state", return_value={"vpin": 0.35, "cancel_counts": [0]*10, "add_counts": [0]*10}), \
             patch("sqlite3.connect") as mock_conn:
            mock_conn.return_value.__enter__.return_value.execute.return_value.fetchone.return_value = (0.0, 0, 0.0, 0.0, 1.0)
            passed, reason = asyncio.run(guard.evaluate_pre_trade_veto({"symbol": "BTCUSDT", "side": "long"}))
            self.assertTrue(passed)

if __name__ == "__main__":
    unittest.main()

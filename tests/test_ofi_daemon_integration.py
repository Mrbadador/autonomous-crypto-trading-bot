import unittest
import numpy as np
from ofi_daemon import SymbolOFIState

class TestOFIDaemonIntegration(unittest.TestCase):
    def test_symbol_ofi_state_metrics_contain_mlofi_and_microprice(self):
        state = SymbolOFIState(symbol="BTCUSDT", window_seconds=60, levels=10)
        
        # Ingest initial snapshot
        bids = [[60000.0, 1.5], [59990.0, 2.0], [59980.0, 3.0]]
        asks = [[60010.0, 1.0], [60020.0, 2.5], [60030.0, 3.5]]
        state.process_depth(bids, asks)
        
        # Ingest second tick (bid pressure)
        bids2 = [[60000.0, 3.5], [59990.0, 2.0], [59980.0, 3.0]]
        asks2 = [[60010.0, 0.8], [60020.0, 2.5], [60030.0, 3.5]]
        state.process_depth(bids2, asks2)
        
        metrics = state.get_metrics()
        self.assertIn("mlofi", metrics)
        self.assertIn("micro_price", metrics)
        self.assertIn("micro_price_bps", metrics)
        self.assertIn("vpin", metrics)
        self.assertIn("hawkes_spoof_index", metrics)
        self.assertIn("hawkes_rho", metrics)
        self.assertIn("hawkes_veto", metrics)
        
        self.assertGreater(metrics["micro_price"], 60000.0)
        self.assertGreater(metrics["mlofi"], 0.0) # Bid pressure should be positive
        self.assertLess(metrics["hawkes_rho"], 1.0) # Subcritical stationary flow
        self.assertFalse(metrics["hawkes_veto"]) # Normal flow should not veto

if __name__ == "__main__":
    unittest.main()

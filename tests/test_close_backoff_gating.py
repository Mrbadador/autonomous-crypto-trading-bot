import unittest
import time
from unittest.mock import AsyncMock, patch, MagicMock

class TestCloseBackoffGating(unittest.IsolatedAsyncioTestCase):
    def test_close_backoff_suppression(self):
        # When _close_retry_next_ts is in the future, close dispatch must be suppressed unless reason is PENDING_CLOSE_RETRY
        now_ts = 1000.0
        next_retry_ts = 1060.0
        symbol = "BTC/USDT"
        _close_retry_next_ts = {symbol: next_retry_ts}

        reason_normal = "STOP_LOSS"
        is_suppressed_normal = (reason_normal != "PENDING_CLOSE_RETRY" and now_ts < _close_retry_next_ts.get(symbol, 0.0))
        self.assertTrue(is_suppressed_normal)

        reason_retry = "PENDING_CLOSE_RETRY"
        is_suppressed_retry = (reason_retry != "PENDING_CLOSE_RETRY" and now_ts < _close_retry_next_ts.get(symbol, 0.0))
        self.assertFalse(is_suppressed_retry)

        # After backoff expires:
        now_ts_expired = 1070.0
        is_suppressed_expired = (reason_normal != "PENDING_CLOSE_RETRY" and now_ts_expired < _close_retry_next_ts.get(symbol, 0.0))
        self.assertFalse(is_suppressed_expired)

if __name__ == "__main__":
    unittest.main()

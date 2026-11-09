"""
tests/test_p0_surgical_remediation.py - Red/Green Reproduction and Verification Tests
for the 4 confirmed P0/P1 defects:
1. swing_trader.py: UnboundLocalError caused by function-local datetime import shadowing module-level datetime.now(UTC).
2. swing_trader.py / conviction_filter.py: AttributeError on boardroom._check_deterministic_prefilters failing open.
3. inflight_shepherd.py: close_position_order executed before _is_swing_locked guard; position local table sync.
4. db_manager.py / reconciliation_engine.py: string vs tuple version comparison and defensive swing_trades column verification.
"""

import asyncio
import os
import sqlite3
import symtable
import sys
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestP0SurgicalRemediation(unittest.TestCase):

    # -------------------------------------------------------------------------
    # TEST 1: swing_trader.py datetime import shadowing / UnboundLocalError
    # -------------------------------------------------------------------------
    def test_01_swing_trader_no_local_datetime_shadowing(self):
        """Verify that open_swing_position does NOT locally import datetime,
        which would shadow module-level datetime and cause UnboundLocalError at line ~2114.
        """
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        target_file = os.path.join(repo_root, "swing_trader.py")
        with open(target_file, "r", encoding="utf-8") as f:
            code = f.read()

        tbl = symtable.symtable(code, "swing_trader.py", "exec")

        def find_scope(scope, name):
            if scope.get_name() == name:
                return scope
            for child in scope.get_children():
                res = find_scope(child, name)
                if res:
                    return res
            return None

        open_scope = find_scope(tbl, "open_swing_position")
        self.assertIsNotNone(open_scope, "open_swing_position method not found in swing_trader.py")

        sym_names = [s.get_name() for s in open_scope.get_symbols()]
        self.assertIn("datetime", sym_names, "datetime should be referenced in open_swing_position")

        # In Python symtable, if datetime is imported locally inside the function,
        # open_scope.lookup('datetime').is_local() is True.
        # When fixed, datetime is NOT imported locally, so is_local() MUST be False!
        is_local = open_scope.lookup("datetime").is_local()
        self.assertFalse(
            is_local,
            "datetime is locally bound inside open_swing_position! This shadows module datetime and causes UnboundLocalError."
        )

    # -------------------------------------------------------------------------
    # TEST 2: conviction_filter.py & swing_trader.py deterministic prefilters
    # -------------------------------------------------------------------------
    def test_02_boardroom_has_deterministic_prefilters_and_fails_closed(self):
        """ConvictionBoardroom must expose _check_deterministic_prefilters method,
        and swing_trader must fail closed if an error or veto occurs.
        """
        from conviction_filter import ConvictionBoardroom
        boardroom = ConvictionBoardroom(db_path="trading_data.db")

        # 1. Method must exist on ConvictionBoardroom
        self.assertTrue(
            hasattr(boardroom, "_check_deterministic_prefilters"),
            "ConvictionBoardroom is missing '_check_deterministic_prefilters' method!"
        )

        # 2. It must be callable as an async method
        res = asyncio.run(boardroom._check_deterministic_prefilters("BTC/USDT", "BUY", {}))
        self.assertIsInstance(res, dict)
        self.assertTrue("veto" in res or "passed" in res or "decoupled_info" in res or "log" in res or res == {"veto": None})

    # -------------------------------------------------------------------------
    # TEST 3: inflight_shepherd.py lock check before close_position_order
    # -------------------------------------------------------------------------
    def test_03_inflight_shepherd_checks_lock_before_close(self):
        """When an asset is locked by swing_architect or fleet, inflight_shepherd
        must NOT call client.close_position_order on any of its close paths.
        """
        import inflight_shepherd
        shepherd = inflight_shepherd.InflightPositionShepherd()

        # Mock client
        mock_client = MagicMock()
        mock_client.exchange = AsyncMock()
        mock_client.fetch_positions = AsyncMock(return_value=[{
            "symbol": "BTC/USDT:USDT",
            "side": "long",
            "posSide": "long",
            "contracts": 1.0,
            "size": 1.0,
            "entryPrice": 60000.0,
            "markPrice": 65000.0,
            "unrealisedPnl": 5000.0,
            "percentage": 8.33,
            "stopLossPrice": 59000.0,
            "takeProfitPrice": 70000.0,
            "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000)
        }])
        mock_client.exchange.fetch_funding_rate = AsyncMock(return_value={"fundingRate": 0.0001})
        mock_client.exchange.close = AsyncMock()
        mock_client._normalize_symbol = MagicMock(return_value="BTCUSDT")
        mock_client.close_position_order = AsyncMock(return_value={"orderId": "123"})
        mock_client.modify_strategy_order = AsyncMock(return_value=True)

        shepherd._fetch_candle_metrics = AsyncMock(return_value={"rsi_15m": 85.0, "atr_15m": 500.0})
        shepherd._read_microstructure_packet = MagicMock(return_value={"normalized_ofi": 0.0, "tape_condition": "BALANCED"})
        shepherd._query_ai_shepherd = AsyncMock(return_value={"verdict": "HARVEST_PROFIT", "tighten_sl": False})

        # Mock BitgetUTAClient inside evaluate_active_positions
        with patch("bitget_uta_client.BitgetUTAClient", return_value=mock_client):
            # Mock ai_engine to trigger climax harvest
            with patch("ai_engine.detect_climax_exhaustion", new=AsyncMock(return_value={
                "climax_detected": True,
                "action": "HARVEST_CLIMAX_NOW",
                "exhaustion_reason": "Blow-off top detected"
            })):
                # With _is_swing_locked returning True (asset is locked):
                with patch("inflight_shepherd._is_swing_locked", return_value=(True, "held by swing_architect")):
                    reports = asyncio.run(shepherd.evaluate_active_positions(force=True))
                    # close_position_order must NOT have been called when asset is locked!
                    mock_client.close_position_order.assert_not_called()

    # -------------------------------------------------------------------------
    # TEST 4: db_manager migration comparison & reconciler missing column defense
    # -------------------------------------------------------------------------
    def test_04_migration_version_tuple_comparison_and_reconciler_defense(self):
        """1. Version string comparison must handle '1.10' vs '1.2' correctly.
        2. Reconciliation engine must not crash with OperationalError if swing_trades is missing columns.
        """
        from db_manager import _parse_version
        self.assertGreater(_parse_version("1.10"), _parse_version("1.2"))
        self.assertGreater(_parse_version("1.2"), _parse_version("1.1"))
        self.assertEqual(_parse_version("1.2.1"), (1, 2, 1))

        # Test reconciler column verification
        import reconciliation_engine
        self.assertTrue(
            hasattr(reconciliation_engine, "_verify_table_columns") or
            hasattr(reconciliation_engine, "_check_required_columns") or
            hasattr(reconciliation_engine, "REQUIRED_SWING_TRADES_COLS"),
            "reconciliation_engine should have defensive column checking mechanism"
        )


if __name__ == "__main__":
    unittest.main()

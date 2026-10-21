"""
Issue #4 — strategy-order cross-mutation on Bitget UTA.

modify_strategy_order() pushed every supplied field (stopLoss AND takeProfit) to
EVERY unfilled strategy order for the symbol. The bot registers the two legs as
separate plans (planType "pos_loss" at bitget_uta_client.py:338 and "pos_profit"
at :420), so a stop ratchet also rewrote the take-profit plan and vice versa, and
a leg that had already gone resting-or-gone was reported with a soft debug line.

Contract under test:
  * a stop ratchet touches ONLY the pos_loss plan, and carries triggerPrice (the
    field the exchange actually fires on) alongside stopLoss, exactly like placement;
  * a take-profit move touches ONLY the pos_profit plan;
  * a leg with no resting plan is re-registered from the live position size;
  * a leg that cannot be re-registered safely logs CRITICAL stop-coverage and
    returns False instead of pretending the shield is live.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

import tests.conftest  # noqa: F401 — enforce D-11 quarantine

from bitget_uta_client import BitgetUTAClient


def _loss_order(order_id="LOSS-1", pos_side="long", size="5.0", price="98.8500"):
    return {"orderId": order_id, "symbol": "NEARUSDT", "planType": "pos_loss",
            "posSide": pos_side, "side": "sell", "size": size,
            "triggerPrice": price, "executePrice": price, "triggerType": "mark_price"}


def _profit_order(order_id="PROFIT-1", pos_side="long", size="5.0", price="101.3500"):
    return {"orderId": order_id, "symbol": "NEARUSDT", "planType": "pos_profit",
            "posSide": pos_side, "side": "sell", "size": size,
            "triggerPrice": price, "executePrice": price, "triggerType": "mark_price"}


def _client(resting_orders, modify_code="00000", live_total=5.0, place_code="00000"):
    client = BitgetUTAClient.__new__(BitgetUTAClient)
    client._markets_loaded = False
    client._time_synced = True
    client.exchange = MagicMock()
    client.ensure_time_sync = AsyncMock()
    client.exchange.private_uta_get_v3_trade_unfilled_strategy_orders = AsyncMock(
        return_value={"code": "00000", "data": list(resting_orders)})
    client.exchange.private_uta_post_v3_trade_modify_strategy_order = AsyncMock(
        return_value={"code": modify_code, "msg": "mock"})
    client.exchange.private_uta_post_v3_trade_place_strategy_order = AsyncMock(
        return_value={"code": place_code, "data": {"orderId": "REBUILT-1"}})
    client.exchange.private_uta_post_v3_trade_cancel_order = AsyncMock(
        return_value={"code": "00000", "data": {"orderId": "CANCELLED-1"}})
    client.fetch_positions = AsyncMock(
        return_value=[{"posSide": "long", "total": live_total, "contracts": live_total}])
    return client


def _modify_calls(client):
    return [c.args[0] for c in
            client.exchange.private_uta_post_v3_trade_modify_strategy_order.await_args_list]


def _place_calls(client):
    return [c.args[0] for c in
            client.exchange.private_uta_post_v3_trade_place_strategy_order.await_args_list]


def _cancel_calls(client):
    return [c.args[0] for c in
            client.exchange.private_uta_post_v3_trade_cancel_order.await_args_list]


class TestStrategyOrderPlanIsolation(unittest.TestCase):

    def test_01_sl_only_never_touches_the_profit_plan(self):
        client = _client([_loss_order(), _profit_order()])
        asyncio.run(client.modify_strategy_order("NEAR/USDT", sl=100.0))
        calls = _modify_calls(client)
        self.assertEqual(len(calls), 1, "a stop ratchet must modify exactly one plan")
        self.assertEqual(calls[0]["orderId"], "LOSS-1")
        self.assertIn("stopLoss", calls[0])
        self.assertNotIn("takeProfit", calls[0])

    def test_02_tp_only_never_touches_the_loss_plan(self):
        client = _client([_loss_order(), _profit_order()])
        asyncio.run(client.modify_strategy_order("NEAR/USDT", tp=103.0))
        calls = _modify_calls(client)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["orderId"], "PROFIT-1")
        self.assertIn("takeProfit", calls[0])
        self.assertNotIn("stopLoss", calls[0])

    def test_03_both_legs_route_to_their_own_plans(self):
        client = _client([_loss_order(), _profit_order()])
        res = asyncio.run(client.modify_strategy_order("NEAR/USDT", sl=100.0, tp=103.0))
        self.assertTrue(res)
        by_id = {c["orderId"]: c for c in _modify_calls(client)}
        self.assertEqual(set(by_id), {"LOSS-1", "PROFIT-1"})
        self.assertIn("stopLoss", by_id["LOSS-1"])
        self.assertNotIn("takeProfit", by_id["LOSS-1"])
        self.assertIn("takeProfit", by_id["PROFIT-1"])
        self.assertNotIn("stopLoss", by_id["PROFIT-1"])

    def test_04_stop_modify_carries_trigger_price(self):
        """The exchange fires on triggerPrice; writing only stopLoss leaves the OLD
        trigger armed, which silently defeats every ratchet."""
        client = _client([_loss_order(), _profit_order()])
        asyncio.run(client.modify_strategy_order("NEAR/USDT", sl=100.0))
        call = _modify_calls(client)[0]
        self.assertEqual(call["triggerPrice"], "100.0000")
        self.assertEqual(call["stopLoss"], "100.0000")

    def test_05_missing_loss_leg_is_rebuilt_from_the_live_position(self):
        client = _client([_profit_order()])
        with self.assertLogs("bitget_uta", level="WARNING") as cm:
            res = asyncio.run(client.modify_strategy_order("NEAR/USDT", sl=100.0))
        self.assertTrue(res)
        placed = _place_calls(client)
        self.assertEqual(len(placed), 1)
        self.assertEqual(placed[0]["planType"], "pos_loss")
        self.assertEqual(placed[0]["posSide"], "long")
        self.assertEqual(placed[0]["side"], "sell")
        self.assertEqual(placed[0]["triggerPrice"], "100.0000")
        self.assertTrue(any("REBUILD" in m for m in cm.output), cm.output)

    def test_06_rejected_modify_cancels_the_stale_plan_then_rebuilds(self):
        client = _client([_loss_order(), _profit_order()], modify_code="25416")
        res = asyncio.run(client.modify_strategy_order("NEAR/USDT", sl=100.0))
        self.assertTrue(res)
        self.assertEqual([c["orderId"] for c in _cancel_calls(client)], ["LOSS-1"])
        placed = _place_calls(client)
        self.assertEqual(len(placed), 1)
        self.assertEqual(placed[0]["planType"], "pos_loss")

    def test_07_no_plan_and_no_live_position_fails_loud_not_silent(self):
        client = _client([])
        client.fetch_positions = AsyncMock(return_value=[])
        with self.assertLogs("bitget_uta", level="CRITICAL") as cm:
            res = asyncio.run(client.modify_strategy_order("NEAR/USDT", sl=100.0))
        self.assertFalse(res)
        self.assertTrue(any("STOP-COVERAGE" in m for m in cm.output), cm.output)
        self.assertEqual(_place_calls(client), [])

    def test_08_rebuild_size_follows_a_partial_harvest(self):
        """After the grid partial take-profit the live position is halved; the rebuilt
        stop must cover the REMAINING size, not the size recorded at entry."""
        client = _client([_profit_order()], live_total=5.0)
        asyncio.run(client.modify_strategy_order("NEAR/USDT", sl=100.0))
        self.assertEqual(_place_calls(client)[0]["size"], "5")

    def test_09_already_up_to_date_counts_as_success(self):
        client = _client([_loss_order(), _profit_order()], modify_code="25600")
        self.assertTrue(asyncio.run(client.modify_strategy_order("NEAR/USDT", sl=100.0)))
        self.assertEqual(_place_calls(client), [])
        self.assertEqual(_cancel_calls(client), [])

    def test_10_rebuild_failure_is_reported_as_coverage_loss(self):
        client = _client([_profit_order()], place_code="40001")
        with self.assertLogs("bitget_uta", level="CRITICAL") as cm:
            res = asyncio.run(client.modify_strategy_order("NEAR/USDT", sl=100.0))
        self.assertFalse(res)
        self.assertTrue(any("STOP-COVERAGE" in m for m in cm.output), cm.output)

    def test_11_trailing_plans_are_never_rewritten_as_flat_stops(self):
        """A track_plan order is its own mechanism and still protects the position, so it
        must not be overwritten with a flat stop, and must not trigger a duplicate rebuild."""
        trail = {"orderId": "TRAIL-1", "symbol": "NEARUSDT", "planType": "track_plan",
                 "posSide": "long", "side": "sell", "size": "5.0", "triggerPrice": "99.0000"}
        client = _client([trail, _profit_order()])
        with self.assertLogs("bitget_uta", level="WARNING") as cm:
            res = asyncio.run(client.modify_strategy_order("NEAR/USDT", sl=100.0))
        self.assertFalse(res, "ratchet did not land, callers must not assume it did")
        self.assertNotIn("TRAIL-1", [c["orderId"] for c in _modify_calls(client)])
        self.assertEqual(_place_calls(client), [], "trailing plan already covers the stop leg")
        self.assertTrue(any("TRAILING" in m for m in cm.output), cm.output)


if __name__ == "__main__":
    unittest.main()

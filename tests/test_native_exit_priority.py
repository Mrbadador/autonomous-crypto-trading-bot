"""
Issue #2 — the software exit races the exchange-side plan.

manage_active_positions() closes on the ticker's last-trade print the moment it crosses
stop_loss / take_profit, while Bitget already rests a mark-price triggered pos_loss and
pos_profit plan on the same position. Both paths market-close, so the software leg fires
first on wick prints and pays for an exit the exchange plan was about to fill anyway.

Contract under test: the on-exchange plan is the PRIMARY exit, and the software close
survives only as a verified fallback —
  * shallow breach + resting plan  -> defer, protection is live;
  * breach deeper than the confirm buffer, no resting plan, or an unreadable book
    -> close immediately (protection is never traded for optimisation);
  * plan still resting after the grace window -> close and label it a fallback so the
    ledger can measure how often the exchange leg failed to trigger.
"""

import asyncio
import time
import unittest
from datetime import timedelta
from unittest.mock import AsyncMock, patch

import ai_engine
from swing_trader import SwingArchitectTrader
from tests.grid_exit_harness import SwingExitHarnessCase, aged, grid_pos, wire_client


def _hold_ai():
    return (patch.object(ai_engine, "check_position_health",
                         AsyncMock(return_value={"verdict": "HOLD", "suggested_sl": None,
                                                "diagnosis": "", "health_grade": "OK"})),
            patch.object(ai_engine, "evaluate_runner_vs_harvest", AsyncMock(return_value=None)))


class TestNativeExitPriority(SwingExitHarnessCase):

    def _run(self, price, plan_families=("loss", "profit"), coverage_error=None,
             sl=98.85, tp=101.35, armed_age=None, side="long", entry=100.0):
        async def body():
            trader = SwingArchitectTrader()
            wire_client(trader, ticker_price=price, ohlcv_close=price,
                        plan_families=plan_families, coverage_error=coverage_error)
            pos = grid_pos(entry=entry, sl=sl, tp=tp, size=10.0, peak=price,
                           engine="SWING_4H", side=side, harvested=True,
                           opened_at=aged(90), risk_usd=11.5)
            trader.active_positions = {"NEAR/USDT": pos}
            trader._last_ai_health_check_ts = {}
            trader.close_swing_position = AsyncMock()
            if armed_age is not None:
                trader._native_exit_armed_ts = {"NEAR/USDT:loss": time.time() - armed_age}
            p1, p2 = _hold_ai()
            with p1, p2:
                await trader.manage_active_positions()
            return trader
        return asyncio.run(body())

    # ------------------------------- stop leg -------------------------------- #
    def test_01_shallow_breach_with_resting_plan_defers(self):
        """0.05% past the stop with a live pos_loss plan: let the exchange trigger fire."""
        trader = self._run(price=98.80, plan_families=("loss", "profit"))
        trader.close_swing_position.assert_not_called()

    def test_02_deep_breach_still_closes_in_software(self):
        """0.50% past the stop means the mark-price plan is not doing its job — exit now."""
        trader = self._run(price=98.35, plan_families=("loss", "profit"))
        trader.close_swing_position.assert_called_once()
        self.assertEqual(trader.close_swing_position.call_args.kwargs["reason"], "STOP_LOSS")

    def test_03_grace_expiry_closes_and_is_labelled_fallback(self):
        trader = self._run(price=98.80, plan_families=("loss", "profit"), armed_age=400)
        trader.close_swing_position.assert_called_once()
        self.assertEqual(trader.close_swing_position.call_args.kwargs["reason"], "STOP_LOSS_FALLBACK")

    def test_04_missing_stop_plan_never_waits(self):
        """Coverage is the whole point: with no resting stop plan the software stop is
        the only shield, so it fires on the first shallow breach."""
        trader = self._run(price=98.80, plan_families=("profit",))
        trader.close_swing_position.assert_called_once()
        self.assertEqual(trader.close_swing_position.call_args.kwargs["reason"], "STOP_LOSS")

    def test_05_unreadable_plan_book_fails_safe(self):
        trader = self._run(price=98.80, coverage_error=RuntimeError("53001 service busy"))
        trader.close_swing_position.assert_called_once()
        self.assertEqual(trader.close_swing_position.call_args.kwargs["reason"], "STOP_LOSS")

    def test_06_short_side_mirrors_the_same_rule(self):
        trader = self._run(price=101.20, sl=101.15, tp=98.0, side="short",
                           plan_families=("loss", "profit"))
        trader.close_swing_position.assert_not_called()

    def test_07_trailing_plan_counts_as_stop_coverage(self):
        trader = self._run(price=98.80, plan_families=("trailing", "profit"))
        trader.close_swing_position.assert_not_called()

    # ------------------------------ target leg ------------------------------- #
    def test_08_target_overshoot_with_resting_plan_defers(self):
        """Let the pos_profit plan fill the maker target instead of market-dumping it."""
        trader = self._run(price=101.40, sl=98.85, tp=101.35,
                           plan_families=("loss", "profit"))
        trader.close_swing_position.assert_not_called()

    def test_09_target_crossed_with_no_profit_plan_closes(self):
        trader = self._run(price=101.40, sl=98.85, tp=101.35, plan_families=("loss",))
        trader.close_swing_position.assert_called_once()
        self.assertEqual(trader.close_swing_position.call_args.kwargs["reason"], "TAKE_PROFIT")

    def test_10_defer_state_is_cleared_once_the_position_exits(self):
        trader = self._run(price=98.80, plan_families=("loss", "profit"))
        self.assertIn("NEAR/USDT:loss", trader._native_exit_armed_ts)
        trader = self._run(price=98.35, plan_families=("loss", "profit"))
        self.assertEqual(trader._native_exit_armed_ts, {})

    def test_11_no_breach_means_no_coverage_call_and_no_close(self):
        trader = self._run(price=100.20, sl=98.85, tp=101.35, plan_families=("loss", "profit"))
        trader.close_swing_position.assert_not_called()
        trader.uta_client.get_resting_plan_families.assert_not_called()


if __name__ == "__main__":
    unittest.main()

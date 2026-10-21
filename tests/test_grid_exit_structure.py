"""
Item #3 — grid exit structure ("breakeven at +0.75% caps winners").

Covers the four production behaviours requested by the audit:
  #3a  R-multiple must be measured from the stop at ENTRY, not from the live
       (already-ratcheted) stop, otherwise every R-based gate fires early.
  #3b  A grid winner reaching the micro-breakeven trigger closes ~50% at market
       and moves the stop to TRUE entry breakeven, leaving the rest to ride to TP.
  #3c  The AI stop-tightener may only move a stop in the risk-REDUCING direction;
       a suggestion that would sit above breakeven (in price) is vetoed.
  #3d  AI_EARLY_TP does not fire for grid positions (the maker TP limit is still
       35bps away and the early harvest pays 8bps to give it up).
"""

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import ai_engine
from swing_trader import SwingArchitectTrader
from tests.grid_exit_harness import (SwingExitHarnessCase, _orig_sqlite_connect,
                                     _LONG_NOTES, grid_pos as _grid_pos,
                                     wire_client as _manage)


class TestGridExitStructure(SwingExitHarnessCase):

    # ------------------------------------------------------------------ #
    # #3a  R measured from the initial stop
    # ------------------------------------------------------------------ #
    def test_01_initial_risk_unit_survives_a_ratcheted_stop(self):
        trader = SwingArchitectTrader()
        pos = _grid_pos(entry=100.0, sl=100.35, size=10.0)  # stop already above entry
        pos["initial_risk_usd"] = 11.5  # 1.15 price risk x 10 contracts at entry
        unit = trader._initial_risk_unit(pos)
        self.assertAlmostEqual(unit, 1.15, places=6)

    def test_02_initial_risk_unit_falls_back_to_stop_geometry(self):
        trader = SwingArchitectTrader()
        no_risk_usd = {"entry_price": 100.0, "size": 10.0, "stop_loss": 98.85, "initial_risk_usd": None}
        self.assertAlmostEqual(trader._initial_risk_unit(no_risk_usd), 1.15, places=6)
        no_risk_usd["stop_loss"] = 50.0  # nonsense stop, 50% away
        self.assertAlmostEqual(trader._initial_risk_unit(no_risk_usd), 2.0, places=6)
        self.assertAlmostEqual(trader._initial_risk_unit({"entry_price": 0.0}), 0.0, places=6)

    def test_03_grid_ratchet_does_not_inflate_r(self):
        """A grid winner whose stop was already ratcheted to entry+0.35% must still read 0.39R, not 1.29R."""
        async def run():
            trader = SwingArchitectTrader()
            _manage(trader, ticker_price=100.45, ohlcv_close=100.45)
            pos = _grid_pos(sl=100.35, size=10.0)
            pos["initial_risk_usd"] = 11.5  # pinned at entry: 1.15 price risk x 10 contracts
            trader.active_positions = {"NEAR/USDT": pos}
            await trader.manage_active_positions()
            self.assertAlmostEqual(trader.active_positions["NEAR/USDT"]["r_multiple"], 0.39, places=2)
        asyncio.run(run())

    def test_04_ratchet_1_be_not_fired_by_grid_ratchet(self):
        """The 2.0R breakeven ratchet must not fire on a +0.45% grid winner (0.39R)."""
        async def run():
            trader = SwingArchitectTrader()
            _manage(trader, ticker_price=100.45, ohlcv_close=100.45)
            pos = _grid_pos(sl=100.35, size=10.0)
            pos["initial_risk_usd"] = 11.5
            trader.active_positions = {"NEAR/USDT": pos}
            await trader.manage_active_positions()
            self.assertEqual(trader.active_positions["NEAR/USDT"]["stop_loss"], 100.35)
            self.assertAlmostEqual(trader.active_positions["NEAR/USDT"]["r_multiple"], 0.39, places=2)
        asyncio.run(run())

    # ------------------------------------------------------------------ #
    # #3b  partial take-profit at the micro-breakeven trigger
    # ------------------------------------------------------------------ #
    def test_05_grid_winner_partially_harvests_at_trigger(self):
        async def run():
            trader = SwingArchitectTrader()
            _manage(trader, ticker_price=100.75, ohlcv_close=100.75)
            trader.active_positions = {"NEAR/USDT": _grid_pos(sl=98.85, size=10.0)}
            await trader.manage_active_positions()
            pos = trader.active_positions["NEAR/USDT"]

            trader.uta_client.close_position_order.assert_called_once()
            kwargs = trader.uta_client.close_position_order.call_args.kwargs
            self.assertEqual(kwargs["side"], "long")
            self.assertEqual(kwargs["qty"], 5.0)

            # stop to TRUE breakeven (entry), not entry+0.35%
            self.assertAlmostEqual(pos["stop_loss"], 100.0, places=4)
            # remaining half still rides to the original maker TP
            self.assertEqual(pos["take_profit"], 101.35)
            self.assertEqual(pos["size"], 5.0)
            self.assertEqual(int(pos["grid_partial_tp_harvested"]), 1)
        asyncio.run(run())

    def test_06_partial_harvest_is_once_only(self):
        async def run():
            trader = SwingArchitectTrader()
            _manage(trader, ticker_price=100.90, ohlcv_close=100.90)
            trader.active_positions = {"NEAR/USDT": _grid_pos(sl=100.0, size=5.0, peak=100.90, harvested=True)}
            await trader.manage_active_positions()
            trader.uta_client.close_position_order.assert_not_called()
            self.assertAlmostEqual(trader.active_positions["NEAR/USDT"]["stop_loss"], 100.0, places=4)
        asyncio.run(run())

    def test_07_dust_and_sub_minimum_positions_fall_back_to_lock(self):
        """ADA/USDT is a whole-contract coin: half of 1 contract is not tradeable,
        so fall back to the old breakeven lock instead of closing dust."""
        async def run():
            trader = SwingArchitectTrader()
            _manage(trader, ticker_price=100.75, ohlcv_close=100.75)
            pos = _grid_pos(symbol="ADA/USDT", sl=98.85, size=1.0)
            trader.active_positions = {"ADA/USDT": pos}
            with self.assertLogs("swing_architect", level="INFO") as cm:
                await trader.manage_active_positions()
            trader.uta_client.close_position_order.assert_not_called()
            self.assertAlmostEqual(trader.active_positions["ADA/USDT"]["stop_loss"], 100.35, places=4)
            self.assertEqual(trader.active_positions["ADA/USDT"]["size"], 1.0)
            self.assertTrue(any("GRID-PARTIAL-TP-SKIP" in m for m in cm.output), cm.output)
            # marker is set even on the fallback lock, so the clip is never retried
            self.assertEqual(int(trader.active_positions["ADA/USDT"]["grid_partial_tp_harvested"]), 1)
        asyncio.run(run())

    def test_07b_tiny_notional_position_falls_back_to_lock(self):
        """0.0008 BTC halves to 0.0004 contracts = $0.04 notional — far below the $5 floor,
        so the position keeps the old entry+0.35% lock instead of being clipped."""
        async def run():
            trader = SwingArchitectTrader()
            _manage(trader, ticker_price=100.75, ohlcv_close=100.75)
            pos = _grid_pos(symbol="BTC/USDT", sl=98.85, size=0.0008)
            trader.active_positions = {"BTC/USDT": pos}
            await trader.manage_active_positions()
            trader.uta_client.close_position_order.assert_not_called()
            self.assertAlmostEqual(trader.active_positions["BTC/USDT"]["stop_loss"], 100.35, places=4)
        asyncio.run(run())

    def test_07c_partial_harvest_failure_falls_back_to_lock(self):
        async def run():
            trader = SwingArchitectTrader()
            _manage(trader, ticker_price=100.75, ohlcv_close=100.75)
            trader.uta_client.close_position_order = AsyncMock(side_effect=Exception("25227 no position available"))
            pos = _grid_pos(sl=98.85, size=10.0)
            trader.active_positions = {"NEAR/USDT": pos}
            with self.assertLogs("swing_architect", level="WARNING") as cm:
                await trader.manage_active_positions()
            self.assertTrue(any("GRID-PARTIAL-TP-FAIL" in m for m in cm.output), cm.output)
            self.assertAlmostEqual(trader.active_positions["NEAR/USDT"]["stop_loss"], 100.35, places=4)
            self.assertEqual(trader.active_positions["NEAR/USDT"]["size"], 10.0)
        asyncio.run(run())

    def test_08_scalp_short_keeps_its_fast_lock(self):
        """SCALP_SHORT is a deliberate fast-harvest design and must not be partially harvested."""
        async def run():
            trader = SwingArchitectTrader()
            _manage(trader, ticker_price=99.20, ohlcv_close=99.20)
            pos = _grid_pos(symbol="NEAR/USDT", sl=101.00, tp=98.80, peak=99.20,
                            side="short", engine="SCALP_SHORT")
            trader.active_positions = {"NEAR/USDT": pos}
            await trader.manage_active_positions()
            trader.uta_client.close_position_order.assert_not_called()
            self.assertAlmostEqual(trader.active_positions["NEAR/USDT"]["stop_loss"], 99.65, places=4)
        asyncio.run(run())

    # ------------------------------------------------------------------ #
    # #3c  AI stop-tightener demoted to a veto
    # ------------------------------------------------------------------ #
    def _ai_case(self, verdict, suggested_sl, price=101.20, harvested=True, sl=100.0):
        async def run():
            trader = SwingArchitectTrader()
            _manage(trader, ticker_price=price, ohlcv_close=price)
            pos = _grid_pos(sl=sl, size=5.0, peak=price, harvested=harvested,
                            opened_at=datetime.now(UTC) - timedelta(minutes=90))
            trader.active_positions = {"NEAR/USDT": pos}
            trader._last_ai_health_check_ts = {}
            health = {"verdict": verdict, "suggested_sl": suggested_sl,
                      "diagnosis": "test", "health_grade": "WARNING"}
            with patch.object(ai_engine, "check_position_health", AsyncMock(return_value=health)), \
                 patch.object(ai_engine, "evaluate_runner_vs_harvest", AsyncMock(return_value=None)):
                await trader.manage_active_positions()
            return trader
        return run()

    def test_09_ai_tighten_above_breakeven_is_vetoed(self):
        """A 100.60 suggestion on a 100.00 entry stop sits IN PRICE — veto, never move."""
        trader = asyncio.run(self._ai_case("TIGHTEN_SL", 100.60))
        trader.uta_client.modify_strategy_order.assert_not_called()
        self.assertAlmostEqual(trader.active_positions["NEAR/USDT"]["stop_loss"], 100.0, places=4)

    def test_10_ai_move_below_entry_is_still_allowed(self):
        """Risk-REDUCING AI moves (stop lowered toward the original) are honoured."""
        trader = asyncio.run(self._ai_case("TIGHTEN_SL", 99.60, sl=99.30))
        trader.uta_client.modify_strategy_order.assert_called_once()
        self.assertAlmostEqual(trader.active_positions["NEAR/USDT"]["stop_loss"], 99.60, places=4)

    def test_11_ai_close_now_demote_cannot_cross_breakeven(self):
        trader = asyncio.run(self._ai_case("CLOSE_NOW", 100.45))
        trader.uta_client.modify_strategy_order.assert_not_called()
        self.assertAlmostEqual(trader.active_positions["NEAR/USDT"]["stop_loss"], 100.0, places=4)

    # ------------------------------------------------------------------ #
    # #3d  AI_EARLY_TP disabled for grid
    # ------------------------------------------------------------------ #
    def test_12_ai_early_tp_skipped_for_grid(self):
        async def run():
            trader = SwingArchitectTrader()
            _manage(trader, ticker_price=101.20, ohlcv_close=101.20)
            trader.active_positions = {"NEAR/USDT": _grid_pos(
                sl=100.0, size=5.0, peak=101.20, harvested=True,
                opened_at=datetime.now(UTC) - timedelta(minutes=90))}
            trader._last_ai_health_check_ts = {}
            trader.close_swing_position = AsyncMock()
            health = {"verdict": "TAKE_PROFIT_EARLY", "suggested_sl": None,
                      "diagnosis": "stalling", "health_grade": "WARNING"}
            with patch.object(ai_engine, "check_position_health", AsyncMock(return_value=health)), \
                 patch.object(ai_engine, "evaluate_runner_vs_harvest", AsyncMock(return_value=None)):
                await trader.manage_active_positions()
            trader.close_swing_position.assert_not_called()
        asyncio.run(run())

    def test_13_ai_early_tp_still_fires_for_swing(self):
        async def run():
            trader = SwingArchitectTrader()
            _manage(trader, ticker_price=101.20, ohlcv_close=101.20)
            pos = _grid_pos(symbol="NEAR/USDT", sl=97.0, tp=106.0, size=5.0, peak=101.20,
                            engine="SWING_4H", lev=5.0, risk_usd=15.0, harvested=True,
                            opened_at=datetime.now(UTC) - timedelta(minutes=90))
            trader.active_positions = {"NEAR/USDT": pos}
            trader._last_ai_health_check_ts = {}
            trader.close_swing_position = AsyncMock()
            health = {"verdict": "TAKE_PROFIT_EARLY", "suggested_sl": None,
                      "diagnosis": "stalling", "health_grade": "WARNING"}
            with patch.object(ai_engine, "check_position_health", AsyncMock(return_value=health)), \
                 patch.object(ai_engine, "evaluate_runner_vs_harvest", AsyncMock(return_value=None)):
                await trader.manage_active_positions()
            trader.close_swing_position.assert_called_once()
            self.assertEqual(trader.close_swing_position.call_args.kwargs["reason"], "AI_EARLY_TP")
        asyncio.run(run())


    def test_14_partial_harvest_persists_the_reduced_size(self):
        """Restart-safety: _load_active_positions_sync re-reads the row, so the halved
        size and the once-only marker must both be written back, not only held in memory."""
        async def run():
            trader = SwingArchitectTrader()
            _manage(trader, ticker_price=100.75, ohlcv_close=100.75)
            pos = _grid_pos(sl=98.85, size=10.0)
            trader.active_positions = {"NEAR/USDT": pos}
            with _orig_sqlite_connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO swing_open_positions (symbol, side, entry_price, size, "
                    "stop_loss, take_profit, trailing_stop, peak_price, leverage, opened_at, updated_at, "
                    "status, r_multiple, initial_risk_usd, notes, engine_type, grid_partial_tp_harvested) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (pos["symbol"], "long", pos["entry_price"], pos["size"], pos["stop_loss"],
                     pos["take_profit"], pos["trailing_stop"], pos["peak_price"], pos["leverage"],
                     pos["opened_at"], pos["updated_at"], "ACTIVE", pos["r_multiple"],
                     pos["initial_risk_usd"], pos["notes"], pos["engine_type"], 0))
                conn.commit()

            await trader.manage_active_positions()

            with _orig_sqlite_connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT size, grid_partial_tp_harvested, stop_loss FROM swing_open_positions "
                    "WHERE symbol = 'NEAR/USDT'").fetchone()
            self.assertAlmostEqual(row[0], 5.0, places=6, msg="halved size was not persisted")
            self.assertEqual(row[1], 1)
            self.assertAlmostEqual(row[2], 100.0, places=4)
        asyncio.run(run())


    def test_15_ratchet_never_puts_the_stop_at_or_above_the_last_price(self):
        """A 2R ratchet computed from a degenerate risk unit can land the stop ABOVE the
        ticker price; a long stop sitting above spot fires instantly on the exchange, so
        the ratchet must be refused, not pushed."""
        async def run():
            trader = SwingArchitectTrader()
            _manage(trader, ticker_price=100.20, ohlcv_close=100.20)
            pos = _grid_pos(symbol="NEAR/USDT", entry=100.0, sl=99.90, tp=106.0, size=1.0,
                            peak=100.20, engine="SWING_4H", risk_usd=0.05, harvested=True,
                            opened_at=datetime.now(UTC) - timedelta(minutes=90))
            trader.active_positions = {"NEAR/USDT": pos}
            trader._last_ai_health_check_ts = {}
            with patch.object(ai_engine, "check_position_health",
                              AsyncMock(return_value={"verdict": "HOLD", "suggested_sl": None,
                                                      "diagnosis": "", "health_grade": "OK"})), \
                 patch.object(ai_engine, "evaluate_runner_vs_harvest", AsyncMock(return_value=None)):
                await trader.manage_active_positions()
            self.assertAlmostEqual(trader.active_positions["NEAR/USDT"]["stop_loss"], 99.90, places=4)
            trader.uta_client.modify_strategy_order.assert_not_called()
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()

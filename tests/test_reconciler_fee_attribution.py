"""
Issue #6 — the reconciler misprices fees and mislabels exits.

audit_live_exchange_positions() reconciles a close the exchange performed by itself. Two
of its ledger fields were systematically wrong:

  * every entry leg was charged the 0.06% taker rate, while grid and maker-sniper entries
    rest a post-only limit at 0.02% and record the real amount in fee_entry_usd. The fee
    drag on those trades was overstated by 2x on the entry leg, and fee_maker_usd was
    hardcoded to 0, so the maker/taker columns could not be used for fee analysis.
  * an exit that lands on the breakeven price with a negative net result (the usual case:
    the stop sits a few bps past entry and the round-trip fee eats it) was still labelled
    breakeven_hit, which flattered the loss column in every stats rollup.

Contract under test: the reconciler must mirror the attribution close_swing_position()
already uses — entry fee from fee_entry_usd (or the rate implied by execution_type, taker
when unknown), exit fee from the fills (taker rate estimated when absent), fee_paid_usd =
round trip split across maker/taker, and breakeven_hit reserved for non-negative closes.
"""

import asyncio
import os
import sqlite3
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import reconciliation_engine as rec  # noqa: E402
from tests.test_reconciler_ledger_guard import DDL  # production DDL snapshot  # noqa: E402

SWING = "TESTB/USDT"
TAKER_RATE = 0.0006


class _FakeExchange:
    def __getattr__(self, name):
        async def _any(*args, **kwargs):
            return {"data": {}}
        return _any


class _FakeUTA:
    """Reports no live position (so the local row looks closed) and the configured fills."""

    fills = []

    def __init__(self, *args, **kwargs):
        self.exchange = _FakeExchange()

    async def fetch_balance(self, *args, **kwargs):
        return 340.0

    async def fetch_positions(self, *args, **kwargs):
        return []

    async def get_recent_fills(self, *args, **kwargs):
        return list(self.fills)

    @staticmethod
    def _normalize_symbol(sym):
        return str(sym).replace("/", "").upper()

    async def cancel_order(self, *args, **kwargs):
        return {}

    async def close_position_order(self, *args, **kwargs):
        return {}

    async def close(self, *args, **kwargs):
        return None


def _close_fill(exit_p, qty, pnl):
    notional = exit_p * qty
    return {
        "symbol": "TESTBUSDT",
        "tradeSide": "close_long",
        "execQty": str(qty),
        "execPrice": str(exit_p),
        "execValue": str(notional),
        "fee": str(notional * TAKER_RATE),
        "execPnl": str(pnl),
    }


class ReconcilerFeeAttributionTest(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="recon_fees_")
        self.db_path = os.path.join(self.tmpdir, "trading_data.db")
        self._orig = {
            "DATA_DB": rec.DATA_DB,
            "LEDGER_DB": rec.LEDGER_DB,
            "MODE": getattr(config, "MODE", None),
            "PAPER_TRADE": getattr(config, "PAPER_TRADE", None),
            "EXCHANGE_ID": getattr(config, "EXCHANGE_ID", None),
            "fetch_live_price": rec.fetch_live_price,
        }
        rec.DATA_DB = self.db_path
        rec.LEDGER_DB = os.path.join(self.tmpdir, "trading_ledger.db")
        config.MODE = "live"
        config.PAPER_TRADE = False
        config.EXCHANGE_ID = "bitget"
        rec.fetch_live_price = lambda symbol: None

        self._orig_modules = {}
        for name in ("bitget_uta_client", "email_dispatcher"):
            self._orig_modules[name] = sys.modules.get(name)
            sys.modules[name] = types.ModuleType(name)
        sys.modules["bitget_uta_client"].BitgetUTAClient = _FakeUTA
        sys.modules["email_dispatcher"].send_trade_log_async = lambda *a, **k: asyncio.sleep(0)
        sys.modules["email_dispatcher"].send_trade_log = lambda *a, **k: None

    def tearDown(self):
        for name, mod in self._orig_modules.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
        for key, attr in (("DATA_DB", "DATA_DB"), ("LEDGER_DB", "LEDGER_DB"), ("fetch_live_price", "fetch_live_price")):
            setattr(rec, attr, self._orig[key])
        config.MODE = self._orig["MODE"]
        config.PAPER_TRADE = self._orig["PAPER_TRADE"]
        config.EXCHANGE_ID = self._orig["EXCHANGE_ID"]

    def _seed(self, execution_type="maker", fee_entry_usd=0.20, exit_p=None, qty=10.0, pnl=None):
        with sqlite3.connect(self.db_path) as conn:
            for ddl in DDL:
                conn.execute(ddl)
            conn.execute(
                "INSERT INTO swing_open_positions (symbol, side, entry_price, size, stop_loss, take_profit, "
                "leverage, opened_at, status, execution_type, fee_entry_usd) "
                "VALUES (?, 'long', 100.0, ?, 98.85, 101.35, 3.0, datetime('now', '-2 hours'), 'OPEN', ?, ?)",
                (SWING, qty, execution_type, fee_entry_usd),
            )
            conn.commit()
        _FakeUTA.fills = [] if exit_p is None else [_close_fill(exit_p, qty, pnl if pnl is not None else (exit_p - 100.0) * qty)]

    def _ledger_row(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM swing_trades").fetchone()
        return dict(row) if row else None

    async def _reconcile(self):
        await rec.audit_live_exchange_positions()
        row = self._ledger_row()
        self.assertIsNotNone(row, "reconciler must write a swing_trades row for the closed position")
        return row

    # --------------------------- fee attribution ----------------------------- #
    async def test_maker_entry_uses_recorded_entry_fee_and_splits_legs(self):
        self._seed(execution_type="maker", fee_entry_usd=0.20, exit_p=101.40)
        row = await self._reconcile()
        exit_fee = 101.40 * 10.0 * TAKER_RATE
        self.assertAlmostEqual(row["fee_maker_usd"], 0.20, places=6,
                               msg="the recorded maker entry fee belongs in the maker column")
        self.assertAlmostEqual(row["fee_taker_usd"], exit_fee, places=6,
                               msg="the exit fill fee belongs in the taker column")
        self.assertAlmostEqual(row["fee_paid_usd"], 0.20 + exit_fee, places=6)

    async def test_maker_entry_without_recorded_fee_is_estimated_at_maker_rate(self):
        self._seed(execution_type="maker", fee_entry_usd=None, exit_p=101.40)
        row = await self._reconcile()
        self.assertAlmostEqual(row["fee_maker_usd"], 100.0 * 10.0 * 0.0002, places=6)

    async def test_taker_entry_charges_both_legs_as_taker(self):
        self._seed(execution_type="taker", fee_entry_usd=0.60, exit_p=101.40)
        row = await self._reconcile()
        self.assertAlmostEqual(row["fee_maker_usd"], 0.0, places=6)
        self.assertAlmostEqual(row["fee_taker_usd"], 0.60 + 101.40 * 10.0 * TAKER_RATE, places=6)

    async def test_unknown_execution_type_falls_back_to_taker(self):
        self._seed(execution_type=None, fee_entry_usd=None, exit_p=101.40)
        row = await self._reconcile()
        self.assertAlmostEqual(row["fee_taker_usd"], 100.0 * 10.0 * TAKER_RATE + 101.40 * 10.0 * TAKER_RATE, places=6)

    async def test_net_pnl_nets_the_full_round_trip_fee(self):
        self._seed(execution_type="maker", fee_entry_usd=0.20, exit_p=101.40, pnl=14.0)
        row = await self._reconcile()
        exit_fee = 101.40 * 10.0 * TAKER_RATE
        self.assertAlmostEqual(row["pnl"], 14.0 - (0.20 + exit_fee), places=6)

    async def test_estimated_fee_path_is_round_trip_when_no_fills(self):
        self._seed(execution_type="maker", fee_entry_usd=None, exit_p=None)
        # No fills at all -> the UNRECONCILED invariant must win over any fabricated price.
        row = await self._reconcile()
        self.assertIsNone(row["exit_price"])
        self.assertEqual(row["status"], "UNRECONCILED")

    # ------------------------------ exit labels ------------------------------ #
    async def test_breakeven_price_with_negative_net_is_labelled_sl_hit(self):
        """Gross +0.50 against a 0.80 round-trip fee is a losing stop exit, not a scratch trade."""
        self._seed(execution_type="maker", fee_entry_usd=0.20, exit_p=100.05, pnl=0.50)
        row = await self._reconcile()
        self.assertLess(row["pnl"], 0.0)
        self.assertEqual(row["exit_reason"], "sl_hit")

    async def test_breakeven_price_with_non_negative_net_stays_breakeven_hit(self):
        self._seed(execution_type="maker", fee_entry_usd=0.20, exit_p=100.30, pnl=3.0)
        row = await self._reconcile()
        self.assertGreaterEqual(row["pnl"], 0.0)
        self.assertEqual(row["exit_reason"], "breakeven_hit")

    async def test_target_price_still_labels_tp_hit(self):
        self._seed(execution_type="maker", fee_entry_usd=0.20, exit_p=101.40, pnl=14.0)
        row = await self._reconcile()
        self.assertEqual(row["exit_reason"], "tp_hit")


if __name__ == "__main__":
    unittest.main()

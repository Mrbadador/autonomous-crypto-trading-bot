import unittest
import sqlite3
import asyncio
from datetime import datetime, UTC, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

import config
from super_bot import SuperBotTrader
from swing_trader import SwingArchitectTrader


import tempfile
import os
import gc

_orig_connect = sqlite3.connect

class TestScalpShortEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.db_path = os.path.join(cls.temp_dir.name, "test_trading.db")
        cls.test_symbol = "NEAR/USDT"
        with _orig_connect(cls.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS swing_open_positions (
                    symbol TEXT PRIMARY KEY, side TEXT, entry_price REAL, size REAL,
                    stop_loss REAL, take_profit REAL, trailing_stop REAL, peak_price REAL,
                    leverage REAL, opened_at TEXT, updated_at TEXT, status TEXT,
                    r_multiple REAL, initial_risk_usd REAL, notes TEXT
                )
            """)
            conn.execute("CREATE TABLE IF NOT EXISTS mean_reversion_open_positions (symbol TEXT)")
            conn.execute("CREATE TABLE IF NOT EXISTS asset_locks (symbol TEXT PRIMARY KEY, locked_by TEXT, heartbeat TEXT)")
            conn.commit()

    @classmethod
    def tearDownClass(cls):
        gc.collect()
        try:
            cls.temp_dir.cleanup()
        except Exception:
            pass

    def setUp(self):
        # Clean test records from DB
        with _orig_connect(self.db_path) as conn:
            conn.execute("DELETE FROM swing_open_positions WHERE symbol = ?", (self.test_symbol,))
            conn.execute("DELETE FROM asset_locks WHERE symbol = ?", (self.test_symbol,))
            conn.commit()

    def tearDown(self):
        with _orig_connect(self.db_path) as conn:
            conn.execute("DELETE FROM swing_open_positions WHERE symbol = ?", (self.test_symbol,))
            conn.execute("DELETE FROM asset_locks WHERE symbol = ?", (self.test_symbol,))
            conn.commit()

    def test_01_configuration_parameters(self):
        """Verify all SCALP_SHORT configuration invariants are defined and within bounded limits."""
        self.assertTrue(hasattr(config, "SCALP_SHORT_ENABLED"))
        self.assertFalse(config.SCALP_SHORT_ENABLED)
        self.assertEqual(config.SCALP_SHORT_TP_PCT, 0.0120)  # +1.20% TP (1.20 R:R)
        self.assertEqual(config.SCALP_SHORT_SL_PCT, 0.0100)  # -1.00% SL
        self.assertEqual(config.SCALP_SHORT_BE_TRIGGER_PCT, 0.0050)  # +0.50% BE trigger
        self.assertEqual(config.SCALP_SHORT_BE_FEE_BUFFER_PCT, 0.0018)  # 0.18% fee buffer
        self.assertEqual(config.SCALP_SHORT_MAX_HOLD_MINS, 45.0)  # 45 min harvest/tighten checkpoint
        self.assertEqual(config.SCALP_SHORT_HARD_EXPIRY_MINS, 90.0)  # 90 min hard expiry
        self.assertEqual(config.SCALP_SHORT_MIN_RSI, 38.0)  # Never short < 38 RSI
        self.assertEqual(config.MAX_CONCURRENT_SHORTS, 1)  # Safeguard 1: strict max 1 short
        self.assertTrue(config.SCALP_SHORT_BTC_PUMP_GUARD)  # Safeguard 2: BTC macro guard
        self.assertTrue(config.SCALP_SHORT_REQUIRE_15M_EXHAUST)  # Safeguard 3: 15m exhaustion requirement
        self.assertEqual(config.SCALP_SHORT_MIN_HARVEST_PCT, 0.0045)  # Safeguard 4: +0.45% min harvest
        self.assertEqual(config.SCALP_SHORT_MAX_5M_ADX, 22.0)  # Option B: 5m ADX < 22 ceiling
        self.assertEqual(config.SCALP_SHORT_MAX_24H_GAIN_PCT, 3.0)  # Option B: 24h gain <= +3.0% ceiling
        self.assertEqual(config.SCALP_SHORT_MAX_DI_SPREAD, 4.0)  # Option B: Market +DI/-DI spread <= +4.0 ceiling


    def test_02_scalp_short_position_adoption(self):
        """Verify SuperBotTrader correctly adopts and tags SCALP_SHORT positions."""
        now_iso = datetime.now(UTC).isoformat()
        with _orig_connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO swing_open_positions 
                (symbol, side, entry_price, size, stop_loss, take_profit, trailing_stop, peak_price, leverage, opened_at, updated_at, status, r_multiple, initial_risk_usd, notes)
                VALUES (?, 'short', 100.0, 1.0, 101.00, 98.80, 101.00, 100.0, 3.0, ?, ?, 'ACTIVE', 0.0, 1.00, 'Quick Scalp Short [Post-Only Limit] (TP: +1.2%)')
            """, (self.test_symbol, now_iso, now_iso))
            conn.commit()

        bot = SuperBotTrader()
        with patch('super_bot.sqlite3.connect', side_effect=lambda *a, **kw: _orig_connect(self.temp_db if hasattr(self, 'temp_db') else self.db_path)):
            bot._load_all_positions_sync()

        self.assertIn(self.test_symbol, bot.active_positions)
        pos = bot.active_positions[self.test_symbol]
        self.assertEqual(pos["engine_type"], "SCALP_SHORT")
        self.assertEqual(pos["side"], "short")
        self.assertEqual(pos["entry_price"], 100.0)

    def test_03_scalp_short_breakeven_ratchet(self):
        """Verify that gaining +0.50% ratchets the stop loss to Entry - 0.18% fee buffer."""
        async def run_test():
            now_iso = datetime.now(UTC).isoformat()
            trader = SwingArchitectTrader()
            trader.uta_client = MagicMock()
            trader.uta_client._normalize_symbol.return_value = "TESTSCALPUSDT"
            trader.active_positions = {
                self.test_symbol: {
                    "symbol": self.test_symbol, "side": "short", "entry_price": 100.0,
                    "size": 1.0, "stop_loss": 101.00, "take_profit": 98.80, "trailing_stop": 101.00,
                    "peak_price": 100.0, "leverage": 3.0, "opened_at": now_iso, "updated_at": now_iso,
                    "status": "ACTIVE", "r_multiple": 0.0, "initial_risk_usd": 1.00,
                    "notes": "Quick Scalp Short [Post-Only Limit] (TP: +1.2%)"
                }
            }
            trader._load_active_positions_sync = MagicMock()

            # Return current price 99.50 (entry was 100.0, profit = +0.50% >= 0.50% trigger)
            trader.uta_client.exchange = MagicMock()
            trader.uta_client.exchange.fetch_ticker = AsyncMock(return_value={"last": 99.50, "close": 99.50})
            trader.uta_client.exchange.fetch_ohlcv = AsyncMock(return_value=[
                [1000, 100.0, 100.5, 99.0, 99.5, 100.0] for _ in range(50)
            ])
            trader.uta_client.modify_strategy_order = AsyncMock(return_value={"retCode": 0})

            await trader.manage_active_positions()

            pos = trader.active_positions.get(self.test_symbol)
            self.assertIsNotNone(pos)
            # Stop loss must be ratcheted down to 100 * (1 - 0.0018) = 99.82 (covering 2x taker fees + buffer)
            self.assertLess(pos["stop_loss"], 100.0)
            self.assertEqual(pos["stop_loss"], 99.82)

        asyncio.run(run_test())

    def test_04_scalp_short_time_decay_exit(self):
        """Verify that holding >= 45m with profit >= +0.45% harvests profit immediately."""
        async def run_test():
            # Opened 50 minutes ago (>= 45 min harvest checkpoint)
            opened_iso = (datetime.now(UTC) - timedelta(minutes=50)).isoformat()
            trader = SwingArchitectTrader()
            trader.uta_client = MagicMock()
            trader.uta_client._normalize_symbol.return_value = "TESTSCALPUSDT"
            trader.active_positions = {
                self.test_symbol: {
                    "symbol": self.test_symbol, "side": "short", "entry_price": 100.0,
                    "size": 1.0, "stop_loss": 101.00, "take_profit": 98.80, "trailing_stop": 101.00,
                    "peak_price": 99.50, "leverage": 3.0, "opened_at": opened_iso, "updated_at": opened_iso,
                    "status": "ACTIVE", "r_multiple": 0.50, "initial_risk_usd": 1.00,
                    "notes": "Quick Scalp Short [Post-Only Limit] (TP: +1.2%)"
                }
            }
            trader._load_active_positions_sync = MagicMock()

            # Current price is 99.50 (in profit by +0.50% >= +0.45% min_harvest_pct)
            trader.uta_client.exchange = MagicMock()
            trader.uta_client.exchange.fetch_ticker = AsyncMock(return_value={"last": 99.50, "close": 99.50})
            trader.uta_client.exchange.fetch_ohlcv = AsyncMock(return_value=[
                [1000, 100.0, 100.5, 99.0, 99.50, 100.0] for _ in range(50)
            ])
            trader.uta_client.modify_strategy_order = AsyncMock(return_value=True)
            trader.close_swing_position = AsyncMock()

            await trader.manage_active_positions()

            # Must have called close_swing_position with reason SCALP_SHORT_TIME_PROFIT
            trader.close_swing_position.assert_called_once_with(
                self.test_symbol,
                exit_price=99.50,
                reason="SCALP_SHORT_TIME_PROFIT"
            )

        asyncio.run(run_test())

    def test_05_concurrent_short_limit_veto(self):
        """Verify that holding active_shorts >= max_concurrent_shorts blocks further short scans."""
        async def run_test():
            bot = SuperBotTrader()
            bot.scalp_short_enabled = True
            bot.uta_client = MagicMock()
            bot.uta_client.exchange = MagicMock()
            bot.uta_client.exchange.fetch_tickers = AsyncMock()
            
            # Simulate 1 active short already held
            bot.active_positions = {
                "SOL/USDT": {"symbol": "SOL/USDT", "side": "short", "engine_type": "SCALP_SHORT"}
            }
            bot.max_concurrent_shorts = 1
            
            await bot.scan_quick_scalp_short_opportunities(mode="CHOP")
            
            # Tickers/OHLCV should NOT even be fetched because cap veto fires immediately
            bot.uta_client.exchange.fetch_tickers.assert_not_called()

        asyncio.run(run_test())

    def test_06_scalp_short_sub_harvest_time_decay_exit(self):
        """Verify Stage 1 (45m-90m) tightens SL instead of market-dumping, and Stage 2 (>=90m) exits as SCALP_SHORT_TIME_EXPIRY."""
        async def run_test():
            # Stage 1: 50 minutes (between 45m and 90m) with +0.15% gain (< +0.45% min_harvest_pct)
            opened_iso_50 = (datetime.now(UTC) - timedelta(minutes=50)).isoformat()
            trader = SwingArchitectTrader()
            trader.uta_client = MagicMock()
            trader.uta_client._normalize_symbol.return_value = "TESTSCALPUSDT"
            trader.active_positions = {
                self.test_symbol: {
                    "symbol": self.test_symbol, "side": "short", "entry_price": 100.0,
                    "size": 1.0, "stop_loss": 101.00, "take_profit": 98.80, "trailing_stop": 101.00,
                    "peak_price": 99.85, "leverage": 3.0, "opened_at": opened_iso_50, "updated_at": opened_iso_50,
                    "status": "ACTIVE", "r_multiple": 0.0, "initial_risk_usd": 1.00,
                    "notes": "Quick Scalp Short [Post-Only Limit]"
                }
            }
            trader._load_active_positions_sync = MagicMock()

            trader.uta_client.exchange = MagicMock()
            trader.uta_client.exchange.fetch_ticker = AsyncMock(return_value={"last": 99.85, "close": 99.85})
            trader.uta_client.exchange.fetch_ohlcv = AsyncMock(return_value=[
                [1000, 100.0, 100.5, 99.0, 99.85, 100.0] for _ in range(50)
            ])
            trader.uta_client.modify_strategy_order = AsyncMock(return_value=True)
            trader.close_swing_position = AsyncMock()

            await trader.manage_active_positions()

            # At 50m (< 90m), must NOT force market close; must tighten SL to 100.55 (-0.55%)
            trader.close_swing_position.assert_not_called()
            self.assertEqual(trader.active_positions[self.test_symbol]["stop_loss"], 100.55)

            # Stage 2: 95 minutes (>= 90m hard expiry) -> must close as SCALP_SHORT_TIME_EXPIRY
            opened_iso_95 = (datetime.now(UTC) - timedelta(minutes=95)).isoformat()
            trader.active_positions[self.test_symbol]["opened_at"] = opened_iso_95
            await trader.manage_active_positions()

            trader.close_swing_position.assert_called_once_with(
                self.test_symbol,
                exit_price=99.85,
                reason="SCALP_SHORT_TIME_EXPIRY"
            )

        asyncio.run(run_test())

    def test_07_btc_pump_guard_veto(self):
        """Verify Safeguard 2: BTC 5m bullish impulse blocks scanning altcoin shorts."""
        async def run_test():
            bot = SuperBotTrader()
            bot.scalp_short_enabled = True
            bot.uta_client = MagicMock()
            bot.uta_client._normalize_symbol.side_effect = lambda s: s.replace("/", "")
            bot.uta_client.exchange = MagicMock()
            
            # Simulate BTC pumping: price is 100,000, VWAP is 95,000, EMA9 > EMA21
            btc_candles = [[i * 300000, 95000 + (i * 200), 95000 + (i * 200) + 100, 95000 + (i * 200) - 50, 95000 + (i * 200) + 80, 10.0] for i in range(30)]
            bot.uta_client.exchange.fetch_ohlcv = AsyncMock(return_value=btc_candles)
            bot.uta_client.exchange.fetch_tickers = AsyncMock()
            
            bot.active_positions = {}
            bot.max_concurrent_shorts = 1
            bot.scalp_short_btc_guard = True
            
            await bot.scan_quick_scalp_short_opportunities(mode="CHOP")
            
            # fetch_tickers for altcoins must NOT be called because BTC guard triggers an early return
            bot.uta_client.exchange.fetch_tickers.assert_not_called()

        asyncio.run(run_test())

    def test_08_15m_exhaustion_veto(self):
        """Verify Safeguard 3: Candidate with 15m candle advancing green with < 20% wick is vetoed."""
        async def run_test():
            bot = SuperBotTrader()
            bot.scalp_short_enabled = True
            bot.uta_client = MagicMock()
            bot.uta_client._normalize_symbol.side_effect = lambda s: s.replace("/", "")
            bot.uta_client.exchange = MagicMock()
            
            bot.active_positions = {}
            bot.max_concurrent_shorts = 1
            bot.scalp_short_btc_guard = False  # Isolate test to 15m check
            bot.scalp_short_require_15m = True
            bot.symbols = ["NEAR/USDT"]
            
            # Ticker showing +1.5% gain in chop (under the +3.0% breakout ceiling)
            bot.uta_client.exchange.fetch_tickers = AsyncMock(return_value={"NEARUSDT": {"percentage": 1.5}})
            
            # 5m candles showing upper band touch + RSI > 68 + red rejection candle on 5m
            c5m = [[i * 300000, 5.0, 5.2, 4.9, 5.15, 100.0] for i in range(50)]
            # Last 5m candle: open=5.20, high=5.50, low=4.95, close=5.15 (red candle with >25% upper wick)
            c5m[-1] = [50 * 300000, 5.20, 5.50, 4.95, 5.15, 500.0]
            
            # 15m candles: last candle is STRONG GREEN (open=4.90, high=5.50, low=4.90, close=5.48)
            # Range = 0.60, upper wick = 5.50 - 5.48 = 0.02 -> 0.02 / 0.60 = 3.3% wick (< 20%)
            c15m = [[i * 900000, 4.8, 5.0, 4.7, 4.9, 200.0] for i in range(10)]
            c15m[-1] = [10 * 900000, 4.90, 5.50, 4.90, 5.48, 1000.0]
            
            async def mock_fetch_ohlcv(symbol, timeframe="5m", limit=60):
                if timeframe == "15m":
                    return c15m
                return c5m
                
            bot.uta_client.exchange.fetch_ohlcv = AsyncMock(side_effect=mock_fetch_ohlcv)
            bot._execute_grid_maker_entry = AsyncMock()
            
            await bot.scan_quick_scalp_short_opportunities(mode="CHOP")
            
            # Entry must NOT be executed because 15m exhaustion veto fired
            bot._execute_grid_maker_entry.assert_not_called()

        asyncio.run(run_test())

    def test_09_anti_breakout_shields(self):
        """Verify Option B Shields: Market +DI/-DI > 4.0 and 24h gain > 3.0% block CHOP shorts."""
        async def run_test():
            bot = SuperBotTrader()
            bot.scalp_short_enabled = True
            bot.uta_client = MagicMock()
            bot.uta_client._normalize_symbol.side_effect = lambda s: s.replace("/", "")
            bot.uta_client.exchange = MagicMock()
            bot.uta_client.exchange.fetch_tickers = AsyncMock()
            bot.uta_client.exchange.fetch_ohlcv = AsyncMock()
            bot.active_positions = {}
            bot.scalp_short_btc_guard = False
            bot.symbols = ["NEAR/USDT", "APT/USDT"]

            # 1. Market DI Spread Shield: +DI exceeds -DI by +18.5 (> +4.0 ceiling)
            bot.market_regime_di_spread = 18.5
            await bot.scan_quick_scalp_short_opportunities(mode="CHOP")
            bot.uta_client.exchange.fetch_tickers.assert_not_called()

            # 2. 24h Breakout Veto: Reset DI spread to 0.0, but NEAR is +15% and APT is +12% (> +3.0% ceiling)
            bot.market_regime_di_spread = 0.0
            bot.uta_client.exchange.fetch_tickers = AsyncMock(return_value={
                "NEARUSDT": {"percentage": 15.2},
                "APTUSDT": {"percentage": 12.4},
            })
            await bot.scan_quick_scalp_short_opportunities(mode="CHOP")
            # Neither symbol should have 5m OHLCV fetched because both exceeded the +3.0% 24h breakout ceiling
            bot.uta_client.exchange.fetch_ohlcv.assert_not_called()

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()


import unittest
import sqlite3
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, UTC

import config
from swing_trader import SwingArchitectTrader

class TestSwingArchitectAndMutex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_path = "trading_data.db"
        cls.test_symbol = "TEST/USDT"

    def test_01_mutex_lock_lifecycle(self):
        """Verify lock acquisition, capacity enforcement, and release."""
        trader = SwingArchitectTrader()
        
        # Ensure clean state for test symbol
        trader._release_lock_sync(self.test_symbol)

        # 1. Acquire lock
        acquired = trader._acquire_lock_sync(self.test_symbol, reason="TEST_LOCK")
        self.assertTrue(acquired, "Should successfully acquire lock")

        # Verify in DB
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT locked_by, reason FROM asset_locks WHERE symbol = ?", (self.test_symbol,)).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], "swing_architect")
            self.assertEqual(row[1], "TEST_LOCK")

        # 2. Release lock
        trader._release_lock_sync(self.test_symbol)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT locked_by FROM asset_locks WHERE symbol = ?", (self.test_symbol,)).fetchone()
            self.assertIsNone(row, "Lock should be deleted after release")

    def test_02_scalper_lock_detection(self):
        """Verify that live_trader and berserker_trader lock checks detect active locks."""
        # Insert a temporary lock for TEST_SOL/USDT (never touch live SOL/USDT)
        now_iso = datetime.now(UTC).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO asset_locks (symbol, locked_by, locked_at, reason, heartbeat)
                VALUES ('TEST_SOL/USDT', 'swing_architect', ?, '4H_SWING_LONG', ?)
            """, (now_iso, now_iso))
            conn.commit()

        # Simulate the exact query used in live_trader and berserker_trader
        with sqlite3.connect(self.db_path, timeout=5) as conn:
            res_locked = conn.execute("SELECT locked_by, reason FROM asset_locks WHERE symbol = ?", ('TEST_SOL/USDT',)).fetchone()
            res_unlocked = conn.execute("SELECT locked_by, reason FROM asset_locks WHERE symbol = ?", ('NON_EXISTENT_TEST/USDT',)).fetchone()

        self.assertIsNotNone(res_locked, "TEST_SOL/USDT must be detected as locked")
        self.assertEqual(res_locked[0], "swing_architect")
        self.assertIsNone(res_unlocked, "NON_EXISTENT_TEST/USDT must be detected as unlocked")

        # Clean up test lock
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM asset_locks WHERE symbol = 'TEST_SOL/USDT'")
            conn.commit()

    def test_03_ichimoku_and_adx_computation(self):
        """Verify mathematical calculation of Ichimoku Kumo and ADX on synthetic 4H data."""
        trader = SwingArchitectTrader()

        # Generate 100 periods of synthetic trending OHLCV data
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(UTC), periods=100, freq='4h')
        base_price = 100.0 + np.cumsum(np.random.normal(0.5, 1.0, 100))
        highs = base_price + np.random.uniform(0.5, 2.0, 100)
        lows = base_price - np.random.uniform(0.5, 2.0, 100)
        closes = base_price + np.random.uniform(-0.5, 0.5, 100)
        volumes = np.random.uniform(1000, 5000, 100)

        df = pd.DataFrame({
            "timestamp": dates,
            "open": base_price,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes
        })

        ind = trader._compute_4h_indicators(df)
        self.assertIn("tenkan", ind)
        self.assertIn("kijun", ind)
        self.assertIn("cloud_top", ind)
        self.assertIn("cloud_bottom", ind)
        self.assertIn("atr", ind)
        self.assertIn("adx", ind)
        self.assertGreater(ind["atr"], 0.0)
        self.assertGreaterEqual(ind["adx"], 0.0)

    def test_04_live_position_preservation(self):
        """Verify that live positions in swing_open_positions remain active and uncorrupted."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT symbol, side, entry_price, size, stop_loss, leverage FROM swing_open_positions").fetchall()
            for r in rows:
                self.assertTrue(len(r[0]) > 0, "Symbol must be non-empty")
                self.assertIn(r[1].lower(), ["long", "short"], "Side must be long or short")
                self.assertGreater(float(r[2]), 0.0, "Entry price must be positive")
                self.assertGreater(float(r[3]), 0.0, "Size must be positive")
                self.assertGreater(float(r[5]), 0.0, "Leverage must be positive")
                print(f"Verified Active Position: {r[0]} {r[1]} Entry: ${r[2]} Size: {r[3]} Leverage: {r[5]}x (Intact)")

    def test_05_dynamic_tiered_leverage(self):
        """Verify dynamic 3x / 5x tiered leverage evaluation rules."""
        trader = SwingArchitectTrader()
        self.assertEqual(trader.standard_leverage, 3)
        self.assertEqual(trader.high_quality_leverage, 5)

        # High Quality Case: Conviction >= 0.80, ADX >= 30.0, Optimal RSI (40-65 for LONG)
        hq_ind = {"close": 100.0, "atr": 2.0, "adx": 32.5, "rsi": 54.0}
        hq_conviction = 0.85
        is_long = True
        is_optimal_rsi = (40.0 <= hq_ind["rsi"] <= 65.0) if is_long else (35.0 <= hq_ind["rsi"] <= 60.0)
        is_high_quality = (
            hq_conviction >= trader.high_quality_conviction and
            hq_ind["adx"] >= trader.high_quality_adx and
            is_optimal_rsi
        )
        assigned_leverage = trader.high_quality_leverage if is_high_quality else trader.standard_leverage
        self.assertTrue(is_high_quality)
        self.assertEqual(assigned_leverage, 5)

        # Standard Quality Case 1: Lower Conviction (0.72 < 0.80)
        std_conviction = 0.72
        is_high_quality_std = (
            std_conviction >= trader.high_quality_conviction and
            hq_ind["adx"] >= trader.high_quality_adx and
            is_optimal_rsi
        )
        assigned_leverage_std = trader.high_quality_leverage if is_high_quality_std else trader.standard_leverage
        self.assertFalse(is_high_quality_std)
        self.assertEqual(assigned_leverage_std, 3)

        # Standard Quality Case 2: Lower ADX (26.0 < 30.0)
        low_adx_ind = {"close": 100.0, "atr": 2.0, "adx": 26.0, "rsi": 54.0}
        is_high_quality_adx = (
            hq_conviction >= trader.high_quality_conviction and
            low_adx_ind["adx"] >= trader.high_quality_adx and
            is_optimal_rsi
        )
        assigned_leverage_adx = trader.high_quality_leverage if is_high_quality_adx else trader.standard_leverage
        self.assertFalse(is_high_quality_adx)
        self.assertEqual(assigned_leverage_adx, 3)

        # Standard Quality Case 3: Overbought RSI (RSI 72 > 65)
        overbought_ind = {"close": 100.0, "atr": 2.0, "adx": 35.0, "rsi": 72.0}
        is_optimal_rsi_ob = (40.0 <= overbought_ind["rsi"] <= 65.0)
        is_high_quality_ob = (
            hq_conviction >= trader.high_quality_conviction and
            overbought_ind["adx"] >= trader.high_quality_adx and
            is_optimal_rsi_ob
        )
        assigned_leverage_ob = trader.high_quality_leverage if is_high_quality_ob else trader.standard_leverage
        self.assertFalse(is_high_quality_ob)
        self.assertEqual(assigned_leverage_ob, 3)

    def test_06_direction_aware_cooldowns(self):
        """Verify direction-aware cooldowns: short stop-outs do not block long entries and LONG_ prefixes remain authoritative."""
        from datetime import timedelta
        trader = SwingArchitectTrader()
        sym = "ZZTEST_CD/USDT"
        now = datetime.now(UTC)
        fut = (now + timedelta(hours=6)).isoformat()
        recent = (now - timedelta(minutes=30)).isoformat()

        def _reset():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM swing_cooldowns WHERE symbol = ?", (sym,))
                conn.execute("DELETE FROM swing_trades WHERE symbol = ?", (sym,))
                conn.commit()

        try:
            # 1. Prefixed SHORT_ cooldown must NOT block LONG, must block SHORT
            _reset()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT OR REPLACE INTO swing_cooldowns VALUES (?, ?, ?, ?)", (sym, fut, "SHORT_STOP_LOSS", now.isoformat()))
                conn.commit()
            self.assertIsNone(trader._get_cooldown_sync(sym, "LONG"))
            self.assertIsNotNone(trader._get_cooldown_sync(sym, "SHORT"))

            # 2. Prefixed LONG_ cooldown must block LONG even if a later short trade won
            _reset()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT OR REPLACE INTO swing_cooldowns VALUES (?, ?, ?, ?)", (sym, fut, "LONG_STOP_LOSS", now.isoformat()))
                conn.execute("""
                    INSERT INTO swing_trades (timestamp, symbol, side, entry_price, exit_price, size, pnl, roe_pct, holding_hours, exit_reason, notes)
                    VALUES (?, ?, 'short', 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 'TAKE_PROFIT', '')
                """, (recent, sym))
                conn.commit()
            self.assertIsNotNone(trader._get_cooldown_sync(sym, "LONG"))
            self.assertIsNone(trader._get_cooldown_sync(sym, "SHORT"))
        finally:
            _reset()

if __name__ == "__main__":
    unittest.main()


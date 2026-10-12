"""
Bot #5: The Swing Architect (4H / Daily Macro Trend Engine)
============================================================
Autonomous multi-day trend capture engine that scans the top 10 fleet coins
on 4-Hour and Daily timeframes.

Key Architecture:
- Strategy: 4H Ichimoku Kumo Cloud Breakout + 4H ADX > 25 + 1D EMA 50/200 Trend Alignment
- Exclusive Asset Mutex: Acquires exclusive lock on asset in `asset_locks` table,
  preventing 15m (Sniper) and 5m (Berserker) from opening conflicting scalps.
- Institutional AI Gates: Multi-Agent Conviction Boardroom, Sentinel News Veto,
  Macro Posture Check, HTF Resistance Runway Gate.
- Position Management: +1.5R Breakeven Ratchet + Dynamic 4H Kijun-sen Trailing Stop.
"""

import asyncio
import json
import logging
import os
import signal
import sqlite3
import sys
import time
from datetime import datetime, UTC, timedelta
import pandas as pd
import numpy as np

import config
from bitget_uta_client import BitgetUTAClient
from conviction_filter import ConvictionBoardroom
from db_manager import DBWriteQueue
from indicators import detect_momentum_divergence, compute_volume_profile, compute_td_sequential


# BUG-07 FIX: Use explicit handler attachment instead of basicConfig()
# basicConfig() is a no-op when the root logger already has handlers (set by start_system.py).
log = logging.getLogger("swing_architect")
log.setLevel(logging.INFO)
log.propagate = False
if not log.handlers:
    _fmt = logging.Formatter("%(asctime)s [%(levelname)s] swing_architect: %(message)s")
    _sh = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(_fmt)
    log.addHandler(_sh)
    try:
        os.makedirs("logs", exist_ok=True)
        _fh = logging.FileHandler("logs/swing_trader.log", encoding="utf-8")
        _fh.setFormatter(_fmt)
        log.addHandler(_fh)
    except Exception:
        pass


class SwingArchitectTrader:
    def __init__(self):
        self.symbols = getattr(config, "SYMBOLS", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
        self.timeframe_entry = "4h"
        self.timeframe_macro = "1d"
        self.allocation_usdt = getattr(config, "SWING_TRADE_ALLOCATION_USDT", 85.0)
        self.balance_pct = getattr(config, "SWING_BALANCE_PCT", 0.175)
        self.standard_leverage = getattr(config, "SWING_STANDARD_LEVERAGE", 3)
        self.high_quality_leverage = getattr(config, "SWING_HIGH_QUALITY_LEVERAGE", 5)
        self.high_quality_conviction = getattr(config, "SWING_HIGH_QUALITY_CONVICTION_MIN", 0.85)
        self.high_quality_adx = getattr(config, "SWING_HIGH_QUALITY_ADX_MIN", 32.0)
        self.leverage = self.standard_leverage
        self.max_locks = getattr(config, "SWING_MAX_CONCURRENT_LOCKS", 4)
        self.atr_sl_mult = getattr(config, "SWING_ATR_SL_MULT", 3.0)
        self.runner_tp_mult = getattr(config, "SWING_RUNNER_TP_MULT", 4.0)
        self.be_threshold_r = getattr(config, "SWING_BREAKEVEN_R_THRESHOLD", 2.0)
        self.min_adx = getattr(config, "SWING_MIN_ADX", 22.0)
        self.check_interval = getattr(config, "SWING_CHECK_INTERVAL_SECS", 300)

        self.uta_client = None
        self.boardroom = ConvictionBoardroom()
        from sentinel_agent import SentinelAgent
        self.sentinel = SentinelAgent()
        self._news_size_multipliers = {}  # symbol -> float (1.0 CLEAR, 0.5 WARNING, 0.0 VETO)
        self.db_queue = DBWriteQueue.get_instance("trading_data.db")
        self._stop_event = asyncio.Event()
        self.active_positions = {}  # symbol -> position dict
        self._background_tasks = set()  # SRE-P0: Anchors tasks to prevent GC
        self._last_ai_health_check_ts = {}  # symbol -> last health check timestamp
        self._ai_veto_cache = {}  # (symbol, direction) -> dict(ts, candle_ts, price, reason)
        self._trap_pass_cache = {}  # (symbol, direction) -> dict(ts, candle_ts, price, score, reason)

        self._init_database()
        self._load_active_positions_sync()

    def _track_task(self, coro, name: str = "unnamed_task"):
        """SRE-P0 Fix: Anchors coroutines to prevent garbage collection mid-execution."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        def _on_done(t):
            self._background_tasks.discard(t)
            try:
                exc = t.exception()
                if exc:
                    log.error(f"[TASK-ERROR] Background task '{name}' failed: {exc}")
            except (asyncio.CancelledError, Exception):
                pass
        task.add_done_callback(_on_done)
        return task

    def _init_database(self):
        """Ensures required tables exist in trading_data.db."""
        try:
            with sqlite3.connect("trading_data.db", timeout=10.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA busy_timeout=10000;")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS asset_locks (
                        symbol TEXT PRIMARY KEY,
                        locked_by TEXT NOT NULL,
                        locked_at TEXT NOT NULL,
                        reason TEXT,
                        heartbeat TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS swing_open_positions (
                        symbol TEXT PRIMARY KEY,
                        side TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        size REAL NOT NULL,
                        stop_loss REAL NOT NULL,
                        take_profit REAL NOT NULL,
                        trailing_stop REAL,
                        peak_price REAL,
                        leverage REAL DEFAULT 3.0,
                        opened_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        status TEXT DEFAULT 'ACTIVE',
                        r_multiple REAL DEFAULT 0.0,
                        initial_risk_usd REAL,
                        notes TEXT,
                        intended_entry_price REAL,
                        actual_fill_price REAL,
                        slippage_bps REAL,
                        fee_entry_usd REAL,
                        execution_type TEXT,
                        gates_snapshot TEXT
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS swing_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_price REAL,
                        size REAL NOT NULL,
                        pnl REAL,
                        roe_pct REAL,
                        holding_hours REAL,
                        exit_reason TEXT,
                        notes TEXT,
                        intended_entry_price REAL,
                        actual_fill_price REAL,
                        slippage_bps REAL,
                        fee_paid_usd REAL,
                        fee_maker_usd REAL,
                        fee_taker_usd REAL,
                        funding_usd REAL,
                        funding_source TEXT,
                        gates_snapshot TEXT,
                        status TEXT DEFAULT 'CLOSED'
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS swing_cooldowns (
                        symbol TEXT PRIMARY KEY,
                        cooldown_until TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                """)
                conn.commit()
            log.info("[INIT] Swing Architect tables verified in trading_data.db.")
        except Exception as e:
            log.error(f"[INIT] Database init error: {e}")

    def _get_cooldown_sync(self, symbol: str, direction: str = None) -> str | None:
        """
        Checks if symbol is currently in an active post-stop cooldown for the target direction.
        Verifies both swing_cooldowns and actual stop loss / loss exit timestamps in swing_trades.
        Opposite-direction stop-outs (e.g. an old SHORT scalp stopped out by a bullish breakout)
        do NOT block LONG entries.
        """
        try:
            now_utc = datetime.now(UTC)
            now_iso = now_utc.isoformat()
            min_cooldown_secs = 4.0 * 3600.0  # 4 hour strict minimum cooldown
            target_dir = direction.upper() if direction else None

            with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                # Fetch last trade side to verify whether any active cooldown was from the same direction
                trade_row = conn.execute("""
                    SELECT timestamp, exit_reason, pnl, side FROM swing_trades
                    WHERE symbol = ?
                    ORDER BY id DESC LIMIT 1
                """, (symbol,)).fetchone()
                last_side = str(trade_row[3] or "").upper() if (trade_row and len(trade_row) > 3) else ""

                # 1. Check explicit swing_cooldowns table
                row = conn.execute("SELECT cooldown_until, reason FROM swing_cooldowns WHERE symbol = ?", (symbol,)).fetchone()
                if row and row[0]:
                    cd_str = str(row[0])
                    cd_reason = str(row[1] or "").upper()
                    if cd_str > now_iso:
                        if target_dir:
                            if cd_reason.startswith("LONG_"):
                                return cd_str if target_dir == "LONG" else None
                            elif cd_reason.startswith("SHORT_") or "SCALP_SHORT" in cd_reason:
                                return cd_str if target_dir == "SHORT" else None
                            else:
                                # Legacy unprefixed reason: infer from last closed trade side
                                if last_side and last_side != target_dir:
                                    return None
                        return cd_str

                # 2. Check actual exit timestamps in swing_trades for stops or losses (persists across restarts/bypasses)
                if trade_row and trade_row[0]:
                    if target_dir and last_side and last_side != target_dir:
                        return None
                    last_ts_str, exit_reason, last_pnl = str(trade_row[0]), str(trade_row[1] or ""), float(trade_row[2] or 0.0)
                    is_stop_or_loss = ("stop" in exit_reason.lower() or "sl" in exit_reason.lower() or last_pnl < 0)
                    if is_stop_or_loss:
                        try:
                            # Normalize SQLite timestamp format
                            clean_ts = last_ts_str.replace("Z", "+00:00")
                            if "T" not in clean_ts and " " in clean_ts:
                                clean_ts = clean_ts.replace(" ", "T")
                            if "+" not in clean_ts and not clean_ts.endswith("Z"):
                                clean_ts += "+00:00"
                            exit_dt = datetime.fromisoformat(clean_ts)
                            elapsed_secs = (now_utc - exit_dt).total_seconds()
                            if elapsed_secs < min_cooldown_secs:
                                cd_until_dt = exit_dt + timedelta(seconds=min_cooldown_secs)
                                cd_until_str = cd_until_dt.isoformat()
                                return cd_until_str
                        except Exception as parse_err:
                            log.debug(f"[COOLDOWN-PARSE-NOTE] {symbol}: {parse_err}")
        except Exception:
            pass
        return None

    def _count_fleet_long_positions_sync(self) -> int:
        """
        Queries all fleet open position tables to count active long positions across the entire bot.
        """
        total_longs = 0
        fleet_tables = ["swing_open_positions", "open_positions", "berserker_open_positions", "marl_live_open_positions"]
        try:
            with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                for tbl in fleet_tables:
                    try:
                        rows = conn.execute(f"SELECT side FROM {tbl} WHERE status = 'ACTIVE' OR status = 'open'").fetchall()
                        for r in rows:
                            if "long" in str(r[0]).lower():
                                total_longs += 1
                    except Exception:
                        pass
        except Exception as e:
            log.debug(f"[FLEET-COUNT-NOTE] Error counting fleet longs: {e}")
            total_longs = len([p for p in self.active_positions.values() if "long" in str(p.get("side", "")).lower()])
        return total_longs

    def _set_cooldown_sync(self, symbol: str, hours: float = 12.0, reason: str = "STOP_LOSS", enforce_floor: bool = True):
        """Places a symbol into a temporary cool-off period following an on-exchange stop-out or unfilled sniper abort."""
        try:
            effective_hours = max(float(hours), 4.0) if enforce_floor else max(float(hours), 0.25)
            cooldown_until = (datetime.now(UTC) + timedelta(hours=effective_hours)).isoformat()
            now_iso = datetime.now(UTC).isoformat()
            with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO swing_cooldowns (symbol, cooldown_until, reason, created_at)
                    VALUES (?, ?, ?, ?)
                """, (symbol, cooldown_until, reason, now_iso))
                conn.commit()
            log.info(f"[COOLDOWN-SET] {symbol} locked in {effective_hours:.2f}h cooldown until {cooldown_until} (Anti-Whipsaw Rule #166 | Reason: {reason})")
        except Exception as e:
            log.warning(f"[COOLDOWN-NOTE] Error setting cooldown for {symbol}: {e}")

    def _record_shadow_signal_sync(self, strategy: str, symbol: str, direction: str, entry_price: float, stop_loss: float = None, take_profit: float = None, veto_reason: str = "", candle_key: str = ""):
        """Stage 5: Persists unexecuted signals for counterfactual sample size expansion with candle-key deduplication."""
        try:
            now_iso = datetime.now(UTC).isoformat()
            ck = candle_key or f"{symbol}|{direction.upper()}|{now_iso}"
            with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS shadow_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        stop_loss REAL,
                        take_profit REAL,
                        veto_reason TEXT,
                        exit_price REAL,
                        pnl_pct REAL,
                        status TEXT DEFAULT 'OPEN',
                        candle_key TEXT
                    )
                """)
                conn.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_shadow_signals_unique ON shadow_signals (symbol, direction, candle_key);
                """)
                conn.execute("""
                    INSERT OR IGNORE INTO shadow_signals (timestamp, strategy, symbol, direction, entry_price, stop_loss, take_profit, veto_reason, exit_price, pnl_pct, status, candle_key)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (now_iso, strategy, symbol, direction.upper(), float(entry_price), float(stop_loss) if stop_loss else None, float(take_profit) if take_profit else None, str(veto_reason), None, None, "OPEN", str(ck)))
                conn.commit()
        except Exception as se:
            log.debug(f"[SHADOW-SIGNAL-NOTE] Failed to log shadow signal: {se}")

    def _load_active_positions_sync(self):
        """Synchronizes in-memory positions with swing_open_positions."""
        try:
            with sqlite3.connect("trading_data.db", timeout=10.0) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("SELECT * FROM swing_open_positions WHERE status = 'ACTIVE'").fetchall()
                active = {}
                for r in rows:
                    pos = dict(r)
                    active[pos["symbol"]] = pos
                self.active_positions = active
        except Exception as e:
            log.error(f"[SYNC] Error loading active positions: {e}")

    async def initialize_clients(self):
        """Initializes Bitget UTA async exchange client and starts Sentinel RSS background refresh."""
        try:
            self.uta_client = BitgetUTAClient()
            self.sentinel.start_background_refresh([s.split("/")[0] for s in self.symbols])
            log.info("[CLIENT] BitgetUTAClient connected and Sentinel background refresh started.")
        except Exception as e:
            log.error(f"[CLIENT] Failed to initialize BitgetUTAClient: {e}")
            raise

    # ---------------------------------------------------------------------------
    # Exclusive Asset Mutex Lock Management
    # ---------------------------------------------------------------------------
    def _acquire_lock_sync(self, symbol: str, reason: str = "4H_SWING_LONG") -> bool:
        """Acquires exclusive asset lock in asset_locks table without overwriting other engines' locks."""
        now_utc = datetime.now(UTC)
        now_iso = now_utc.isoformat()
        try:
            with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                # Check active swing position count against maximum capacity
                active_count = len(self.active_positions)
                if active_count >= self.max_locks:
                    log.warning(f"[LOCK-GATE] Cannot lock {symbol}: Active swing positions ({active_count}) at max ({self.max_locks}).")
                    return False

                # Verify no other fleet engine holds a live lock on this symbol
                existing = conn.execute(
                    "SELECT locked_by, heartbeat FROM asset_locks WHERE symbol = ?",
                    (symbol,)
                ).fetchone()
                if existing and existing[0] and str(existing[0]) != "swing_architect":
                    hb_str = str(existing[1] or "")
                    lock_is_fresh = True
                    try:
                        clean_hb = hb_str.replace("Z", "+00:00")
                        if "+" not in clean_hb:
                            clean_hb += "+00:00"
                        hb_dt = datetime.fromisoformat(clean_hb)
                        if (now_utc - hb_dt).total_seconds() > 900.0:
                            lock_is_fresh = False
                    except Exception:
                        lock_is_fresh = True
                    if lock_is_fresh:
                        log.warning(f"[LOCK-GATE] Cannot lock {symbol}: Already locked by '{existing[0]}' (heartbeat: {hb_str}).")
                        return False

                # Acquire lock for swing_architect
                conn.execute("""
                    INSERT OR REPLACE INTO asset_locks (symbol, locked_by, locked_at, reason, heartbeat)
                    VALUES (?, 'swing_architect', ?, ?, ?)
                """, (symbol, now_iso, reason, now_iso))
                conn.commit()
            log.info(f"[LOCK-ACQUIRED] Exclusive Mutex Lock granted for {symbol} (Reason: {reason}).")
            return True
        except Exception as e:
            log.error(f"[LOCK-ERROR] Failed to acquire lock for {symbol}: {e}")
            return False

    def _release_lock_sync(self, symbol: str):
        """Releases exclusive asset lock owned by swing_architect."""
        try:
            with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                conn.execute("DELETE FROM asset_locks WHERE symbol = ? AND locked_by = 'swing_architect'", (symbol,))
                conn.commit()
            log.info(f"[LOCK-RELEASED] Exclusive Mutex Lock freed for {symbol}. Intraday bots unblocked.")
        except Exception as e:
            log.error(f"[LOCK-ERROR] Failed to release lock for {symbol}: {e}")

    def _refresh_heartbeats_sync(self):
        """Refreshes heartbeat for all actively locked swing assets and prunes only orphaned swing_architect locks."""
        now_iso = datetime.now(UTC).isoformat()
        try:
            with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                active_symbols = list(self.active_positions.keys())
                for sym in active_symbols:
                    conn.execute("""
                        UPDATE asset_locks SET heartbeat = ?
                        WHERE symbol = ? AND locked_by = 'swing_architect'
                    """, (now_iso, sym))
                # Prune only swing_architect locks whose positions are no longer active
                if active_symbols:
                    placeholders = ",".join("?" for _ in active_symbols)
                    conn.execute(f"""
                        DELETE FROM asset_locks
                        WHERE locked_by = 'swing_architect' AND symbol NOT IN ({placeholders})
                    """, active_symbols)
                else:
                    conn.execute("DELETE FROM asset_locks WHERE locked_by = 'swing_architect'")
                conn.commit()
        except Exception as e:
            log.debug(f"[HEARTBEAT-NOTE] Lock heartbeat refresh error: {e}")

    # ---------------------------------------------------------------------------
    # Quantitative Technical Analysis (4H & 1D)
    # ---------------------------------------------------------------------------
    def _compute_4h_indicators(self, df: pd.DataFrame) -> dict:
        """Computes Ichimoku Kumo (26-period shifted Senkou Span A/B), ADX(14), and ATR(14) on 4H candles."""
        if len(df) < 55:
            return {}

        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Ichimoku Components
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        tenkan = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2.0

        # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        kijun = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2.0

        # Senkou Span A & B: Classic 26-period forward shift (cloud at current bar was projected 26 bars ago)
        span_a_raw = (tenkan + kijun) / 2.0
        span_b_raw = (high.rolling(window=52).max() + low.rolling(window=52).min()) / 2.0
        if len(df) >= 78:
            span_a = span_a_raw.shift(26)
            span_b = span_b_raw.shift(26)
        else:
            span_a = span_a_raw
            span_b = span_b_raw

        # ATR(14)
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr_14 = tr.rolling(window=14).mean()

        # ADX(14)
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        tr_sum = tr.rolling(window=14).sum()
        plus_di = 100.0 * (pd.Series(plus_dm, index=df.index).rolling(window=14).sum() / tr_sum)
        minus_di = 100.0 * (pd.Series(minus_dm, index=df.index).rolling(window=14).sum() / tr_sum)
        dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx_14 = dx.rolling(window=14).mean()

        # RSI(14) for Overbought/Oversold exhaustion filters
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_14 = 100.0 - (100.0 / (1.0 + rs))
        rsi = float(rsi_14.iloc[-1]) if not pd.isna(rsi_14.iloc[-1]) else 50.0

        o = float(df["open"].iloc[-1])
        h = float(high.iloc[-1])
        l = float(low.iloc[-1])
        c = float(close.iloc[-1])
        prev_c = float(close.iloc[-2]) if len(close) >= 2 else o
        t = float(tenkan.iloc[-1])
        k = float(kijun.iloc[-1])
        sa_val = span_a.iloc[-1] if not pd.isna(span_a.iloc[-1]) else span_a_raw.iloc[-1]
        sb_val = span_b.iloc[-1] if not pd.isna(span_b.iloc[-1]) else span_b_raw.iloc[-1]
        sa = float(sa_val)
        sb = float(sb_val)
        atr = float(atr_14.iloc[-1])
        adx = float(adx_14.iloc[-1]) if not pd.isna(adx_14.iloc[-1]) else 0.0

        cloud_top = max(sa, sb)
        cloud_bottom = min(sa, sb)

        bar_range = max(1e-8, h - l)
        upper_wick_ratio = (h - max(o, c)) / bar_range
        lower_wick_ratio = (min(o, c) - l) / bar_range
        bullish_candle_4h = (c > o) and (c >= prev_c) and (upper_wick_ratio <= 0.60)
        bearish_candle_4h = (c < o) and (c <= prev_c) and (lower_wick_ratio <= 0.60)

        chikou_ref = float(close.iloc[-27]) if len(df) >= 27 else float(close.iloc[0])
        chikou_bull_clear = bool(c > chikou_ref) if len(df) >= 27 else True
        chikou_bear_clear = bool(c < chikou_ref) if len(df) >= 27 else True

        # Bullish Breakout: Close > Cloud Top AND Close > Kijun AND Chikou Clear AND (Tenkan >= Kijun * 0.995 or Close > Cloud Top * 1.002)
        bullish_kumo = (c > cloud_top) and (c > k) and chikou_bull_clear and ((t >= k * 0.995) or (c > cloud_top * 1.002))

        # Bearish Breakdown: Close < Cloud Bottom AND Close < Kijun AND Chikou Clear AND (Tenkan <= Kijun * 1.005 or Close < Cloud Bottom * 0.998)
        bearish_kumo = (c < cloud_bottom) and (c < k) and chikou_bear_clear and ((t <= k * 1.005) or (c < cloud_bottom * 0.998))

        ts_col = "timestamp" if "timestamp" in df.columns else ("ts" if "ts" in df.columns else None)
        candle_ts = int(df[ts_col].iloc[-1]) if ts_col else int(time.time() // 14400 * 14400)
        vol_col = "volume" if "volume" in df.columns else ("vol" if "vol" in df.columns else None)
        vol_ratio_20ma = 1.0
        if vol_col and len(df) >= 20:
            v_ma20 = float(df[vol_col].rolling(window=20).mean().iloc[-1] or 0.0)
            if v_ma20 > 0:
                vol_ratio_20ma = round(float(df[vol_col].iloc[-1]) / v_ma20, 2)

        return {
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "prev_close": prev_c,
            "tenkan": t,
            "kijun": k,
            "span_a": sa,
            "span_b": sb,
            "cloud_top": cloud_top,
            "cloud_bottom": cloud_bottom,
            "atr": atr,
            "adx": adx,
            "rsi": rsi,
            "bullish_kumo": bullish_kumo,
            "bearish_kumo": bearish_kumo,
            "bullish_candle_4h": bullish_candle_4h,
            "bearish_candle_4h": bearish_candle_4h,
            "upper_wick_ratio": round(upper_wick_ratio, 3),
            "lower_wick_ratio": round(lower_wick_ratio, 3),
            "candle_ts": candle_ts,
            "vol_ratio_20ma": vol_ratio_20ma,
            "chikou_bull_clear": chikou_bull_clear,
            "chikou_bear_clear": chikou_bear_clear
        }

    def _compute_1d_trend(self, df_1d: pd.DataFrame) -> dict:
        """Computes 1D EMA 50 & 200 for macro trend alignment."""
        if len(df_1d) < 200:
            return {"bullish_alignment": True, "bearish_alignment": True, "ema50": 0.0, "ema200": 0.0}

        close = df_1d["close"]
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
        curr_price = close.iloc[-1]

        bullish = (curr_price > ema200) and (ema50 >= (ema200 * 0.98))
        bearish = (curr_price < ema200) and (ema50 <= (ema200 * 1.02))
        daily_trend_str = "BULLISH" if bullish else ("BEARISH" if bearish else "NEUTRAL")
        return {
            "bullish_alignment": bool(bullish),
            "bearish_alignment": bool(bearish),
            "ema50": float(ema50),
            "ema200": float(ema200),
            "close": float(curr_price),
            "daily_trend": daily_trend_str
        }

    # ---------------------------------------------------------------------------
    # Institutional AI Defense Gates
    # ---------------------------------------------------------------------------
    async def _check_macro_safety(self, direction: str = "LONG") -> tuple[bool, str]:
        """Validates macro environment from system_state (Fail-Closed on DB error)."""
        def _read_macro():
            try:
                with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                    kill = conn.execute("SELECT value FROM system_state WHERE key = 'macro_global_killswitch'").fetchone()
                    bias = conn.execute("SELECT value FROM system_state WHERE key = 'ai_macro_bias'").fetchone()
                    reg = conn.execute("SELECT value FROM system_state WHERE key = 'market_regime_label'").fetchone()
                    pb = conn.execute("SELECT value FROM system_state WHERE key = 'ai_event_playbook'").fetchone()
                    summary = conn.execute("SELECT value FROM system_state WHERE key = 'ai_event_summary'").fetchone()
                    return (
                        (kill[0] if kill else "0"),
                        (bias[0] if bias else "NEUTRAL"),
                        (reg[0] if reg else "NEUTRAL"),
                        (pb[0] if pb else "NORMAL_OPERATIONS"),
                        (summary[0] if summary else ""),
                        True
                    )
            except Exception as db_err:
                return "1", "DEFENSIVE_BEAR", "BEAR_TREND", "NORMAL_OPERATIONS", str(db_err), False

        killswitch, macro_bias, market_regime, event_playbook, event_summary, db_ok = await asyncio.to_thread(_read_macro)
        if not db_ok:
            return False, f"Macro state DB read failed (fail-closed): {event_summary}"
        if killswitch == "1":
            return False, "Global Macro Killswitch is ENGAGED."
        
        dir_upper = direction.upper()

        # V40.0 Event-Driven Catalyst Directives (Override baseline indicators)
        if event_playbook == "SECULAR_SHORT_BREAKDOWN":
            if dir_upper == "LONG":
                return False, f"Structural Bear Event Active ({event_summary}). Halting long entries."
            elif dir_upper == "SHORT":
                return True, f"Macro clear: Structural Bear Event Active ({event_summary}). Breakdown shorts authorized."

        if event_playbook == "FLASH_REBOUND_TRAP":
            if dir_upper == "SHORT":
                return False, f"Flash Liquidity Shock Active ({event_summary}). Anti-bottom short shield engaged to prevent short squeeze."

        if event_playbook == "BULL_MOMENTUM_EXPANSION":
            if dir_upper == "SHORT":
                return False, f"Major Pro-Crypto Catalyst Active ({event_summary}). Halting counter-trend shorts."
            elif dir_upper == "LONG":
                return True, f"Macro clear: Pro-Crypto Catalyst Active ({event_summary}). Breakout longs authorized."

        if dir_upper == "LONG" and (macro_bias.upper() in ("BEARISH", "DEFENSIVE_BEAR") or market_regime in ("BEAR_TREND", "BEARISH")):
            return False, f"Macro/Regime is {macro_bias}/{market_regime}. Halting long swing entries (anti-dead-cat-bounce)."
        if dir_upper == "SHORT" and (macro_bias.upper() not in ("BEARISH", "DEFENSIVE_BEAR", "BEAR_TREND") and market_regime not in ("BEAR_TREND", "BEARISH")):
            return False, f"Macro Bias is {macro_bias} and Regime is {market_regime}. Halting short swing entries (requires confirmed bear regime to avoid short squeezes)."
        return True, "Macro clear."

    async def _check_news_veto(self, symbol: str, direction: str = "LONG") -> tuple[bool, str]:
        """Queries unified Sentinel RSS + sentiment_state.json (2h max-age, fail-closed, WARNING=0.5x size-down)."""
        ticker = symbol.split("/")[0]
        try:
            verdict, reason, size_mult = await self.sentinel.async_perform_directional_audit(
                ticker, direction=direction
            )
            self._news_size_multipliers[symbol] = float(size_mult)
            if verdict in ("VETO", "BLOCK") or size_mult <= 0.0:
                return False, f"Sentinel News Veto: {reason}"
            if verdict == "WARNING" or size_mult < 1.0:
                log.warning(f"[SENTINEL-WARNING] Caution headline for {symbol} — scaling position size by {size_mult:.2f}x ({reason})")
                return True, f"News WARNING (0.5x size-down): {reason}"
            return True, "News safe."
        except Exception as e:
            self._news_size_multipliers[symbol] = 0.0
            log.warning(f"[SENTINEL-VETO] News check failed (fail-closed) for {symbol}: {e}")
            return False, f"Sentinel news audit error (fail-closed): {e}"

    async def _check_htf_runway(self, symbol: str, current_price: float, direction: str = "LONG") -> tuple[bool, str]:
        """Enforces minimum runway to major resistance/support from asset_scout."""
        def _read_confluence():
            try:
                with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                    row = conn.execute("SELECT value FROM system_state WHERE key = 'ai_mtf_confluence'").fetchone()
                    if row:
                        data = json.loads(row[0])
                        matrix = data.get("confluence_matrix", {})
                        return matrix.get(symbol, {})
            except Exception:
                pass
            return {}

        conf = await asyncio.to_thread(_read_confluence)
        if not conf:
            return True, "No barrier data."

        dir_upper = direction.upper()
        min_runway_ratio = float(getattr(config, "MTF_MIN_RUNWAY_PCT", 0.004))
        if dir_upper == "LONG":
            if conf.get("veto_long", False):
                strat_rsn = str(conf.get("structural_rationale", ""))
                # If veto was merely a 15m counter-trend pullback within a bullish 4H trend, don't kill the swing entry
                if "15m" in strat_rsn and ("counter-trend" in strat_rsn or "pullback" in strat_rsn):
                    log.info(f"[RUNWAY-ADAPTIVE] 4H swing candidate {symbol} authorized despite 15m pullback: {strat_rsn}")
                else:
                    return False, f"MTF Confluence veto_long active: {strat_rsn}"

            key_barrier = conf.get("key_htf_barrier", 0.0)
            if key_barrier > current_price:
                dist_pct = (key_barrier - current_price) / current_price
                if dist_pct < min_runway_ratio:  # Require at least min_runway_ratio clearance for 4H swing entry
                    return False, f"Insufficient runway ({dist_pct*100:.2f}% < {min_runway_ratio*100:.1f}%) to HTF barrier ${key_barrier:,.2f}"
        else:
            if conf.get("veto_short", False):
                strat_rsn = str(conf.get("structural_rationale", ""))
                if "15m" in strat_rsn and ("counter-trend" in strat_rsn or "bounce" in strat_rsn):
                    log.info(f"[RUNWAY-ADAPTIVE] 4H swing breakdown {symbol} authorized despite 15m bounce: {strat_rsn}")
                else:
                    return False, f"MTF Confluence veto_short active: {strat_rsn}"

            key_floor = conf.get("key_htf_floor", conf.get("key_htf_support", 0.0))
            if key_floor > 0.0 and key_floor < current_price:
                dist_pct = (current_price - key_floor) / current_price
                if dist_pct < min_runway_ratio:  # Require at least min_runway_ratio clearance above major support floor
                    return False, f"Insufficient runway ({dist_pct*100:.2f}% < {min_runway_ratio*100:.1f}%) to HTF support ${key_floor:,.2f}"

        return True, "Runway clear."

    async def _check_intraday_15m_alignment(self, symbol: str, direction: str = "LONG") -> tuple[bool, str]:
        """
        Gate 4.4: 15m Intraday Trend & EMA20 Alignment Gate (Zero AI Cost — Fail-Closed).
        Prevents entering late 4H Kumo breakouts directly into an active 15m lower-high/lower-low cascade.
        """
        try:
            clean_sym = self.uta_client._normalize_symbol(symbol)
            candles_15m = await self.uta_client.exchange.fetch_ohlcv(clean_sym, timeframe="15m", limit=25)
            if not candles_15m or len(candles_15m) < 21:
                return False, "Insufficient 15m candles for intraday alignment check (fail-closed)."

            df15 = pd.DataFrame(candles_15m, columns=["ts", "open", "high", "low", "close", "vol"])
            close15 = df15["close"].astype(float)
            open15 = df15["open"].astype(float)
            ema20_15m = float(close15.ewm(span=20, adjust=False).mean().iloc[-1])

            curr_c = float(close15.iloc[-1])
            curr_o = float(open15.iloc[-1])
            prev_c = float(close15.iloc[-2])
            prev_o = float(open15.iloc[-2])

            is_long = direction.upper() == "LONG"
            if is_long:
                if curr_c < ema20_15m * 0.999:
                    return False, f"15m Price (${curr_c:,.4f}) below 15m EMA20 (${ema20_15m:,.4f}) — active intraday pullback in progress."
                if curr_c < curr_o and prev_c < prev_o and curr_c < prev_c:
                    return False, f"Two consecutive red 15m candles (${prev_c:,.4f} -> ${curr_c:,.4f}) — waiting for intraday stabilization."
            else:
                if curr_c > ema20_15m * 1.001:
                    return False, f"15m Price (${curr_c:,.4f}) above 15m EMA20 (${ema20_15m:,.4f}) — active intraday bounce in progress."
                if curr_c > curr_o and prev_c > prev_o and curr_c > prev_c:
                    return False, f"Two consecutive green 15m candles (${prev_c:,.4f} -> ${curr_c:,.4f}) — waiting for intraday rollover."

            return True, f"15m aligned with {direction.upper()} (Price=${curr_c:,.4f} vs EMA20=${ema20_15m:,.4f})."
        except Exception as e:
            log.warning(f"[15M-ALIGNMENT-VETO] {symbol} 15m alignment check error (fail-closed): {e}")
            return False, f"15m alignment check failed (fail-closed): {e}"

    async def _check_breakout_trap_gate(self, symbol: str, ind_4h: dict, current_price: float, direction: str = "LONG") -> tuple[bool, float, str]:
        """
        Feature 6: AI Breakout vs. Bull-Trap Gate (Zero-Lag Fakeout Shield — Deterministic + Fail-Closed).
        Fetches 5m candle wick metrics and queries Gemini/Groq to detect retail stop-run traps.
        """
        try:
            clean_sym = self.uta_client._normalize_symbol(symbol)
            candles = await self.uta_client.exchange.fetch_ohlcv(clean_sym, timeframe="5m", limit=10)
            if not candles or len(candles) < 3:
                return False, 1.0, "Insufficient 5m candles for trap check (fail-closed)."

            df5 = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "vol"])
            c_last = df5.iloc[-3:].copy()
            c_high = c_last["high"].values
            c_low = c_last["low"].values
            c_open = c_last["open"].values
            c_close = c_last["close"].values

            wick_tops = [(c_high[i] - max(c_open[i], c_close[i])) / (c_high[i] - c_low[i] + 1e-6) for i in range(len(c_last))]
            wick_bottoms = [(min(c_open[i], c_close[i]) - c_low[i]) / (c_high[i] - c_low[i] + 1e-6) for i in range(len(c_last))]
            avg_upper_wick = float(np.mean(wick_tops))
            avg_lower_wick = float(np.mean(wick_bottoms))

            is_long = direction.upper() == "LONG"
            # Deterministic hard wick shield before calling AI
            if is_long and avg_upper_wick >= 0.58:
                return False, avg_upper_wick, f"Deterministic 5m upper-wick exhaustion ({avg_upper_wick*100:.1f}% >= 58%)."
            if (not is_long) and avg_lower_wick >= 0.58:
                return False, avg_lower_wick, f"Deterministic 5m lower-wick absorption ({avg_lower_wick*100:.1f}% >= 58%)."

            trap_ctx = {
                "symbol": symbol,
                "direction": direction.upper(),
                "current_price": current_price,
                "cloud_top": ind_4h.get("cloud_top", current_price),
                "cloud_bottom": ind_4h.get("cloud_bottom", current_price),
                "adx_4h": round(ind_4h.get("adx", 20.0), 2),
                "rsi_4h": round(ind_4h.get("rsi", 50.0), 2),
                "avg_5m_upper_wick_ratio": round(avg_upper_wick, 3),
                "avg_5m_lower_wick_ratio": round(avg_lower_wick, 3),
                "last_5m_upper_wick_ratio": round(wick_tops[-1], 3),
                "recent_5m_candles": [
                    {"close": round(c_close[i], 4), "upper_wick_pct": round(wick_tops[i] * 100, 1)}
                    for i in range(len(c_last))
                ]
            }

            import ai_engine
            trap_res = await ai_engine.detect_breakout_trap(trap_ctx, timeout_sec=8.0)
            verdict = trap_res.get("verdict", "CONFIRMED_BREAKOUT")
            trap_score = float(trap_res.get("trap_risk_score", 0.35))
            rationale = trap_res.get("structural_rationale", "Standard breakout verification.")

            # Rule: If trap_risk_score >= 0.75 or verdict is VETO
            if trap_score >= 0.75 or verdict in ("VETO_BULL_TRAP", "VETO_BEAR_TRAP"):
                log.warning(f"[BREAKOUT-TRAP-GATE] 🪤 VETO for {symbol} ({direction}): {verdict} (Risk: {trap_score*100:.1f}%). {rationale}")
                return False, trap_score, f"AI Breakout Trap Gate VETO ({verdict}): {rationale}"

            log.info(f"[BREAKOUT-TRAP-GATE] ✅ CONFIRMED for {symbol}: Risk={trap_score*100:.1f}%. {rationale}")
            return True, trap_score, f"Breakout confirmed ({rationale})"
        except Exception as e:
            log.warning(f"[BREAKOUT-TRAP-GATE] Error during trap check for {symbol} (fail-closed): {e}")
            return False, 1.0, f"Trap gate failed (fail-closed): {e}"

    def _check_divergence_gate(self, symbol: str, df_4h: pd.DataFrame, direction: str = "LONG") -> tuple[bool, str]:
        """
        Gate 4.6: Automated Momentum Divergence Gate (go-stock Quantitative Upgrade).
        Identifies regular bearish/bullish divergence between price peaks/troughs and RSI/MACD.
        Vetoes LONG entries on Bearish Divergence (Bull-Trap / Exhaustion).
        Vetoes SHORT entries on Bullish Divergence (Bear-Trap / Exhaustion).
        """
        if not getattr(config, "SWING_DIVERGENCE_GATE_ENABLED", True):
            return True, "Divergence gate disabled in config."

        try:
            lookback = getattr(config, "SWING_DIVERGENCE_LOOKBACK_BARS", 14)
            div = detect_momentum_divergence(df_4h, lookback=lookback)
            is_long = (direction.upper() == "LONG")

            if is_long and div.get("bearish_divergence", False):
                rsn = f"4H Bearish Divergence detected ({div.get('details', '')}). Price pushing higher on deteriorating momentum (Bull-Trap risk)."
                log.warning(f"[DIVERGENCE-GATE] [VETO] for {symbol} LONG: {rsn}")
                return False, rsn
            elif not is_long and div.get("bullish_divergence", False):
                rsn = f"4H Bullish Divergence detected ({div.get('details', '')}). Price pushing lower on rising momentum (Bear-Trap risk)."
                log.warning(f"[DIVERGENCE-GATE] [VETO] for {symbol} SHORT: {rsn}")
                return False, rsn

            log.info(f"[DIVERGENCE-GATE] [PASS] for {symbol} {direction}: No adverse divergence ({div.get('details', '')}).")
            return True, "Divergence alignment confirmed."
        except Exception as e:
            log.warning(f"[DIVERGENCE-GATE] Error for {symbol} (fail-closed): {e}")
            return False, f"Divergence gate error (fail-closed): {e}"

    def _check_volume_profile_gate(self, symbol: str, df_4h: pd.DataFrame, current_price: float, direction: str = "LONG") -> tuple[bool, str]:
        """
        Gate 4.7: Volume Profile & Value Area Clearance Gate (go-stock Quantitative Upgrade).
        Computes Value Area (POC, VAH, VAL) over rolling 48 candles.
        Requires 4H breakouts to clear VAH into low-liquidity discovery zones rather than high-volume chop zones.
        """
        if not getattr(config, "SWING_VOLUME_PROFILE_GATE_ENABLED", True):
            return True, "Volume Profile gate disabled in config."

        try:
            lookback = getattr(config, "SWING_VOLUME_PROFILE_LOOKBACK", 48)
            va_pct = getattr(config, "SWING_VOLUME_PROFILE_VA_PCT", 0.50)
            vp = compute_volume_profile(df_4h, lookback=lookback, value_area_pct=va_pct)
            is_long = (direction.upper() == "LONG")

            vah = vp.get("vah", current_price)
            val = vp.get("val", current_price)
            poc = vp.get("poc", current_price)

            vah_tol = float(getattr(config, "SWING_VOLUME_PROFILE_VAH_TOLERANCE", 0.985))
            val_tol = 2.0 - vah_tol

            if is_long:
                # Require price to be at or near VAH (within tolerance) to capture high-pressure breakout builds
                if current_price < (vah * vah_tol):
                    rsn = f"Trapped inside {va_pct*100:.0f}% Value Area (${val:.4f} - ${vah:.4f}, POC: ${poc:.4f}, Close: ${current_price:.4f}). Breakout requires pressure above {vah*vah_tol:.4f}."
                    log.warning(f"[VOLUME-PROFILE-GATE] [VETO] for {symbol} LONG: {rsn}")
                    return False, rsn
            else:
                # For SHORT, require price to be at or below VAL
                if current_price > (val * val_tol):
                    rsn = f"Trapped inside {va_pct*100:.0f}% Value Area (${val:.4f} - ${vah:.4f}, POC: ${poc:.4f}, Close: ${current_price:.4f}). Breakdown requires clearance below {val*val_tol:.4f}."
                    log.warning(f"[VOLUME-PROFILE-GATE] [VETO] for {symbol} SHORT: {rsn}")
                    return False, rsn

            log.info(f"[VOLUME-PROFILE-GATE] [PASS] for {symbol} {direction}: Cleared Value Area ({vp.get('details', '')}).")
            return True, f"Volume profile clearance confirmed ({vp.get('details', '')})."
        except Exception as e:
            log.warning(f"[VOLUME-PROFILE-GATE] Error for {symbol} (fail-closed): {e}")
            return False, f"Volume profile gate error (fail-closed): {e}"

    def _check_td_sequential_gate(self, symbol: str, df_4h: pd.DataFrame, direction: str = "LONG") -> tuple[bool, str]:
        """
        Gate 4.8: Tom DeMark TD Sequential Exhaustion Gate (go-stock Quantitative Upgrade).
        Counts consecutive closes relative to close 4 bars prior.
        Vetoes LONG entries on TD-9 Upward Exhaustion Climax (Buying Climax / Distribution Top).
        Vetoes SHORT entries on TD-9 Downward Exhaustion Climax (Selling Climax / Capitulation Bottom).
        """
        if not getattr(config, "SWING_TD_SEQUENTIAL_GATE_ENABLED", True):
            return True, "TD Sequential gate disabled in config."

        try:
            td = compute_td_sequential(df_4h)
            is_long = (direction.upper() == "LONG")

            if is_long and td.get("is_td9_upward_exhaustion", False):
                rsn = f"4H TD Sequential Upward Exhaustion Climax reached (Count: {td.get('up_count', 9)}). High statistical probability of immediate mean-reverting pullback."
                log.warning(f"[TD-SEQUENTIAL-GATE] [VETO] for {symbol} LONG: {rsn}")
                return False, rsn
            elif not is_long and td.get("is_td9_downward_exhaustion", False):
                rsn = f"4H TD Sequential Downward Exhaustion Climax reached (Count: {td.get('down_count', 9)}). High statistical probability of immediate mean-reverting bounce."
                log.warning(f"[TD-SEQUENTIAL-GATE] [VETO] for {symbol} SHORT: {rsn}")
                return False, rsn

            log.info(f"[TD-SEQUENTIAL-GATE] [PASS] for {symbol} {direction}: Setup count healthy ({td.get('details', '')}).")
            return True, f"TD Sequential setup healthy ({td.get('details', '')})."
        except Exception as e:
            log.warning(f"[TD-SEQUENTIAL-GATE] Error for {symbol} (fail-closed): {e}")
            return False, f"TD Sequential gate error (fail-closed): {e}"

    async def _deliberate_boardroom(self, symbol: str, ind_4h: dict, ind_1d: dict, direction: str = "LONG") -> tuple[bool, float, str]:

        """Convenes Multi-Agent Conviction Boardroom for swing trade audit."""
        is_long = (direction.upper() == "LONG")
        side = "long" if is_long else "short"
        cloud_dist = (ind_4h["close"] - ind_4h["cloud_top"]) if is_long else (ind_4h["cloud_bottom"] - ind_4h["close"])
        is_1d_bull = ind_1d.get("bullish_alignment", False)
        is_1d_bear = ind_1d.get("bearish_alignment", False)
        daily_trend_str = ind_1d.get("daily_trend", "BULLISH" if is_1d_bull else ("BEARISH" if is_1d_bear else "NEUTRAL"))
        daily_close = ind_1d.get("close", ind_4h.get("close", 0.0))

        chikou_clear = ind_4h.get("chikou_bull_clear", True) if is_long else ind_4h.get("chikou_bear_clear", True)
        context = {
            "symbol": symbol,
            "current_price": round(ind_4h["close"], 4),
            "timeframe": "4h",
            "direction": direction.upper(),
            "adx": round(ind_4h["adx"], 2),
            "atr": round(ind_4h["atr"], 4),
            "rsi": round(ind_4h.get("rsi", 50.0), 2),
            "cloud_distance_pct": round(cloud_dist / ind_4h["close"] * 100, 2),
            "tenkan_sen": round(ind_4h.get("tenkan", 0.0), 4),
            "kijun_sen": round(ind_4h.get("kijun", 0.0), 4),
            "volume_ratio_vs_20ma": ind_4h.get("vol_ratio_20ma", 1.0),
            "chikou_span_clear": chikou_clear,
            "planned_sl_atr_mult": self.atr_sl_mult,
            "planned_tp_rr_mult": self.runner_tp_mult,
            "daily_close": round(daily_close, 4),
            "daily_trend": daily_trend_str,
            "daily_trend_confirmed_bullish": is_1d_bull,
            "daily_trend_confirmed_bearish": is_1d_bear,
            "daily_ema50": round(ind_1d["ema50"], 2),
            "daily_ema200": round(ind_1d["ema200"], 2),
            "strategy": f"Ichimoku 4H {'Breakout' if is_long else 'Breakdown'}"
        }

        try:
            res = await self.boardroom.get_conviction(
                symbol=symbol,
                side=side,
                context_data=context
            )
            execute = res.get("execute", False)
            score = res.get("conviction_score", 0.0)
            rationale = res.get("final_rationale", "No details")
            return execute, score, rationale
        except Exception as e:
            log.error(f"[BOARDROOM-ERROR] Boardroom deliberation failed for {symbol}: {e}")
            return False, 0.0, f"Boardroom crash: {e}"

    # ---------------------------------------------------------------------------
    # Order Execution & Lifecycle Management
    # ---------------------------------------------------------------------------
    async def open_swing_position(self, symbol: str, ind_4h: dict, direction: str = "LONG", conviction_score: float = 0.70, gates_snapshot: dict = None):
        """Executes market entry, acquires asset lock, and records position."""
        # P0-4 DEFENSE: Check if Portfolio Risk Agent or Operator has activated RISK_HALT / OPERATOR_HALT (offloaded via to_thread, Fail-Closed)
        def _read_risk_halt():
            with sqlite3.connect("trading_data.db", timeout=5.0) as rh_conn:
                rh_rows = rh_conn.execute(
                    "SELECT value FROM system_state WHERE key IN ('RISK_HALT', 'OPERATOR_HALT')"
                ).fetchall()
                return any(r and str(r[0]).strip().upper() in ("1", "TRUE") for r in rh_rows)

        try:
            if await asyncio.to_thread(_read_risk_halt):
                log.warning(f"[SWING-VETO] RISK_HALT / OPERATOR_HALT active in system_state! Blocking new swing position for {symbol}.")
                return
        except Exception as rhe:
            log.warning(f"[SWING-RISK-HALT-VETO] Failed to read RISK_HALT state (fail-closed) for {symbol}: {rhe}")
            return

        # P0-5 DEFENSE: Check Pre-Trade Microstructure Guard (VPIN Toxicity & Hawkes Spoof Veto)
        try:
            from microstructure_guard import MicrostructureStateCache
            clean_s = symbol.replace("/", "").upper()
            ms_state = MicrostructureStateCache.get_state(clean_s)
            vpin_val = float(ms_state.get("vpin", 0.0) or 0.0)
            h_spoof = float(ms_state.get("hawkes_spoof_index", 0.0) or 0.0)
            h_rho = float(ms_state.get("hawkes_rho", 0.0) or 0.0)
            h_veto = bool(ms_state.get("hawkes_veto", False))

            if vpin_val >= 0.70:
                log.warning(f"[SWING-MICROSTRUCTURE-VETO] {symbol} VPIN={vpin_val:.2f} >= 0.70 (Informed Toxicity Cascade). Blocking entry.")
                return

            h_spoof_max = float(getattr(config, "HAWKES_SPOOF_VETO_THRESHOLD", 0.78))
            h_rho_max = float(getattr(config, "HAWKES_CASCADE_RHO_THRESHOLD", 1.05))
            if h_veto or h_spoof > h_spoof_max or h_rho >= h_rho_max:
                log.warning(f"[SWING-MICROSTRUCTURE-VETO] {symbol} Hawkes Spoof={h_spoof:.2f} (rho={h_rho:.2f}). Predatory phantom liquidity detected. Blocking entry.")
                return
        except Exception as ms_err:
            log.warning(f"[SWING-MICROSTRUCTURE-VETO] Microstructure guard error (fail-closed) for {symbol}: {ms_err}")
            return

        # Live price discovery to ensure order execution precision
        try:
            clean_sym = self.uta_client._normalize_symbol(symbol)
            ticker = await self.uta_client.exchange.fetch_ticker(clean_sym)
            entry_price = float(ticker.get("last", 0.0) or ticker.get("close", 0.0) or ind_4h["close"])
        except Exception:
            entry_price = ind_4h["close"]
        atr = ind_4h["atr"]
        adx = float(ind_4h.get("adx", 0.0))
        rsi = float(ind_4h.get("rsi", 50.0))
        is_long = direction.upper() == "LONG"

        # Enforce Fleet-Wide Long Cap: Max 2 Active Longs across all engines
        if is_long:
            fleet_longs = await asyncio.to_thread(self._count_fleet_long_positions_sync)
            max_fleet_longs = getattr(config, "MAX_FLEET_LONG_POSITIONS", 2)
            if fleet_longs >= max_fleet_longs:
                log.warning(f"[SWING-LONG-CAP-VETO] Active fleet longs ({fleet_longs}) >= cap ({max_fleet_longs}). Blocking new long entry for {symbol}.")
                return

        # Dynamic Tiered Leverage Protocol:
        # 5x for Very Good / Premium Quality setups:
        #   - Boardroom Conviction >= SWING_HIGH_QUALITY_CONVICTION_MIN (default 0.80)
        #   - ADX >= SWING_HIGH_QUALITY_ADX_MIN (default 30.0, strong momentum)
        #   - RSI in optimal non-exhausted trajectory (40 <= RSI <= 65 for LONG, 35 <= RSI <= 60 for SHORT)
        # 3x for Standard / Lower Quality setups:
        #   - Valid setups with conviction 0.65-0.79 or ADX 25.0-29.9
        is_optimal_rsi = (40.0 <= rsi <= 65.0) if is_long else (35.0 <= rsi <= 60.0)
        is_high_quality = (
            conviction_score >= self.high_quality_conviction and
            adx >= self.high_quality_adx and
            is_optimal_rsi
        )

        assigned_leverage = self.high_quality_leverage if is_high_quality else self.standard_leverage
        quality_tier = "PREMIUM_5X" if is_high_quality else "STANDARD_3X"

        # Rule #165 & #166: Enforce minimum stop-loss distance floor (1.5%) to survive intraday churn
        min_sl_dist = entry_price * getattr(config, "SWING_MIN_SL_PCT", 0.015)
        sl_distance = max(self.atr_sl_mult * atr, min_sl_dist)

        if is_long:
            stop_loss = round(entry_price - sl_distance, 4)
            risk_per_unit = entry_price - stop_loss
            if risk_per_unit <= 0:
                return
            take_profit = round(entry_price + (self.runner_tp_mult * risk_per_unit), 4)
        else:
            stop_loss = round(entry_price + sl_distance, 4)
            risk_per_unit = stop_loss - entry_price
            if risk_per_unit <= 0:
                return
            take_profit = round(entry_price - (self.runner_tp_mult * risk_per_unit), 4)

        # Feature 3: Dynamic AI Exit Geometry (Structure-Calibrated SL & TP)
        # Enhances fixed ATR with structural anchors (FVGs, order blocks, key swing pivots)
        ai_geometry_adopted = False
        try:
            import ai_engine
            geom_ctx = {
                "symbol": symbol,
                "side": "LONG" if is_long else "SHORT",
                "entry_price": entry_price,
                "atr": atr,
                "adx": adx,
                "rsi": rsi,
                "baseline_sl": stop_loss,
                "baseline_tp": take_profit,
                "cloud_top": float(ind_4h.get("cloud_top", 0.0)),
                "cloud_bottom": float(ind_4h.get("cloud_bottom", 0.0)),
                "kijun_sen": float(ind_4h.get("kijun", 0.0))
            }
            geom_res = await ai_engine.tune_exit_geometry(geom_ctx, timeout_sec=8.0)
            ai_sl = float(geom_res.get("suggested_sl") or 0.0)
            ai_tp = float(geom_res.get("suggested_tp") or 0.0)
            conf = float(geom_res.get("confidence", 0.0))
            if ai_sl > 0.0 and ai_tp > 0.0 and conf >= 0.70:
                ai_dist = abs(entry_price - ai_sl)
                if ai_dist >= min_sl_dist:
                    reward = abs(ai_tp - entry_price)
                    if reward >= 1.8 * ai_dist:
                        log.info(f"[EXIT-GEOMETRY] 📐 AI Structure-Calibrated SL/TP adopted for {symbol}: SL ${stop_loss} -> ${ai_sl} ({geom_res.get('structural_anchor_sl')}) | TP ${take_profit} -> ${ai_tp} ({geom_res.get('structural_anchor_tp')}) | R:R={reward/ai_dist:.2f}R")
                        stop_loss = round(ai_sl, 4)
                        take_profit = round(ai_tp, 4)
                        risk_per_unit = ai_dist
                        sl_distance = ai_dist
                        ai_geometry_adopted = True
        except Exception as geom_err:
            log.debug(f"[EXIT-GEOMETRY-NOTE] {symbol}: {geom_err}")

        base_stop_loss = stop_loss
        base_take_profit = take_profit

        # Dynamic Balance Sizing: 70% Wallet Utilization Protocol
        # Allows the fleet to deploy up to 70% of total wallet equity across concurrent positions (30% cash buffer).
        max_wallet_margin_pct = getattr(config, "GLOBAL_MARGIN_LIMIT", 70.0) / 100.0  # 0.70
        margin_collateral = self.allocation_usdt
        try:
            live_bal = await self.uta_client.fetch_balance()
            if live_bal and live_bal > 10.0:
                # Calculate currently locked margin across active positions
                total_locked_margin = sum(
                    (float(p.get("size", 0.0)) * float(p.get("entry_price", 0.0))) / max(1.0, float(p.get("leverage", 3.0)))
                    for p in self.active_positions.values()
                )
                max_allowed_margin = live_bal * max_wallet_margin_pct
                available_margin_buffer = max(0.0, max_allowed_margin - total_locked_margin)

                if available_margin_buffer < 5.0:
                    log.warning(f"[MARGIN-CEILING] 70% Wallet Utilization Ceiling reached! Locked: ${total_locked_margin:.2f} / Max: ${max_allowed_margin:.2f} ({total_locked_margin/live_bal*100:.1f}%). Vetoing entry.")
                    return

                # Target fair share (e.g. 70% / 4 locks = 17.5% of wallet), bounded by remaining 70% headroom
                target_per_trade = live_bal * (max_wallet_margin_pct / max(1, self.max_locks))
                
                # Empirical Growth-Optimal Kelly Sizing + FinRL Rolling Sharpe Ensemble Weight
                try:
                    from risk_manager import RiskManager
                    rm = RiskManager()
                    kelly_frac = await asyncio.to_thread(rm.calculate_empirical_kelly_fraction)
                    conv_mult = max(0.6, min(1.2, float(conviction_score or 0.7)))
                    finrl_mult = 1.0
                    try:
                        rep_path = os.path.join("cache", "quant_validation_report.json")
                        if os.path.exists(rep_path):
                            with open(rep_path, "r", encoding="utf-8") as rf:
                                rep_data = json.load(rf)
                            sw = ((rep_data.get("finrl_ensemble_selector") or {}).get("strategy_weights") or {}).get("SWING_4H") or {}
                            finrl_mult = max(0.75, min(1.25, float(sw.get("allocation_weight_mult", 1.0))))
                    except Exception:
                        pass
                    target_kelly_margin = live_bal * kelly_frac * conv_mult * finrl_mult
                    target_per_trade = max(target_per_trade * 0.7, min(target_per_trade * 1.35, target_kelly_margin))
                    log.info(f"[SWING-KELLY-SIZING] Empirical Kelly: {kelly_frac:.2%} | FinRL Sharpe Mult: {finrl_mult:.2f}x -> Target Margin: ${target_per_trade:.2f}")
                except Exception as ke:
                    log.debug(f"[SWING-KELLY-NOTE] {ke}")

                margin_collateral = max(6.0, min(round(target_per_trade, 2), round(available_margin_buffer, 2)))

                # Hard Per-Trade Dollar Risk Cap (default 2.0% of account equity)
                sl_pct_actual = abs(entry_price - stop_loss) / entry_price if entry_price > 0 else 0.015
                max_risk_pct = float(getattr(config, "RISK_PER_TRADE_PCT", 2.0)) / 100.0
                if sl_pct_actual > 0 and assigned_leverage > 0:
                    max_risk_usd = live_bal * max_risk_pct
                    max_margin_by_risk = max_risk_usd / (assigned_leverage * sl_pct_actual)
                    if margin_collateral > max_margin_by_risk:
                        log.info(
                            f"[RISK-CAP] Clamping {symbol} margin ${margin_collateral:.2f} -> ${max_margin_by_risk:.2f} "
                            f"to enforce {max_risk_pct*100:.1f}% max equity risk (${max_risk_usd:.2f} on {sl_pct_actual*100:.2f}% SL at {assigned_leverage}x)."
                        )
                        margin_collateral = max(6.0, round(max_margin_by_risk, 2))
        except Exception as be:
            log.warning(f"[BALANCE-NOTE] Using fallback allocation (${margin_collateral:,.2f} USDT): {be}")

        # Apply Sentinel WARNING size-down multiplier (e.g. 0.5x on caution news)
        news_mult = float(self._news_size_multipliers.get(symbol, 1.0))
        if 0.0 < news_mult < 1.0:
            prev_margin = margin_collateral
            margin_collateral = max(6.0, round(margin_collateral * news_mult, 2))
            log.warning(f"[SENTINEL-SIZE-DOWN] {symbol} WARNING tier active -> scaling margin ${prev_margin:.2f} -> ${margin_collateral:.2f} ({news_mult:.2f}x)")

        target_notional = margin_collateral * assigned_leverage
        qty = target_notional / entry_price

        # Format minimum order qty per asset to satisfy Bitget contract minimums
        if "BTC" in symbol:
            qty = max(round(qty, 4), 0.0001)
        elif "ETH" in symbol or "BNB" in symbol:
            qty = max(round(qty, 2), 0.01)
        elif "SOL" in symbol:
            qty = max(round(qty, 1), 0.1)
        else:
            qty = max(float(round(qty)), 1.0)

        # 1. Acquire exclusive lock FIRST before placing order
        lock_reason = f"4H_SWING_{direction.upper()}_{quality_tier}"
        lock_acquired = await asyncio.to_thread(self._acquire_lock_sync, symbol, lock_reason)
        if not lock_acquired:
            log.warning(f"[ABORT] Cannot open {symbol} swing trade: Lock acquisition failed.")
            return

        tp_dist_pct = abs(take_profit - entry_price) / entry_price * 100.0 if entry_price > 0 else 0.0
        log.info("=" * 60)
        log.info(f"   EXECUTING 4H SWING {direction.upper()} [{quality_tier}]: {symbol}")
        log.info(f"   Entry Price: ${entry_price:,.4f} | Margin: ${margin_collateral:,.2f} USDT | Lev: {assigned_leverage}x")
        log.info(f"   Quality Metrics: Conviction={conviction_score:.2f} | ADX={adx:.1f} | RSI={rsi:.1f}")
        log.info(f"   Notional Size: ${target_notional:,.2f} ({qty} units)")
        log.info(f"   Initial SL: ${stop_loss:,.4f} ({'+' if not is_long else '-'}{sl_distance/entry_price*100:.2f}%)")
        log.info(f"   Target TP:  ${take_profit:,.4f} ({'-' if not is_long else '+'}{tp_dist_pct:.2f}%)")
        log.info("=" * 60)

        order_success = False
        side_str = "long" if is_long else "short"
        actual_entry_price = entry_price
        execution_order_type = "MARKET"
        exec_mode = "RUNAWAY_IMPULSE"

        # Feature 2: Pre-Entry Retest & Limit Order Sniper
        # Evaluates 5m/15m wick rejection and L2 depth to avoid buying wick tops;
        # snipes entries 0.3%–0.6% lower on EMA/Kijun pullbacks via post-only maker orders.
        try:
            entry_ctx = {
                "symbol": symbol,
                "direction": direction,
                "current_price": entry_price,
                "adx": adx,
                "rsi": rsi,
                "atr": atr,
                "kijun_price": ind_4h.get("kijun", entry_price),
                "cloud_top": ind_4h.get("cloud_top", entry_price),
                "cloud_bottom": ind_4h.get("cloud_bottom", entry_price),
                "conviction_score": conviction_score
            }
            import ai_engine
            sniper_rec = await ai_engine.optimize_entry_timing(entry_ctx, timeout_sec=8.0)
            exec_mode = sniper_rec.get("execution_mode", "RUNAWAY_IMPULSE")
            rec_entry = float(sniper_rec.get("recommended_entry_price") or entry_price)
            discount_pct = float(sniper_rec.get("price_discount_pct", 0.0))
            rationale = sniper_rec.get("rationale", "Standard algorithmic execution.")

            log.info(f"[ENTRY-SNIPER] {symbol} Execution Mode: {exec_mode} | Rec Price: ${rec_entry:,.4f} | Discount: {discount_pct:.2f}% | Rationale: {rationale}")

            # If Retest Pullback recommended with a discount <= 1.5% for LONG or SHORT:
            if exec_mode == "RETEST_PULLBACK":
                if is_long and rec_entry < entry_price:
                    discount_diff = (entry_price - rec_entry) / entry_price
                    if discount_diff <= 0.015:
                        actual_entry_price = rec_entry
                        execution_order_type = "POST_ONLY_LIMIT"
                        log.info(f"[ENTRY-SNIPER] Attempting Maker Post-Only Retest Snipe on {symbol} (LONG) at ${actual_entry_price:,.4f} (-{discount_diff*100:.2f}% rebate discount)...")
                elif (not is_long) and rec_entry > entry_price:
                    discount_diff = (rec_entry - entry_price) / entry_price
                    if discount_diff <= 0.015:
                        actual_entry_price = rec_entry
                        execution_order_type = "POST_ONLY_LIMIT"
                        log.info(f"[ENTRY-SNIPER] Attempting Maker Post-Only Retest Snipe on {symbol} (SHORT) at ${actual_entry_price:,.4f} (+{discount_diff*100:.2f}% rebate discount)...")
        except Exception as sniper_err:
            log.warning(f"[ENTRY-SNIPER-NOTE] Sniper optimization fallback to market: {sniper_err}")

        try:
            if execution_order_type == "POST_ONLY_LIMIT":
                try:
                    # Preserve AI structure-calibrated SL/TP if adopted and valid relative to limit price;
                    # otherwise recalculate from ATR relative to discounted entry price.
                    valid_ai_levels = (
                        ai_geometry_adopted and
                        ((is_long and base_stop_loss < actual_entry_price < base_take_profit) or
                         (not is_long and base_take_profit < actual_entry_price < base_stop_loss))
                    )
                    if valid_ai_levels:
                        stop_loss = base_stop_loss
                        take_profit = base_take_profit
                        risk_per_unit = abs(actual_entry_price - stop_loss)
                    else:
                        sl_dist_sniper = max(self.atr_sl_mult * atr, actual_entry_price * getattr(config, "SWING_MIN_SL_PCT", 0.015))
                        stop_loss = round(actual_entry_price - sl_dist_sniper, 4) if is_long else round(actual_entry_price + sl_dist_sniper, 4)
                        risk_per_unit = abs(actual_entry_price - stop_loss)
                        take_profit = round(actual_entry_price + (self.runner_tp_mult * risk_per_unit), 4) if is_long else round(actual_entry_price - (self.runner_tp_mult * risk_per_unit), 4)

                    po_res = await self.uta_client.create_post_only_order(
                        symbol=symbol,
                        side=side_str,
                        qty=qty,
                        price=actual_entry_price,
                        sl=stop_loss,
                        tp=take_profit,
                        leverage_val=assigned_leverage
                    )
                    po_order_id = po_res.get("orderId") or po_res.get("id") if po_res else None
                    if po_order_id:
                        sniper_timeout = getattr(config, "SWING_MAKER_SNIPER_TIMEOUT_SECS", 180)
                        log.info(f"[ENTRY-SNIPER-PLACED] Maker Post-Only order {po_order_id} placed for {symbol} at ${actual_entry_price:,.4f}. Monitoring fill queue (TTL: {sniper_timeout}s)...")
                        
                        poll_interval = 2.0
                        start_time = time.time()
                        is_filled = False
                        
                        while (time.time() - start_time) < sniper_timeout:
                            await asyncio.sleep(poll_interval)
                            try:
                                o_info = await self.uta_client.fetch_order(symbol, po_order_id)
                                o_status = str(o_info.get("order_status") or o_info.get("status") or o_info.get("state") or "").lower()
                                filled_amount = float(o_info.get("filled_qty") or o_info.get("cumExecQty") or 0.0)
                                
                                if o_status in ("filled", "closed") or (qty > 0 and filled_amount >= qty * 0.95):
                                    is_filled = True
                                    deal_p = float(o_info.get("deal_avg_price") or o_info.get("avgPrice") or 0.0)
                                    if deal_p > 0:
                                        actual_entry_price = deal_p
                                    log.info(f"[ENTRY-SNIPER-FILLED] Maker limit order {po_order_id} FILLED on {symbol}! Qty: {filled_amount} @ ${actual_entry_price:,.4f}. Maker fee rebate captured!")
                                    break
                                elif o_status in ("canceled", "cancelled"):
                                    log.warning(f"[ENTRY-SNIPER-CANCELED] Limit order {po_order_id} was cancelled by exchange.")
                                    break
                            except Exception as poll_err:
                                log.debug(f"[ENTRY-SNIPER-POLL] Polling note on {po_order_id}: {poll_err}")
                        
                        if not is_filled:
                            log.info(f"[ENTRY-SNIPER-TIMEOUT] Limit order {po_order_id} for {symbol} unfilled after {sniper_timeout}s. Cancelling on Bitget matching engine...")
                            try:
                                await self.uta_client.cancel_order(symbol, po_order_id)
                            except Exception as c_err:
                                log.warning(f"[ENTRY-SNIPER-CANCEL] Cancel note: {c_err}")
                            
                            # Final state check to avoid race-condition fill at boundary
                            try:
                                final_o = await self.uta_client.fetch_order(symbol, po_order_id)
                                final_st = str(final_o.get("order_status") or final_o.get("status") or "").lower()
                                final_fill = float(final_o.get("filled_qty") or final_o.get("cumExecQty") or 0.0)
                                if final_st in ("filled", "closed") or (qty > 0 and final_fill >= qty * 0.95):
                                    is_filled = True
                                    deal_p = float(final_o.get("deal_avg_price") or final_o.get("avgPrice") or 0.0)
                                    if deal_p > 0:
                                        actual_entry_price = deal_p
                                    log.info(f"[ENTRY-SNIPER-FILLED] Order {po_order_id} filled right at cancel boundary!")
                            except Exception:
                                pass
                        
                        if is_filled:
                            order_success = True
                            # Clean quantity and prices for Bitget exchange precision
                            actual_qty = filled_amount if (filled_amount > 0 and filled_amount <= qty) else qty
                            qty_str = self.uta_client.format_quantity_precision(symbol, actual_qty) if hasattr(self.uta_client, "format_quantity_precision") else str(actual_qty)
                            opp_side = "sell" if is_long else "buy"
                            sl_str = self.uta_client.format_price_precision(symbol, stop_loss) if hasattr(self.uta_client, "format_price_precision") else str(round(stop_loss, 4) if stop_loss < 10 else round(stop_loss, 2))
                            
                            # Register native strategy stop loss on Bitget with 3-attempt backoff retry
                            sl_attached = False
                            last_sl_err = None
                            for sl_attempt in range(3):
                                try:
                                    sl_p = {
                                        "category": "USDT-FUTURES",
                                        "symbol": self.uta_client._normalize_symbol(symbol),
                                        "marginMode": getattr(config, "MARGIN_MODE", "isolated").lower(),
                                        "marginCoin": "USDT",
                                        "planType": "pos_loss",
                                        "posSide": "long" if is_long else "short",
                                        "side": opp_side,
                                        "triggerPrice": sl_str,
                                        "stopLoss": sl_str,
                                        "triggerType": "mark_price",
                                        "size": qty_str
                                    }
                                    await self.uta_client.exchange.private_uta_post_v3_trade_place_strategy_order(sl_p)
                                    log.info(f"[ENTRY-SNIPER-SHIELD] Native Stop Loss REGISTERED on Bitget! (SL=${stop_loss:,.4f})")
                                    sl_attached = True
                                    break
                                except Exception as sl_reg_err:
                                    last_sl_err = sl_reg_err
                                    log.warning(f"[ENTRY-SNIPER] Native SL registration attempt {sl_attempt+1}/3 failed for {symbol}: {sl_reg_err}")
                                    await asyncio.sleep(0.5 * (sl_attempt + 1))

                            if not sl_attached:
                                log.critical(f"[ENTRY-SNIPER] 🚨 CRITICAL NAKED STOP FAILURE: Failed to attach native SL on {symbol} after 3 attempts ({last_sl_err})! Aborting position to eliminate naked risk.")
                                try:
                                    await self.uta_client.close_position_order(symbol=symbol, side="long" if is_long else "short", qty=actual_qty)
                                    log.warning(f"[ENTRY-SNIPER] Emergency position abort SUCCESS for {symbol}.")
                                except Exception as abort_err:
                                    log.error(f"[ENTRY-SNIPER] Emergency position abort FAILED for {symbol}: {abort_err}")
                                await asyncio.to_thread(self._release_lock_sync, symbol)
                                raise RuntimeError(f"Native Stop Loss placement failed on sniper order for {symbol} after 3 retries ({last_sl_err}). Position aborted.")

                            # Register native Take Profit
                            if take_profit and take_profit > 0.0:
                                try:
                                    tp_str = self.uta_client.format_price_precision(symbol, take_profit) if hasattr(self.uta_client, "format_price_precision") else str(round(take_profit, 4) if take_profit < 10 else round(take_profit, 2))
                                    tp_p = {
                                        "category": "USDT-FUTURES",
                                        "symbol": self.uta_client._normalize_symbol(symbol),
                                        "marginMode": getattr(config, "MARGIN_MODE", "isolated").lower(),
                                        "marginCoin": "USDT",
                                        "planType": "pos_profit",
                                        "posSide": "long" if is_long else "short",
                                        "side": opp_side,
                                        "triggerPrice": tp_str,
                                        "takeProfit": tp_str,
                                        "triggerType": "mark_price",
                                        "size": qty_str
                                    }
                                    await self.uta_client.exchange.private_uta_post_v3_trade_place_strategy_order(tp_p)
                                    log.info(f"[ENTRY-SNIPER-TARGET] Native Take Profit REGISTERED on Bitget! (TP=${take_profit:,.4f})")
                                except Exception as tp_reg_err:
                                    log.warning(f"[ENTRY-SNIPER] Native TP registration note: {tp_reg_err}")
                        else:
                            # Do NOT market-chase the top when AI Sniper explicitly mandated RETEST_PULLBACK due to exhaustion!
                            curr_mkt_p = entry_price
                            try:
                                clean_sym_chk = self.uta_client._normalize_symbol(symbol)
                                t_chk = await self.uta_client.exchange.fetch_ticker(clean_sym_chk)
                                curr_mkt_p = float(t_chk.get("last", 0.0) or t_chk.get("close", 0.0) or entry_price)
                            except Exception:
                                curr_mkt_p = entry_price

                            near_pullback_zone = (
                                (is_long and curr_mkt_p <= actual_entry_price * 1.0010) or
                                ((not is_long) and curr_mkt_p >= actual_entry_price * 0.9990)
                            )
                            if near_pullback_zone:
                                log.info(
                                    f"[ENTRY-SNIPER-FALLBACK] Live price ${curr_mkt_p:,.4f} reached within 0.10% of pullback target "
                                    f"${actual_entry_price:,.4f}. Executing market fill at pullback zone."
                                )
                                execution_order_type = "MARKET"
                                entry_price = curr_mkt_p
                                stop_loss = base_stop_loss
                                take_profit = base_take_profit
                                risk_per_unit = abs(entry_price - stop_loss)
                            else:
                                abort_cd_hours = float(getattr(config, "SWING_ABORT_COOLDOWN_HOURS", 0.35))
                                log.info(
                                    f"[ENTRY-SNIPER-NO-CHASE] 🛡️ AI Sniper mandated RETEST_PULLBACK at ${actual_entry_price:,.4f} "
                                    f"and live price (${curr_mkt_p:,.4f}) never reached the pullback zone. "
                                    f"Refusing to market-buy the top; aborting cleanly ({abort_cd_hours*60:.0f}m cooldown)."
                                )
                                await asyncio.to_thread(self._release_lock_sync, symbol)
                                await asyncio.to_thread(
                                    self._set_cooldown_sync,
                                    symbol,
                                    abort_cd_hours,
                                    f"{direction.upper()}_SNIPER_UNFILLED_ABORT",
                                    False,
                                )
                                return
                except Exception as po_err:
                    if exec_mode == "RETEST_PULLBACK":
                        abort_cd_hours = float(getattr(config, "SWING_ABORT_COOLDOWN_HOURS", 0.35))
                        log.warning(
                            f"[ENTRY-SNIPER-FAIL-CLOSED] Post-only limit order failed ({po_err}) during RETEST_PULLBACK. "
                            f"Refusing to market-chase top; releasing lock and setting {abort_cd_hours*60:.0f}m cooldown."
                        )
                        await asyncio.to_thread(self._release_lock_sync, symbol)
                        await asyncio.to_thread(
                            self._set_cooldown_sync,
                            symbol,
                            abort_cd_hours,
                            f"{direction.upper()}_SNIPER_ERROR_ABORT",
                            False,
                        )
                        return
                    log.warning(f"[ENTRY-SNIPER-ERROR] Post-only limit handling failed ({po_err}). Re-routing RUNAWAY_IMPULSE to taker market order.")
                    execution_order_type = "MARKET"
                    stop_loss = base_stop_loss
                    take_profit = base_take_profit
                    risk_per_unit = abs(entry_price - stop_loss)

            if not order_success:
                # Route live market order to Bitget UTA with assigned tiered leverage
                res = await self.uta_client.create_market_order(
                    symbol=symbol,
                    side=side_str,
                    qty=qty,
                    sl=stop_loss,
                    tp=take_profit,
                    leverage_val=assigned_leverage
                )
                order_success = bool(res and (res.get("orderId") or res.get("id")))
                actual_entry_price = entry_price
        except Exception as oe:
            log.error(f"[EXECUTION-ERROR] Failed to execute live order on Bitget: {oe}")

        if not order_success:
            log.error(f"[ROLLBACK] Order placement failed on Bitget. Releasing lock for {symbol}.")
            await asyncio.to_thread(self._release_lock_sync, symbol)
            return

        # 2. Record active position in database
        now_iso = datetime.now(UTC).isoformat()
        order_tag = "POST_ONLY_MAKER" if execution_order_type == "POST_ONLY_LIMIT" else "MARKET_TAKER"
        final_size = actual_qty if (execution_order_type == "POST_ONLY_LIMIT" and 'actual_qty' in locals()) else qty
        intended_price = float(entry_price)
        actual_price = float(actual_entry_price)
        slippage_bps = ((actual_price - intended_price) / intended_price * 10000.0) if intended_price > 0 else 0.0
        exec_type = "maker" if execution_order_type == "POST_ONLY_LIMIT" else "taker"
        entry_fee_rate = 0.0002 if exec_type == "maker" else 0.0006
        fee_entry_usd = actual_price * final_size * entry_fee_rate
        gates_json = json.dumps(gates_snapshot) if gates_snapshot else None

        pos_record = {
            "symbol": symbol,
            "side": side_str,
            "entry_price": actual_price,
            "size": final_size,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "trailing_stop": stop_loss,
            "peak_price": actual_price,
            "leverage": assigned_leverage,
            "opened_at": now_iso,
            "updated_at": now_iso,
            "status": "ACTIVE",
            "r_multiple": 0.0,
            "initial_risk_usd": risk_per_unit * final_size,
            "engine_type": "SWING_4H",
            "order_type": "post_only" if execution_order_type == "POST_ONLY_LIMIT" else "market",
            "execution_type": exec_type,
            "intended_entry_price": intended_price,
            "actual_fill_price": actual_price,
            "slippage_bps": slippage_bps,
            "fee_entry_usd": fee_entry_usd,
            "gates_snapshot": gates_json,
            "notes": (
                f"Ichimoku 4H {'Breakout' if is_long else 'Breakdown'} "
                f"[{quality_tier} (Conviction: {conviction_score:.2f}, ADX: {adx:.1f})] "
                f"[Order: {order_tag} | Slippage: {slippage_bps:+.1f}bps | Engine: SWING_4H]"
            ),
        }
        self.active_positions[symbol] = pos_record

        def _save_pos():
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    with sqlite3.connect("trading_data.db", timeout=15.0) as conn:
                        conn.execute("PRAGMA journal_mode=WAL;")
                        conn.execute("PRAGMA busy_timeout=15000;")
                        conn.execute("""
                            INSERT OR REPLACE INTO swing_open_positions
                            (symbol, side, entry_price, size, stop_loss, take_profit, trailing_stop, peak_price,
                             leverage, opened_at, updated_at, status, r_multiple, initial_risk_usd, notes,
                             intended_entry_price, actual_fill_price, slippage_bps, fee_entry_usd, execution_type, gates_snapshot)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            pos_record["symbol"], pos_record["side"], pos_record["entry_price"], pos_record["size"],
                            pos_record["stop_loss"], pos_record["take_profit"], pos_record["trailing_stop"],
                            pos_record["peak_price"], pos_record["leverage"], pos_record["opened_at"],
                            pos_record["updated_at"], pos_record["status"], pos_record["r_multiple"],
                            pos_record["initial_risk_usd"], pos_record["notes"],
                            pos_record["intended_entry_price"], pos_record["actual_fill_price"],
                            pos_record["slippage_bps"], pos_record["fee_entry_usd"],
                            pos_record["execution_type"], pos_record["gates_snapshot"]
                        ))
                        conn.commit()
                        return
                except sqlite3.OperationalError as e:
                    if "locked" in str(e).lower() or "busy" in str(e).lower():
                        time.sleep(0.5)
                    else:
                        log.error(f"[DB-SAVE-ERROR] Operational error saving swing position {symbol}: {e}")
                        break
                except Exception as e:
                    log.error(f"[DB-SAVE-ERROR] Unexpected error saving swing position {symbol}: {e}")
                    break

            # Emergency persistence fallback so exchange position is never orphaned
            try:
                emerg_file = "emergency_positions.json"
                existing = {}
                if os.path.exists(emerg_file):
                    with open(emerg_file, "r") as f:
                        existing = json.load(f)
                existing[symbol] = pos_record
                with open(emerg_file, "w") as f:
                    json.dump(existing, f, indent=2)
                log.critical(f"[EMERGENCY-SAVE] Position {symbol} written to emergency_positions.json due to SQLite failure!")
            except Exception as ef:
                log.critical(f"[FATAL-PERSISTENCE] Failed to write emergency file for {symbol}: {ef}")

        await asyncio.to_thread(_save_pos)
        log.info(f"[POSITION-OPENED] {symbol} ({direction.upper()}) registered in swing_open_positions [{quality_tier} | {order_tag}]. Mutex lock active.")

        # 3. Dispatch Email Alert for Trade Open
        try:
            from email_dispatcher import send_trade_log_async
            trade_ctx = {
                "engine": f"The Swing Architect (4H) [{quality_tier}]",
                "side": direction.upper(),
                "qty": qty,
                "entry_price": actual_entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "leverage": assigned_leverage,
                "conviction": f"{round(conviction_score * 10)}/10",
                "notes": f"4H Ichimoku Kumo {'Breakout' if is_long else 'Breakdown'} [{quality_tier} | {order_tag}] (Risk: ${pos_record['initial_risk_usd']:.2f})",
                "ai_reasoning": f"4H Ichimoku Kumo {'Breakout' if is_long else 'Breakdown'} [{quality_tier}] with Conviction={conviction_score:.2f}, ADX={adx:.1f}, assigned {assigned_leverage}x leverage."
            }
            self._track_task(send_trade_log_async(symbol=symbol, side=direction.upper(), price=actual_entry_price, status="OPENED", trade_context=trade_ctx), "open_trade_alert")
        except Exception as mail_err:
            log.warning(f"[EMAIL-ALERT-NOTE] Trade open email note: {mail_err}")

    @staticmethod
    def _classify_position_engine(pos: dict) -> str:
        """Explicitly classifies a shared swing_open_positions row into SWING_4H, GRID_MAKER, or SCALP_SHORT."""
        explicit = str(pos.get("engine_type") or "").upper()
        if explicit in ("SWING_4H", "GRID_MAKER", "SCALP_SHORT"):
            return explicit
        notes_str = str(pos.get("notes", "")).lower()
        if ("scalp_short" in notes_str) or ("scalp short" in notes_str) or ("short scalp" in notes_str):
            return "SCALP_SHORT"
        if ("grid" in notes_str) or ("chop" in notes_str):
            return "GRID_MAKER"
        return "SWING_4H"

    async def manage_active_positions(self):
        """Monitors open positions in swing_open_positions, dispatching lifecycle rules by engine_type (SWING_4H, GRID_MAKER, SCALP_SHORT)."""
        await asyncio.to_thread(self._load_active_positions_sync)
        if not self.active_positions:
            return

        for symbol, pos in list(self.active_positions.items()):
            try:
                # Fetch live ticker price
                clean_sym = self.uta_client._normalize_symbol(symbol)
                ticker = await self.uta_client.exchange.fetch_ticker(clean_sym)
                curr_price = float(ticker.get("last", 0.0) or ticker.get("close", 0.0))
                if curr_price <= 0:
                    continue

                entry_p = pos["entry_price"]
                sl = pos["stop_loss"]
                tp = pos["take_profit"]
                is_long = "long" in str(pos.get("side", "long")).lower()
                engine_role = self._classify_position_engine(pos)
                is_grid_pos = (engine_role == "GRID_MAKER")
                is_scalp_short = (engine_role == "SCALP_SHORT")
                is_swing_4h = (engine_role == "SWING_4H")

                if is_long:
                    peak_p = max(pos.get("peak_price", entry_p), curr_price)
                    pos["peak_price"] = peak_p
                    initial_risk_unit = (entry_p - sl) if entry_p > sl else (entry_p * 0.02)
                    r_multiple = (curr_price - entry_p) / initial_risk_unit if initial_risk_unit > 0 else 0.0
                    pos["r_multiple"] = round(r_multiple, 2)
                else:
                    peak_p = min(pos.get("peak_price", entry_p), curr_price)
                    pos["peak_price"] = peak_p
                    initial_risk_unit = (sl - entry_p) if sl > entry_p else (entry_p * 0.02)
                    r_multiple = (entry_p - curr_price) / initial_risk_unit if initial_risk_unit > 0 else 0.0
                    pos["r_multiple"] = round(r_multiple, 2)

                # Fetch latest 4H candle for Kijun-sen trailing line and AI Shepherding indicators
                candles = await self.uta_client.exchange.fetch_ohlcv(clean_sym, timeframe="4h", limit=100)
                df_4h = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                ind_4h = self._compute_4h_indicators(df_4h)
                kijun = ind_4h.get("kijun", sl)
                tenkan = ind_4h.get("tenkan")

                if is_long:
                    # Ratchet 0: Dedicated Micro-Breakeven Ratchet for Chop Grid (GRID_MAKER) positions
                    # Protects winning trades by ratcheting stop to Entry * (1 + 0.0018) when Peak Profit >= +0.50% (covering 2x taker fees + buffer)
                    peak_profit_pct = (peak_p - entry_p) / entry_p
                    grid_be_trigger = max(float(getattr(config, "GRID_BREAKEVEN_TRIGGER_PCT", 0.0050)), 0.0050)
                    grid_fee_buffer = max(float(getattr(config, "GRID_BREAKEVEN_FEE_BUFFER_PCT", 0.0018)), 0.0018)

                    # Only ratchet if peak profit has cleanly cleared the hurdle (minimum peak ROE >= +0.50%)
                    if is_grid_pos and peak_profit_pct >= grid_be_trigger and sl < (entry_p * (1.0 + grid_fee_buffer)):
                        new_sl = entry_p * (1.0 + grid_fee_buffer)
                        pos["stop_loss"] = round(new_sl, 4)
                        pos["trailing_stop"] = round(new_sl, 4)
                        log.info(f"[GRID-RATCHET-BE] 🛡️ {symbol} (LONG) reached peak +{peak_profit_pct*100:.2f}% gain! Ratcheted SL to Breakeven (+{grid_fee_buffer*100:.2f}% fee hurdle): ${new_sl:,.4f}")
                        self._track_task(self.uta_client.modify_strategy_order(symbol, sl=new_sl, tp=pos.get("take_profit")), f"grid_be_long_{symbol}")

                    # Ratchet 1: Breakeven at +1.5R (LONG)
                    if r_multiple >= self.be_threshold_r and sl < (entry_p * 1.0018):
                        fee_buffer = max(entry_p * 0.0018, 0.05 * initial_risk_unit)
                        new_sl = entry_p + fee_buffer  # Lock in guaranteed positive net exit after round-trip fees
                        pos["stop_loss"] = round(new_sl, 4)
                        pos["trailing_stop"] = round(new_sl, 4)
                        log.info(f"[RATCHET-BE] {symbol} (LONG) at +{r_multiple:.2f}R! Stop-loss ratcheted to Breakeven (+${fee_buffer:.4f} fee buffer): ${new_sl:,.4f}")
                        self._track_task(self.uta_client.modify_strategy_order(symbol, sl=new_sl, tp=pos.get("take_profit")), f"ratchet_be_long_{symbol}")

                    # Ratchet 1.5: High-Water Mark Dynamic Profit Protection (Rule #154, #155, #156)
                    # Prevents winning runners from round-tripping peak paper gains back into breakeven
                    peak_roe = ((peak_p - entry_p) / entry_p) * pos.get("leverage", self.leverage) * 100.0
                    t2_roe = getattr(config, "SWING_PROFIT_LOCK_T2_ROE", 30.0)
                    t2_ratio = getattr(config, "SWING_PROFIT_LOCK_T2_RATIO", 0.70)
                    t1_roe = getattr(config, "SWING_PROFIT_LOCK_T1_ROE", 15.0)
                    t1_ratio = getattr(config, "SWING_PROFIT_LOCK_T1_RATIO", 0.50)

                    if peak_roe >= t2_roe:
                        floor_sl = entry_p + ((peak_p - entry_p) * t2_ratio)
                        if floor_sl > pos["stop_loss"] and curr_price > floor_sl:
                            pos["stop_loss"] = round(floor_sl, 4)
                            pos["trailing_stop"] = round(floor_sl, 4)
                            log.info(f"[PROFIT-LOCK-70%] {symbol} (LONG) Peak at ${peak_p:.4f} (+{peak_roe:.1f}% ROE)! Ratcheting stop to 70% profit floor: ${floor_sl:,.4f}")
                            self._track_task(self.uta_client.modify_strategy_order(symbol, sl=round(floor_sl, 4), tp=pos.get("take_profit")), f"profit_lock_70_long_{symbol}")
                    elif peak_roe >= t1_roe:
                        floor_sl = entry_p + ((peak_p - entry_p) * t1_ratio)
                        if floor_sl > pos["stop_loss"] and curr_price > floor_sl:
                            pos["stop_loss"] = round(floor_sl, 4)
                            pos["trailing_stop"] = round(floor_sl, 4)
                            log.info(f"[PROFIT-LOCK-50%] {symbol} (LONG) Peak at ${peak_p:.4f} (+{peak_roe:.1f}% ROE)! Ratcheting stop to 50% profit floor: ${floor_sl:,.4f}")
                            self._track_task(self.uta_client.modify_strategy_order(symbol, sl=round(floor_sl, 4), tp=pos.get("take_profit")), f"profit_lock_50_long_{symbol}")

                    # Ratchet 2: Dynamic 4H Trailing Stop (LONG - SWING_4H only) - only move UP
                    if is_swing_4h:
                        trail_anchor = tenkan if (r_multiple >= 3.0 and tenkan) else kijun
                        new_sl_val = round(trail_anchor, 4) if trail_anchor else None
                        if new_sl_val and new_sl_val > pos["stop_loss"] and curr_price > new_sl_val:
                            old_sl_val = pos["stop_loss"]
                            pos["stop_loss"] = new_sl_val
                            pos["trailing_stop"] = new_sl_val
                            anchor_name = "4H Tenkan" if (r_multiple >= 3.0 and tenkan) else "4H Kijun"
                            log.info(f"[TRAIL-RATCHET] {symbol} (LONG) Trailing stop advanced to {anchor_name}: ${new_sl_val:,.4f}")
                            self._track_task(self.uta_client.modify_strategy_order(symbol, sl=new_sl_val, tp=pos.get("take_profit")), f"trail_ratchet_long_{symbol}")

                            # Alert via email when stop moves meaningfully (> 0.5% advance)
                            if ((new_sl_val - old_sl_val) / max(0.001, old_sl_val) >= 0.005) or (old_sl_val < entry_p <= new_sl_val):
                                try:
                                    from email_dispatcher import send_trailing_ratchet_alert_async
                                    locked_profit = max(0.0, (new_sl_val - entry_p) * float(pos.get("size", 1.0) or 1.0))
                                    self._track_task(send_trailing_ratchet_alert_async(
                                        symbol=symbol,
                                        side="LONG",
                                        old_sl=old_sl_val,
                                        new_sl=new_sl_val,
                                        current_price=curr_price,
                                        peak_price=peak_p,
                                        r_multiple=r_multiple,
                                        locked_profit_usd=locked_profit,
                                        anchor_name=anchor_name
                                    ), f"trail_alert_long_{symbol}")
                                except Exception as trail_mail_err:
                                    log.debug(f"[TRAIL-ALERT-NOTE] {trail_mail_err}")
                else:
                    # Ratchet 0A: Fast Micro-Breakeven Ratchet for Quick Scalp Short & Short Grid positions
                    peak_profit_pct = (entry_p - peak_p) / entry_p
                    short_be_trigger = max(float(getattr(config, "GRID_BREAKEVEN_TRIGGER_PCT", 0.0050)), 0.0050)
                    short_fee_buffer = max(float(getattr(config, "SCALP_SHORT_BE_FEE_BUFFER_PCT", 0.0018)), 0.0018)

                    if (is_scalp_short or is_grid_pos) and peak_profit_pct >= short_be_trigger and sl > (entry_p * (1.0 - short_fee_buffer)):
                        new_sl = entry_p * (1.0 - short_fee_buffer)
                        pos["stop_loss"] = round(new_sl, 4)
                        pos["trailing_stop"] = round(new_sl, 4)
                        tag = "QUICK SCALP SHORT" if is_scalp_short else "SHORT GRID"
                        log.info(f"[SCALP-SHORT-BE] 🛡️ {symbol} ({tag}) reached peak +{peak_profit_pct*100:.2f}% gain! Ratcheted SL to Breakeven (-{short_fee_buffer*100:.2f}% fee hurdle): ${new_sl:,.4f}")
                        self._track_task(self.uta_client.modify_strategy_order(symbol, sl=new_sl, tp=pos.get("take_profit")), f"scalp_be_short_{symbol}")

                    # Ratchet 1: Breakeven at +1.5R (SHORT)
                    if r_multiple >= self.be_threshold_r and sl > (entry_p * 0.9982):
                        fee_buffer = max(entry_p * 0.0018, 0.05 * initial_risk_unit)
                        new_sl = entry_p - fee_buffer  # Lock in guaranteed positive net exit below entry after round-trip fees
                        pos["stop_loss"] = round(new_sl, 4)
                        pos["trailing_stop"] = round(new_sl, 4)
                        log.info(f"[RATCHET-BE] {symbol} (SHORT) at +{r_multiple:.2f}R! Stop-loss ratcheted to Breakeven (-${fee_buffer:.4f} fee buffer): ${new_sl:,.4f}")
                        self._track_task(self.uta_client.modify_strategy_order(symbol, sl=new_sl, tp=pos.get("take_profit")), f"ratchet_be_short_{symbol}")

                    # Ratchet 1.5: High-Water Mark Dynamic Profit Protection (Rule #154, #155, #156)
                    peak_roe = ((entry_p - peak_p) / entry_p) * pos.get("leverage", self.leverage) * 100.0
                    t2_roe = getattr(config, "SWING_PROFIT_LOCK_T2_ROE", 30.0)
                    t2_ratio = getattr(config, "SWING_PROFIT_LOCK_T2_RATIO", 0.70)
                    t1_roe = getattr(config, "SWING_PROFIT_LOCK_T1_ROE", 15.0)
                    t1_ratio = getattr(config, "SWING_PROFIT_LOCK_T1_RATIO", 0.50)

                    if peak_roe >= t2_roe:
                        floor_sl = entry_p - ((entry_p - peak_p) * t2_ratio)
                        if floor_sl < pos["stop_loss"] and curr_price < floor_sl:
                            pos["stop_loss"] = round(floor_sl, 4)
                            pos["trailing_stop"] = round(floor_sl, 4)
                            log.info(f"[PROFIT-LOCK-70%] {symbol} (SHORT) Peak at ${peak_p:.4f} (+{peak_roe:.1f}% ROE)! Ratcheting stop to 70% profit floor: ${floor_sl:,.4f}")
                            self._track_task(self.uta_client.modify_strategy_order(symbol, sl=round(floor_sl, 4), tp=pos.get("take_profit")), f"profit_lock_70_short_{symbol}")
                    elif peak_roe >= t1_roe:
                        floor_sl = entry_p - ((entry_p - peak_p) * t1_ratio)
                        if floor_sl < pos["stop_loss"] and curr_price < floor_sl:
                            pos["stop_loss"] = round(floor_sl, 4)
                            pos["trailing_stop"] = round(floor_sl, 4)
                            log.info(f"[PROFIT-LOCK-50%] {symbol} (SHORT) Peak at ${peak_p:.4f} (+{peak_roe:.1f}% ROE)! Ratcheting stop to 50% profit floor: ${floor_sl:,.4f}")
                            self._track_task(self.uta_client.modify_strategy_order(symbol, sl=round(floor_sl, 4), tp=pos.get("take_profit")), f"profit_lock_50_short_{symbol}")

                    # Ratchet 2: Dynamic 4H Trailing Stop (SHORT - SWING_4H only) - only move DOWN
                    if is_swing_4h:
                        trail_anchor = tenkan if (r_multiple >= 3.0 and tenkan) else kijun
                        new_sl_val = round(trail_anchor, 4) if trail_anchor else None
                        if new_sl_val and new_sl_val < pos["stop_loss"] and curr_price < new_sl_val:
                            old_sl_val = pos["stop_loss"]
                            pos["stop_loss"] = new_sl_val
                            pos["trailing_stop"] = new_sl_val
                            anchor_name = "4H Tenkan" if (r_multiple >= 3.0 and tenkan) else "4H Kijun"
                            log.info(f"[TRAIL-RATCHET] {symbol} (SHORT) Trailing stop dropped to {anchor_name}: ${new_sl_val:,.4f}")
                            self._track_task(self.uta_client.modify_strategy_order(symbol, sl=new_sl_val, tp=pos.get("take_profit")), f"trail_ratchet_short_{symbol}")

                            # Alert via email when stop moves meaningfully (> 0.5% advance)
                            if ((old_sl_val - new_sl_val) / max(0.001, old_sl_val) >= 0.005) or (old_sl_val > entry_p >= new_sl_val):
                                try:
                                    from email_dispatcher import send_trailing_ratchet_alert_async
                                    locked_profit = max(0.0, (entry_p - new_sl_val) * float(pos.get("size", 1.0) or 1.0))
                                    self._track_task(send_trailing_ratchet_alert_async(
                                        symbol=symbol,
                                        side="SHORT",
                                        old_sl=old_sl_val,
                                        new_sl=new_sl_val,
                                        current_price=curr_price,
                                        peak_price=peak_p,
                                        r_multiple=r_multiple,
                                        locked_profit_usd=locked_profit,
                                        anchor_name=anchor_name
                                    ), f"trail_alert_short_{symbol}")
                                except Exception as trail_mail_err:
                                    log.debug(f"[TRAIL-ALERT-NOTE] {trail_mail_err}")

                # Ratchet 3: AI Continuous Active Shepherding & Runner vs. Harvest Architect
                # Evaluates active position health every 1200s (20 minutes)
                # Enforces Rule #173 & Rule #175 Breathing Room: strictly mutes micro-tightening during
                # the first 60 minutes UNLESS price has already expanded >= 1.0x ATR above entry!
                last_ai_check = self._last_ai_health_check_ts.get(symbol, 0)
                try:
                    opened_dt = datetime.fromisoformat(str(pos.get("opened_at") or "").replace("Z", "+00:00"))
                    if opened_dt.tzinfo is None:
                        opened_dt = opened_dt.replace(tzinfo=UTC)
                    holding_mins = max(1.0, (datetime.now(UTC) - opened_dt).total_seconds() / 60.0)
                except Exception:
                    holding_mins = 60.0

                if (time.time() - last_ai_check) >= 1200 and not is_scalp_short:
                    try:
                        import ai_engine
                        unr_pnl_pct = ((curr_price - entry_p) / entry_p * 100.0) if is_long else ((entry_p - curr_price) / entry_p * 100.0)
                        atr_val = float(ind_4h.get("atr", 0.0) or (entry_p * 0.02))
                        price_advance = (curr_price - entry_p) if is_long else (entry_p - curr_price)
                        has_atr_expansion = (price_advance >= atr_val)
                        allow_tightening = (holding_mins >= 60.0 or has_atr_expansion)

                        ind_dict = {
                            "rsi": float(ind_4h.get("rsi", 50.0)),
                            "adx": float(ind_4h.get("adx", 20.0)),
                            "atr": atr_val,
                            "ema20": float(ind_4h.get("ema20", curr_price)),
                            "vol_ratio": float(ind_4h.get("vol_ratio", 1.0))
                        }

                        pos_context = {
                            "symbol": symbol,
                            "side": "LONG" if is_long else "SHORT",
                            "entry_price": entry_p,
                            "current_price": curr_price,
                            "unrealized_pnl_pct": unr_pnl_pct,
                            "r_multiple": r_multiple,
                            "holding_mins": holding_mins,
                            "adx": ind_dict["adx"],
                            "rsi": ind_dict["rsi"]
                        }

                        # If position is well in profit (>= +1.0R), deploy Dynamic Runner vs Harvest Architect
                        runner_decision = None
                        if r_multiple >= 1.0:
                            runner_decision = await ai_engine.evaluate_runner_vs_harvest(pos_context, timeout_sec=8.0)
                            if runner_decision:
                                log.info(f"[RUNNER-ARCHITECT] 🏃 {symbol} (+{r_multiple:.2f}R): Action={runner_decision.get('action')} | TargetR={runner_decision.get('target_r')}R | Rationale={runner_decision.get('rationale')}")

                        health_res = await ai_engine.check_position_health(
                            symbol=symbol,
                            side="LONG" if is_long else "SHORT",
                            entry_price=entry_p,
                            current_price=curr_price,
                            unrealized_pnl_pct=unr_pnl_pct,
                            holding_mins=holding_mins,
                            indicators=ind_dict,
                            current_sl=pos["stop_loss"],
                            current_tp=pos.get("take_profit", 0.0),
                            timeout_sec=15.0
                        )
                        self._last_ai_health_check_ts[symbol] = time.time()

                        if runner_decision:
                            health_res["runner_vs_harvest"] = runner_decision

                        # Persist fresh health & shepherding report to system_state
                        sym_clean = symbol.upper().replace("/", "").replace("-", "")
                        def _save_ai_health(s_name, h_res):
                            try:
                                with sqlite3.connect("trading_data.db", timeout=5.0) as s_conn:
                                    s_conn.execute(
                                        "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                                        (f"ai_health_{s_name}", json.dumps(h_res), datetime.now(UTC).isoformat())
                                    )
                                    s_conn.execute(
                                        "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                                        (f"ai_shepherd_{s_name}", json.dumps(h_res), datetime.now(UTC).isoformat())
                                    )
                                    s_conn.commit()
                            except Exception:
                                pass
                        await asyncio.to_thread(_save_ai_health, sym_clean, health_res)

                        h_verdict = health_res.get("verdict", "HOLD")
                        h_sl = health_res.get("suggested_sl")
                        h_diag = health_res.get("diagnosis", "")

                        # Execute Shepherding Actions
                        if h_verdict == "TIGHTEN_SL" and h_sl is not None:
                            # Rule #173 & Rule #175 Invariant: Check Breathing Room
                            if not allow_tightening:
                                log.info(f"[BREATHING-ROOM] 🛡️ Rule #173/175 Active: Muting micro-tightening for {symbol} (holding {holding_mins:.1f}m < 60m & advance < 1.0 ATR). Giving trade room to expand.")
                            else:
                                min_delta = entry_p * 0.002
                                if (is_long and (h_sl - pos["stop_loss"]) >= min_delta and h_sl < curr_price) or \
                                   (not is_long and (pos["stop_loss"] - h_sl) >= min_delta and h_sl > curr_price):
                                    old_sl_val = pos["stop_loss"]
                                    pos["stop_loss"] = round(h_sl, 4)
                                    pos["trailing_stop"] = round(h_sl, 4)
                                    log.info(f"[AI-SHEPHERD] 🛡️ {symbol} Stop Tightened by AI Shepherding: ${old_sl_val:.4f} -> ${h_sl:.4f} ({h_diag})")
                                    self._track_task(
                                        self.uta_client.modify_strategy_order(symbol, sl=round(h_sl, 4), tp=pos.get("take_profit")),
                                        f"ai_tighten_sl_{symbol}"
                                    )

                        elif (h_verdict == "TAKE_PROFIT_EARLY" or (runner_decision and runner_decision.get("action") == "HARVEST_MODE")) and unr_pnl_pct > 1.0:
                            log.info(f"[AI-SHEPHERD] 🎯 {symbol} AI Early Profit Harvest (+{unr_pnl_pct:.2f}%): {h_diag or 'Momentum stalling at major resistance'}")
                            await self.close_swing_position(symbol, exit_price=curr_price, reason="AI_EARLY_TP")
                            continue

                        elif h_verdict == "CLOSE_NOW":
                            log.warning(f"[AI-SHEPHERD] 🚨 {symbol} AI Emergency Structural Close: {h_diag}")
                            await self.close_swing_position(symbol, exit_price=curr_price, reason="AI_STRUCTURAL_INVALIDATION")
                            continue

                        else:
                            log.info(f"[AI-SHEPHERD] 🩺 {symbol} Health Check: {h_verdict} ({health_res.get('health_grade', 'PRISTINE')}) | PnL: {unr_pnl_pct:+.2f}% | Holding: {holding_mins:.1f}m | {h_diag}")

                    except Exception as ai_chk_err:
                        log.debug(f"[AI-HEALTH-NOTE] {symbol} health check note: {ai_chk_err}")

                # Exit Condition 0A: Two-Stage Time Decay Protocol for Quick Scalp Shorts (45m Tighten/Harvest -> 90m Hard Expiry)
                if is_scalp_short and not is_long:
                    try:
                        opened_dt = datetime.fromisoformat(str(pos.get("opened_at") or "").replace("Z", "+00:00"))
                        if opened_dt.tzinfo is None:
                            opened_dt = opened_dt.replace(tzinfo=UTC)
                        holding_mins = max(0.0, (datetime.now(UTC) - opened_dt).total_seconds() / 60.0)
                    except Exception:
                        holding_mins = 0.0

                    max_scalp_mins = getattr(config, "SCALP_SHORT_MAX_HOLD_MINS", 45.0)
                    hard_expiry_mins = getattr(config, "SCALP_SHORT_HARD_EXPIRY_MINS", 90.0)
                    min_harvest_pct = getattr(config, "SCALP_SHORT_MIN_HARVEST_PCT", 0.0045) * 100.0
                    if holding_mins >= max_scalp_mins:
                        short_pnl_pct = (entry_p - curr_price) / entry_p * 100.0
                        if short_pnl_pct >= (min_harvest_pct - 1e-5):
                            log.info(f"[SCALP-SHORT-TIMEOUT] ⏱️ {symbol} (QUICK SCALP SHORT) reached {holding_mins:.1f}m hold time (checkpoint {max_scalp_mins:.0f}m) with +{short_pnl_pct:.2f}% profit (>= min harvest {min_harvest_pct:.2f}%). Harvesting market TP.")
                            await self.close_swing_position(symbol, exit_price=curr_price, reason="SCALP_SHORT_TIME_PROFIT")
                            continue
                        elif holding_mins >= hard_expiry_mins:
                            log.info(f"[SCALP-SHORT-TIMEOUT] ⏱️ {symbol} (QUICK SCALP SHORT) reached {holding_mins:.1f}m hard expiry (limit {hard_expiry_mins:.0f}m) with PnL {short_pnl_pct:+.2f}%. Closing to enforce zero long-held short exposure.")
                            await self.close_swing_position(symbol, exit_price=curr_price, reason="SCALP_SHORT_TIME_EXPIRY")
                            continue
                        else:
                            # Stage 1 Defense (45m - 90m): Tighten SL instead of market-dumping a flat trade into taker fees
                            target_sl = round(entry_p * (1.0 - 0.0018), 4) if short_pnl_pct >= 0.22 else round(entry_p * 1.0055, 4)
                            if target_sl < pos["stop_loss"] and curr_price < target_sl:
                                old_sl = pos["stop_loss"]
                                pos["stop_loss"] = target_sl
                                pos["trailing_stop"] = target_sl
                                log.info(f"[SCALP-SHORT-45M-TIGHTEN] 🛡️ {symbol} at {holding_mins:.1f}m (PnL {short_pnl_pct:+.2f}% < {min_harvest_pct:.2f}% harvest). Tightened SL ${old_sl:.4f} -> ${target_sl:.4f} until {hard_expiry_mins:.0f}m hard expiry.")
                                self._track_task(self.uta_client.modify_strategy_order(symbol, sl=target_sl, tp=pos.get("take_profit")), f"scalp_45m_tighten_{symbol}")

                # Exit Condition 0: Global AI Macro Killswitch
                def _check_emergency_killswitch():
                    for attempt in range(2):
                        try:
                            with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                                row = conn.execute("SELECT value FROM system_state WHERE key = 'macro_global_killswitch'").fetchone()
                                return bool(row and str(row[0]) == "1")
                        except Exception as k_err:
                            if attempt == 0:
                                time.sleep(0.1)
                                continue
                            log.warning(f"[KILLSWITCH-READ-TRANSIENT-ERR] Error reading macro_global_killswitch: {k_err}. Preserving active position, will retry next cycle.")
                            return False
                    return False

                is_emergency = await asyncio.to_thread(_check_emergency_killswitch)
                if is_emergency:
                    log.warning(f"[MACRO-EMERGENCY] Global Killswitch engaged! Liquidating {symbol} swing trade to CASH.")
                    await self.close_swing_position(symbol, exit_price=curr_price, reason="MACRO_KILLSWITCH")
                    continue

                # Exit Condition 1: Take Profit Reached
                if tp and tp > 0.0:
                    if (is_long and curr_price >= tp) or (not is_long and curr_price <= tp):
                        await self.close_swing_position(symbol, exit_price=curr_price, reason="TAKE_PROFIT")
                        continue

                # Exit Condition 2: Stop Loss Breached
                if (is_long and curr_price <= pos["stop_loss"]) or (not is_long and curr_price >= pos["stop_loss"]):
                    await self.close_swing_position(symbol, exit_price=curr_price, reason="STOP_LOSS")
                    continue

                # Update state in DB
                now_iso = datetime.now(UTC).isoformat()
                pos["updated_at"] = now_iso
                def _update_pos():
                    with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                        conn.execute("""
                            UPDATE swing_open_positions
                            SET stop_loss = ?, trailing_stop = ?, peak_price = ?, r_multiple = ?, updated_at = ?
                            WHERE symbol = ?
                        """, (pos["stop_loss"], pos["trailing_stop"], pos["peak_price"], pos["r_multiple"], now_iso, symbol))
                        conn.commit()
                await asyncio.to_thread(_update_pos)

            except Exception as e:
                log.error(f"[MANAGE-ERROR] Error managing {symbol} swing position: {e}")

    async def close_swing_position(self, symbol: str, exit_price: float, reason: str):
        """Closes position on Bitget UTA, records closed trade, and releases mutex lock."""
        pos = self.active_positions.get(symbol)
        if not pos:
            return

        entry_p = pos["entry_price"]
        qty = pos["size"]
        is_long = "long" in str(pos.get("side", "long")).lower()
        pos_side = "long" if is_long else "short"

        if is_long:
            gross_pnl = (exit_price - entry_p) * qty
        else:
            gross_pnl = (entry_p - exit_price) * qty

        # Accurate Transaction Fee Attribution (Maker 0.02% if post-only maker entry, else Taker 0.06%)
        notes_lower = str(pos.get("notes", "")).lower()
        is_maker_entry = (
            "maker" in notes_lower
            or "post_only" in notes_lower
            or pos.get("order_type") == "post_only"
            or pos.get("execution_type") == "maker"
        )
        entry_fee_rate = 0.0002 if is_maker_entry else 0.0006
        exit_fee_rate = 0.0006  # Market close order incurs taker fee
        entry_fee = entry_p * qty * entry_fee_rate
        exit_fee = exit_price * qty * exit_fee_rate
        total_fees = entry_fee + exit_fee
        pnl = gross_pnl - total_fees
        roe_pct = (pnl / ((entry_p * qty) / max(1.0, float(pos.get("leverage", 5.0))))) * 100.0 if (entry_p * qty) > 0 else 0.0

        try:
            raw_opened = str(pos.get("opened_at") or "").replace("Z", "+00:00")
            opened_dt = datetime.fromisoformat(raw_opened)
            if opened_dt.tzinfo is None:
                opened_dt = opened_dt.replace(tzinfo=UTC)
            holding_hours = max(0.0, (datetime.now(UTC) - opened_dt).total_seconds() / 3600.0)
        except Exception:
            holding_hours = 0.0

        log.info("=" * 60)
        log.info(f"   CLOSING 4H SWING {pos_side.upper()}: {symbol} | Reason: {reason}")
        log.info(f"   Entry: ${entry_p:,.4f} -> Exit: ${exit_price:,.4f}")
        fee_desc = f"maker {entry_fee_rate*100:.2f}% + taker {exit_fee_rate*100:.2f}%" if is_maker_entry else "0.06% taker x2"
        log.info(f"   Gross PnL: ${gross_pnl:+,.2f} | Fees: ${total_fees:,.2f} ({fee_desc}) -> Net PnL: ${pnl:+,.2f} USD ({roe_pct:+.2f}% ROE) | Held: {holding_hours:.1f}h")
        log.info("=" * 60)

        # Route market close on Bitget
        close_res = None
        close_error = None
        try:
            close_res = await self.uta_client.close_position_order(symbol=symbol, side=pos_side, qty=qty)
        except Exception as ce:
            err_txt = str(ce)
            if "25227" in err_txt or "no position available to close" in err_txt.lower():
                log.info(f"[CLOSE-ALREADY-SETTLED] Position {symbol} ({pos_side.upper()}) already closed on Bitget (code 25227). Proceeding to ledger settlement.")
                close_res = {"id": None, "already_closed": True, "symbol": symbol, "side": pos_side, "qty": qty}
            else:
                close_error = ce
                log.critical(f"[CLOSE-ORDER-FAILED] 🚨 Bitget close order failed on {symbol}: {ce}")

        # Safety Check: If close order failed, do NOT delete position or release lock!
        if close_error or not close_res:
            def _mark_pending_close():
                with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                    conn.execute("""
                        UPDATE swing_open_positions
                        SET status = 'PENDING_CLOSE', notes = ?
                        WHERE symbol = ?
                    """, (f"PENDING_CLOSE: {reason} error: {close_error}", symbol))
                    conn.commit()
            await asyncio.to_thread(_mark_pending_close)
            if symbol in self.active_positions:
                self.active_positions[symbol]["status"] = "PENDING_CLOSE"
            log.warning(f"[CLOSE-SAFETY] Position {symbol} marked PENDING_CLOSE. Locks retained; position NOT purged.")
            return

        # Query and Aggregate Order Fills (Multi-tranche / split-fill aggregation)
        order_id = close_res.get("id") or close_res.get("orderId")
        actual_exit_price = exit_price
        actual_exit_fee = exit_fee
        actual_realized_pnl = None

        if order_id or close_res.get("already_closed"):
            try:
                await asyncio.sleep(0.5)  # brief pause for exchange fill settlement
                if order_id:
                    fills = await self.uta_client.get_order_fills(symbol=symbol, order_id=order_id)
                else:
                    recent_all = await self.uta_client.get_recent_fills(symbol=symbol, lookback_mins=30)
                    opp_order_side = "sell" if is_long else "buy"
                    fills = [
                        f for f in recent_all
                        if str(f.get("tradeSide", "")).lower() == "close"
                        or str(f.get("side", "")).lower() == opp_order_side
                    ]
                if fills:
                    total_fill_qty = sum(float(f.get("execQty", 0.0) or 0.0) for f in fills)
                    total_fill_val = sum(float(f.get("execValue", 0.0) or (float(f.get("execQty", 0.0) or 0.0) * float(f.get("execPrice", 0.0) or 0.0))) for f in fills)
                    total_fill_fee = sum(sum(float(fd.get("fee", 0.0) or 0.0) for fd in f.get("feeDetail", [])) if f.get("feeDetail") else float(f.get("fee", 0.0) or 0.0) for f in fills)
                    fill_pnl_list = [float(f.get("execPnl", 0.0) or 0.0) for f in fills if f.get("execPnl") is not None]

                    if total_fill_qty > 0 and total_fill_val > 0:
                        actual_exit_price = total_fill_val / total_fill_qty
                    if total_fill_fee > 0:
                        actual_exit_fee = total_fill_fee
                    if fill_pnl_list:
                        actual_realized_pnl = sum(fill_pnl_list) - (entry_fee + actual_exit_fee)

                    log.info(f"[CLOSE-SPLIT-FILLS] Aggregated {len(fills)} fill tranche(s) for {symbol}: Avg Price=${actual_exit_price:,.4f}, Total Fees=${actual_exit_fee:.4f}")
            except Exception as fill_err:
                log.debug(f"[CLOSE-FILLS-NOTE] Note aggregating fills for {symbol}: {fill_err}")

        # Stage 1 Instrumentation: Query Bitget Funding Fee History across holding window
        funding_usd = 0.0
        funding_source = "unavailable"
        try:
            opened_ms = int(opened_dt.timestamp() * 1000) if 'opened_dt' in locals() and opened_dt else None
            funding_records = await self.uta_client.get_funding_history(symbol=symbol, since_ms=opened_ms)
            if funding_records:
                # CCXT funding history records: positive is income, negative is fee paid
                funding_usd = sum(float(fr.get("amount", 0.0) or 0.0) for fr in funding_records)
                funding_source = "bitget_api"
            else:
                funding_usd = 0.0
                funding_source = "bitget_api" if opened_ms else "unavailable"
        except Exception as f_err:
            log.warning(f"[FUNDING-NOTE] Could not fetch funding history for {symbol}: {f_err}")
            funding_usd = 0.0
            funding_source = "unavailable"

        # Fee attribution split
        fee_maker_usd = entry_fee if is_maker_entry else 0.0
        fee_taker_usd = (0.0 if is_maker_entry else entry_fee) + actual_exit_fee
        fee_paid_usd = fee_maker_usd + fee_taker_usd

        # Net PnL incorporates trading gross pnl, fees, and funding payments
        pnl = actual_realized_pnl if actual_realized_pnl is not None else (gross_pnl - fee_paid_usd + funding_usd)
        roe_pct = (pnl / ((entry_p * qty) / max(1.0, float(pos.get("leverage", 5.0))))) * 100.0 if (entry_p * qty) > 0 else 0.0

        # Retrieve intended entry, fill price, slippage bps, and gates snapshot
        intended_entry = float(pos.get("intended_entry_price", entry_p) or entry_p)
        actual_fill_entry = float(pos.get("actual_fill_price", entry_p) or entry_p)
        pos_slippage_bps = float(pos.get("slippage_bps", 0.0) or 0.0)
        gates_snapshot_str = pos.get("gates_snapshot")
        if not gates_snapshot_str and "gates_snapshot" in pos:
            gates_snapshot_str = json.dumps(pos["gates_snapshot"])

        # Record in swing_trades history (with deduplication check)
        now_iso = datetime.now(UTC).isoformat()
        notes_str = f"Gross: ${gross_pnl:+.4f} | Fees: ${fee_paid_usd:.4f} (M:${fee_maker_usd:.4f}/T:${fee_taker_usd:.4f}) | Funding: ${funding_usd:+.4f} ({funding_source}) | {pos.get('notes', '')}".strip(" |")
        def _finalize_db():
            with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                existing = conn.execute("""
                    SELECT id FROM swing_trades
                    WHERE symbol = ? AND ABS(entry_price - ?) < 0.0001
                    AND (timestamp >= datetime('now', '-15 minutes') OR timestamp >= datetime('now', '-15 minutes', 'utc'))
                """, (symbol, entry_p)).fetchone()
                if existing:
                    log.info(f"[SWING-DEDUP] Trade {symbol} already recorded by reconciler (ID: {existing[0]}). Updating with accurate fills.")
                    conn.execute("""
                        UPDATE swing_trades
                        SET exit_price = ?, pnl = ?, roe_pct = ?, exit_reason = ?, notes = ?,
                            intended_entry_price = ?, actual_fill_price = ?, slippage_bps = ?,
                            fee_paid_usd = ?, fee_maker_usd = ?, fee_taker_usd = ?,
                            funding_usd = ?, funding_source = ?, gates_snapshot = ?, status = 'CLOSED'
                        WHERE id = ?
                    """, (
                        actual_exit_price, pnl, roe_pct, reason, notes_str,
                        intended_entry, actual_fill_entry, pos_slippage_bps,
                        fee_paid_usd, fee_maker_usd, fee_taker_usd,
                        funding_usd, funding_source, gates_snapshot_str,
                        existing[0]
                    ))
                else:
                    conn.execute("""
                        INSERT INTO swing_trades (
                            timestamp, symbol, side, entry_price, exit_price, size, pnl, roe_pct,
                            holding_hours, exit_reason, notes, intended_entry_price, actual_fill_price,
                            slippage_bps, fee_paid_usd, fee_maker_usd, fee_taker_usd,
                            funding_usd, funding_source, gates_snapshot, status
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        now_iso, symbol, pos_side, entry_p, actual_exit_price, qty, pnl, roe_pct,
                        holding_hours, reason, notes_str, intended_entry, actual_fill_entry,
                        pos_slippage_bps, fee_paid_usd, fee_maker_usd, fee_taker_usd,
                        funding_usd, funding_source, gates_snapshot_str, 'CLOSED'
                    ))
                conn.execute("DELETE FROM swing_open_positions WHERE symbol = ?", (symbol,))
                conn.commit()

        await asyncio.to_thread(_finalize_db)

        # Clean memory
        self.active_positions.pop(symbol, None)

        # Release the exclusive asset lock so scalpers can resume
        await asyncio.to_thread(self._release_lock_sync, symbol)
        log.info(f"[POSITION-CLOSED] {symbol} trade finalized. Mutex lock released.")

        # Enforce Anti-Whipsaw Cooldown (Rule #166) if closed by stop-loss or loss
        if "stop" in reason.lower() or "sl" in reason.lower() or pnl < 0:
            cooldown_h = getattr(config, "SWING_COOLDOWN_HOURS", 12.0)
            await asyncio.to_thread(self._set_cooldown_sync, symbol, cooldown_h, f"{pos_side.upper()}_{reason}")

            # Feature 5: Trigger Instant Loss Autopsy to distill binding learned rules
            try:
                import post_mortem_agent
                self._track_task(post_mortem_agent.trigger_instant_loss_autopsy(symbol=symbol, pnl=pnl, exit_reason=reason), f"instant_autopsy_{symbol}")
            except Exception as pm_err:
                log.warning(f"[AUTOPSY-TRIGGER-NOTE] {pm_err}")

        # Dispatch Email Alert for Trade Close
        try:
            from email_dispatcher import send_trade_log_async
            close_ctx = {
                "engine": "The Swing Architect (4H)",
                "side": pos_side.upper(),
                "qty": qty,
                "entry_price": entry_p,
                "price": actual_exit_price,
                "pnl": pnl,
                "leverage": pos.get("leverage", self.leverage),
                "notes": f"4H Swing {reason} ({holding_hours:.1f}h hold)",
                "close_trigger": reason,
                "close_reason": f"Position closed at ${actual_exit_price:,.4f} due to {reason}. Realized PnL: ${pnl:+,.2f} ({roe_pct:+.2f}% ROE)."
            }
            self._track_task(send_trade_log_async(symbol=symbol, side=pos_side.upper(), price=actual_exit_price, pnl=pnl, status="CLOSED", trade_context=close_ctx), "close_trade_alert")
        except Exception as mail_err:
            log.warning(f"[EMAIL-ALERT-NOTE] Trade close email note: {mail_err}")

    # ---------------------------------------------------------------------------
    # Main Daemon Loop & Telemetry
    # ---------------------------------------------------------------------------
    def publish_telemetry(self):
        """Emits real-time telemetry into system_state table."""
        now_iso = datetime.now(UTC).isoformat()
        telemetry = {
            "engine": "swing_architect",
            "status": "ONLINE",
            "timestamp": now_iso,
            "active_positions_count": len(self.active_positions),
            "max_locks": self.max_locks,
            "active_positions": self.active_positions,
            "timeframe": self.timeframe_entry,
            "allocation_usdt": self.allocation_usdt,
            "leverage": self.leverage
        }
        query = "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('swing_trader_telemetry', ?, ?)"
        self.db_queue.enqueue_write("system_state", query, (json.dumps(telemetry), now_iso))

    async def scan_fleet(self, allowed_direction: str = None):
        """Evaluates 4H setups across the 10 fleet coins (Long & Short)."""
        # Refresh heartbeats for existing locks
        await asyncio.to_thread(self._refresh_heartbeats_sync)

        # Check lock capacity and parameters (dynamically refreshed from config)
        self.max_locks = getattr(config, "SWING_MAX_CONCURRENT_LOCKS", 4)
        self.min_adx = getattr(config, "SWING_MIN_ADX", 22.0)
        if len(self.active_positions) >= self.max_locks:
            log.info(f"[CAPACITY] Max swing positions ({self.max_locks}) reached. Monitoring active trades.")
            return

        # Filter candidate symbols
        candidates = [s for s in self.symbols if s not in self.active_positions]
        if not candidates:
            return

        # Dynamically rank candidates by 24h Relative Strength vs BTC (V36.5)
        # Prioritizes momentum leaders for Longs and breakdown laggards for Shorts
        try:
            clean_syms = [self.uta_client._normalize_symbol(s) for s in candidates]
            tickers = await self.uta_client.exchange.fetch_tickers(clean_syms)
            is_long_scan = (allowed_direction or "LONG").upper() == "LONG"
            candidates.sort(
                key=lambda s: float((tickers.get(self.uta_client._normalize_symbol(s)) or {}).get("percentage", 0.0) or 0.0),
                reverse=is_long_scan
            )
            top_rs = ", ".join([f"{s.split('/')[0]}: {(tickers.get(self.uta_client._normalize_symbol(s)) or {}).get('percentage', 0.0):+.2f}%" for s in candidates[:3]])
            log.debug(f"[SWING-RS-RANKING] 4H scan priority ({'LONG' if is_long_scan else 'SHORT'}): {top_rs}")
        except Exception as sort_err:
            log.debug(f"[SWING-RS-NOTE] RS sorting fallback: {sort_err}")

        for symbol in candidates:

            # 0a. Check Blacklist (e.g. DOGE/USDT)
            blacklist = getattr(config, "SWING_BLACKLIST", ["DOGE/USDT"])
            if any(b.replace("/", "").upper() == symbol.replace("/", "").upper() for b in blacklist):
                continue

            # 0b. Check Anti-Whipsaw Cooldown (Rule #166)
            cooldown_until = await asyncio.to_thread(self._get_cooldown_sync, symbol, allowed_direction or "LONG")
            if cooldown_until:
                try:
                    clean_cd = str(cooldown_until).replace("Z", "+00:00")
                    if "T" not in clean_cd and " " in clean_cd: clean_cd = clean_cd.replace(" ", "T")
                    if "+" not in clean_cd: clean_cd += "+00:00"
                    cd_epoch = datetime.fromisoformat(clean_cd).timestamp()
                    if cd_epoch > datetime.now(UTC).timestamp():
                        log.debug(f"[COOLDOWN-GATE] Skipping {symbol}: active cool-off until {cooldown_until} (Anti-Whipsaw Rule #166)")
                        continue
                except Exception as cd_parse_err:
                    log.debug(f"[COOLDOWN-PARSE] {symbol}: {cd_parse_err}")

            try:
                # 1. Fetch 4H (limit=95 for 52+26 shifted Ichimoku Kumo) and 1D data
                clean_sym = self.uta_client._normalize_symbol(symbol)
                candles_4h = await self.uta_client.exchange.fetch_ohlcv(clean_sym, timeframe="4h", limit=95)
                candles_1d = await self.uta_client.exchange.fetch_ohlcv(clean_sym, timeframe="1d", limit=210)
                if len(candles_4h) < 80:
                    continue

                df_4h = pd.DataFrame(candles_4h, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df_1d = pd.DataFrame(candles_1d, columns=["timestamp", "open", "high", "low", "close", "volume"])

                # Enforce evaluation on confirmed closed 4H candles to eliminate intraday wick false breakouts
                df_4h_eval = df_4h.iloc[:-1] if len(df_4h) > 80 else df_4h
                df_1d_eval = df_1d.iloc[:-1] if len(df_1d) > 200 else df_1d

                ind_4h = self._compute_4h_indicators(df_4h_eval)
                ind_1d = self._compute_1d_trend(df_1d_eval)

                # Determine candidate direction
                cand_direction = None
                rsi = ind_4h.get("rsi", 50.0)
                adx = ind_4h.get("adx", 0.0)
                live_4h_o = float(df_4h["open"].iloc[-1])
                live_4h_c = float(df_4h["close"].iloc[-1])

                # LONG Checklist:
                # 1) 4H Bullish Kumo breakout + Bullish Closed 4H Expansion Candle (Close > Open & Close >= Prior Close)
                # 2) Live forming 4H bar not actively dumping (live_close >= live_open * 0.9985)
                # 3) ADX >= min_adx (trending market)
                # 4) 1D Bullish alignment (Price > 200 EMA & 50 EMA >= 0.98 * 200 EMA) -> Anti-Dead-Cat-Bounce
                # 5) RSI <= 70.0 (Not overbought)
                if (ind_4h.get("bullish_kumo", False) and
                    ind_4h.get("bullish_candle_4h", False) and
                    live_4h_c >= live_4h_o * 0.9985 and
                    live_4h_c >= ind_4h["close"] * 0.9970 and
                    adx >= self.min_adx and 
                    ind_1d.get("bullish_alignment", False) and 
                    rsi <= 70.0):
                    cand_direction = "LONG"

                # SHORT Checklist:
                # 1) 4H Bearish Kumo breakdown + Bearish Closed 4H Breakdown Candle (Close < Open & Close <= Prior Close)
                # 2) Live forming 4H bar not actively squeezing up (live_close <= live_open * 1.0015)
                # 3) ADX >= self.min_adx (trending breakdown)
                # 4) 1D Bearish alignment (Price < 200 EMA & 50 EMA <= 1.02 * 200 EMA)
                # 5) RSI >= 35.0 (Anti-Bottom-Shorting / Not oversold)
                elif (ind_4h.get("bearish_kumo", False) and
                      ind_4h.get("bearish_candle_4h", False) and
                      live_4h_c <= live_4h_o * 1.0015 and
                      live_4h_c <= ind_4h["close"] * 1.0030 and
                      adx >= self.min_adx and 
                      ind_1d.get("bearish_alignment", False) and 
                      rsi >= 35.0):
                    cand_direction = "SHORT"

                if not cand_direction:
                    continue

                if allowed_direction and cand_direction != allowed_direction.upper():
                    continue

                # Enforce Fleet-Wide Long Cap: Max 2 Active Longs across the fleet
                if cand_direction == "LONG":
                    fleet_longs = await asyncio.to_thread(self._count_fleet_long_positions_sync)
                    max_fleet_longs = getattr(config, "MAX_FLEET_LONG_POSITIONS", 2)
                    if fleet_longs >= max_fleet_longs:
                        log.debug(f"[FLEET-LONG-CAP] Active fleet longs ({fleet_longs}) >= cap ({max_fleet_longs}). Vetoing {symbol} LONG candidate.")
                        continue

                log.info(f"[CANDIDATE-FOUND] 4H Kumo {cand_direction} confirmed on {symbol}! (ADX: {adx:.1f}, RSI: {rsi:.1f})")
                candle_4h_ts = ind_4h.get("candle_ts", 0)
                candle_key = f"{symbol}|{cand_direction}|{candle_4h_ts}"

                # 2. Check Macro Safety for candidate direction
                macro_ok, macro_rsn = await self._check_macro_safety(direction=cand_direction)
                if not macro_ok:
                    log.info(f"[MACRO-GATE] {symbol} {cand_direction} candidate blocked: {macro_rsn}")
                    await asyncio.to_thread(self._record_shadow_signal_sync, "SWING_4H", symbol, cand_direction, ind_4h["close"], veto_reason=f"Macro: {macro_rsn}", candle_key=candle_key)
                    continue

                # 3. Sentinel News Veto
                news_ok, news_rsn = await self._check_news_veto(symbol, direction=cand_direction)
                if not news_ok:
                    log.warning(f"[NEWS-VETO] {symbol} candidate blocked: {news_rsn}")
                    await asyncio.to_thread(self._record_shadow_signal_sync, "SWING_4H", symbol, cand_direction, ind_4h["close"], veto_reason=f"News: {news_rsn}", candle_key=candle_key)
                    continue

                # 4. HTF Resistance/Support Runway Gate
                runway_ok, runway_rsn = await self._check_htf_runway(symbol, ind_4h["close"], direction=cand_direction)
                if not runway_ok:
                    log.warning(f"[RUNWAY-VETO] {symbol} candidate blocked: {runway_rsn}")
                    await asyncio.to_thread(self._record_shadow_signal_sync, "SWING_4H", symbol, cand_direction, ind_4h["close"], veto_reason=f"Runway: {runway_rsn}", candle_key=candle_key)
                    continue

                # 4.4 15m Intraday Trend & EMA20 Alignment Gate (Zero AI Cost)
                align15_ok, align15_rsn = await self._check_intraday_15m_alignment(symbol, direction=cand_direction)
                if not align15_ok:
                    log.warning(f"[15M-ALIGNMENT-VETO] {symbol} {cand_direction} candidate blocked: {align15_rsn}")
                    await asyncio.to_thread(self._record_shadow_signal_sync, "SWING_4H", symbol, cand_direction, ind_4h["close"], veto_reason=f"15mAlign: {align15_rsn}", candle_key=candle_key)
                    continue

                # 4.6 Momentum Divergence Gate (Go-Stock Quantitative Upgrade — Zero AI Cost)
                div_ok, div_rsn = self._check_divergence_gate(symbol, df_4h_eval, direction=cand_direction)
                if not div_ok:
                    log.warning(f"[DIVERGENCE-VETO] {symbol} {cand_direction} candidate blocked: {div_rsn}")
                    await asyncio.to_thread(self._record_shadow_signal_sync, "SWING_4H", symbol, cand_direction, ind_4h["close"], veto_reason=f"Divergence: {div_rsn}", candle_key=candle_key)
                    continue

                # 4.7 Volume Profile & Value Area Gate (Go-Stock Quantitative Upgrade — Zero AI Cost)
                vp_ok, vp_rsn = self._check_volume_profile_gate(symbol, df_4h_eval, ind_4h["close"], direction=cand_direction)
                if not vp_ok:
                    log.warning(f"[VOLUME-PROFILE-VETO] {symbol} {cand_direction} candidate blocked: {vp_rsn}")
                    await asyncio.to_thread(self._record_shadow_signal_sync, "SWING_4H", symbol, cand_direction, ind_4h["close"], veto_reason=f"VolumeProfile: {vp_rsn}", candle_key=candle_key)
                    continue

                # 4.8 TD Sequential Exhaustion Gate (Go-Stock Quantitative Upgrade — Zero AI Cost)
                td_ok, td_rsn = self._check_td_sequential_gate(symbol, df_4h_eval, direction=cand_direction)
                if not td_ok:
                    log.warning(f"[TD-SEQUENTIAL-VETO] {symbol} {cand_direction} candidate blocked: {td_rsn}")
                    await asyncio.to_thread(self._record_shadow_signal_sync, "SWING_4H", symbol, cand_direction, ind_4h["close"], veto_reason=f"TD9: {td_rsn}", candle_key=candle_key)
                    continue

                # Per-Candle Boardroom & Trap Gate Cache (cuts 4H scan AI calls by ~95%)
                # Cache per (symbol, direction, 4h_candle_timestamp) for up to 60m unless price moves > 0.5%
                cache_key = (symbol, cand_direction)
                curr_p = float(ind_4h["close"])
                cached_veto = self._ai_veto_cache.get(cache_key)
                if isinstance(cached_veto, dict):
                    age_sec = time.time() - float(cached_veto.get("ts", 0.0))
                    same_candle = (candle_4h_ts > 0 and cached_veto.get("candle_ts") == candle_4h_ts)
                    ref_price = float(cached_veto.get("price", curr_p) or curr_p)
                    price_moved_pct = abs(curr_p - ref_price) / ref_price if ref_price > 0 else 1.0
                    if (same_candle or age_sec < 3600.0) and price_moved_pct <= 0.005:
                        log.info(
                            f"[AI-CANDLE-CACHE] Skipping {symbol} {cand_direction} on same 4H bar "
                            f"(vetoed {int(age_sec / 60)}m ago, Δprice={price_moved_pct*100:.2f}% <= 0.50%: {str(cached_veto.get('reason', ''))[:75]})"
                        )
                        continue

                # 4.5 AI Breakout vs. Bull-Trap Gate (Zero-Lag Fakeout Shield — with Per-Candle Pass Cache)
                cached_trap = self._trap_pass_cache.get(cache_key)
                trap_cached_ok = False
                if isinstance(cached_trap, dict):
                    same_candle = (candle_4h_ts > 0 and cached_trap.get("candle_ts") == candle_4h_ts)
                    ref_price = float(cached_trap.get("price", curr_p) or curr_p)
                    price_moved_pct = abs(curr_p - ref_price) / ref_price if ref_price > 0 else 1.0
                    if same_candle and price_moved_pct <= 0.005:
                        trap_cached_ok = True
                        log.info(f"[TRAP-CANDLE-CACHE] Reusing 4H Trap Gate pass for {symbol} {cand_direction} (Δprice={price_moved_pct*100:.2f}%)")

                if not trap_cached_ok:
                    trap_ok, trap_score, trap_rsn = await self._check_breakout_trap_gate(symbol, ind_4h, curr_p, direction=cand_direction)
                    if not trap_ok:
                        self._ai_veto_cache[cache_key] = {
                            "ts": time.time(),
                            "candle_ts": candle_4h_ts,
                            "price": curr_p,
                            "reason": f"TrapGate: {trap_rsn}"
                        }
                        log.warning(f"[BREAKOUT-TRAP-VETO] {symbol} {cand_direction} blocked: {trap_rsn}")
                        await asyncio.to_thread(self._record_shadow_signal_sync, "SWING_4H", symbol, cand_direction, curr_p, veto_reason=f"TrapGate: {trap_rsn}", candle_key=candle_key)
                        continue
                    self._trap_pass_cache[cache_key] = {
                        "ts": time.time(),
                        "candle_ts": candle_4h_ts,
                        "price": curr_p,
                        "score": trap_score,
                        "reason": trap_rsn
                    }

                # 5. Multi-Agent Conviction Boardroom (Unified 1-Call Deliberation)
                log.info(f"[BOARDROOM] Convening Boardroom deliberation for {symbol} 4H Swing {cand_direction}...")

                execute, score, rationale = await self._deliberate_boardroom(symbol, ind_4h, ind_1d, direction=cand_direction)
                if not execute or score < 0.65:
                    self._ai_veto_cache[cache_key] = {
                        "ts": time.time(),
                        "candle_ts": candle_4h_ts,
                        "price": curr_p,
                        "reason": f"Boardroom ({score:.2f}): {rationale}"
                    }
                    log.warning(f"[BOARDROOM-VETO] {symbol} rejected (Score: {score:.2f}): {rationale}")
                    await asyncio.to_thread(self._record_shadow_signal_sync, "SWING_4H", symbol, cand_direction, curr_p, veto_reason=f"Boardroom: {rationale}", candle_key=candle_key)
                    continue

                log.info(f"[BOARDROOM-APPROVED] {symbol} passed deliberation for {cand_direction} (Score: {score:.2f})!")

                # Assemble Gate Snapshot for Truthful Quantitative Logging
                gates_snapshot = {
                    "1d_trend": ind_1d.get("daily_trend", "UNKNOWN"),
                    "1d_bullish_alignment": bool(ind_1d.get("bullish_alignment", False)),
                    "1d_bearish_alignment": bool(ind_1d.get("bearish_alignment", False)),
                    "4h_adx": round(float(adx), 2),
                    "4h_rsi": round(float(rsi), 2),
                    "macro_safety": {"pass": bool(macro_ok), "reason": str(macro_rsn)},
                    "sentinel_news": {"pass": bool(news_ok), "reason": str(news_rsn)},
                    "htf_runway": {"pass": bool(runway_ok), "reason": str(runway_rsn)},
                    "15m_alignment": {"pass": bool(align15_ok), "reason": str(align15_rsn)},
                    "momentum_divergence": {"pass": bool(div_ok), "reason": str(div_rsn)},
                    "volume_profile": {"pass": bool(vp_ok), "reason": str(vp_rsn)},
                    "td_sequential": {"pass": bool(td_ok), "reason": str(td_rsn)},
                    "breakout_trap": {"pass": True, "score": float(trap_score if 'trap_score' in locals() else 0.0), "reason": str(trap_rsn if 'trap_rsn' in locals() else "cached")},
                    "boardroom_conviction": {"score": float(score), "rationale": str(rationale)}
                }

                # 6. Open swing position & acquire exclusive lock
                await self.open_swing_position(
                    symbol, ind_4h, direction=cand_direction, conviction_score=score, gates_snapshot=gates_snapshot
                )
                break  # Open at most one position per cycle to allow portfolio re-weighting

            except Exception as sym_err:
                log.error(f"[SCAN-ERROR] Error evaluating {symbol}: {sym_err}", exc_info=True)

    async def run(self):
        """Main autonomous execution loop."""
        log.info("=" * 65)
        log.info("   LAUNCHING THE SWING ARCHITECT (4H / DAILY TREND ENGINE)")
        log.info(f"   Allocation: ${self.allocation_usdt:,.2f} | Leverage: {self.leverage}x | Max Locks: {self.max_locks}")
        log.info(f"   Timeframe: {self.timeframe_entry} (Entry) + {self.timeframe_macro} (Macro Trend)")
        log.info("   Defense Suite: Boardroom, Sentinel, Macro Posture, Asset Mutex")
        log.info("=" * 65)

        await self.initialize_clients()

        while not self._stop_event.is_set():
            try:
                # 1. Manage active positions (breakeven & trailing stops)
                await self.manage_active_positions()

                # 2. Scan fleet for new 4H setups
                await self.scan_fleet()

                # 3. Emit telemetry
                self.publish_telemetry()

            except Exception as loop_e:
                log.error(f"[LOOP-ERROR] Unexpected error in Swing Architect loop: {loop_e}", exc_info=True)

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.check_interval)
            except asyncio.TimeoutError:
                pass

        log.info("[SHUTDOWN] The Swing Architect daemon halting cleanly.")
        if self.uta_client:
            await self.uta_client.close()

    def stop(self):
        self._stop_event.set()


def main():
    # Check both config.py and system_state in DB for dynamic activation
    db_enabled = False
    try:
        with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
            row = conn.execute("SELECT value FROM system_state WHERE key = 'SWING_TRADER_ENABLED'").fetchone()
            if row and str(row[0]).strip().lower() in ("1", "true"):
                db_enabled = True
    except Exception:
        pass

    if not getattr(config, "SWING_TRADER_ENABLED", False) and not db_enabled:
        log.info("[DISABLED] The Swing Architect is disabled via config (SWING_TRADER_ENABLED=false).")
        try:
            with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                now_iso = datetime.now(UTC).isoformat()
                parked = json.dumps({"status": "DISABLED", "timestamp": now_iso})
                conn.execute(
                    "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('swing_trader_telemetry', ?, ?)",
                    (parked, now_iso)
                )
                conn.commit()
        except Exception:
            pass
        sys.exit(0)

    trader = SwingArchitectTrader()

    def _sig_handler(sig, frame):
        log.warning(f"[SIGNAL] Shutdown signal ({sig}) received. Stopping daemon...")
        trader.stop()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        asyncio.run(trader.run())
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as e:
        log.critical(f"[FATAL] Swing Architect daemon crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

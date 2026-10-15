"""
super_bot.py — Unified Dual-Regime Sovereign Engine (V36.0 Enterprise)
======================================================================
Unified execution engine that consolidates:
1. BULL REGIME: The Swing Architect (4H Ichimoku Breakouts, 3x/5x Longs Only, Dynamic Kijun Trailing)
2. CHOP REGIME: Mean Reversion Maker Grid (5m/15m VWAP - 2.0 Sigma Dips, 3x Longs Only, Post-Only Limit Orders)
3. BEAR/CRASH REGIME: 100% Cash / USDT Preservation (Zero Shorts, Zero Exposure, Capital Shield)

Critical Features:
- Seamless State Inheritance: Adopts, tracks, and manages all existing live positions (LINK, SOL, NEAR, etc.).
- Active Position Management: +1.5R Breakeven Ratchet, +15%/+30% Dynamic Profit Floors, Kijun-sen Trailing.
- Maker Order Advantage: Post-only limit orders reduce fee drag by 66% (0.02% vs 0.06%).
- Exclusive Mutex Locking: Coordinates asset allocation so swing and grid strategies never clash.
"""

import asyncio
import json
import logging
import os
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone, UTC, timedelta
import pandas as pd
import numpy as np
import ta

import config
from bitget_uta_client import BitgetUTAClient
from conviction_filter import ConvictionBoardroom
from db_manager import DBWriteQueue
from swing_trader import SwingArchitectTrader
from indicators import detect_momentum_divergence, compute_volume_profile, compute_td_sequential
from microstructure_guard import MicrostructureStateCache
from strategy_base import SwingKumoStrategy, ChopGridMakerStrategy, BearScalpShortStrategy


try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Explicit logging configuration
log = logging.getLogger("super_bot")
log.setLevel(logging.INFO)
log.propagate = False
if not log.handlers:
    _fmt = logging.Formatter("%(asctime)s [%(levelname)s] [SUPER_BOT] %(message)s")
    _sh = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(_fmt)
    log.addHandler(_sh)
    try:
        os.makedirs("logs", exist_ok=True)
        _fh = logging.FileHandler("logs/super_bot.log", encoding="utf-8")
        _fh.setFormatter(_fmt)
        log.addHandler(_fh)
    except Exception:
        pass


def compute_grid_5m_indicators(df: pd.DataFrame) -> dict:
    """
    Pure indicator pipeline for 5m Chop Grid & Scalp Short engines.
    Zero-lookahead: computes RSI (14), 24-period rolling VWAP & deviation bands,
    EMA20, candle wick ratios, and 14-period ADX strictly up to the current bar.
    """
    if df is None or len(df) < 25:
        return {}

    c = df["close"]
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi_series = 100.0 - (100.0 / (1.0 + rs))
    rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
    rsi_prev = float(rsi_series.iloc[-2]) if len(rsi_series) >= 2 and not pd.isna(rsi_series.iloc[-2]) else rsi

    cum_pv = (c * df["volume"]).rolling(24).sum()
    cum_v = df["volume"].rolling(24).sum()
    vwap_series = cum_pv / (cum_v + 1e-9)
    vwap = float(vwap_series.iloc[-1]) if not pd.isna(vwap_series.iloc[-1]) else float(c.iloc[-1])

    # True VWAP deviation band: σ of (price - rolling VWAP), fallback to std-of-closes
    vwap_diff = c - vwap_series
    std_series = vwap_diff.rolling(24).std()
    rolling_std = float(std_series.iloc[-1]) if not pd.isna(std_series.iloc[-1]) else 0.0
    if rolling_std <= 0:
        c_std = c.rolling(24).std()
        rolling_std = float(c_std.iloc[-1]) if not pd.isna(c_std.iloc[-1]) else 0.0

    lower_band = vwap - (2.0 * rolling_std)
    upper_band = vwap + (2.0 * rolling_std)

    ema20_series = c.ewm(span=20, adjust=False).mean()
    ema20 = float(ema20_series.iloc[-1]) if not pd.isna(ema20_series.iloc[-1]) else float(c.iloc[-1])

    c_open = float(df["open"].iloc[-1])
    c_high = float(df["high"].iloc[-1])
    c_low = float(df["low"].iloc[-1])
    c_close = float(df["close"].iloc[-1])
    c_range = max(c_high - c_low, 1e-6)
    upper_wick = c_high - max(c_open, c_close)
    upper_wick_ratio = upper_wick / c_range

    adx_5m = 0.0
    try:
        adx_series = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
        if not pd.isna(adx_series.iloc[-1]):
            adx_5m = float(adx_series.iloc[-1])
    except Exception:
        adx_5m = 0.0

    is_red_candle = c_close < c_open
    is_shooting_star = upper_wick_ratio >= 0.35
    is_rsi_falling = rsi < (rsi_prev - 0.5)
    has_bearish_reversal = (is_red_candle and upper_wick_ratio >= 0.25) or (is_shooting_star and is_rsi_falling)

    return {
        "rsi": rsi,
        "rsi_prev": rsi_prev,
        "vwap": vwap,
        "rolling_std": rolling_std,
        "lower_band": lower_band,
        "upper_band": upper_band,
        "ema20": ema20,
        "adx_5m": adx_5m,
        "curr_price": c_close,
        "open": c_open,
        "high": c_high,
        "low": c_low,
        "close": c_close,
        "upper_wick_ratio": upper_wick_ratio,
        "is_red_candle": is_red_candle,
        "is_shooting_star": is_shooting_star,
        "is_rsi_falling": is_rsi_falling,
        "has_bearish_reversal": has_bearish_reversal,
    }


def evaluate_chop_grid_signal(
    ind: dict,
    curr_price: float,
    rsi_thresh: float = 30.0,
    max_adx: float = 24.0,
    regime_adx: float = 20.0,
    di_spread: float = 0.0
) -> tuple[bool, str]:
    """
    Pure evaluation of Chop Grid Maker Long Dip-Buy entry.
    """
    if not ind or "rsi" not in ind:
        return False, "NO_INDICATORS"

    # Regime gates
    if regime_adx >= 25.0 and di_spread < -4.0:
        return False, f"REGIME_SELLER_DOMINANCE (ADX={regime_adx:.1f}, DI_SPREAD={di_spread:+.1f})"
    if regime_adx >= 28.0:
        return False, f"REGIME_EXPANSION (ADX={regime_adx:.1f})"

    # Asset 5m ADX exhaustion guard
    if ind.get("adx_5m", 0.0) >= max_adx:
        return False, f"ASSET_5M_ADX_WATERFALL (ADX={ind.get('adx_5m', 0.0):.1f} >= {max_adx:.1f})"

    rsi = ind["rsi"]
    lower_band = ind["lower_band"]

    if rsi < rsi_thresh and curr_price <= lower_band:
        return True, f"CHOP_DIP_CONFIRMED (RSI={rsi:.1f} < {rsi_thresh:.1f}, Price={curr_price} <= Lower={lower_band:.4f})"

    return False, "CONDITIONS_NOT_MET"


def evaluate_scalp_short_signal(
    ind: dict,
    curr_price: float,
    mode: str = "CHOP",
    min_rsi: float = 38.0,
    rsi_overbought: float = 68.0,
    max_adx: float = 22.0
) -> tuple[bool, str]:
    """
    Pure evaluation of Quick Scalp Short entry.
    """
    if not ind or "rsi" not in ind:
        return False, "NO_INDICATORS"

    rsi = ind["rsi"]
    if rsi < min_rsi:
        return False, f"ANTI_BOTTOM_RSI (RSI={rsi:.1f} < {min_rsi:.1f})"

    adx_5m = ind.get("adx_5m", 0.0)
    if mode == "CHOP" and adx_5m >= max_adx:
        return False, f"ACTIVE_BREAKOUT_ADX (5m ADX={adx_5m:.1f} >= {max_adx:.1f})"

    upper_band = ind["upper_band"]
    has_bearish_reversal = ind.get("has_bearish_reversal", False)
    vwap = ind["vwap"]
    ema20 = ind["ema20"]
    is_red_candle = ind.get("is_red_candle", False)

    if mode == "CHOP":
        if curr_price >= (upper_band * 0.998) and rsi >= rsi_overbought:
            if not has_bearish_reversal:
                return False, f"LACKS_REVERSAL_CONFIRMATION (Wick={ind.get('upper_wick_ratio', 0.0):.1%})"
            return True, f"CHOP_RANGE_TOP_FADE (Price={curr_price} >= Upper={upper_band:.4f}, RSI={rsi:.1f}, ADX={adx_5m:.1f})"
    elif mode == "BEAR":
        bounced_to_resistance = (curr_price >= vwap * 0.997) or (curr_price >= ema20 * 0.998)
        in_relief_rsi_window = (45.0 <= rsi <= 68.0)
        if bounced_to_resistance and in_relief_rsi_window and is_red_candle:
            return True, f"BEAR_RELIEF_BOUNCE_FADE (Price={curr_price}, RSI={rsi:.1f})"

    return False, "CONDITIONS_NOT_MET"


class SuperBotTrader:
    def __init__(self):
        # 1. Configuration & Fleet
        self.symbols = getattr(config, "SYMBOLS", [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT",
            "AVAX/USDT", "XRP/USDT", "ADA/USDT", "BNB/USDT", "NEAR/USDT"
        ])
        self.max_locks = getattr(config, "SWING_MAX_CONCURRENT_LOCKS", 4)
        
        # 2. Capital & Leverage Allocation (70% Wallet Utilization Policy)
        self.swing_allocation_usdt = getattr(config, "SWING_TRADE_ALLOCATION_USDT", 85.0)
        self.grid_allocation_usdt = getattr(config, "GRID_ALLOCATION_USDT", 85.0)
        self.standard_leverage = getattr(config, "SWING_STANDARD_LEVERAGE", 3)
        self.high_quality_leverage = getattr(config, "SWING_HIGH_QUALITY_LEVERAGE", 5)
        self.grid_leverage = getattr(config, "GRID_LEVERAGE", 3)
        
        # 3. Strategy Parameters
        self.grid_tp_pct = getattr(config, "GRID_TP_PCT", 0.0135)     # +1.35% limit target
        self.grid_sl_pct = getattr(config, "GRID_SL_PCT", 0.0115)     # -1.15% tight stop
        self.grid_rsi_thresh = getattr(config, "GRID_RSI_THRESHOLD", 30.0)
        self.grid_max_adx = getattr(config, "GRID_MAX_5M_ADX", 24.0)   # 5m ADX exhaustion ceiling (blocks waterfall dips)
        
        # Quick Scalp Short Parameters (Fast Turnover Protocol V36.6)
        self.scalp_short_enabled = getattr(config, "SCALP_SHORT_ENABLED", True)
        self.scalp_short_tp_pct = getattr(config, "SCALP_SHORT_TP_PCT", 0.0120)     # +1.20% limit target
        self.scalp_short_sl_pct = getattr(config, "SCALP_SHORT_SL_PCT", 0.0100)     # -1.00% tight stop
        self.scalp_short_be_trigger = getattr(config, "SCALP_SHORT_BE_TRIGGER_PCT", 0.0050) # +0.50% profit triggers breakeven ratchet
        self.scalp_short_max_hold_mins = getattr(config, "SCALP_SHORT_MAX_HOLD_MINS", 45.0) # 45m soft harvest/tighten checkpoint
        self.scalp_short_hard_expiry_mins = getattr(config, "SCALP_SHORT_HARD_EXPIRY_MINS", 90.0) # 90m hard time decay limit
        self.scalp_short_rsi_overbought = getattr(config, "SCALP_SHORT_RSI_OVERBOUGHT", 68.0)
        self.scalp_short_min_rsi = getattr(config, "SCALP_SHORT_MIN_RSI", 38.0)     # Strict anti-bottom filter (never short < 38)
        self.max_concurrent_shorts = getattr(config, "MAX_CONCURRENT_SHORTS", 1)   # Strictly max 1 short across fleet
        self.scalp_short_btc_guard = getattr(config, "SCALP_SHORT_BTC_PUMP_GUARD", True) # Veto alt shorts if BTC pumps
        self.scalp_short_require_15m = getattr(config, "SCALP_SHORT_REQUIRE_15M_EXHAUST", True) # 15m exhaustion requirement
        self.scalp_short_max_adx = getattr(config, "SCALP_SHORT_MAX_5M_ADX", 22.0)       # 5m ADX exhaustion ceiling (< 22.0)
        self.scalp_short_max_24h_gain = getattr(config, "SCALP_SHORT_MAX_24H_GAIN_PCT", 3.0) # Ban shorting > +3.0% 24h gainers
        self.scalp_short_max_di_spread = getattr(config, "SCALP_SHORT_MAX_DI_SPREAD", 4.0)   # Mute CHOP shorts when +DI > -DI + 4.0

        self.check_interval_fast = 15   # 15s position management & grid check
        self.check_interval_4h = 300    # 5 min 4H candle scan
        
        # 4. Engine & State
        self.uta_client = None
        self.boardroom = ConvictionBoardroom()
        self.db_queue = DBWriteQueue.get_instance("trading_data.db")
        self._stop_event = asyncio.Event()
        self._background_tasks = set()
        
        # Sub-engine for Swing Architecture + Declarative Strategy Registry (Jesse Pattern)
        self.swing_engine = SwingArchitectTrader()
        self.swing_strategy = SwingKumoStrategy()
        self.grid_strategy = ChopGridMakerStrategy()
        self.scalp_short_strategy = BearScalpShortStrategy()
        
        # Internal state
        self.active_positions = {}  # symbol -> dict
        self.grid_open_orders = {}  # symbol -> order_id
        self.recent_loss_cooldowns = {}  # symbol -> epoch timestamp (Anti-knife local quarantine)
        self.current_regime = "UNKNOWN"
        self.regime_adx = 20.0
        self.regime_hurst = 0.50
        self.market_regime_di_spread = 0.0
        self.turbulence_index = 0.0
        self.turbulence_threshold = 22.0
        self.turbulence_state = "CALM_NORMAL"
        # Fail-closed regime health: new entries are blocked while the regime row is missing,
        # stale, unparseable or non-finite. Position management/exits ignore this flag.
        self.regime_data_ok = False
        self.regime_stale_reason = "no regime read yet"
        self._regime_health_logged = None
        
        self._init_database()

    def _init_database(self):
        """Ensures Super Bot tables and state exist."""
        try:
            with sqlite3.connect("trading_data.db", timeout=10.0) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS super_bot_telemetry (
                        timestamp TEXT PRIMARY KEY,
                        regime TEXT,
                        adx REAL,
                        hurst REAL,
                        active_swing_count INTEGER,
                        active_grid_count INTEGER,
                        total_unrealized_pnl REAL,
                        status TEXT
                    )
                """)
                conn.commit()
            log.info("[DB-INIT] Super Bot telemetry schema initialized.")
        except Exception as e:
            log.error(f"[DB-INIT-ERR] {e}")

    def _track_task(self, coro, name="task"):
        """SRE-P0: Anchors background coroutines in a persistent set to prevent garbage collection."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def initialize_clients(self):
        """Initializes Bitget UTA async exchange client and links to sub-engines + Sentinel."""
        try:
            self.uta_client = BitgetUTAClient()
            self.swing_engine.uta_client = self.uta_client
            if hasattr(self.swing_engine, "sentinel") and self.swing_engine.sentinel:
                self.swing_engine.sentinel.start_background_refresh([s.split("/")[0] for s in self.symbols])
            log.info("[CLIENT] BitgetUTAClient connected and Sentinel background refresh started for Super Bot.")
        except Exception as e:
            log.critical(f"[CLIENT] Failed to initialize BitgetUTAClient: {e}")
            raise

    # ---------------------------------------------------------------------------
    # Live Position Adoption & Management (LINK, SOL, NEAR, etc.)
    # ---------------------------------------------------------------------------
    def _load_all_positions_sync(self):
        """Adopts and syncs active positions from both swing_open_positions and mean_reversion_open_positions."""
        try:
            with sqlite3.connect("trading_data.db", timeout=10.0) as conn:
                conn.row_factory = sqlite3.Row
                
                # 1. Adopt active 4H Swing and Chop Grid positions (including PENDING_CLOSE)
                rows_swing = conn.execute("SELECT * FROM swing_open_positions WHERE status IN ('ACTIVE', 'PENDING_CLOSE')").fetchall()
                active = {}
                for r in rows_swing:
                    pos = dict(r)
                    notes = str(pos.get("notes", "")).lower()
                    if "scalp_short" in notes or "scalp short" in notes or "short scalp" in notes:
                        pos["engine_type"] = "SCALP_SHORT"
                    elif "grid" in notes or "chop" in notes:
                        pos["engine_type"] = "GRID_MAKER"
                    else:
                        pos["engine_type"] = "SWING_4H"
                    active[pos["symbol"]] = pos
                    
                # 2. Adopt active Mean Reversion Grid positions (legacy fallback if any)
                try:
                    rows_grid = conn.execute("SELECT * FROM mean_reversion_open_positions").fetchall()
                    for r in rows_grid:
                        pos = dict(r)
                        pos["engine_type"] = "GRID_MAKER"
                        active[pos["symbol"]] = pos
                except Exception:
                    pass
                    
                self.active_positions = active
                # Mirror ALL active positions to swing engine so manage_active_positions tracks them with ratchets
                self.swing_engine.active_positions = dict(active)
                
        except Exception as e:
            log.error(f"[SYNC-ERR] Error adopting positions: {e}")

    @staticmethod
    def _regime_row_age_sec(raw_ts):
        """Age in seconds of a regime row's publication timestamp; None if unparseable."""
        if not raw_ts:
            return None
        try:
            cleaned = str(raw_ts).strip().replace("Z", "+00:00")
            if "T" not in cleaned and " " in cleaned:
                cleaned = cleaned.replace(" ", "T")
            parsed = datetime.fromisoformat(cleaned)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return max(0.0, time.time() - parsed.timestamp())
        except Exception:
            return None

    @staticmethod
    def _finite_or_none(value):
        try:
            as_float = float(value)
        except (TypeError, ValueError):
            return None
        return as_float if np.isfinite(as_float) else None

    def _note_regime_health(self, is_ok, reason):
        """Log once per health transition rather than on every 5-second routing loop."""
        if self._regime_health_logged == is_ok:
            return
        self._regime_health_logged = is_ok
        if is_ok:
            log.info("[REGIME-GATE] ✅ Regime telemetry valid again — new entries re-enabled.")
        else:
            log.warning(f"[REGIME-GATE] 🛑 Blocking NEW entries: {reason}. Live positions keep being managed and exited normally.")

    def _get_market_regime_sync(self):
        """Reads current market regime & FinRL Mahalanobis Turbulence determined by regime_classifier.py.

        Fail-closed for NEW entries: an absent row, a missing/too-old/unparseable publication
        timestamp, an UNKNOWN label, or non-finite ADX/Hurst/DI telemetry leaves
        self.regime_data_ok False so the router stops opening positions. Trailing stops,
        breakeven ratchets, profit floors and exits are deliberately not gated on it.
        """
        regime = "UNKNOWN"
        adx = 0.0
        hurst = 0.50
        di_spread = 0.0
        turb_idx = 0.0
        turb_thresh = 22.0
        turb_state = "UNKNOWN"
        problem = None

        max_age = self._finite_or_none(getattr(config, "REGIME_MAX_AGE_SEC", None))
        if max_age is None or max_age <= 0:
            max_age = 900.0

        try:
            with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                row = conn.execute(
                    "SELECT regime, adx, hurst, plus_di, minus_di, turbulence_index, turbulence_threshold, turbulence_state, timestamp "
                    "FROM market_regime ORDER BY id DESC LIMIT 1"
                ).fetchone()

            if not row:
                problem = "market_regime table has no rows"
            else:
                regime = str(row[0] or "").strip().upper() or "UNKNOWN"
                adx_val = self._finite_or_none(row[1])
                hurst_val = self._finite_or_none(row[2])
                p_di = self._finite_or_none(row[3])
                m_di = self._finite_or_none(row[4])
                raw_ts = row[8] if len(row) > 8 else None
                age = self._regime_row_age_sec(raw_ts)

                if adx_val is not None:
                    adx = adx_val
                if hurst_val is not None:
                    hurst = hurst_val
                if p_di is not None and m_di is not None:
                    di_spread = p_di - m_di
                if len(row) > 5:
                    tv = self._finite_or_none(row[5])
                    if tv is not None:
                        turb_idx = tv
                    tv = self._finite_or_none(row[6])
                    if tv is not None:
                        turb_thresh = tv
                    if row[7]:
                        turb_state = str(row[7])

                if adx_val is None:
                    problem = f"ADX is missing or non-finite ({row[1]!r})"
                elif not 0.0 <= adx_val <= 100.0:
                    problem = f"ADX out of range ({adx_val:.2f})"
                elif hurst_val is None or not 0.0 <= hurst_val <= 1.0:
                    problem = f"Hurst is missing, non-finite or out of range ({row[2]!r})"
                elif p_di is None or m_di is None:
                    problem = f"+DI/-DI missing or non-finite ({row[3]!r}, {row[4]!r})"
                elif age is None:
                    problem = f"regime timestamp missing or unparseable ({raw_ts!r})"
                elif age > max_age:
                    problem = f"regime row is {age:.0f}s old (max {max_age:.0f}s)"
                elif regime == "UNKNOWN":
                    problem = "regime label is UNKNOWN"
        except Exception as e:
            log.warning(f"[REGIME-READ-FAIL-CLOSED] Error reading market_regime: {e}. Forcing regime=UNKNOWN.")
            regime = "UNKNOWN"
            adx = 0.0
            problem = f"market_regime read failed: {e}"

        self.regime_data_ok = problem is None
        self.regime_stale_reason = problem or "regime telemetry valid"
        self._note_regime_health(self.regime_data_ok, problem)

        self.current_regime = regime
        self.regime_adx = adx
        self.regime_hurst = hurst
        self.market_regime_di_spread = di_spread
        self.turbulence_index = turb_idx
        self.turbulence_threshold = turb_thresh
        self.turbulence_state = turb_state
        return regime, adx, hurst

    def _get_finrl_strategy_weight(self, engine_tag: str) -> float:
        """
        Reads FinRL Rolling Validation Sharpe Strategy Ensemble multiplier from quant_validation_report.json.
        Overweights the highest-Sharpe strategy (up to 1.20x) and throttles low-Sharpe strategies (0.75x-0.90x).
        """
        try:
            report_path = os.path.join("cache", "quant_validation_report.json")
            if os.path.exists(report_path):
                with open(report_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                weights = (data.get("finrl_ensemble_selector") or {}).get("strategy_weights") or {}
                w_info = weights.get(engine_tag) or {}
                mult = float(w_info.get("allocation_weight_mult", 1.0))
                return max(0.75, min(1.25, mult))
        except Exception:
            pass
        return 1.0

    # ---------------------------------------------------------------------------
    # Mean Reversion Maker Engine (CHOP Mode)
    # ---------------------------------------------------------------------------
    async def scan_chop_grid_opportunities(self):
        """Scans unallocated assets during CHOP regime for Maker dip-buys."""
        # Check available capacity
        active_count = len(self.active_positions)
        if active_count >= self.max_locks:
            return

        # FinRL Cross-Asset Mahalanobis Turbulence Circuit Breaker
        if self.turbulence_state == "EXTREME_SHOCK":
            log.warning(
                f"[FINRL-TURBULENCE-SHIELD] 🚨 Cross-asset turbulence breach active "
                f"(d_t={self.turbulence_index:.2f} > {self.turbulence_threshold:.2f}). Chop Grid dip-buying muted."
            )
            return

        # Safeguard 0: Fleet-Wide Long Exposure Cap (Strictly Max 2 Active Longs Across Fleet)
        in_mem_longs = len([p for p in self.active_positions.values() if "long" in str(p.get("side", "long")).lower()])
        db_longs = await asyncio.to_thread(self.swing_engine._count_fleet_long_positions_sync)
        active_longs = max(in_mem_longs, db_longs)
        if active_longs >= 2:
            log.debug(f"[GRID-LONG-CAP-VETO] 🛑 Already holding {active_longs} active LONG position(s) (cap: 2). Blocking correlated long risk.")
            return

        # 0. Global Market Trend & Downtrend Guard
        # If market ADX >= 25.0 and DI spread is negative (sellers dominate), dip-buying is strictly prohibited.
        if self.regime_adx >= 25.0 and self.market_regime_di_spread < -4.0:
            log.info(f"[GRID-MARKET-SHIELD] 🛡️ Market ADX={self.regime_adx:.1f} >= 25.0 with seller dominance (DI spread={self.market_regime_di_spread:+.1f}). Chop Grid dip-buys muted.")
            return

        if self.regime_adx >= 28.0:
            log.info(f"[GRID-MARKET-SHIELD] 🛡️ Market ADX={self.regime_adx:.1f} >= 28.0 indicates powerful directional expansion. Chop Grid muted.")
            return

        # Eligible symbols: not currently locked
        unlocked = [s for s in self.symbols if s not in self.active_positions]
        if not unlocked:
            return

        # Only skip symbols explicitly placed in emergency CLOSE_ONLY posture
        close_only_set = set()
        try:
            with sqlite3.connect("trading_data.db", timeout=5.0) as s_conn:
                row_matrix = s_conn.execute("SELECT value FROM system_state WHERE key = 'ai_decoupled_regime_matrix'").fetchone()
                if row_matrix and row_matrix[0]:
                    matrix_data = json.loads(row_matrix[0])
                    for sym_k, sym_v in matrix_data.items():
                        if isinstance(sym_v, dict) and sym_v.get("posture") == "CLOSE_ONLY":
                            close_only_set.add(sym_k)
        except Exception as st_e:
            log.debug(f"[GRID-STATE-CHECK-NOTE] {st_e}")

        # 1. Filter out symbols under active post-stop cooldown
        now_iso = datetime.now(UTC).isoformat()
        now_t = time.time()
        eligible_candidates = []
        for s in unlocked:
            # Check local in-memory loss cooldown
            if s in self.recent_loss_cooldowns and now_t < self.recent_loss_cooldowns[s]:
                rem_mins = (self.recent_loss_cooldowns[s] - now_t) / 60.0
                log.debug(f"[GRID-COOLDOWN-GATE] ⏳ Skipping {s}: local loss cooldown active ({rem_mins:.1f}m remaining)")
                continue

            # Check database swing_cooldowns (for LONG entries only)
            cooldown_until = await asyncio.to_thread(self.swing_engine._get_cooldown_sync, s, "LONG")
            if cooldown_until and cooldown_until > now_iso:
                log.debug(f"[GRID-COOLDOWN-GATE] ⏳ Skipping {s}: active cool-off until {cooldown_until} (Anti-Churn Protection)")
                continue

            if s in close_only_set:
                log.debug(f"[GRID-CLOSE-ONLY] 🛑 Skipping {s}: marked CLOSE_ONLY in macro matrix")
                continue

            eligible_candidates.append(s)

        if not eligible_candidates:
            return

        # 2. Dynamic Relative Strength Ranking: Sort candidates by 24h performance descending
        # Prioritizes momentum leaders (NEAR, ADA, AVAX, LINK) over laggards
        try:
            clean_syms = [self.uta_client._normalize_symbol(s) for s in eligible_candidates]
            tickers = await self.uta_client.exchange.fetch_tickers(clean_syms)
            eligible_candidates.sort(
                key=lambda s: float((tickers.get(self.uta_client._normalize_symbol(s)) or {}).get("percentage", 0.0) or 0.0),
                reverse=True
            )
            top_leaders = ", ".join([f"{s.split('/')[0]}: {(tickers.get(self.uta_client._normalize_symbol(s)) or {}).get('percentage', 0.0):+.2f}%" for s in eligible_candidates[:3]])
            log.debug(f"[GRID-RS-RANKING] Candidates sorted by 24h RS: {top_leaders}")
        except Exception as rs_err:
            log.debug(f"[GRID-RS-NOTE] RS sorting fallback: {rs_err}")

        for symbol in eligible_candidates:
            try:
                clean_sym = self.uta_client._normalize_symbol(symbol)
                # Pull recent 5m bars
                candles = await self.uta_client.exchange.fetch_ohlcv(clean_sym, timeframe="5m", limit=60)
                if not candles or len(candles) < 30:
                    continue
                    
                df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                ind = compute_grid_5m_indicators(df)
                if not ind:
                    continue
                
                curr_price = ind["curr_price"]
                rsi = ind["rsi"]
                lower_band = ind["lower_band"]

                is_sig, sig_reason = evaluate_chop_grid_signal(
                    ind=ind,
                    curr_price=curr_price,
                    rsi_thresh=self.grid_rsi_thresh,
                    max_adx=getattr(self, "grid_max_adx", 24.0),
                    regime_adx=self.regime_adx,
                    di_spread=self.market_regime_di_spread
                )

                if is_sig:
                    # Microstructure & Markov Gates for Dip-Buy
                    from microstructure_guard import MicrostructureStateCache
                    ms_state = MicrostructureStateCache.get_state(clean_sym)
                    tox_halt = bool(ms_state.get("toxicity_halt", False))
                    vpin_val = float(ms_state.get("vpin", 0.0) or 0.0)
                    micro_p = float(ms_state.get("micro_price", 0.0) or curr_price)
                    if micro_p <= 0.0:
                        micro_p = curr_price

                    # N-3 FIX: If microstructure state is empty (missing key or stale daemon), apply defensive sizing
                    ms_is_degraded = (not ms_state)
                    if ms_is_degraded:
                        log.info(f"[GRID-MS-DEGRADED] Microstructure state for {clean_sym} is missing or stale (>60s). Scaling margin 50% for defense.")

                    # P0-3 FIX: hawkes_rho is mathematically frozen (0.30314) due to static branching ratios.
                    # Excised rho comparison from entry veto path; hawkes_spoof_index and VPIN remain live controls.
                    hawkes_spoof = float(ms_state.get("hawkes_spoof_index", 0.0) or 0.0)
                    hawkes_veto = bool(ms_state.get("hawkes_veto", False))
                    spoof_cut = float(getattr(config, "HAWKES_SPOOF_VETO_THRESHOLD", 0.78))

                    if hawkes_veto or hawkes_spoof >= spoof_cut:
                        log.warning(f"[GRID-HAWKES-VETO] 🚨 Order flow spoofing detected on {symbol}: spoof={hawkes_spoof:.2f}>={spoof_cut}, veto_flag={hawkes_veto}. Suppressing dip-buy.")
                        continue

                    if tox_halt or vpin_val >= 0.68:
                        log.warning(f"[GRID-VPIN-ADVISORY] ⚠️ Microstructure Toxicity alert for {symbol}: tox_halt={tox_halt}, VPIN={vpin_val:.2f} (Advisory log-only until bucket calibration).")
                        # Held advisory / log-only to prevent false-positive lockouts on low-unit alts


                    # Markov Regime Cascade & Tradability Duration Screen (Playbook Step 9)
                    def _check_markov_telemetry():
                        try:
                            max_age = float(getattr(config, "REGIME_MAX_AGE_SEC", 900.0) or 900.0)
                        except (TypeError, ValueError):
                            max_age = 900.0
                        try:
                            with sqlite3.connect("trading_data.db", timeout=3.0) as conn:
                                row = conn.execute("SELECT value, updated_at FROM system_state WHERE key = 'markov_regime_telemetry'").fetchone()
                                if row and row[0]:
                                    raw_val, raw_ts = row[0], row[1]
                                    age = self._regime_row_age_sec(raw_ts)
                                    if age is not None and age > max_age:
                                        log.warning(f"[GRID-MARKOV-STALE] ⚠️ markov_regime_telemetry is stale ({age:.1f}s > {max_age:.1f}s). Skipping Markov gate.")
                                        return {}
                                    parsed = json.loads(raw_val)
                                    if "is_tradable" not in parsed:
                                        age_str = f"{age:.1f}s" if age is not None else "unknown"
                                        log.warning(f"[GRID-MARKOV-KEY-MISSING] ⚠️ markov_regime_telemetry missing 'is_tradable' key (age: {age_str}).")
                                    if "sizing_multiplier" not in parsed:
                                        age_str = f"{age:.1f}s" if age is not None else "unknown"
                                        log.warning(f"[GRID-MARKOV-KEY-MISSING] ⚠️ markov_regime_telemetry missing 'sizing_multiplier' key (age: {age_str}).")
                                    return parsed
                        except Exception:
                            pass
                        return {}
                    m_data = await asyncio.to_thread(_check_markov_telemetry)
                    if m_data:
                        # 1. Bear cascade gate
                        p_bear = float(m_data.get("state_probabilities", {}).get("BEAR_CASCADE", 0.0) or 0.0)
                        if p_bear >= 0.55:
                            log.warning(f"[GRID-VETO] 🛑 Markov Bear Cascade Probability {p_bear:.2f} >= 0.55 for {symbol}. Skipping dip-buy into cascade.")
                            continue

                        # 2. Tradability duration gate (anti-churn fee shield)
                        is_tradable = m_data.get("is_tradable", True)
                        if not is_tradable:
                            tradable_rsn = m_data.get("tradability_reason", "Duration too short")
                            log.warning(f"[GRID-VETO] 🛑 Markov Tradability Screen Veto for {symbol}: {tradable_rsn}. Blocking entry to prevent fee churn.")
                            continue

                    # Maker entry pricing: align with micro-price (or current price if lower)
                    maker_entry_price = min(curr_price, micro_p)

                    log.info(f"[GRID-OPPORTUNITY] [SIGNAL] {symbol} Oversold Dip! Price: ${curr_price:,.4f} | MicroPrice: ${micro_p:,.4f}")
                    
                    conv_score = 0.75
                    rationale = f"Maker Grid dip: RSI {rsi:.1f}, MicroPrice ${micro_p:,.4f}, Entry ${maker_entry_price:,.4f}, TP +{self.grid_tp_pct*100:.2f}%"
                    log.info(f"[GRID-APPROVED] ⚡ {symbol} Approved via Maker Engine (Score: {conv_score:.2f} | {rationale})")
                        
                    # Execute Post-Only Maker Order
                    await self._execute_grid_maker_entry(
                        symbol=symbol,
                        current_price=maker_entry_price,
                        rsi=rsi,
                        conv_score=conv_score,
                        rationale=rationale,
                        ms_is_degraded=ms_is_degraded
                    )
                    break  # One entry per cycle
                    
            except Exception as e:
                log.debug(f"[GRID-SCAN-NOTE] {symbol}: {e}")

    async def _execute_grid_maker_entry(
        self,
        symbol: str,
        current_price: float,
        rsi: float = 0.0,
        conv_score: float = 0.0,
        rationale: str = "",
        side: str = "long",
        ms_is_degraded: bool = False
    ):
        """Places a post-only limit buy or sell order on Bitget UTA using dynamic 70% wallet allocation."""
        try:
            side = side.lower()
            is_long = (side == "long")
            pos_side = "long" if is_long else "short"
            opp_side = "sell" if is_long else "buy"
            order_verb = "Buy" if is_long else "Sell"
            engine_tag = "GRID_MAKER" if is_long else "SCALP_SHORT"

            # P0-4 DEFENSE: Check if Portfolio Risk Agent or Operator has activated RISK_HALT / OPERATOR_HALT (Fail-Closed)
            def _check_risk_halt():
                with sqlite3.connect("trading_data.db", timeout=5.0) as rh_conn:
                    rh_rows = rh_conn.execute(
                        "SELECT value FROM system_state WHERE key IN ('RISK_HALT', 'OPERATOR_HALT')"
                    ).fetchall()
                    return any(r and str(r[0]).strip().upper() in ("1", "TRUE") for r in rh_rows)

            try:
                if await asyncio.to_thread(_check_risk_halt):
                    log.warning(f"[GRID-VETO] RISK_HALT / OPERATOR_HALT active in system_state! Blocking new {pos_side.upper()} entry for {symbol}.")
                    return
            except Exception as rhe:
                log.warning(f"[GRID-RISK-HALT-VETO] Failed to read RISK_HALT state (fail-closed) for {symbol}: {rhe}")
                return

            # Check if another fleet engine (e.g. swing_architect) currently holds an active fresh lock
            def _check_asset_lock():
                with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                    row = conn.execute("SELECT locked_by, heartbeat FROM asset_locks WHERE symbol = ?", (symbol,)).fetchone()
                    if row and row[0]:
                        holder = str(row[0])
                        if holder not in ('super_bot_grid', 'super_bot_scalp_short'):
                            hb_str = str(row[1] or "")
                            try:
                                clean_hb = hb_str.replace("Z", "+00:00")
                                if "+" not in clean_hb: clean_hb += "+00:00"
                                hb_dt = datetime.fromisoformat(clean_hb)
                                if (datetime.now(timezone.utc) - hb_dt).total_seconds() <= 900.0:
                                    return holder, hb_str
                            except Exception:
                                return holder, hb_str
                    return None, None

            locked_by, locked_hb = await asyncio.to_thread(_check_asset_lock)
            if locked_by:
                log.warning(f"[{engine_tag}-LOCK-VETO] Cannot enter {symbol}: Already locked by '{locked_by}' (heartbeat: {locked_hb}). Aborting entry.")
                return

            # Unified Sentinel News & Sentiment Gate (2h max-age, fail-closed, WARNING=0.5x size-down)
            news_ok, news_rsn = await self.swing_engine._check_news_veto(
                symbol, direction="LONG" if is_long else "SHORT"
            )
            if not news_ok:
                log.warning(f"[{engine_tag}-NEWS-VETO] {symbol} ({pos_side.upper()}) blocked: {news_rsn}")
                return

            # Enforce Fleet-Wide Long Cap (Strictly Max 2 Active Longs Across Fleet)
            if is_long:
                in_mem_longs = len([p for p in self.active_positions.values() if "long" in str(p.get("side", "long")).lower()])
                db_longs = await asyncio.to_thread(self.swing_engine._count_fleet_long_positions_sync)
                if max(in_mem_longs, db_longs) >= 2:
                    log.warning(f"[GRID-LONG-CAP-VETO] 🛑 Fleet already holds >= 2 active LONG positions. Vetoing {symbol} LONG execution.")
                    return

            # Dynamic Balance Sizing: 70% Wallet Utilization Protocol
            max_wallet_margin_pct = getattr(config, "GLOBAL_MARGIN_LIMIT", 70.0) / 100.0  # 0.70
            margin = self.grid_allocation_usdt
            try:
                live_bal = await self.uta_client.fetch_balance()
                if live_bal and live_bal > 10.0:
                    total_locked_margin = sum(
                        (float(p.get("size", 0.0)) * float(p.get("entry_price", 0.0))) / max(1.0, float(p.get("leverage", 3.0)))
                        for p in self.active_positions.values()
                    )
                    max_allowed_margin = live_bal * max_wallet_margin_pct
                    available_margin_buffer = max(0.0, max_allowed_margin - total_locked_margin)

                    if available_margin_buffer < 5.0:
                        log.warning(f"[GRID-MARGIN-CEILING] 70% Wallet Utilization reached! Locked: ${total_locked_margin:.2f} / Max: ${max_allowed_margin:.2f} ({total_locked_margin/live_bal*100:.1f}%). Vetoing {pos_side.upper()} entry.")
                        return

                    target_per_trade = live_bal * (max_wallet_margin_pct / max(1, self.max_locks))
                    
                    # Empirical Growth-Optimal Kelly Sizing + FinRL Rolling Sharpe Ensemble Weight + Markov Regime Multiplier
                    try:
                        from risk_manager import RiskManager
                        rm = RiskManager()
                        kelly_frac = rm.calculate_empirical_kelly_fraction()
                        conv_mult = max(0.5, min(1.2, conv_score))
                        finrl_mult = self._get_finrl_strategy_weight(engine_tag)
                        
                        # Continuous Markov Risk Multiplier (Playbook Step 8)
                        markov_mult = 1.0
                        def _get_markov_mult():
                            try:
                                max_age = float(getattr(config, "REGIME_MAX_AGE_SEC", 900.0) or 900.0)
                            except (TypeError, ValueError):
                                max_age = 900.0
                            try:
                                with sqlite3.connect("trading_data.db", timeout=3.0) as conn:
                                    row = conn.execute("SELECT value, updated_at FROM system_state WHERE key = 'markov_regime_telemetry'").fetchone()
                                    if row and row[0]:
                                        raw_val, raw_ts = row[0], row[1]
                                        age = self._regime_row_age_sec(raw_ts)
                                        if age is not None and age > max_age:
                                            log.warning(f"[{engine_tag}-MARKOV-STALE] ⚠️ markov_regime_telemetry is stale ({age:.1f}s > {max_age:.1f}s). Defaulting sizing_multiplier to 1.0.")
                                            return 1.0
                                        m_d = json.loads(raw_val)
                                        if "sizing_multiplier" not in m_d:
                                            age_str = f"{age:.1f}s" if age is not None else "unknown"
                                            log.warning(f"[{engine_tag}-MARKOV-KEY-MISSING] ⚠️ markov_regime_telemetry missing 'sizing_multiplier' key (age: {age_str}). Defaulting to 1.0.")
                                            return 1.0
                                        return float(m_d.get("sizing_multiplier", 1.0) or 1.0)
                            except Exception:
                                pass
                            return 1.0
                        markov_mult = await asyncio.to_thread(_get_markov_mult)
                        
                        target_kelly_margin = live_bal * kelly_frac * conv_mult * finrl_mult * markov_mult
                        target_per_trade = max(target_per_trade * 0.7, min(target_per_trade * 1.35, target_kelly_margin))
                        log.info(f"[{engine_tag}-KELLY] Empirical Kelly: {kelly_frac:.2%} | FinRL: {finrl_mult:.2f}x | Markov Mult: {markov_mult:.2f}x -> Target Margin: ${target_per_trade:.2f}")
                    except Exception as ke:
                        log.debug(f"[{engine_tag}-KELLY-NOTE] {ke}")

                    margin = max(6.0, min(round(target_per_trade, 2), round(available_margin_buffer, 2)))
            except Exception as be:
                log.warning(f"[{engine_tag}-BALANCE-NOTE] Using fallback allocation (${margin:,.2f} USDT): {be}")

            # Apply Sentinel WARNING size-down multiplier (0.5x on caution news)
            news_mult = float(getattr(self.swing_engine, "_news_size_multipliers", {}).get(symbol, 1.0))
            if 0.0 < news_mult < 1.0:
                prev_margin = margin
                margin = max(6.0, round(margin * news_mult, 2))
                log.warning(f"[{engine_tag}-NEWS-SIZE-DOWN] {symbol} WARNING tier active -> scaling margin ${prev_margin:.2f} -> ${margin:.2f} ({news_mult:.2f}x)")

            # P0-4 DEFENSE: Apply 50% margin haircut if microstructure state is missing or stale
            if ms_is_degraded:
                prev_margin = margin
                margin = max(6.0, round(margin * 0.50, 2))
                log.warning(f"[{engine_tag}-MS-DEGRADED-SIZING] {symbol} Microstructure degraded -> scaling margin ${prev_margin:.2f} -> ${margin:.2f} (0.50x)")

            notional = margin * self.grid_leverage
            qty = self.uta_client.format_qty_precision(symbol, notional / current_price)
            
            # Formulate TP & SL
            if is_long:
                tp_price = current_price * (1.0 + self.grid_tp_pct)
                sl_price = current_price * (1.0 - self.grid_sl_pct)
            else:
                tp_price = current_price * (1.0 - self.scalp_short_tp_pct)
                sl_price = current_price * (1.0 + self.scalp_short_sl_pct)
            
            log.info(f"[{engine_tag}] [POST-ONLY-{order_verb.upper()}] Placing True Post-Only Maker Limit {order_verb} for {symbol}: {qty:.4f} units @ ${current_price:,.4f}")
            
            # Place TRUE Maker Post-Only order via UTA client (guarantees maker rebate and zero taker fee crossing)
            order = await self.uta_client.create_post_only_order(
                symbol=symbol,
                side=pos_side,
                qty=qty,
                price=current_price,
                leverage_val=self.grid_leverage
            )
            order_id = order.get("id") or order.get("orderId") if order else None
            if not order_id:
                log.warning(f"[{engine_tag}] Maker post-only order placement rejected or crossed spread on {symbol}.")
                return

            log.info(f"[{engine_tag}] Maker order {order_id} placed for {symbol}. Monitoring fill queue (30s TTL)...")
            start_t = time.time()
            is_filled = False
            actual_entry = current_price

            while (time.time() - start_t) < 30.0:
                await asyncio.sleep(1.5)
                try:
                    info = await self.uta_client.fetch_order(symbol, order_id)
                    st = str(info.get("order_status") or info.get("status") or info.get("state") or "").lower()
                    f_qty = float(info.get("filled_qty") or info.get("cumExecQty") or 0.0)
                    if st in ("filled", "closed") or (qty > 0 and f_qty >= qty * 0.95):
                        is_filled = True
                        if f_qty > 0:
                            qty = self.uta_client.format_qty_precision(symbol, f_qty)
                        deal_p = float(info.get("deal_avg_price") or info.get("avgPrice") or 0.0)
                        if deal_p > 0:
                            actual_entry = deal_p
                        log.info(f"[{engine_tag}-FILLED] Maker order {order_id} FILLED on {symbol}! Qty: {qty} @ ${actual_entry:,.4f}")
                        break
                    elif st in ("canceled", "cancelled"):
                        log.warning(f"[{engine_tag}-CANCELED] Order {order_id} cancelled by exchange (post-only protect).")
                        break
                except Exception as pe:
                    log.debug(f"[{engine_tag}-POLL-NOTE] {pe}")

            if not is_filled:
                log.info(f"[{engine_tag}-TIMEOUT] Order {order_id} for {symbol} not filled in 30s. Cancelling on Bitget...")
                try:
                    await self.uta_client.cancel_order(symbol, order_id)
                except Exception as ce:
                    log.debug(f"[{engine_tag}-CANCEL-NOTE] {ce}")

                # Late-fill safety check: Check if order executed right before cancellation
                try:
                    live_pos = await self.uta_client.fetch_positions(symbol=symbol)
                    matching = [p for p in live_pos if float(p.get("total", 0.0) or 0.0) > 0 and (p.get("posSide") or p.get("holdSide")) == pos_side]
                    if matching:
                        log.info(f"[{engine_tag}-LATE-FILL] Order {order_id} for {symbol} executed on Bitget! Adopting live position.")
                        is_filled = True
                        actual_entry = float(matching[0].get("avgPrice") or actual_entry)
                        qty = self.uta_client.format_qty_precision(symbol, float(matching[0].get("total") or qty))
                except Exception as le:
                    log.debug(f"[{engine_tag}-LATE-FILL-NOTE] {le}")

                if not is_filled:
                    return

            # Formulate TP & SL accurately based on ACTUAL execution fill price (not pre-fill quote)
            if is_long:
                tp_price = actual_entry * (1.0 + self.grid_tp_pct)
                sl_price = actual_entry * (1.0 - self.grid_sl_pct)
            else:
                tp_price = actual_entry * (1.0 - self.scalp_short_tp_pct)
                sl_price = actual_entry * (1.0 + self.scalp_short_sl_pct)

            # Register native strategy stop loss on Bitget with aggressive retry
            sl_attached = False
            last_se = None
            sl_qty_str = self.uta_client.format_quantity_precision(symbol, qty) if hasattr(self.uta_client, "format_quantity_precision") else str(self.uta_client.format_qty_precision(symbol, qty))
            for sl_att in range(3):
                try:
                    sl_str = self.uta_client.format_price_precision(symbol, sl_price)
                    sl_p = {
                        "category": "USDT-FUTURES",
                        "symbol": self.uta_client._normalize_symbol(symbol),
                        "marginMode": getattr(config, "MARGIN_MODE", "isolated").lower(),
                        "marginCoin": "USDT",
                        "planType": "pos_loss",
                        "posSide": pos_side,
                        "side": opp_side,
                        "triggerPrice": sl_str,
                        "stopLoss": sl_str,
                        "triggerType": "mark_price",
                        "size": sl_qty_str
                    }
                    await self.uta_client.exchange.private_uta_post_v3_trade_place_strategy_order(sl_p)
                    log.info(f"[{engine_tag}-SHIELD] Native Bitget Stop Loss attached at ${sl_price:,.4f} (Qty: {sl_qty_str})")
                    sl_attached = True
                    break
                except Exception as se:
                    last_se = se
                    log.warning(f"[{engine_tag}-SHIELD-NOTE] Stop loss attach attempt {sl_att+1}/3 failed: {se}")
                    await asyncio.sleep(0.5 * (sl_att + 1))

            if not sl_attached:
                log.critical(f"[{engine_tag}-SHIELD] 🚨 Native Stop Loss failed on {symbol} after 3 retries ({last_se})! Aborting position to eliminate naked capital exposure.")
                try:
                    await self.uta_client.close_position_order(symbol=symbol, side=pos_side, qty=qty)
                    log.warning(f"[{engine_tag}-SHIELD] Emergency position abort SUCCESS for {symbol}.")
                except Exception as ab_err:
                    log.error(f"[{engine_tag}-SHIELD] Failed to abort position on {symbol}: {ab_err}")
                return

            # Register native strategy Take Profit (pos_profit) on Bitget so TP executes immediately on-exchange
            if tp_price and tp_price > 0.0:
                try:
                    tp_str = self.uta_client.format_price_precision(symbol, tp_price)
                    tp_p = {
                        "category": "USDT-FUTURES",
                        "symbol": self.uta_client._normalize_symbol(symbol),
                        "marginMode": getattr(config, "MARGIN_MODE", "isolated").lower(),
                        "marginCoin": "USDT",
                        "planType": "pos_profit",
                        "posSide": pos_side,
                        "side": opp_side,
                        "triggerPrice": tp_str,
                        "takeProfit": tp_str,
                        "triggerType": "mark_price",
                        "size": sl_qty_str
                    }
                    await self.uta_client.exchange.private_uta_post_v3_trade_place_strategy_order(tp_p)
                    log.info(f"[{engine_tag}-TARGET] Native Bitget Take Profit attached at ${tp_price:,.4f} (Qty: {sl_qty_str})")
                except Exception as tpe:
                    log.debug(f"[{engine_tag}-TARGET-NOTE] Native TP attach note on {symbol}: {tpe}")

            now_iso = datetime.now(UTC).isoformat()
            init_risk = (actual_entry - sl_price) * qty if is_long else (sl_price - actual_entry) * qty
            intended_p = float(current_price)
            actual_fill_p = float(actual_entry)
            slippage_bps = ((actual_fill_p - intended_p) / intended_p * 10000.0) if intended_p > 0 else 0.0
            fee_entry_usd = actual_fill_p * qty * 0.0002  # Maker fee 0.02%
            grid_gates = {
                "engine": engine_tag,
                "5m_rsi": round(float(rsi), 2),
                "conviction_score": round(float(conv_score), 2),
                "rationale": str(rationale),
                "intended_entry": intended_p,
                "order_type": "post_only_limit"
            }
            grid_gates_json = json.dumps(grid_gates)

            notes_str = (
                f"Chop Maker Grid [Post-Only Limit] (TP: +{self.grid_tp_pct*100:.1f}%)"
                if is_long
                else f"Quick Scalp Short [Post-Only Limit] (TP: +{self.scalp_short_tp_pct*100:.1f}%)"
            )
            pos_record = {
                "symbol": symbol,
                "side": pos_side,
                "entry_price": actual_fill_p,
                "size": qty,
                "stop_loss": sl_price,
                "take_profit": tp_price,
                "trailing_stop": sl_price,
                "peak_price": actual_fill_p,
                "leverage": self.grid_leverage,
                "opened_at": now_iso,
                "updated_at": now_iso,
                "status": "ACTIVE",
                "r_multiple": 0.0,
                "initial_risk_usd": init_risk,
                "notes": notes_str,
                "engine_type": engine_tag,
                "intended_entry_price": intended_p,
                "actual_fill_price": actual_fill_p,
                "slippage_bps": slippage_bps,
                "fee_entry_usd": fee_entry_usd,
                "execution_type": "maker",
                "gates_snapshot": grid_gates_json
            }
            
            # Record in swing_open_positions for unified management
            def _save():
                with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO swing_open_positions 
                        (symbol, side, entry_price, size, stop_loss, take_profit, trailing_stop, peak_price,
                         leverage, opened_at, updated_at, status, r_multiple, initial_risk_usd, notes,
                         intended_entry_price, actual_fill_price, slippage_bps, fee_entry_usd, execution_type, gates_snapshot)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol, pos_side, actual_fill_p, qty, sl_price, tp_price, sl_price, actual_fill_p,
                        self.grid_leverage, now_iso, now_iso, "ACTIVE", 0.0, pos_record["initial_risk_usd"], pos_record["notes"],
                        intended_p, actual_fill_p, slippage_bps, fee_entry_usd, "maker", grid_gates_json
                    ))
                    # Acquire exclusive lock
                    lock_who = 'super_bot_grid' if is_long else 'super_bot_scalp_short'
                    lock_why = 'CHOP_GRID_LONG' if is_long else 'SCALP_SHORT'
                    conn.execute("""
                        INSERT OR REPLACE INTO asset_locks (symbol, locked_by, locked_at, reason, heartbeat)
                        VALUES (?, ?, ?, ?, ?)
                    """, (symbol, lock_who, now_iso, lock_why, now_iso))
                    conn.commit()
                    
            await asyncio.to_thread(_save)
            self.active_positions[symbol] = pos_record
            log.info(f"[{engine_tag}-ACTIVE] [OK] {symbol} {pos_side.upper()} trade registered and locked.")
                
            # Dispatch Trade Open Alert Email
            try:
                from email_dispatcher import send_trade_log_async
                trade_ctx = {
                    "symbol": symbol,
                    "side": pos_side.upper(),
                    "price": current_price,
                    "qty": qty,
                    "stop_loss": sl_price,
                    "take_profit": tp_price,
                    "leverage": self.grid_leverage,
                    "engine": "Super Bot (Chop Maker Grid)" if is_long else "Super Bot (Quick Scalp Short)",
                    "ai_reasoning": (
                        f"Mean-Reversion Oversold Dip-Buy. RSI: {rsi:.1f} <= {self.grid_rsi_thresh}. AI Boardroom Conviction: {conv_score:.2f} — {rationale}"
                        if is_long
                        else f"Quick Turnover Scalp Short. RSI: {rsi:.1f}. {rationale}"
                    ),
                    "notes": pos_record["notes"]
                }
                t_em = asyncio.create_task(send_trade_log_async(
                    symbol=symbol,
                    side=pos_side.upper(),
                    price=current_price,
                    status="OPENED",
                    trade_context=trade_ctx
                ))
                self._background_tasks.add(t_em)
                t_em.add_done_callback(self._background_tasks.discard)
                log.info(f"[{engine_tag}-EMAIL] Dispatched trade OPEN alert email for {symbol} {pos_side.upper()}.")
            except Exception as em_err:
                log.error(f"[{engine_tag}-EMAIL-ERR] Failed to dispatch open email for {symbol}: {em_err}")
                
        except Exception as e:
            log.error(f"[EXEC-ERR] Failed to execute maker entry for {symbol} ({side}): {e}")

    # ---------------------------------------------------------------------------
    # Quick Scalp Short Engine (Fast Turnover Protocol V36.6)
    # ---------------------------------------------------------------------------
    async def scan_quick_scalp_short_opportunities(self, mode: str = "CHOP"):
        """
        Scans unallocated assets for high-expectancy Quick Scalp Shorts.
        Never holds long-term swing shorts — strictly rapid turnover:
          - Fast TP: +0.90% limit target
          - Tight SL: 1.35% with native Bitget stop-loss attached
          - Fast Breakeven Ratchet: at +0.40% gain, locks SL to Entry - 0.05%
          - Hard Time-Decay Stop: 45-minute maximum lifetime to guarantee zero lingering short exposure.
        
        Two Entry Modes:
          1. mode == 'CHOP': Overbought Range Top Fade (VWAP + 2.0 std upper band touch, RSI >= 68)
          2. mode == 'BEAR': Relief Bounce Fade in Bear Trend (RSI 48-68 relief bounce into VWAP/EMA20)
        """
        if not self.scalp_short_enabled:
            return

        # Check available capacity
        active_count = len(self.active_positions)
        if active_count >= self.max_locks:
            return

        # Safeguard 0: Global Market Bullish Polarity & Expansion Shield (Symmetry with Long Grid Shield)
        # In CHOP mode, if market +DI dominates -DI by > 4.0 points or ADX >= 28.0, range-top shorting is strictly prohibited.
        if mode == "CHOP":
            if self.market_regime_di_spread > self.scalp_short_max_di_spread:
                log.info(f"[SCALP-SHORT-DI-SHIELD] 🛡️ Market +DI dominates -DI (DI spread={self.market_regime_di_spread:+.1f} > +{self.scalp_short_max_di_spread:.1f}). Chop Scalp Shorts muted.")
                return
            if self.regime_adx >= 28.0:
                log.info(f"[SCALP-SHORT-ADX-SHIELD] 🛡️ Market ADX={self.regime_adx:.1f} >= 28.0 indicates directional expansion. Chop Scalp Shorts muted.")
                return

        # Safeguard 1: Fleet Short Exposure Cap (Strictly Max 1 Active Short Across Fleet)
        # Prevents correlated "herd shorting" where multiple altcoins enter simultaneously during a relief bounce
        active_shorts = len([p for p in self.active_positions.values() if str(p.get("side", "")).lower() == "short" or p.get("engine_type") == "SCALP_SHORT"])
        if active_shorts >= self.max_concurrent_shorts:
            log.debug(f"[SCALP-SHORT-CAP-VETO] Already holding {active_shorts} active short(s) (cap: {self.max_concurrent_shorts}). Blocking concurrent short risk.")
            return

        # Eligible symbols: not currently locked
        unlocked = [s for s in self.symbols if s not in self.active_positions]
        if not unlocked:
            return

        # Check if Macro Sentinel has marked symbols as AVOID or global risk halt
        decoupled_avoid_set = set()
        try:
            with sqlite3.connect("trading_data.db", timeout=5.0) as s_conn:
                row_matrix = s_conn.execute("SELECT value FROM system_state WHERE key = 'ai_decoupled_regime_matrix'").fetchone()
                if row_matrix and row_matrix[0]:
                    matrix_data = json.loads(row_matrix[0])
                    for sym_k, sym_v in matrix_data.items():
                        if isinstance(sym_v, dict) and (sym_v.get("tier") == "AVOID" or sym_v.get("posture") == "CLOSE_ONLY"):
                            decoupled_avoid_set.add(sym_k)
        except Exception as st_e:
            log.debug(f"[SCALP-SHORT-STATE-NOTE] {st_e}")

        now_iso = datetime.now(UTC).isoformat()
        now_t = time.time()
        eligible_candidates = []
        for s in unlocked:
            # Check local in-memory loss cooldown
            if s in self.recent_loss_cooldowns and now_t < self.recent_loss_cooldowns[s]:
                rem_mins = (self.recent_loss_cooldowns[s] - now_t) / 60.0
                log.debug(f"[SCALP-COOLDOWN-GATE] ⏳ Skipping {s}: local loss cooldown active ({rem_mins:.1f}m remaining)")
                continue

            # Check database swing_cooldowns
            cooldown_until = await asyncio.to_thread(self.swing_engine._get_cooldown_sync, s)
            if cooldown_until and cooldown_until > now_iso:
                log.debug(f"[SCALP-COOLDOWN-GATE] ⏳ Skipping {s}: active cool-off until {cooldown_until}")
                continue

            # Decoupled matrix AVOID / CLOSE_ONLY check
            if s in decoupled_avoid_set:
                log.debug(f"[SCALP-DECOUPLED-VETO] 🛑 Skipping {s}: marked AVOID/CLOSE_ONLY in macro matrix")
                continue

            eligible_candidates.append(s)

        if not eligible_candidates:
            return

        # Safeguard 2: Bitcoin Macro Anchor Veto for Shorts
        # If Bitcoin (the market leader) is actively in a 5m bullish push, strictly veto shorting altcoins
        if self.scalp_short_btc_guard:
            try:
                btc_clean = self.uta_client._normalize_symbol("BTC/USDT")
                btc_candles = await self.uta_client.exchange.fetch_ohlcv(btc_clean, timeframe="5m", limit=30)
                if btc_candles and len(btc_candles) >= 20:
                    btc_df = pd.DataFrame(btc_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                    btc_c = btc_df["close"]
                    btc_curr = float(btc_c.iloc[-1])
                    btc_cum_pv = (btc_c * btc_df["volume"]).rolling(24).sum()
                    btc_cum_v = btc_df["volume"].rolling(24).sum()
                    btc_vwap = float((btc_cum_pv / (btc_cum_v + 1e-9)).iloc[-1])
                    btc_ema9 = float(btc_c.ewm(span=9, adjust=False).mean().iloc[-1])
                    btc_ema21 = float(btc_c.ewm(span=21, adjust=False).mean().iloc[-1])
                    
                    if btc_curr > (btc_vwap * 1.001) and btc_ema9 > btc_ema21:
                        log.info(f"[SCALP-SHORT-BTC-VETO] 🛑 Bitcoin is in an active 5m bullish impulse (${btc_curr:,.1f} > VWAP ${btc_vwap:,.1f} & EMA9 > EMA21). Altcoin shorts strictly vetoed.")
                        return
            except Exception as btc_e:
                log.debug(f"[SCALP-SHORT-BTC-NOTE] {btc_e}")

        # Safeguard 2B: Ban Shorting 24h Bull Breakouts (> +3.0%) & Sort by Weakest 24h Return
        # Never step in front of top-gaining bull breakouts (e.g. NEAR/APT); always prioritize relative laggards.
        try:
            clean_syms = [self.uta_client._normalize_symbol(s) for s in eligible_candidates]
            tickers = await self.uta_client.exchange.fetch_tickers(clean_syms)
            if isinstance(tickers, dict):
                filtered_by_24h = []
                for s in eligible_candidates:
                    t_info = tickers.get(self.uta_client._normalize_symbol(s)) or {}
                    pct_24h = float(t_info.get("percentage", 0.0) or 0.0) if isinstance(t_info, dict) else 0.0
                    if mode == "CHOP" and pct_24h > self.scalp_short_max_24h_gain:
                        log.info(f"[SCALP-SHORT-BREAKOUT-VETO] 🛑 Skipping {s}: 24h gain +{pct_24h:.2f}% > +{self.scalp_short_max_24h_gain:.1f}% breakout ceiling. Never short top gainers.")
                        continue
                    filtered_by_24h.append(s)
                eligible_candidates = filtered_by_24h
                eligible_candidates.sort(
                    key=lambda s: float((tickers.get(self.uta_client._normalize_symbol(s)) or {}).get("percentage", 0.0) or 0.0),
                    reverse=False
                )
        except Exception as sort_err:
            log.debug(f"[SCALP-SHORT-SORT-NOTE] {sort_err}")

        if not eligible_candidates:
            return

        for symbol in eligible_candidates:
            try:
                clean_sym = self.uta_client._normalize_symbol(symbol)
                candles = await self.uta_client.exchange.fetch_ohlcv(clean_sym, timeframe="5m", limit=60)
                if not candles or len(candles) < 30:
                    continue

                df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                ind = compute_grid_5m_indicators(df)
                if not ind:
                    continue

                curr_price = ind["curr_price"]
                rsi = ind["rsi"]
                upper_band = ind["upper_band"]
                upper_wick_ratio = ind["upper_wick_ratio"]
                adx_5m = ind["adx_5m"]

                # STRICT ANTI-BOTTOM SHORTING GATE:
                if rsi < self.scalp_short_min_rsi:
                    log.debug(f"[SCALP-SHORT-VETO] {symbol} RSI={rsi:.1f} < {self.scalp_short_min_rsi:.0f} (Oversold flush). Never short the bottom.")
                    continue

                # Safeguard 3A: Asset-Specific 5m ADX Exhaustion Ceiling (< 22.0)
                if mode == "CHOP" and adx_5m >= self.scalp_short_max_adx:
                    log.info(f"[SCALP-SHORT-ADX-VETO] 🛑 {symbol} 5m ADX={adx_5m:.1f} >= {self.scalp_short_max_adx:.1f} (Active Breakout Expansion). Never fade high-ADX momentum.")
                    continue

                is_sig, sig_reason = evaluate_scalp_short_signal(
                    ind=ind,
                    curr_price=curr_price,
                    mode=mode,
                    min_rsi=self.scalp_short_min_rsi,
                    rsi_overbought=self.scalp_short_rsi_overbought,
                    max_adx=self.scalp_short_max_adx
                )

                if not is_sig:
                    if mode == "CHOP" and curr_price >= (upper_band * 0.998) and rsi >= self.scalp_short_rsi_overbought and not ind.get("has_bearish_reversal", False):
                        log.debug(f"[SCALP-SHORT-CHOP-VETO] {symbol} At upper band (${curr_price:,.4f}) but lacks rejection wick/candle (wick={upper_wick_ratio:.1%}). Waiting for rejection.")
                    continue

                trigger_reason = sig_reason

                # Safeguard 3: 15m Higher-Timeframe Exhaustion Confirmation
                # Reject fading if the 15m candle is actively advancing green without an upper rejection wick (>= 20%)
                if self.scalp_short_require_15m:
                    try:
                        candles_15m = await self.uta_client.exchange.fetch_ohlcv(clean_sym, timeframe="15m", limit=10)
                        if candles_15m and len(candles_15m) >= 2:
                            last_15m = candles_15m[-1]
                            o15, h15, l15, c15 = float(last_15m[1]), float(last_15m[2]), float(last_15m[3]), float(last_15m[4])
                            rng15 = max(h15 - l15, 1e-6)
                            upper_wick_15 = h15 - max(o15, c15)
                            wick_ratio_15 = upper_wick_15 / rng15
                            
                            # If 15m candle is advancing green and upper wick is less than 20%, buyers are powering through
                            if c15 > o15 and wick_ratio_15 < 0.20:
                                log.info(f"[SCALP-SHORT-15M-VETO] 🛑 {symbol} 15m candle advancing green with only {wick_ratio_15:.1%} upper wick (< 20%). Skipping short fade against strong HTF momentum.")
                                continue
                    except Exception as htf_err:
                        log.debug(f"[SCALP-SHORT-15M-NOTE] {symbol}: {htf_err}")

                # Quantitative Divergence check (veto if bullish divergence is developing)
                div_info = detect_momentum_divergence(df, lookback=24)
                if div_info.get("bullish_divergence", False):
                    log.info(f"[SCALP-SHORT-DIVERGENCE-VETO] [VETO] {symbol} Bullish Divergence active on bounce ({div_info.get('details')}). Skipping short.")
                    continue

                # Microstructure Guard & VPIN
                ms_is_degraded = False
                try:
                    clean_sym = symbol.replace("/", "").replace("-", "").upper()
                    ms_state = MicrostructureStateCache.get_state(clean_sym)
                    ms_is_degraded = (not ms_state)
                    vpin_score = float(ms_state.get("vpin", 0.0) or 0.0)
                    if vpin_score >= 0.75:
                        log.info(f"[SCALP-SHORT-VPIN-VETO] [VETO] {symbol} VPIN={vpin_score:.2f} >= 0.75. Toxic flow squeeze risk. Skipping.")
                        continue
                except Exception:
                    ms_is_degraded = True

                log.info(f"[SCALP-SHORT-OPPORTUNITY] 🎯 {symbol} Quick Scalp Short Signal! {trigger_reason} | RSI: {rsi:.1f} | Wick: {upper_wick_ratio:.1%}")

                # Execute Maker Short Entry (Post-Only Limit Sell)
                conv_score = 0.78
                await self._execute_grid_maker_entry(
                    symbol=symbol,
                    current_price=curr_price,
                    rsi=rsi,
                    conv_score=conv_score,
                    rationale=trigger_reason,
                    side="short",
                    ms_is_degraded=ms_is_degraded
                )
                break  # One entry per cycle
            except Exception as e:
                log.debug(f"[SCALP-SHORT-SCAN-NOTE] {symbol}: {e}")

    # ---------------------------------------------------------------------------
    # Main Super Bot Orchestration Loops
    # ---------------------------------------------------------------------------
    async def run(self):
        """Main lifecycle of the Unified Super Bot."""
        log.info("=" * 65)
        log.info("      SUPER BOT V36.6 CORE SWING + MAKER GRID ENGAGED")
        log.info("=" * 65)
        log.info(f"Fleet Universe: {len(self.symbols)} assets | Max Concurrent Locks: {self.max_locks}")
        log.info(f"4H Swing Engine: 4H Ichimoku Breakout + Multi-Agent AI Boardroom (3x/5x Longs)")
        log.info(f"Chop Maker Grid: Clean 5m Oversold Dip-Buy Maker Engine (3x Longs)")
        log.info(f"Bear Mode: BEAR_SHIELD (100% Cash Preservation)")
        log.info("=" * 65)

        await self.initialize_clients()
        await asyncio.to_thread(self._load_all_positions_sync)

        log.info(f"[STARTUP-AUDIT] Adopted {len(self.active_positions)} active position(s): {list(self.active_positions.keys())}")
        for sym, pos in self.active_positions.items():
            log.info(f"   -> {sym}: {pos.get('side', 'long').upper()} | Size: {pos.get('size')} | Entry: ${pos.get('entry_price')} | SL: ${pos.get('stop_loss')} | TP: ${pos.get('take_profit')}")

        last_4h_scan = 0
        last_telemetry = 0

        while not self._stop_event.is_set():
            try:
                now_ts = time.time()
                
                # 1. Update Market Regime
                regime, adx, hurst = await asyncio.to_thread(self._get_market_regime_sync)
                
                # 2. Manage ALL Active Positions (LINK, SOL, NEAR, etc.)
                # Runs Kijun trailing, Breakeven ratchet, and Profit floors
                await self.swing_engine.manage_active_positions()
                await asyncio.to_thread(self._load_all_positions_sync)
                
                is_bull = any(x in regime for x in ("BULL", "TRENDING")) and self.market_regime_di_spread >= 0.0
                is_bear = any(x in regime for x in ("BEAR", "CRASH", "BREAKDOWN")) or (adx >= 26.0 and self.market_regime_di_spread < -5.0)
                is_chop = (not is_bear) and any(x in regime for x in ("CHOP", "SIDEWAYS", "UNKNOWN", "RANGING", "NEUTRAL", "CONSOLIDATION"))

                # 3. Route Scans Based on Live Regime
                if not self.regime_data_ok:
                    # Fail-closed: no NEW entries while regime telemetry is absent, stale,
                    # UNKNOWN or non-finite. Position management already ran above this loop,
                    # so trailing stops, breakeven ratchets and exits are unaffected.
                    log.debug(f"[REGIME-GATE] Entry scans skipped this cycle: {self.regime_stale_reason}")
                elif is_bear:
                    # Bear Mode: BEAR_SHIELD Active (Zero Longs, 100% Cash Preservation)
                    if now_ts - last_4h_scan >= self.check_interval_4h:
                        log.info(f"[REGIME: BEAR_SHIELD] 🛡️ (ADX: {adx:.1f}, DI Spread: {self.market_regime_di_spread:+.1f}) -> Bear Shield Active (Zero Longs).")
                        last_4h_scan = now_ts
                    if self.scalp_short_enabled:
                        await self.scan_quick_scalp_short_opportunities(mode="BEAR")

                else:
                    # 3a. 4H Swing Architect + AI Boardroom Scan (every 5m when not in Bear Shield and DI spread >= -2.0)
                    # Unchains decoupled momentum leaders (NEAR, LINK, AVAX, SUI, APT) even when BTC ADX is in CHOP
                    if now_ts - last_4h_scan >= self.check_interval_4h:
                        if self.turbulence_state == "EXTREME_SHOCK":
                            log.warning(
                                f"[FINRL-TURBULENCE-SHIELD] 🚨 Cross-asset turbulence breach active "
                                f"(d_t={self.turbulence_index:.2f} > {self.turbulence_threshold:.2f}). 4H Swing LONG scan muted."
                            )
                        elif is_bull or self.market_regime_di_spread >= -2.0:
                            log.info(
                                f"[REGIME: {regime}] (ADX: {adx:.1f}, DI Spread: {self.market_regime_di_spread:+.1f}, "
                                f"Hurst: {hurst:.2f}, Turbulence: {self.turbulence_index:.1f}/{self.turbulence_threshold:.1f} [{self.turbulence_state}]) "
                                f"-> Running 4H Swing Architect + AI Boardroom fleet scan (LONG)..."
                            )
                            await self.swing_engine.scan_fleet(allowed_direction="LONG")
                        last_4h_scan = now_ts

                    # 3b. Clean Sept 30 Chop Maker Grid Dip-Buy Scan (in CHOP regime when ADX < 25.0)
                    if is_chop:
                        if adx >= 25.0:
                            log.info(f"[CHOP-GRID-VETO] Market ADX={adx:.1f} >= 25.0 indicates directional expansion. Chop Grid dip-buying suppressed.")
                        else:
                            await self.scan_chop_grid_opportunities()
                        if self.scalp_short_enabled:
                            await self.scan_quick_scalp_short_opportunities(mode="CHOP")

                # 4. Telemetry Logging (every 60s)
                if now_ts - last_telemetry >= 60:
                    last_telemetry = now_ts
                    now_iso = datetime.now(UTC).isoformat()
                    # S2-11 FIX: Compute truthful aggregate unrealized PnL from active positions
                    total_unrealized_pnl = 0.0
                    for p in self.active_positions.values():
                        try:
                            e_price = float(p.get("entry_price", 0.0) or 0.0)
                            c_price = float(p.get("current_price", e_price) or e_price)
                            p_size = float(p.get("size", 0.0) or 0.0)
                            p_side = str(p.get("side", "long")).lower()
                            if e_price > 0 and p_size > 0:
                                pos_upnl = (c_price - e_price) * p_size if "long" in p_side else (e_price - c_price) * p_size
                                total_unrealized_pnl += pos_upnl
                        except Exception:
                            pass
                    def _write_telemetry():
                        with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                            conn.execute("""
                                INSERT OR REPLACE INTO super_bot_telemetry
                                (timestamp, regime, adx, hurst, active_swing_count, active_grid_count, total_unrealized_pnl, status)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                now_iso, regime, adx, hurst,
                                len([p for p in self.active_positions.values() if p.get("engine_type") == "SWING_4H"]),
                                len([p for p in self.active_positions.values() if p.get("engine_type") in ("GRID_MAKER", "SCALP_SHORT")]),
                                round(total_unrealized_pnl, 4), "OPERATIONAL"
                            ))
                            conn.commit()
                    await asyncio.to_thread(_write_telemetry)

            except Exception as e:
                log.error(f"[SUPER-BOT-LOOP-ERR] {e}")

            # Responsive sleep loop
            for _ in range(self.check_interval_fast):
                if self._stop_event.is_set():
                    break
                await asyncio.sleep(1)

        log.info("[SUPER-BOT] Graceful shutdown completed.")

    def stop(self):
        """Triggers graceful engine shutdown."""
        log.info("[SUPER-BOT] Stopping Super Bot...")
        self._stop_event.set()
        self.swing_engine.stop()


def main():
    bot = SuperBotTrader()

    def handle_signal(sig, frame):
        log.info(f"[SHUTDOWN] Signal {sig} received.")
        bot.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        asyncio.run(bot.run())
    except (KeyboardInterrupt, SystemExit):
        log.info("[SHUTDOWN] Process terminated.")
    except Exception as e:
        log.critical(f"[FATAL] Super Bot crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

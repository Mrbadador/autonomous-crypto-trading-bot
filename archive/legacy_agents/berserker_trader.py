import asyncio
import socket
import aiohttp
import json
import logging
import pandas as pd
import dns_fix  # noqa: F401 — patches ccxt.Exchange.open() with ThreadedResolver
import ccxt.async_support as ccxt
import os
import traceback
import random
from datetime import datetime, UTC, timedelta
import config
from strategist_agent import StrategistAgent
from analyst_agent import AnalystAgent
from executor_agent import ExecutorAgent
from indicators import compute_indicators, classify_regime_hard, get_candle_patterns, check_ema_stack
from email_dispatcher import send_trade_log_async, send_alert_async
from live_paper_exchange import LivePaperExchange
import ai_engine
from core_engine import get_deterministic_action, calculate_risk_qty, calculate_exits, build_trend_context, calculate_dynamic_risk
from state_manager import EngineState, TradeEntry
from analyst_agent import AnalystAgent
from strategist_agent import StrategistAgent
from db_manager import DatabaseManager
from warden_60m import Warden60Minute
from sentinel_agent import SentimentSentinel
from risk_manager import RiskManager
from latency_tracer import LatencyTracer, reset_trace, log_full_trace
from analyst_agent import detect_volume_divergence
from conviction_filter import ConvictionBoardroom
from macro_breaker_agent import MacroCircuitBreaker
import sys
import numpy as np
from microstructure_guard import MicrostructureGuard


# V21.0 Module-level Sentinel (shared across all symbols — one RSS poll per trade attempt)
news_sentinel = SentimentSentinel()

log = logging.getLogger("berserker")

# ── [V32.6-BLACKLIST] Hard asset exclusion list ───────────────────────────────
# Add any ticker that demonstrates consistent losses in a trending regime.
# Checked BEFORE any indicator compute, AI call, or exchange interaction.
# To re-enable an asset: remove it from this list and restart the fleet node.
BLACKLIST = [
    'GUA/USDT',   # [2026-05-29] 0W/3L in TRENDING regime. -10.81 USDT in 24h. Quarantined.
    'XLM/USDT',   # [2026-05-30] 1W/3L (33% WR), -$1.38 PnL, vol_pct=34% (dead market). Quarantined.
]

# ── [V32.7] Minimum Volume Activity Gate ──────────────────────────────────────
# vol_pct is the 5m candle volume vs. the 20-period EMA of volume.
# Assets below this threshold are trading on thin, stale liquidity.
# Signals generated in sub-75% vol environments have historically been noise.
# Set to 0.0 to disable. Adjust via .env: BERSERKER_MIN_VOL_PCT
MIN_VOL_PCT_GATE: float = float(os.getenv("BERSERKER_MIN_VOL_PCT", "75.0"))

# Setup Logging
def setup_logging(log_name="berserker"):
    l = logging.getLogger(log_name)
    if not l.handlers:
        l.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        l.addHandler(sh)
        
        os.makedirs("logs", exist_ok=True)
        fh = logging.FileHandler(f"logs/{log_name}.log")
        fh.setFormatter(formatter)
        l.addHandler(fh)
    l.propagate = False # Prevent duplicate bubbling
    return l

class BerserkerTrader:
    def __init__(self, symbol: str, shared_warden=None, shared_sentinel=None, fleet_state=None, shared_analyst=None, shared_strategist=None, shared_risk_manager=None, scan_offset_secs: float = 0.0):
        self.fleet_state = fleet_state
        self.scan_offset_secs = scan_offset_secs  # V38.0 AI-call stagger: per-node offset to prevent simultaneous Groq bursts
        safe_name = symbol.encode('ascii', 'ignore').decode('ascii').replace('/', '-')
        self.log_name = f"berserker-{safe_name}"
        self.log = setup_logging(self.log_name)
        # V26.6 Unicode Display Sanitization
        display_symbol = symbol.encode('ascii', 'ignore').decode('ascii') if not symbol.isascii() else symbol
        self.log.info(f"[BOOT] Fleet Node for {display_symbol} initializing.")
        
        self.symbol = symbol
        self.symbols = [symbol]
        self.timeframe = "5m"
        self.exchange = None
        self.paper_ex = LivePaperExchange(initial_balance=config.INITIAL_BALANCE, account_name="berserker")
        self.db = DatabaseManager(db_path="trading_data.db") # V27.0 Dedicated DB Utility Instance
        self.dfs = {s: pd.DataFrame() for s in self.symbols}
        self.active_positions = {s: None for s in self.symbols}
        self.last_btc_trend = "UNKNOWN"
        self.last_candle_t = {s: 0 for s in self.symbols}
        self.order_lock = asyncio.Lock()
        self.veto_pause_until = {}
        self.tp_cooldown = {}
        self.last_macro_fetch = None
        self.macro_fetch_ttl = timedelta(minutes=10)
        self.state = EngineState()
        self.background_tasks = set()
        self.active_strategy_mode = "Trend" # V26.3 Strategy Swapper Default

        self._last_ticker: dict[str, dict] = {}
        self._last_balance: float = 0.0
        self._balance_fetched_at: float = 0.0
        self._balance_ttl: float = 10.0
        
        # V30.1 Dynamic Sovereignty Engines (Shared across nodes to prevent model bloat and event-loop lag)
        self.rl_analyst = shared_analyst if shared_analyst else AnalystAgent()
        self.rl_strategist = shared_strategist if shared_strategist else StrategistAgent()
        self.executor = ExecutorAgent(min_confidence=config.MIN_NN_CONFIDENCE)
        # policy_model.pth 14 times (7 nodes x 2) was pure memory waste.
        
        self.owns_warden = (shared_warden is None)
        self.macro_warden = shared_warden if shared_warden else Warden60Minute(db_path="trading_ledger.db")
        self.news_sentinel = shared_sentinel if shared_sentinel else news_sentinel
        
        # V22.0 Dynamic Risk Manager (Fractional Kelly Criterion)
        self.owns_risk_manager = (shared_risk_manager is None)
        self.risk_manager = shared_risk_manager if shared_risk_manager else RiskManager(
            historical_win_rate=0.65,
            avg_risk_reward=1.5,
            max_risk_cap=0.05  # Hard cap: never risk more than 5% per trade
        )
        
        # V28.7 Pre-Trade Order Book Microstructure Guard (Hawkes/VPIN veto layer)
        self.microstructure_guard = MicrostructureGuard()
        self.boardroom = ConvictionBoardroom()
        self.macro_breaker = MacroCircuitBreaker()
        # V31.0 P1: Per-node WebSocket stale-connection alert deduplication flag.
        # Prevents email floods during network outages (7 nodes × reconnect storm).
        self._ws_stale_alerted: bool = False

    def _run_background_task(self, coro):
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    async def _set_status(self, msg, veto=None):
        """Broadcasts internal state to the system_state table for the HUD."""
        await self.paper_ex.update_system_state(f"current_status_{self.log_name}", msg)
        if veto is not None:
            await self.paper_ex.update_system_state(f"last_veto_{self.log_name}", veto)

    def _get_dynamic_parameters(self, regime_or_trend=None):
        profile = config.get_regime_profile(regime_or_trend or self.last_btc_trend)
        params = {
            "min_nn_confidence": config.MIN_NN_CONFIDENCE,
            "vol_gate_mult": config.BERSERKER_VOL_GATE_MULT,
            "veto_threshold": profile.get("min_conviction", config.BERSERKER_VETO_THRESHOLD),
            "tp_atr_mult": profile.get("tp_atr_mult", 4.5),
            "sl_atr_mult": min(1.2, float(profile.get("sl_atr_mult", 1.2))),
            "trail_activation_atr": float(profile.get("trail_activation_atr", 2.0)),
            "leverage": profile.get("leverage", config.LEVERAGE),
            "size_multiplier": 0.5 * float(profile.get("size_multiplier", 1.0))
        }
        try:
            import sqlite3
            with sqlite3.connect("trading_data.db", timeout=10) as conn:
                c = conn.cursor()
                c.execute("SELECT param_name, param_value FROM bot_parameters WHERE bot_name = 'berserker'")
                for row in c.fetchall():
                    if row[0] in params:
                        params[row[0]] = float(row[1])
        except Exception as e:
            self.log.error(f"Failed to fetch dynamic params: {e}")
        return params

    async def prime_data(self):
        for attempt in range(1, 6):
            try:
                await self.exchange.load_markets()
                self.log.info(f"Berserker Exchange markets loaded (Attempt {attempt})")
                break
            except Exception as e:
                if attempt == 5:
                    self.log.critical(f"FATAL: Failed to load markets after 5 attempts: {e}")
                    raise
                self.log.warning(f"Market load failed: {e}. Retrying in 10s... ({attempt}/5)")
                await asyncio.sleep(10)
                
        for symbol in self.symbols:
            display_symbol = symbol.encode('ascii', 'ignore').decode('ascii') if not symbol.isascii() else symbol
            self.log.info(f"Priming {display_symbol} with 5m historical data...")
            try:
                raw = await self.exchange.fetch_ohlcv(symbol, timeframe='5m', limit=200)
                df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df.set_index("timestamp", inplace=True)
                self.dfs[symbol] = compute_indicators(df)
                self.log.info(f"Berserker Primed {display_symbol}. Price: ${self.dfs[symbol]['close'].iloc[-1]:.2f}")
            except Exception as e:
                self.log.error(f"Silent failure caught in [prime_data]: {e}")
        
        # V15.4: Push initial metric heartbeat immediately on startup
        for symbol in self.symbols:
            if symbol in self.dfs and not self.dfs[symbol].empty:
                r1 = self.dfs[symbol].iloc[-1]
                price = float(r1["close"])
                atr = max(float(r1["atr14"]), 0.0001)
                vol, vol_sma = float(r1["volume"]), float(r1["vol_sma20"])
                metrics = {
                    "price": price,
                    "ema200": float(r1.get("ema200", price)),
                    "vol_ratio": vol / vol_sma if vol_sma > 0 else 1.0,
                    "atr": atr
                }
                await self.paper_ex.update_system_state(f"{self.log_name}_metrics", json.dumps(metrics))
                await self.paper_ex.save_log("INFO", f"[{self.log_name.upper()}] Initialized. Balance: ${self.paper_ex.balance:,.2f}")
            self.log.info(f"[{self.log_name}] SYNC COMPLETE: {len(self.paper_ex.positions)} positions recovered.")

        # V22.0: Start Sentinel background RSS cache (non-blocking)
        import time as _time
        self._time = _time   # make time available as self._time throughout the class
        self.news_sentinel.start_background_refresh([s.split("/")[0] for s in self.symbols])
        self.log.info("[SENTINEL] Background RSS refresh loop launched.")

    async def fetch_btc_macro(self):
        """
        V30.4 Centrally Cached Macro Trend:
        Instead of query-hammering the exchange for BTC klines on every candle close,
        the fleet pulls the central macro trend calculated by the Live Oracle
        from the SQLite trading_data.db (system_state table).

        V30.6 WAL-Retry Guard: If the Oracle is mid-checkpoint (PRAGMA wal_checkpoint(TRUNCATE))
        when this read fires, SQLite raises OperationalError "database is locked". We retry
        once after 100ms before falling back to the last known cached trend.

        V31.0 P2: Added TTL guard using self.last_macro_fetch and self.macro_fetch_ttl.
        These attributes were reserved at init (L82-83) but never implemented, causing
        7 fleet nodes to each hit trading_data.db on every 5m candle = 84 reads/hour
        for a value that only changes once per hour.
        """
        import asyncio
        import sqlite3 as _sqlite3

        # V31.0 TTL Gate: skip the DB read entirely if last fetch was < 10 minutes ago
        now_dt = datetime.now()
        if self.last_macro_fetch is not None and (now_dt - self.last_macro_fetch) < self.macro_fetch_ttl:
            self.log.debug(f"[MACRO] TTL hit — using cached BTC trend: {self.last_btc_trend}")
            return

        async def _load_with_wal_retry(key: str, default: str) -> str:
            try:
                return await asyncio.to_thread(self.db.load_state, key, default)
            except _sqlite3.OperationalError as wal_err:
                self.log.warning(f"[MACRO] WAL lock on DB read ({wal_err}). Retrying in 100ms...")
                await asyncio.sleep(0.1)
                try:
                    return await asyncio.to_thread(self.db.load_state, key, default)
                except _sqlite3.OperationalError:
                    self.log.error(f"[MACRO] Retry also locked. Holding last known trend: {self.last_btc_trend}")
                    return default

        try:
            cached_trend = await _load_with_wal_retry("btc_macro_trend", "UNKNOWN")
            if cached_trend != "UNKNOWN":
                self.last_btc_trend = cached_trend
                self.log.info(f"[MACRO] Pulling cached BTC trend: {self.last_btc_trend}")
            else:
                self.log.warning("[MACRO] Centralized BTC trend is UNKNOWN or not initialized in DB yet.")
            # Mark successful fetch regardless of value to enforce TTL
            self.last_macro_fetch = now_dt
        except Exception as e:
            self.log.error(f"Failed to read central BTC trend from DB: {e}")

    def _sync_compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to run synchronous and CPU-bound pandas indicator calculations off-thread."""
        return compute_indicators(df)

    async def handle_candle_close(self, symbol, kline_data):
        try:
            # V25.6: Re-sync state from DB so Recovery Agent DCAs don't get orphaned
            await self.paper_ex.sync_positions_from_db()
            self.active_positions[symbol] = self.paper_ex.positions.get(symbol)

            # ── [V32.6-BLACKLIST] Pre-execution quarantine gate ───────────────
            # FIRST check — fires before any indicator compute, AI call, or
            # exchange interaction. Zero CPU wasted on excluded assets.
            if symbol in BLACKLIST:
                self.log.warning(
                    f"[BLACKLIST] Signal for {symbol} received but QUARANTINED. "
                    f"Asset is on the hard exclusion list. Skipping all processing."
                )
                return

            # ── [V38.0] Per-Node AI-Call Stagger ─────────────────────────────
            # Each fleet node has a deterministic offset (set at launch, 0–45s).
            # This spreads 10 concurrent Groq API calls evenly across time so
            # no two nodes fire simultaneously, eliminating 429 burst exhaustion
            # on the 2-key pool without requiring additional API keys.
            # NOTE: The position-management path (has_pos) bypasses the offset
            # so that exit / SL / TP logic is never delayed.
            if self.scan_offset_secs > 0.0 and self.active_positions.get(symbol) is None:
                self.log.debug(f"[STAGGER] Node offset {self.scan_offset_secs:.1f}s applied for {symbol}")
                await asyncio.sleep(self.scan_offset_secs)

            # ── [V32.7] Minimum Volume Activity Pre-Gate ──────────────────────
            # Reject signals from illiquid/thin markets before any heavy compute.
            # vol_pct is already stored per-symbol in berserker_telemetry; read
            # it from the kline_data vol field if available to stay zero-latency.
            if MIN_VOL_PCT_GATE > 0.0:
                try:
                    raw_vol  = float(kline_data.get('v', 0) or 0)    # current candle volume
                    # Use the EMA-vol ref stored on self if available (set by telemetry writer)
                    ema_vol  = getattr(self, f"_ema_vol_{symbol.replace('/','_')}", None)
                    if ema_vol and ema_vol > 0:
                        vol_pct = (raw_vol / ema_vol) * 100.0
                        if vol_pct < MIN_VOL_PCT_GATE:
                            self.log.info(
                                f"[VOL-GATE] {symbol} vol_pct={vol_pct:.1f}% < "
                                f"{MIN_VOL_PCT_GATE:.0f}% gate. Skipping — thin market."
                            )
                            return
                except Exception as _vg_err:
                    self.log.debug(f"[VOL-GATE] vol check error for {symbol}: {_vg_err}")
            # ─────────────────────────────────────────────────────────────────

            kline_timestamp = int(kline_data['t'])
            if self.last_candle_t.get(symbol, 0) >= kline_timestamp:
                self.log.info(f"[GUARD] Rejecting duplicate candle close tick for {symbol} at {kline_timestamp}. Already processed.")
                return
            self.last_candle_t[symbol] = kline_timestamp

            sym_veto = self.veto_pause_until.get(symbol)
            if sym_veto and datetime.now() < sym_veto:
                pause_msg = f"Cooling from Veto until {sym_veto.strftime('%H:%M:%S')}"
                self.log.info(f"[PAUSED] Berserker {symbol} {pause_msg}")
                await self._set_status(f"Status: {symbol} {pause_msg}")
                return

            # V25.1 Block 1: Macro Gate
            try:
                def _get_killswitch():
                    import sqlite3
                    conn = sqlite3.connect("trading_ledger.db", timeout=5.0)
                    try:
                        return conn.execute("SELECT global_killswitch FROM macro_market_context ORDER BY id DESC LIMIT 1").fetchone()
                    finally:
                        conn.close()
                macro_row = await asyncio.to_thread(_get_killswitch)
                if macro_row and macro_row[0] == 1:
                    self.log.warning(f"[MACRO-GATE] SPY dump detected. Execution halted for {symbol}.")
                    await self._set_status("Status: Halted by Macro Gate")
                    return
            except Exception as e:
                self.log.error(f"Failed to check Macro Gate: {e}")

            reset_trace()  # V22.0: Start fresh latency trace for this candle
            self.log.info(f"[BERSERKER-PULSE] 5m Candle Closed for {symbol}.")
            price = float(kline_data['c'])
            if price <= 0: return

            # V25.3 Block 3: OFI Micro-Audit Math
            total_volume = float(kline_data.get('v', 0.0))
            taker_buys = float(kline_data.get('V', 0.0))
            taker_sells = total_volume - taker_buys
            ofi = (taker_buys - taker_sells) / total_volume if total_volume > 0 else 0.0

            new_row = {"open": float(kline_data['o']), "high": float(kline_data['h']), "low": float(kline_data['l']), "close": price, "volume": float(kline_data['v'])}
            new_timestamp = pd.to_datetime(kline_data['t'], unit='ms', utc=True)
            temp_df = self.dfs[symbol].copy()
            temp_df.loc[new_timestamp] = new_row
            if len(temp_df) > 1000: temp_df = temp_df.tail(1000)
            self.dfs[symbol] = await asyncio.to_thread(self._sync_compute_indicators, temp_df)
            
            # V14.0 P0: Fix Look-ahead bias by ensuring we use ONLY the closed candle history.
            # In handle_candle_close, the last row is the one that JUST closed.
            # We strictly avoid the live candle by utilizing the stable historical dataframe.
            closed_df = self.dfs[symbol].iloc[:-1] 
            r1 = self.dfs[symbol].iloc[-1] 
            
            atr = max(float(r1["atr14"]), 0.0001)
            vol, vol_sma = float(r1["volume"]), float(r1["vol_sma20"])
            
            rsi = float(r1.get("rsi14", 50))
            # Metrics update deferred until after VARSI calculation
            # [CP35-S1-C1] Anchored to background_tasks to prevent orphan GC
            self._run_background_task(self.paper_ex.save_log("DEBUG", f"[{symbol}] Decision Matrix Running. Price: {price}"))
            ema50, ema20, ema200 = float(r1["ema50"]), float(r1["ema20"]), float(r1["ema200"])
            adx_val, adx_sma = float(r1.get("adx", 0)), float(r1.get("adx_avg", 0))
            vwap = float(r1.get("vwap", price))
            hard_regime = classify_regime_hard(r1)
            trend_ctx = build_trend_context(r1)
            await self.fetch_btc_macro()

            # Fetch dynamic params using macro trend and regime profile (offloaded to thread)
            dyn_params = await asyncio.to_thread(self._get_dynamic_parameters, self.last_btc_trend or hard_regime)
            dyn_min_confidence = dyn_params["min_nn_confidence"]

            dyn_vol_gate_mult = dyn_params["vol_gate_mult"]
            dyn_veto_threshold = dyn_params["veto_threshold"]

            # V24.0 Adaptive Constraints: VARSI (Volatility-Adjusted RSI)
            try:
                # V30.1: Await asynchronous state construction to prevent loop starvation
                state_vec = await self.rl_analyst.build_state_tensor()
                # NOTE: Fast-gate neural confidence check DISABLED — the model weights have not been
                # sufficiently trained on live data and consistently produce ~36% confidence regardless
                # of market conditions, causing the gate to block 100% of candles. The deterministic
                # entry logic (volume displacement, RSI band, EMA50, 60M macro warden) provides all
                # necessary trade quality filtering. Re-enable once PPO training converges above 65%.
                neural_decision = self.rl_strategist.get_action(state_vec)

                def _q_varsi():
                    import sqlite3
                    conn = sqlite3.connect("trading_ledger.db", timeout=30.0)  # V24.5 P1: corrected to system standard
                    res = conn.execute("SELECT volatility_state, trend_strength FROM market_context ORDER BY timestamp DESC LIMIT 1").fetchone()
                    conn.close()
                    return res
                row = await asyncio.to_thread(_q_varsi)
                
                if row:
                    volatility, trend_strength = row
                    dynamic_rsi_target = 30.0 + (trend_strength * 10.0) - ((volatility - 1.0) * 10.0)
                    dynamic_rsi_target = max(22.0, min(38.0, dynamic_rsi_target))
                else:
                    volatility, trend_strength = 1.0, 0.0
                    dynamic_rsi_target = 30.0
                    
                dyn_rsi = dynamic_rsi_target
                dyn_vwap = 0.98 # Retained for vwap constraint
                
                self.log.info(f"[SOVEREIGNTY-BERSERKER] Adaptive Constraints: Vol {volatility:.4f} | Trend {trend_strength:+.4f} -> New RSI Target: {dynamic_rsi_target:.1f}")
            except Exception as re:
                self.log.warning(f"VARSI query failed: {re}. Reverting to benchmark constants.")
                dyn_rsi, dyn_vwap = 30.0, 0.98

            # V26.3 Sovereign Master Router: Load Current Market Regime
            regime_label, hurst_val = "UNKNOWN", 0.5
            try:
                import sqlite3 as _sqlite3
                def _q_regime():
                    with _sqlite3.connect("trading_data.db", timeout=5) as c:
                        return c.execute("SELECT regime, hurst FROM market_regime ORDER BY rowid DESC LIMIT 1").fetchone()
                reg_row = await asyncio.to_thread(_q_regime)
                if reg_row:
                    regime_label, hurst_val = str(reg_row[0]).upper(), float(reg_row[1])
            except Exception as reg_e:
                self.log.warning(f"Failed to fetch market regime: {reg_e}. Defaulting to Trend stance.")
            
            # V28.0 Stage 3: Fetch 60M Macro Warden Regime Upfront for Directional Bias
            current_macro_regime = "UNKNOWN"
            try:
                current_macro_regime = await self.macro_warden.determine_regime()
            except Exception as we:
                self.log.warning(f"Pre-fetch 60m Warden failed: {we}")

            # 1.2 Hurst Exponent Filtering & Master Routing Logic Switch
            hurst_floor = getattr(config, "HURST_TREND_FLOOR", 0.52)
            if regime_label == 'CHOP' or hurst_val < hurst_floor:
                self.active_strategy_mode = "MeanRev"
                self.log.info(f"[HURST-ROUTER] Low Hurst ({hurst_val:.3f} < {hurst_floor}) or CHOP regime. Selecting MeanRev stance.")
            elif regime_label == 'TRENDING' and hurst_val >= hurst_floor:
                self.active_strategy_mode = "Trend"
            else:
                self.active_strategy_mode = "Trend" if hurst_val >= 0.50 else "MeanRev"
            
            self.log.info(f"[MASTER-ROUTER] Regime: {regime_label} (Hurst: {hurst_val:.2f}). Active Stance: {self.active_strategy_mode}")

            # V15.2: Update System State — deferred (P4: non-blocking create_task)
            metrics = {
                "price": price,
                "ema200": float(r1.get("ema200", price)),
                "vol_ratio": vol / vol_sma if vol_sma > 0 else 1.0,
                "atr": atr,
                "rsi": rsi,
                "dyn_rsi": dyn_rsi,
                "strategy_mode": self.active_strategy_mode
            }
            # [CP35-S1-C1] Anchored to background_tasks to prevent orphan GC
            self._run_background_task(self.paper_ex.update_system_state(f"{self.log_name}_metrics", json.dumps(metrics)))

            # --- Berserker DETERMINISTIC ENTRY GATE (V11.0: vol gate from config) ---
            vol_ratio = vol / vol_sma if vol_sma > 0 else 0
            is_volume_spike = vol_ratio >= dyn_vol_gate_mult

            is_trending = adx_val > 25.0
            is_not_oversold = rsi > 25.0
            
            # V17.5 Dynamic Sovereignty Alignment: Flexible trend momentum band (not overbought, not collapsed)
            v17_long_valid = (rsi < 75.0) and (rsi > 35.0) and (price > vwap * dyn_vwap)

            # Determine intended entry signal
            intended_signal = None
            if self.active_positions.get(symbol) is None:
                if self.active_strategy_mode == "MeanRev":
                    # --- V26.3 Sovereign Ghost-Grid (Mean Reversion) ---
                    intended_signal = self._execute_mean_reversion_logic({
                        "price": price, 
                        "bb_lower": float(r1.get("bb_lower", price)),
                        "bb_upper": float(r1.get("bb_upper", price)),
                        "rsi": rsi,
                        "macro_regime": current_macro_regime
                    })
                    if intended_signal:
                        self.log.info(f"[GHOST-GRID] Mean Reversion trigger fired for {symbol}! Executing Snap-back.")
                else:
                    # --- Standard Berserker Trend Logic ---
                    if is_volume_spike:
                        if price > ema50 and self.last_btc_trend == "BULLISH" and is_trending and v17_long_valid:
                            intended_signal = "long"
                        elif price < ema50 and self.last_btc_trend == "BEARISH" and is_trending and is_not_oversold:
                            intended_signal = "short"

            # V25.3 Block 3: Gate 8 Micro-Flow Veto
            if intended_signal:
                if intended_signal == "long" and ofi < -0.2:
                    self.log.warning(f"[MICRO-AUDIT] VETO: Signal is LONG but Order Flow is Bearish (OFI: {ofi:.2f}).")
                    await self._set_status(f"Status: VETO by Micro-Flow (OFI: {ofi:.2f})")
                    intended_signal = None
                elif intended_signal == "short" and ofi > 0.2:
                    self.log.warning(f"[MICRO-AUDIT] VETO: Signal is SHORT but Order Flow is Bullish (OFI: {ofi:.2f}).")
                    await self._set_status(f"Status: VETO by Micro-Flow (OFI: {ofi:.2f})")
                    intended_signal = None

            # [V20.1] THE 60-MINUTE MASTER LOCK — Top-Down Multi-Timeframe Veto Gate
            if intended_signal:
                try:
                    # V28.0 Stage 3: Utilizing pre-fetched current_macro_regime
                    
                    # V28.0 Stage 2: Explicit Bypass for Mean-Reversion logic during Macro Chop
                    allow_execution = False
                    
                    if current_macro_regime == "CHOP_GRID_ACTIVE":
                        if self.active_strategy_mode == "MeanRev":
                            allow_execution = True
                            self.log.info(f"[60M-APPROVED] Mean-Reversion setup authorized during CHOP_GRID_ACTIVE regime.")
                        else:
                            # Explicit Block for Trend logic during chop
                            pass 
                    elif intended_signal == "long" and current_macro_regime == "LONG_ONLY":
                        allow_execution = True
                        self.log.info(f"[60M-APPROVED] {symbol} LONG setup aligns with 60m Tide. Proceeding.")
                    elif intended_signal == "short" and current_macro_regime == "SHORT_ONLY":
                        allow_execution = True
                        self.log.info(f"[60M-APPROVED] {symbol} SHORT setup aligns with 60m Tide. Proceeding.")

                    if not allow_execution:
                        self.log.warning(f"[60M-VETO] {symbol} {intended_signal.upper()} blocked — 60m Warden is in {current_macro_regime}. Target Stance: {self.active_strategy_mode}")
                        await self._set_status(f"Status: VETO by 60m Warden ({current_macro_regime})", veto=f"60m Warden: {current_macro_regime} blocked {intended_signal.upper()}")
                        intended_signal = None
                        
                except Exception as warden_err:
                    self.log.warning(f"[60M-WARDEN] Regime check failed: {warden_err}. Allowing signal through as safety fallback.")

            # V11.0: TP Re-entry Cooldown — block same-direction entry for 3 candles (15m)
            signal = intended_signal
            if signal:
                cd = self.tp_cooldown.get(symbol)
                if cd and datetime.now() < cd["until"] and signal == cd["direction"]:
                    cooldown_msg = f"TP cooldown active until {cd['until'].strftime('%H:%M:%S UTC')}"
                    self.log.info(f"[COOLDOWN] {symbol} {signal.upper()} blocked — {cooldown_msg}")
                    await self._set_status("Status: Blocked by Cooldown", veto=cooldown_msg)
                    signal = None

            action = f"OPEN_{signal.upper()}" if signal else "HOLD"
            if action == "HOLD":
                 await self._set_status("Status: Scanning for Volume Displacement")

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            tel_packet = {
                "timestamp": current_time,
                "reasoning": f"[{action}] Vol Ratio: {vol_ratio:.2f} | Macro: {self.last_btc_trend}",
                "price": price, "ema_ref": ema50, "vol_pct": vol_ratio * 100, "action": action,
                "quality": 0.0
            }
            # [CP35-S1-C1] Anchored to background_tasks to prevent orphan GC
            self._run_background_task(self.paper_ex.save_telemetry(symbol, tel_packet))

            # V22.0 P4: Defer dashboard gauge write — [CP35-S1-C1] Anchored to background_tasks
            self._run_background_task(self.paper_ex.update_system_state(f"{self.log_name}_metrics", json.dumps({
                "price": price, "ema200": ema200, "ema50": ema50, "atr": atr, "vol_ratio": vol_ratio, "rsi": rsi
            })))

            if signal and self.active_positions[symbol] is None:
                # -------------------------------------------------------------------
                # V23.1 FLEET OPTIMIZER: Alpha Strike & Circuit Breaker
                # -------------------------------------------------------------------
                if self.fleet_state:
                    try:
                        # Use timestamp string as canonical candle key
                        candle_key = str(r1.name)
                        
                        # A. Register Intent & Enter Async Buffer for rivals (200ms)
                        await self.fleet_state.register_opportunity(candle_key, symbol, vol_ratio)
                        self.log.info(f"[ALPHA-BUFFER] Staking claim for {symbol} (Vol: {vol_ratio:.2f}x). Waiting 200ms for rivals...")
                        await asyncio.sleep(0.2)
                        
                        # B. Alpha Arbitration
                        is_leader = await self.fleet_state.is_alpha_leader(candle_key, symbol)
                        if not is_leader:
                            self.log.warning(f"[ALPHA-FILTER] Abandoning strike for {symbol}. Rival coin is dominating the pulse.")
                            await self._set_status("Status: Blocked by Alpha Filter", veto="Yielding to Alpha Rival")
                            signal = None
                        
                        # C. Circuit Breaker enforced at final gate
                        if signal:
                            open_count = await self.fleet_state.get_active_positions_count()
                            max_c = self.fleet_state.MAX_CONCURRENT_TRADES
                            if open_count >= max_c:
                                self.log.warning(f"[CIRCUIT-BREAKER] Strike blocked for {symbol}. Portfolio heat maximized ({open_count}/{max_c}).")
                                await self._set_status("Status: Circuit Breaker Blocked", veto=f"Portfolio Full ({open_count})")
                                signal = None
                    except Exception as fe:
                        self.log.error(f"[OPTIMIZER-ERROR] Fleet check failed: {fe}. Bypassing for safety.")

            existing_pos = self.active_positions.get(symbol)
            is_scale_in = existing_pos is not None
            max_entries_per_coin = int(getattr(config, "MAX_POSITIONS_PER_SYMBOL", 2))
            
            # Check per-symbol entry limit (allows up to 2 positions per coin regardless of PnL)
            sym_entry_count = 1 if is_scale_in else 0
            if sym_entry_count >= max_entries_per_coin:
                self.log.info(f"[MAX-ENTRIES-REACHED] {symbol} already has {sym_entry_count}/{max_entries_per_coin} entries. Skipping.")
                signal = None

            if signal and (not is_scale_in or getattr(config, "ALLOW_SAME_COIN_SCALE_IN", True)):
                await self._set_status(f"Status: Pinging Groq AI for Audit on {symbol}...")
                async with self.order_lock:
                    self.log.info(f"[HYBRID-GATEWAY] Requesting PRE-TRADE AI Audit for {symbol} {signal.upper()} ({'ENTRY 2' if is_scale_in else 'ENTRY 1'})...")
                    is_ema_stack = check_ema_stack(price, ema20, ema50, ema200, signal)
                    patterns = get_candle_patterns(self.dfs[symbol])
                    current_session_pnl = self.state.get_session_performance()

                    hist_p = self.dfs[symbol]['close'].tail(5).tolist()
                    hist_v = self.dfs[symbol]['volume'].tail(5).tolist()
                    div_data = detect_volume_divergence(hist_p, hist_v)

                    # V22.0 P2: Read sentiment from DB off-thread (non-blocking)
                    async def _read_sentiment():
                        try:
                            import sqlite3 as _sqlite3
                            def _query():
                                with _sqlite3.connect("trading_ledger.db", timeout=3) as c:
                                    r = c.execute("SELECT score, rationale FROM sentiment_logs ORDER BY timestamp DESC LIMIT 1").fetchone()
                                    return (float(r[0]), r[1]) if r else (0.0, "Market neutral.")
                            return await asyncio.to_thread(_query)
                        except Exception:
                            return 0.0, "Market neutral."

                    # V30.6: Fetch true MLOFI signal from Oracle's centralized system_state.
                    # Falls back to kline-derived OFI proxy if Oracle hasn't published yet.
                    try:
                        mlofi_normalized = await asyncio.to_thread(
                            self.db.load_state, "mlofi_normalized", None
                        )
                        ofi_for_guard = float(mlofi_normalized) if mlofi_normalized is not None else ofi
                    except Exception:
                        ofi_for_guard = ofi

                    vol_series_data = self.dfs[symbol]['volume'].tail(100).values
                    vol_24h = float(self.dfs[symbol]['volume'].tail(288).sum())
                    micro_ctx = {
                        "ofi": ofi_for_guard,
                        "trade_volumes": np.array([taker_buys, taker_sells], dtype=np.float64),
                        "trade_signs": np.array([1.0, -1.0], dtype=np.float64),
                        "vol_24h": vol_24h if vol_24h > 0 else 1000.0,
                        "volume_series": vol_series_data,
                        "funding_rate": config.FUNDING_RATE_PERCENT,
                        "spread": 1.0, # Default proxy
                        "dt": 300.0 # 5-minute candle interval
                    }

                    # V22.0 P2: Read sentiment from DB off-thread (non-blocking)
                    s_score, s_reason = await _read_sentiment()

                    # Define concurrent wrapper tasks to run with LatencyTracer
                    async def run_ai_audit():
                        async with LatencyTracer("AI_AUDIT", budget_ms=180000):
                            try:
                                return await ai_engine.get_llm(engine_name=self.log_name).get_risk_params(
                                    regime=hard_regime, atr_pct=float(r1["atr_pct"]), signal=signal,
                                    phase=trend_ctx.phase, session_pnl=current_session_pnl, streak=0,
                                    btc_context=self.last_btc_trend,
                                    market_context=str(self.dfs[symbol]['close'].tail(10).tolist()),
                                    htf_trend="unknown", current_regime=hard_regime,
                                    ema_slope=float(r1.get("ema_slope", 0.0)),
                                    patterns=patterns, ema_stack=is_ema_stack, timeout_sec=175.0, symbol=symbol
                                )
                            except Exception as e:
                                self.log.error(f"[AI-ENGINE] Risk params query failed: {e}")
                                return {"entry_quality": 0.0, "notes": f"AI Audit failed: {e}"}

                    async def run_boardroom():
                        async with LatencyTracer("BOARDROOM", budget_ms=180000):
                            try:
                                return await self.boardroom.get_conviction(
                                    symbol=symbol, side=signal,
                                    context_data={
                                        "regime": hard_regime, "phase": trend_ctx.phase,
                                        "quality": "pending", "atr": atr, "notes": "pending",
                                        "active_strategy_mode": self.active_strategy_mode,
                                        "rl_dynamic_rsi": dyn_rsi, "rl_dynamic_vwap": dyn_vwap,
                                        "narrative_sentiment_score": round(s_score, 2),
                                        "narrative_sentiment_rationale": s_reason,
                                        "recent_prices": [round(x, 2) for x in hist_p],
                                        "recent_volumes": [round(x, 2) for x in hist_v],
                                        "vol_divergence_math_score": div_data
                                    }
                                )
                            except Exception as e:
                                self.log.error(f"[BOARDROOM] Conviction query failed: {e}")
                                return {"execute": False, "conviction_score": 0.0, "final_rationale": f"Boardroom failed: {e}"}

                    async def run_microstructure():
                        async with LatencyTracer("MICROSTRUCTURE_VETO", budget_ms=200):
                            try:
                                return await self.microstructure_guard.evaluate_pre_trade_veto(micro_ctx)
                            except Exception as e:
                                self.log.error(f"[MICROSTRUCTURE] Microstructure veto calculation failed: {e}")
                                return True, f"Microstructure guard error: {e}"

                    try:
                        ai_res, boardroom_res, (m_passed, m_reason) = await asyncio.wait_for(
                            asyncio.gather(
                                run_ai_audit(),
                                run_boardroom(),
                                run_microstructure()
                            ),
                            timeout=180.0
                        )
                    except asyncio.TimeoutError:
                        self.log.warning("[PERF] Parallel AI Audits / Veto timed out (>180s). Using safe fallbacks.")
                        ai_res = {"entry_quality": 0.5, "notes": "LLM timeout fallback."}
                        boardroom_res = {"execute": False, "conviction_score": 0.0, "final_rationale": "Timeout fallback."}
                        m_passed, m_reason = True, "Microstructure veto timeout."

                    # [CP35-S1-C1] Anchored to background_tasks to prevent orphan GC
                    self._run_background_task(
                        self.paper_ex.update_system_state("last_ai_thoughts_berserker", ai_res.get("notes", "No notes."))
                    )

                    # Dynamic Observation Mode Gating
                    if os.getenv("OBSERVATION_MODE", "0") == "1":
                        self.log.warning("[OBSERVATION MODE] AI risk engine bypassed — OBSERVATION_MODE=1 is set.")
                        quality = 100.0
                        notes = ai_res.get("notes", "No AI notes") if 'ai_res' in locals() else "Observation Mode Override"
                    else:
                        quality = ai_res.get("entry_quality", 0.0) if 'ai_res' in locals() else 0.0
                        notes = ai_res.get("notes", "No AI notes") if 'ai_res' in locals() else "AI Audit failed."

                    if quality < dyn_veto_threshold:
                        veto_reason = f"AI Conviction too low: {quality:.1f} | {notes}"
                        self.log.warning(f"[VETO] Pre-trade AI Audit REJECTED {symbol} (Quality: {quality:.1f}). Reason: {notes}")
                        self.veto_pause_until[symbol] = datetime.now() + timedelta(minutes=30)
                        tel_packet["reasoning"] = f"[VETO {quality:.1f}] {notes}"
                        tel_packet["quality"] = quality
                        # [CP35-S1-C1] Anchored to background_tasks to prevent orphan GC
                        self._run_background_task(self.paper_ex.save_telemetry(symbol, tel_packet))
                        await self._set_status("Status: Cooling Down from Veto", veto=veto_reason)

                    else:
                        # V22.0 P1: Use cached ticker price from monitor_position (zero network roundtrip)
                        cached_ticker = self._last_ticker.get(symbol)
                        if cached_ticker and cached_ticker.get('ask'):
                            exec_p = cached_ticker['ask'] if signal == "long" else cached_ticker['bid']
                            self.log.debug(f"[PERF] Using cached ticker price: {exec_p}")
                        else:
                            # Fallback: live fetch only if cache is empty
                            async with LatencyTracer("TICKER_FETCH", budget_ms=200):
                                ticker = await self.exchange.fetch_ticker(symbol)
                            if ticker is None or ticker.get('ask') is None:
                                exec_p = price
                            else:
                                exec_p = ticker['ask'] if signal == "long" else ticker['bid']

                        # V16.4: Volatility Warden
                        vol_mult = 1.0
                        try:
                            def _read_vol_state():
                                if os.path.exists("volatility_state.json"):
                                    with open("volatility_state.json", "r") as f:
                                        return json.load(f)
                                return None
                            v_data = await asyncio.to_thread(_read_vol_state)
                            if v_data:
                                vol_mult = v_data.get("recommended_risk_multiplier", 1.0)
                                if v_data.get("market_vibe") != "TRENDING":
                                    self.log.info(f"[WARDEN] Adaptive Risk Active: {v_data.get('market_vibe')} market. Applying {vol_mult}x.")
                        except Exception as ve:
                            self.log.error(f"[WARDEN] Volatility Warden state read error: {ve}. Reverting to 1.0 multiplier.")
                            vol_mult = 1.0

                        qty = calculate_risk_qty(self.paper_ex, symbol, exec_p, atr, 0.5 * vol_mult, atr_mult=1.2, is_scale_in=is_scale_in)

                        # V43.0 Unified Sentiment Oracle Kill-Switch (2h max-age, fail-closed, WARNING=0.5x size-down)
                        try:
                            def _read_sent_state():
                                if os.path.exists("sentiment_state.json"):
                                    with open("sentiment_state.json", "r", encoding="utf-8") as f:
                                        return json.load(f)
                                return None
                            s_data = await asyncio.to_thread(_read_sent_state)
                            if not s_data or not s_data.get("last_update"):
                                self.log.warning(f"[SENTIMENT] sentiment_state.json missing or lacks last_update (fail-closed) for {symbol}")
                                return
                            lu_dt = datetime.fromisoformat(s_data["last_update"])
                            if lu_dt.tzinfo is None:
                                lu_dt = lu_dt.replace(tzinfo=UTC)
                            if (datetime.now(UTC) - lu_dt).total_seconds() > 7200:
                                self.log.warning(f"[SENTIMENT] sentiment_state.json stale (>2h: {s_data['last_update']}) — blocking {symbol} (fail-closed)")
                                return

                            base = symbol.split('/')[0].upper()
                            g_info = s_data.get("GLOBAL", {}) if isinstance(s_data.get("GLOBAL"), dict) else {}
                            b_info = s_data.get(base, {}) if isinstance(s_data.get(base), dict) else {}
                            g_status = str(g_info.get("status", "SAFE")).upper()
                            b_status = str(b_info.get("status", "SAFE")).upper()

                            if g_status == "BLOCK" or b_status == "BLOCK":
                                reason = b_info.get("reason") if b_status == "BLOCK" else g_info.get("reason", "Global block")
                                self.log.warning(f"[SENTIMENT] TRADE BLOCKED BY ORACLE ({symbol}): {reason}")
                                self._run_background_task(self.paper_ex.save_log("SENTIMENT", f"BLOCKED {symbol}: {reason}"))
                                return
                            if g_status == "WARNING" or b_status == "WARNING":
                                warn_reason = b_info.get("reason") if b_status == "WARNING" else g_info.get("reason", "Caution")
                                qty = round(qty * 0.5, 6)
                                self.log.warning(f"[SENTIMENT] WARNING tier for {symbol} -> scaling qty 0.5x ({warn_reason})")
                        except Exception as se:
                            self.log.error(f"Sentiment check failed (fail-closed): {se}")
                            return

                        if qty > 0:
                            # V15.8: Macro Circuit Breaker
                            try:
                                res = await self.macro_breaker.fetch()
                                if not res.get("safe_to_trade", True):
                                    self.log.warning(f"[WARDEN] MACRO HALT: {symbol}. Reason: {res.get('reason')}")
                                    # [CP35-S1-C1] Anchored to background_tasks to prevent orphan GC
                                    self._run_background_task(self.paper_ex.save_log("WARDEN", f"MACRO HALT on {symbol}: {res.get('reason')}"))
                                    await self._set_status(f"Status: Halted by Macro Circuit Breaker ({symbol})")
                                    return
                            except Exception as mbe:
                                self.log.critical(f"[WARDEN] Macro Circuit Breaker fetch failed: {mbe}. Refusing trade as safety fallback.")
                                await self._set_status(f"Status: VETO by Macro Circuit Breaker failure ({symbol})", veto=f"Macro breaker error: {mbe}")
                                return

                            # V19.0 Wall Street Circuit Breaker — P2: async DB read
                            if signal == 'long':
                                try:
                                    import sqlite3 as _sqlite3
                                    def _spy_query():
                                        conn = _sqlite3.connect("trading_ledger.db", timeout=3.0)
                                        try:
                                            # Check table exists before querying
                                            tbl = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='macro_logs'").fetchone()
                                            if not tbl:
                                                return None
                                            return conn.execute("SELECT spy_health FROM macro_logs ORDER BY timestamp DESC LIMIT 1").fetchone()
                                        finally:
                                            conn.close()
                                    res = await asyncio.to_thread(_spy_query)
                                    if res and float(res[0]) < -0.05:
                                        spy_h = float(res[0])
                                        self.log.critical(f"[MACRO HALT] SPY at {spy_h:+.2%}. Strangling Long Entries.")
                                        await self._set_status(f"Status: [MACRO HALT] Wall Street Contagion ({symbol})")
                                        return
                                except Exception as e:
                                    self.log.error(f"V19.0 Circuit Breaker Error: {e}")

                            # V26.1 Alpha Auditor Gate: Block execution if Berserker Strategy is disabled.
                            try:
                                import sqlite3 as _sqlite3
                                def _audit_check():
                                    conn = _sqlite3.connect("trading_data.db", timeout=5.0)
                                    try:
                                        res = conn.execute("SELECT value FROM system_state WHERE key = 'DISABLED_SIGNAL_Berserker-Strategy'").fetchone()
                                        return res[0] if res else "0"
                                    finally:
                                        conn.close()
                                auditor_v = await asyncio.to_thread(_audit_check)
                                if auditor_v == "1":
                                    self.log.critical(f"[ALPHA-AUDITOR VETO] Berserker Strategy is currently DISABLED due to poor performance.")
                                    await self._set_status(f"Status: VETO by Alpha Auditor ({symbol})")
                                    return
                            except Exception as e:
                                self.log.error(f"V26.1 Auditor Gate Error: {e}")

                            # V35.0 Exclusive Asset Mutex Lock Gate (Bot #5 Swing Architect)
                            try:
                                def _check_asset_lock():
                                    try:
                                        with _sqlite3.connect("trading_data.db", timeout=5.0) as l_conn:
                                            row = l_conn.execute("SELECT locked_by, reason, heartbeat FROM asset_locks WHERE symbol = ?", (symbol,)).fetchone()
                                            if row:
                                                hb = row[2]
                                                if hb:
                                                    hb_dt = datetime.fromisoformat(hb)
                                                    if (datetime.now(UTC) - hb_dt).total_seconds() > 900:
                                                        return None  # Stale lock ignored (>15m)
                                                return row[0], row[1]
                                    except Exception:
                                        return None
                                    return None
                                lock_res = await asyncio.to_thread(_check_asset_lock)
                                if lock_res:
                                    locked_by, lock_rsn = lock_res
                                    self.log.info(f"[ASSET-LOCK] [SKIP] {symbol} is locked by {locked_by} ({lock_rsn}). Skipping 5m entry.")
                                    await self._set_status(f"Status: Asset Locked ({locked_by})")
                                    return
                            except Exception as le:
                                self.log.debug(f"Asset lock check note: {le}")

                            # V26.2 Iceberg Absorption Gate: 1m Resolution Scan
                            is_iceberg = False
                            try:
                                async with LatencyTracer("ICEBERG_AUDIT", budget_ms=1000):
                                    k_1m = await self.exchange.fetch_ohlcv(symbol, timeframe='1m', limit=21)
                                    if len(k_1m) >= 5:
                                        # Extract OHLCV to Dataframe
                                        df_1m = pd.DataFrame(k_1m, columns=["t","o","h","l","c","v"])
                                        last_row = df_1m.iloc[-1]
                                        v_hist = df_1m["v"].iloc[:-1].astype(float)
                                        
                                        avg_v_20 = float(v_hist.tail(20).mean())
                                        cur_v = float(last_row["v"])
                                        cur_o = float(last_row["o"])
                                        cur_c = float(last_row["c"])
                                        candle_range = abs(cur_c - cur_o)
                                        
                                        # Iceberg condition: Massive Volume (>3x) AND tight consolidative movement
                                        if cur_v > (avg_v_20 * 3) and candle_range < (atr * 0.1):
                                            is_iceberg = True
                                            self.log.info(f"[ICEBERG DETECTED] Valid absorption footprint found on {symbol}!")
                            except Exception as ice_e:
                                self.log.warning(f"Iceberg Audit processing failure: {ice_e}")

                            # Check Boardroom verdict (already retrieved in parallel gather)
                            try:
                                score = boardroom_res.get("conviction_score", 0.0)
                                execute = boardroom_res.get("execute", False)
                                rationale = boardroom_res.get("final_rationale", "No reason given.")

                                # V26.2 Front-Run Logic: Apply boost for confirmed Absorption
                                if is_iceberg:
                                    self.log.info(f"[ICEBERG BOOST] Applying +20% conviction bonus to signal on {symbol}")
                                    score = min(1.0, score + 0.20)
                                    rationale += " [ICEBERG]"
                                    notes += " [ICEBERG]"

                                # Dynamic Boardroom Gating
                                if os.getenv("OBSERVATION_MODE", "0") == "1":
                                    self.log.warning("[OBSERVATION MODE] Boardroom conviction bypassed — OBSERVATION_MODE=1 is set.")
                                    execute = True
                                    score = 1.0

                                required_score = 0.75
                                if s_score < -0.5:
                                    required_score *= 1.20
                                    self.log.warning(f"[THRESHOLD HIKE] Severe Fear ({s_score:+.2f}). Threshold → {required_score:.2f}")

                                if not execute or score < required_score:
                                    self.log.warning(f"[BOARDROOM] Strike Aborted: Low Conviction ({score}). Rationale: {rationale}")
                                    # [CP35-S1-C1] Anchored to background_tasks to prevent orphan GC
                                    self._run_background_task(self.paper_ex.save_log("BOARDROOM", f"Strike Aborted ({score}): {rationale}"))

                                    await self._set_status(f"Status: Vetoed by Boardroom ({symbol})")
                                    return
                                self.log.info(f"[BOARDROOM] APPROVED by Portfolio Manager (Score: {score}). Rationale: {rationale}")

                                # [V22.0] DYNAMIC KELLY SIZING
                                # Resolves real balance dynamically — exchange API first, paper_ex fallback
                                try:
                                    # V22.0 P1: TTL-cached balance — avoids live roundtrip per-trade
                                    import time as _t
                                    _now_t = _t.perf_counter()
                                    if (_now_t - self._balance_fetched_at) > self._balance_ttl or self._last_balance <= 0:
                                        try:
                                            async with LatencyTracer("BALANCE_FETCH", budget_ms=200):
                                                if hasattr(self.paper_ex, "uta_client") and self.paper_ex.uta_client:
                                                    fetched = await self.paper_ex.uta_client.fetch_balance()
                                                elif self.exchange:
                                                    raw_bal = await self.exchange.fetch_balance()
                                                    fetched = float(
                                                        raw_bal.get("free", {}).get("USDT")
                                                        or raw_bal.get("USDT", {}).get("free", 0.0)
                                                    )
                                                else:
                                                    fetched = float(self.paper_ex.balance)
                                            if fetched > 0:
                                                self._last_balance = fetched
                                                self._balance_fetched_at = _now_t
                                        except Exception as bal_err:
                                            self.log.warning(f"[RISK-MANAGER] Balance fetch failed: {bal_err}.")
                                    current_balance = self._last_balance if self._last_balance > 0 else float(self.paper_ex.balance)
                                    self.log.info(f"[RISK-MANAGER] Balance: ${current_balance:.2f}")


                                    # Convert 0-1 boardroom score to 0-100 for the Risk Manager
                                    llm_conviction_score = float(score) * 100.0
                                    
                                    # V25.3 Block 3: Now async, called directly on event loop
                                    allocated_capital, risk_pct = await self.risk_manager.get_position_size(
                                        current_balance, llm_conviction_score,
                                        symbol, list(self.active_positions.keys())
                                    )
                                    if allocated_capital <= 0:
                                        self.log.warning("[RISK-MANAGER] Kelly edge negative or zero conviction. Trade vaporized.")
                                        await self._set_status(f"Status: VETO by Risk Manager (zero edge)")
                                        return
                                    # Derive qty from Kelly-allocated capital at current leverage (50% position scaling for 5m high-frequency safety)
                                    kelly_qty = ((allocated_capital * 0.5) * config.BERSERKER_LEVERAGE) / exec_p
                                    if kelly_qty > 0:
                                        qty = kelly_qty  # Override the static calculate_risk_qty result
                                    self.log.info(f"[RISK-MANAGER] Kelly-sized qty={qty:.6f} ({risk_pct*100:.2f}% | ${allocated_capital*0.5:.2f} of ${current_balance:.2f} balance).")
                                except Exception as rme:
                                    self.log.warning(f"[RISK-MANAGER] Sizing block failed: {rme}. Falling back to static qty.")
                            except Exception as be:
                                self.log.error(f"[BOARDROOM-CRASH] Execution aborted on {symbol}: {be}")
                                await self._set_status(f"Status: Vetoed by Boardroom Crash ({symbol})")
                                return

                            # V26.3 Adaptive Exits Switch
                            # V31.0 P1: Capture final_sl / final_tp BEFORE execution so the Ghost-Grid
                            # custom stops are not silently discarded by a second calculate_exits() call.
                            if self.active_strategy_mode == "MeanRev":
                                # Ghost-Grid Mode Targets: Mean TP, tight ATR SL (supports Bi-Directional)
                                tp_raw = r1.get("sma20")
                                if signal == "short":
                                    final_tp = float(tp_raw) if (tp_raw is not None and not pd.isna(tp_raw)) else float(exec_p) * 0.99
                                    final_sl = float(exec_p) + (1.2 * atr)
                                else:
                                    final_tp = float(tp_raw) if (tp_raw is not None and not pd.isna(tp_raw)) else float(exec_p) * 1.01
                                    final_sl = float(exec_p) - (1.2 * atr)
                                self.log.info(f"[GHOST-GRID] Custom Exit ({signal.upper()}) applied: TP=${final_tp:.2f}, SL=${final_sl:.2f}")
                            else:
                                sl_mult = float(dyn_params.get("sl_atr_mult", 1.2))
                                tp_mult = float(dyn_params.get("tp_atr_mult", 4.5))
                                if signal == "long":
                                    final_sl = float(exec_p) - (sl_mult * atr)
                                    final_tp = float(exec_p) + (tp_mult * atr)
                                else:
                                    final_sl = float(exec_p) + (sl_mult * atr)
                                    final_tp = float(exec_p) - (tp_mult * atr)
                                self.log.info(f"[DYNAMIC-REGIME-EXITS] {signal.upper()} Exits Applied: SL=${final_sl:.2f} ({sl_mult}x ATR), TP=${final_tp:.2f} ({tp_mult}x ATR)")

                            # V26.2 Value at Risk (VaR) Hard Gate (Late execution audit)
                            try:
                                new_risk_total = float(qty) * abs(float(exec_p) - float(final_sl))
                                is_var_valid = await asyncio.to_thread(
                                    self.risk_manager.get_portfolio_var, current_balance, new_risk_total
                                )
                                if not is_var_valid:
                                    self.log.critical(f"[VaR-HALT] VETO on {symbol}: Combined trade risk exceeds 10% Portfolio limit!")
                                    await self._set_status(f"Status: VETO by Portfolio VaR ({symbol})")
                                    return
                            except Exception as ve:
                                self.log.error(f"VaR Validation system exception: {ve}. Defaulting to standard block execution.")

                            # [V21.0] SENTINEL — pure in-memory cache lookup (< 1ms, V22.0)
                            try:
                                _ticker = symbol.split("/")[0]
                                sentinel_status, sentinel_msg = self.news_sentinel.perform_disaster_audit(_ticker)
                                if sentinel_status == "VETO":
                                    self.log.error(f"[SENTINEL FATAL VETO] Execution halted! {sentinel_msg}")
                                    await self._set_status(f"Status: VETO by Sentinel ({_ticker})", veto=sentinel_msg)
                                    # [CP35-S1-C1] Anchored to background_tasks to prevent orphan GC
                                    self._run_background_task(self.paper_ex.save_log("SENTINEL", f"FATAL VETO: {sentinel_msg}"))
                                    return
                                self.log.info(f"[SENTINEL] {sentinel_msg} — Strike approved.")
                            except Exception as se:
                                self.log.warning(f"[SENTINEL] Audit failed: {se}. Allowing trade.")

                            # 🎯 LOG FULL SEGMENT TIMINGS before placing order
                            log_full_trace(symbol)

                            # V24.0 Order Book Wall Audit
                            # OBSERVATION MODE: Bypass orderbook wall veto
                            self.log.warning("[CRITICAL] !!! ORDERBOOK WALL AUDIT OBSERVATION MODE FORCED - BYPASSING VETO !!!")
                            is_supported = True



                            # V30.0 Standardized Execution Routing: Almgren-Chriss Stealth Slicing
                            exec_success = await self.executor.execute_trade(
                                strategy_decision={
                                    "action_name": signal.upper(),
                                    "confidence": score
                                },
                                current_price=exec_p,
                                market_state={"regime": hard_regime, "phase": trend_ctx.phase, "quality": quality},
                                symbol=symbol
                            )

                            if exec_success:
                                # V31.0 P1: Use final_sl/final_tp captured before execution.
                                # Flash Crash Filter: compute hard_sl as the wider Bitget-native backstop
                                _hard_mult = float(getattr(config, "HARD_SL_ATR_MULTIPLIER", 3.5))
                                _hard_sl = (float(exec_p) - (_hard_mult * atr)) if signal == "long" else (float(exec_p) + (_hard_mult * atr))
                                pos = await self.paper_ex.open_position(
                                    symbol, signal, qty, exec_p,
                                    sl=final_sl, tp=final_tp, hard_sl=_hard_sl,
                                    metadata={"regime": hard_regime, "phase": trend_ctx.phase, "quality": quality, "notes": rationale, "ai_rationale": rationale}
                                )
                                if not pos:
                                    self.log.warning(f"[EXECUTION-ABORT] open_position rejected trade for {symbol}.")
                                    await self._set_status(f"Status: Trade Rejected by Exchange ({symbol})")
                                    return
                                self.log.info(f"[BERSERKER-ALERT] Executing {symbol} | Side: {signal.upper()} | Price: ${exec_p:.4f} | SoftSL: {final_sl:.4f} | HardSL: {_hard_sl:.4f} | TP: {final_tp:.4f}")
                                self.active_positions[symbol] = pos
                                
                                trade_ctx = {
                                    "symbol": symbol,
                                    "side": signal,
                                    "price": exec_p,
                                    "qty": qty,
                                    "stop_loss": final_sl,
                                    "take_profit": final_tp,
                                    "regime": hard_regime,
                                    "conviction": f"{round(float(score)*10)}/10",
                                    "ai_reasoning": rationale,
                                    "quality": quality,
                                    "engine": "Berserker (5m)",
                                    "leverage": pos.get("leverage", 5.0)
                                }
                                self._run_background_task(send_trade_log_async(symbol, signal, exec_p, status="OPENED", trade_context=trade_ctx))
                                tel_packet["reasoning"] = f"[EXECUTED {signal.upper()}] AI Quality: {quality:.1f}"
                                tel_packet["quality"] = quality
                                self._run_background_task(self.paper_ex.save_telemetry(symbol, tel_packet))
                            else:
                                self.log.warning(f"[EXECUTION-ABORT] Stealth router declined trade for {symbol}.")
                                await self._set_status(f"Status: VETO by Stealth Router ({symbol})")
                                return

            self.log.debug(f"[BERSERKER-CLOCK] 5m Cycle for {symbol} complete.")
        except Exception as e:
            self.log.error(f"Silent failure caught in [handle_candle_close]: {e}\n{traceback.format_exc()}")
        
        # V27.0 Explicit Terminal Heartbeat Ensure Cycle Completed
        try:
            self.db.update_heartbeat(self.log_name)
        except: pass


    async def _check_orderbook_wall(self, symbol: str, side: str, entry_p: float) -> bool:
        try:
            async with LatencyTracer("ORDERBOOK_AUDIT", budget_ms=200):
                ob = await self.exchange.fetch_order_book(symbol, limit=50)
                target_side = 'bids' if side == 'long' else 'asks'
                data = ob.get(target_side, [])
                
                if not data or len(data) < 5:
                    self.log.warning(f"[ORDERBOOK] Insufficient depth on {target_side}. Blocking strike.")
                    return False
                
                vols = [float(x[1]) for x in data]
                avg_vol = sum(vols) / len(vols)
                
                boundary_mult = 0.995 if side == 'long' else 1.005
                boundary_price = entry_p * boundary_mult
                
                relevant_vols = []
                for p_raw, v_raw in data:
                    p_pt, v_pt = float(p_raw), float(v_raw)
                    if side == 'long':
                        if boundary_price <= p_pt <= entry_p:
                            relevant_vols.append(v_pt)
                    else:
                        if entry_p <= p_pt <= boundary_price:
                            relevant_vols.append(v_pt)
                
                if not relevant_vols:
                    self.log.warning(f"[ORDERBOOK] Zero {target_side.upper()} walls found within 0.5% range. Veto.")
                    return False
                
                max_local = max(relevant_vols)
                if max_local >= (avg_vol * 2.5):
                    self.log.info(f"[ORDERBOOK] Verified {target_side[:-1].upper()} Wall: {max_local:.1f} ({max_local/avg_vol:.1f}x avg). Support Confirmed.")
                    return True
                else:
                    self.log.warning(f"[ORDERBOOK] VETO: Max wall {max_local:.1f} < avg*2.5 ({avg_vol*2.5:.1f}).")
                    return False
        except Exception as e:
            self.log.error(f"[ORDERBOOK] Audit Exception: {e}. Fallback FALSE.")
            return False


    async def monitor_position(self, symbol, current_price):
        # V22.0 P1: Cache last known price here — fires every WebSocket tick,
        # so by the time a candle closes, we already have a fresh price.
        self._last_ticker[symbol] = {"ask": current_price, "bid": current_price, "last": current_price}
        try:
            # V15.5: Added safety check to prevent KeyError during startup/shutdown
            if symbol not in self.active_positions: return
            pos = self.active_positions.get(symbol)
            if not pos or pos.get("is_closing"): return
            
            # V25.6 Sync Step: Prevent Memory Stale State from external DCA Rescue
            await self.paper_ex.sync_positions_from_db()
            if self.paper_ex.positions.get(symbol):
                self.active_positions[symbol] = self.paper_ex.positions[symbol]
                pos = self.active_positions[symbol]
            else:
                self.active_positions[symbol] = None
                return
                
            side, price = pos["side"], current_price
            if side == "long": pos["peak_price"] = max(pos["peak_price"], price)
            else: pos["peak_price"] = min(pos["peak_price"], price)
            atr = self.dfs[symbol]["atr14"].iloc[-1]
            import math as _math
            atr = max(float(atr), 0.0001) if (atr is not None and not _math.isnan(float(atr))) else 0.0001  # V_AUDIT_FIX: NaN guard
            
            # V16.1: Whale Watcher - Adaptive Trailing Stop (Flash-Crash Protection)
            try:
                def _read_whale_state():
                    if os.path.exists("whale_state.json"):
                        with open("whale_state.json", "r") as f:
                            return json.load(f)
                    return None
                w_data = await asyncio.to_thread(_read_whale_state)
                if w_data and w_data.get("whale_status") == "DANGER":
                    # Move SL to 2% below current market price (Trailing Stop)
                    danger_sl = price * 0.98 if side == "long" else price * 1.02
                    if (side == "long" and danger_sl > pos["stop_loss"]) or (side == "short" and danger_sl < pos["stop_loss"]):
                        pos["stop_loss"] = danger_sl
                        self.log.warning(f"[WHALE ALERT] DANGER DETECTED. Tightening SL to 2% Trailing: {danger_sl:.2f}")
                        await self.paper_ex.save_log("WHALE", f"DANGER: Tightened {symbol} SL to {danger_sl:.2f}")
                        await self.paper_ex.sync_position_updates(symbol, pos)
            except Exception as we:
                self.log.error(f"Whale check failed: {we}")

            p_dist = (price - pos["entry_price"] if side == "long" else pos["entry_price"] - price)
            if p_dist >= 3.0 * atr and not pos.get("partial_tp_hit"):
                from live_paper_exchange import get_contract_specs
                specs = get_contract_specs(symbol)
                min_step = specs["min_qty"]
                # Only take partial TP if BOTH the closed portion and remaining portion are >= min_qty
                if (pos["qty"] / 2.0) >= min_step and (pos["qty"] - (pos["qty"] / 2.0)) >= min_step:
                    partial_qty = round(pos["qty"] / 2.0, specs["decimals"]) if specs["decimals"] > 0 else float(int(pos["qty"] // 2))
                    await self.paper_ex.update_position_flag(symbol, "partial_tp_hit", True)
                    pnl_or_none = await self.paper_ex.close_position(symbol, price, t_type="partial_tp", partial_qty=partial_qty)
                    if pnl_or_none is None:
                        self.log.warning(f"[PARTIAL TP] {symbol} was already 0 on exchange. Dropping memory state.")
                        self.active_positions[symbol] = None
                        return
                    else:
                        self.log.info(f"[PARTIAL TP] {symbol} secured 50% at 3.0x ATR momentum expansion. PnL: ${pnl_or_none:+.4f}")
                        partial_actual_pnl = pnl_or_none if isinstance(pnl_or_none, (int, float)) else (p_dist * partial_qty)
                        partial_ctx = {
                            "qty": partial_qty,
                            "pnl": partial_actual_pnl,
                            "engine": f"Berserker ({self.timeframe})",
                            "entry_price": pos["entry_price"],
                            "peak_price": pos.get("peak_price", price),
                            "entry_time": pos.get("entry_time"),
                            "close_trigger": "PARTIAL TAKE-PROFIT (50%)",
                            "close_reason": f"Secured 50% partial profit at 3.0x ATR momentum expansion (${price:,.4f}) to lock in gains and de-risk position while letting runner trail."
                        }
                        self._run_background_task(send_trade_log_async(symbol, side, price, pnl=partial_actual_pnl, status="PARTIAL_TP", trade_context=partial_ctx))
                else:
                    self.log.info(f"[PARTIAL TP] {symbol} quantity ({pos['qty']}) is at exchange minimum ({min_step}). Skipping 50% split to preserve position on Bitget; trailing full position.")
                await self.paper_ex.update_position_flag(symbol, "partial_tp_hit", True)
                
            if p_dist >= 2.0 * atr:
                trail_mult = 1.0 if pos.get("partial_tp_hit") else 1.8
                trail_sl = (pos["peak_price"] - (trail_mult * atr) if side == "long" else pos["peak_price"] + (trail_mult * atr))
                if (side == "long" and trail_sl > pos["stop_loss"]) or (side == "short" and trail_sl < pos["stop_loss"]):
                    pos["stop_loss"] = trail_sl
                    await self.paper_ex.sync_position_updates(symbol, pos)

            # V11.0: Detect SL vs TP hit separately so cooldown can be set on TP only
            is_sl = (side == "long" and price <= pos["stop_loss"]) or (side == "short" and price >= pos["stop_loss"])
            is_tp = (pos.get("take_profit", 0) > 0) and (
                (side == "long" and price >= pos["take_profit"]) or
                (side == "short" and price <= pos["take_profit"])
            )

            # ── Flash Crash Filter (Soft SL confirmation window) ─────────────────────────
            # Berserker runs on 5m candles — price ticks arrive ~every second from WebSocket.
            # A 60-second window safely spans 1 full 5m bar before we treat the breach as real.
            # TPs bypass the filter — they fire immediately.
            if is_sl and not is_tp:
                import time as _time
                confirmation_secs = int(getattr(config, "FLASH_CRASH_CONFIRMATION_SECS", 60))
                breach_ts = pos.get("sl_breach_time")

                if breach_ts is None:
                    pos["sl_breach_time"] = _time.time()
                    await self.paper_ex.sync_position_updates(symbol, pos)
                    self.log.warning(
                        f"[FLASH FILTER] ⚡ {symbol} breached Soft SL at ${price:,.4f} "
                        f"(SL=${pos['stop_loss']:,.4f}). Starting {confirmation_secs}s confirmation window."
                    )
                    is_sl = False
                elif (_time.time() - breach_ts) < confirmation_secs:
                    elapsed = int(_time.time() - breach_ts)
                    self.log.info(
                        f"[FLASH FILTER] ⏳ {symbol} still below Soft SL "
                        f"(${price:,.4f} < ${pos['stop_loss']:,.4f}) — {elapsed}s/{confirmation_secs}s. Holding."
                    )
                    is_sl = False
                else:
                    elapsed = int(_time.time() - breach_ts)
                    self.log.info(
                        f"[FLASH FILTER] ✅ {symbol} Soft SL confirmed after {elapsed}s. "
                        f"Genuine move — closing at ${price:,.4f}."
                    )
                    pos["sl_breach_time"] = None
                    await self.paper_ex.sync_position_updates(symbol, pos)
            elif not is_sl and pos.get("sl_breach_time") is not None:
                import time as _time
                elapsed = int(_time.time() - pos["sl_breach_time"])
                self._run_background_task(send_alert_async(
                    f"⚡ [{symbol}] Flash Crash Filter Saved Position (Berserker)",
                    f"Price breached Soft SL ${pos['stop_loss']:,.4f} but recovered within {elapsed}s.\nPosition preserved. No loss taken."
                ))
                self.log.info(
                    f"[FLASH FILTER] 🛡️ {symbol} recovered above Soft SL after {elapsed}s — "
                    f"FLASH CRASH confirmed. Position preserved."
                )
                pos["sl_breach_time"] = None
                await self.paper_ex.sync_position_updates(symbol, pos)
            # ─────────────────────────────────────────────────────────────────────────────

            if is_sl or is_tp:
                close_type = "tp_hit" if is_tp else "sl_hit"
                # Check if stop loss was dynamic trailing stop or initial stop loss
                was_trailed = pos.get("peak_price") is not None and (
                    (side == "long" and pos["stop_loss"] > pos["entry_price"]) or 
                    (side == "short" and pos["stop_loss"] < pos["entry_price"])
                )
                
                if is_tp:
                    close_trig = "TAKE-PROFIT TARGET HIT"
                    close_rsn = f"Price successfully reached Take-Profit target at ${price:,.4f} (Target level was ${pos.get('take_profit', price):,.4f}), securing full profits."
                elif was_trailed:
                    close_trig = "DYNAMIC TRAILING STOP"
                    close_rsn = f"Dynamic Trailing Stop triggered at ${price:,.4f} after price reached a peak of ${pos.get('peak_price', price):,.4f}. Trailing stop locked in accrued profits against retracement."
                else:
                    close_trig = "INITIAL STOP-LOSS HIT"
                    close_rsn = f"Initial Stop-Loss hit at ${price:,.4f} (Stop level was ${pos.get('stop_loss', price):,.4f}). Automated risk firewall executed to strictly contain drawdown."

                # V15.6: Flag as closing to prevent re-entry into this block during DB latency/restarts
                await self.paper_ex.update_position_flag(symbol, "is_closing", True)
                pnl_or_none = await self.paper_ex.close_position(symbol, price, t_type=close_type)
                if pnl_or_none is None:
                    # Reset flag on fail to avoid permanent lock
                    await self.paper_ex.update_position_flag(symbol, "is_closing", False)
                    return
                
                # Fix: Calculate exact dollar PnL rather than raw price distance per coin
                actual_pnl = pnl_or_none if isinstance(pnl_or_none, (int, float)) else ((price - pos["entry_price"] if side == "long" else pos["entry_price"] - price) * pos["qty"])

                # V16.3: Dispatch Non-blocking Email Alert with actual dollar PnL and full reason context
                close_ctx = {
                    "qty": pos["qty"],
                    "pnl": actual_pnl,
                    "engine": f"Berserker ({self.timeframe})",
                    "entry_price": pos["entry_price"],
                    "peak_price": pos.get("peak_price", price),
                    "entry_time": pos.get("entry_time"),
                    "stop_loss": pos.get("stop_loss"),
                    "take_profit": pos.get("take_profit"),
                    "close_trigger": close_trig,
                    "close_reason": close_rsn,
                    "ai_reasoning": pos.get("metadata", {}).get("ai_rationale") or pos.get("metadata", {}).get("notes", "5m high-velocity momentum exit.")
                }
                self._run_background_task(send_trade_log_async(symbol, side, price, pnl=actual_pnl, status="CLOSED", trade_context=close_ctx))
                
                # FIX: Record trade in state manager for PnL tracking
                entry = TradeEntry(
                    symbol=symbol, side=side, entry_price=pos["entry_price"], exit_price=price,
                    qty=pos["qty"], pnl=(price - pos["entry_price"] if side == "long" else pos["entry_price"] - price) * pos["qty"],
                    balance=self.paper_ex.balance, ai_action=pos.get("metadata", {}).get("action", "unknown"),
                    final_action=close_type, ai_notes=close_rsn,
                    regime=pos.get("metadata", {}).get("regime", ""), phase=pos.get("metadata", {}).get("phase", ""),
                    quality=pos.get("metadata", {}).get("quality", 0.0), sl_mode="", tp_mode="", trail_mode="",
                    timestamp=datetime.now()
                )
                self.state.record_trade(entry)
                
                self.active_positions[symbol] = None
                if is_tp:
                    # 3-candle cooldown: 3 × 5m = 15 minutes
                    self.tp_cooldown[symbol] = {
                        "direction": side,
                        "until": datetime.now() + timedelta(minutes=15)
                    }
                    self.log.info(f"[COOLDOWN] {symbol} TP hit ({side.upper()}). Blocking same-direction re-entry for 15m.")
        except Exception as e:
            self.log.error(f"Silent failure caught in [monitor_position]: {e}")


    async def websocket_loop(self):
        sym_raw = self.symbol.replace("/", "").lower()
        url = f"wss://fstream.binance.com/market/stream?streams={sym_raw}@kline_5m"
        self.log.info(f"Connecting to Berserker Stream (5m): {url}")
        resolver = aiohttp.ThreadedResolver()
        # SECURE SSL AUTO-HANDLED BY WSS:// URI
        connector = aiohttp.TCPConnector(resolver=resolver, family=socket.AF_INET)
        async with aiohttp.ClientSession(connector=connector, connector_owner=True) as session:
            try:
                # SSL=True or False removed as URI handles it
                async with session.ws_connect(url, timeout=20.0, heartbeat=30.0, receive_timeout=300.0) as ws:
                    self.log.info("[NETWORK] Berserker Handshake Successful (5m).")
                    await self._set_status("Status: ACTIVE (Wiping Stale Telemetry)", veto="ACTIVE")
                    while True:
                        if os.path.exists("STOP_TRADING"):
                            self.log.critical("!!! STOP_TRADING FILE DETECTED !!!")
                            for s in self.symbols:
                                if self.active_positions[s]:
                                    self.log.warning(f"Kill Switch: Closing {s}")
                                    await self.paper_ex.close_position(s, float(self.dfs[s]['close'].iloc[-1]), t_type="kill_switch")
                            self.log.critical("Berserker Engine Halted.")
                            break
                        # V31.0 P1: Reset stale-connection alert flag on each successful
                        # message receipt so a future outage can send one fresh alert.
                        self._ws_stale_alerted = False

                        try:
                            symbol = self.symbol # Pre-define symbol to prevent UnboundLocalError in outer except
                            msg = await asyncio.wait_for(ws.receive(), timeout=300.0)
                            if msg.type == aiohttp.WSMsgType.CLOSED: break
                            if msg.type != aiohttp.WSMsgType.TEXT: continue
                            
                            # V28.1 native [BERSERKER-FATAL] error trap for incoming message operations
                            try:
                                payload = json.loads(msg.data)
                                data = payload.get("data", payload)
                                if 'e' not in data or data['e'] != 'kline':
                                    if isinstance(payload, dict) and ('error' in payload or 'code' in payload or payload.get('e') == 'error'):
                                        self.log.warning(f"[WS-ERROR-FRAME] Received exchange error payload: {payload}")
                                    continue
                                k = data.get('k', {})
                                
                                # V28.1 Surgical Fix: Dynamic symbol assignment bypasses obsolete BTC/SOL limit
                                symbol = self.symbol
                                
                                await self.monitor_position(symbol, float(k['c']))
                                if k.get('x', False): 
                                    self._run_background_task(self.handle_candle_close(symbol, k))
                                
                                # V27.0: Standardized Heartbeat Logic Replacement (Throttled to 30s to prevent handle leaks)
                                try:
                                    import time
                                    now_time = time.time()
                                    if not hasattr(self, "_last_heartbeat_time") or now_time - self._last_heartbeat_time >= 30.0:
                                        self.db.update_heartbeat(self.log_name)
                                        self._last_heartbeat_time = now_time
                                except: pass
                                
                            except Exception as inner_e:
                                import traceback
                                self.log.error(f"[BERSERKER-FATAL] Unhandled error in on_message: {inner_e}")
                                self.log.error(traceback.format_exc())
                                
                        except asyncio.TimeoutError:
                            self.log.warning("[NETWORK] Berserker Watchdog: Stale Connection.")
                            # V31.0 P1: Deduplicate stale-connection alerts per node.
                            # The reconnect backoff loop previously fired one email per attempt,
                            # causing floods of 7-node × reconnect storm emails.
                            if not self._ws_stale_alerted:
                                await send_alert_async(
                                    "CRITICAL WARNING: Berserker Stale Connection",
                                    f"Berserker Watchdog detected a stale WebSocket connection for {self.log_name}. Attempting reconnection sequence."
                                )
                                self._ws_stale_alerted = True
                            break
                        except Exception as e:
                            self.log.error(f"Monitor Loop Crash for {symbol}: {e}\n{traceback.format_exc()}")
                            continue
            except Exception as e:
                self.log.error(f"Berserker Stream Failed: {e}")

    def _execute_mean_reversion_logic(self, context):
        """
        V28.0 Sovereign Bi-Directional Ghost-Grid Logic.
        Requirement (LONG): Price < Lower BB AND RSI < 30.
        Requirement (SHORT): Price > Upper BB AND RSI > 70.
        Returns: "long" / "short" trigger or None
        """
        try:
            price = context['price']
            bb_lower = context['bb_lower']
            bb_upper = context.get('bb_upper', price * 1.02)
            rsi = context['rsi']
            macro = context.get('macro_regime', 'UNKNOWN')
            
            # V28.0 Stage 3: Dual-Directional Constraints
            # 1. LONG Mean Reversion (Permitted in LONG_ONLY or CHOP_GRID_ACTIVE)
            if macro in ["LONG_ONLY", "CHOP_GRID_ACTIVE", "UNKNOWN"]:
                if price < bb_lower and rsi < 30.0:
                    return "long"
            
            # 2. SHORT Mean Reversion (Permitted in SHORT_ONLY or CHOP_GRID_ACTIVE)
            if macro in ["SHORT_ONLY", "CHOP_GRID_ACTIVE", "UNKNOWN"]:
                if price > bb_upper and rsi > 70.0:
                    return "short"
        except Exception as e:
            self.log.warning(f"Mean Reversion logic audit failure: {e}")
            
        return None

    async def run(self):
        try:
            # DNS FIX: Build a dedicated aiohttp session with ThreadedResolver so
            # ccxt bypasses the broken aiodns/c-ares resolver in this environment.
            import socket as _socket
            _resolver  = aiohttp.ThreadedResolver()
            _connector = aiohttp.TCPConnector(
                resolver=_resolver,
                family=_socket.AF_INET,
                use_dns_cache=True,
                ttl_dns_cache=300,
            )
            _session = aiohttp.ClientSession(connector=_connector, connector_owner=True)
            ex_id = getattr(config, "EXCHANGE_ID", "bitget").lower()
            ex_class = getattr(ccxt, ex_id, None) or ccxt.bitget
            ex_opts = {
                "apiKey": config.ACTIVE_API_KEY,
                "secret": config.ACTIVE_SECRET_KEY,
                "enableRateLimit": True,
                "sandbox": config.SANDBOX,
                "options": {
                    "defaultType": "swap" if ex_id == "bitget" else "future",
                    "adjustForTimeDifference": True,
                    "fetchCurrencies": False,
                },
                "timeout": 30000,
                "aiohttp_session": _session,
            }
            if ex_id == "bitget":
                ex_opts["password"] = config.ACTIVE_PASSPHRASE
            self.exchange = ex_class(ex_opts)

            try:
                await self.paper_ex.clear_ui_state()
                # V15.4: Sync recovered positions from DB to engine memory
                # V15.7: Ensure all symbols are represented in active_positions
                for sym, pos in self.paper_ex.positions.items():
                    self.active_positions[sym] = pos
                await self.prime_data()
                backoff = 1
                while True:
                    try: 
                        await self.websocket_loop()
                        backoff = 1 # Reset on success
                    except Exception as e:
                        self.log.error(f"WebSocket connection dropped: {e}. Reconnecting in {backoff} seconds...")
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 60) # Exponential backoff
                    await asyncio.sleep(1)
            finally:
                if hasattr(self, "exchange") and self.exchange:
                    await self.exchange.close()
                    self.log.info("Berserker Exchange session closed.")
                if hasattr(self, "owns_warden") and self.owns_warden and hasattr(self, "macro_warden") and self.macro_warden:
                    await self.macro_warden.close()
                    self.log.info("Berserker owned Warden session closed.")
                if hasattr(self, "owns_risk_manager") and self.owns_risk_manager and hasattr(self, "risk_manager") and self.risk_manager:
                    await self.risk_manager.close()
                    self.log.info("Berserker owned RiskManager session closed.")
                if hasattr(self, "paper_ex") and self.paper_ex:
                    try:
                        self.paper_ex.close()
                        self.log.info("Berserker paper exchange session closed.")
                    except Exception:
                        pass
                if '_session' in locals() and _session:
                    await _session.close()
                    self.log.info("Berserker aiohttp ClientSession closed.")
        except Exception as main_e:
            import traceback
            self.log.error(f"[BERSERKER-FATAL] Unhandled error in main run loop: {main_e}")
            self.log.error(traceback.format_exc())

if __name__ == "__main__":
    import sys
    exec_sym = sys.argv[1] if len(sys.argv) > 1 else config.SYMBOLS[0]
    
    # Silence ccxt base exchange warning
    logging.getLogger("ccxt").setLevel(logging.ERROR)
    
    # Custom asyncio exception handler to silence unclosed sessions warning
    def custom_handler(loop, context):
        msg = context.get("message", "")
        if "Unclosed client session" in msg or "unclosed" in msg.lower():
            return
        loop.default_exception_handler(context)
        
    async def main():
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(custom_handler)
        await BerserkerTrader(symbol=exec_sym).run()
        
    asyncio.run(main())


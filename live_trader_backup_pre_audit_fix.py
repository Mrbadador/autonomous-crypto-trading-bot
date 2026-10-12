import asyncio
import socket
import aiohttp
import json
import logging
import pandas as pd
import sqlite3
import dns_fix  # noqa: F401 — patches ccxt.Exchange.open() with ThreadedResolver
import ccxt.async_support as ccxt
import os
import traceback
import random
from datetime import datetime, UTC, timedelta
import config
from executor_agent import ExecutorAgent
from email_dispatcher import send_trade_log_async, send_alert_async
from indicators import compute_indicators, classify_regime_hard, get_candle_patterns, check_ema_stack
from live_paper_exchange import LivePaperExchange
import ai_engine
from core_engine import get_deterministic_action, calculate_risk_qty, calculate_exits, build_trend_context, calculate_dynamic_risk
from state_manager import EngineState, TradeEntry
from analyst_agent import AnalystAgent
from strategist_agent import StrategistAgent
from macro_breaker_agent import MacroCircuitBreaker
from conviction_filter import ConvictionBoardroom
import sys

log = logging.getLogger("sniper")

# Setup Logging
def setup_logging(log_name="sniper"):
    global log
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"logs/{log_name}.log")
        ],
        force=True
    )
    log = logging.getLogger(log_name)
    return log

class LiveTrader:
    def __init__(self, log_name="sniper"):
        setup_logging(log_name)
        self.log_name = log_name
        self.symbols = config.SYMBOLS if config.SYMBOLS else ["SOL/USDT", "BTC/USDT"]
        self.binance_symbols = [s.lower().replace("/", "") for s in self.symbols]
        self.symbol_map = {s.lower().replace("/", ""): s for s in self.symbols}
        self.timeframe = "15m"
        self.exchange = None
        # Account segregation for SNIPER
        self.paper_ex = LivePaperExchange(initial_balance=config.INITIAL_BALANCE, account_name="sniper")
        self.dfs = {s: pd.DataFrame() for s in self.symbols}
        self.active_positions = {s: None for s in self.symbols}
        self.last_btc_trend = "UNKNOWN"
        self.last_candle_t = {s: 0 for s in self.symbols}
        self.order_lock = asyncio.Lock()
        # V11.0: TP cooldown tracker
        self.tp_cooldown = {}  # symbol -> {"direction": str, "until": datetime}
        self.state = EngineState()
        self.background_tasks = set()
        self._last_heartbeat_time = 0.0
        
        # V17.5 Dynamic Sovereignty Engines
        self.rl_analyst = AnalystAgent()
        self.rl_strategist = StrategistAgent()
        
        # We start executor with a default, but it might not be dynamic inside yet.
        # It's fine, we will use dynamic gating in handle_candle_close.
        self.executor = ExecutorAgent(min_confidence=config.MIN_NN_CONFIDENCE)
        self.macro_breaker = MacroCircuitBreaker()
        self.boardroom = ConvictionBoardroom()
        # [FIX L2] EconomicCalendarGate singleton — avoids re-instantiation on every WebSocket tick
        try:
            from economic_calendar import EconomicCalendarGate
            self.cal_gate = EconomicCalendarGate()
        except Exception as _cge:
            log.warning(f"EconomicCalendarGate init failed: {_cge}. Macro de-risk guard disabled.")
            self.cal_gate = None

    # [FIX P0-1] Converted to async — SQLite read dispatched via asyncio.to_thread()
    # so the event loop thread is never blocked during candle-close decision cycles.
    async def _get_dynamic_parameters(self):
        params = {
            "min_nn_confidence": config.MIN_NN_CONFIDENCE,
            "vol_gate_mult": config.SNIPER_VOL_GATE_MULT
        }
        def _fetch():
            try:
                with sqlite3.connect("trading_data.db", timeout=10) as conn:
                    c = conn.cursor()
                    c.execute("SELECT param_name, param_value FROM bot_parameters WHERE bot_name = 'sniper'")
                    return c.fetchall()
            except Exception as e:
                log.error(f"Failed to fetch dynamic params: {e}")
                return []
        rows = await asyncio.to_thread(_fetch)
        for row in rows:
            if row[0] in params:
                params[row[0]] = float(row[1])
        return params

    # [FIX P0-1] Converted to async — SQLite read dispatched via asyncio.to_thread()
    # so the event loop thread is never blocked waiting for a potentially-locked DB.
    async def _check_risk_halt(self):
        def _read():
            try:
                with sqlite3.connect("trading_data.db", timeout=10) as conn:
                    res = conn.execute("SELECT value FROM system_state WHERE key = 'RISK_HALT'").fetchone()
                    if res and res[0] == "1":
                        return True
            except Exception as e:
                log.error(f"Failed to check RISK_HALT: {e}")
            return False
        return await asyncio.to_thread(_read)

    def _run_background_task(self, coro):
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    async def _set_status(self, msg, veto=None):
        """Broadcasts internal state to the system_state table for the HUD."""
        await self.paper_ex.update_system_state("current_status_sniper", msg)
        if veto is not None:
            await self.paper_ex.update_system_state("last_veto_sniper", veto)

    async def prime_data(self):
        """Fetch historical bars for all assets."""
        for attempt in range(1, 6):
            try:
                await self.exchange.load_markets()
                log.info(f"Sniper Exchange markets loaded (Attempt {attempt})")
                break
            except Exception as e:
                if attempt == 5:
                    log.critical(f"FATAL: Failed to load markets after 5 attempts: {e}")
                    raise
                log.warning(f"Market load failed: {e}. Retrying in 10s... ({attempt}/5)")
                await asyncio.sleep(10)

        for symbol in self.symbols:
            log.info(f"Priming {symbol} with {self.timeframe} historical data...")
            for attempt in range(1, 6):
                try:
                    # Fetch raw CCXT data
                    raw = await self.exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, limit=200)
                    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                    df.set_index("timestamp", inplace=True)
                    self.dfs[symbol] = await asyncio.to_thread(compute_indicators, df)
                    log.info(f"Primed {symbol}. Current Price: ${self.dfs[symbol]['close'].iloc[-1]:.2f}")
                    break
                except Exception as e:
                    if attempt == 5:
                        log.critical(f"FATAL: Failed to prime {symbol} after 5 attempts: {e}")
                        raise
                    log.warning(f"Data fetch failed for {symbol}: {e}. Retrying... ({attempt}/5)")
                    await asyncio.sleep(5)
        
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
            log.info(f"[{self.log_name}] SYNC COMPLETE: {len(self.paper_ex.positions)} positions recovered.")

    async def fetch_btc_macro(self):
        """BTC macro trend is used as a global filter, pulled from centralized system_state."""
        try:
            # 1. Primary: Centralized Macro Sentinel trend from system_state
            def _get_central_macro():
                with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                    row = conn.execute("SELECT value FROM system_state WHERE key = 'btc_macro_trend'").fetchone()
                    if row and row[0]:
                        return row[0].upper()
                return None

            central_trend = await asyncio.to_thread(_get_central_macro)
            if central_trend in ("BULLISH", "BEARISH"):
                self.last_btc_trend = central_trend
                return

            # 2. Fallback to local 15m BTC klines if system_state is uninitialized
            df = self.dfs.get("BTC/USDT")
            if df is not None and not df.empty:
                last = df.iloc[-1]
                ema200 = last.get("ema200")
                if pd.isna(ema200):
                    self.last_btc_trend = "UNKNOWN"
                else:
                    self.last_btc_trend = "BULLISH" if last["close"] > ema200 else "BEARISH"
        except Exception as e:
            log.error(f"Silent failure caught in [fetch_btc_macro]: {e}")

    async def handle_candle_close(self, symbol, kline_data):
        try:
            kline_timestamp = int(kline_data['t'])
            if self.last_candle_t.get(symbol, 0) >= kline_timestamp:
                log.info(f"[GUARD] Rejecting duplicate candle close tick for {symbol} at {kline_timestamp}. Already processed.")
                return
            self.last_candle_t[symbol] = kline_timestamp
            
            log.info(f"[CLOCK] {self.timeframe} Candle Closed for {symbol}. Running V10.0 Decision Matrix...")
            price = float(kline_data['c'])
            if price <= 0: return
            new_row = {"open": float(kline_data['o']), "high": float(kline_data['h']), "low": float(kline_data['l']), "close": price, "volume": float(kline_data['v'])}
            new_timestamp = pd.to_datetime(kline_data['t'], unit='ms', utc=True)
            temp_df = self.dfs[symbol].copy()
            temp_df.loc[new_timestamp] = new_row
            if len(temp_df) > 1000: temp_df = temp_df.tail(1000)
            self.dfs[symbol] = await asyncio.to_thread(compute_indicators, temp_df)
            
            # V14.0 P0: Fix Look-ahead bias (Using last closed candle history)
            # We reference iloc[:-1] logic to ensure only closed/stable rows drive signals.
            closed_df = self.dfs[symbol].iloc[:-1]
            r1 = self.dfs[symbol].iloc[-1]
            
            atr = max(float(r1["atr14"]), 0.0001)
            vol, vol_sma = float(r1["volume"]), float(r1["vol_sma20"])
            ema200, ema50, ema20 = float(r1.get("ema200", price)), float(r1.get("ema50", price)), float(r1.get("ema20", price))

            # V15.2: Update System State with latest metrics for HUD visibility
            metrics = {
                "price": price,
                "ema200": ema200,
                "vol_ratio": vol / vol_sma if vol_sma > 0 else 1.0,
                "atr": atr
            }
            await self.paper_ex.update_system_state(f"{self.log_name}_metrics", json.dumps(metrics))
            await self.paper_ex.save_log("DEBUG", f"[{symbol}] Decision Matrix Running. Price: {price}")
            adx_val, adx_sma = float(r1.get("adx", 0)), float(r1.get("adx_avg", 0))
            rsi, s_high, s_low = float(r1.get("rsi14", 50)), float(r1.get("swing_high", price * 1.01)), float(r1.get("swing_low", price * 0.99))
            vwap = float(r1.get("vwap", price))
            hard_regime = classify_regime_hard(r1)
            trend_ctx = build_trend_context(r1)
            patterns = get_candle_patterns(self.dfs[symbol])
            await self.fetch_btc_macro()

            # Fetch dynamic params
            dyn_params = await self._get_dynamic_parameters()
            dyn_min_confidence = dyn_params["min_nn_confidence"]
            dyn_vol_gate_mult = dyn_params["vol_gate_mult"]

            # V17.5 Dynamic Sovereignty: Interrogate the Policy Brain for dynamic gate thresholds
            try:
                state_vec = await self.rl_analyst.build_state_tensor()
                # NOTE: Fast-gate neural confidence check DISABLED — the model weights have not been
                # sufficiently trained on live data and consistently produce ~36% confidence regardless
                # of market conditions, causing the gate to block 100% of candles. The deterministic
                # entry logic (volume displacement, RSI band, EMA200, 60M macro warden) provides all
                # necessary trade quality filtering. Re-enable once PPO training converges above 65%.
                neural_decision = self.rl_strategist.get_action(state_vec)
                neural_confidence = neural_decision["confidence"]

                dyn_limits = self.rl_strategist.get_dynamic_thresholds(state_vec)
                dyn_rsi = dyn_limits["dynamic_rsi"]
                dyn_vwap = dyn_limits["dynamic_vwap"]
                log.info(f"[SOVEREIGNTY] RL Brain Eval: Confidence={neural_confidence:.2%} | RSI={dyn_rsi:.1f} | VWAP={dyn_vwap:.4f} | [FAST-GATE: DISABLED - training in progress]")
            except Exception as re:
                log.warning(f"Dynamic sovereignty brain retrieval offline: {re}. Reverting to standard V17.0.")
                dyn_rsi, dyn_vwap = 30.0, 0.98
                neural_decision = {"confidence": 0.0}

            action, reason, q_mult = get_deterministic_action(
                hard_regime, trend_ctx, adx_val, adx_sma, price, ema200,
                self.last_btc_trend.lower(), rsi, vol, vol_sma, s_high, s_low, atr, patterns,
                has_pos=(self.active_positions.get(symbol) is not None),
                vol_gate_mult=dyn_vol_gate_mult,
                vwap=vwap,
                dynamic_rsi=dyn_rsi,
                dynamic_vwap=dyn_vwap
            )

            # V11.0 / V36.5: Post-Exit Re-entry Cooldown & Top-Buying Guard
            if "open" in action:
                signal_dir = "long" if "long" in action else "short"
                cd = self.tp_cooldown.get(symbol)
                if cd and datetime.now() < cd["until"] and signal_dir == cd["direction"]:
                    log.info(
                        f"[COOLDOWN] {symbol} {signal_dir.upper()} blocked "
                        f"— Post-exit cooldown active until {cd['until'].strftime('%H:%M:%S UTC')}"
                    )
                    action = "hold"
                    reason = "Post-exit cooldown active"
                    await self._set_status("Blocked by Cooldown Window", veto=reason)
                elif signal_dir == "long" and price > (ema20 * 1.015):
                    # Top-buying protection: price extended >1.5% above 15m EMA20
                    log.info(
                        f"[TOP-GUARD] {symbol} LONG blocked — price (${price:,.4f}) stretched "
                        f">1.5% above EMA20 (${ema20:,.4f})"
                    )
                    action = "hold"
                    reason = "Price over-extended above EMA20"
                    await self._set_status("Blocked by EMA20 Extension Guard", veto=reason)
                elif await self._check_risk_halt():
                    log.warning(f"[GLOBAL RISK GATE] {symbol} trade blocked. Portfolio at max exposure.")
                    action = "hold"
                    reason = "Global Risk Halt Active"
                    await self._set_status("Blocked by Global Risk Halt", veto=reason)
            elif action == "hold" and "TP cooldown active" not in reason and "Risk Halt" not in reason:
                await self._set_status("Scanning for Volume Displacement")

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            tel_packet = {"timestamp": current_time, "reasoning": f"[{action.upper()}] {reason}", "price": price, "ema_ref": ema200, "vol_pct": (vol / vol_sma * 100) if vol_sma > 0 else 100.0, "action": action.upper(), "quality": 0.0}
            await self.paper_ex.save_telemetry(symbol, tel_packet)
            
            # Broadcast metrics for Dashboard Gauges
            metrics = {
                "price": price,
                "ema200": ema200,
                "atr": atr,
                "vol_ratio": vol / vol_sma if vol_sma > 0 else 1.0
            }
            await self.paper_ex.update_system_state("sniper_metrics", json.dumps(metrics))

            # [V36.2-TELEMETRY] Push live Sniper balance on every candle close so the
            # HUD dashboard always reflects current capital (resolves 24-day stale issue).
            await self.paper_ex.update_system_state(
                "balance_sniper", str(round(self.paper_ex.balance, 8))
            )

            if "open" in action and self.active_positions[symbol] is None:
                if self.last_btc_trend == "UNKNOWN":
                    log.warning(f"[WARMUP] {symbol} skip — BTC Trend is UNKNOWN (EMA200 not ready)")
                    await self._set_status("Status: Waiting for BTC EMA200 Warmup")
                    return

                await self._set_status(f"Pinging Groq AI for Audit on {symbol}...")
                async with self.order_lock:
                    signal = "long" if "long" in action else "short"

                    # Macro Direction Lock: Suppress counter-trend trades on BTC itself
                    # (Altcoin breakdown signals have already passed get_deterministic_action criteria)
                    if symbol == "BTC/USDT":
                        if signal == "short" and self.last_btc_trend.upper() in ("BULLISH", "LONG_ONLY"):
                            log.info(f"[MACRO-LOCK] BTC SHORT audit suppressed — Macro trend is {self.last_btc_trend}.")
                            await self._set_status("Scanning for Long Pullbacks", veto="Macro Bullish Lock")
                            return
                        if signal == "long" and self.last_btc_trend.upper() in ("BEARISH", "SHORT_ONLY"):
                            log.info(f"[MACRO-LOCK] BTC LONG audit suppressed — Macro trend is {self.last_btc_trend}.")
                            await self._set_status("Scanning for Short Breakdowns", veto="Macro Bearish Lock")
                            return

                    # DETERMINISTIC EMA STACK VALIDATION
                    is_ema_stack = check_ema_stack(price, ema20, ema50, ema200, signal)
                    
                    # Fetch the real PnL from the state manager
                    current_session_pnl = self.state.get_session_performance()
                    
                    ai_res = await ai_engine.get_llm(engine_name=self.log_name).get_risk_params(
                        regime=hard_regime, atr_pct=float(r1["atr_pct"]), signal=signal, phase=trend_ctx.phase,
                        session_pnl=current_session_pnl, streak=0, btc_context=self.last_btc_trend, 
                        market_context=str(self.dfs[symbol]['close'].tail(10).tolist()), 
                        htf_trend="unknown", current_regime=hard_regime, ema_slope=float(r1.get("ema_slope", 0)), 
                        patterns=patterns, ema_stack=is_ema_stack, timeout_sec=20.0, symbol=symbol
                    )
                    await self.paper_ex.update_system_state("last_ai_thoughts_sniper", ai_res.get("notes", "No notes."))
                    if os.getenv("OBSERVATION_MODE", "0") == "1":
                        log.warning("[OBSERVATION MODE] AI Audit conviction bypassed — OBSERVATION_MODE=1 is set.")
                        quality = 100.0
                    else:
                        quality = float(ai_res.get("quality", 0.0))

                    # [PATCH V32.4] SHADOW-REHABILITATION: Log ghost trade with entry TP/SL
                    try:
                        db_path = self.paper_ex.db_path
                        def _log_ghost():
                            with sqlite3.connect(db_path, timeout=5.0) as conn:
                                # [FIX P0-4] WAL pragma + busy_timeout for contention safety during 8-daemon flush bursts
                                conn.execute("PRAGMA journal_mode=WAL;")
                                conn.execute("PRAGMA busy_timeout=10000;")
                                _cursor = conn.execute(
                                    "SELECT value FROM system_state WHERE key='DISABLED_SIGNAL_Sniper-Strategy'"
                                )
                                _flag_row = _cursor.fetchone()
                                if _flag_row and _flag_row[0] == '1':
                                    _ghost_ts = datetime.now(UTC).isoformat()
                                    _paper_sl, _paper_tp = calculate_exits(price, signal, atr)
                                    conn.execute(
                                        "INSERT INTO ghost_trades "
                                        "(timestamp, symbol, signal, entry_price, paper_sl, paper_tp, signal_type) VALUES (?,?,?,?,?,?,?)",
                                        (_ghost_ts, symbol, signal, price, _paper_sl, _paper_tp, 'Sniper-Strategy')
                                    )
                                    conn.commit()
                                    return _paper_sl, _paper_tp
                            return None

                        ghost_res = await asyncio.to_thread(_log_ghost)
                        if ghost_res:
                            _paper_sl, _paper_tp = ghost_res
                            log.info(
                                f"[GHOST-TRADE] Shadow signal logged: "
                                f"{signal.upper()} {symbol} @ ${price:.2f} | Stored SL: {_paper_sl:.2f}, TP: {_paper_tp:.2f}"
                            )
                    except Exception as _ge:
                        log.error(f"[GHOST-TRADE] Critical write error: {_ge}")
                    # --- END PATCH V32.4 ---

                    # PATCH: Define the dynamic minimum conviction threshold (Default 50.0 from config)
                    dyn_min_conv = getattr(config, "MIN_AI_CONVICTION_SCORE", 50.0)
                    if hasattr(self, 'min_confidence') and self.min_confidence:
                        dyn_min_conv = self.min_confidence
                    if dyn_min_conv <= 1.0:
                        dyn_min_conv *= 100.0
                    if quality < dyn_min_conv:
                        veto_msg = f"AI Conviction too low: {quality:.1f} (Needs {dyn_min_conv:.1f})"
                        tel_packet["reasoning"] = f"[HOLD (AI)] {veto_msg}"
                        tel_packet["quality"] = quality
                        await self.paper_ex.save_telemetry(symbol, tel_packet)
                        await self._set_status("Cooling Down from Veto", veto=veto_msg)
                    else:
                        ticker = await self.exchange.fetch_ticker(symbol)
                        # V14.4 Fallback: If ticker fails, use candle close price
                        if ticker is None or ticker.get('ask') is None or ticker.get('bid') is None:
                            log.warning(f"[{symbol}] Ticker failed. Falling back to candle close (${price:.2f}).")
                            exec_p = price
                        else:
                            exec_p = ticker['ask'] if signal == "long" else ticker['bid']
                        # V16.4: Volatility Warden - Adaptive Risk Sizing (Chop Protection)
                        vol_mult = 1.0
                        try:
                            if os.path.exists("volatility_state.json"):
                                with open("volatility_state.json", "r") as f:
                                    v_data = json.load(f)
                                    vol_mult = v_data.get("recommended_risk_multiplier", 1.0)
                                    if v_data.get("market_vibe") != "TRENDING":
                                        log.info(f"[WARDEN] Adaptive Risk Active: {v_data.get('market_vibe')} market. Applying {vol_mult}x multiplier.")
                        except: pass

                        qty = calculate_risk_qty(self.paper_ex, symbol, exec_p, atr, ai_res.get("size_mult", 1.0) * vol_mult, atr_mult=1.5)
                        
                        # V16.0: Sentiment Oracle Kill-Switch
                        try:
                            if os.path.exists("sentiment_state.json"):
                                with open("sentiment_state.json", "r") as f:
                                    s_data = json.load(f)
                                    # Base symbol (e.g., BTC from BTC/USDT)
                                    base = symbol.split('/')[0]
                                    s_status = s_data.get(base, {}).get("status", "SAFE")
                                    if s_status == "BLOCK":
                                        reason = s_data.get(base, {}).get("reason", "Unknown")
                                        log.warning(f"[SENTIMENT] TRADE BLOCKED BY ORACLE: {reason}")
                                        await self.paper_ex.save_log("SENTIMENT", f"BLOCKED {symbol}: {reason}")
                                        return
                        except Exception as se:
                            log.error(f"Sentiment check failed: {se}")

                        if qty > 0:
                            try:
                                res = await self.macro_breaker.fetch()
                                if not res.get("safe_to_trade", True):
                                    log.warning(f"[WARDEN] MACRO HALT: Trade aborted on {symbol}. Reason: {res.get('reason')}")
                                    await self.paper_ex.save_log("WARDEN", f"MACRO HALT on {symbol}: {res.get('reason')}")
                                    await self._set_status(f"Status: Halted by Macro Circuit Breaker ({symbol})")
                                    return
                            except Exception as me:
                                log.error(f"Macro Circuit Breaker evaluation failed: {me}")

                            # V16.0: Conviction Filter (The Boardroom)
                            try:
                                # V18.0 Narrative Awareness: Query the latest linguistics vector
                                s_score, s_reason = 0.0, "Market sentiment neutral."
                                try:
                                    with sqlite3.connect("trading_ledger.db", timeout=5) as s_conn:
                                        s_res = s_conn.execute("SELECT score, rationale FROM sentiment_logs ORDER BY timestamp DESC LIMIT 1").fetchone()
                                        if s_res: s_score, s_reason = float(s_res[0]), s_res[1]
                                except: pass

                                from analyst_agent import detect_volume_divergence
                                
                                hist_p = self.dfs[symbol]['close'].tail(5).tolist()
                                hist_v = self.dfs[symbol]['volume'].tail(5).tolist()
                                div_data = detect_volume_divergence(hist_p, hist_v)

                                boardroom_res = await self.boardroom.get_conviction(
                                    symbol=symbol, 
                                    side=signal, 
                                    context_data={
                                        "regime": hard_regime, 
                                        "phase": trend_ctx.phase, 
                                        "quality": quality, 
                                        "atr": atr,
                                        "rl_dynamic_rsi": dyn_rsi,
                                        "rl_dynamic_vwap": dyn_vwap,
                                        "narrative_sentiment_score": round(s_score, 2),
                                        "narrative_sentiment_rationale": s_reason,
                                        "recent_prices": [round(x, 2) for x in hist_p],
                                        "recent_volumes": [round(x, 2) for x in hist_v],
                                        "vol_divergence_math_score": div_data
                                    }
                                )
                                if os.getenv("OBSERVATION_MODE", "0") == "1":
                                    log.warning("[OBSERVATION MODE] Boardroom conviction bypassed — OBSERVATION_MODE=1 is set.")
                                    execute = True
                                    score = 1.0
                                else:
                                    execute = boardroom_res.get("execute", False)
                                    score = boardroom_res.get("conviction_score", 0.0)
                                rationale = boardroom_res.get("final_rationale", "No details")
                                if not execute or score < 0.75:
                                    log.warning(f"[BOARDROOM] Strike Aborted: Low Conviction ({score}). Rationale: {rationale}")
                                    await self.paper_ex.save_log("BOARDROOM", f"Strike Aborted ({score}): {rationale}")
                                    await self._set_status(f"Status: Vetoed by Boardroom ({symbol})")
                                    return
                                log.info(f"[BOARDROOM] APPROVED by Portfolio Manager (Score: {score}). Rationale: {rationale}")
                            except Exception as be:
                                log.error(f"Boardroom execution gate crashed: {be}")

                            tel_packet["reasoning"] = f"[EXECUTED {signal.upper()}] Strategy & AI Handshake verified."
                            tel_packet["quality"] = quality
                            await self.paper_ex.save_telemetry(symbol, tel_packet)
                            
                            # V30.0 Standardized Execution Routing: Almgren-Chriss Stealth Slicing
                            exec_success = await self.executor.execute_trade(
                                strategy_decision={
                                    "action_name": signal.upper(),
                                    "confidence": quality / 100.0 if quality > 1.0 else quality
                                },
                                current_price=exec_p,
                                market_state=state_vec,
                                symbol=symbol
                            )
                            
                            if exec_success:
                                sl, tp, hard_sl = calculate_exits(exec_p, signal, atr, regime=hard_regime)
                                pos = await self.paper_ex.open_position(symbol, signal, qty, exec_p, sl=sl, tp=tp, hard_sl=hard_sl, metadata={"regime": hard_regime, "phase": trend_ctx.phase, "quality": quality, "notes": ai_res.get("notes", ""), "ai_rationale": ai_res.get("notes", "")})
                                if not pos:
                                    log.warning(f"[EXECUTION-ABORT] open_position rejected trade for {symbol}.")
                                    await self._set_status(f"Status: Trade Rejected by Exchange ({symbol})")
                                    return
                                log.info(f"[LIVE ALERT] Executing {symbol} | Side: {signal.upper()} | Price: ${exec_p:.4f} | SL: {sl:.4f} | TP: {tp:.4f}")
                                self.active_positions[symbol] = pos
                                
                                trade_ctx = {
                                    "symbol": symbol,
                                    "side": signal,
                                    "price": exec_p,
                                    "qty": qty,
                                    "stop_loss": sl,
                                    "take_profit": tp,
                                    "regime": hard_regime,
                                    "conviction": f"{round(float(score)*10)}/10" if 'score' in locals() else f"{quality:.1f}%",
                                    "ai_reasoning": rationale if 'rationale' in locals() else ai_res.get("notes", ""),
                                    "quality": quality,
                                    "engine": f"Live Engine ({self.timeframe})",
                                    "leverage": pos.get("leverage", 5.0)
                                }
                                # Dispatch Non-blocking Alert with complete trade context
                                self._run_background_task(send_trade_log_async(symbol, signal, exec_p, status="OPENED", trade_context=trade_ctx))
                                await self._set_status(f"Monitoring Active {signal.upper()} Position")
                            else:
                                log.warning(f"[EXECUTION-ABORT] Stealth router declined trade for {symbol}.")
                                await self._set_status(f"Status: VETO by Stealth Router ({symbol})")
                                return
            log.debug(f"[CLOCK] {self.timeframe} cycle for {symbol} complete.")
        except Exception as e:
            log.error(f"Silent failure caught in [handle_candle_close]: {e}\n{traceback.format_exc()}")

    async def monitor_position(self, symbol, current_price):
        try:
            # V15.5: Added safety check to prevent KeyError
            if symbol not in self.active_positions: return
            pos = self.active_positions.get(symbol)
            if not pos or pos.get("is_closing"): return

            # Sync Step: Drop memory if position was closed externally on exchange / reconciled
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
            df = self.dfs.get(symbol)
            if df is None or df.empty or "atr14" not in df.columns:
                return
            raw_atr = df["atr14"].iloc[-1]
            atr = max(float(raw_atr), 0.0001) if pd.notna(raw_atr) else 0.0
            
            # V16.1: Whale Watcher - Adaptive Trailing Stop (Flash-Crash Protection)
            try:
                if os.path.exists("whale_state.json"):
                    with open("whale_state.json", "r") as f:
                        w_data = json.load(f)
                        if w_data.get("whale_status") == "DANGER":
                            # Move SL to 2% below current market price (Trailing Stop)
                            danger_sl = price * 0.98 if side == "long" else price * 1.02
                            # Only tighten (don't loosen)
                            if (side == "long" and danger_sl > pos["stop_loss"]) or (side == "short" and danger_sl < pos["stop_loss"]):
                                pos["stop_loss"] = danger_sl
                                log.warning(f"[WHALE ALERT] DANGER DETECTED. Tightening SL to 2% Trailing: {danger_sl:.2f}")
                                await self.paper_ex.save_log("WHALE", f"DANGER: Tightened {symbol} SL to {danger_sl:.2f}")
                                await self.paper_ex.sync_position_updates(symbol, pos)
            except Exception as we:
                log.error(f"Whale check failed: {we}")

            # Macro News Active De-risking Guard: If high-impact news (CPI/NFP) is within 15 mins, tighten profitable SL to Breakeven
            try:
                # [FIX L2] Use __init__ singleton instead of instantiating a new gate every tick
                if self.cal_gate is not None:
                    is_news, news_reason, diff_mins = self.cal_gate.check_event_window()
                    if is_news and diff_mins < 0 and abs(diff_mins) <= 15.0:
                        entry_p = float(pos.get("entry_price", 0.0))
                        if (side == "long" and price > entry_p and entry_p > pos["stop_loss"]) or \
                           (side == "short" and price < entry_p and entry_p < pos["stop_loss"]):
                            pos["stop_loss"] = entry_p
                            log.warning(f"[MACRO DE-RISK] HIGH IMPACT NEWS IMMINENT ({news_reason}). Moving {symbol} SL to Breakeven (${entry_p:,.4f})!")
                            await self.paper_ex.save_log("MACRO_NEWS", f"HIGH IMPACT NEWS IMMINENT ({news_reason}): Moved {symbol} SL to Breakeven.")
                            await self.paper_ex.sync_position_updates(symbol, pos)
            except Exception as ne:
                log.debug(f"Macro news de-risk check note: {ne}")


            p_dist = (price - pos["entry_price"] if side == "long" else pos["entry_price"] - price)

            # V36.5 Early Scratch Defense: When price moves +1.0x ATR in profit, ratchet SL to Breakeven (+ 0.1% fee cushion)
            be_mult = float(getattr(config, "EARLY_BE_ATR_MULT", 1.0))
            if p_dist >= be_mult * atr and atr > 0:
                fee_buf_pct = float(getattr(config, "BREAKEVEN_FEE_BUFFER_PCT", 0.001))
                fee_buffer = pos["entry_price"] * fee_buf_pct
                be_sl = (pos["entry_price"] + fee_buffer) if side == "long" else (pos["entry_price"] - fee_buffer)
                if (side == "long" and be_sl > pos["stop_loss"]) or (side == "short" and be_sl < pos["stop_loss"]):
                    pos["stop_loss"] = be_sl
                    log.info(f"[EARLY-BE] 🛡️ {symbol} reached +{be_mult:.1f}x ATR gain (${price:,.4f}). Ratcheting SL to Breakeven (${be_sl:,.4f}) to guarantee zero downside!")
                    await self.paper_ex.sync_position_updates(symbol, pos)

            if p_dist >= 3.0 * atr and atr > 0 and not pos.get("partial_tp_hit"):
                from live_paper_exchange import get_contract_specs
                specs = get_contract_specs(symbol)
                min_step = specs["min_qty"]
                # Only take partial TP if BOTH the closed portion and remaining portion are >= min_qty
                if (pos["qty"] / 2.0) >= min_step and (pos["qty"] - (pos["qty"] / 2.0)) >= min_step:
                    partial_qty = round(pos["qty"] / 2.0, specs["decimals"]) if specs["decimals"] > 0 else float(int(pos["qty"] // 2))
                    p_pnl = await self.paper_ex.close_position(symbol, price, t_type="partial_tp", partial_qty=partial_qty)
                    if p_pnl is None:
                        log.warning(f"[PARTIAL TP] {symbol} was already 0 on exchange. Dropping memory state.")
                        self.active_positions[symbol] = None
                        return
                    log.info(f"[PARTIAL TP] {symbol} secured 50% at 3.0x ATR momentum expansion. PnL: ${p_pnl:+.4f}")
                    # Dispatch Partial TP Email Alert
                    partial_actual_pnl = p_pnl
                    partial_ctx = {
                        "qty": partial_qty,
                        "pnl": partial_actual_pnl,
                        "engine": f"Sniper ({self.timeframe})",
                        "entry_price": pos["entry_price"],
                        "peak_price": pos.get("peak_price", price),
                        "entry_time": pos.get("entry_time"),
                        "close_trigger": "PARTIAL TAKE-PROFIT (50%)",
                        "close_reason": f"Secured 50% partial profit at 3.0x ATR momentum expansion (${price:,.4f}) to lock in gains and de-risk position while letting runner trail."
                    }
                    self._run_background_task(send_trade_log_async(symbol, side, price, pnl=partial_actual_pnl, status="PARTIAL_TP", trade_context=partial_ctx))
                else:
                    log.info(f"[PARTIAL TP] {symbol} quantity ({pos['qty']}) is at exchange minimum ({min_step}). Skipping 50% split to preserve position on Bitget; trailing full position.")
                pos["partial_tp_hit"] = True
                
            if p_dist >= 2.0 * atr and atr > 0:
                trail_mult = 1.0 if pos.get("partial_tp_hit") else 2.0
                trail_sl = (pos["peak_price"] - (trail_mult * atr) if side == "long" else pos["peak_price"] + (trail_mult * atr))
                if (side == "long" and trail_sl > pos["stop_loss"]) or (side == "short" and trail_sl < pos["stop_loss"]):
                    pos["stop_loss"] = trail_sl
                    await self.paper_ex.sync_position_updates(symbol, pos)

            # V11.0: Detect SL vs TP hit separately so cooldown is only set on TP
            is_sl = (side == "long" and price <= pos["stop_loss"]) or (side == "short" and price >= pos["stop_loss"])
            is_tp = (pos.get("take_profit", 0) > 0) and (
                (side == "long" and price >= pos["take_profit"]) or
                (side == "short" and price <= pos["take_profit"])
            )

            # ── Flash Crash Filter (Soft SL confirmation window) ──────────────────────
            # When a soft SL breach is detected, wait FLASH_CRASH_CONFIRMATION_SECS before
            # actually closing. If price recovers above the SL within that window the breach
            # is treated as a flash crash / exchange flush and the position is preserved.
            # TPs bypass the filter entirely — they execute immediately.
            if is_sl and not is_tp:
                import time as _time
                confirmation_secs = int(getattr(config, "FLASH_CRASH_CONFIRMATION_SECS", 60))
                breach_ts = pos.get("sl_breach_time")  # float unix timestamp or None

                if breach_ts is None:
                    # First tick below soft SL — start the countdown, don't close yet
                    pos["sl_breach_time"] = _time.time()
                    await self.paper_ex.sync_position_updates(symbol, pos)
                    log.warning(
                        f"[FLASH FILTER] ⚡ {symbol} breached Soft SL at ${price:,.4f} "
                        f"(SL=${pos['stop_loss']:,.4f}). Starting {confirmation_secs}s confirmation window. "
                        f"Position preserved until confirmed."
                    )
                    is_sl = False  # Do NOT close yet
                elif (_time.time() - breach_ts) < confirmation_secs:
                    # Still within the confirmation window — let price breathe
                    elapsed = int(_time.time() - breach_ts)
                    log.info(
                        f"[FLASH FILTER] ⏳ {symbol} still below Soft SL "
                        f"(${price:,.4f} < ${pos['stop_loss']:,.4f}) — "
                        f"{elapsed}s / {confirmation_secs}s elapsed. Holding."
                    )
                    is_sl = False  # Still waiting
                else:
                    # Price has remained below soft SL for confirmation_secs → genuine move, close now
                    elapsed = int(_time.time() - breach_ts)
                    log.info(
                        f"[FLASH FILTER] ✅ {symbol} Soft SL confirmed after {elapsed}s. "
                        f"Genuine move — closing position at ${price:,.4f}."
                    )
                    pos["sl_breach_time"] = None  # Reset so we don't carry stale state
                    await self.paper_ex.sync_position_updates(symbol, pos)
                    # is_sl remains True — fall through to close block below
            elif not is_sl and pos.get("sl_breach_time") is not None:
                # Price has recovered above the Soft SL — this was a flash crash!
                import time as _time
                elapsed = int(_time.time() - pos["sl_breach_time"])
                self._run_background_task(send_alert_async(
                    f"⚡ [{symbol}] Flash Crash Filter Saved Position",
                    f"Price breached Soft SL ${pos['stop_loss']:,.4f} but recovered within {elapsed}s.\nPosition preserved. No loss taken."
                ))
                log.info(
                    f"[FLASH FILTER] 🛡️ {symbol} recovered above Soft SL after {elapsed}s — "
                    f"FLASH CRASH confirmed. Position preserved. Resetting breach timer."
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

                # V15.6: Flag as closing
                pnl_closed = await self.paper_ex.close_position(symbol, price, t_type=close_type)
                
                # Only dispatch email alert and record trade if the close actually executed
                if pnl_closed is not None:
                    actual_closed_pnl = pnl_closed if isinstance(pnl_closed, (int, float)) else (p_dist * pos["qty"])
                    # V16.3: Dispatch Non-blocking Email Alert with actual dollar PnL and full reason context
                    close_ctx = {
                        "qty": pos["qty"],
                        "pnl": actual_closed_pnl,
                        "engine": f"Sniper ({self.timeframe})",
                        "entry_price": pos["entry_price"],
                        "peak_price": pos.get("peak_price", price),
                        "entry_time": pos.get("entry_time"),
                        "stop_loss": pos.get("stop_loss"),
                        "take_profit": pos.get("take_profit"),
                        "close_trigger": close_trig,
                        "close_reason": close_rsn,
                        "ai_reasoning": pos.get("metadata", {}).get("ai_rationale") or pos.get("metadata", {}).get("notes", "Quantitative strategy momentum exit.")
                    }
                    self._run_background_task(send_trade_log_async(symbol, side, price, pnl=actual_closed_pnl, status="CLOSED", trade_context=close_ctx))
                    
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

                # [V36.2-TELEMETRY] Flush post-trade balance to DB immediately after
                # every close event so the HUD capital delta is never stale post-trade.
                self._run_background_task(
                    self.paper_ex.update_system_state(
                        "balance_sniper", str(round(self.paper_ex.balance, 8))
                    )
                )
                log.info(
                    f"[BALANCE-SYNC] balance_sniper flushed to system_state: "
                    f"${self.paper_ex.balance:,.4f} USDT (post-{close_type})"
                )

                # Post-Exit Cooldown: Trigger on BOTH full TP and profitable trailing stop exits
                if is_tp or was_trailed:
                    cd_mins = int(getattr(config, "POST_TP_COOLDOWN_MINUTES", 120))
                    self.tp_cooldown[symbol] = {
                        "direction": side,
                        "until": datetime.now() + timedelta(minutes=cd_mins)
                    }
                    exit_label = "TP TARGET" if is_tp else "PROFITABLE TRAIL"
                    log.info(f"[COOLDOWN] {symbol} {exit_label} hit ({side.upper()}). Blocking same-direction re-entry for {cd_mins}m.")
        except Exception as e:
            log.error(f"Silent failure caught in [monitor_position]: {e}")
        # [V32.3] Ghost Position Tracker — resolve open ghost entries against live price
        await self._resolve_ghost_positions(symbol, current_price)


    async def _resolve_ghost_positions(self, symbol: str, current_price: float):
        """
        [V32.4] Ghost Position Tracker.
        Evaluates all open ghost_trades for `symbol` against the live price using
        stored exit prices, eliminating look-ahead/look-back ATR bias.
        """
        try:
            db_path = self.paper_ex.db_path
            def _fetch_ghosts():
                with sqlite3.connect(db_path, timeout=5.0) as conn:
                    return conn.execute(
                        "SELECT id, signal, entry_price, paper_sl, paper_tp FROM ghost_trades "
                        "WHERE symbol = ? AND outcome IS NULL AND pnl IS NULL",
                        (symbol,)
                    ).fetchall()
            
            rows = await asyncio.to_thread(_fetch_ghosts)

            if not rows:
                return

            updates = []
            for ghost_id, signal, entry_price, paper_sl, paper_tp in rows:
                # [V32.4] Backward compatibility: Fallback to dynamic ATR if stored levels are missing
                if paper_sl is None or paper_tp is None:
                    df = self.dfs.get(symbol)
                    if df is not None and not df.empty and "atr14" in df.columns:
                        raw_atr = df["atr14"].iloc[-1]
                        if pd.notna(raw_atr) and float(raw_atr) > 0:
                            paper_sl, paper_tp = calculate_exits(entry_price, signal, float(raw_atr))
                    
                    if paper_sl is None or paper_tp is None:
                        continue  # Skip resolving this entry if ATR is unavailable

                is_tp = (signal == "long" and current_price >= paper_tp) or \
                        (signal == "short" and current_price <= paper_tp)
                is_sl = (signal == "long" and current_price <= paper_sl) or \
                        (signal == "short" and current_price >= paper_sl)

                if is_tp or is_sl:
                    outcome = "tp_hit" if is_tp else "sl_hit"
                    # Notional unit PnL — no leverage, pure signal-quality metric
                    ghost_pnl = (current_price - entry_price) if signal == "long" \
                                else (entry_price - current_price)

                    updates.append((current_price, ghost_pnl, outcome, ghost_id, symbol, signal, entry_price, paper_sl, paper_tp))

            if updates:
                def _update_ghosts(upd_list):
                    with sqlite3.connect(db_path, timeout=5.0) as conn:
                        for u in upd_list:
                            conn.execute(
                                "UPDATE ghost_trades "
                                "SET exit_price=?, pnl=?, outcome=? WHERE id=?",
                                (u[0], u[1], u[2], u[3])
                            )
                        conn.commit()
                
                await asyncio.to_thread(_update_ghosts, updates)

                for u in updates:
                    sign = "+" if u[1] >= 0 else ""
                    log.info(
                        f"[GHOST-CLOSE] {u[4]} ghost #{u[3]} "
                        f"{u[5].upper()} resolved: {u[2]} | "
                        f"Entry=${u[6]:.2f} Exit=${u[0]:.2f} | "
                        f"PnL={sign}{u[1]:.2f} (Stored SL={u[7]:.2f}, TP={u[8]:.2f})"
                    )

        except Exception as e:
            log.debug(f"[GHOST-RESOLVER] Non-critical error: {e}")

    async def websocket_loop(self):
        streams = "/".join([f"{bs}@kline_{self.timeframe}" for bs in self.binance_symbols])
        url = f"wss://fstream.binance.com/market/stream?streams={streams}"
        log.info(f"Connecting to Sniper Stream ({len(self.symbols)} symbols): {url}")
        resolver = aiohttp.ThreadedResolver()
        # SECURE SSL AUTO-HANDLED BY WSS:// URI
        connector = aiohttp.TCPConnector(resolver=resolver, family=socket.AF_INET)
        async with aiohttp.ClientSession(connector=connector, connector_owner=True) as session:
            try:
                # SSL removed as per 2026 URI best practices
                async with session.ws_connect(url, timeout=20.0, heartbeat=30.0, receive_timeout=300.0) as ws:
                    log.info("[NETWORK] Sniper Handshake Successful.")
                    await self._set_status("Status: ACTIVE (Wiping Stale Telemetry)", veto="ACTIVE")
                    while True:
                        if os.path.exists("STOP_TRADING"):
                            log.critical("!!! STOP_TRADING FILE DETECTED !!!")
                            for s in self.symbols:
                                if self.active_positions[s]:
                                    log.warning(f"Kill Switch: Closing {s}")
                                    await self.paper_ex.close_position(s, float(self.dfs[s]['close'].iloc[-1]), t_type="kill_switch")
                            log.critical("Sniper Engine Halted.")
                            break

                        try:
                            msg = await asyncio.wait_for(ws.receive(), timeout=300.0)
                            if msg.type == aiohttp.WSMsgType.CLOSED: break
                            if msg.type != aiohttp.WSMsgType.TEXT: continue
                            payload = json.loads(msg.data)
                            data = payload.get("data", payload)
                            if 'e' not in data or data['e'] != 'kline': continue
                            k = data.get('k', {})
                            symbol = self.symbol_map.get(data.get('s', '').lower())
                            if not symbol: continue
                            await self.monitor_position(symbol, float(k['c']))
                            if k.get('x', False): self._run_background_task(self.handle_candle_close(symbol, k))
                            
                            # V14.3: High-frequency Health Pulse (Throttled to 30s to prevent handle leaks)
                            import time
                            now_time = time.time()
                            if now_time - self._last_heartbeat_time >= 30.0:
                                await self.paper_ex.update_system_state(f"heartbeat_{self.log_name}", datetime.now(UTC).isoformat())
                                self._last_heartbeat_time = now_time
                        except asyncio.TimeoutError:
                            log.warning("[NETWORK] Sniper Watchdog: Stale Connection.")
                            await send_alert_async(
                                "CRITICAL WARNING: Sniper Stale Connection",
                                f"Sniper Watchdog detected a stale WebSocket connection for {self.log_name}. Attempting reconnection sequence."
                            )
                            break
                        except Exception as e:
                            log.error(f"Monitor Crash for {symbol}: {e}\n{traceback.format_exc()}")
                            continue
            except Exception as e:
                log.error(f"Sniper Stream Failed: {e}")

    async def run(self):
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
                    log.error(f"WebSocket connection dropped: {e}. Reconnecting in {backoff} seconds...")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60) # Exponential backoff
                await asyncio.sleep(1)
        finally:
            if hasattr(self, "exchange") and self.exchange:
                await self.exchange.close()
                log.info("Sniper Exchange session closed.")
            if hasattr(self, "paper_ex") and self.paper_ex:
                try:
                    self.paper_ex.close()
                    log.info("Sniper paper exchange session closed.")
                except Exception:
                    pass
            if '_session' in locals() and _session:
                await _session.close()
                log.info("Sniper aiohttp ClientSession closed.")

if __name__ == "__main__":
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
        await LiveTrader().run()
        
    asyncio.run(main())

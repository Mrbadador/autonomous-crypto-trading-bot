import sqlite3
import aiohttp
import asyncio
import pandas as pd
import numpy as np
import logging
import os
import dns_fix  # noqa: F401 — patches ccxt.Exchange.open() with ThreadedResolver
import ccxt.async_support as ccxt
from dotenv import load_dotenv
from datetime import datetime, UTC, timezone
from core.entropy_engine import compute_signal_readiness
from core.liquidity_hunter import LiquidityHunter
from indicators import compute_indicators
import sys
import json
import config

# ENV CUSTODIAN: Force-load .env from the project root before any ccxt/aiohttp clients are created.
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] LIVE_ORACLE: %(message)s")
log = logging.getLogger("live_oracle")

# ENV INTEGRITY CHECK
_api_key = os.getenv("BITGET_API_KEY", "")
if _api_key:
    log.info(f"[BOOT] ENV OK — BITGET_API_KEY loaded (last 4: ...{_api_key[-4:]}).")
else:
    log.warning("[BOOT] BITGET_API_KEY is empty. Public endpoints only — authenticated calls will fail.")

class LiveOracle:
    """
    Agent 1: The Oracle (V30.1 Alpha Edition).
    Pulls live data via async ccxt, calculates entropy and fractal dimensions,
    and commits high-fidelity quantitative features to oracle_state.
    """
    def __init__(self, symbol="BTC/USDT", db_path="trading_ledger.db"):
        self.symbol = symbol
        self.db_path = db_path
        self.fear_greed_index = 50.0
        self.hunter = LiquidityHunter()
        self._init_db()
        
        # Persistent connections for connection pooling & caching
        self.session = None
        self.exchange = None
        self.last_exchange_info_time = None
        
        # V30.1: Background DB Worker Thread to prevent event-loop starvation
        import queue
        import threading
        self.db_queue = queue.Queue()
        self.db_worker_thread = threading.Thread(target=self._db_worker, daemon=True)
        self.db_worker_thread.start()
        log.info("[BOOT] Background DB Persistence Worker launched.")
        self._prune_counter = 0

        # [V36.2-DEDUP] Tracks the last 5-minute bucket epoch written to oracle_state.
        # Prevents duplicate INSERTs when the main_update_loop and any sub-daemon
        # fire concurrently within the same 5-minute candle window.
        self._last_oracle_write_ts: str | None = None

        # [TASK 01: Database Sentinel Bootstrapping]
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp FROM oracle_state ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            conn.close()
            if row and row[0]:
                ts_str = row[0]
                if ts_str.endswith('Z'):
                    ts_str = ts_str[:-1] + '+00:00'
                dt = datetime.fromisoformat(ts_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                self._last_oracle_write_ts = dt.strftime("%Y-%m-%dT%H:") + str((dt.minute // 5) * 5).zfill(2)
                log.info(f"[BOOT] Loaded _last_oracle_write_ts sentinel from database: '{self._last_oracle_write_ts}'")
            else:
                log.info("[BOOT] No previous oracle_state records found in DB to bootstrap sentinel.")
        except Exception as db_err:
            log.warning(f"[BOOT] Database sentinel bootstrap failed: {db_err}")

    async def init_session(self):
        """Initializes the persistent aiohttp session and ccxt exchange client."""
        try:
            if self.session is None or self.session.closed:
                # DNS FIX: aiodns/c-ares cannot contact DNS in this environment.
                # ThreadedResolver delegates to the OS socket layer (getaddrinfo)
                # which resolves correctly, matching what PowerShell/ping can see.
                import socket as _socket
                connector = aiohttp.TCPConnector(
                    resolver=aiohttp.ThreadedResolver(),
                    family=_socket.AF_INET,
                    use_dns_cache=True,
                    ttl_dns_cache=300,
                )
                self.session = aiohttp.ClientSession(connector=connector)
            if self.exchange is None:
                ex_id = getattr(config, "EXCHANGE_ID", "bitget").lower()
                ex_class = getattr(ccxt, ex_id, None) or ccxt.bitget
                self.exchange = ex_class({
                    "enableRateLimit": True,
                    "options": {
                        "defaultType": "swap" if ex_id == "bitget" else "future",
                        "adjustForTimeDifference": True,
                    },
                    "timeout": 30000,
                    "aiohttp_session": self.session
                })
        except Exception as e:
            log.error(f"[ORACLE] Failed to initialize connection session: {e}")
            if self.exchange is not None:
                try:
                    await self.exchange.close()
                except Exception:
                    pass
                self.exchange = None
            if self.session is not None and not self.session.closed:
                try:
                    await self.session.close()
                except Exception:
                    pass
                self.session = None
            raise e


    def _db_worker(self):
        """
        Daemon thread consuming SQL tasks to ensure the main event loop 
        is never blocked by disk I/O or SQLite lock contention.
        Supports 2-tuples and 3-tuples (query, params, optional_db_path) 
        for dynamic database routing.
        """
        import sqlite3
        conns = {}
        while True:
            task = self.db_queue.get()
            if task is None: 
                for c in conns.values():
                    c.close()
                break
            
            db_path = self.db_path
            if len(task) == 3:
                query, params, custom_db = task
                if custom_db:
                    db_path = custom_db
            else:
                query, params = task
                
            try:
                if db_path not in conns:
                    conns[db_path] = sqlite3.connect(db_path, timeout=60.0)
                    conns[db_path].execute("PRAGMA journal_mode=WAL")
                    conns[db_path].execute("PRAGMA busy_timeout=15000")
                conn = conns[db_path]
                
                if query.strip().upper().startswith("PRAGMA"):
                    conn.execute(query)
                else:
                    with conn:
                        if params is not None and len(params) > 0:
                            conn.execute(query, params)
                        else:
                            conn.execute(query)
            except sqlite3.OperationalError as e:
                # P1 FIX: Evict stale/corrupted connection handle so the next
                # task always gets a fresh, validated connection object.
                log.error(f"[DB WORKER] OperationalError on {db_path}: {e}. Evicting connection from cache.", exc_info=True)
                if db_path in conns:
                    try:
                        conns[db_path].close()
                    except Exception:
                        pass
                    del conns[db_path]
            except Exception as e:
                # Surfacing database errors so lock timeouts or transaction fails are visible
                log.error(f"[DB WORKER] Critical Write Failure on {db_path}: {e}", exc_info=True)
            finally:
                self.db_queue.task_done()

    def _init_db(self):
        """Ensures the oracle_state and system_state tables exist with correct schemas."""
        try:
            # V30.4: Initialize system_state table in trading_data.db for shared macro state
            conn_data = sqlite3.connect("trading_data.db", timeout=10.0)
            conn_data.execute("PRAGMA journal_mode=WAL;")
            conn_data.execute("PRAGMA busy_timeout=15000;")
            with conn_data:
                conn_data.execute("""
                    CREATE TABLE IF NOT EXISTS system_state (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
            conn_data.close()
            
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=15000;")
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS oracle_state (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        symbol TEXT,
                        volatility REAL,
                        trend_strength REAL,
                        vwap REAL,
                        rsi REAL,
                        apen REAL,
                        hurst REAL,
                        signal_readiness REAL
                    )
                """)
                # Legacy table for backward compatibility with V30.0 Analysts
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS market_context (
                        id INTEGER PRIMARY KEY DEFAULT 1,
                        timestamp TEXT,
                        fear_greed_index REAL,
                        volatility_state REAL,
                        trend_strength REAL,
                        vwap REAL,
                        rsi_14 REAL,
                        signal_readiness REAL,
                        whale_threat REAL
                    )
                """)
                # V30.1: Microstructure Table for LiquidityHunter
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS oracle_microstructure (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        symbol TEXT,
                        liquidity_imbalance REAL,
                        vacuum_detected INTEGER,
                        bid_wall_price REAL,
                        bid_wall_size REAL,
                        ask_wall_price REAL,
                        ask_wall_size REAL,
                        signal_multiplier REAL
                    )
                """)
                # Phase 16.9 Schema Patch: Create missing tables consumed by AnalystAgent
                # sentiment_logs: stores aggregated sentiment scores for the policy tensor
                # Schema matches sentiment_warden.py exactly for cross-daemon compatibility.
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sentiment_logs (
                        timestamp TEXT,
                        score REAL,
                        rationale TEXT
                    )
                """)
                # macro_logs: stores macro-economic health indicators (SPY proxy, DXY momentum)
                # Schema matches macro_warden.py exactly for cross-daemon compatibility.
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS macro_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT DEFAULT 'BTC/USDT',
                        spy_health REAL NOT NULL DEFAULT 0.0,
                        dxy_momentum REAL NOT NULL DEFAULT 0.0,
                        source TEXT DEFAULT 'macro_warden'
                    )
                """)
            conn.close()
            log.info(f"Database successfully initialized: {self.db_path}")
        except Exception as e:
            log.critical(f"Failed to initialize database: {e}")
            raise

    async def fetch_fear_greed(self):
        """Pulls current crypto market sentiment async."""
        url = "https://api.alternative.me/fng/"
        try:
            await self.init_session()
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    self.fear_greed_index = float(data["data"][0]["value"])
                    return self.fear_greed_index
        except Exception as e:
            log.warning(f"Failed to fetch live Fear & Greed: {e}. Falling back to cached value.")
        return self.fear_greed_index

    async def start_depth_stream(self):
        """V30.1: Real-time Order Book Depth Stream Subscription."""
        stream_symbol = self.symbol.lower().replace("/", "")
        url = f"wss://fstream.binance.com/ws/{stream_symbol}@depth20@100ms"
        log.info(f"Connecting to Order Book Depth Stream: {url}")
        
        while True:
            try:
                await self.init_session()
                async with self.session.ws_connect(
                    url, 
                    timeout=20.0, 
                    heartbeat=30.0, 
                    receive_timeout=300.0
                ) as ws:
                    log.info(f"[NETWORK] Depth Stream connected for {self.symbol}")
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                                analysis = self.hunter.analyze_order_book(data.get('b', []), data.get('a', []))
                                self._commit_microstructure(analysis)
                            except Exception as inner_e:
                                log.error(f"Error processing depth message: {inner_e}")
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            await asyncio.sleep(5)
                            break
            except Exception as e:
                log.warning(f"Depth Stream connection error: {e}. Reconnecting...")
                await asyncio.sleep(5)

    # ─── Pruning counter to avoid overhead on every 100ms message ────────────
    _prune_counter: int = 0
    _PRUNE_EVERY: int = 600  # prune every 600 inserts (~60 seconds at 100ms)

    def _commit_microstructure(self, analysis):
        """
        Offloads LiquidityHunter results to the background DB worker.
        V30.5: Also routes normalized MLOFI to system_state so the
        MicrostructureGuard FlowClassifier receives a real OFI signal.
        V30.6 [P0-FIX]: Auto-prunes oracle_microstructure to prevent unbounded disk growth.
          At 100ms inserts the table grows at ~86,400 rows/day (~8 MB/day raw).
          We retain the last 10,000 rows (~17 minutes) which is sufficient for all
          dashboard queries and microstructure lookups.
        """
        walls = analysis.get("target_walls", {})
        mlofi  = analysis.get("mlofi", 0.0)
        ts     = datetime.now(timezone.utc).isoformat()

        # 1. Persist full microstructure snapshot to oracle_microstructure HUD table
        q_ms = """
            INSERT INTO oracle_microstructure 
            (timestamp, symbol, liquidity_imbalance, vacuum_detected, bid_wall_price, bid_wall_size, ask_wall_price, ask_wall_size, signal_multiplier) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        p_ms = (
            ts,
            self.symbol,
            analysis["liquidity_imbalance"],
            1 if analysis["vacuum_detected"] else 0,
            walls.get("bid_wall_price"),
            walls.get("bid_wall_size"),
            walls.get("ask_wall_price"),
            walls.get("ask_wall_size"),
            analysis["signal_multiplier"]
        )
        self.db_queue.put((q_ms, p_ms))

        # 2. V30.5: Distribute rolling Z-score normalized MLOFI to system_state
        #    so MicrostructureGuard.evaluate_pre_trade_veto() reads a live OFI
        #    instead of defaulting to 0.0. Routed to trading_data.db (WAL-safe).
        q_ofi = """
            INSERT OR REPLACE INTO system_state (key, value, updated_at)
            VALUES (?, ?, ?)
        """
        p_ofi = ("mlofi_normalized", str(round(mlofi, 6)), ts)
        self.db_queue.put((q_ofi, p_ofi, "trading_data.db"))

        log.debug(f"[MLOFI] Committed normalized MLOFI={mlofi:.4f} to system_state.")

        # [P0-FIX] Periodic row pruning — every _PRUNE_EVERY inserts to avoid
        # per-message overhead while still bounding the table size.
        self._prune_counter += 1
        if self._prune_counter % LiveOracle._PRUNE_EVERY == 0:
            # Keep only the last 10,000 rows (>17 minutes of 100ms data)
            q_prune = """
                DELETE FROM oracle_microstructure
                WHERE symbol = ? AND id NOT IN (
                    SELECT id FROM oracle_microstructure WHERE symbol = ?
                    ORDER BY id DESC LIMIT 10000
                )
            """
            self.db_queue.put((q_prune, (self.symbol, self.symbol)))
            log.debug("[PRUNE] Queued oracle_microstructure pruning (retaining last 10,000 rows).")

    async def main_update_loop(self):
        log.info(f"Starting LiveOracle update loop for {self.symbol}...")
        while True:
            try:
                # Ensure the persistent session and CCXT exchange client are active
                await self.init_session()

                # Cache invalidation for exchangeInfo / load_markets (12h reload)
                now_time = datetime.now(timezone.utc)
                if (self.last_exchange_info_time is None or 
                    (now_time - self.last_exchange_info_time).total_seconds() > 12 * 3600):
                    log.info("Refreshing exchangeInfo/markets cache...")
                    await self.exchange.load_markets(reload=True)
                    self.last_exchange_info_time = now_time
                else:
                    await self.exchange.load_markets(reload=False)

                # 1b. Centralized BTC Macro Fetch (V30.4 Compliance Upgrade - 600-Bar Warmup)
                btc_trend = "UNKNOWN"
                try:
                    btc_raw = await self.exchange.fetch_ohlcv("BTC/USDT", timeframe='1h', limit=600)
                    def process_btc(raw):
                        df_1h = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
                        df_1h["timestamp"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
                        df_1h.set_index("timestamp", inplace=True)
                        df_1h = compute_indicators(df_1h, warmup_bars=600)
                        last_row = df_1h.iloc[-1]
                        is_warm = bool(last_row.get("warm", False))
                        e20 = float(last_row.get("ema20", np.nan))
                        e50 = float(last_row.get("ema50", np.nan))
                        e200 = float(last_row.get("ema200", np.nan))
                        close = float(last_row["close"])
                        prev_close = float(df_1h["close"].iloc[-2]) if len(df_1h) >= 2 else close
                        roc_1h = ((close - prev_close) / prev_close) * 100
                        return is_warm, e20, e50, e200, close, roc_1h

                    is_warm, ema20_1h, ema50_1h, ema200_1h, btc_price, roc_1h = await asyncio.to_thread(process_btc, btc_raw)
                    
                    # Multi-Horizon Trend Determination with Unbiased NEUTRAL Fallback:
                    if not is_warm or np.isnan(ema200_1h):
                        btc_trend = "UNKNOWN"
                    elif roc_1h <= -0.65 or (btc_price < ema20_1h and btc_price < ema50_1h and btc_price < ema200_1h):
                        btc_trend = "BEARISH"
                    elif roc_1h >= +0.65 or (btc_price > ema20_1h and btc_price > ema50_1h and btc_price > ema200_1h):
                        btc_trend = "BULLISH"
                    elif btc_price > ema200_1h:
                        btc_trend = "BULLISH" if btc_price > ema50_1h else "NEUTRAL"
                    elif btc_price < ema200_1h:
                        btc_trend = "BEARISH" if btc_price < ema50_1h else "NEUTRAL"
                    else:
                        btc_trend = "NEUTRAL"
                    log.info(f"[MACRO-CENTRAL] Multi-Horizon BTC 1H trend: {btc_trend} (Price: {btc_price:.2f} | 1h ROC: {roc_1h:+.2f}% | EMA20: {ema20_1h:.2f} | EMA50: {ema50_1h:.2f} | EMA200: {ema200_1h:.2f} | Warm: {is_warm})")
                except Exception as macro_err:
                    log.error(f"[MACRO-CENTRAL] Centralized BTC Macro fetch/calculation failed: {macro_err}")
                    btc_trend = "UNKNOWN"

                # 1. Fetching historical candles and volumes using the persistent client (600 bars for causal indicator warmup)
                bars = await self.exchange.fetch_ohlcv(self.symbol, timeframe='1h', limit=600)
                
                # Extract raw arrays for the Numba engine
                prices = np.array([float(b[4]) for b in bars], dtype=np.float64)
                
                # Volume splits: Taker buy base asset volume is at index 9 if exchange supports it.
                # If unavailable (e.g. Bitget 6-element OHLCV), derive directional volume from candle body delta.
                try:
                    buy_vol = np.array([float(b[9]) for b in bars], dtype=np.float64) 
                    sell_vol = np.array([float(b[5]) - float(b[9]) for b in bars], dtype=np.float64) 
                    neutral_vol = np.zeros_like(buy_vol) 
                except (IndexError, ValueError, TypeError):
                    # Directional volume derivation from candle close vs open
                    total_v = np.array([float(b[5]) for b in bars], dtype=np.float64)
                    opens = np.array([float(b[1]) for b in bars], dtype=np.float64)
                    closes = prices
                    bullish_mask = closes >= opens
                    buy_vol = np.where(bullish_mask, total_v * 0.70, total_v * 0.30)
                    sell_vol = total_v - buy_vol
                    neutral_vol = np.zeros_like(total_v)

                # 2. Information-Theoretic Compute Block with JIT Fallback
                try:
                    def compute_sync(p, bv, sv, nv):
                        s_r = compute_signal_readiness(p, bv, sv, nv)
                        return float(s_r.get('readiness', 0.50)), float(s_r.get('apen', 0.0)), float(s_r.get('hurst', 0.5))
                    readiness_value, apen_value, hurst_value = await asyncio.to_thread(compute_sync, prices, buy_vol, sell_vol, neutral_vol)
                except Exception as e:
                    log.warning(f"[ORACLE] Numba Engine JIT Fault: {e}. Defaulting to Signal Readiness 0.50")
                    readiness_value, apen_value, hurst_value = 0.50, 0.0, 0.5

                # 3. Standard Quantitative Metrics unified via indicators.py compute_indicators
                def process_sym_indicators(raw_bars):
                    df_sym = pd.DataFrame(raw_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
                    df_sym["timestamp"] = pd.to_datetime(df_sym["timestamp"], unit="ms", utc=True)
                    df_sym.set_index("timestamp", inplace=True)
                    df_sym = compute_indicators(df_sym, warmup_bars=min(len(df_sym), 600))
                    last = df_sym.iloc[-1]
                    
                    vol = float(last.get("atr_pct", 1.0)) if not np.isnan(last.get("atr_pct", np.nan)) else 1.0
                    vol = float(np.clip(vol, 0.5, 2.0))
                    
                    t_str = float(last.get("trend_str", 0.0))
                    # Map trend_str [0, 100] with directional slope sign into [-1.0, 1.0]
                    slope_val = float(last.get("ema_slope", 0.0))
                    direction = 1.0 if slope_val >= 0 else -1.0
                    t_strength = float(np.clip((t_str / 100.0) * direction, -1.0, 1.0))
                    
                    v_val = float(last.get("vwap", last["close"]))
                    if np.isnan(v_val):
                        v_val = float(last["close"])
                        
                    r_val = float(last.get("rsi14", 50.0))
                    if np.isnan(r_val):
                        r_val = 50.0
                    return vol, t_strength, v_val, r_val

                volatility, trend_strength, vwap, rsi = await asyncio.to_thread(process_sym_indicators, bars)

                # 4. Telemetry Output Upgrade
                log.info("==================================================")
                log.info("         LIVE MARKET ORACLE METRICS COMMITTED     ")
                log.info("==================================================")
                log.info(f" Timestamp      : {datetime.now(timezone.utc).isoformat()}")
                log.info(f" Volatility     : {volatility:.4f} (Base: 1.0)")
                log.info(f" Trend Strength : {trend_strength:.4f} (Range: [-1.0, 1.0])")
                log.info(f" VWAP           : ${vwap:.2f}")
                log.info(f" RSI (14)       : {rsi:.1f}")
                log.info(f" ApEn           : {apen_value:.4f}")
                log.info(f" Hurst          : {hurst_value:.4f}")
                log.info(f" Signal Readiness: {readiness_value:.4f}")
                log.info("==================================================")

                # 5. Offload Database Commit to background worker
                try:
                    fng = await self.fetch_fear_greed()
                    ts = datetime.now(timezone.utc).isoformat()

                    # [V36.2-DEDUP] Compute the current 5-minute candle bucket.
                    # Deduplicate writes so each 5-minute cycle only writes once.
                    now_utc = datetime.now(timezone.utc)
                    candle_bucket = now_utc.strftime("%Y-%m-%dT%H:") + str((now_utc.minute // 5) * 5).zfill(2)

                    if candle_bucket == self._last_oracle_write_ts:
                        log.debug(
                            f"[DEDUP] Telemetry write suppressed — bucket '{candle_bucket}' "
                            "already committed this candle cycle."
                        )
                    else:
                        # V30.1 Telemetry Task — only executes once per 5-minute bucket
                        q1 = """
                            INSERT INTO oracle_state 
                            (timestamp, symbol, volatility, trend_strength, vwap, rsi, apen, hurst, signal_readiness) 
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """
                        p1 = (ts, self.symbol, volatility, trend_strength, vwap, rsi, apen_value, hurst_value, readiness_value)
                        self.db_queue.put((q1, p1))
                        self._last_oracle_write_ts = candle_bucket

                        # V30.0 Legacy Support Task (idempotent UPSERT at id=1)
                        q2 = """
                            INSERT OR REPLACE INTO market_context 
                            (id, timestamp, fear_greed_index, volatility_state, trend_strength, vwap, rsi_14, signal_readiness, whale_threat)
                            VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
                        """
                        p2 = (ts, fng, volatility, trend_strength, vwap, rsi, readiness_value, 0.0)
                        self.db_queue.put((q2, p2))

                        # Dense Chronological Series Task (auto-incrementing id for temporal sequence extraction)
                        q2_seq = """
                            INSERT INTO market_context 
                            (timestamp, fear_greed_index, volatility_state, trend_strength, vwap, rsi_14, signal_readiness, whale_threat)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """
                        self.db_queue.put((q2_seq, p2))

                        # Retain last 500 chronological rows
                        q2_prune = """
                            DELETE FROM market_context WHERE id > 1 AND id NOT IN (
                                SELECT id FROM market_context WHERE id > 1 ORDER BY id DESC LIMIT 500
                            )
                        """
                        self.db_queue.put((q2_prune, ()))
                        log.debug(f"[DEDUP] oracle_state & market_context committed for bucket '{candle_bucket}'.")
                    
                    # V30.4 Centralized Macro Trend Distribution Task (idempotent UPSERT)
                    q3 = """
                        INSERT OR REPLACE INTO system_state (key, value, updated_at)
                        VALUES (?, ?, ?)
                    """
                    p3 = ("btc_macro_trend", btc_trend, ts)
                    self.db_queue.put((q3, p3, "trading_data.db"))
                    
                    # 8.3 Centralized Fear & Greed Sentiment Distribution
                    q_fng = """
                        INSERT OR REPLACE INTO system_state (key, value, updated_at)
                        VALUES ('fear_greed_index', ?, ?)
                    """
                    p_fng = (str(round(fng, 1)), ts)
                    self.db_queue.put((q_fng, p_fng, "trading_data.db"))
                    log.info(f"[MACRO-CENTRAL] Dispatched BTC trend '{btc_trend}' and F&G '{fng:.1f}' to trading_data.db system_state.")
                    
                except Exception as db_err:
                    log.error(f"[ORACLE] Failed to queue telemetry: {db_err}")

            except Exception as loop_err:
                log.error(f"[ORACLE] Critical fault in update cycle: {loop_err}")
            
            # Non-blocking WAL checkpoint: PASSIVE on ledger only (avoids locking trading_data.db)
            try:
                self.db_queue.put(("PRAGMA wal_checkpoint(PASSIVE);", ()))
                log.info("[WAL-CHECKPOINT] Queued passive WAL checkpoint for trading_ledger.db.")
            except Exception as checkpoint_err:
                log.error(f"[ORACLE] Failed to queue WAL checkpoint: {checkpoint_err}")

            if "--once" in sys.argv:
                log.info("Single-pass execution flagged. Terminating loop.")
                break

            await asyncio.sleep(300)

async def main():
    db_path = "trading_ledger.db"
    oracle = LiveOracle(symbol="BTC/USDT", db_path=db_path)
    
    # Initialize the persistent session and CCXT exchange client
    await oracle.init_session()
    
    try:
        if "--once" in sys.argv:
            # ONLY run the update loop once, skip the perpetual websocket stream
            await oracle.main_update_loop()
        else:
            # Run both the update loop and the depth stream concurrently
            await asyncio.gather(
                oracle.main_update_loop(),
                oracle.start_depth_stream()
            )
    finally:
        # Cleanup persistent connections on shutdown
        if oracle.session and not oracle.session.closed:
            await oracle.session.close()
        if oracle.exchange:
            await oracle.exchange.close()
        oracle.db_queue.put(None)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Oracle manually terminated.")

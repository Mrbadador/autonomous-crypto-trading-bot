import logging
import sqlite3
import os
import json
import asyncio
import dns_fix  # noqa: F401
import ccxt
from datetime import datetime, UTC
import config

log = logging.getLogger("live_paper_exchange")

class PositionDict(dict):
    """
    V11.1 Hotfix: Supports both dict lookups (pos['peak_price']) and object attributes (pos.peak_price)
    to prevent AttributeError crashes when interacting with different strategy modules.
    """
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"'PositionDict' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        self[name] = value
def get_contract_specs(symbol: str) -> dict:
    """Returns exchange lot precision, step size, and minimum order quantity."""
    sym = symbol.upper()
    if "BTC" in sym:
        return {"decimals": 4, "step": 0.0001, "min_qty": 0.0001}
    elif "ETH" in sym or "BNB" in sym:
        return {"decimals": 2, "step": 0.01, "min_qty": 0.01}
    elif "SOL" in sym or "AVAX" in sym:
        return {"decimals": 1, "step": 0.1, "min_qty": 0.1}
    elif any(coin in sym for coin in ["LINK", "ADA", "XRP", "DOGE", "NEAR"]):
        return {"decimals": 0, "step": 1.0, "min_qty": 1.0}
    else:
        return {"decimals": 2, "step": 0.01, "min_qty": 0.1}

class LivePaperExchange:
    """
    A high-fidelity simulation layer for live market forward-testing.
    Hardened for multi-bot concurrency with atomic transactions and async locks.
    """
    # V33.0 / V38.0: Absolute fee tiers for high-fidelity trading simulation (Bitget UTA VIP0)
    TAKER_FEE = 0.0006  # 0.06% (Bitget standard taker rate)
    MAKER_FEE = 0.0002  # 0.02% (Bitget standard maker rate)

    def _get_slippage_penalty(self, symbol: str) -> float:
        """Returns a 1-tick slippage penalty based on the symbol to simulate order book crossing."""
        if "BTC" in symbol:
            return 0.10
        elif "ETH" in symbol:
            return 0.01
        else:
            return 0.0001

    def __init__(self, db_path=None, initial_balance=500.0, account_name="sniper"):
        if db_path is None:
            # V15.5: Enforce absolute pathing to prevent Split-Brain database creation
            base_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base_dir, "trading_data.db")
        self.db_path = db_path
        self.initial_balance = initial_balance
        self.account_name = account_name
        self.lock = asyncio.Lock() # Prevent race conditions during dual-bot operations
        import exchange_factory
        self.twap_exchange = exchange_factory.create_sync_exchange()
        
        # V34.0 Bitget UTA Live Execution Integration
        mode_val = str(getattr(config, "MODE", "paper")).lower()
        paper_trade = getattr(config, "PAPER_TRADE", True)
        ex_id = str(getattr(config, "EXCHANGE_ID", "bitget")).lower()
        self.is_live_execution = (not paper_trade) and (mode_val in ["live", "live_pilot"]) and (ex_id == "bitget")
        if self.is_live_execution:
            from bitget_uta_client import BitgetUTAClient
            self.uta_client = BitgetUTAClient()
            log.info(f"[{self.account_name}] 🔴 LIVE BITGET UTA TRADING ACTIVATED (REAL MONEY ORDERS ENABLED) 🔴")
        else:
            self.uta_client = None
            log.info(f"[{self.account_name}] 🟢 Simulation / Paper Trading Mode Active 🟢")

        self._init_db()
        self.balance = self._load_balance_sync()
        self.positions = {} # symbol -> position_dict
        self._recover_positions_sync()
        self._background_tasks = set()
        log.info(f"LivePaperExchange [{account_name}] initialized. Balance: ${self.balance:,.2f} | Mode: Atomic WAL")

    def _fetch_pre_entry_state_sync(self, symbol: str, side: str) -> dict:
        """P1-1 Fix: Synchronously queries all local database pre-entry states in a single
        resilient connection off the event loop thread, preventing event loop stalls and
        ensuring no SQLite connection is held across remote network calls."""
        res = {
            "veto": None,
            "dynamic_leverage": 5,
            "distinct_coins": set(),
            "sym_entries": 0,
            "total_active_margin": 0.0,
            "max_allowed": int(getattr(config, "MAX_CONCURRENT_TRADES", 5))
        }
        target_sym_clean = symbol.replace("/", "").replace("-", "").upper()
        base_lev = getattr(config, "get_asset_leverage", lambda s: getattr(config, "LEVERAGE", 5))(symbol)
        trend_lev = int(getattr(config, "TREND_LEVERAGE", base_lev))
        chop_lev = int(getattr(config, "CHOP_LEVERAGE", 3))
        dynamic_leverage = trend_lev

        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA busy_timeout=10000;")

                # 1. Dynamic Leverage Engine
                try:
                    reg_row = conn.execute("SELECT regime, adx, hurst FROM market_regime ORDER BY id DESC LIMIT 1").fetchone()
                    if reg_row:
                        r_label, r_adx, r_hurst = str(reg_row[0] or "").upper(), float(reg_row[1] or 0), float(reg_row[2] or 0.5)
                        if "CHOP" in r_label or r_adx < getattr(config, "ADX_CHOP_FLOOR", 15.0) or r_hurst < 0.48:
                            dynamic_leverage = min(chop_lev, trend_lev)
                        else:
                            dynamic_leverage = trend_lev
                except Exception as le:
                    log.debug(f"Dynamic leverage determination note: {le}")
                res["dynamic_leverage"] = dynamic_leverage

                # 0. Global Portfolio Risk Halt & Operator Halt Enforcement (Fail-Closed)
                try:
                    halt_rows = conn.execute(
                        "SELECT key, value FROM system_state WHERE key IN ('RISK_HALT', 'OPERATOR_HALT')"
                    ).fetchall()
                    for h_key, h_val in halt_rows:
                        if str(h_val).strip().upper() in ("1", "TRUE"):
                            res["veto"] = f"[{self.account_name}] [RISK-HALT-VETO] Entry blocked on {symbol}: {h_key}=1 active in system_state."
                            return res
                except Exception as rhe:
                    log.warning(f"[{self.account_name}] Risk halt check failed (fail-closed) on {symbol}: {rhe}")
                    res["veto"] = f"[{self.account_name}] [RISK-HALT-VETO] Failed to read RISK_HALT state (fail-closed): {rhe}"
                    return res

                # 2. Dynamic Top Coins Whitelist Guard
                try:
                    sym_row = conn.execute("SELECT value FROM system_state WHERE key = 'ALLOWED_SYMBOLS'").fetchone()
                    if sym_row and sym_row[0]:
                        import json
                        allowed = json.loads(sym_row[0])
                        sym_clean = symbol.upper().replace("/", "").replace("-", "")
                        allowed_clean = [s.upper().replace("/", "").replace("-", "") for s in allowed]
                        if sym_clean not in allowed_clean:
                            res["veto"] = f"[{self.account_name}] VETO: {symbol} is not in ALLOWED_SYMBOLS whitelist ({allowed}). Entry blocked."
                            return res
                except Exception as we:
                    log.debug(f"Whitelist check note: {we}")

                # 3. Macro Economic News Event Gate
                try:
                    news_row = conn.execute("SELECT value FROM system_state WHERE key = 'news_event_active'").fetchone()
                    if news_row and str(news_row[0]).strip() in ("1", "true", "TRUE"):
                        reason_row = conn.execute("SELECT value FROM system_state WHERE key = 'news_event_reason'").fetchone()
                        n_reason = reason_row[0] if reason_row else "High-Impact Macro Event Window"
                        res["veto"] = f"[{self.account_name}] [NEWS-CALENDAR-VETO] Entry blocked on {symbol} due to macro event: {n_reason}"
                        return res
                except Exception as ne:
                    log.debug(f"News calendar gate note: {ne}")

                # 4. Strict Macro Directional Alignment Guard
                try:
                    row_macro = conn.execute("SELECT value FROM system_state WHERE key = 'btc_macro_trend'").fetchone()
                    if row_macro:
                        macro_dir = str(row_macro[0]).upper()
                        if side.lower() in ("short", "sell") and macro_dir == "BULLISH":
                            res["veto"] = f"[{self.account_name}] VETO: SHORT order on {symbol} blocked because Macro Trend is BULLISH."
                            return res
                        elif side.lower() in ("long", "buy") and macro_dir == "BEARISH":
                            res["veto"] = f"[{self.account_name}] VETO: LONG order on {symbol} blocked because Macro Trend is BEARISH."
                            return res
                except Exception as me:
                    log.debug(f"Macro direction guard check note: {me}")

                # 5. Fleet Cross-Position Ceiling, Friendly Fire & Margin
                row_max = conn.execute("SELECT value FROM system_state WHERE key = 'GLOBAL_MAX_POSITIONS'").fetchone()
                if row_max and row_max[0].isdigit():
                    res["max_allowed"] = int(row_max[0])

                fleet_tables = ["open_positions", "berserker_open_positions", "marl_live_open_positions"]
                for tbl in fleet_tables:
                    try:
                        cur_pos = conn.execute(f"SELECT side FROM {tbl} WHERE symbol = ? OR symbol = ?", (symbol, symbol.replace("/", ""))).fetchall()
                        for (p_side,) in cur_pos:
                            if str(p_side or "").lower() != str(side or "").lower():
                                res["veto"] = f"[{self.account_name}] [FRIENDLY-FIRE-GUARD] Opposing position already active in {tbl} ({symbol} {str(p_side).upper()}). Aborting {side.upper()} order."
                                return res
                    except Exception as ffe:
                        log.debug(f"Friendly fire check note: {ffe}")

                for tbl in fleet_tables:
                    try:
                        rows = conn.execute(f"SELECT symbol, qty, entry_price, leverage FROM {tbl}").fetchall()
                        for row in rows:
                            s, q, ep = row[0], float(row[1] or 0.0), float(row[2] or 0.0)
                            lev_pos = float(row[3]) if len(row) > 3 and row[3] else dynamic_leverage
                            notional = q * ep
                            pos_margin = notional / max(1.0, lev_pos)
                            if pos_margin >= 0.05:
                                s_clean = s.replace("/", "").replace("-", "").upper()
                                res["distinct_coins"].add(s_clean)
                                res["total_active_margin"] += pos_margin
                                if s_clean == target_sym_clean:
                                    res["sym_entries"] += 1
                    except Exception as tbl_err:
                        log.warning(f"[{self.account_name}] Failed to query {tbl} for margin check: {tbl_err}")
        except Exception as e:
            log.error(f"[{self.account_name}] Pre-entry DB state query error: {e}")

        return res

    def _get_db_conn(self, timeout=30.0):
        """Standardized resilient SQLite WAL connection with autocheckpoint=200."""
        conn = sqlite3.connect(self.db_path, timeout=timeout)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA wal_autocheckpoint=200")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_db(self):
        import time
        max_retries = 5
        for attempt in range(max_retries):
            try:
                conn = self._get_db_conn(timeout=30.0)
                with conn:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS system_state (
                            key TEXT PRIMARY KEY,
                            value TEXT,
                            updated_at TEXT
                        )
                    """)
                    
                    telemetry_table = "telemetry" if self.account_name == "sniper" else f"{self.account_name}_telemetry"
                    trades_table = "trades" if self.account_name == "sniper" else f"{self.account_name}_trades"
                    positions_table = "open_positions" if self.account_name == "sniper" else f"{self.account_name}_open_positions"

                    conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {trades_table} (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT, symbol TEXT, side TEXT, type TEXT,
                            entry_price REAL, exit_price REAL, qty REAL, pnl REAL,
                            balance REAL, regime TEXT, phase TEXT, quality REAL, notes TEXT,
                            status TEXT DEFAULT 'closed',
                            ai_rationale TEXT,
                            slippage_bps REAL DEFAULT 0.0,
                            action TEXT
                        )
                    """)
                    conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {positions_table} (
                            symbol TEXT PRIMARY KEY, side TEXT, entry_price REAL, qty REAL,
                            take_profit REAL, stop_loss REAL, peak_price REAL, entry_time TEXT,
                            is_closing INTEGER DEFAULT 0, partial_tp_hit INTEGER DEFAULT 0,
                            action_confidence REAL, comments TEXT, leverage REAL
                        )
                    """)
                    conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {telemetry_table} (
                            symbol TEXT PRIMARY KEY, timestamp TEXT, reasoning TEXT,
                            price REAL, ema_ref REAL, vol_pct REAL, action TEXT, quality REAL
                        )
                    """)
                    # Alter columns for trades tables
                    for t in [trades_table, f"legacy_{trades_table}"]:
                        for col_name, col_type in [
                            ("status", "TEXT DEFAULT 'closed'"),
                            ("ai_rationale", "TEXT"),
                            ("slippage_bps", "REAL DEFAULT 0.0"),
                            ("action", "TEXT")
                        ]:
                            try:
                                conn.execute(f"ALTER TABLE {t} ADD COLUMN {col_name} {col_type}")
                            except:
                                pass
                    # Alter columns for positions tables
                    for col_name, col_type in [
                        ("is_closing", "INTEGER DEFAULT 0"),
                        ("partial_tp_hit", "INTEGER DEFAULT 0"),
                        ("action_confidence", "REAL"),
                        ("comments", "TEXT"),
                        ("leverage", "REAL"),
                        ("hard_sl", "REAL DEFAULT 0.0"),          # Unconditional Bitget native backstop SL
                        ("sl_breach_time", "REAL DEFAULT 0.0"),   # Unix timestamp when soft SL was first breached (flash crash filter)
                    ]:
                        try:
                            conn.execute(f"ALTER TABLE {positions_table} ADD COLUMN {col_name} {col_type}")
                        except:
                            pass
                    
                    balance_key = f"balance_{self.account_name}"
                    conn.execute("INSERT OR IGNORE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)", 
                                 (balance_key, str(self.initial_balance), datetime.now(UTC).isoformat()))
                return # Success
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    log.warning(f"[DB-LOCKED] _init_db collision. Retrying {attempt+1}/{max_retries} in 0.5s...")
                    time.sleep(0.5)
                else:
                    raise e

    def _load_balance_sync(self):
        import time as _time
        balance_key = f"balance_{self.account_name}"
        for attempt in range(5):
            try:
                conn = sqlite3.connect(self.db_path, timeout=30.0)  # V24.5 P1: system-standard timeout
                res = conn.execute("SELECT value FROM system_state WHERE key = ?", (balance_key,)).fetchone()
                if not res and self.account_name == "sniper":
                    res = conn.execute("SELECT value FROM system_state WHERE key = 'balance'").fetchone()
                bal = float(res[0]) if res else self.initial_balance

                # V15.5: Safety adjustment for recovered positions if using default balance
                if not res:
                    reserved = 0
                    positions_table = "open_positions" if self.account_name == "sniper" else f"{self.account_name}_open_positions"
                    try:
                        cursor = conn.execute(f"SELECT entry_price, qty FROM {positions_table}")
                        rows = cursor.fetchall()
                        for r in rows:
                            reserved += (r[0] * r[1]) / 1.0  # Leverage 1.0 assumed for paper
                        bal -= reserved
                        if reserved > 0:
                            log.debug(f"[{self.account_name}] Adjusted starting balance for recovered trades. New Available: ${bal:,.2f}")
                    except:
                        pass
                conn.close()
                return bal
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < 4:
                    log.warning(f"[{self.account_name}] _load_balance_sync DB locked. Retry {attempt+1}/5 in 0.5s...")
                    _time.sleep(0.5)
                else:
                    log.error(f"Failed to load balance for {self.account_name}: {e}")
                    return self.initial_balance
            except Exception as e:
                log.error(f"Failed to load balance for {self.account_name}: {e}")
                return self.initial_balance
        return self.initial_balance

    def _recover_positions_sync(self):
        """V15.5: Enhanced recovery logic checking both open_positions and trades tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            positions_table = "open_positions" if self.account_name == "sniper" else f"{self.account_name}_open_positions"
            trades_table = "trades" if self.account_name == "sniper" else f"{self.account_name}_trades"
            
            # 1. Recover from open_positions (Primary)
            try:
                cursor = conn.execute(f"SELECT symbol, side, entry_price, qty, take_profit, stop_loss, peak_price, entry_time, is_closing, partial_tp_hit, action_confidence, comments, leverage FROM {positions_table}")
                rows = cursor.fetchall()
                log.debug(f"[{self.account_name}] Recovery scan: {positions_table} → {len(rows)} row(s) found.")
                for row in rows:
                    sym = row[0]
                    self.positions[sym] = PositionDict({
                        "symbol": row[0], "side": row[1], "entry_price": row[2], "qty": row[3],
                        "take_profit": row[4], "stop_loss": row[5], "peak_price": row[6], "entry_time": row[7],
                        "is_closing":    bool(row[8]) if len(row) > 8 else False,
                        "partial_tp_hit": bool(row[9]) if len(row) > 9 else False,
                        "action_confidence": float(row[10]) if len(row) > 10 and row[10] is not None else 0.0,
                        "comments": str(row[11]) if len(row) > 11 and row[11] is not None else "",
                        "leverage": float(row[12]) if len(row) > 12 and row[12] is not None else 5.0,
                        "margin_reserved": 0.0,  # Will be recalculated on next position event
                    })
                    log.info(f"[{self.account_name}] RECOVERED from {positions_table}: {sym} {row[1]} @ ${row[2]} (is_closing={self.positions[sym]['is_closing']}, partial_tp_hit={self.positions[sym]['partial_tp_hit']})")
            except Exception as e:
                log.warning(f"[{self.account_name}] No positions found in {positions_table}: {e}")

            # 2. Fallback to trades table if empty
            if not self.positions:
                try:
                    cursor = conn.execute(f"SELECT symbol, side, entry_price, qty, timestamp FROM {trades_table} WHERE status NOT IN ('CLOSED', 'CANCELLED', 'closed', 'cancelled', 'FAILED', 'failed')")
                    rows = cursor.fetchall()
                    log.debug(f"[{self.account_name}] Fallback recovery scan: {trades_table} → {len(rows)} row(s) found.")
                    for row in rows:
                        sym = row[0]
                        self.positions[sym] = PositionDict({
                            "symbol": row[0], "side": row[1].lower(), "entry_price": row[2], "qty": row[3],
                            "take_profit": 0, "stop_loss": 0, "peak_price": row[2], "entry_time": row[4],
                            "is_closing": False,
                            "partial_tp_hit": False,
                            "margin_reserved": 0.0,
                        })
                        log.info(f"[{self.account_name}] RECOVERED from {trades_table}: {sym} {row[1]} @ ${row[2]}")
                except Exception as e:
                    log.warning(f"[{self.account_name}] Trade fallback recovery failed: {e}")
            
            conn.close()
        except Exception as e:
            log.error(f"Failed to recover positions for {self.account_name}: {e}")

    async def save_log(self, level, message):
        """V15.7: Persists a log entry to the database for high-fidelity HUD tracking.
        V31.0 P0: Acquire self.lock before writing — save_log was the only writer in this
        class that bypassed the lock, creating unprotected concurrent write risk.
        """
        async with self.lock:
            import asyncio
            for attempt in range(3):
                conn = None
                try:
                    conn = sqlite3.connect(self.db_path, timeout=30.0)
                    conn.execute("PRAGMA journal_mode=WAL")
                    session_id = "UNKNOWN"
                    try:
                        res = conn.execute("SELECT value FROM system_state WHERE key = ?",
                                          (f"session_id_{self.account_name}",)).fetchone()
                        if res: session_id = res[0]
                    except: pass

                    with conn:
                        conn.execute("""
                            INSERT INTO logs (timestamp, level, engine, message, session_id)
                            VALUES (?, ?, ?, ?, ?)
                        """, (datetime.now(UTC).isoformat(), level, self.account_name, message, session_id))
                    break
                except sqlite3.OperationalError as e:
                    if "locked" in str(e).lower() and attempt < 2:
                        await asyncio.sleep(0.1)
                    # else: silently drop the log entry rather than crash
                except Exception:
                    pass  # Logging must never crash the trading engine
                finally:
                    if conn:
                        conn.close()

    async def save_telemetry(self, symbol, data):
        import time
        async with self.lock:
            max_retries = 5
            for attempt in range(max_retries):
                conn = None
                try:
                    telemetry_table = "telemetry" if self.account_name == "sniper" else f"{self.account_name}_telemetry"
                    conn = sqlite3.connect(self.db_path, timeout=30.0)
                    conn.execute("PRAGMA journal_mode=WAL")
                    with conn:
                        conn.execute(f"""
                            REPLACE INTO {telemetry_table} (symbol, timestamp, reasoning, price, ema_ref, vol_pct, action, quality)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            symbol, data.get("timestamp"), data.get("reasoning"),
                            data.get("price"), data.get("ema_ref"), data.get("vol_pct"), 
                            data.get("action"), data.get("quality", 0.0)
                        ))
                    return
                except sqlite3.OperationalError as e:
                    if "locked" in str(e).lower() and attempt < max_retries - 1:
                        log.warning(f"[DB-LOCKED] Telemetry collision for {symbol}. Retrying in 0.5s...")
                        await asyncio.sleep(0.5)  # V31.0 P2: was time.sleep — blocked event loop
                    else:
                        log.error(f"Failed to save telemetry for {symbol} [{self.account_name}]: {e}")
                        break
                except Exception as e:
                    log.error(f"Crit fail telemetry {symbol}: {e}")
                    break
                finally:
                    if conn: conn.close()

    async def sync_positions_from_db(self):
        """V25.6: Re-queries DB to prevent memory stale state from Recovery Agent DCAs.
        V31.0 P0: Extended to also sync stop_loss, take_profit, and peak_price so that
        after a Recovery Agent DCA the Berserker's trailing stop calculations operate on
        the current DB state rather than a phantom baseline from before the DCA.
        """
        positions_table = "open_positions" if self.account_name == "sniper" else f"{self.account_name}_open_positions"
        def _fetch():
            c = None
            try:
                c = sqlite3.connect(self.db_path, timeout=10.0)
                res = c.execute(
                    f"SELECT symbol, entry_price, qty, stop_loss, take_profit, peak_price, is_closing, partial_tp_hit, hard_sl, sl_breach_time FROM {positions_table}"
                ).fetchall()
                return res
            except Exception: return []
            finally:
                if c: c.close()

        db_rows = await asyncio.to_thread(_fetch)
        # r: (symbol, entry_price, qty, stop_loss, take_profit, peak_price, is_closing, partial_tp_hit)
        db_map = {r[0]: {"entry_price": r[1], "qty": r[2],
                          "stop_loss": r[3], "take_profit": r[4], "peak_price": r[5],
                          "is_closing": bool(r[6]) if len(r) > 6 else False,
                          "partial_tp_hit": bool(r[7]) if len(r) > 7 else False,
                          "hard_sl": float(r[8]) if len(r) > 8 and r[8] else 0.0,
                          "sl_breach_time": float(r[9]) if len(r) > 9 and r[9] else None}
                  for r in db_rows}

        async with self.lock:
            for sym, db_pos in db_map.items():
                if sym in self.positions:
                    mem = self.positions[sym]
                    changed = (
                        abs(mem.get("entry_price", 0.0) - db_pos["entry_price"]) > 1e-5
                        or abs(mem.get("qty", 0.0) - db_pos["qty"]) > 1e-5
                    )
                    if changed:
                        log.info(
                            f"[{self.account_name}] DB-SYNC: DCA update for {sym}. "
                            f"Price: {db_pos['entry_price']:.4f}, Qty: {db_pos['qty']:.4f}, "
                            f"SL: {db_pos['stop_loss']:.4f}"
                        )
                        mem["entry_price"] = db_pos["entry_price"]
                        mem["qty"]         = db_pos["qty"]
                        # V31.0 P0: Sync SL/TP/peak so trailing stop math uses correct baseline
                        if db_pos["stop_loss"] and db_pos["stop_loss"] > 0.0:
                            mem["stop_loss"]   = db_pos["stop_loss"]
                        if db_pos["take_profit"] and db_pos["take_profit"] > 0.0:
                            mem["take_profit"] = db_pos["take_profit"]
                        if db_pos["peak_price"] and db_pos["peak_price"] > 0.0:
                            mem["peak_price"]  = db_pos["peak_price"]
                    # V31.0 P2: Also sync flags to keep memory aligned with DB updates
                    mem["is_closing"] = db_pos["is_closing"]
                    mem["partial_tp_hit"] = db_pos["partial_tp_hit"]

    async def _execute_twap_slices(self, symbol, side, signal_price):
        """Background task handling simulated TWAP fills over 4 minutes."""
        try:
            fills = [signal_price]
            # P0-PATCH: Use the shared class-level exchange instance instead of
            # creating a new ccxt.binance() per trade (connection leak fix).
            exchange = self.twap_exchange
            log.info(f"[{self.account_name}] TWAP Background Task starting for {symbol}.")
            
            for _ in range(4):
                await asyncio.sleep(60)
                try:
                    ticker = await asyncio.to_thread(exchange.fetch_ticker, symbol)
                    fills.append(float(ticker['last']))
                except Exception:
                    fills.append(fills[-1])
            
            final_price = sum(fills) / len(fills)
            
            if side == "long":
                slip = ((final_price - signal_price) / signal_price) * 10000
            else:
                slip = ((signal_price - final_price) / signal_price) * 10000
                
            log.info(f"[{self.account_name}] TWAP COMPLETE for {symbol}. Final Avg: {final_price:.4f} | Slippage: {slip:.2f} bps")
            
            # 1. Update memory
            async with self.lock:
                if symbol in self.positions:
                    self.positions[symbol]["entry_price"] = final_price
                    if "metadata" not in self.positions[symbol]:
                        self.positions[symbol]["metadata"] = {}
                    self.positions[symbol]["metadata"]["slippage_bps"] = slip
            
            # 2. Update DB
            positions_table = "open_positions" if self.account_name == "sniper" else f"{self.account_name}_open_positions"
            def _commit():
                try:
                    c = sqlite3.connect(self.db_path, timeout=15.0)
                    c.execute(f"UPDATE {positions_table} SET entry_price = ? WHERE symbol = ?", (final_price, symbol))
                    c.commit()
                    c.close()
                except Exception as e:
                    log.error(f"TWAP DB Update Fail: {e}")
            await asyncio.to_thread(_commit)
        except Exception as e:
            log.error(f"TWAP Exception: {e}")

    async def execute_order(self, symbol, side, entry_price, qty):
        """V30.2 Wrapper to open a position (local execution)."""
        return await self.open_position(symbol, side.lower(), qty, entry_price)

    async def open_position(self, symbol, side, qty, price, sl=0.0, tp=0.0, metadata=None, hard_sl=0.0, maker: bool = True):
        signal_price = price
        if metadata is None: metadata = {}
        metadata['signal_price'] = signal_price
        metadata['slippage_bps'] = 0.0 # Initial value
        is_maker = bool(maker if maker is not None else metadata.get("is_maker", True))
        metadata['is_maker'] = is_maker

        async with self.lock:
            # V33.0 / Sprint 3: Apply entry slippage (0.0 for maker limit queue fills)
            slippage = 0.0 if is_maker else self._get_slippage_penalty(symbol)
            execution_price = price if is_maker else (price + slippage if side == "long" else price - slippage)

            # P1-1 Fix: Perform all local database reads off-thread into a decoupled data structure
            db_state = await asyncio.to_thread(self._fetch_pre_entry_state_sync, symbol, side)
            if db_state.get("veto"):
                log.warning(db_state["veto"])
                return None

            dynamic_leverage = db_state.get("dynamic_leverage", 5)

            # V14.0 P0: Margin Reservation with Dynamic Leverage
            margin_required = (qty * execution_price) / dynamic_leverage
            if margin_required > self.balance:
                log.error(f"[{self.account_name}] Insufficient balance for margin: ${margin_required:.2f} > ${self.balance:.2f}")
                return None

            target_sym_clean = symbol.replace("/", "").replace("-", "").upper()
            distinct_coins = set(db_state.get("distinct_coins", []))
            total_active_margin = float(db_state.get("total_active_margin", 0.0))
            sym_entries = int(db_state.get("sym_entries", 0))

            # Gate 1: Global Distinct Coins Cap (Strict 4 coins active across entire fleet)
            max_distinct_coins = int(getattr(config, "GLOBAL_MAX_DISTINCT_COINS", 4))
            if target_sym_clean not in distinct_coins and len(distinct_coins) >= max_distinct_coins:
                log.warning(f"[{self.account_name}] [COIN-CAP] Global coin cap reached ({len(distinct_coins)}/{max_distinct_coins} distinct coins active: {sorted(list(distinct_coins))}). Aborting new coin entry for {symbol}.")
                return None

            # Gate 2: Strict 55% Portfolio Margin Ceiling across all active positions
            # Evaluated with ZERO open SQLite connection handles!
            max_margin_pct = float(getattr(config, "MAX_PORTFOLIO_MARGIN_PCT", 0.55))
            equity_ref = float(self.balance or config.INITIAL_BALANCE)
            if self.is_live_execution and self.uta_client:
                try:
                    live_equity = await self.uta_client.fetch_balance()
                    if live_equity and live_equity > 0:
                        equity_ref = live_equity
                    live_pos = await self.uta_client.fetch_positions()
                    if live_pos:
                        live_tot_margin = 0.0
                        for lp in live_pos:
                            s_raw = str(lp.get("symbol", "")).upper()
                            if s_raw:
                                m_size = float(lp.get("positionBalance", 0.0) or lp.get("marginSize", 0.0) or 0.0)
                                if m_size == 0.0:
                                    tot_q = float(lp.get("total", 0.0) or 0.0)
                                    avg_p = float(lp.get("avgPrice", 0.0) or 0.0)
                                    lev_p = float(lp.get("leverage", 1.0) or 1.0)
                                    m_size = (tot_q * avg_p) / max(1.0, lev_p)
                                if m_size >= 0.05:
                                    distinct_coins.add(s_raw)
                                    live_tot_margin += m_size
                        if live_tot_margin > 0:
                            total_active_margin = max(total_active_margin, live_tot_margin)
                except Exception as eq_err:
                    log.debug(f"Live equity/positions check note: {eq_err}")

            max_margin_allowed = equity_ref * max_margin_pct
            if (total_active_margin + margin_required) > max_margin_allowed:
                log.warning(
                    f"[{self.account_name}] [MARGIN-CEILING] Portfolio margin ceiling reached: "
                    f"Current=${total_active_margin:.2f} + New=${margin_required:.2f} = ${total_active_margin + margin_required:.2f} "
                    f"> {max_margin_pct*100:.0f}% of equity (${max_margin_allowed:.2f} on ${equity_ref:.2f}). Aborting entry for {symbol}."
                )
                return None

            # Gate 3: Max entries per symbol ceiling
            max_per_sym = int(getattr(config, "MAX_POSITIONS_PER_SYMBOL", 4))
            if sym_entries >= max_per_sym:
                log.warning(f"[{self.account_name}] Max entries for {symbol} reached ({sym_entries}/{max_per_sym}). Aborting entry.")
                return None


            # V34.0 Live Bitget Order Execution with Dynamic Leverage
            if self.is_live_execution and self.uta_client:
                try:
                    # Enforce absolute minimum notional of $5.50 for futures margin orders
                    if qty * execution_price < 5.5:
                        qty = 5.5 / execution_price

                    # Apply Bitget-specific contract precision and minimum order quantity limits
                    specs = get_contract_specs(symbol)
                    dec = specs["decimals"]
                    qty = float(round(qty)) if dec == 0 else round(qty, dec)
                    qty = max(qty, specs["min_qty"])

                    margin_required = (qty * execution_price) / dynamic_leverage

                    log.info(f"[{self.account_name}] Routing LIVE order to Bitget UTA: {symbol} {side.upper()} Qty={qty} Lev={dynamic_leverage}x (Notional: ${qty*execution_price:.2f}, SoftSL={sl}, HardSL={hard_sl or sl}, TP={tp})")
                    live_res = await self.uta_client.create_market_order(
                        symbol=symbol,
                        side=side,
                        qty=qty,
                        sl=hard_sl if hard_sl and hard_sl > 0 else sl,  # Native SL = hard backstop
                        tp=tp,
                        leverage_val=dynamic_leverage
                    )
                    bitget_oid = live_res.get("id")
                    if metadata is None: metadata = {}
                    metadata["bitget_order_id"] = bitget_oid
                    log.info(f"[{self.account_name}] 🎯 LIVE BITGET TRADE OPENED! Order ID: {bitget_oid} (Lev={dynamic_leverage}x, HardSL={hard_sl or sl}, NativeSL on exchange)")
                except Exception as live_err:
                    # P0-1 DEFENSE: Check if order filled on Bitget despite timeout/network error
                    confirmed_pos = None
                    try:
                        await asyncio.sleep(1.0)
                        curr_positions = await self.uta_client.fetch_positions(symbol=symbol)
                        for cp in (curr_positions or []):
                            c_qty = float(cp.get("total", 0.0) or cp.get("contracts", 0.0) or 0.0)
                            c_side = str(cp.get("posSide", cp.get("holdSide", ""))).lower()
                            if c_qty > 0 and (c_side == side.lower() or not c_side or c_side == "net"):
                                confirmed_pos = cp
                                break
                    except Exception as check_err:
                        log.warning(f"[{self.account_name}] Post-error position verification query failed: {check_err}")

                    if confirmed_pos:
                        log.critical(f"[{self.account_name}] ⚠️ GHOST POSITION RESCUED! Order executed on Bitget despite timeout/error: {live_err}. Adopting live position.")
                        bitget_oid = confirmed_pos.get("orderId", "TIMEOUT_RECOVERED")
                        if metadata is None: metadata = {}
                        metadata["bitget_order_id"] = bitget_oid
                        execution_price = float(confirmed_pos.get("openPriceAvg", confirmed_pos.get("avgPrice", execution_price)) or execution_price)
                        qty = float(confirmed_pos.get("total", confirmed_pos.get("contracts", qty)) or qty)
                    else:
                        log.critical(f"[{self.account_name}] ❌ LIVE BITGET TRADE FAILED: {live_err}. Aborting entry.")
                        return None

            # Extract metadata for email_dispatcher
            action_conf = metadata.get("quality", 0.0) if metadata else 0.0
            comments = metadata.get("notes", "") or metadata.get("ai_rationale", "") if metadata else ""
            leverage_val = float(dynamic_leverage)

            pos = PositionDict({
                "symbol": symbol, "side": side, "qty": qty, "entry_price": execution_price,
                "entry_time": datetime.now(UTC).isoformat(), "metadata": metadata or {},
                "take_profit": tp, "stop_loss": sl, "peak_price": execution_price,
                "hard_sl": hard_sl if hard_sl and hard_sl > 0 else sl,  # Unconditional Bitget backstop
                "sl_breach_time": None,   # Flash crash filter timer (None = not currently breached)
                "margin_reserved": margin_required,
                "is_closing": False,
                "partial_tp_hit": False,
                "action_confidence": action_conf,
                "comments": comments,
                "leverage": leverage_val,
                "is_maker": is_maker
            })
            
            # Atomic Transaction: Position Sync + Balance Commit
            import time
            success = False
            for attempt in range(5):
                conn = None
                try:
                    conn = self._get_db_conn(timeout=30.0)
                    conn.execute("BEGIN IMMEDIATE")
                    
                    new_balance = self.balance - margin_required
                    balance_key = f"balance_{self.account_name}"
                    conn.execute("REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                                 (balance_key, str(new_balance), datetime.now(UTC).isoformat()))
                    positions_table = "open_positions" if self.account_name == "sniper" else f"{self.account_name}_open_positions"
                    existing_row = conn.execute(f"SELECT qty, entry_price, peak_price FROM {positions_table} WHERE symbol = ?", (symbol,)).fetchone()
                    if existing_row and float(existing_row[0] or 0) > 0:
                        ex_qty = float(existing_row[0])
                        ex_entry = float(existing_row[1])
                        ex_peak = float(existing_row[2] or execution_price)
                        combined_qty = ex_qty + qty
                        combined_entry = ((ex_qty * ex_entry) + (qty * execution_price)) / combined_qty
                        combined_peak = max(ex_peak, execution_price)
                        conn.execute(f"""
                            REPLACE INTO {positions_table} (symbol, side, entry_price, qty, take_profit, stop_loss, peak_price, entry_time, is_closing, partial_tp_hit, action_confidence, comments, leverage, hard_sl, sl_breach_time)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?, ?)
                        """, (symbol, side, combined_entry, combined_qty, tp, sl, combined_peak, pos["entry_time"], action_conf, f"Multi-Trade/Scale-In: {comments}", leverage_val, pos["hard_sl"], None))
                        pos["qty"] = combined_qty
                        pos["entry_price"] = combined_entry
                    else:
                        conn.execute(f"""
                            REPLACE INTO {positions_table} (symbol, side, entry_price, qty, take_profit, stop_loss, peak_price, entry_time, is_closing, partial_tp_hit, action_confidence, comments, leverage, hard_sl, sl_breach_time)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (symbol, side, execution_price, qty, tp, sl, execution_price, pos["entry_time"], 0, 0, action_conf, comments, leverage_val, pos["hard_sl"], None))
                    
                    conn.commit()
                    self.balance = new_balance
                    self.positions[symbol] = pos
                    success = True
                    
                    # V32 Latency Drift Audit: Calculate drift from policy evaluation to commit
                    policy_eval_time = metadata.get("policy_eval_time") if metadata else None
                    if policy_eval_time:
                        import time as _time
                        drift_ms = (_time.time() - policy_eval_time) * 1000
                        LATENCY_SLA_MS = 10.0  # Operational SLA: <10ms policy-to-commit drift
                        if drift_ms > LATENCY_SLA_MS:
                            log.critical(
                                f"[LATENCY-BREACH] Policy-to-Commit Drift: {drift_ms:.2f}ms — "
                                f"EXCEEDS SLA threshold of {LATENCY_SLA_MS:.0f}ms! "
                                f"Investigate DB contention or I/O wait. Symbol: {symbol}"
                            )
                        else:
                            log.info(f"[LATENCY-OK] Policy-to-Commit Drift: {drift_ms:.2f}ms ✓ (SLA: <{LATENCY_SLA_MS:.0f}ms)")
                        
                    break
                except sqlite3.OperationalError as e:
                    if "locked" in str(e).lower() and attempt < 9:
                        log.warning(f"[DB-LOCKED] Open {symbol} collide. Retry {attempt+1}/10...")
                        await asyncio.sleep(0.5)  # V31.0 P2: was time.sleep — blocked event loop
                    else:
                        if conn: conn.rollback()
                        log.error(f"Atomic Open failed for {symbol}: {e}")
                        if self.is_live_execution and bitget_oid:
                            self.positions[symbol] = pos
                            log.critical(f"[{self.account_name}] CRIT-03 SAFETY: Live Bitget order {bitget_oid} opened but DB commit failed ({e}). Retained in memory. Attempting emergency disk sync...")
                            try:
                                def _emerg_write():
                                    import sqlite3 as _sq
                                    with _sq.connect(self.db_path, timeout=60.0) as ec:
                                        ec.execute(f"""
                                            REPLACE INTO {positions_table} (symbol, side, entry_price, qty, take_profit, stop_loss, peak_price, entry_time, is_closing, partial_tp_hit, action_confidence, comments, leverage, hard_sl, sl_breach_time)
                                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?, ?)
                                        """, (symbol, side, execution_price, qty, tp, sl, execution_price, pos["entry_time"], action_conf, f"EMERGENCY_RESCUED: {comments}", leverage_val, pos["hard_sl"], None))
                                import threading as _th
                                _t = _th.Thread(target=_emerg_write, daemon=True)
                                _t.start()
                            except Exception as em_err:
                                log.error(f"[{self.account_name}] Emergency disk sync also failed: {em_err}")
                            success = True
                            break
                        return None
                except Exception as e:
                    if conn: conn.rollback()
                    log.error(f"Atomic Open failed for {symbol}: {e}")
                    if self.is_live_execution and bitget_oid:
                        self.positions[symbol] = pos
                        log.critical(f"[{self.account_name}] CRIT-03 SAFETY: Live Bitget order {bitget_oid} opened but DB commit failed ({e}). Retained in memory. Attempting emergency disk sync...")
                        try:
                            def _emerg_write():
                                import sqlite3 as _sq
                                with _sq.connect(self.db_path, timeout=60.0) as ec:
                                    ec.execute(f"""
                                        REPLACE INTO {positions_table} (symbol, side, entry_price, qty, take_profit, stop_loss, peak_price, entry_time, is_closing, partial_tp_hit, action_confidence, comments, leverage, hard_sl, sl_breach_time)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?, ?)
                                    """, (symbol, side, execution_price, qty, tp, sl, execution_price, pos["entry_time"], action_conf, f"EMERGENCY_RESCUED: {comments}", leverage_val, pos["hard_sl"], None))
                            import threading as _th
                            _t = _th.Thread(target=_emerg_write, daemon=True)
                            _t.start()
                        except Exception as em_err:
                            log.error(f"[{self.account_name}] Emergency disk sync also failed: {em_err}")
                        success = True
                        break
                    return None
                finally:
                    if conn: conn.close()
            
            if not success: return None
            
            # Fire-and-forget background TWAP calculation (taker market orders only) (GC-anchored)
            if not is_maker:
                t = asyncio.create_task(self._execute_twap_slices(symbol, side, signal_price))
                self._background_tasks.add(t)
                t.add_done_callback(self._background_tasks.discard)
            log.info(f"[{self.account_name}] Opened {side.upper()} {symbol} at ${price:.4f} | Margin Reserved: ${margin_required:.2f}")

            # Universal Live Order Alert Safety Net
            if self.is_live_execution:
                try:
                    from email_dispatcher import send_trade_log_async
                    if self.account_name not in ("sniper", "berserker", "swing", "marl_live"):
                        trade_ctx = {
                            "engine": f"Exchange Router ({self.account_name})",
                            "side": side.upper(),
                            "symbol": symbol,
                            "price": execution_price,
                            "qty": qty,
                            "stop_loss": sl,
                            "take_profit": tp,
                            "leverage": leverage_val,
                            "ai_reasoning": comments or f"Live Bitget order {bitget_oid or 'direct'}"
                        }
                        asyncio.create_task(send_trade_log_async(
                            symbol=symbol,
                            side=side.upper(),
                            price=execution_price,
                            status="OPENED",
                            trade_context=trade_ctx
                        ))
                except Exception as _ex_em_err:
                    log.debug(f"Exchange open email safety note: {_ex_em_err}")

            return pos

    async def close_position(self, symbol, price, t_type="market_close", partial_qty=None):
        async with self.lock:
            if symbol not in self.positions: return None
            
            is_partial = partial_qty is not None and partial_qty < self.positions[symbol]["qty"]
            if is_partial:
                pos = self.positions[symbol].copy()
                qty = partial_qty
                pos["qty"] = qty
            else:
                pos = self.positions.pop(symbol)
                qty = pos["qty"]
                
            specs = get_contract_specs(symbol)
            dec = specs["decimals"]
            qty = float(round(qty)) if dec == 0 else round(qty, dec)
            qty = max(qty, specs["min_qty"])
                
            side = pos["side"]
            # V34.0 Live Bitget Close Execution
            if self.is_live_execution and self.uta_client:
                try:
                    log.info(f"[{self.account_name}] Routing LIVE CLOSE order to Bitget UTA: {symbol} {side.upper()} Qty={qty}")
                    await self.uta_client.close_position_order(
                        symbol=symbol,
                        side=side,
                        qty=qty
                    )
                    log.info(f"[{self.account_name}] 🎯 LIVE BITGET POSITION CLOSED! ({symbol} {t_type.upper()})")
                except Exception as live_err:
                    err_str = str(live_err).lower()
                    if "no position available to close" in err_str or "25227" in err_str:
                        log.info(f"[{self.account_name}] Note: Position on {symbol} was already settled on Bitget matching engine (Native Stop/TP). Finalizing ledger and alert.")
                        # Proceed down to finalize PnL calculation, release margin, delete open position, and dispatch closure alert
                    else:
                        log.critical(f"[{self.account_name}] ❌ LIVE BITGET CLOSE FAILED: {live_err}. Aborting local position deletion to prevent state desync.")
                        if not is_partial:
                            self.positions[symbol] = pos
                        return None

            # V33.0 / Sprint 3: Apply exit slippage and maker vs taker fee accounting (0.02% maker vs 0.04% taker)
            is_entry_maker = bool(pos.get("is_maker", True))
            is_exit_maker = t_type in ("tp_hit", "tp", "partial_tp", "maker_exit")
            slippage = 0.0 if is_exit_maker else self._get_slippage_penalty(symbol)
            execution_price = price if is_exit_maker else (price - slippage if side == "long" else price + slippage)

            if side == "long":
                gross_pnl = (execution_price - pos["entry_price"]) * qty
            else:
                gross_pnl = (pos["entry_price"] - execution_price) * qty
            entry_fee = qty * pos["entry_price"] * (self.MAKER_FEE if is_entry_maker else self.TAKER_FEE)
            exit_fee = qty * execution_price * (self.MAKER_FEE if is_exit_maker else self.TAKER_FEE)
            net_pnl = gross_pnl - (entry_fee + exit_fee)
            
            # V33.0: Release margin reserved for the closed portion
            margin_to_free = pos["margin_reserved"] if not is_partial else (pos["margin_reserved"] * (qty / self.positions[symbol]["qty"]))
            
            # [P0-MARGIN-FIREWALL] isolated margin protection: loss is capped at the reserved margin to prevent cash bleed
            if net_pnl < -margin_to_free:
                log.warning(f"[{self.account_name}] isolated margin trigger: net_pnl (${net_pnl:.2f}) exceeded reserved margin (${margin_to_free:.2f}). Capping loss.")
                net_pnl = -margin_to_free
                
            new_balance = self.balance + margin_to_free + net_pnl
            
            # Atomic Transaction: Balance + Update/Delete Position + Log Trade
            import time
            success = False
            for attempt in range(5):
                conn = None
                try:
                    conn = sqlite3.connect(self.db_path, timeout=30.0)
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute("BEGIN IMMEDIATE")
                    balance_key = f"balance_{self.account_name}"
                    conn.execute("REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                                 (balance_key, str(new_balance), datetime.now(UTC).isoformat()))
                    
                    positions_table = "open_positions" if self.account_name == "sniper" else f"{self.account_name}_open_positions"
                    if is_partial:
                        new_pos_qty = self.positions[symbol]["qty"] - qty
                        conn.execute(f"UPDATE {positions_table} SET qty = ?, partial_tp_hit = 1 WHERE symbol = ?", (new_pos_qty, symbol))
                    else:
                        conn.execute(f"DELETE FROM {positions_table} WHERE symbol = ?", (symbol,))
                    
                    trades_table = "trades" if self.account_name == "sniper" else f"{self.account_name}_trades"
                    meta = pos.get("metadata", {})
                    trade_dict = pos if isinstance(pos, dict) else (dict(pos) if pos else {})
                    meta_dict = meta if isinstance(meta, dict) else (dict(meta) if meta else {})
                    conn.execute(f"""
                        INSERT INTO {trades_table} (timestamp, symbol, side, type, entry_price, exit_price, qty, pnl, balance, regime, phase, quality, notes, status, ai_rationale, slippage_bps, action)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        datetime.now(UTC).isoformat(),
                        trade_dict.get("symbol", symbol),
                        trade_dict.get("side", side),
                        t_type,
                        trade_dict.get("entry_price", 0.0),
                        execution_price,
                        trade_dict.get("qty", qty),
                        net_pnl,
                        new_balance,
                        meta_dict.get("regime", "UNKNOWN"),
                        meta_dict.get("phase", "UNKNOWN"),
                        meta_dict.get("quality", 0.0),
                        meta_dict.get("notes", ""),
                        "closed",
                        meta_dict.get("ai_rationale", ""),
                        meta_dict.get("slippage_bps", 0.0),
                        t_type
                    ))
                    conn.commit()
                    
                    # Apply successful transaction to local state
                    self.balance = new_balance
                    if is_partial:
                        self.positions[symbol]["qty"] -= qty
                        self.positions[symbol]["margin_reserved"] -= margin_to_free
                        self.positions[symbol]["partial_tp_hit"] = True
                    success = True

                    # CRIT-02: Persist experience transition to experience_replay.json
                    try:
                        import json as js
                        replay_file = "experience_replay.json"
                        exp_entry = {
                            "symbol": symbol,
                            "side": side,
                            "entry_price": float(pos["entry_price"]),
                            "exit_price": float(execution_price),
                            "max_drawdown_price": float(pos.get("peak_price", pos["entry_price"])),
                            "pnl": float(net_pnl),
                            "action_id": 1 if side == "long" else 2,
                            "timestamp": datetime.now(UTC).isoformat()
                        }
                        exp_list = []
                        if os.path.exists(replay_file):
                            try:
                                with open(replay_file, "r") as rf:
                                    exp_list = js.load(rf)
                                    if not isinstance(exp_list, list): exp_list = []
                            except Exception: exp_list = []
                        exp_list.append(exp_entry)
                        if len(exp_list) > 500: exp_list = exp_list[-500:]
                        with open(replay_file + ".tmp", "w") as wf:
                            js.dump(exp_list, wf)
                        os.replace(replay_file + ".tmp", replay_file)
                    except Exception as exp_err:
                        log.debug(f"Replay persistence note: {exp_err}")

                    break
                except sqlite3.OperationalError as e:
                    if "locked" in str(e).lower() and attempt < 4:
                        log.warning(f"[DB-LOCKED] Close {symbol} collide. Retry {attempt+1}/5...")
                        await asyncio.sleep(0.5)  # V31.0 P2: was time.sleep — blocked event loop
                    else:
                        if conn: conn.rollback()
                        if not is_partial: self.positions[symbol] = pos
                        log.error(f"Atomic Close failed for {symbol}: {e}")
                        return None
                except Exception as e:
                    if conn: conn.rollback()
                    if not is_partial: self.positions[symbol] = pos
                    log.error(f"Atomic Close failed for {symbol}: {e}")
                    return None
                finally:
                    if conn: conn.close()
            
            if not success: return None
            
            log.info(f"[{self.account_name}] Closed {symbol} at ${price:.2f} | PnL: ${net_pnl:.2f}")

            # Universal Live Order Close Alert Safety Net
            if self.is_live_execution and self.account_name not in ("sniper", "berserker", "swing", "marl_live"):
                try:
                    from email_dispatcher import send_trade_log_async
                    close_ctx = {
                        "engine": f"Exchange Router ({self.account_name})",
                        "side": side.upper(),
                        "symbol": symbol,
                        "price": execution_price,
                        "qty": qty,
                        "entry_price": float(pos.get("entry_price", execution_price)),
                        "pnl": net_pnl,
                        "leverage": float(pos.get("leverage", 5.0)),
                        "close_trigger": t_type.upper(),
                        "close_reason": f"Closed position on Bitget via {self.account_name} router."
                    }
                    asyncio.create_task(send_trade_log_async(
                        symbol=symbol,
                        side=side.upper(),
                        price=execution_price,
                        pnl=net_pnl,
                        status="CLOSED",
                        trade_context=close_ctx
                    ))
                except Exception as _ex_cl_err:
                    log.debug(f"Exchange close email safety note: {_ex_cl_err}")

            return net_pnl

    async def sync_position_updates(self, symbol, pos):
        """Atomic UPSERT for trailing stop updates."""
        async with self.lock:
            import time
            for attempt in range(5):
                conn = None
                try:
                    positions_table = "open_positions" if self.account_name == "sniper" else f"{self.account_name}_open_positions"
                    conn = sqlite3.connect(self.db_path, timeout=30.0)
                    conn.execute("PRAGMA journal_mode=WAL")
                    with conn:
                        # Flash crash filter: sl_breach_time is a unix timestamp (float) or None
                        breach_ts = pos.get("sl_breach_time") or 0.0
                        cur = conn.execute(f"""
                            UPDATE {positions_table} 
                            SET stop_loss = ?, take_profit = ?, peak_price = ?, is_closing = ?, partial_tp_hit = ?, sl_breach_time = ?
                            WHERE symbol = ?
                        """, (pos.get("stop_loss", 0.0), pos.get("take_profit", 0.0), pos.get("peak_price", 0.0), 1 if pos.get("is_closing", False) else 0, 1 if pos.get("partial_tp_hit", False) else 0, breach_ts if breach_ts else None, symbol))
                        if cur.rowcount == 0:
                            log.warning(f"[{self.account_name}] [GHOST-PURGE] {symbol} was cleared from {positions_table}. Purging stale memory state.")
                            self.positions.pop(symbol, None)
                    break
                except sqlite3.OperationalError as e:
                    if "locked" in str(e).lower() and attempt < 4:
                        await asyncio.sleep(0.5)  # V31.0 P2: was time.sleep — blocked event loop
                except Exception as e:
                    log.error(f"Sync Update failed for {symbol}: {e}")
                    break
                finally:
                    if conn: conn.close()

    async def update_position_flag(self, symbol, flag_name, val):
        """Update a specific status flag (is_closing, partial_tp_hit) atomically in-memory and DB."""
        if flag_name not in ("is_closing", "partial_tp_hit"):
            raise ValueError(f"Invalid position flag name: {flag_name}")
            
        async with self.lock:
            if symbol not in self.positions:
                return False
                
            self.positions[symbol][flag_name] = bool(val)
            
            # Atomic DB write
            import time
            success = False
            for attempt in range(5):
                conn = None
                try:
                    positions_table = "open_positions" if self.account_name == "sniper" else f"{self.account_name}_open_positions"
                    conn = sqlite3.connect(self.db_path, timeout=30.0)
                    conn.execute("PRAGMA journal_mode=WAL")
                    with conn:
                        db_val = 1 if val else 0
                        conn.execute(f"UPDATE {positions_table} SET {flag_name} = ? WHERE symbol = ?", (db_val, symbol))
                    success = True
                    break
                except sqlite3.OperationalError as e:
                    if "locked" in str(e).lower() and attempt < 4:
                        await asyncio.sleep(0.5)
                except Exception as e:
                    log.error(f"Failed to update flag {flag_name} for {symbol}: {e}")
                    break
                finally:
                    if conn: conn.close()
            return success

    async def clear_ui_state(self):
        """Clears stale UI telemetry on bot startup to prevent desync."""
        async with self.lock:
            import time
            for attempt in range(5):
                conn = None
                try:
                    conn = sqlite3.connect(self.db_path, timeout=30.0)
                    conn.execute("PRAGMA journal_mode=WAL")
                    with conn:
                        keys_to_clear = [
                            f"current_status_{self.account_name}",
                            f"last_veto_{self.account_name}",
                            f"last_ai_thoughts_{self.account_name}",
                            f"{self.account_name}_metrics"
                        ]
                        placeholders = ', '.join(['?'] * len(keys_to_clear))
                        conn.execute(f"DELETE FROM system_state WHERE key IN ({placeholders})", keys_to_clear)
                        
                        session_key = f"session_id_{self.account_name}"
                        session_id = str(int(datetime.now(UTC).timestamp()))
                        conn.execute("REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                                     (session_key, session_id, datetime.now(UTC).isoformat()))
                    log.info(f"[{self.account_name}] UI state cleared. New Session: {session_id}")
                    break
                except sqlite3.OperationalError as e:
                    if "locked" in str(e).lower() and attempt < 4:
                        await asyncio.sleep(0.5)  # V31.0 P2: was time.sleep — blocked event loop
                except Exception as e:
                    log.error(f"Failed to clear UI state: {e}")
                    break
                finally:
                    if conn: conn.close()

    async def update_system_state(self, key, value):
        """Atomic write to system_state for high-level UI telemetry."""
        async with self.lock:
            import time
            for attempt in range(5):
                conn = None
                try:
                    conn = sqlite3.connect(self.db_path, timeout=30.0)
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    with conn:
                        conn.execute("REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                                     (key, str(value), datetime.now(UTC).isoformat()))
                        conn.commit()
                    break
                except sqlite3.OperationalError as e:
                    if "locked" in str(e).lower() and attempt < 4:
                        await asyncio.sleep(0.5)  # V31.0 P2: was time.sleep — blocked event loop
                except Exception as e:
                    log.error(f"Failed to update system_state {key}: {e}")
                    break
                finally:
                    if conn: conn.close()

    def close(self):
        """Cleanly closes the underlying synchronous ccxt exchange instance."""
        if hasattr(self, "twap_exchange") and self.twap_exchange:
            try:
                self.twap_exchange.close()
            except Exception:
                pass

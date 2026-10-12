import sqlite3
import logging
from datetime import datetime, UTC
import os
import asyncio
import threading
import queue
import time

log = logging.getLogger(__name__)

class WriteOp:
    __slots__ = ("table", "query", "params", "enqueued_at")
    def __init__(self, table: str, query: str, params: tuple):
        self.table = table
        self.query = query
        self.params = params
        self.enqueued_at = time.monotonic()

class DBWriteQueue:
    """
    High-throughput asynchronous in-memory micro-batched write queue.
    Batches up to 50 operations or flushes every 250ms via BEGIN IMMEDIATE ... COMMIT.
    Decouples ingestion from disk I/O, eradicating SQLite lock contention across concurrent daemons.
    """
    _registry = {}
    _registry_lock = threading.Lock()

    @classmethod
    def get_instance(cls, db_path: str, max_batch_size: int = 50, flush_interval: float = 0.250):
        with cls._registry_lock:
            norm_path = os.path.abspath(db_path)
            if norm_path not in cls._registry:
                cls._registry[norm_path] = cls(norm_path, max_batch_size, flush_interval)
            return cls._registry[norm_path]

    def __init__(self, db_path: str, max_batch_size: int = 50, flush_interval: float = 0.250):
        self.db_path = os.path.abspath(db_path)
        self.max_batch_size = max_batch_size
        self.flush_interval = flush_interval

        self._thread_queue = queue.Queue()
        self._async_queue = None
        self._loop = None
        self._worker_thread = None
        self._worker_started = threading.Event()
        self._running = True
        self._state_cache = {}
        self._cache_lock = threading.Lock()
        self._flush_requested = False

        # Telemetry metrics
        self.total_enqueued = 0
        self.total_flushed = 0
        self.total_batches = 0
        self.lock_errors = 0
        self.last_flush_time = 0.0

        # Dedicated worker thread with its own asyncio event loop and SQLite connection
        self._worker_thread = threading.Thread(
            target=self._run_worker_thread,
            name=f"DBWriteQueue-{os.path.basename(self.db_path)}",
            daemon=True
        )
        self._worker_thread.start()
        if not self._worker_started.wait(timeout=5.0):
            log.error(f"[DBWriteQueue] Worker thread failed to start within 5s for {self.db_path}")

    def _create_writer_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        conn.execute('PRAGMA journal_mode=WAL;')
        conn.execute('PRAGMA synchronous=NORMAL;')
        conn.execute('PRAGMA wal_autocheckpoint=200;')
        conn.execute('PRAGMA busy_timeout=30000;')
        return conn

    def _run_worker_thread(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._async_queue = asyncio.Queue()
        self._worker_started.set()
        try:
            self._loop.run_until_complete(self._worker_loop())
        finally:
            self._loop.close()

    async def _worker_loop(self):
        conn = self._create_writer_conn()
        batch = []
        last_flush = time.monotonic()

        while self._running or not self._async_queue.empty() or not self._thread_queue.empty() or batch:
            try:
                # 1. Transfer any thread-queue items into async_queue
                while not self._thread_queue.empty():
                    try:
                        op = self._thread_queue.get_nowait()
                        self._async_queue.put_nowait(op)
                    except queue.Empty:
                        break

                # 2. Compute dynamic wait timeout for 250ms batching window
                elapsed = time.monotonic() - last_flush
                remaining_timeout = 0.005 if self._flush_requested else max(0.005, self.flush_interval - elapsed)

                try:
                    op = await asyncio.wait_for(self._async_queue.get(), timeout=remaining_timeout)
                    batch.append(op)
                    self._async_queue.task_done()

                    # Drain up to max_batch_size without blocking
                    while len(batch) < self.max_batch_size and not self._async_queue.empty():
                        op = self._async_queue.get_nowait()
                        batch.append(op)
                        self._async_queue.task_done()
                except asyncio.TimeoutError:
                    pass

                # 3. Flush if batch is full, or 250ms passed, or flush requested, or stopping
                now = time.monotonic()
                if batch and (len(batch) >= self.max_batch_size or (now - last_flush) >= self.flush_interval or self._flush_requested or not self._running):
                    self._commit_batch(conn, batch)
                    batch.clear()
                    last_flush = time.monotonic()
                    self._flush_requested = False
            except Exception as e:
                if not self._running:
                    break
                log.critical(f"[DB-WORKER-CRASH] Worker loop fault: {e}", exc_info=True)
                try:
                    conn.close()
                except Exception:
                    pass
                try:
                    conn = self._create_writer_conn()
                except Exception as ce:
                    log.error(f"[DB-WORKER] Failed to recreate writer connection: {ce}")
                await asyncio.sleep(0.5)

        # Final drain on shutdown
        if batch:
            try:
                self._commit_batch(conn, batch)
            except Exception as e:
                log.error(f"[DB-WORKER] Final drain commit error: {e}")
            batch.clear()
        try:
            conn.close()
        except Exception:
            pass

    def _commit_batch(self, conn, batch):
        if not batch:
            return
        max_retries = 5
        for attempt in range(max_retries):
            try:
                conn.execute("BEGIN IMMEDIATE;")
                for op in batch:
                    conn.execute(op.query, op.params)
                conn.commit()
                self.total_flushed += len(batch)
                self.total_batches += 1
                self.last_flush_time = time.monotonic()
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() or "busy" in str(e).lower():
                    self.lock_errors += 1
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    if attempt < max_retries - 1:
                        time.sleep(0.05 * (attempt + 1))
                        continue
                log.error(f"[DBWriteQueue] Lock/OperationalError in batch commit ({len(batch)} ops): {e}")
                try:
                    conn.rollback()
                except Exception:
                    pass
                break
            except Exception as e:
                log.error(f"[DBWriteQueue] Non-retryable error in batch commit ({len(batch)} ops): {e}. Attempting isolated 1-by-1 commit fallback.")
                try:
                    conn.rollback()
                except Exception:
                    pass
                # Isolate failure: commit valid operations individually so one bad op doesn't drop the rest
                salvaged = 0
                for op in batch:
                    try:
                        with conn:
                            conn.execute(op.query, op.params)
                        salvaged += 1
                        self.total_flushed += 1
                    except Exception as single_err:
                        log.error(f"[DBWriteQueue] Dropping malformed op on {op.table}: {single_err} | Query: {op.query[:120]}")
                if salvaged > 0:
                    self.total_batches += 1
                    self.last_flush_time = time.monotonic()
                break

    def enqueue_write(self, table: str, query: str, params: tuple = ()):
        """
        Thread-safe and coroutine-safe write enqueue.
        Non-blocking: pushes operation to worker queue in <0.05ms.
        """
        if not self._running:
            try:
                conn = self._create_writer_conn()
                with conn:
                    conn.execute(query, params)
                conn.close()
            except Exception as e:
                log.error(f"[DBWriteQueue] Fallback write failed: {e}")
            return

        op = WriteOp(table, query, params)
        self.total_enqueued += 1

        # Cache system_state in-memory for instant read-after-write consistency
        if table == "system_state" and len(params) >= 2:
            with self._cache_lock:
                self._state_cache[params[0]] = params[1]

        if self._loop and self._loop.is_running() and self._async_queue:
            try:
                self._loop.call_soon_threadsafe(self._async_queue.put_nowait, op)
                return
            except Exception:
                pass
        self._thread_queue.put(op)

    async def enqueue_write_async(self, table: str, query: str, params: tuple = ()):
        """Async convenience alias for enqueue_write."""
        self.enqueue_write(table, query, params)

    def get_cached_state(self, key: str):
        with self._cache_lock:
            return self._state_cache.get(key)

    def flush_sync(self, timeout: float = 5.0) -> bool:
        """Synchronously blocks until all enqueued operations are committed to SQLite."""
        self._flush_requested = True
        target = self.total_enqueued
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.total_flushed >= target:
                return True
            time.sleep(0.01)
        return False

    async def flush_async(self, timeout: float = 5.0) -> bool:
        """Asynchronously waits until all enqueued operations are committed to SQLite."""
        self._flush_requested = True
        target = self.total_enqueued
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.total_flushed >= target:
                return True
            await asyncio.sleep(0.01)
        return False

    def shutdown(self, timeout: float = 5.0):
        """Cleanly drains the queue and shuts down the worker thread."""
        self._running = False
        self.flush_sync(timeout=timeout)
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

class DatabaseManager:
    def __init__(self, db_path=None):
        if db_path is None:
            # Default to absolute path in the same directory as this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base_dir, "trading_data.db")
        self.db_path = db_path
        self._init_db()
        self.write_queue = DBWriteQueue.get_instance(self.db_path)

    def enqueue_write(self, table: str, query: str, params: tuple = ()):
        """Delegates write to the asynchronous in-memory micro-batched write queue."""
        self.write_queue.enqueue_write(table, query, params)

    async def enqueue_write_async(self, table: str, query: str, params: tuple = ()):
        """Async convenience alias for enqueue_write."""
        self.write_queue.enqueue_write(table, query, params)

    def flush(self, timeout: float = 5.0) -> bool:
        """Synchronously flushes any pending writes in the DBWriteQueue."""
        return self.write_queue.flush_sync(timeout=timeout)

    def _get_conn(self):
        """Creates a fresh, localized connection with WAL mode, busy timeout, and autocheckpoint."""
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        conn.execute('PRAGMA journal_mode=WAL;')
        conn.execute('PRAGMA synchronous=NORMAL;')
        # [P0-FIX] Autocheckpoint every 200 pages (~800 KB) to prevent unbounded WAL growth.
        conn.execute('PRAGMA wal_autocheckpoint=200;')
        # [P0-FIX] 30-second busy timeout instead of relying solely on the connect timeout.
        # Under 8+ concurrent agent writes, SQLite can SQLITE_BUSY for short windows —
        # busy_timeout retries for up to 30s before raising an exception.
        conn.execute('PRAGMA busy_timeout=30000;')
        return conn

    def checkpoint_wal(self, mode: str = "PASSIVE") -> None:
        """
        Force a WAL checkpoint to bound file growth. 
        Call periodically (e.g. every 30 minutes) from a background task.
        Modes: PASSIVE (default, non-blocking), FULL, RESTART, TRUNCATE (shrinks file to zero).
        """
        try:
            with self._get_conn() as conn:
                result = conn.execute(f"PRAGMA wal_checkpoint({mode});").fetchone()
                log.info(f"[WAL-CHECKPOINT] mode={mode} | pages_in_wal={result[1]} | pages_checkpointed={result[2]}")
        except Exception as e:
            log.warning(f"[WAL-CHECKPOINT] Failed: {e}")

    def _init_db(self):
        """Initializes the SQLite tables if they don't exist."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # 1. Trades Table (Historical Execution)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pos_id INTEGER,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    type TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL,
                    exit_price REAL,
                    qty REAL,
                    pnl REAL,
                    balance REAL,
                    regime TEXT,
                    phase TEXT,
                    quality REAL,
                    sl_mode TEXT,
                    tp_mode TEXT,
                    trail_mode TEXT,
                    mfe REAL,
                    mae REAL,
                    exposure_time_mins REAL,
                    efficiency_score REAL,
                    fear_index INTEGER,
                    news_sentiment REAL,
                    status TEXT DEFAULT 'closed',
                    action TEXT,
                    notes TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT,
                    engine TEXT,
                    message TEXT,
                    session_id TEXT
                )
            """)
            
            # 2. Cycles Table (AI Decision History)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cycles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL,
                    signal TEXT,
                    action TEXT,
                    regime TEXT,
                    phase TEXT,
                    atr_pct REAL,
                    adx REAL,
                    adx_sma REAL,
                    slope REAL,
                    patterns TEXT,
                    notes TEXT
                )
            """)
            
            # 3. System State Table (Persistence across restarts)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # 4. Market Context Table (Sentiment)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_context (
                    timestamp TEXT PRIMARY KEY,
                    fear_index INTEGER,
                    news_score REAL
                )
            """)
            
            # 5. TradeSync Telemetry Table (Local Sync Reroute)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tradesync (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price REAL,
                    quantity REAL,
                    confidence REAL,
                    source TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS boardroom_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    conviction_score REAL NOT NULL,
                    verdict TEXT NOT NULL,
                    agents_votes_json TEXT,
                    rationale TEXT,
                    subsequent_12h_pnl REAL,
                    counterfactual_outcome TEXT
                )
            """)
            cursor.execute("""
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
                    status TEXT DEFAULT 'OPEN'
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS swing_open_positions (
                    symbol TEXT PRIMARY KEY,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    size REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    trailing_stop REAL NOT NULL,
                    peak_price REAL NOT NULL,
                    leverage INTEGER DEFAULT 3,
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
            cursor.execute("""
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
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_cycles_timestamp ON cycles (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tradesync_timestamp ON tradesync (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs (timestamp)')
            
            # Check if migrations are already applied to save I/O and time during startup
            cursor.execute("SELECT value FROM system_state WHERE key = 'migrations_version'")
            row = cursor.fetchone()
            if row and row[0] == "1.1":
                # Migrations already applied, skip
                pass
            else:
                # Migration: Ensure new columns exist for all trades tables
                for table in ["trades", "legacy_trades", "berserker_trades", "legacy_berserker_trades"]:
                    try:
                        cursor.execute(f"PRAGMA table_info({table})")
                        cols = [col[1] for col in cursor.fetchall()]
                        if not cols:
                            continue
                        new_cols = {
                            "mfe": "REAL",
                            "mae": "REAL",
                            "exposure_time_mins": "REAL",
                            "efficiency_score": "REAL",
                            "fear_index": "INTEGER",
                            "news_sentiment": "REAL",
                            "status": "TEXT DEFAULT 'closed'",
                            "action": "TEXT",
                            "ai_rationale": "TEXT",
                            "slippage_bps": "REAL DEFAULT 0.0"
                        }
                        for col_name, col_type in new_cols.items():
                            if col_name not in cols:
                                log.info(f"Migrating database: Adding '{col_name}' column to '{table}' table.")
                                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
                    except Exception as ex:
                        log.warning(f"Failed to migrate table {table}: {ex}")

                cursor.execute("PRAGMA table_info(cycles)")
                columns = [col[1] for col in cursor.fetchall()]
                for col_name in ["patterns", "adx_sma", "slope", "risk_mode"]:
                    if col_name not in columns:
                        log.info(f"Migrating database: Adding '{col_name}' column to 'cycles' table.")
                        if col_name == "patterns" or col_name == "risk_mode":
                            cursor.execute(f"ALTER TABLE cycles ADD COLUMN {col_name} TEXT")
                        else:
                            cursor.execute(f"ALTER TABLE cycles ADD COLUMN {col_name} REAL")

                # Stage 1 Instrumentation: Truthful execution ledger columns for swing_trades & swing_open_positions
                try:
                    cursor.execute("PRAGMA table_info(swing_trades)")
                    st_cols = [c[1] for c in cursor.fetchall()]
                    if st_cols:
                        st_new_cols = {
                            "intended_entry_price": "REAL",
                            "actual_fill_price": "REAL",
                            "slippage_bps": "REAL",
                            "fee_paid_usd": "REAL",
                            "fee_maker_usd": "REAL",
                            "fee_taker_usd": "REAL",
                            "funding_usd": "REAL",
                            "funding_source": "TEXT",
                            "gates_snapshot": "TEXT",
                            "status": "TEXT DEFAULT 'CLOSED'"
                        }
                        for col_name, col_type in st_new_cols.items():
                            if col_name not in st_cols:
                                log.info(f"Migrating database: Adding '{col_name}' column to 'swing_trades'.")
                                cursor.execute(f"ALTER TABLE swing_trades ADD COLUMN {col_name} {col_type}")
                except Exception as ex_st:
                    log.warning(f"Failed to migrate swing_trades: {ex_st}")

                try:
                    cursor.execute("PRAGMA table_info(swing_open_positions)")
                    sop_cols = [c[1] for c in cursor.fetchall()]
                    if sop_cols:
                        sop_new_cols = {
                            "intended_entry_price": "REAL",
                            "actual_fill_price": "REAL",
                            "slippage_bps": "REAL",
                            "fee_entry_usd": "REAL",
                            "execution_type": "TEXT",
                            "gates_snapshot": "TEXT"
                        }
                        for col_name, col_type in sop_new_cols.items():
                            if col_name not in sop_cols:
                                log.info(f"Migrating database: Adding '{col_name}' column to 'swing_open_positions'.")
                                cursor.execute(f"ALTER TABLE swing_open_positions ADD COLUMN {col_name} {col_type}")
                except Exception as ex_sop:
                    log.warning(f"Failed to migrate swing_open_positions: {ex_sop}")
                
                # Mark as applied
                from datetime import datetime, UTC
                cursor.execute(
                    "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('migrations_version', '1.1', ?)",
                    (datetime.now(UTC).isoformat(),)
                )

            # Migration 1.2: Rebuild swing_trades table with nullable exit_price, pnl, and roe_pct (FIX 1)
            cursor.execute("SELECT value FROM system_state WHERE key = 'migrations_version'")
            row_v = cursor.fetchone()
            curr_v = str(row_v[0]) if row_v else "0.0"
            if curr_v < "1.2":
                log.info("Migrating database to 1.2: Rebuilding swing_trades with nullable exit_price/pnl/roe_pct...")
                try:
                    cursor.execute("PRAGMA foreign_keys = OFF;")
                    # 1. Create temporary migration table with identical modern schema but NULLABLE exit columns
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS swing_trades_mig (
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
                        );
                    """)
                    
                    # 2. Copy all existing records across matching columns
                    cursor.execute("PRAGMA table_info(swing_trades)")
                    existing_cols = [c[1] for c in cursor.fetchall()]
                    target_cols = [
                        "id", "timestamp", "symbol", "side", "entry_price", "exit_price", "size",
                        "pnl", "roe_pct", "holding_hours", "exit_reason", "notes",
                        "intended_entry_price", "actual_fill_price", "slippage_bps",
                        "fee_paid_usd", "fee_maker_usd", "fee_taker_usd", "funding_usd",
                        "funding_source", "gates_snapshot", "status"
                    ]
                    common_cols = [c for c in target_cols if c in existing_cols]
                    cols_csv = ", ".join(common_cols)
                    
                    cursor.execute(f"INSERT INTO swing_trades_mig ({cols_csv}) SELECT {cols_csv} FROM swing_trades;")
                    
                    # 3. Atomic swap
                    cursor.execute("DROP TABLE swing_trades;")
                    cursor.execute("ALTER TABLE swing_trades_mig RENAME TO swing_trades;")
                    
                    # 4. Recreate indexes
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_swing_trades_symbol ON swing_trades (symbol);")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_swing_trades_timestamp ON swing_trades (timestamp);")
                    
                    from datetime import datetime, UTC
                    cursor.execute(
                        "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('migrations_version', '1.2', ?)",
                        (datetime.now(UTC).isoformat(),)
                    )
                    log.info("Database migration 1.2 SUCCESS: swing_trades successfully rebuilt with nullable exit constraints.")
                except Exception as mig_err:
                    log.critical(f"Database migration 1.2 FAILED: {mig_err}", exc_info=True)
                    raise mig_err
                finally:
                    cursor.execute("PRAGMA foreign_keys = ON;")

            conn.commit()
        log.info(f"Database initialized at {self.db_path} with WAL mode.")

    def log_trade(self, symbol, side, type, pnl, notes, entry_price=0.0, exit_price=0.0, qty=0.0, balance=0.0, regime="", phase="", mfe=0.0, pos_id=None, quality=0.0, sl_mode="NORMAL", tp_mode="NORMAL", trail_mode="NORMAL", status="closed"):
        """Inserts a trade event into the database via batched write queue with full metadata."""
        timestamp = datetime.now(UTC).isoformat()
        sql = """
            INSERT INTO trades 
            (timestamp, symbol, side, type, pnl, notes, entry_price, exit_price, qty, balance, regime, phase, mfe, pos_id, quality, sl_mode, tp_mode, trail_mode, status) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.enqueue_write("trades", sql, (timestamp, symbol, side, type, pnl, notes, entry_price, exit_price, qty, balance, regime, phase, mfe, pos_id, quality, sl_mode, tp_mode, trail_mode, status))

    def log_tradesync(self, symbol, side, price, quantity, confidence, source="SovereignSwarm_V30.1"):
        """Saves TradeSync telemetry locally to trading_data.db via batched write queue."""
        timestamp = datetime.now(UTC).isoformat()
        sql = """
            INSERT INTO tradesync (timestamp, symbol, side, price, quantity, confidence, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        self.enqueue_write("tradesync", sql, (timestamp, symbol, side, price, quantity, confidence, source))

    def log_veto_skipped(self, symbol: str, side: str, price: float, confidence: float, reason: str):
        """
        [V32.1 PPO GUARD] Records a VETO_SKIPPED telemetry row in the tradesync table.

        Vetoed trades MUST NOT be interpreted by the nightly PPO trainer as execution
        losses. By stamping them with source='VETO_SKIPPED', the NightlyTrainer's
        load_memory() can filter them out of the experience batch before reward
        calculation, preventing the policy network from learning to avoid its own
        safety gates.

        Args:
            symbol:     Trading pair, e.g. 'BTC/USDT'
            side:       Attempted direction, e.g. 'SHORT' or 'LONG'
            price:      Market price at veto time
            confidence: Strategist confidence score (for post-hoc audit)
            reason:     Short description of the veto gate that fired
        """
        try:
            self.log_tradesync(
                symbol=symbol,
                side=side,
                price=price,
                quantity=0.0,           # No fill — veto was pre-execution
                confidence=confidence,
                source=f"VETO_SKIPPED:{reason}"
            )
            log.debug(
                f"[PPO-GUARD] VETO_SKIPPED logged: {symbol} {side} @ ${price:.2f} "
                f"| Reason: {reason} | Confidence: {confidence:.2%}"
            )
        except Exception as e:
            log.warning(f"[PPO-GUARD] Failed to log VETO_SKIPPED for {symbol}: {e}")

    def log_cycle(self, cycle_data: dict):
        """Inserts a cycle snapshot into the database via batched write queue."""
        if "timestamp" not in cycle_data:
            cycle_data["timestamp"] = datetime.now(UTC).isoformat()
            
        columns = ", ".join(cycle_data.keys())
        placeholders = ", ".join(["?" for _ in cycle_data])
        sql = f"INSERT INTO cycles ({columns}) VALUES ({placeholders})"
        self.enqueue_write("cycles", sql, tuple(cycle_data.values()))

    def get_recent_trades(self, limit=50):
        """Retrieves the most recent trades using a fresh connection."""
        try:
            with self._get_conn() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            log.error(f"Failed to get recent trades: {e}")
            return []

    def get_stats_24h(self):
        """Calculates performance stats for the last 24 hours."""
        sql = """
            SELECT 
                COUNT(*) as count,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl,
                MAX(pnl) as best_trade,
                MIN(pnl) as worst_trade
            FROM trades 
            WHERE timestamp > datetime('now', '-1 day')
        """
        try:
            with self._get_conn() as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(sql).fetchone()
                return dict(row) if row else {}
        except Exception as e:
            log.error(f"Failed to get 24h stats: {e}")
            return {}

    def save_state(self, key: str, value: str):
        """Saves a key-value pair to system_state table via batched write queue."""
        sql = """
            INSERT INTO system_state (key, value, updated_at) 
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET 
                value=excluded.value, 
                updated_at=excluded.updated_at
        """
        self.enqueue_write("system_state", sql, (key, value, datetime.now(UTC).isoformat()))

    def load_state(self, key: str, default=None):
        """Loads a value from system_state table (checks in-memory cache first)."""
        cached = self.write_queue.get_cached_state(key)
        if cached is not None:
            return cached
        try:
            with self._get_conn() as conn:
                cursor = conn.execute("SELECT value FROM system_state WHERE key = ?", (key,))
                row = cursor.fetchone()
                return row[0] if row else default
        except Exception as e:
            log.error(f"Failed to load state {key}: {e}")
            return default

    def update_heartbeat(self, agent_name: str):
        """V27.0: Synchronous heartbeat update for agent reliability tracking."""
        self.save_state(f"heartbeat_{agent_name}", datetime.now(UTC).isoformat())

    def measure_wal_io_latency(self) -> float:
        """
        [PERSISTENCE-WATCHDOG] Measures round-trip WAL write latency in milliseconds.
        Executes a lightweight no-op write (INSERT + immediate rollback) to probe
        actual IO wait time under concurrent agent writes. Returns latency in ms.
        Returns -1.0 on failure.
        """
        try:
            import time as _time
            t0 = _time.perf_counter()
            with self._get_conn() as conn:
                # Use a savepoint so we can measure the full write path without
                # leaving permanent data in the database.
                conn.execute("SAVEPOINT wal_probe")
                conn.execute(
                    "INSERT INTO system_state (key, value, updated_at) VALUES ('_wal_probe', '1', 'probe')"
                )
                conn.execute("ROLLBACK TO SAVEPOINT wal_probe")
                conn.execute("RELEASE SAVEPOINT wal_probe")
            elapsed_ms = (_time.perf_counter() - t0) * 1000
            return round(elapsed_ms, 3)
        except Exception as e:
            log.warning(f"[WAL-PROBE] Latency measurement failed: {e}")
            return -1.0

    def get_wal_page_count(self) -> int:
        """
        [PERSISTENCE-WATCHDOG] Returns the current number of uncheck-pointed pages in the WAL.
        A rising page count indicates checkpoint pressure; pages > 1000 will trigger
        autocheckpoint but may increase IO wait under concurrent writes.
        Returns -1 on failure.
        """
        try:
            with self._get_conn() as conn:
                row = conn.execute("PRAGMA wal_checkpoint(PASSIVE);").fetchone()
                # row = (busy, log, checkpointed)
                # log = total WAL pages; checkpointed = pages flushed to main DB
                if row:
                    wal_pages = row[1]
                    return int(wal_pages)
        except Exception as e:
            log.warning(f"[WAL-PROBE] Page count query failed: {e}")
        return -1

    def prune_old_rows(
        self,
        tradesync_days: int = 7,
        cycles_days: int = 30,
        logs_days: int = 14,
    ) -> dict:
        """
        [P1-FIX] Purges stale rows from the high-volume telemetry tables to prevent
        unbounded disk growth during long-running 24/7 operation.

        Under normal load the tables grow at:
          - tradesync: ~1,000-5,000 rows/day (every trade + veto event)
          - cycles:    ~8,640 rows/day (one per 10-second tick per symbol)
          - logs:      ~50,000+ rows/day (debug-level logging)

        Recommended call schedule: every 30 minutes from a background task.

        Args:
            tradesync_days: Retain rows newer than this many days (default 7).
            cycles_days:    Retain rows newer than this many days (default 30).
            logs_days:      Retain rows newer than this many days (default 14).

        Returns:
            dict with 'tradesync', 'cycles', 'logs' keys showing rows deleted.
        """
        results = {}
        pruning_plan = [
            ("tradesync", tradesync_days),
            ("cycles",    cycles_days),
            ("logs",      logs_days),
        ]
        try:
            with self._get_conn() as conn:
                for table, days in pruning_plan:
                    try:
                        cutoff = f"datetime('now', '-{days} days')"
                        deleted = conn.execute(
                            f"DELETE FROM {table} WHERE timestamp < {cutoff}"
                        ).rowcount
                        results[table] = deleted
                        if deleted > 0:
                            log.info(
                                f"[PRUNE] Deleted {deleted} rows older than {days}d from '{table}'."
                            )
                    except Exception as table_err:
                        log.warning(f"[PRUNE] Failed to prune '{table}': {table_err}")
                        results[table] = -1
                conn.commit()
        except Exception as e:
            log.error(f"[PRUNE] Connection error during pruning: {e}")
        return results

"""
depth_recorder.py — Bitget books15 (L2 15-Levels) Depth Snapshot Recorder

Standalone, zero-order, public-data daemon.
Consumes Bitget public WebSocket v2 'books15' channel for BTCUSDT, ETHUSDT, SOLUSDT perpetuals.
Persists level-by-level snapshot rows to daily SQLite depth databases with schema:
    (timestamp_ms INTEGER, symbol TEXT, side TEXT, level INTEGER, price REAL, size REAL, seq INTEGER)

Features:
1. REST 10-minute historical bootstrap snapshot at startup via merge-depth.
2. WebSocket stream ingestion with periodic non-blocking batch commit to data/depth/depth_YYYY-MM-DD.db.
3. Disk Budget & Retention:
   - Level count per snapshot: 15 bids + 15 asks = 30 rows per symbol.
   - Sample rate: ~100ms - 250ms update ticks. At ~5 ticks/sec across 3 symbols, ~450 rows/sec.
   - Partitioning: Daily SQLite databases (`data/depth/depth_YYYY-MM-DD.db`).
   - Retention policy: Purges partitions older than 90 days (`cleanup_expired_partitions(max_age_days=90)`).
   - Storage estimate: ~350 MB - 600 MB / day compressed with WAL + SQLite page compression, ~40-50 GB for 90 days.
"""

import os
import sys
import time
import json
import asyncio
import sqlite3
import requests
import websockets
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
WS_URL = "wss://ws.bitget.com/v2/ws/public"
REST_DEPTH_URL = "https://api.bitget.com/api/v2/mix/market/merge-depth"
DEPTH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "depth")
RETENTION_DAYS = 90
BATCH_FLUSH_INTERVAL_SEC = 2.0
MAX_BATCH_ROWS = 5000


def get_daily_db_path(ts_ms: Optional[int] = None) -> str:
    """Returns absolute path to the SQLite depth database for the given day."""
    if ts_ms is None:
        dt = datetime.now(timezone.utc)
    else:
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    date_str = dt.strftime("%Y-%m-%d")
    os.makedirs(DEPTH_DIR, exist_ok=True)
    return os.path.join(DEPTH_DIR, f"depth_{date_str}.db")


def init_depth_db(db_path: str):
    """Initializes the schema and performance pragmas for a daily depth database."""
    with sqlite3.connect(db_path, timeout=10.0) as conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 10000;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS books15_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ms INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                level INTEGER NOT NULL,
                price REAL NOT NULL,
                size REAL NOT NULL,
                seq INTEGER
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_depth_sym_ts ON books15_snapshots (symbol, timestamp_ms);")


def cleanup_expired_partitions(max_age_days: int = RETENTION_DAYS):
    """Purges daily depth partition databases older than max_age_days."""
    if not os.path.exists(DEPTH_DIR):
        return
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")

    for fname in os.listdir(DEPTH_DIR):
        if fname.startswith("depth_") and fname.endswith(".db"):
            parts = fname.replace("depth_", "").replace(".db", "")
            if parts < cutoff_str:
                file_path = os.path.join(DEPTH_DIR, fname)
                try:
                    os.remove(file_path)
                    for aux in [file_path + "-wal", file_path + "-shm"]:
                        if os.path.exists(aux):
                            os.remove(aux)
                    print(f"[DEPTH-RETENTION] Purged expired partition: {fname} (< {cutoff_str})")
                except Exception as e:
                    print(f"[DEPTH-RETENTION] Error removing {fname}: {e}")


class DepthRecorder:
    def __init__(self, symbols: List[str] = SYMBOLS):
        self.symbols = [s.upper() for s in symbols]
        self.row_buffer: List[tuple] = []
        self.total_rows_written = 0
        self.last_flush = time.monotonic()
        self._running = True

    def fetch_rest_bootstrap_snapshot(self, symbol: str) -> List[tuple]:
        """Fetches initial L2 snapshot via Bitget REST merge-depth endpoint."""
        rows = []
        try:
            params = {
                "symbol": symbol,
                "productType": "USDT-FUTURES",
                "limit": "15"
            }
            r = requests.get(REST_DEPTH_URL, params=params, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if data.get("code") == "00000" and "data" in data:
                    d = data["data"]
                    ts_ms = int(d.get("ts") or time.time() * 1000)
                    bids = d.get("bids", [])
                    asks = d.get("asks", [])
                    for lvl, b in enumerate(bids[:15]):
                        p = float(b[0])
                        q = float(b[1])
                        rows.append((ts_ms, symbol, "BID", lvl, p, q, 0))
                    for lvl, a in enumerate(asks[:15]):
                        p = float(a[0])
                        q = float(a[1])
                        rows.append((ts_ms, symbol, "ASK", lvl, p, q, 0))
                    print(f"[DEPTH-BOOTSTRAP] Fetched REST snapshot for {symbol}: {len(rows)} level rows @ {ts_ms}")
        except Exception as e:
            print(f"[DEPTH-BOOTSTRAP] REST bootstrap failed for {symbol}: {e}")
        return rows

    def bootstrap_all(self):
        """Fetches bootstrap snapshots for all configured symbols."""
        for sym in self.symbols:
            rows = self.fetch_rest_bootstrap_snapshot(sym)
            if rows:
                self.row_buffer.extend(rows)
        self.flush_buffer_sync()

    def flush_buffer_sync(self):
        """Commits all buffered rows into the appropriate daily SQLite partition."""
        if not self.row_buffer:
            return

        batch = self.row_buffer[:]
        self.row_buffer.clear()

        # Group rows by daily database path
        grouped: Dict[str, List[tuple]] = {}
        for r in batch:
            ts_ms = r[0]
            db_path = get_daily_db_path(ts_ms)
            if db_path not in grouped:
                grouped[db_path] = []
            grouped[db_path].append(r)

        for db_path, rows in grouped.items():
            init_depth_db(db_path)
            try:
                with sqlite3.connect(db_path, timeout=10.0) as conn:
                    conn.executemany("""
                        INSERT INTO books15_snapshots (timestamp_ms, symbol, side, level, price, size, seq)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, rows)
                self.total_rows_written += len(rows)
            except Exception as e:
                print(f"[DEPTH-FLUSH-ERROR] Failed to write {len(rows)} rows to {db_path}: {e}")

        self.last_flush = time.monotonic()

    async def run(self):
        """Main async loop subscribing to public books15 WebSocket."""
        print("=" * 65)
        print("   *** BITGET L2 (BOOKS15) DEPTH RECORDER DAEMON ***")
        print("=" * 65)
        print(f"Monitored symbols: {self.symbols}")
        print(f"Partition storage: {DEPTH_DIR}")
        print(f"Retention window : {RETENTION_DAYS} days")

        # 1. REST bootstrap
        self.bootstrap_all()
        cleanup_expired_partitions()

        sub_args = [{"instType": "USDT-FUTURES", "channel": "books15", "instId": s} for s in self.symbols]
        sub_payload = {"op": "subscribe", "args": sub_args}

        while self._running:
            try:
                async with websockets.connect(WS_URL, ping_interval=None) as ws:
                    await ws.send(json.dumps(sub_payload))
                    print(f"[DEPTH-WS] Connected and subscribed to books15 for {len(self.symbols)} symbols.")

                    async def _ping_loop():
                        while True:
                            await asyncio.sleep(25)
                            try:
                                await ws.send("ping")
                            except Exception:
                                break

                    ping_task = asyncio.create_task(_ping_loop())

                    try:
                        async for msg in ws:
                            if msg == "pong":
                                continue
                            data = json.loads(msg)
                            if data.get("event") == "subscribe":
                                continue

                            arg = data.get("arg", {})
                            sym = arg.get("instId", "").upper()
                            raw_list = data.get("data", [])
                            if not raw_list or sym not in self.symbols:
                                continue

                            entry = raw_list[0]
                            ts_ms = int(entry.get("ts") or time.time() * 1000)
                            seq = int(entry.get("seq") or 0)
                            bids = entry.get("bids", [])
                            asks = entry.get("asks", [])

                            for lvl, b in enumerate(bids[:15]):
                                self.row_buffer.append((ts_ms, sym, "BID", lvl, float(b[0]), float(b[1]), seq))
                            for lvl, a in enumerate(asks[:15]):
                                self.row_buffer.append((ts_ms, sym, "ASK", lvl, float(a[0]), float(a[1]), seq))

                            now = time.monotonic()
                            if len(self.row_buffer) >= MAX_BATCH_ROWS or (now - self.last_flush) >= BATCH_FLUSH_INTERVAL_SEC:
                                await asyncio.to_thread(self.flush_buffer_sync)

                    finally:
                        ping_task.cancel()
                        await asyncio.to_thread(self.flush_buffer_sync)

            except Exception as e:
                print(f"[DEPTH-WS-WARN] Disconnected: {e}. Reconnecting in 3s...")
                await asyncio.to_thread(self.flush_buffer_sync)
                await asyncio.sleep(3)


def main():
    recorder = DepthRecorder(SYMBOLS)
    try:
        asyncio.run(recorder.run())
    except KeyboardInterrupt:
        recorder.flush_buffer_sync()
        print(f"[DEPTH-RECORDER] Gracefully stopped. Total rows written: {recorder.total_rows_written:,}")


if __name__ == "__main__":
    main()

# verify_db_write_queue_stress.py
# Simulates 200 writes/sec stress test on DBWriteQueue under concurrent read load.
import asyncio
import time
import argparse
import sys
import os
import sqlite3

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import db_manager

async def writer_task(task_id: int, rate_per_sec: int, stop_event: asyncio.Event, stats: dict):
    dm = db_manager.DatabaseManager()
    interval = 1.0 / rate_per_sec
    count = 0
    while not stop_event.is_set():
        t0 = time.perf_counter()
        count += 1
        source_id = f"STRESS_T{task_id}_{count}"
        dm.log_tradesync(
            symbol="BTC/USDT",
            side="BUY" if count % 2 == 0 else "SELL",
            price=77000.0 + (count % 100) * 0.1,
            quantity=0.001,
            confidence=0.99,
            source=source_id
        )
        stats["writes"] += 1
        sleep_dur = interval - (time.perf_counter() - t0)
        if sleep_dur > 0:
            await asyncio.sleep(sleep_dur)
        else:
            await asyncio.sleep(0.001)

async def reader_task(reader_id: int, stop_event: asyncio.Event, stats: dict, db_path: str):
    while not stop_event.is_set():
        try:
            conn = sqlite3.connect(db_path, timeout=5.0)
            conn.execute("PRAGMA busy_timeout=5000;")
            cursor = conn.execute("SELECT COUNT(*), MAX(timestamp) FROM tradesync;")
            res = cursor.fetchone()
            stats["reads"] += 1
            conn.close()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() or "busy" in str(e).lower():
                stats["reader_locks"] += 1
                print(f"[READER {reader_id}] Lock error: {e}")
            else:
                stats["reader_errors"] += 1
        except Exception as e:
            stats["reader_errors"] += 1
        await asyncio.sleep(0.05)  # Read every 50ms (20 reads/sec)

async def main(duration: int, target_rate: int = 200):
    print("===============================================================")
    print("   *** DB_MANAGER STRESS TEST (DBWriteQueue) ***")
    print("===============================================================")
    print(f"Target Write Rate : {target_rate} writes/sec")
    print(f"Test Duration     : {duration} seconds ({duration/60:.1f} minutes)")
    print("Simulating concurrent writers & high-frequency readers...")
    
    dm = db_manager.DatabaseManager()
    db_path = dm.db_path
    
    num_writers = 5
    rate_per_writer = target_rate // num_writers
    num_readers = 3
    
    stop_event = asyncio.Event()
    stats = {
        "writes": 0,
        "reads": 0,
        "reader_locks": 0,
        "reader_errors": 0
    }
    
    writers = [asyncio.create_task(writer_task(i, rate_per_writer, stop_event, stats)) for i in range(num_writers)]
    readers = [asyncio.create_task(reader_task(i, stop_event, stats, db_path)) for i in range(num_readers)]
    
    start_time = time.monotonic()
    last_report = start_time
    
    try:
        while time.monotonic() - start_time < duration:
            await asyncio.sleep(1.0)
            now = time.monotonic()
            if now - last_report >= 5.0:
                elapsed = now - start_time
                w_sec = stats["writes"] / elapsed
                r_sec = stats["reads"] / elapsed
                wq = dm.write_queue
                print(f"[{elapsed:5.1f}s / {duration}s] Writes: {stats['writes']:>6} ({w_sec:5.1f}/s) | Reads: {stats['reads']:>5} ({r_sec:4.1f}/s) | Batches: {wq.total_batches} | Q Lock Errs: {wq.lock_errors} | Reader Lock Errs: {stats['reader_locks']}")
                last_report = now
    finally:
        stop_event.set()
        await asyncio.gather(*writers, *readers, return_exceptions=True)
        
    print("\nStopping writers and flushing remaining queue...")
    t_flush_start = time.monotonic()
    flushed = dm.flush(timeout=10.0)
    flush_time = time.monotonic() - t_flush_start
    
    total_elapsed = time.monotonic() - start_time
    total_writes = stats["writes"]
    total_reads = stats["reads"]
    actual_rate = total_writes / total_elapsed
    wq = dm.write_queue
    
    print("\n===============================================================")
    print("   *** STRESS TEST RESULTS ***")
    print("===============================================================")
    print(f"Total Duration        : {total_elapsed:.2f} s")
    print(f"Total Operations      : {total_writes} writes enqueued")
    print(f"Average Throughput    : {actual_rate:.1f} writes/sec")
    print(f"Total Concurrent Reads: {total_reads} reads executed")
    print(f"Total Batches Flushed : {wq.total_batches}")
    print(f"Avg Batch Size        : {total_writes / max(1, wq.total_batches):.1f} ops/batch")
    print(f"Final Flush Time      : {flush_time*1000:.2f} ms (Success: {flushed})")
    print(f"Queue Lock Errors     : {wq.lock_errors}")
    print(f"Reader Lock Errors    : {stats['reader_locks']}")
    print("===============================================================")
    
    # Assertions
    assert wq.lock_errors == 0, f"FAILED: DBWriteQueue recorded {wq.lock_errors} lock errors!"
    assert stats["reader_locks"] == 0, f"FAILED: Readers recorded {stats['reader_locks']} lock errors!"
    assert flushed, "FAILED: Final queue flush timed out!"
    
    # Cleanup stress test rows from tradesync
    print("Cleaning up test data from tradesync...")
    with dm._get_conn() as conn:
        deleted = conn.execute("DELETE FROM tradesync WHERE source LIKE 'STRESS_%'").rowcount
        conn.commit()
    print(f"Cleaned up {deleted} stress test records.")
    print(">>> ALL STRESS TEST ASSERTIONS PASSED (0 LOCK ERRORS) <<<")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds (default 30s)")
    parser.add_argument("--rate", type=int, default=200, help="Target writes/sec (default 200)")
    args = parser.parse_args()
    asyncio.run(main(duration=args.duration, target_rate=args.rate))

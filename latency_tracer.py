# latency_tracer.py — V22.0 Critical-Path Latency Profiler
#
# Wraps each stage of the Berserker order pipeline with microsecond-precision
# timing. Writes segment-by-segment results to the log and flags any stage
# that exceeds its budget with log.critical().
#
# Usage (async context manager):
#   async with LatencyTracer("AI_AUDIT", budget_ms=5000) as t:
#       result = await ai_engine.get_risk_params(...)

import asyncio
import logging
import time
import sqlite3
import json
from contextlib import asynccontextmanager
from datetime import datetime, UTC

log = logging.getLogger("latency_tracer")

# ---------------------------------------------------------------------------
# Per-segment budget table (milliseconds)
# ---------------------------------------------------------------------------
SEGMENT_BUDGETS_MS = {
    "INDICATOR_COMPUTE":    50,
    "DB_WRITE_METRICS":     20,
    "RL_STATE_TENSOR":      30,
    "MACRO_REGIME":         10,
    "AI_AUDIT":         180000,   # Groq network call — hard 180s deadline
    "TICKER_FETCH":        200,   # Binance REST ticker
    "SENTIMENT_READ":       10,
    "MACRO_BREAKER":       200,
    "SQLITE_SENTIMENT":     15,
    "BOARDROOM":        180000,   # 3-LLM debate — hard 180s deadline
    "BALANCE_FETCH":       200,
    "SENTINEL_AUDIT":       50,   # Pre-cached — must be in-memory
    "ORDER_PLACEMENT":     100,
    "TOTAL_CRITICAL_PATH": 300,   # 🎯 The SLA
}

# ---------------------------------------------------------------------------
# In-process trace accumulator (per-candle, reset on each invocation)
# ---------------------------------------------------------------------------
_active_trace: dict = {}

def reset_trace():
    """Call at the start of each handle_candle_close invocation."""
    global _active_trace
    _active_trace = {}

def get_trace() -> dict:
    return dict(_active_trace)


@asynccontextmanager
async def LatencyTracer(segment_name: str, budget_ms: float | None = None):
    """
    Async context manager that measures wall-clock time for a named segment.
    Logs a WARNING if budget is breached, CRITICAL if > 2× budget.

    Args:
        segment_name: One of the SEGMENT_BUDGETS_MS keys, or any custom name.
        budget_ms: Override budget. Defaults to SEGMENT_BUDGETS_MS lookup.
    """
    budget = budget_ms or SEGMENT_BUDGETS_MS.get(segment_name, 1000)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _active_trace[segment_name] = round(elapsed_ms, 2)

        if elapsed_ms > budget * 2:
            log.critical(
                f"[LATENCY] ⛔ {segment_name}: {elapsed_ms:.0f}ms "
                f"(budget={budget}ms — exceeded by {elapsed_ms - budget:.0f}ms)"
            )
        elif elapsed_ms > budget:
            log.warning(
                f"[LATENCY] ⚠️  {segment_name}: {elapsed_ms:.0f}ms "
                f"(budget={budget}ms — over by {elapsed_ms - budget:.0f}ms)"
            )
        else:
            log.debug(f"[LATENCY] ✅ {segment_name}: {elapsed_ms:.0f}ms / {budget}ms budget")


def log_full_trace(symbol: str, db_path: str = "trading_ledger.db"):
    """
    Call once at the end of the critical path to persist the complete
    segment breakdown to the DB and emit a summary log line.
    """
    trace = get_trace()
    total = sum(trace.values())
    summary = " | ".join(f"{k}={v:.0f}ms" for k, v in trace.items())
    sla = SEGMENT_BUDGETS_MS.get("TOTAL_CRITICAL_PATH", 300)

    if total > sla:
        log.warning(f"[LATENCY-REPORT] {symbol} TOTAL={total:.0f}ms > SLA={sla}ms | {summary}")
    else:
        log.info(f"[LATENCY-REPORT] {symbol} TOTAL={total:.0f}ms ✅ | {summary}")

    # Persist to DB for dashboard visibility (best-effort, non-blocking)
    import threading
    def _persist():
        try:
            conn = sqlite3.connect(db_path, timeout=2)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS latency_logs "
                "(timestamp TEXT, symbol TEXT, total_ms REAL, segments TEXT)"
            )
            conn.execute(
                "INSERT INTO latency_logs VALUES (?, ?, ?, ?)",
                (datetime.now(UTC).isoformat(), symbol, round(total, 2), json.dumps(trace))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            log.debug(f"[LATENCY] DB persist failed (non-critical): {e}")
            
    threading.Thread(target=_persist, daemon=True).start()

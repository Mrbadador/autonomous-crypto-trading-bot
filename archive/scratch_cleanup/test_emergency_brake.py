import sqlite3
import asyncio
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] LIVE_TRADER: %(message)s")
log = logging.getLogger("live_trader")

def query_sentiment_metrics():
    try:
        with sqlite3.connect("trading_ledger.db", timeout=15.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            cursor = conn.execute("SELECT fear_greed_index FROM market_context ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            fg = float(row[0]) if row and row[0] is not None else 50.0
            
            cursor = conn.execute("SELECT spy_health FROM macro_logs ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            spy = float(row[0]) if row and row[0] is not None else 0.0
            
            return fg, spy
    except Exception as e:
        log.warning(f"[MACRO-GUARD] Failed to query sentiment metrics: {e}")
        return 50.0, 0.0

async def test():
    fg, spy = await asyncio.to_thread(query_sentiment_metrics)
    is_extreme_panic = (fg < 25.0) or (spy < -0.6)
    
    action_name = "LONG"
    print(f"Checking trade: {action_name} | F&G={fg} | SPY={spy}")
    
    if action_name == "LONG" and is_extreme_panic:
        log.critical(f"[EMERGENCY-BRAKE] LONG trade vetoed! Macro Sentiment is in EXTREME PANIC. F&G={fg:.1f}, SPY_Health={spy:.2f}. Forcing HOLD/CASH state.")
        return False
    return True

if __name__ == "__main__":
    result = asyncio.run(test())
    print("Execution allowed:" if result else "Execution BLOCKED.")

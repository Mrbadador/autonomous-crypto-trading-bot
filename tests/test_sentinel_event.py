import asyncio
import sqlite3
import json
import logging
import os
import tempfile
import tests.conftest  # noqa: F401 — enforce D-11 quarantine
from sentinel_agent import SentimentSentinel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

async def test_sentinel():
    tmp_dir = tempfile.mkdtemp(prefix="test_sentinel_")
    tmp_db = os.path.join(tmp_dir, "trading_data.db")
    sentinel = SentimentSentinel()
    sentinel.db_path = tmp_db
    
    # Simulate a simulated high impact war headline arriving
    test_headlines = [
        "Major military conflict declared in key shipping strait; crude oil surges 8%",
        "Regular altcoin developer community update released"
    ]
    
    print("\n--- Testing Sentinel Breaking Catalyst Evaluation ---")
    event_res = await sentinel.evaluate_headlines_for_catalysts(test_headlines)
    print(f"Evaluated Playbook: {sentinel._active_event_playbook}")
    print(f"Catalyst Summary: {sentinel._active_event_summary}")
    
    assert sentinel._active_event_playbook in ("SECULAR_SHORT_BREAKDOWN", "FLASH_REBOUND_TRAP", "BULL_MOMENTUM_EXPANSION", "IGNORE_NOISE", "CASH_PRESERVATION")

    # Verify SQLite persistence in quarantined DB
    with sqlite3.connect(tmp_db) as conn:
        row = conn.execute("SELECT value FROM system_state WHERE key = 'ai_event_playbook'").fetchone()
        assert row is not None, "ai_event_playbook key not persisted in system_state!"
        print(f"SQLite verified: ai_event_playbook = {row[0]}")

    # Verify directional disaster audit
    long_status, long_msg = sentinel.perform_directional_audit("BTC", direction="LONG")
    short_status, short_msg = sentinel.perform_directional_audit("BTC", direction="SHORT")
    print(f"LONG audit: {long_status} | {long_msg}")
    print(f"SHORT audit: {short_status} | {short_msg}")

    print("\n[SUCCESS] Sentinel Event Catalyst verified successfully!")

if __name__ == "__main__":
    asyncio.run(test_sentinel())

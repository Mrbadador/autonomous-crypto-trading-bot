import asyncio
import sqlite3
from swing_trader import SwingArchitectTrader

async def test_swing_event_safety():
    architect = SwingArchitectTrader()

    # Case 1: Secular Bear Event Active
    with sqlite3.connect("trading_data.db") as conn:
        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('macro_global_killswitch', '0', 'now')")
        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_macro_bias', 'BEARISH', 'now')")
        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('market_regime_label', 'NEUTRAL', 'now')")
        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_event_playbook', 'SECULAR_SHORT_BREAKDOWN', 'now')")
        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_event_summary', 'War outbreak causes sustained energy shock', 'now')")
        conn.commit()

    long_ok, long_rsn = await architect._check_macro_safety(direction="LONG")
    short_ok, short_rsn = await architect._check_macro_safety(direction="SHORT")
    print("\nCase 1 (SECULAR_SHORT_BREAKDOWN):")
    print(f"LONG ok: {long_ok} | Reason: {long_rsn}")
    print(f"SHORT ok: {short_ok} | Reason: {short_rsn}")
    assert long_ok is False, "Longs should be halted during SECULAR_SHORT_BREAKDOWN"
    assert short_ok is True, "Shorts should be authorized during SECULAR_SHORT_BREAKDOWN"

    # Case 2: Flash Rebound Trap (Liquidation cascade)
    with sqlite3.connect("trading_data.db") as conn:
        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_event_playbook', 'FLASH_REBOUND_TRAP', 'now')")
        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_event_summary', 'Exchange FUD rumor causes 15m flash dump', 'now')")
        conn.commit()

    long_ok_2, long_rsn_2 = await architect._check_macro_safety(direction="LONG")
    short_ok_2, short_rsn_2 = await architect._check_macro_safety(direction="SHORT")
    print("\nCase 2 (FLASH_REBOUND_TRAP):")
    print(f"LONG ok: {long_ok_2} | Reason: {long_rsn_2}")
    print(f"SHORT ok: {short_ok_2} | Reason: {short_rsn_2}")
    assert short_ok_2 is False, "Shorts should be blocked during FLASH_REBOUND_TRAP to avoid squeeze"

    # Clean up state to baseline
    with sqlite3.connect("trading_data.db") as conn:
        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_event_playbook', 'NORMAL_OPERATIONS', 'now')")
        conn.commit()

    print("\n[SUCCESS] Swing Architect Event-Driven Gates verified!")

if __name__ == "__main__":
    asyncio.run(test_swing_event_safety())

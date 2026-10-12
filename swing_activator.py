"""
Swing Activator: Autonomous Transition Watcher
==============================================
Monitors `marl_live_open_positions` in `trading_data.db`.
The exact moment the active BTC trade closes:
1. Detects position close and records final exit PnL from `marl_live_trades`.
2. Flips `SWING_TRADER_ENABLED=true` in `.env`.
3. Sets `SWING_TRADER_ENABLED = 1` in `trading_data.db` `system_state`.
4. Spawns `swing_trader.py` as an autonomous background daemon.
5. Emits transition log telemetry.
6. Gracefully exits.
"""

import os
import sys
import time
import sqlite3
import subprocess
import logging
from datetime import datetime, UTC

# Setup Logging
log = logging.getLogger("swing_activator")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] swing_activator: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/swing_activator.log", encoding="utf-8")
    ]
)

DB_PATH = "trading_data.db"
ENV_PATH = ".env"

def check_marl_btc_open():
    try:
        with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM marl_live_open_positions WHERE symbol LIKE '%BTC%'").fetchone()
            return dict(row) if row else None
    except Exception as e:
        log.error(f"Error checking open positions: {e}")
        return "ERROR"

def get_last_closed_trade():
    try:
        with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM marl_live_trades ORDER BY id DESC LIMIT 1").fetchone()
            return dict(row) if row else None
    except Exception:
        return None

def update_env_to_enabled():
    try:
        if not os.path.exists(ENV_PATH):
            return
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        if "SWING_TRADER_ENABLED=false" in content:
            new_content = content.replace("SWING_TRADER_ENABLED=false", "SWING_TRADER_ENABLED=true")
        elif "SWING_TRADER_ENABLED" in content:
            import re
            new_content = re.sub(r"SWING_TRADER_ENABLED\s*=\s*\w+", "SWING_TRADER_ENABLED=true", content)
        else:
            new_content = content + "\nSWING_TRADER_ENABLED=true\n"

        with open(ENV_PATH, "w", encoding="utf-8") as f:
            f.write(new_content)
        log.info("[ACTIVATOR] Updated .env: SWING_TRADER_ENABLED=true")
    except Exception as e:
        log.error(f"[ACTIVATOR] Error updating .env: {e}")

def update_db_system_state():
    now_iso = datetime.now(UTC).isoformat()
    try:
        with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO system_state (key, value, updated_at)
                VALUES ('SWING_TRADER_ENABLED', '1', ?)
            """, (now_iso,))
            conn.commit()
        log.info("[ACTIVATOR] Updated system_state: SWING_TRADER_ENABLED = 1")
    except Exception as e:
        log.error(f"[ACTIVATOR] Error updating system_state: {e}")

def launch_swing_trader():
    log.info("[ACTIVATOR] Spawning swing_trader.py process...")
    cmd = [sys.executable, "swing_trader.py"]
    try:
        flags = 0
        if os.name == 'nt':
            flags = subprocess.CREATE_NEW_PROCESS_GROUP
        p = subprocess.Popen(cmd, creationflags=flags)
        log.info(f"[ACTIVATOR] swing_trader.py launched successfully (PID: {p.pid})!")
        return p.pid
    except Exception as e:
        log.error(f"[ACTIVATOR] Failed to spawn swing_trader.py: {e}")
        return None

def main():
    log.info("=" * 65)
    log.info("   SWING ACTIVATOR: AUTONOMOUS TRANSITION WATCHER ENGAGED")
    log.info("   Target: Monitor active BTC trade in marl_live_open_positions")
    log.info("   Trigger: On trade close -> Enable & launch swing_trader.py")
    log.info("=" * 65)

    # Initial check
    initial_pos = check_marl_btc_open()
    if initial_pos and initial_pos != "ERROR":
        entry_p = initial_pos.get("entry_price")
        tp = initial_pos.get("take_profit")
        sl = initial_pos.get("stop_loss")
        log.info(f"[WATCHING] Active BTC Trade found: Entry=${entry_p:,.2f} | TP=${tp:,.2f} | SL=${sl:,.2f}")
    else:
        log.warning("[WATCHING] No active BTC trade found on boot! Will monitor continuously...")

    zero_count = 0
    heartbeat_counter = 0

    while True:
        try:
            pos = check_marl_btc_open()
            if pos == "ERROR":
                time.sleep(5)
                continue

            if pos is not None:
                zero_count = 0
                heartbeat_counter += 1
                if heartbeat_counter % 60 == 0:  # Every 5 minutes
                    peak_p = pos.get("peak_price", pos.get("entry_price"))
                    log.info(f"[HEARTBEAT] BTC trade active. Entry: ${pos.get('entry_price'):,.2f} | Peak: ${peak_p:,.2f}")
            else:
                # Position is None
                zero_count += 1
                log.info(f"[PROBE] BTC position not found in open positions (Confirmation {zero_count}/2)...")
                if zero_count >= 2:
                    # Confirmed closed!
                    log.info("=" * 65)
                    log.info("   [EVENT DETECTED] ACTIVE BTC TRADE HAS CLOSED!")
                    log.info("=" * 65)

                    last_trade = get_last_closed_trade()
                    if last_trade:
                        log.info(f"[CLOSED TRADE DETAILS] PnL: ${last_trade.get('pnl', 0):+,.2f} | Exit: ${last_trade.get('exit_price', 0):,.2f} | Type: {last_trade.get('type')}")

                    # 1. Update .env
                    update_env_to_enabled()

                    # 2. Update database system_state
                    update_db_system_state()

                    # 3. Launch swing_trader.py
                    pid = launch_swing_trader()

                    log.info("=" * 65)
                    log.info(f"   THE SWING ARCHITECT IS NOW LIVE (PID: {pid})")
                    log.info("   Autonomous transition complete. Watcher exiting.")
                    log.info("=" * 65)
                    break

            time.sleep(5)
        except Exception as loop_e:
            log.error(f"[WATCHER-ERROR] Unexpected error: {loop_e}")
            time.sleep(5)

if __name__ == "__main__":
    main()

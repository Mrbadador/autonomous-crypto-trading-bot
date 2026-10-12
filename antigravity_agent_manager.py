# antigravity_agent_manager.py
import os
import sys
import json
import time
import asyncio
import logging
import sqlite3
import subprocess
from datetime import datetime, UTC, timezone, timedelta
import websockets
from regime_monitor import run_regime_monitor  # [V32.7] Hurst/ADX regime shift alerts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] AGENT_MANAGER: %(message)s")
log = logging.getLogger("agent_manager")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH  = os.path.join(BASE_DIR, "launcher.log")
MAIN_LOG_PATH = os.path.join(BASE_DIR, "logs", "main.log")  # [V32.6] Watchdog feed
RULES_HANDLER = os.path.join(BASE_DIR, "antigravity_rules_handler.py")

# [V32.6] Re-bind main.log FileHandler on daemon start.
# main.py's FileHandler was severed during the host crash.
# The agent manager now feeds main.log directly so the
# antigravity_rules_handler watchdog always has a live stream.
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
_main_fh = logging.FileHandler(MAIN_LOG_PATH, mode="a", encoding="utf-8", errors="replace")
_main_fh.setLevel(logging.INFO)
_main_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
log.addHandler(_main_fh)  # agent_manager logger now dual-writes: console + main.log

# Port 9812 connection clients
CONNECTED_CLIENTS = set()

async def ws_handler(websocket):
    CONNECTED_CLIENTS.add(websocket)
    log.info(f"New dashboard connection from {websocket.remote_address}")
    try:
        # Send initial registration message
        await websocket.send(json.dumps({
            "title": "SYSTEM_INFO",
            "content": "Anti-Gravity Autonomous Swarm Manager Live. Dashboard connected."
        }))
        async for message in websocket:
            # We just keep it open, no complex incoming protocols needed
            pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        CONNECTED_CLIENTS.discard(websocket)
        log.info(f"Dashboard disconnected: {websocket.remote_address}")

async def broadcast_telemetry(title: str, content: str):
    if not CONNECTED_CLIENTS:
        return
    payload = json.dumps({
        "title": title,
        "content": content,
        "timestamp": datetime.now(UTC).isoformat()
    })
    websockets_to_remove = set()
    for ws in CONNECTED_CLIENTS:
        try:
            await ws.send(payload)
        except Exception:
            websockets_to_remove.add(ws)
    for ws in websockets_to_remove:
        CONNECTED_CLIENTS.discard(ws)

_rules_running = False
_last_rules_invocation_ts = 0.0

async def monitor_logs_and_trigger_rules():
    global _rules_running, _last_rules_invocation_ts
    # Prefer active swarm log logs/main.log, falling back to launcher.log
    watch_log_path = MAIN_LOG_PATH if os.path.exists(MAIN_LOG_PATH) else LOG_PATH
    log.info(f"Starting background log watcher on: {watch_log_path}")

    last_size = 0
    if os.path.exists(watch_log_path):
        last_size = os.path.getsize(watch_log_path)

    critical_triggers = [
        "[WAL-BREACH]",
        "[PRIORITY-CRITICAL]",
        "database is locked",
        "OperationalError: disk I/O error",
        "[CRITICAL]",
        "Traceback (most recent call last)",
    ]

    while True:
        await asyncio.sleep(5)
        # Dynamically follow logs/main.log if it appears
        if not os.path.exists(watch_log_path) and os.path.exists(MAIN_LOG_PATH):
            watch_log_path = MAIN_LOG_PATH
            last_size = os.path.getsize(watch_log_path)

        if not os.path.exists(watch_log_path):
            continue

        try:
            current_size = os.path.getsize(watch_log_path)
            if current_size < last_size:
                # Log rotated or truncated
                last_size = 0

            if current_size > last_size:
                with open(watch_log_path, "r", encoding="utf-8", errors="ignore") as f:
                    f.seek(last_size)
                    new_data = f.read()
                last_size = current_size

                # Check for genuine critical triggers (excludes healthy [WAL-WATCHDOG] and generic ERROR)
                matched_kw = [kw for kw in critical_triggers if kw in new_data]
                if matched_kw:
                    now_ts = time.time()
                    if _rules_running:
                        log.info("Rules Handler already running. Debouncing concurrent invocation.")
                    elif (now_ts - _last_rules_invocation_ts) < 60.0:
                        log.debug(f"Rules Handler on 60s cooldown ({now_ts - _last_rules_invocation_ts:.1f}s elapsed). Skipping.")
                    else:
                        _rules_running = True
                        _last_rules_invocation_ts = now_ts
                        log.warning(f"Critical trigger detected in log ({', '.join(matched_kw)}). Invoking Rules Handler...")

                        def run_handler(target_file: str):
                            global _rules_running
                            try:
                                subprocess.run(
                                    [sys.executable, RULES_HANDLER, "--log", target_file, "--analyze"],
                                    cwd=BASE_DIR,
                                    capture_output=True,
                                    text=True,
                                    timeout=30.0,
                                )
                            except Exception as sub_e:
                                log.warning(f"Rules Handler subprocess execution note: {sub_e}")
                            finally:
                                _rules_running = False

                        await asyncio.to_thread(run_handler, watch_log_path)
                        await broadcast_telemetry("RULES_TRIGGERED", f"Rules analysis executed on log breach ({', '.join(matched_kw)}).")
        except Exception as watch_e:
            log.debug(f"Log watcher loop note: {watch_e}")

async def send_periodic_heartbeat():
    while True:
        await asyncio.sleep(10)
        # Check DB stats for telemetry
        db_path = os.path.join(BASE_DIR, "trading_data.db")
        stats_msg = "Sovereign Swarm active."
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path, timeout=5.0)
                r = conn.execute("SELECT COUNT(*) FROM marl_live_trades").fetchone()
                stats_msg = f"Sovereign Swarm Live. Closed trades count: {r[0]}"
                conn.close()
            except Exception as e:
                stats_msg = f"Sovereign Swarm active. Telemetry err: {e}"
        # [V32.6] Write heartbeat to main.log via the agent_manager logger
        # (which now has a dual FileHandler). This keeps the watchdog feed alive
        # even when main.py's own FileHandler is severed.
        log.info(f"[HEARTBEAT] {stats_msg}")
        await broadcast_telemetry("HEARTBEAT", stats_msg)


async def run_6h_email_cron():
    """
    [V32.6] 6-Hour Heartbeat Email Cron.
    Fires at 00:00, 06:00, 12:00, and 18:00 UTC by sleeping until the next
    6-hour boundary, then dispatching via email_notifier.send_6h_heartbeat_async().
    Falls back gracefully if email_notifier is unavailable.
    """
    log.info("[6H-CRON] 6-hour email heartbeat cron registered.")

    def _seconds_to_next_slot() -> float:
        """Seconds until the next 00/06/12/18 UTC boundary."""
        now   = datetime.now(timezone.utc)
        # Round up to the next 6-hour slot
        slot  = (now.hour // 6 + 1) * 6
        if slot >= 24:
            # Crosses midnight
            next_dt = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        else:
            next_dt = now.replace(hour=slot, minute=0, second=0, microsecond=0)
        secs = (next_dt - now).total_seconds()
        return max(secs, 0.0)

    while True:
        wait_secs = _seconds_to_next_slot()
        slot_time  = datetime.now(timezone.utc) + timedelta(seconds=wait_secs)
        log.info(
            f"[6H-CRON] Next heartbeat email scheduled at "
            f"{slot_time.strftime('%Y-%m-%d %H:%M UTC')} "
            f"(in {wait_secs/3600:.2f}h / {wait_secs:.0f}s)"
        )

        await asyncio.sleep(wait_secs)

        # Dispatch — offloaded to a thread so SMTP never blocks the event loop
        log.info("[6H-CRON] Dispatching 6-hour heartbeat email...")
        await broadcast_telemetry("6H_HEARTBEAT", "Generating and dispatching 6-hour email report...")
        try:
            from email_notifier import send_6h_heartbeat_async
            ok = await send_6h_heartbeat_async()
            status = "DISPATCHED" if ok else "DISPATCH FAILED (check GMAIL credentials)"
        except ImportError:
            ok     = False
            status = "SKIPPED (email_notifier.py not found)"
        except Exception as e:
            ok     = False
            status = f"ERROR: {e}"

        log.info(f"[6H-CRON] Email heartbeat result: {status}")
        await broadcast_telemetry("6H_HEARTBEAT_RESULT", status)

        # Safety buffer: sleep 60s after dispatch to avoid double-fire on boundary
        await asyncio.sleep(60)

async def run_morning_briefing_cron():
    """
    Daily Morning Briefing Cron.
    Fires daily at 04:15 UTC (06:15 local time) by generating and emailing morning_briefing.
    """
    log.info("[BRIEFING-CRON] Daily Morning Briefing cron registered.")
    
    def _seconds_to_morning_slot() -> float:
        now = datetime.now(timezone.utc)
        target = now.replace(hour=4, minute=0, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        return max((target - now).total_seconds(), 0.0)

    while True:
        wait_secs = _seconds_to_morning_slot()
        slot_time = datetime.now(timezone.utc) + timedelta(seconds=wait_secs)
        log.info(f"[BRIEFING-CRON] Next morning briefing scheduled at {slot_time.strftime('%Y-%m-%d %H:%M UTC')} (in {wait_secs/3600:.2f}h)")
        await asyncio.sleep(wait_secs)
        
        log.info("[BRIEFING-CRON] Generating and dispatching daily morning briefing...")
        try:
            from morning_briefing import generate_briefing, send_morning_briefing_email, BRIEFING_PATH
            briefing = await asyncio.to_thread(generate_briefing)
            with open(BRIEFING_PATH, "w", encoding="utf-8") as f:
                f.write(briefing)
            await asyncio.to_thread(send_morning_briefing_email, briefing)
            log.info("[BRIEFING-CRON] Morning briefing successfully dispatched.")
        except Exception as e:
            log.error(f"[BRIEFING-CRON] Failed to dispatch morning briefing: {e}")
            
        await asyncio.sleep(60)

async def start_ws_server():
    log.info("Initializing WebSocket telemetry server on ws://localhost:9812...")
    async with websockets.serve(ws_handler, "localhost", 9812):
        await asyncio.Future() # run forever

async def main():
    # Start tasks
    server_task    = asyncio.create_task(start_ws_server())
    monitor_task   = asyncio.create_task(monitor_logs_and_trigger_rules())
    heartbeat_task = asyncio.create_task(send_periodic_heartbeat())
    cron_6h_task   = asyncio.create_task(run_6h_email_cron())       # [V32.6] 6-hour email cron
    briefing_task  = asyncio.create_task(run_morning_briefing_cron()) # Daily Morning Briefing cron
    regime_task    = asyncio.create_task(run_regime_monitor())       # [V32.7] Hurst/ADX regime shift alerts

    log.info("All Anti-Gravity autonomous manager services running.")
    log.info("[6H-CRON] Heartbeat email pipeline active (fires at 00:00, 06:00, 12:00, 18:00 UTC).")
    log.info("[BRIEFING-CRON] Daily Morning Briefing active (fires at 04:15 UTC / 06:15 Local).")
    log.info("[REGIME-MONITOR] Hurst/ADX watchdog active (gate: Hurst>0.55 & ADX>25).")
    await asyncio.gather(server_task, monitor_task, heartbeat_task, cron_6h_task, briefing_task, regime_task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Agent Manager shutdown.")

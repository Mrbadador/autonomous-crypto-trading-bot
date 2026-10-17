# start_system.py — Swarm Unified Subprocess Orchestrator

import sys
import os
import time
import subprocess
import signal
import logging
import pathlib
import threading
import atexit as _atexit
from datetime import datetime, timezone
from dotenv import load_dotenv

# ENV CUSTODIAN: Load .env before spawning any subprocesses.
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)

def _get_script_identity(cmd):
    py_file = None
    for arg in cmd:
        if arg.endswith(".py"):
            py_file = arg
            break
    if py_file:
        extra = []
        for arg in cmd:
            if arg.startswith("--"):
                extra.append(arg.lstrip("-"))
        if extra:
            return f"{pathlib.Path(py_file).stem}_{'_'.join(extra)}.py"
        return py_file
    return cmd[-1] if "run" in cmd else cmd[1]

def _get_log_stream(cmd):
    log_dir = pathlib.Path("logs")
    log_dir.mkdir(exist_ok=True)
    identity = _get_script_identity(cmd)
    stem = pathlib.Path(identity).stem
    return open(log_dir / f"{stem}.log", "a", encoding="utf-8")

# DNS FIX: Patch ccxt's Exchange.open() to use ThreadedResolver globally.
# Must be imported after load_dotenv and before any ccxt usage.
import dns_fix  # noqa: F401
import config

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] LAUNCHER_SYSTEM: %(message)s")
log = logging.getLogger("launcher_system")

# ENV INTEGRITY CHECK
_bitget_key = os.getenv("BITGET_API_KEY", "")
_bitget_secret = os.getenv("BITGET_API_SECRET", "")
if _bitget_key and _bitget_secret:
    log.info(f"[ENV] START_SYSTEM ENV OK — BITGET_API_KEY loaded (last 4: ...{_bitget_key[-4:]}).")
else:
    log.critical("[ENV] START_SYSTEM FATAL: BITGET_API_KEY or BITGET_API_SECRET is EMPTY. Check .env file! Aborting launch.")
    sys.exit(1)

class ManagedProcess:
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None
        self.restarts = 0
        self.restart_at = 0.0  # Epoch timestamp for when to restart
        self._last_successful_start = 0.0
        self.log_file = None

managed = []

def _checkpoint_databases():
    """
    [DD-01 FIX] Best-effort WAL TRUNCATE checkpoint on both active databases.
    Called by shutdown_handler after all subprocesses are terminated to ensure
    the WAL journals are in a clean, read-consistent state before exit.
    Any failure is logged but does NOT block process termination.
    """
    import sqlite3 as _sqlite3
    for db in ["trading_data.db", "trading_ledger.db"]:
        try:
            _c = _sqlite3.connect(db, timeout=5.0)
            _c.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            _c.close()
            log.info(f"[SHUTDOWN] WAL checkpoint TRUNCATE completed for {db}.")
        except Exception as _e:
            log.warning(f"[SHUTDOWN] WAL checkpoint failed for {db}: {_e}. WAL will be replayed on next boot.")

_briefing_cycle_count = 0
_briefing_lock = threading.Lock()

def _run_in_process_briefing():
    """Generates the autonomous 5-minute executive briefing directly in-process without extra daemons."""
    global _briefing_cycle_count
    with _briefing_lock:
        _briefing_cycle_count += 1
        curr_cycle = _briefing_cycle_count
    try:
        import importlib
        import executive_briefing
        importlib.reload(executive_briefing)
        log.info(f"[BRIEFING] Generating autonomous in-process 5-minute briefing (Cycle #{curr_cycle})...")
        payload = executive_briefing.run_single_briefing_cycle(curr_cycle)
        log.info(f"[BRIEFING] Briefing Cycle #{curr_cycle} completed & updated in SQLite.")
    except Exception as e:
        log.error(f"[BRIEFING] In-process briefing error (Cycle #{curr_cycle}): {e}")

_shepherd_lock = threading.Lock()

def _run_in_process_shepherd(force: bool = False):
    """Runs continuous in-flight position shepherding every 15 mins whenever positions are open."""
    with _shepherd_lock:
        try:
            import importlib
            import inflight_shepherd
            importlib.reload(inflight_shepherd)
            log.info("[SHEPHERD] Running autonomous in-process 15-minute in-flight position shepherd...")
            reports = inflight_shepherd.run_single_shepherd_cycle(force=force)
            if reports:
                log.info(f"[SHEPHERD] In-flight evaluation completed for {len(reports)} active positions.")
            else:
                log.debug("[SHEPHERD] Zero open positions; 100% cash shield intact.")
        except Exception as e:
            log.error(f"[SHEPHERD] In-process shepherding error: {e}")

_rs_lock = threading.Lock()

def _run_in_process_relative_strength(force: bool = False):
    """Runs continuous 30-minute Multi-Asset Relative Strength Matrix directly in-process."""
    with _rs_lock:
        try:
            import importlib
            import relative_strength_matrix
            importlib.reload(relative_strength_matrix)
            log.info("[RELATIVE-STRENGTH] Running autonomous in-process 30-minute fleet matrix evaluation...")
            res = relative_strength_matrix.run_single_relative_strength_cycle(force=force)
            if res:
                leaders = res.get("priority_leaders", [])
                log.info(f"[RELATIVE-STRENGTH] Relative strength updated. Leaders: {leaders}")
            else:
                log.debug("[RELATIVE-STRENGTH] Cycle completed with no change.")
        except Exception as e:
            log.error(f"[RELATIVE-STRENGTH] In-process matrix error: {e}")

_hourly_lock = threading.Lock()

def _run_in_process_hourly_report(force: bool = False):
    """Generates and dispatches the 1-hour executive email rollup summarizing the 5m cycles."""
    with _hourly_lock:
        try:
            import importlib
            import hourly_briefing
            importlib.reload(hourly_briefing)
            log.info("[HOURLY-BRIEFING] Generating autonomous in-process 1-hour executive email rollup...")
            success = hourly_briefing.run_single_hourly_briefing(force=force)
            if success:
                log.info("[HOURLY-BRIEFING] 1-hour executive email rollup dispatched successfully.")
            else:
                log.warning("[HOURLY-BRIEFING] Hourly executive email dispatch returned false.")
        except Exception as e:
            log.error(f"[HOURLY-BRIEFING] In-process hourly briefing error: {e}")

def get_depth_health(now_ts=None, depth_dir=None, date_str=None):
    """
    FIX 3 — supervision-facing depth staleness check.

    Delegates the read-only probe to depth_recorder.read_depth_health() and routes the
    result through the launcher logger: WARNING when age > 15 min, ERROR when age > 60 min.
    Returns the structured result dict (status/age_sec/rows/snapshots/db_path).
    """
    import depth_recorder
    res = depth_recorder.read_depth_health(now_ts=now_ts, depth_dir=depth_dir, date_str=date_str)
    status = res.get("status")
    age = res.get("age_sec")
    if status == "MISSING":
        log.error(f"[DEPTH-HEALTH] ERROR: today's depth partition is missing: {res['db_path']}")
    elif status == "READ_ERROR":
        log.error(f"[DEPTH-HEALTH] ERROR: cannot read depth partition {res['db_path']}: {res.get('error')}")
    elif status == "EMPTY":
        log.warning("[DEPTH-HEALTH] WARNING: depth partition exists but holds zero snapshot rows.")
    elif status == "STALE_ERROR":
        log.error(
            f"[DEPTH-HEALTH] ERROR: books15 depth stream stale {age/60.0:.1f} min (>60min) "
            f"— recorder likely down (rows={res['rows']}, snapshots={res['snapshots']})."
        )
    elif status == "STALE_WARN":
        log.warning(
            f"[DEPTH-HEALTH] WARNING: depth stream lagging {age/60.0:.1f} min (>15min) "
            f"(rows={res['rows']}, snapshots={res['snapshots']})."
        )
    else:
        log.info(f"[DEPTH-HEALTH] OK: depth stream fresh (age {age:.0f}s); rows={res['rows']}, snapshots={res['snapshots']}.")
    return res

import psutil
import socket

_SINGLE_INSTANCE_PORT = 49199
_instance_socket = None
_shutdown_in_progress = False

def _recycle_existing_orchestrator(my_pid):
    """Gracefully recycles any running start_system.py orchestrator and child daemons."""
    log.info("[RECYCLE] Initiating graceful process recycling of running fleet...")
    
    target_names = [
        "start_system.py", "antigravity_agent_manager.py", "ofi_daemon.py", "main.py", "live_trader.py",
        "marl_live_trader.py", "mean_reversion_trader.py", "live_oracle.py",
        "regime_classifier.py", "portfolio_risk_agent.py", "post_mortem_agent.py",
        "macro_sentinel.py", "asset_scout.py", "recovery_agent.py", "reconciliation_engine.py",
        "db_maintenance_daemon.py", "prometheus_exporter.py", "basis_trader.py", "swing_trader.py",
        "super_bot.py", "executive_briefing.py", "depth_recorder.py"
    ]
    
    procs_to_terminate = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if p.pid == my_pid:
                continue
            cmdline = " ".join(p.info.get('cmdline') or [])
            if any(t in cmdline for t in target_names) or ("streamlit" in cmdline and "dashboard.py" in cmdline):
                procs_to_terminate.append(p)
        except Exception:
            pass
            
    if procs_to_terminate:
        log.info(f"[RECYCLE] Signaling {len(procs_to_terminate)} running processes to flush active state and terminate gracefully...")
        for p in procs_to_terminate:
            try:
                p.terminate()
            except Exception:
                pass
                
        # Wait up to 5 seconds for graceful shutdown
        _, alive = psutil.wait_procs(procs_to_terminate, timeout=5)
        for p in alive:
            try:
                log.warning(f"[RECYCLE] Process {p.pid} did not exit gracefully, enforcing termination...")
                p.kill()
            except Exception:
                pass
        time.sleep(1)
        
    # Checkpoint WAL journals to guarantee clean state
    _checkpoint_databases()
    log.info("[RECYCLE] Graceful process recycling and WAL checkpoint complete.")

def _enforce_single_instance():
    """Guarantees strictly ONE instance of start_system.py runs using atomic loopback socket bind."""
    global _instance_socket
    my_pid = os.getpid()
    
    try:
        _instance_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Prevent SO_REUSEADDR on Windows to ensure strict mutual exclusion
        if sys.platform != "win32":
            _instance_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        _instance_socket.bind(("127.0.0.1", _SINGLE_INSTANCE_PORT))
        _instance_socket.listen(1)
        log.info(f"[ORCHESTRATOR] Acquired single-instance execution lock on port {_SINGLE_INSTANCE_PORT} (PID {my_pid}).")
    except Exception as e:
        log.warning(f"[RECYCLE] Port {_SINGLE_INSTANCE_PORT} is bound by an active start_system.py orchestrator ({e}).")
        log.info("[RECYCLE] Executing process recycling to update running fleet bytecode...")
        _recycle_existing_orchestrator(my_pid)
        
        # Wait up to 5s for the port to be fully released
        acquired = False
        for attempt in range(10):
            try:
                _instance_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                if sys.platform != "win32":
                    _instance_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
                _instance_socket.bind(("127.0.0.1", _SINGLE_INSTANCE_PORT))
                _instance_socket.listen(1)
                log.info(f"[ORCHESTRATOR] Acquired single-instance execution lock on port {_SINGLE_INSTANCE_PORT} (PID {my_pid}) after recycling.")
                acquired = True
                break
            except Exception:
                time.sleep(0.5)
                
        if not acquired:
            log.critical(f"[RECYCLE FATAL] Port {_SINGLE_INSTANCE_PORT} still bound after recycling attempt. Aborting.")
            sys.exit(1)
            
    # Cleanup any stale/orphaned child processes from previous crashed sessions
    _cleanup_stale_children(my_pid)

def _cleanup_stale_children(my_pid):
    """Terminates orphaned trading bot daemons to ensure a clean slate."""
    target_names = [
        "antigravity_agent_manager.py", "ofi_daemon.py", "main.py", "live_trader.py",
        "marl_live_trader.py", "mean_reversion_trader.py", "live_oracle.py",
        "regime_classifier.py", "portfolio_risk_agent.py", "post_mortem_agent.py",
        "macro_sentinel.py", "asset_scout.py", "recovery_agent.py", "reconciliation_engine.py",
        "db_maintenance_daemon.py", "prometheus_exporter.py", "basis_trader.py",
        "swing_trader.py", "super_bot.py", "executive_briefing.py", "depth_recorder.py"
    ]
    cleaned = 0
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if p.pid == my_pid:
                continue
            cmdline = " ".join(p.info.get('cmdline') or [])
            if any(t in cmdline for t in target_names) or ("streamlit" in cmdline and "dashboard.py" in cmdline):
                log.info(f"[CLEANUP] Terminating orphaned process (PID {p.pid}): {cmdline[:60]}...")
                p.kill()
                cleaned += 1
        except Exception:
            pass
    if cleaned > 0:
        log.info(f"[CLEANUP] Cleaned {cleaned} orphaned processes before launch.")
        time.sleep(1)

def shutdown_handler(signum, frame):
    global _shutdown_in_progress
    if _shutdown_in_progress:
        return
    _shutdown_in_progress = True

    """Graceful termination handler closing all active subprocess streams on Ctrl+C."""
    log.warning("[SYSTEM] Shutdown signal received. Terminating all active processes...")
    for mp in managed:
        if mp.process and mp.process.poll() is None:
            log.info(f"[SYSTEM] Terminating process (PID {mp.process.pid})...")
            mp.process.terminate()
            try:
                mp.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                log.warning(f"[SYSTEM] Process (PID {mp.process.pid}) did not exit. Forcing kill.")
                mp.process.kill()
        if mp.log_file:
            try:
                mp.log_file.close()
            except Exception:
                pass
    # [DD-01 FIX] Checkpoint WAL journals after all subprocesses are terminated.
    _checkpoint_databases()
    
    # Release single-instance socket
    global _instance_socket
    if _instance_socket:
        try:
            _instance_socket.close()
        except Exception:
            pass
        _instance_socket = None
            
    log.info("[SYSTEM] All background agents halted. Session complete.")
    if signum is not None:
        sys.exit(0)

# Register shutdown hooks
signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)
_atexit.register(shutdown_handler, None, None)

def _spawn_managed_process(mp):
    if mp.log_file:
        try:
            mp.log_file.close()
        except Exception:
            pass
    mp.log_file = _get_log_stream(mp.cmd)
    mp.process = subprocess.Popen(mp.cmd, env=os.environ, stdout=mp.log_file, stderr=mp.log_file)

def main():
    _enforce_single_instance()
    log.info("=" * 60)
    log.info("     LAUNCHING V15.0 SWARM ECOSYSTEM COMMAND CENTER")
    log.info("=" * 60)

    # Spawn background processes sequentially with staggered delays
    subprocess_configs = [
        [sys.executable, "antigravity_agent_manager.py"],
        [sys.executable, "ofi_daemon.py"],
    ]

    # Scalper Fleet Gate: Suspended under Swing-Only Consolidation to stop taker fee drag
    if getattr(config, "SCALPERS_ENABLED", False):
        log.info("[FLEET-ROUTER] ⚡ Intraday scalper fleet ENABLED.")
        subprocess_configs.extend([
            [sys.executable, "main.py"],
            [sys.executable, "main.py", "--berserker"],
            [sys.executable, "live_trader.py"],
            [sys.executable, "marl_live_trader.py"],
        ])
    else:
        log.info("[FLEET-ROUTER] 🛡️ Scalpers deactivated by Swing-Only Consolidation policy. High-frequency taker friction eliminated.")

    # Unified Super Bot Dual-Regime Engine (Consolidates 4H Swing + Chop Maker Grid)
    if getattr(config, "SUPER_BOT_ENABLED", True):
        log.info("[FLEET-ROUTER] ⚡ Super Bot Dual-Regime Engine ENABLED (Bull Swing + Chop Maker Grid).")
        subprocess_configs.append([sys.executable, "super_bot.py"])
    elif getattr(config, "SWING_TRADER_ENABLED", True):
        log.info("[FLEET-ROUTER] ⚡ Swing Architect Engine ENABLED.")
        subprocess_configs.append([sys.executable, "swing_trader.py"])

    # Core Swarm Telemetry, Risk & Macro daemons
    subprocess_configs.extend([
        # [sys.executable, "mean_reversion_trader.py"],     # Suspended: 0 trades placed
        [sys.executable, "live_oracle.py"],
        [sys.executable, "regime_classifier.py"],
        [sys.executable, "portfolio_risk_agent.py"],
        [sys.executable, "post_mortem_agent.py"],
        [sys.executable, "macro_sentinel.py"],
        [sys.executable, "asset_scout.py"],
        [sys.executable, "recovery_agent.py"],
        [sys.executable, "reconciliation_engine.py"],
        [sys.executable, "db_maintenance_daemon.py"],
        [sys.executable, "prometheus_exporter.py"],
    ])
    # FIX 3: supervised books15 L2 depth recorder (zero-order public-data daemon).
    # Accrues queue-model calibration snapshots (TRIAL-003 target: >=50,000 across >=14 days).
    if getattr(config, "DEPTH_RECORDER_ENABLED", True):
        log.info("[FLEET-ROUTER] 📡 books15 depth recorder ENABLED (supervised; auto-restart + backoff).")
        subprocess_configs.append([sys.executable, "depth_recorder.py"])
    else:
        log.info("[FLEET-ROUTER] 📡 books15 depth recorder disabled via DEPTH_RECORDER_ENABLED=false.")
    if getattr(config, "BASIS_TRADER_ENABLED", False):
        subprocess_configs.append([sys.executable, "basis_trader.py"])
    subprocess_configs.append([
        sys.executable, "-m", "streamlit", "run", "dashboard.py",
        "--server.port", "8501",
        "--server.address", "127.0.0.1",
        "--server.headless", "true"
    ])

    global managed
    managed = [ManagedProcess(cmd) for cmd in subprocess_configs]

    for mp in managed:
        script_name = _get_script_identity(mp.cmd)
        log.info(f"Initializing subprocess for {script_name}...")
        
        try:
            # Spawn non-blocking subprocess with explicit env to guarantee .env propagation
            _spawn_managed_process(mp)
            mp._last_successful_start = time.time()
            log.info(f"Subprocess {script_name} started successfully (PID {mp.process.pid}).")
        except Exception as e:
            log.critical(f"Failed to start subprocess {script_name}: {e}")
            shutdown_handler(None, None)
            
        # V32.1: Enforce staggered 4-second start delay between nodes (upgraded from 2s).
        # Root cause of WAL 6.52ms SLA breach: simultaneous 7-node state reconciliation
        # burst during ignition phase exceeded disk IO capacity within the 2s window.
        time.sleep(4)

    log.info("=" * 60)
    log.info("     ALL background subprocesses are running.")
    log.info("     Watching active streams. Press Ctrl+C to terminate.")
    log.info("=" * 60)

    _STABLE_RUNTIME_THRESHOLD_SECS = 300  # 5 minutes stable = fresh restart budget
    _last_briefing_time = time.time()
    _last_shepherd_time = time.time()
    _last_relative_strength_time = time.time()
    _last_hourly_report_time = time.time()
    # FIX 3: fire the depth staleness health check immediately on boot, then every 5 minutes.
    _last_depth_health_time = time.time() - 300.0
    try:
        import sqlite3
        conn_h = sqlite3.connect("trading_ledger.db", timeout=3.0)
        row_h = conn_h.execute("SELECT generated_at FROM hourly_briefing_history ORDER BY id DESC LIMIT 1").fetchone()
        conn_h.close()
        if row_h and row_h[0]:
            ts_str = row_h[0].replace(" UTC", "").strip()
            last_dt = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
            secs_since_last = (datetime.now(timezone.utc) - last_dt).total_seconds()
            if secs_since_last < 3600.0:
                _last_hourly_report_time = time.time() - secs_since_last
            else:
                _last_hourly_report_time = time.time() - 3540.0  # Trigger 60s after boot
        else:
            _last_hourly_report_time = time.time() - 3540.0
    except Exception as h_init_err:
        log.debug(f"[ORCHESTRATOR] Note initializing hourly report schedule: {h_init_err}")
        _last_hourly_report_time = time.time()
    # Trigger autonomous background threads on boot
    threading.Thread(target=_run_in_process_briefing, daemon=True, name="in_process_briefing").start()
    threading.Thread(target=_run_in_process_shepherd, kwargs={"force": False}, daemon=True, name="in_process_shepherd").start()
    threading.Thread(target=_run_in_process_relative_strength, kwargs={"force": False}, daemon=True, name="in_process_relative_strength").start()

    while True:
        try:
            now = time.time()
            for mp in managed:
                script_name = _get_script_identity(mp.cmd)
                if mp.process is None:
                    # Never started or failed initial launch
                    if mp.restart_at > 0.0 and now >= mp.restart_at:
                        log.info(f"[RESTART] Retrying initial launch for {script_name}...")
                        try:
                            _spawn_managed_process(mp)
                            mp.restart_at = 0.0
                            mp._last_successful_start = now
                            log.info(f"Subprocess {script_name} started successfully (PID {mp.process.pid}).")
                        except Exception as e:
                            log.error(f"Retry initial launch failed for {script_name}: {e}")
                            mp.restart_at = now + 10.0
                elif mp.process.poll() is not None:
                    # Process exited
                    if mp.restart_at == -1.0:
                        continue
                    if _get_script_identity(mp.cmd) == "basis_trader.py" and not getattr(config, "BASIS_TRADER_ENABLED", False):
                        log.info(f"[SUPERVISOR] {script_name} is disabled via BASIS_TRADER_ENABLED=false. De-registering process cleanly.")
                        mp.restart_at = -1.0
                        continue
                    if _get_script_identity(mp.cmd) == "swing_trader.py" and not getattr(config, "SWING_TRADER_ENABLED", False):
                        log.info(f"[SUPERVISOR] {script_name} is disabled via SWING_TRADER_ENABLED=false. De-registering process cleanly.")
                        mp.restart_at = -1.0
                        continue
                    if _get_script_identity(mp.cmd) == "depth_recorder.py" and not getattr(config, "DEPTH_RECORDER_ENABLED", True):
                        log.info(f"[SUPERVISOR] {script_name} is disabled via DEPTH_RECORDER_ENABLED=false. De-registering process cleanly.")
                        mp.restart_at = -1.0
                        continue
                    if mp.restart_at == 0.0:
                        if mp.restarts >= 15:
                            log.critical(f"[SYSTEM] Subprocess {script_name} has exceeded the maximum restart limit (15). HALTING restart attempts.")
                            mp.restart_at = -1.0
                            continue
                        backoff = min(60.0, 5.0 * (2 ** mp.restarts))
                        mp.restart_at = now + backoff
                        log.error(f"[SYSTEM] Subprocess {script_name} (PID {mp.process.pid}) exited unexpectedly. Scheduling restart in {backoff:.1f}s (attempt {mp.restarts + 1}).")
                    elif now >= mp.restart_at:
                        log.info(f"[RESTART] Restarting {script_name} (attempt {mp.restarts + 1})...")
                        try:
                            _spawn_managed_process(mp)
                            mp.restarts += 1
                            mp.restart_at = 0.0
                            mp._last_successful_start = now
                            log.info(f"Subprocess {script_name} restarted successfully (PID {mp.process.pid}).")
                        except Exception as e:
                            log.error(f"Failed to restart subprocess {script_name}: {e}")
                            mp.restart_at = now + 10.0  # retry in 10s
                else:
                    # [DD-03 OPT] Process is alive. Reset restart counter if it has been
                    # stable long enough so the next crash starts fresh on backoff budget.
                    _last_start = getattr(mp, '_last_successful_start', 0)
                    if (_last_start > 0 and
                            now - _last_start > _STABLE_RUNTIME_THRESHOLD_SECS and
                            mp.restarts > 0):
                        mp.restarts = 0
                        log.debug(f"[RESTART] Reset backoff counter for {script_name} after stable run.")

            # Autonomous in-process 5-minute executive briefing check
            if now - _last_briefing_time >= 300.0:
                _last_briefing_time = now
                threading.Thread(target=_run_in_process_briefing, daemon=True, name="in_process_briefing").start()

            # FIX 3: books15 depth-recorder staleness health check (every 5 min; WARNING >15min, ERROR >60min)
            if now - _last_depth_health_time >= 300.0:
                _last_depth_health_time = now
                get_depth_health()

            # Autonomous in-process 15-minute in-flight position shepherd check
            if now - _last_shepherd_time >= 900.0:
                _last_shepherd_time = now
                threading.Thread(target=_run_in_process_shepherd, kwargs={"force": False}, daemon=True, name="in_process_shepherd").start()

            # Autonomous in-process 30-minute Multi-Asset Relative Strength Matrix check
            if now - _last_relative_strength_time >= 1800.0:
                _last_relative_strength_time = now
                threading.Thread(target=_run_in_process_relative_strength, kwargs={"force": False}, daemon=True, name="in_process_relative_strength").start()

            # Autonomous in-process 1-hour executive briefing email rollup check
            if now - _last_hourly_report_time >= 3600.0:
                _last_hourly_report_time = now
                threading.Thread(target=_run_in_process_hourly_report, kwargs={"force": False}, daemon=True, name="in_process_hourly_report").start()

            time.sleep(1)
        except KeyboardInterrupt:
            log.info("[SHUTDOWN] Keyboard interrupt received.")
            shutdown_handler(None, None)
            break
        except Exception as loop_e:
            log.error(f"[MONITOR] Unexpected exception in monitoring loop: {loop_e}", exc_info=True)
            time.sleep(2)

if __name__ == "__main__":
    main()

"""
run_fleet_detached.py — Independent Windows Process Spawner for Sovereign Swarm
Launches start_system.py in a fully detached process group on Windows so that it
continues running 24/7 even if interactive subagent or tool sessions terminate.
"""

import os
import sys
import subprocess
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] DETACHED_LAUNCHER: %(message)s")
log = logging.getLogger("detached_launcher")

def launch_detached():
    venv_python = sys.executable
    cmd = [venv_python, "start_system.py"]
    
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
    
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    out_log = open(os.path.join(log_dir, "start_system_detached.log"), "a", encoding="utf-8")
    
    proc = subprocess.Popen(
        cmd,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=out_log,
        stderr=out_log,
        creationflags=flags,
        close_fds=True
    )
    
    log.info(f"[SUCCESS] Sovereign Swarm detached with Master PID: {proc.pid}")
    log.info("Output log stream: logs/start_system_detached.log")
    return proc.pid

if __name__ == "__main__":
    launch_detached()

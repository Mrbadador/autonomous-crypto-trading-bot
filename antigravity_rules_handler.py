# antigravity_rules_handler.py
import os
import re
import sys
import logging
import sqlite3
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] RULES_HANDLER: %(message)s")
log = logging.getLogger("rules_handler")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_FILE = os.path.join(BASE_DIR, "logs", "main.log")


def check_weights():
    log.info("Starting weight-divergence check...")
    policy_path = os.path.join(BASE_DIR, "policy_model.pth")
    strat_path = os.path.join(BASE_DIR, "strategist_weights_v26.pth")
    if not os.path.exists(policy_path) or not os.path.exists(strat_path):
        log.info("Weight files not present. Skipping check.")
        return

    t_policy = os.path.getmtime(policy_path)
    t_strat = os.path.getmtime(strat_path)
    sz_policy = os.path.getsize(policy_path)
    sz_strat = os.path.getsize(strat_path)

    log.info(f"policy_model.pth: Size={sz_policy} bytes, Time={t_policy}")
    log.info(f"strategist_weights_v26.pth: Size={sz_strat} bytes, Time={t_strat}")

    if abs(sz_policy - sz_strat) > 1024 or abs(t_policy - t_strat) > 86400:
        log.warning("Weight divergence detected between policy_model.pth and strategist_weights_v26.pth!")
    else:
        log.info("Weights are synchronized.")


def run_db_cleanup():
    log.info("Executing I/O repair: triggering passive checkpoint/vacuum...")
    try:
        from db_manager import DatabaseManager
        db_mgr = DatabaseManager()
        db_mgr.checkpoint_wal("PASSIVE")
    except Exception as e:
        log.debug(f"db_manager checkpoint note: {e}")

    for db_name in ["trading_data.db", "trading_ledger.db"]:
        p = os.path.join(BASE_DIR, db_name)
        if os.path.exists(p):
            try:
                conn = sqlite3.connect(p, timeout=5.0)
                res = conn.execute("PRAGMA wal_checkpoint(PASSIVE);").fetchone()
                log.info(f"Checkpoint on {db_name}: busy={res[0]}, log={res[1]}, checkpointed={res[2]}")
                conn.close()
            except Exception as ex:
                log.warning(f"Failed checkpoint on {db_name}: {ex}")


def analyze_logs(log_path: str):
    if not os.path.exists(log_path):
        log.warning(f"Log file not found: {log_path}")
        return

    log.info(f"Analyzing log file: {log_path}")
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()[-100:]

    io_pattern = re.compile(r"IO WAIT|database is locked|\[WAL-BREACH\]|OperationalError: disk I/O", re.IGNORECASE)
    logic_pattern = re.compile(r"Traceback \(most recent call last\)|Uncaught exception|AssertionError|SyntaxError", re.IGNORECASE)

    detected_io_lines = []
    detected_logic_lines = []

    for line in lines:
        if io_pattern.search(line):
            detected_io_lines.append(line.strip())
        if logic_pattern.search(line):
            detected_logic_lines.append(line.strip())

    if detected_io_lines:
        log.warning(f"[REMEDIATION] I/O issue detected ({len(detected_io_lines)} occurrences). Sample: '{detected_io_lines[-1][:120]}'")
        run_db_cleanup()

    if detected_logic_lines:
        log.warning(f"[REMEDIATION] Uncaught exception detected ({len(detected_logic_lines)} occurrences). Sample: '{detected_logic_lines[-1][:120]}'")
        check_weights()

    if not detected_io_lines and not detected_logic_lines:
        log.info("Log scan complete: no active I/O locks or unhandled crash signatures found.")


def main():
    parser = argparse.ArgumentParser(description="Antigravity Rules Handler Watchdog")
    parser.add_argument("--log", dest="log_path", default=None, help="Target log file to analyze")
    parser.add_argument("--analyze", action="store_true", help="Analyze recent log entries and take corrective actions")
    args = parser.parse_args()

    if args.analyze:
        target = args.log_path
        if not target or not os.path.exists(target):
            if os.path.exists(DEFAULT_LOG_FILE):
                target = DEFAULT_LOG_FILE
            elif os.path.exists(os.path.join(BASE_DIR, "launcher.log")):
                target = os.path.join(BASE_DIR, "launcher.log")
            else:
                target = DEFAULT_LOG_FILE

        analyze_logs(target)
    else:
        log.info("Rules handler loaded. Pass --analyze to run log analysis.")


if __name__ == "__main__":
    main()

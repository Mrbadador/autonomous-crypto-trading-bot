# rebind_mainlog.py — V32.6 Hot-Patch: Re-bind logs/main.log FileHandler
#
# Problem: After the host crash, the root logger's FileHandler for main.log
#          holds a broken fd (file was 0-byte truncated by the OS). New log
#          entries are written to stdout only — main.log stays silent.
#
# Fix: Find the stale FileHandler in the root logger's handler chain,
#      close it gracefully, and replace it with a fresh FileHandler pointing
#      to the same path. Then emit a test heartbeat to confirm the stream is live.
#
# This script is safe to run while the swarm is live — it only touches the
# Python logging module state and never touches the exchange connection.

import os
import sys
import logging
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_LOG_PATH = os.path.join(BASE_DIR, "logs", "main.log")
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

FORMATTER = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def rebind_file_handler(log_path: str, logger_name: str = None) -> bool:
    """
    Replaces any stale/broken FileHandler on the specified logger
    (or the root logger if logger_name is None) with a fresh one.
    Safe to call on a live logger — acquires the handler lock during swap.
    """
    target = logging.getLogger(logger_name)  # None → root logger

    # ── 1. Identify and close stale FileHandlers ──────────────────────────────
    stale = [h for h in target.handlers if isinstance(h, logging.FileHandler)]

    if stale:
        for h in stale:
            print(f"  [REBIND] Closing stale FileHandler: {h.baseFilename}")
            h.acquire()
            try:
                h.flush()
                h.close()
            finally:
                h.release()
            target.removeHandler(h)
        print(f"  [REBIND] Removed {len(stale)} stale FileHandler(s).")
    else:
        print("  [REBIND] No existing FileHandlers found -- attaching fresh handler.")

    # ── 2. Attach fresh FileHandler ───────────────────────────────────────────
    try:
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8", errors="backslashreplace", delay=False)
        fh.setFormatter(FORMATTER)
        fh.setLevel(logging.INFO)
        target.addHandler(fh)
        print(f"  [REBIND] Fresh FileHandler attached: {log_path}")
        return True
    except Exception as e:
        print(f"  [REBIND ERROR] Failed to attach FileHandler: {e}")
        return False


def verify_stream(log_path: str) -> bool:
    """Emit a timestamped heartbeat and confirm it lands in the file."""
    root = logging.getLogger()
    ts   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    root.info(f"[V32.6-REBIND] main.log FileHandler re-bound successfully at {ts}. Heartbeat stream restored.")

    # Flush all handlers
    for h in root.handlers:
        h.flush()

    # Read back and verify
    if os.path.exists(log_path):
        size = os.path.getsize(log_path)
        if size > 0:
            with open(log_path, "r", encoding="utf-8") as f:
                last_line = f.readlines()[-1].strip()
            print(f"  [VERIFY] main.log size: {size} bytes")
            print(f"  [VERIFY] Last entry: {last_line[:100]}")
            return True
        else:
            print("  [VERIFY FAIL] File is still 0 bytes after emit.")
            return False
    print("  [VERIFY FAIL] File does not exist.")
    return False


if __name__ == "__main__":
    print("=" * 60)
    print("  V32.6 — main.log FileHandler Re-bind")
    print("=" * 60)

    # ── Step 1: Configure root logger if it has no level set ─────────────────
    root_logger = logging.getLogger()
    if root_logger.level == logging.NOTSET:
        root_logger.setLevel(logging.INFO)

    # Ensure a StreamHandler exists (stdout safety net)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root_logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(FORMATTER)
        root_logger.addHandler(sh)

    # ── Step 2: Re-bind the FileHandler on the root logger ───────────────────
    print(f"\n[1/3] Scanning root logger handlers...")
    print(f"      Current handlers: {[type(h).__name__ for h in root_logger.handlers]}")
    ok = rebind_file_handler(MAIN_LOG_PATH, logger_name=None)

    # ── Step 3: Also patch the 'launcher' named logger ────────────────────────
    print(f"\n[2/3] Patching 'launcher' named logger...")
    launcher_log = logging.getLogger("launcher")
    launcher_stale = [h for h in launcher_log.handlers if isinstance(h, logging.FileHandler)]
    for h in launcher_stale:
        print(f"      Closing stale launcher FileHandler → {h.baseFilename}")
        h.close()
        launcher_log.removeHandler(h)
    fh2 = logging.FileHandler(MAIN_LOG_PATH, mode="a", encoding="utf-8")
    fh2.setFormatter(FORMATTER)
    launcher_log.addHandler(fh2)
    print(f"      Fresh FileHandler attached to 'launcher' logger.")

    # ── Step 4: Verify the stream is live ────────────────────────────────────
    print(f"\n[3/3] Verifying heartbeat stream...")
    verified = verify_stream(MAIN_LOG_PATH)

    # ── Result ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if ok and verified:
        print("  RESULT: [SUCCESS] main.log FileHandler is LIVE.")
        print(f"  Stream path: {MAIN_LOG_PATH}")
    else:
        print("  RESULT: [PARTIAL] Handler attached but verify step failed.")
        print("  Recommendation: Restart antigravity_agent_manager.py")
    print("=" * 60 + "\n")

    sys.exit(0 if (ok and verified) else 1)

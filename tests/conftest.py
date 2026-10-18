"""
Quarantine & Isolation Fixture for Test Suite (D-11 / P2-9).
============================================================
1. Blocks any import of `start_system` (which registers a module-scope `_atexit`
   handler that kills the live fleet on interpreter exit).
2. Redirects all relative and repo-root SQLite database connections
   (`trading_data.db`, `trading_ledger.db`, `live_paper_trades.db`,
   `paper_trading.db`, `test_ledger.db`, `market_depth.db`) and state JSON files
   (`microstructure_state.json`, `sentiment_state.json`, `volatility_state.json`)
   to an isolated per-session temporary directory, seeded via read-only SQLite
   backup from the production schema/state, and rejects any direct write
   attempt targeting the live production repository paths.
"""

import builtins
import importlib.abc
import os
import shutil
import sqlite3
import sys
import tempfile

PROJECT_ROOT = os.path.normcase(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
PROJECT_ROOT_RAW = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in [os.path.normcase(os.path.abspath(p)) for p in sys.path]:
    sys.path.insert(0, PROJECT_ROOT_RAW)

QUARANTINED_DB_BASENAMES = frozenset({
    "trading_data.db",
    "trading_ledger.db",
    "live_paper_trades.db",
    "paper_trading.db",
    "test_ledger.db",
    "market_depth.db",
})

QUARANTINED_STATE_BASENAMES = frozenset({
    "microstructure_state.json",
    "sentiment_state.json",
    "volatility_state.json",
})

_SESSION_TEMP_DIR = tempfile.mkdtemp(prefix="llama_test_quarantine_")
_ORIG_SQLITE_CONNECT = sqlite3.connect
_ORIG_OPEN = builtins.open
_ORIG_COPYFILE = shutil.copyfile


import atexit

def _seed_temp_files():
    """
    Creates a consistent read-only snapshot of production DBs and state JSONs
    into _SESSION_TEMP_DIR using sqlite3 online backup with mode=ro URI.
    Production databases are NEVER opened writable.
    Fails loudly if a seed backup fails so tests never run against an empty schema.
    """
    for db_name in QUARANTINED_DB_BASENAMES:
        prod_path = os.path.join(PROJECT_ROOT_RAW, db_name)
        dst_path = os.path.join(_SESSION_TEMP_DIR, db_name)
        if os.path.exists(prod_path) and not os.path.exists(dst_path):
            try:
                ro_uri = f"file:{prod_path.replace(os.sep, '/')}?mode=ro"
                src_conn = _ORIG_SQLITE_CONNECT(ro_uri, uri=True, timeout=5.0)
                dst_conn = _ORIG_SQLITE_CONNECT(dst_path, timeout=5.0)
                with dst_conn:
                    src_conn.backup(dst_conn)
                dst_conn.close()
                src_conn.close()
            except Exception as e:
                raise RuntimeError(f"[QUARANTINE-SEED-ERROR] Failed to seed temp DB for {db_name}: {e}") from e

    for st_name in QUARANTINED_STATE_BASENAMES:
        prod_st = os.path.join(PROJECT_ROOT_RAW, st_name)
        dst_st = os.path.join(_SESSION_TEMP_DIR, st_name)
        if os.path.exists(prod_st) and not os.path.exists(dst_st):
            try:
                _ORIG_COPYFILE(prod_st, dst_st)
            except Exception as e:
                raise RuntimeError(f"[QUARANTINE-SEED-ERROR] Failed to seed temp state for {st_name}: {e}") from e


def _cleanup_session_temp_dir():
    """Removes the session quarantine temporary directory on interpreter exit."""
    if os.path.exists(_SESSION_TEMP_DIR):
        try:
            shutil.rmtree(_SESSION_TEMP_DIR, ignore_errors=True)
        except Exception:
            pass


atexit.register(_cleanup_session_temp_dir)
_seed_temp_files()


class _BlockStartSystemFinder(importlib.abc.MetaPathFinder):
    """Prevents importing start_system.py from any test or helper."""

    def find_spec(self, fullname, path, target=None):
        if fullname == "start_system" or fullname.startswith("start_system."):
            raise RuntimeError(
                "CRITICAL FLEET-KILL TRAP BLOCKED (D-11): "
                "Importing 'start_system' is forbidden in tests because "
                "start_system.py registers a module-scope _atexit shutdown_handler "
                "that kills the live trading fleet on interpreter exit."
            )
        return None


if not any(isinstance(f, _BlockStartSystemFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _BlockStartSystemFinder())


def _is_repo_root_target(raw_path: str, protected_basenames: frozenset) -> tuple[bool, str]:
    """
    Returns (is_protected_repo_target, basename) for a path string.
    Matches both bare relative filenames ('trading_data.db') when CWD is PROJECT_ROOT
    and explicit paths pointing directly inside PROJECT_ROOT.
    Paths pointing inside a temporary directory (or when CWD is a temporary directory)
    are permitted to use that temporary directory directly.
    """
    if not isinstance(raw_path, (str, bytes, os.PathLike)):
        return False, ""
    path_str = os.fsdecode(raw_path)
    if path_str == ":memory:" or path_str.startswith("file::memory:"):
        return False, ""

    clean_str = path_str
    if clean_str.startswith("file:"):
        clean_str = clean_str[5:].split("?")[0]

    bname = os.path.basename(clean_str).lower()
    matched_base = ""
    if bname in protected_basenames:
        matched_base = bname
    else:
        for pb in protected_basenames:
            if bname.startswith(pb + "."):
                matched_base = bname
                break

    if not matched_base:
        return False, ""

    # Resolve the directory of clean_str relative to current working directory
    resolved_dir = os.path.normcase(os.path.abspath(os.path.dirname(clean_str) or os.getcwd()))
    if resolved_dir == PROJECT_ROOT:
        return True, matched_base

    return False, matched_base


def _quarantined_sqlite_connect(database, *args, **kwargs):
    """
    Intercepts sqlite3.connect calls:
    - Redirects any PROJECT_ROOT production DB filename (including bare relative
      'trading_data.db' when CWD == PROJECT_ROOT) to _SESSION_TEMP_DIR.
    - Leaves explicit temp-dir paths (and relative paths when CWD is a temp dir)
      untouched so per-test temp DBs work naturally.
    """
    if isinstance(database, (str, bytes, os.PathLike)):
        db_str = os.fsdecode(database)
        is_uri = bool(kwargs.get("uri", False)) or db_str.startswith("file:")
        is_prod_target, bname = _is_repo_root_target(db_str, QUARANTINED_DB_BASENAMES)
        if is_prod_target:
            redirected = os.path.join(_SESSION_TEMP_DIR, bname)
            if is_uri and db_str.startswith("file:"):
                query_part = ("?" + db_str.split("?", 1)[1]) if "?" in db_str else ""
                redirected_uri = f"file:{redirected.replace(os.sep, '/')}{query_part}"
                return _ORIG_SQLITE_CONNECT(redirected_uri, *args, **kwargs)
            return _ORIG_SQLITE_CONNECT(redirected, *args, **kwargs)

    return _ORIG_SQLITE_CONNECT(database, *args, **kwargs)


sqlite3.connect = _quarantined_sqlite_connect


def _quarantined_open(file, mode="r", *args, **kwargs):
    """
    Redirects state JSON reads/writes targeting PROJECT_ROOT into _SESSION_TEMP_DIR,
    and blocks any write to a protected DB in PROJECT_ROOT.
    """
    if isinstance(file, (str, bytes, os.PathLike)):
        f_str = os.fsdecode(file)
        is_state, state_bname = _is_repo_root_target(f_str, QUARANTINED_STATE_BASENAMES)
        if is_state:
            file = os.path.join(_SESSION_TEMP_DIR, state_bname)
        else:
            is_db, db_bname = _is_repo_root_target(f_str, QUARANTINED_DB_BASENAMES)
            if is_db and any(m in str(mode) for m in ("w", "a", "+", "x")):
                file = os.path.join(_SESSION_TEMP_DIR, db_bname)
    return _ORIG_OPEN(file, mode, *args, **kwargs)


builtins.open = _quarantined_open


def _quarantined_copyfile(src, dst, *args, **kwargs):
    """Prevents shutil.copyfile from touching live DBs in PROJECT_ROOT."""
    for p in (src, dst):
        is_db, _ = _is_repo_root_target(str(p), QUARANTINED_DB_BASENAMES)
        if is_db:
            src_redir = os.path.join(_SESSION_TEMP_DIR, os.path.basename(str(src)))
            dst_redir = os.path.join(_SESSION_TEMP_DIR, os.path.basename(str(dst)))
            if not os.path.exists(src_redir):
                _ORIG_SQLITE_CONNECT(src_redir).close()
            return _ORIG_COPYFILE(src_redir, dst_redir, *args, **kwargs)
    return _ORIG_COPYFILE(src, dst, *args, **kwargs)


shutil.copyfile = _quarantined_copyfile

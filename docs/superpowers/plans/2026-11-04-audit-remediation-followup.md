# Audit Remediation Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the six defects that survived the `59bdd91` P0 remediation, verified by audit against HEAD `59bdd91`.

**Architecture:** All six fixes are local, self-contained code changes in existing modules. No new subsystems, no new dependencies, no schema redesign. Four of the six restore a broken invariant that `59bdd91` disturbed (lock acquisition, backtest DB isolation, health signal, breakeven trigger); one moves a database invariant from an out-of-band SQL command into version-controlled schema code; one repairs a module that `59bdd91` left unimportable.

**Tech Stack:** Python 3.14 (`.venv`), `sqlite3` stdlib, `pytest` 9.1.1, `unittest.IsolatedAsyncioTestCase`, Bitget UTA via `ccxt`.

**Spec:** This document. Every finding below was verified by direct code inspection, a reproduced exception, or a live `pytest` run — not inferred from the `59bdd91` commit message.

## Global Constraints

- Interpreter for all commands: `.venv/Scripts/python.exe -m pytest ...` (bare `pytest` collects a vendored interpreter at `Python/` and crashes — 174 collection errors).
- Never open `trading_data.db` in write mode during investigation. Use `sqlite3.connect('file:trading_data.db?mode=ro', uri=True)`.
- Every task's test must be able to go RED. A test that cannot fail proves nothing — verify the red state before implementing.
- Line endings: these files are CRLF. Preserve them; a whole-file rewrite to LF produces a 4000-line phantom diff.
- `docs/superpowers/plans/` is the plan location. Do not commit this plan alongside a fix commit.
- **No task in this plan touches EC2 or the production ledger.** Production steps are in the Runbook at the end and require explicit operator approval.

## Baseline

| Metric | `573b391` (pre-remediation) | `59bdd91` (current HEAD) |
|---|---|---|
| Full suite | 10 failed, 1532 passed | **5 failed, 1537 passed** |
| `tests/trades` rows (local) | 366 (122 dup groups) | 244 (deduped, index added — DB file only) |
| `tests/trades` rows (EC2) | 369 | 369 (unchanged — dedup never reached production) |

The 3 failures introduced by `59bdd91` are all in the rewritten entry path and share one root cause (Task 1).

---

### Task 1: Fix the pre-order asset-lock `IntegrityError` regression

**Files:**
- Modify: `super_bot.py:1037-1061` (the `_acquire_pre_lock` closure)
- Modify: `super_bot.py` (add module-level helper near the top, after the imports)
- Test: `tests/test_asset_lock_acquire.py` (create)

**Interfaces:**
- Consumes: `sqlite3` (stdlib), the `asset_locks` table (`symbol TEXT PRIMARY KEY, locked_by TEXT NOT NULL, locked_at TEXT NOT NULL, reason TEXT, heartbeat TEXT NOT NULL`).
- Produces: `acquire_asset_lock(db_path: str, symbol: str, owner: str, reason: str, now_iso: str) -> bool` — module-level in `super_bot.py`. Returns `True` when `owner` holds the lock for `symbol` afterwards (either newly taken or renewed). Returns `False` only when another engine holds a fresh (<900s) lock.

**Context — why this is a regression, not a pre-existing bug:**

`59bdd91` moved the lock acquisition to *before* order placement so the grid could not race the swing engine. The new SQL was:

```sql
INSERT INTO asset_locks (symbol, locked_by, locked_at, reason, heartbeat)
SELECT ?, ?, ?, ?, ?
WHERE NOT EXISTS (SELECT 1 FROM asset_locks WHERE symbol = ? AND locked_by != ?)
```

`asset_locks.symbol` is `TEXT PRIMARY KEY`. The `NOT EXISTS` guard only excludes rows owned by a *different* engine, so when the grid **already owns** the lock, the guard passes and the `INSERT` hits a primary-key conflict.

Reproduced:

```
PRE-LOCK RAISED: IntegrityError : UNIQUE constraint failed: asset_locks.symbol
```

The exception escapes `_acquire_pre_lock` and is swallowed by the caller at `super_bot.py:776` (`except Exception as e: log.debug(...)`) — so the order is never placed and the failure is invisible at default log level. The intended `[GRID-LOCK-RACE-VETO]` warning never fires, because the `INSERT` raises instead of returning 0 rows.

The scan-side check at `super_bot.py:816` only vetoes locks held by *other* engines, so a self-owned orphaned lock passes the scan and then crashes in the pre-lock. Orphaned locks are pruned only after 900s (`super_bot.py:406-415`), so there is a ~15-minute post-close window where the symbol cannot be re-entered.

- [x] **Step 1: Write the failing test**

Create `tests/test_asset_lock_acquire.py`:

```python
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from super_bot import acquire_asset_lock


class TestAcquireAssetLock(unittest.TestCase):
    """Regression: acquiring a symbol lock must never raise, and must not let a
    self-owned orphan lock block a legitimate entry."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmp.name, "locks.db")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS asset_locks (
                       symbol TEXT PRIMARY KEY,
                       locked_by TEXT NOT NULL,
                       locked_at TEXT NOT NULL,
                       reason TEXT,
                       heartbeat TEXT NOT NULL)"""
            )
        self.now = datetime.now(timezone.utc).isoformat()
        self.old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()

    def tearDown(self):
        self._tmp.cleanup()

    def _seed(self, symbol, owner, heartbeat):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO asset_locks (symbol, locked_by, locked_at, reason, heartbeat)"
                " VALUES (?, ?, ?, ?, ?)",
                (symbol, owner, heartbeat, "seed", heartbeat),
            )

    def test_renews_self_owned_lock_without_raising(self):
        # This is the exact state that produced
        # "sqlite3.IntegrityError: UNIQUE constraint failed: asset_locks.symbol"
        self._seed("BTC/USDT", "super_bot_grid", self.now)
        self.assertTrue(
            acquire_asset_lock(self.db_path, "BTC/USDT", "super_bot_grid", "grid", self.now)
        )

    def test_takes_free_symbol(self):
        self.assertTrue(
            acquire_asset_lock(self.db_path, "ETH/USDT", "super_bot_grid", "grid", self.now)
        )

    def test_refuses_symbol_held_by_other_engine(self):
        self._seed("SOL/USDT", "swing_architect", self.now)
        self.assertFalse(
            acquire_asset_lock(self.db_path, "SOL/USDT", "super_bot_grid", "grid", self.now)
        )

    def test_takes_symbol_whose_other_lock_is_stale(self):
        self._seed("SOL/USDT", "swing_architect", self.old)
        self.assertTrue(
            acquire_asset_lock(self.db_path, "SOL/USDT", "super_bot_grid", "grid", self.now)
        )

    def test_lock_is_released_for_next_owner_after_release(self):
        self.assertTrue(
            acquire_asset_lock(self.db_path, "BTC/USDT", "super_bot_grid", "grid", self.now)
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM asset_locks WHERE symbol = ? AND locked_by = ?",
                ("BTC/USDT", "super_bot_grid"),
            )
        self.assertTrue(
            acquire_asset_lock(self.db_path, "BTC/USDT", "super_bot_scalp_short", "short", self.now)
        )
```

- [x] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_asset_lock_acquire.py -q --tb=line`

Expected: collection error — `ImportError: cannot import name 'acquire_asset_lock' from 'super_bot'`.

- [x] **Step 3: Add the module-level helper**

Add to `super_bot.py`, after the module imports and before the class definitions:

```python
def acquire_asset_lock(db_path: str, symbol: str, owner: str, reason: str, now_iso: str) -> bool:
    """Atomically take or renew the exclusive symbol lock.

    Returns True when `owner` holds the lock for `symbol` once this returns, and
    False only when another engine holds a lock that is still fresh (<900s).
    """
    with sqlite3.connect(db_path, timeout=5.0) as conn:
        conn.execute(
            """DELETE FROM asset_locks
               WHERE symbol = ?
                 AND (julianday('now') - julianday(COALESCE(heartbeat, locked_at))) * 86400.0 > 900.0""",
            (symbol,),
        )
        renewed = conn.execute(
            """UPDATE asset_locks SET locked_at = ?, reason = ?, heartbeat = ?
               WHERE symbol = ? AND locked_by = ?""",
            (now_iso, reason, now_iso, symbol, owner),
        ).rowcount
        if renewed == 0:
            taken = conn.execute(
                """INSERT INTO asset_locks (symbol, locked_by, locked_at, reason, heartbeat)
                   SELECT ?, ?, ?, ?, ?
                   WHERE NOT EXISTS (
                       SELECT 1 FROM asset_locks
                       WHERE symbol = ?
                         AND (julianday('now') - julianday(COALESCE(heartbeat, locked_at))) * 86400.0 <= 900.0
                   )""",
                (symbol, owner, now_iso, reason, now_iso, symbol),
            ).rowcount
            if taken == 0:
                return False
        conn.commit()
        return True
```

- [x] **Step 4: Replace the closure in `_execute_grid_maker_entry`**

Replace the whole `_acquire_pre_lock` closure and its call site (`super_bot.py:1037-1061`) with:

```python
            # P0-2 LOCK ACQUISITION: take the symbol lock BEFORE order placement so the
            # swing engine cannot enter the same symbol while our maker order rests.
            lock_who = "super_bot_grid" if is_long else "super_bot_scalp_short"
            lock_why = f"Chop Maker Grid dip-buy ({conv_reason})" if is_long else f"Scalp Short Maker fade ({conv_reason})"
            now_iso_lock = datetime.now(timezone.utc).isoformat()

            lock_ok = await asyncio.to_thread(
                acquire_asset_lock, "trading_data.db", symbol, lock_who, lock_why, now_iso_lock
            )
            if not lock_ok:
                log.warning(
                    f"[{engine_tag}-LOCK-RACE-VETO] Could not acquire asset_locks for {symbol} "
                    f"prior to order placement. Aborting."
                )
                return
```

Keep the existing `_release_pre_lock` closure unchanged.

- [x] **Step 5: Run the new test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_asset_lock_acquire.py -q --tb=line`

Expected: `5 passed`.

- [x] **Step 6: Run the three tests the regression broke**

Run: `.venv/Scripts/python.exe -m pytest tests/test_grid_risk_cap.py tests/test_microstructure_degraded_haircut.py tests/test_gate_toggle_wiring.py -q --tb=line`

Expected: `3 passed` — these assert `create_post_only_order.called` is `True`, which is exactly what the lock crash prevented.

- [x] **Step 7: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q --tb=no -rf`

Expected: `3 failed, 1539 passed` — the 3 remaining failures are the two `async def` tests (`test_event_catalyst.py`, `test_sentinel_event.py`) plus nothing else. **If any other test fails, stop and investigate before committing.**

- [x] **Step 8: Commit**

```bash
git add super_bot.py tests/test_asset_lock_acquire.py
git commit -m "fix(super_bot): make pre-order asset lock idempotent so a self-owned orphan lock cannot block entries"
```

---

### Task 2: Isolate the backtest database from the production ledger

**Files:**
- Modify: `backtests/run_backtest.py:259-278`
- Test: `tests/test_backtest_db_isolation.py` (create)

**Interfaces:**
- Consumes: `DatabaseManager(db_path: str | None)` from `db_manager.py` — when `db_path` is `None` it resolves to `<repo root>/trading_data.db`, which is the production ledger.
- Produces: the backtest writes only to `backtests/backtest_sim.db`. The production ledger is never opened for write.

**Context — the `59bdd91` fix was incomplete:**

The commit removed `trading_data.db` from the `os.remove` list but left the rest:

```python
    sim_dir = os.path.dirname(os.path.abspath(__file__))
    for f in ["trade_history.csv", "trades.csv", os.path.join(sim_dir, "backtest_sim.db")]:
        if os.path.exists(f):
            os.remove(f)

    db = DatabaseManager()          # <-- resolves to PRODUCTION trading_data.db
    ...
    conn.execute("DELETE FROM trades")
    conn.execute("DELETE FROM cycles")
    conn.execute("REPLACE INTO system_state ... 'balance_total' ...")
```

`DatabaseManager.__init__` defaults `db_path` to the repo-root `trading_data.db`. The file survives; its **contents** are still destroyed, including `system_state.balance_total`. The root `trade_history.csv` and `trades.csv` are also still deleted, CWD-relative.

- [x] **Step 1: Write the failing test**

Create `tests/test_backtest_db_isolation.py`:

```python
import ast
import os
import unittest


class TestBacktestDbIsolation(unittest.TestCase):
    """The backtest must never construct DatabaseManager() with its default path,
    and must never DELETE from a production ledger."""

    def setUp(self):
        self.path = os.path.join("backtests", "run_backtest.py")
        with open(self.path, "r", encoding="utf-8") as f:
            self.src = f.read()
        self.tree = ast.parse(self.src)

    def test_database_manager_is_constructed_with_an_explicit_path(self):
        calls = [
            n for n in ast.walk(self.tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "DatabaseManager"
        ]
        self.assertTrue(calls, "expected a DatabaseManager(...) call")
        for call in calls:
            self.assertTrue(
                call.args or call.keywords,
                "DatabaseManager() with no db_path resolves to the production "
                "trading_data.db and wipes the live ledger",
            )

    def test_no_unscoped_delete_from_trades(self):
        tree = ast.parse(self.src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Exec if hasattr(ast, "Exec") else ast.Call):
                pass
        # Simple textual guard: a DELETE FROM trades must be scoped to a sim database.
        self.assertNotIn(
            "conn.execute(\"DELETE FROM trades\")",
            self.src,
            "unscoped DELETE FROM trades wipes the production ledger",
        )
```

- [x] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backtest_db_isolation.py -q --tb=line`

Expected: `2 failed` — `DatabaseManager()` has no arguments, and the unscoped `DELETE FROM trades` literal is present.

- [x] **Step 3: Scope the backtest to its own database**

Replace the purge-and-open block in `backtests/run_backtest.py` with:

```python
    # Purge old simulation artifacts to avoid ParserErrors.
    # P0-6 FIX: the backtest must never touch the production ledger — every
    # artifact it writes or deletes is scoped to this directory.
    sim_dir = os.path.dirname(os.path.abspath(__file__))
    sim_db = os.path.join(sim_dir, "backtest_sim.db")
    for f in [os.path.join(sim_dir, "trade_history.csv"),
              os.path.join(sim_dir, "trades.csv"),
              sim_db]:
        if os.path.exists(f):
            os.remove(f)

    db = DatabaseManager(db_path=sim_db)
    now_iso = datetime.now(UTC).isoformat()
    with db._get_conn() as conn:
        conn.execute("DELETE FROM trades")
        conn.execute("DELETE FROM cycles")
        conn.execute("REPLACE INTO system_state (key, value, updated_at) VALUES ('balance_total', ?, ?)", (str(config.INITIAL_BALANCE), now_iso))
        conn.execute("REPLACE INTO system_state (key, value, updated_at) VALUES ('session_session_start', ?, ?)", (now_iso, now_iso))
        conn.commit()
```

Note the `session_start` key name in the original is `session_start` — preserve it exactly; do not introduce the typo shown above. The correct final line is:

```python
        conn.execute("REPLACE INTO system_state (key, value, updated_at) VALUES ('session_start', ?, ?)", (now_iso, now_iso))
```

- [x] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backtest_db_isolation.py -q --tb=line`

Expected: `2 passed`.

- [x] **Step 5: Verify the production ledger is untouched**

Run:

```bash
.venv/Scripts/python.exe -c "
import hashlib,sqlite3
h=hashlib.md5(open('trading_data.db','rb').read()).hexdigest()
c=sqlite3.connect('file:trading_data.db?mode=ro',uri=True)
print('md5',h,'rows',c.execute('select count(*) from trades').fetchone()[0])
"
```

Expected: `244` rows — unchanged. Record the hash; re-check it after Task 3 to prove no local regression.

- [x] **Step 6: Commit**

```bash
git add backtests/run_backtest.py tests/test_backtest_db_isolation.py
git commit -m "fix(backtest): scope run_backtest to backtest_sim.db so the production ledger is never wiped"
```

---

### Task 3: Move the ledger uniqueness constraint into version-controlled schema code

**Files:**
- Modify: `db_manager.py:546-550` (index creation after the `trades` table)
- Modify: `db_manager.py:701-702` (index creation after the `swing_trades` migration)
- Modify: `swing_trader.py:169` (the second `CREATE TABLE IF NOT EXISTS swing_trades` definition)
- Create: `scripts/dedup_trade_ledger.py`
- Test: `tests/test_ledger_unique_indexes.py` (create)

**Interfaces:**
- Consumes: `db_manager.DatabaseManager._init_db()` (runs on every `DatabaseManager()` construction), `swing_trader`'s own schema bootstrap.
- Produces: `idx_trades_unique` on `trades(symbol, entry_price, timestamp)` and `idx_swing_trades_unique` on `swing_trades(symbol, entry_price, timestamp)`, created by code on every startup. `scripts/dedup_trade_ledger.py --db <path>` performs a backed-up, idempotent dedup + index creation.

**Context — the constraint is not in version control:**

`59bdd91` applied the unique indexes with ad-hoc SQL against the local `trading_data.db` file. `grep -rn "idx_trades_unique" --include=*.py .` returns **nothing** — no `.py` file creates them. A rebuilt database, a fresh clone, or the EC2 box has no constraint. The remote ledger still holds **369 trades with 122 duplicate groups and no index**.

Creating a `UNIQUE` index over a table that already contains duplicates raises `sqlite3.IntegrityError`, so the schema code must tolerate that and point at the dedup script.

- [x] **Step 1: Write the failing test**

Create `tests/test_ledger_unique_indexes.py`:

```python
import os
import sqlite3
import tempfile
import unittest

import db_manager


class TestLedgerUniqueIndexes(unittest.TestCase):
    """The unique ledger indexes must be created by schema code, not by a
    one-off SQL command against a database file."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmp.name, "ledger.db")

    def tearDown(self):
        self._tmp.cleanup()

    def _indexes(self, table):
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?", (table,)
            ).fetchall()
        return {r[0] for r in rows}

    def test_trades_unique_index_is_created_by_schema_code(self):
        db_manager.DatabaseManager(db_path=self.db_path)
        self.assertIn("idx_trades_unique", self._indexes("trades"))

    def test_swing_trades_unique_index_is_created_by_schema_code(self):
        db_manager.DatabaseManager(db_path=self.db_path)
        self.assertIn("idx_swing_trades_unique", self._indexes("swing_trades"))

    def test_duplicate_insert_is_rejected(self):
        db_manager.DatabaseManager(db_path=self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO trades (timestamp, symbol, type, side, entry_price, qty)"
                " VALUES ('2026-11-04T00:00:00', 'BTC/USDT', 'grid', 'long', 100.0, 1.0)"
            )
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO trades (timestamp, symbol, type, side, entry_price, qty)"
                    " VALUES ('2026-11-04T00:00:00', 'BTC/USDT', 'grid', 'long', 100.0, 1.0)"
                )
```

- [x] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ledger_unique_indexes.py -q --tb=line`

Expected: `2 failed, 1 passed` — the index assertions fail because no code creates the indexes.

- [x] **Step 3: Add the index creation to `db_manager._init_db`**

Immediately after the existing index block at `db_manager.py:546-550`, add:

```python
            # P0-7: ledger uniqueness must be enforced by schema code, not by a one-off
            # SQL command. A ledger with pre-existing duplicates cannot be indexed, so
            # failure here is reported and the dedup script must be run.
            for stmt, idx_name, tbl in (
                ('CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_unique ON trades (symbol, entry_price, timestamp)', 'idx_trades_unique', 'trades'),
                ('CREATE UNIQUE INDEX IF NOT EXISTS idx_swing_trades_unique ON swing_trades (symbol, entry_price, timestamp)', 'idx_swing_trades_unique', 'swing_trades'),
            ):
                try:
                    cursor.execute(stmt)
                except sqlite3.IntegrityError:
                    conn.rollback()
                    log.warning(
                        f"[DB-SCHEMA] Cannot create {idx_name} on {tbl}: table contains duplicate rows. "
                        f"Run scripts/dedup_trade_ledger.py to clean the ledger."
                    )
```

Also add the same two `CREATE UNIQUE INDEX IF NOT EXISTS` statements to the `swing_trades` index block at `db_manager.py:701-702`, inside the same try/except pattern.

- [x] **Step 4: Add the index to the second `swing_trades` schema definition**

`swing_trader.py:169` declares `CREATE TABLE IF NOT EXISTS swing_trades (...)` independently of `db_manager`. After that statement, add:

```python
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_swing_trades_unique
                        ON swing_trades(symbol, entry_price, timestamp);
```

- [x] **Step 5: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ledger_unique_indexes.py -q --tb=line`

Expected: `3 passed`.

- [x] **Step 6: Write the dedup script**

Create `scripts/dedup_trade_ledger.py`:

```python
"""Deduplicate the trade ledger and install the unique indexes.

Destructive: it DELETES duplicate trade rows. Always run with --backup (the
default) so the original rows are recoverable.

Usage:
    python scripts/dedup_trade_ledger.py --db trading_data.db
    python scripts/dedup_trade_ledger.py --db trading_data.db --dry-run
"""
import argparse
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone

DUP_KEYS = {
    "trades": ("symbol", "entry_price", "timestamp"),
    "swing_trades": ("symbol", "entry_price", "timestamp"),
}


def dedup_table(conn, table):
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    if not cols or "id" not in cols:
        return 0, 0
    keys = [c for c in DUP_KEYS[table] if c in cols]
    if len(keys) != len(DUP_KEYS[table]):
        return 0, 0
    before = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    key_sql = ", ".join(keys)
    conn.execute(
        f"""DELETE FROM {table} WHERE id NOT IN (
                SELECT MIN(id) FROM {table} GROUP BY {key_sql}
            )"""
    )
    after = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    conn.commit()
    return before, after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db_path = os.path.abspath(args.db)
    if not os.path.exists(db_path):
        print(f"ERROR: {db_path} does not exist")
        return 1

    if not args.dry_run:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup = f"{db_path}.bak_{stamp}"
        shutil.copy2(db_path, backup)
        print(f"backup written: {backup}")

    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        for table in DUP_KEYS:
            before, after = dedup_table(conn, table)
            print(f"{table}: {before} -> {after} rows")
        if args.dry_run:
            conn.rollback()
            print("dry run: no changes committed")
        else:
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_unique"
                " ON trades (symbol, entry_price, timestamp)"
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_swing_trades_unique"
                " ON swing_trades (symbol, entry_price, timestamp)"
            )
            conn.commit()
            print("unique indexes created")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [x] **Step 7: Verify the script on a copy, never the live ledger**

Run:

```bash
.venv/Scripts/python.exe -c "
import shutil; shutil.copy2('trading_data.db','/tmp/ledger_copy.db')"
.venv/Scripts/python.exe scripts/dedup_trade_ledger.py --db /tmp/ledger_copy.db
```

Expected: `trades: 244 -> 244 rows` (the local ledger is already deduped) and `unique indexes created`. On the EC2 copy it will report `369 -> 247`.

- [x] **Step 8: Commit**

```bash
git add db_manager.py swing_trader.py scripts/dedup_trade_ledger.py tests/test_ledger_unique_indexes.py
git commit -m "fix(db): enforce ledger uniqueness in schema code and add a backed-up dedup migration script"
```

---

### Task 4: Repair the module the security fix broke, and close the secret gap

**Files:**
- Modify: `archive/legacy_agents/mobile_web_bridge.py:34-54`
- Test: `tests/test_mobile_bridge_importable.py` (create)

**Interfaces:**
- Consumes: `APP_PASSWORD` / `SESSION_SECRET_KEY` environment variables.
- Produces: a module that imports cleanly with or without `APP_PASSWORD` set, and never contains a hardcoded credential default.

**Context:**

`59bdd91` replaced the hardcoded fallback password with a random one, but placed the `log.warning(...)` call at line 41 while `log = logging.getLogger("mobile_bridge")` is defined at line 54. The module therefore raises `NameError` at import time:

```
IMPORT FAILED: NameError name 'log' is not defined
```

Verified by executing the module directly. Separately, the secret `SovereignSwarm2026!` is still present in git history and **still present on the EC2 box** — `grep -c 'SovereignSwarm2026' archive/legacy_agents/mobile_web_bridge.py` on the remote returns `1`. `.gitattributes` `export-ignore` only affects future `git archive` bundles; the previously deployed copy was never removed.

- [x] **Step 1: Write the failing test**

Create `tests/test_mobile_bridge_importable.py`:

```python
import importlib.util
import os
import unittest


class TestMobileBridgeImportable(unittest.TestCase):
    """archive/legacy_agents/mobile_web_bridge.py must import without raising,
    and must not ship a hardcoded password default."""

    PATH = os.path.join("archive", "legacy_agents", "mobile_web_bridge.py")

    def _load(self):
        spec = importlib.util.spec_from_file_location("mwb_under_test", self.PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_module_imports_cleanly(self):
        module = self._load()
        self.assertTrue(module.APP_PASSWORD)

    def test_no_hardcoded_password_literal(self):
        with open(self.PATH, "r", encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("SovereignSwarm2026", src)
```

- [x] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_mobile_bridge_importable.py -q --tb=line`

Expected: `2 failed` — the import raises `NameError`, and the secret literal is present.

- [x] **Step 3: Move the logger above the security block**

In `archive/legacy_agents/mobile_web_bridge.py`, cut these two lines from their current position at line 53-54:

```python
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] MOBILE_BRIDGE: %(message)s")
log = logging.getLogger("mobile_bridge")
```

and paste them immediately after the `load_dotenv(...)` call, before the `# Security Configuration` block. The resulting order must be:

```python
# Load workspace environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] MOBILE_BRIDGE: %(message)s")
log = logging.getLogger("mobile_bridge")

# Security Configuration — P0-8: Enforce explicit password from environment, never ship hardcoded defaults
APP_PASSWORD = os.getenv("APP_PASSWORD")
if not APP_PASSWORD:
    # Disable bridge if no strong password provided
    APP_PASSWORD = secrets.token_hex(32)
    log.warning("[SECURITY] No APP_PASSWORD configured in environment. Generated one-time ephemeral secret.")
SESSION_SECRET = os.getenv("SESSION_SECRET_KEY", secrets.token_hex(32))
```

- [x] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_mobile_bridge_importable.py -q --tb=line`

Expected: `2 passed`.

- [x] **Step 5: Commit**

```bash
git add archive/legacy_agents/mobile_web_bridge.py tests/test_mobile_bridge_importable.py
git commit -m "fix(archive): define logger before use so the mobile bridge module imports cleanly"
```

---

### Task 5: Make the reconciler health signal truthful

**Files:**
- Modify: `reconciliation_engine.py:248-353` (`audit_paper_positions` return value)
- Modify: `reconciliation_engine.py:1259-1269` and `:1302-1309` (`audit_live_exchange_positions` return values)
- Modify: `reconciliation_engine.py:1336-1346` (`is_healthy` computation)
- Test: `tests/test_reconciler_health_signal.py` (create)

**Interfaces:**
- Consumes: `update_system_state(key, value)` — writes the `reconciliation_heartbeat` row.
- Produces: every audit return dict gains a `"status"` key with value `"HEALTHY"` or `"DEGRADED"`. `audit_paper_positions()` → `{"audited_at", "desyncs_healed", "positions_active", "status"}`. `audit_live_exchange_positions()` → adds `"status"` alongside the existing `"mode"` key.

**Context:**

`59bdd91` replaced the hardcoded `"status": "HEALTHY"` with:

```python
is_healthy = bool(
    paper_audit.get("status") in ("OK", "HEALTHY") and
    (not live_audit or live_audit.get("status") in ("OK", "HEALTHY"))
)
```

Neither audit function ever sets a `"status"` key — `grep -n '"status"' reconciliation_engine.py` matches only the three lines of this new block. `paper_audit.get("status")` is therefore `None`, `is_healthy` is always `False`, and the heartbeat telemetry reports **`"DEGRADED"` permanently**. The fix inverted a permanent false-HEALTHY into a permanent false-DEGRADED; the signal is still unusable.

- [x] **Step 1: Write the failing test**

Create `tests/test_reconciler_health_signal.py`:

```python
import inspect
import unittest

import reconciliation_engine as recon


class TestReconcilerHealthSignal(unittest.TestCase):
    """Both audit paths must publish a real 'status' key, otherwise the
    heartbeat telemetry is permanently DEGRADED."""

    def test_paper_audit_source_sets_status(self):
        src = inspect.getsource(recon.audit_paper_positions)
        self.assertIn('"status"', src)
        self.assertIn("DEGRADED", src)
        self.assertIn("HEALTHY", src)

    def test_live_audit_source_sets_status(self):
        src = inspect.getsource(recon.audit_live_exchange_positions)
        self.assertIn('"status"', src)

    def test_heartbeat_is_not_hardcoded_healthy(self):
        src = inspect.getsource(recon.reconciliation_main_loop)
        self.assertNotIn('"status": "HEALTHY",', src)
        self.assertIn('"status": "HEALTHY" if is_healthy else "DEGRADED"', src)
```

- [x] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_reconciler_health_signal.py -q --tb=line`

Expected: `2 failed, 1 passed` — neither audit function sets `"status"`.

- [x] **Step 3: Add a status to the paper audit**

In `reconciliation_engine.py`, inside `audit_paper_positions()`:

- Initialise `audit_summary = {"audited_at": datetime.now(UTC).isoformat(), "desyncs_healed": 0, "positions_active": 0, "status": "HEALTHY"}`.
- In the per-table `except Exception as e:` handler, set `audit_summary["status"] = "DEGRADED"` before logging.

- [x] **Step 4: Add a status to the live audit**

In `audit_live_exchange_positions()`:

- The success return becomes:

```python
        return {
            "mode": "live_bitget",
            "status": "HEALTHY",
            "exchange_positions": len(active_exchange_positions),
            "synced_at": datetime.now(UTC).isoformat()
        }
```

- Both failure returns (`{"mode": "live_error", "error": str(e)}` at `:1269` and `:1309`) become:

```python
        return {"mode": "live_error", "status": "DEGRADED", "error": str(e)}
```

- The paper-mode bypass return at `:360` becomes:

```python
        return {"mode": "paper_active", "status": "HEALTHY", "live_reconciliation": "bypassed"}
```

- [x] **Step 5: Correct the heartbeat computation**

Replace the `is_healthy` block with:

```python
            is_healthy = bool(
                paper_audit.get("status") == "HEALTHY" and
                (not live_audit or live_audit.get("status") == "HEALTHY")
            )
```

and make the log line report the real status rather than a hardcoded `Status: OK`:

```python
            log.info(f"[HEARTBEAT] Active: {paper_audit['positions_active']} | Desyncs Healed: {paper_audit['desyncs_healed']} | Status: {telemetry['status']}")
```

- [x] **Step 6: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_reconciler_health_signal.py -q --tb=line`

Expected: `3 passed`.

- [x] **Step 7: Run the reconciler regression tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_reconciler_ledger_guard.py tests/test_reconciler_fee_attribution.py tests/test_p2_7_regression.py tests/test_p2_9_regression.py -q --tb=line`

Expected: all pass.

- [x] **Step 8: Commit**

```bash
git add reconciliation_engine.py tests/test_reconciler_health_signal.py
git commit -m "fix(reconciler): publish a truthful HEALTHY/DEGRADED status instead of a permanently degraded signal"
```

---

### Task 6: Separate the scalp-short and grid breakeven triggers

**Files:**
- Modify: `config.py:130` (add `SCALP_SHORT_BREAKEVEN_TRIGGER_PCT` next to `GRID_BREAKEVEN_TRIGGER_PCT`)
- Modify: `swing_trader.py:2529` and `:2541` (split the shared `short_be_trigger` variable)
- Test: `tests/test_breakeven_trigger_separation.py` (create)

**Interfaces:**
- Consumes: `config.GRID_BREAKEVEN_TRIGGER_PCT` (0.0120) and the new `config.SCALP_SHORT_BREAKEVEN_TRIGGER_PCT` (0.0075).
- Produces: `swing_trader.py` uses a distinct trigger per ratchet — Ratchet 0A (Quick Scalp Short micro-breakeven) reads `SCALP_SHORT_BREAKEVEN_TRIGGER_PCT`; Ratchet 0B (grid short partial TP) reads `GRID_BREAKEVEN_TRIGGER_PCT`.

**Context:**

`59bdd91` changed one line:

```python
-  short_be_trigger = max(float(getattr(config, "GRID_BREAKEVEN_TRIGGER_PCT", 0.0050)), 0.0050)
+  short_be_trigger = float(getattr(config, "SCALP_SHORT_BREAKEVEN_TRIGGER_PCT", 0.0075))
```

`SCALP_SHORT_BREAKEVEN_TRIGGER_PCT` **does not exist in `config.py`**, so the value is a hardcoded magic number, and `GRID_BREAKEVEN_TRIGGER_PCT` (0.0120) is no longer consulted here. The commit message does not mention this change.

`short_be_trigger` gates **two** ratchets:

- Ratchet 0A (`is_scalp_short`) — the fast micro-breakeven, whose own docstring says *"deliberate fast-harvest design"*. 0.75% is correct here and is what `tests/test_scalp_short_engine.py` asserts.
- Ratchet 0B (`is_grid_pos`) — the grid short partial-TP harvest. This one is **live**: the long-side grid ratchet at `swing_trader.py:2439` uses `GRID_BREAKEVEN_TRIGGER_PCT` (1.20%), so the short side is now asymmetric with it, and `config.py:130` documents the 1.20% value as *"Only breakeven at +1.20% (80% of TP)"* — a deliberate anti-micro-clipping floor that `tests/test_config_hygiene.py` enforces.

The production bug is the shared variable, not the 0.75% value. Splitting them restores the intended asymmetry without breaking the scalp-short test.

- [x] **Step 1: Write the failing test**

Create `tests/test_breakeven_trigger_separation.py`:

```python
import inspect
import unittest

import config
import swing_trader


class TestBreakevenTriggerSeparation(unittest.TestCase):
    """Ratchet 0A (scalp short) and Ratchet 0B (grid short) must not share one
    trigger: the grid floor is 1.20%, the scalp-short fast-harvest is 0.75%."""

    def test_scalp_short_trigger_exists_in_config(self):
        self.assertTrue(hasattr(config, "SCALP_SHORT_BREAKEVEN_TRIGGER_PCT"))
        self.assertAlmostEqual(config.SCALP_SHORT_BREAKEVEN_TRIGGER_PCT, 0.0075, places=4)

    def test_grid_trigger_is_unchanged(self):
        self.assertGreaterEqual(config.GRID_BREAKEVEN_TRIGGER_PCT, 0.0120)

    def test_ratchet_0b_uses_the_grid_trigger(self):
        src = inspect.getsource(swing_trader.SwingArchitectTrader.manage_active_positions)
        grid_branch = src.split("Ratchet 0B")[1]
        self.assertIn("grid_be_trigger", grid_branch)
        self.assertNotIn("peak_profit_pct >= short_be_trigger", grid_branch)

    def test_ratchet_0a_uses_the_scalp_short_trigger(self):
        src = inspect.getsource(swing_trader.SwingArchitectTrader.manage_active_positions)
        scalp_branch = src.split("Ratchet 0A")[1].split("Ratchet 0B")[0]
        self.assertIn("short_be_trigger", scalp_branch)
```

- [x] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_breakeven_trigger_separation.py -q --tb=line`

Expected: `2 failed` — `config.SCALP_SHORT_BREAKEVEN_TRIGGER_PCT` does not exist, and Ratchet 0B still reads `short_be_trigger`.

- [x] **Step 3: Add the config key**

In `config.py`, directly after the `GRID_BREAKEVEN_TRIGGER_PCT` line, add:

```python
SCALP_SHORT_BREAKEVEN_TRIGGER_PCT = float(os.getenv("SCALP_SHORT_BREAKEVEN_TRIGGER_PCT", "0.0075")) # Fast micro-breakeven for Quick Scalp Short only (Ratchet 0A). Grid ratchets use GRID_BREAKEVEN_TRIGGER_PCT.
```

- [x] **Step 4: Split the trigger variables**

In `swing_trader.py`, replace:

```python
                    short_be_trigger = float(getattr(config, "SCALP_SHORT_BREAKEVEN_TRIGGER_PCT", 0.0075))
```

with:

```python
                    short_be_trigger = float(getattr(config, "SCALP_SHORT_BREAKEVEN_TRIGGER_PCT", 0.0075))
                    grid_be_trigger = max(float(getattr(config, "GRID_BREAKEVEN_TRIGGER_PCT", 0.0050)), 0.0050)
```

and in Ratchet 0B change:

```python
                    elif is_grid_pos and peak_profit_pct >= short_be_trigger and int(pos.get("grid_partial_tp_harvested") or 0) == 0:
```

to:

```python
                    elif is_grid_pos and peak_profit_pct >= grid_be_trigger and int(pos.get("grid_partial_tp_harvested") or 0) == 0:
```

- [x] **Step 5: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_breakeven_trigger_separation.py -q --tb=line`

Expected: `4 passed`.

- [x] **Step 6: Verify the scalp-short test still passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scalp_short_engine.py tests/test_config_hygiene.py -q --tb=line`

Expected: all pass — the scalp-short ratchet keeps its 0.75% trigger and the grid keeps its 1.20% floor.

- [x] **Step 7: Commit**

```bash
git add config.py swing_trader.py tests/test_breakeven_trigger_separation.py
git commit -m "fix(swing): give the grid short partial-TP its own 1.20% breakeven trigger instead of sharing the scalp-short value"
```

---

## Final verification

- [x] Run the complete suite:

```bash
.venv/Scripts/python.exe -m pytest tests/ -q --tb=no -rf
```

Actual result on `48aa0c9`: **1577 passed, 0 failed, 47 subtests passed** (689s). The two expected async failures no longer occur — `main` has since added `pytest.ini` with `asyncio_mode = auto` and the `pytest-asyncio` plugin, so `tests/test_event_catalyst.py::test_catalyst` and `tests/test_sentinel_event.py::test_sentinel` now execute and pass. The planned expectation of `2 failed, 1542 passed` was written before that upstream change.

- [x] The two async tests remain unfixed by this plan. They are not broken by `59bdd91`; they have never run because no async plugin is installed. To enable them:

```bash
.venv/Scripts/python.exe -m pip install pytest-asyncio
```

then add to `pytest.ini` (new file):

```ini
[pytest]
testpaths = tests
asyncio_mode = auto
```

`pytest.ini` also fixes the bare-`pytest` crash, which currently collects the vendored interpreter under `Python/` and produces 174 collection errors plus an access-violation crash. `testpaths = tests` excludes `scratch/`, `Python/` and `.qoder/`.

- [x] Confirm the production ledger was not modified locally:

```bash
.venv/Scripts/python.exe -c "
import hashlib,sqlite3
print(hashlib.md5(open('trading_data.db','rb').read()).hexdigest())
c=sqlite3.connect('file:trading_data.db?mode=ro',uri=True)
print('trades rows:', c.execute('select count(*) from trades').fetchone()[0])"
```

Expected: `244` rows.

---

## Production Runbook — requires explicit operator approval

**Do not run any of this without the operator's explicit go-ahead.** These steps touch the live EC2 box and the live ledger.

### 1. Deploy the code (Task 1-6 fixes)

```bash
# from the repo root, on the operator's machine
powershell -ExecutionPolicy Bypass -File ./deploy_to_aws.ps1
```

This ships `git archive HEAD`, which — since `59bdd91` added `.gitattributes` with `archive/ export-ignore` — no longer includes `archive/`. That is desirable, but it does **not** delete the copy already on the server. See step 3.

### 2. Cycle the daemons

```bash
ssh -i ~/.ssh/aws_trading_key.pem ubuntu@13.53.41.118 \
  "tmux send-keys -t trading C-c; sleep 2; tmux send-keys -t trading './venv/bin/python start_system.py' C-m"
```

Verify:

```bash
ssh -i ~/.ssh/aws_trading_key.pem ubuntu@13.53.41.118 \
  "ps -eo pid,etime,cmd | grep -E 'super_bot|reconciliation_engine|start_system' | grep -v grep"
```

### 3. Remove the archive tree and rotate the leaked password

The secret `SovereignSwarm2026!` was published in git history and is still on the box. Removing the file is not enough — the value must be treated as compromised.

```bash
ssh -i ~/.ssh/aws_trading_key.pem ubuntu@13.53.41.118 \
  "cd ~/llama_trading_bot && rm -rf archive && grep -rc 'SovereignSwarm2026' . 2>/dev/null | grep -v ':0' | head"
```

Then generate a fresh `APP_PASSWORD` for the operator's `.env` on the box — do not reuse the old value.

### 4. Back up and deduplicate the production ledger

The EC2 ledger currently holds **369 trades with 122 duplicate groups and no unique index**. The dedup **deletes 122 rows**, so back it up first.

```bash
ssh -i ~/.ssh/aws_trading_key.pem ubuntu@13.53.41.118 \
  "cd ~/llama_trading_bot && cp trading_data.db trading_data.db.bak_\$(date +%Y%m%d_%H%M%S) && ls -la trading_data.db*"
```

Then upload and run the script:

```bash
scp -i ~/.ssh/aws_trading_key.pem scripts/dedup_trade_ledger.py ubuntu@13.53.41.118:~/llama_trading_bot/scripts/
ssh -i ~/.ssh/aws_trading_key.pem ubuntu@13.53.41.118 \
  "cd ~/llama_trading_bot && ./venv/bin/python scripts/dedup_trade_ledger.py --db trading_data.db"
```

Expected output: `trades: 369 -> 247 rows` and `unique indexes created`.

Verify afterwards:

```bash
ssh -i ~/.ssh/aws_trading_key.pem ubuntu@13.53.41.118 \
  "cd ~/llama_trading_bot && ./venv/bin/python -c \"
import sqlite3
c=sqlite3.connect('file:trading_data.db?mode=ro',uri=True)
print('rows', c.execute('select count(*) from trades').fetchone()[0])
print('dupes', c.execute('select count(*) from (select count(*) n from trades group by timestamp,symbol,side,entry_price,qty having n>1)').fetchone()[0])\""
```

Expected: `rows 247`, `dupes 0`.

**Note:** the local ledger shows `244` rows and the production one should show `247`. They are not the same dataset — do not copy one over the other.

### 5. Confirm the heartbeat reports HEALTHY

```bash
ssh -i ~/.ssh/aws_trading_key.pem ubuntu@13.53.41.118 \
  "cd ~/llama_trading_bot && ./venv/bin/python -c \"
import json,sqlite3
c=sqlite3.connect('file:trading_data.db?mode=ro',uri=True)
row=c.execute(\\\"select value from system_state where key='reconciliation_heartbeat'\\\").fetchone()
print(json.loads(row[0])['status'])\""
```

Expected: `HEALTHY`. If it reads `DEGRADED` after Task 5 is deployed, the live audit is genuinely failing — check `logs/reconciliation_engine.log`.

---

## Out of scope — found during the audit, not covered by this plan

These are real but were not part of the `59bdd91` remediation and are not required to close it. Listed so they are not lost.

1. **Grid and swing may double-place stops.** The grid now binds `presetStopLossPrice`/`presetTakeProfitPrice` on the maker order *and* places separate strategy SL/TP orders afterwards (`super_bot.py:1187`, `:1293`). The swing engine already did this (`swing_trader.py:1817`, `:1926`). If Bitget does not dedupe, each position carries two conditional stops. Verify with one small live order and inspect the exchange's open conditional orders.
2. **Split-brain database paths.** `reconciliation_engine.py` uses BASE_DIR-absolute paths; `super_bot.py` and `swing_trader.py` use CWD-relative `sqlite3.connect("trading_data.db")`. Run from a different working directory and the engines write to a different, empty database.
3. **Funding is always zero.** `reconciliation_engine.py` reads `total_funding`, a column that does not exist on `open_positions` or `swing_open_positions`, so funding is always `0.0`. The two PnL writers therefore disagree on the same trade.
4. **Lookahead bias in every published backtest metric.** `quant_validation.py` fills at the same bar's close that produced the signal; `backtest_swing_architect.py` attaches the same-day daily bar to 4H bars via `merge_asof`. The live engine is causal — only the backtests leak. Every Sharpe and Profit-Factor published so far describes a strategy that cannot be traded.
5. **`tests/test_key_type.py` makes a live API call at import time and prints the API key.** Move it out of `tests/` or gate it behind an env var.
6. **`main.py:962` imports `nightly_trainer` from the repo root, but it lives in `scripts/`.** The ImportError is swallowed, so the RL/calibration layer never runs and `models/*.pth` are never loaded by any live engine.

---

## Post-verification follow-up (2026-11-04, after the branch was merged)

Running the replay during final verification exposed two defects that the plan's
source-inspection tests could not see. Both are fixed on `main` in commits
`5b7d266` and `13b1806`, with 9 new tests (`tests/test_sandbox_ledger_isolation.py`,
`tests/test_backtest_entry_path.py`). The suite is **1586 passed, 0 failed**.

### F1 — The replay opened the production ledger (P0)

Task 2 sandboxed the replay's **CWD**, but `DatabaseManager()` resolves its default
path from its own module directory, not the CWD. `archive/legacy_agents/state_manager.py:63`
(`EngineState.__init__` → `DatabaseManager()`) therefore opened the **production**
`trading_data.db` in write mode on every replay. Proof: the ledger's md5 changed
during a replay and its `idx_trades_unique` was rebuilt from the 3-column to the
5-column identity.

Fix: a `TRADING_DATA_DB_PATH` override that `DatabaseManager()` honours,
`state_manager.ledger_path()` resolves, and the replay publishes via
`apply_sandbox_ledger_path()`. The state manager's three raw
`sqlite3.connect("trading_data.db")` calls now go through `ledger_path()` too, so
they read the sandbox ledger instead of an empty sandbox file.

**Consequence for the operator:** the local ledger's schema was modified by that
accidental write. No rows were lost (244 `trades` / 181 `swing_trades`, `max(timestamp)`
still 2026-10-21), but the md5 is no longer a valid "untouched" witness — compare row
counts and `max(timestamp)` instead.

### F2 — The replay could never open a position (P0)

`archive/legacy_agents/core_engine.py:123` `calculate_exits()` returns **three**
values `(soft_sl, tp, hard_sl)`, but `archive/bot_v10_backtest.py:421` unpacked two.
Every entry that passed the gates raised `ValueError: too many values to unpack
(expected 2, got 3)` **before the position was created**, so the harness reported
`TOTAL TRADES: 0 / NET PNL $0.00`. That output is indistinguishable from "the
strategy has no edge", and it is what every backtest number in this repo was
measured against. The same stale unpack existed in `archive/experimental_traders/live_trader.py:419,1204`
and `archive/live_trader_backup_pre_audit_fix.py:392,854`.

Fix: unpack all three values; a regression test drives a real entry with a stubbed
LLM verdict and asserts a simulated position opens.

After the fix, one full replay produced **41 trades, +$35.71 net, 7.01% max
drawdown** (BTC/USDT 38.9% win / PF 1.50 / +$3.57; ETH/USDT 34.8% / PF 2.10 / +$32.14).

**Caveat:** the replay still calls the live LLM for every entry verdict
(`archive/bot_v10_backtest.py:367`), so it is non-deterministic and bills real API
calls; a large share of calls here log `All cloud API endpoints exhausted` and are
vetoed with score 0.0. Treat any single replay as a smoke test, not a measurement,
until the AI dependency is stubbed or removed from the replay path.

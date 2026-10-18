# P0 Swarm Audit Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 5 critical P0 defects identified by 10-agent swarm audit before fleet restart

**Architecture:** Atomic, testable fixes applied sequentially with verification after each. Each fix addresses a specific vulnerability that could cause catastrophic loss or system compromise.

**Tech Stack:** Python 3.11+, SQLite, asyncio, ccxt, Streamlit, Bitget UTA V3 API

**Spec:** C:\Users\user\.qoder\tmp\C--Users-user--gemini-antigravity-scratch-llama-trading-bot\attachments\8569da77-b738-4fc3-af66-b6e5aaf2abef\3cef6410-6e1a-4221-a6f8-ccbf93bff80c.txt

## Global Constraints

- All changes must be backward compatible with existing database schema
- No breaking changes to API contracts between modules
- Must pass all 39 existing regression tests after each fix
- Credential rotation requires manual operator action (cannot automate key generation)

---

### Task 1: Secure .env File Permissions and Add Startup Validation

**Files:**
- Modify: `.env` (file permissions only)
- Modify: `start_system.py:1-50` (add credential validation at boot)
- Test: `tests/test_p0_security.py` (new file)

**Interfaces:**
- Consumes: None
- Produces: `_validate_credentials()` function called during startup

- [ ] **Step 1: Set .env file permissions to 600**

```bash
icacls .env /inheritance:r
icacls .env /grant:r "%USERNAME%":(R,W)
```

- [ ] **Step 2: Add credential validation function to start_system.py**

Add at line 45 (after imports):

```python
def _validate_credentials() -> bool:
    """Validate that critical credentials are loaded and warn if .env is world-readable."""
    import os
    import stat
    
    # Check .env file permissions
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        file_stat = os.stat(env_path)
        mode = file_stat.st_mode
        # Check if others have read permission (world-readable)
        if mode & stat.S_IROTH:
            log.critical("[SECURITY] .env file is world-readable! Run: chmod 600 .env")
            return False
    
    # Validate required credentials are present
    required_keys = [
        "BITGET_API_KEY", "BITGET_API_SECRET", "BITGET_PASSPHRASE",
        "GMAIL_USER", "GMAIL_APP_PASSWORD"
    ]
    missing = [k for k in required_keys if not os.getenv(k)]
    if missing:
        log.critical(f"[SECURITY] Missing required credentials: {missing}")
        return False
    
    log.info("[SECURITY] Credential validation passed")
    return True
```

- [ ] **Step 3: Call validation in main() before daemon spawning**

At line 320 (before `_initialize_databases()`), add:

```python
if not _validate_credentials():
    log.critical("Credential validation failed. Refusing to start.")
    sys.exit(1)
```

- [ ] **Step 4: Write test for credential validation**

Create `tests/test_p0_security.py`:

```python
import os
import stat
import tempfile
import unittest
from unittest.mock import patch


class TestCredentialValidation(unittest.TestCase):
    def test_world_readable_env_rejected(self):
        """Verify that world-readable .env files cause startup failure."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("BITGET_API_KEY=test\n")
            temp_env = f.name
        
        try:
            # Set world-readable permissions
            os.chmod(temp_env, 0o644)
            
            # Import the validation function
            import start_system
            result = start_system._validate_credentials.__wrapped__(temp_env)
            
            self.assertFalse(result, "World-readable .env should fail validation")
        finally:
            os.remove(temp_env)
    
    def test_missing_credentials_rejected(self):
        """Verify that missing required credentials cause startup failure."""
        with patch.dict(os.environ, {}, clear=True):
            import start_system
            result = start_system._validate_credentials()
            self.assertFalse(result, "Missing credentials should fail validation")
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd C:\Users\user\.gemini\antigravity\scratch\llama_trading_bot
.venv/Scripts/python.exe -m pytest tests/test_p0_security.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add .env start_system.py tests/test_p0_security.py
git commit -m "P0-1: Secure .env permissions and add credential validation at startup"
```

---

### Task 2: Disable Mobile Bridge RCE Vector

**Files:**
- Modify: `archive/legacy_agents/agent_mobile_bridge.py:37` (remove hardcoded password)
- Modify: `scripts/start_mobile_bridge.py:27,56,75-89` (disable Cloudflare Tunnel, require explicit password)
- Test: `tests/test_p0_mobile_bridge.py` (new file)

**Interfaces:**
- Consumes: None
- Produces: Hardened mobile bridge that refuses to start without explicit APP_PASSWORD

- [ ] **Step 1: Remove hardcoded fallback password**

In `archive/legacy_agents/agent_mobile_bridge.py:37`, change:

```python
# BEFORE:
APP_PASSWORD = os.getenv("APP_PASSWORD", "SovereignSwarm2026!")

# AFTER:
APP_PASSWORD = os.getenv("APP_PASSWORD")
if not APP_PASSWORD:
    raise EnvironmentError("APP_PASSWORD environment variable is required. Set it in .env before starting the mobile bridge.")
```

- [ ] **Step 2: Disable Cloudflare Tunnel in start script**

In `scripts/start_mobile_bridge.py:75-89`, comment out the tunnel launch:

```python
# DISABLED FOR SECURITY: Cloudflare Tunnel exposes admin interface to public internet
# cf_cmd = [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{BRIDGE_PORT}"]
# subprocess.Popen(cf_cmd)
log.warning("[SECURITY] Cloudflare Tunnel disabled. Use VPN or SSH tunnel for remote access.")
```

- [ ] **Step 3: Remove password from stdout**

In `scripts/start_mobile_bridge.py:56`, remove:

```python
# DELETE THIS LINE:
print(f"  App Password  : {APP_PASSWORD}\n")

# REPLACE WITH:
print(f"  App Password  : (set in .env - never share this)\n")
```

- [ ] **Step 4: Bind to localhost only**

In `scripts/start_mobile_bridge.py:75`, ensure:

```python
web_cmd = [sys.executable, "-m", "uvicorn", "mobile_web_bridge:app", "--host", "127.0.0.1", "--port", str(BRIDGE_PORT)]
```

- [ ] **Step 5: Write test verifying password requirement**

Create `tests/test_p0_mobile_bridge.py`:

```python
import os
import unittest
from unittest.mock import patch


class TestMobileBridgeSecurity(unittest.TestCase):
    def test_missing_password_raises_error(self):
        """Verify mobile bridge refuses to start without APP_PASSWORD."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(EnvironmentError) as ctx:
                import archive.legacy_agents.agent_mobile_bridge as bridge
                # Force reload to trigger the check
                import importlib
                importlib.reload(bridge)
            
            self.assertIn("APP_PASSWORD", str(ctx.exception))
    
    def test_hardcoded_password_removed(self):
        """Verify no hardcoded fallback password exists in source."""
        with open("archive/legacy_agents/agent_mobile_bridge.py", "r") as f:
            content = f.read()
        
        self.assertNotIn("SovereignSwarm2026!", content,
                        "Hardcoded password still present in source")
```

- [ ] **Step 6: Run test to verify it passes**

```bash
.venv/Scripts/python.exe -m pytest tests/test_p0_mobile_bridge.py -v
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add archive/legacy_agents/agent_mobile_bridge.py scripts/start_mobile_bridge.py tests/test_p0_mobile_bridge.py
git commit -m "P0-2: Remove hardcoded password and disable Cloudflare Tunnel exposure"
```

---

### Task 3: Unblock Async Event Loop in Reconciler

**Files:**
- Modify: `reconciliation_engine.py:1176` (wrap sync call in asyncio.to_thread)
- Test: Existing tests should pass; add timing assertion

**Interfaces:**
- Consumes: `audit_paper_positions()` (sync function)
- Produces: Non-blocking async reconciliation loop

- [ ] **Step 1: Wrap audit_paper_positions in asyncio.to_thread**

At line 1176 in `reconciliation_engine.py`, change:

```python
# BEFORE:
paper_audit = audit_paper_positions()

# AFTER:
paper_audit = await asyncio.to_thread(audit_paper_positions)
```

- [ ] **Step 2: Verify no other sync HTTP calls in async context**

Search for `requests.get` in reconciliation_engine.py:

```bash
grep -n "requests.get" reconciliation_engine.py
```

Lines 152 and 166 are inside `fetch_live_price()` which is called from `audit_paper_positions()`. Since we wrapped the entire function in `asyncio.to_thread()`, these are now offloaded correctly.

- [ ] **Step 3: Run existing regression tests**

```bash
.venv/Scripts/python.exe -m pytest tests/test_p2_9_regression.py tests/test_p2_11_regression.py -v
```

Expected: All 39 tests PASS

- [ ] **Step 4: Commit**

```bash
git add reconciliation_engine.py
git commit -m "P0-3: Offload sync HTTP calls to thread pool to unblock async event loop"
```

---

### Task 4: Swing Trader Fail-Open on Zero Balance

**Files:**
- Modify: `swing_trader.py:1266-1352` (add explicit else clause to veto entry when balance unresolved)
- Test: `tests/test_p0_swing_balance.py` (new file)

**Interfaces:**
- Consumes: `uta_client.fetch_balance()` (returns float or None)
- Produces: Entry veto when balance is missing or too low

- [ ] **Step 1: Add explicit else clause after balance check**

At line 1348 in `swing_trader.py` (after the `if live_bal and live_bal > 10.0:` block closes), add:

```python
else:
    log.warning(f"[BALANCE-VETO] Live balance unresolved (${live_bal}) or below minimum ($10). Vetoing entry for {symbol}.")
    return
```

This ensures that when `live_bal` is `None`, `0`, or `<= $10.0`, the position entry is blocked entirely rather than falling through with the $85 default margin.

- [ ] **Step 2: Write test for zero balance veto**

Create `tests/test_p0_swing_balance.py`:

```python
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class TestSwingTraderBalanceVeto(unittest.TestCase):
    def test_zero_balance_vetos_entry(self):
        """Verify swing trader blocks entry when balance is zero."""
        from swing_trader import SwingArchitectTrader
        
        trader = SwingArchitectTrader()
        trader.uta_client = MagicMock()
        trader.uta_client.fetch_balance = AsyncMock(return_value=0.0)
        
        # Mock other dependencies
        trader.allocation_usdt = 85.0
        
        result = asyncio.run(trader.open_swing_position(
            symbol="BTC/USDT",
            side="long",
            entry_price=50000.0,
            stop_loss_pct=1.5,
            take_profit_pct=3.0,
            leverage=3,
            strategy="test"
        ))
        
        self.assertIsNone(result, "Position entry should be vetoed when balance is zero")
    
    def test_none_balance_vetos_entry(self):
        """Verify swing trader blocks entry when balance API returns None."""
        from swing_trader import SwingArchitectTrader
        
        trader = SwingArchitectTrader()
        trader.uta_client = MagicMock()
        trader.uta_client.fetch_balance = AsyncMock(return_value=None)
        trader.allocation_usdt = 85.0
        
        result = asyncio.run(trader.open_swing_position(
            symbol="BTC/USDT",
            side="long",
            entry_price=50000.0,
            stop_loss_pct=1.5,
            take_profit_pct=3.0,
            leverage=3,
            strategy="test"
        ))
        
        self.assertIsNone(result, "Position entry should be vetoed when balance is None")
```

- [ ] **Step 3: Run test to verify it passes**

```bash
.venv/Scripts/python.exe -m pytest tests/test_p0_swing_balance.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add swing_trader.py tests/test_p0_swing_balance.py
git commit -m "P0-4: Add explicit balance veto to prevent fail-open on zero/unresolved balance"
```

---

### Task 5: Missing get_open_position Method

**Files:**
- Modify: `swing_trader.py:1724` (replace non-existent method call with fetch_positions)
- Test: Existing tests should cover this path

**Interfaces:**
- Consumes: `uta_client.fetch_positions(symbol=...)` (returns list of position dicts)
- Produces: Single active position dict or None

- [ ] **Step 1: Replace get_open_position with fetch_positions**

At line 1724 in `swing_trader.py`, change:

```python
# BEFORE:
active_ex_pos = await self.uta_client.get_open_position(symbol)

# AFTER:
positions = await self.uta_client.fetch_positions(symbol=symbol)
active_ex_pos = next((p for p in positions if float(p.get("contracts", p.get("total", 0.0))) > 0), None)
```

The `fetch_positions()` method returns a list; we filter for the first position with non-zero size. The field name varies between exchanges (`contracts` for some, `total` for others), so we check both.

- [ ] **Step 2: Run existing regression tests**

```bash
.venv/Scripts/python.exe -m pytest tests/test_p2_9_regression.py tests/test_p2_11_regression.py -v
```

Expected: All 39 tests PASS

- [ ] **Step 3: Commit**

```bash
git add swing_trader.py
git commit -m "P0-5: Replace non-existent get_open_position with fetch_positions call"
```

---

## Verification Checklist

After all 5 tasks complete:

- [ ] All 39 regression tests pass
- [ ] `.env` file has 600 permissions
- [ ] Mobile bridge cannot start without explicit APP_PASSWORD
- [ ] Reconciler cycle completes in <5 seconds (not 30+ seconds)
- [ ] Swing trader vetoes entry when balance API fails
- [ ] Pre-trade drift guard functions correctly
- [ ] Fleet can restart successfully

## Rollback Plan

If any fix causes issues:

```bash
# Revert last commit
git reset --hard HEAD~1

# Or revert all P0 fixes
git reset --hard origin/main
```

Each fix is atomic and independently reversible.

# AI Event Catalyst & Horizon Arbiter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform the news and sentiment sentinel from a blunt keyword veto into an autonomous event-driven intelligence engine that classifies breaking news and volatility events into either temporary flash liquidity shocks (rebound traps) or secular macro regime shifts (authorizing high-conviction 4H breakdown shorts or bull breakouts), with zero API cost ($0.00).

**Architecture:** 
1. `ai_engine.py`: Implement `evaluate_event_catalyst(event_packet)` querying Google Gemini Flash (Groq rotation failover) to analyze headline text alongside live Bitget funding rate, open interest delta, BTC 15m velocity, and SPY/DXY macro moves.
2. `sentinel_agent.py`: Upgrade async RSS headline collector to detect high-urgency catalysts, query the arbiter non-blockingly, and persist structured event directives (`ai_event_catalyst`, `ai_event_playbook`, `ai_event_horizon`) to SQLite `system_state`.
3. `macro_sentinel.py`: Ingest breaking news and event catalyst state into the 20-minute macro synthesis loop, dynamically aligning `ai_macro_bias` and derivatives skew directives.
4. `swing_trader.py`: In `_check_macro_safety`, read `ai_event_playbook` to immediately unblock 2x/3x trend shorts on `SECULAR_SHORT_BREAKDOWN` while shielding against squeeze traps on `FLASH_REBOUND_TRAP`.
5. `dashboard.py`: Display real-time breaking event telemetry, catalyst category, horizon, and tactical playbook badges on Tab 3 & Tab 4.

**Tech Stack:** Python 3.12, asyncio, aiohttp, Google Gemini Flash 2.5, Groq Cloud (Llama 3.3 70B), SQLite WAL (`trading_data.db` and `trading_ledger.db`), Streamlit.

---

### Task 1: AI Engine Catalyst & Horizon Evaluator

**Files:**
- Modify: `ai_engine.py` (add `evaluate_event_catalyst`)
- Test: `test_event_catalyst.py`

**Step 1: Write test script**
Create `test_event_catalyst.py` to test both flash crash and secular war shock classification.

**Step 2: Run test to verify it fails**
Run: `.venv\Scripts\python.exe test_event_catalyst.py`
Expected: FAIL (`AttributeError: module 'ai_engine' has no attribute 'evaluate_event_catalyst'`)

**Step 3: Implement `evaluate_event_catalyst` in `ai_engine.py`**
Provide institutional system prompt that ingests breaking headline, price velocity, funding rate, open interest, and macro indices, returning structured JSON with horizon, category, and tactical playbook.

**Step 4: Run test to verify it passes**
Run: `.venv\Scripts\python.exe test_event_catalyst.py`
Expected: PASS (Returns valid JSON verdict for test cases)

---

### Task 2: Sentiment Sentinel Event Processor & Persistence

**Files:**
- Modify: `sentinel_agent.py`
- Test: `test_sentinel_event.py`

**Step 1: Write test script**
Verify that `sentinel_agent.py` triggers event evaluation when high-impact headlines arrive and persists results to `system_state`.

**Step 2: Implement reactive catalyst trigger in `sentinel_agent.py`**
- Deduplicate evaluated headlines to prevent redundant API calls.
- Classify headlines into high-impact triggers (war, missile, sanctions, sec, etf, rate, inflation, crash, ban).
- Dispatch `ai_engine.evaluate_event_catalyst()` asynchronously.
- Update `system_state` keys (`ai_event_catalyst`, `ai_event_playbook`, `ai_event_horizon`, `ai_event_headline`).
- Update `perform_disaster_audit()` to honor intelligent playbooks rather than blunt keyword rejections.

**Step 3: Verify test passes**
Run: `.venv\Scripts\python.exe test_sentinel_event.py`
Expected: PASS

---

### Task 3: Macro Sentinel Integration

**Files:**
- Modify: `macro_sentinel.py`

**Step 1: Feed event catalyst into `generate_macro_synthesis`**
Read latest `ai_event_catalyst` and `ai_event_headline` from `system_state` and include in `macro_packet`.

**Step 2: Ensure fast-path macro reaction**
If `ai_event_playbook == 'SECULAR_SHORT_BREAKDOWN'`, immediately set `ai_macro_bias = 'BEARISH'` in `system_state` without waiting for the full 20-minute macro timer.

---

### Task 4: Swing Architect Event-Driven Short/Long Gating

**Files:**
- Modify: `swing_trader.py`
- Test: `test_swing_event_gate.py`

**Step 1: Write test verifying macro safety response to event playbooks**
Test that `_check_macro_safety("SHORT")` returns `True` when `ai_event_playbook == 'SECULAR_SHORT_BREAKDOWN'` and `_check_macro_safety("SHORT")` returns `False` when `ai_event_playbook == 'FLASH_REBOUND_TRAP'`.

**Step 2: Modify `swing_trader.py` `_check_macro_safety`**
Add event playbook evaluations to `_check_macro_safety`.

**Step 3: Verify test passes**
Run: `.venv\Scripts\python.exe test_swing_event_gate.py`
Expected: PASS

---

### Task 5: Dashboard Telemetry & Live Verification

**Files:**
- Modify: `dashboard.py`

**Step 1: Add Event Intelligence Card to Tab 3**
Add visual card showing Breaking Event Headline, Horizon (`FLASH_LIQUIDITY_SHOCK` vs `STRUCTURAL_MACRO_BEAR` vs `BULLISH_CATALYST`), Confidence, and Active Playbook badge.

**Step 2: Restart bot supervisor & verify live execution**
Restart system under `start_system.py`, verify Streamlit loads cleanly at `http://localhost:8501`.

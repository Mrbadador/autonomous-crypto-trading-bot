# Open-Source Quantitative Microstructure & Sizing Integration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate top mathematical models and components scouted from open-source quant ecosystems into the live bot: Multi-Level OFI & Micro-Price, VPIN Pre-Trade Toxicity Circuit Breaker, Empirical Distribution-Free Fractional Kelly Sizing, and L2 Order Book CRC32/Sequence Replay.

**Architecture:** Create a zero-dependency, pure-Python mathematical microstructure module (`microstructure_metrics.py`), wire MLOFI and Micro-Price into `ofi_daemon.py`, attach VPIN Pre-Trade Toxicity Gate in `super_bot.py` and `microstructure_guard.py`, integrate empirical growth-optimal Kelly sizing in `risk_manager.py` and `super_bot.py`, and add CRC32 checksum & sequence gap validation in `ofi_daemon.py`. Surface live telemetry on `dashboard.py`.

**Tech Stack:** Python 3.12, asyncio, NumPy, Streamlit, SQLite WAL, Bitget UTA WebSocket v2.

---

### Task 1: Create Zero-Dependency Microstructure Metrics Module (`microstructure_metrics.py`)

**Files:**
- Create: `microstructure_metrics.py`
- Test: `tests/test_microstructure_metrics.py`

**Step 1: Write the failing test**
Create test suite covering:
- `compute_micro_price`: verify with symmetric and asymmetric book depths; verify fallback to mid when sizes are zero.
- `compute_multi_level_ofi`: verify Cont-Cucuringu-Zhang level innovation math and geometric depth decay.
- `compute_vpin`: verify Bulk Volume Classification (BVC) and Abramowitz-Stegun standard normal CDF approximation.
- `calculate_empirical_fractional_kelly`: verify Golden-Section search log-wealth optimization on positive, negative, and mixed return distributions.

**Step 2: Run the test to ensure it fails**
Run pytest on `tests/test_microstructure_metrics.py`.

**Step 3: Implement `microstructure_metrics.py`**
Implement:
- `compute_micro_price(best_bid_p, best_bid_q, best_ask_p, best_ask_q)`
- `compute_multi_level_ofi(prev_bids, curr_bids, prev_asks, curr_asks, levels=5, decay=0.5)`
- `VPINCalculator` with `update_trade(price, size)` and `get_vpin()`
- `calculate_empirical_growth_optimal_kelly(returns, fraction=0.25, max_cap=0.25, min_cap=0.05)`

**Step 4: Run tests and make sure they pass**
Verify all tests pass with zero errors.

---

### Task 2: Enhance `ofi_daemon.py` with Multi-Level OFI, Micro-Price, VPIN & CRC32/Sequence Integrity

**Files:**
- Modify: `ofi_daemon.py`
- Test: `tests/test_ofi_daemon_integration.py`

**Step 1: Write integration test**
Verify `SymbolOFIState` computes `mlofi`, `micro_price`, `micro_price_bps`, and `vpin`, and exports them in the metrics dictionary. Verify CRC32 checksum validation helper.

**Step 2: Update `SymbolOFIState` and `MultiSymbolOFIDaemon` in `ofi_daemon.py`**
- Integrate MLOFI and Micro-Price into the depth update loop.
- Track rolling volume buckets for VPIN from trade and depth ticks.
- Add Bitget CRC32 checksum verification helper.
- Update `microstructure_state.json` payload with `mlofi`, `micro_price`, `micro_price_bps`, and `vpin` per symbol.

**Step 3: Run integration test and verify output**
Run tests to confirm clean execution without blocking the event loop.

---

### Task 3: Attach VPIN Pre-Trade Toxicity Circuit Breaker in `super_bot.py` and `microstructure_guard.py`

**Files:**
- Modify: `super_bot.py`
- Modify: `microstructure_guard.py`
- Test: `tests/test_vpin_gate.py`

**Step 1: Write unit test for VPIN gate**
Test that `super_bot.py` Chop Grid Maker and `microstructure_guard.py` reject trades when `vpin >= 0.70` (informed toxicity alert).

**Step 2: Implement VPIN check in `super_bot.py`**
In `_scan_chop_grid_opportunities()`:
- Query cached symbol microstructure state.
- If `vpin >= 0.70`, log `[GRID-VPIN-TOXICITY-VETO]` and skip trade.

**Step 3: Update `microstructure_guard.py`**
Ensure `evaluate_pre_trade_veto()` checks symbol VPIN and issues a VETO if toxicity breaches threshold.

---

### Task 4: Integrate Empirical Fractional Kelly Sizing in `risk_manager.py` and `super_bot.py`

**Files:**
- Modify: `risk_manager.py`
- Modify: `super_bot.py`
- Modify: `swing_trader.py`
- Test: `tests/test_empirical_kelly.py`

**Step 1: Write unit test for empirical Kelly sizing**
Test on sample trade history from SQLite. Verify that negative expectancy returns $0.0$, while positive edge returns a safe Quarter-Kelly fraction ($0.05 \le f^* \le 0.25$).

**Step 2: Implement in `risk_manager.py`**
- Add `calculate_empirical_kelly()` reading actual return percentages from `trading_data.db`.
- Blend with conviction score and volatility scaling.

**Step 3: Update sizing in `super_bot.py` & `swing_trader.py`**
- Dynamically size trades using the Empirical Kelly fraction while strictly honoring the 70% total margin ceiling.

---

### Task 5: Expose New Alpha Telemetry on Streamlit Dashboard (`dashboard.py`)

**Files:**
- Modify: `dashboard.py`

**Step 1: Update Tab 4 ("RISK & MICROSTRUCTURE")**
- Add Multi-Level OFI (MLOFI), Imbalance-Weighted Micro-Price (with bps spread bias), and Live VPIN Flow Toxicity Gauge to the Microstructure Deck.
- Display VPIN status chips (`🟢 Benign Flow (<0.45)`, `🟡 Caution (0.45-0.70)`, `🔴 Toxic Cascade (>=0.70)`).

**Step 2: Verify Dashboard & Restart System**
- Test dashboard syntax and verify running services.

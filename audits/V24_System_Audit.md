# V24.0 System Architecture Audit
**Generated:** 2026-05-11 | **Status:** Read-Only Deep Scan | **Auditor:** Antigravity  
**Scope:** `main.py`, `live_oracle.py`, `regime_classifier.py`, `berserker_trader.py`, `live_trader.py`, `live_paper_exchange.py`, `warden_60m.py`, `risk_manager.py`, `sentinel_agent.py`, `ai_engine.py`

---

## 1. System Overview

The Llama Trading Engine is a multi-process, multi-agent algorithmic trading framework running in paper-trading mode. It operates two parallel trading engines (Sniper 15m and Berserker 5m Fleet) against a 7-asset cryptocurrency portfolio, governed by a layered hierarchy of quantitative, AI, and news-based kill-switches.

```
main.py (Launcher + Async Supervisor)
├── Process: Live-Oracle          → feeds trading_ledger.db
├── Process: Regime-Classifier    → feeds trading_data.db
├── Process: Sniper-15m           → reads/writes trading_data.db
└── Process: Berserker-5m Fleet   → reads/writes trading_data.db
    ├── BTC/USDT Node
    ├── ETH/USDT Node
    ├── SOL/USDT Node
    ├── AVAX/USDT Node
    ├── LINK/USDT Node
    ├── NEAR/USDT Node
    └── FET/USDT Node
```

---

## 2. Data Pipeline

### 2.1 Market Data Ingestion

**Source:** Binance REST API (public endpoints, no auth required) + Binance WebSocket streams.

The system uses a **dual-channel** ingestion model:

| Channel | Source | Consumers | Interval |
|---|---|---|---|
| REST Poll | `live_oracle.py` | All agents (via DB) | Every 5 minutes |
| REST Poll | `regime_classifier.py` | Trading engines (via DB) | Every 5 minutes |
| WebSocket | `berserker_trader.py` | Berserker only | Real-time, per candle close |
| WebSocket | `live_trader.py` | Sniper only | Real-time, per candle close |
| REST (ccxt) | `warden_60m.py` | All engines (direct call) | On-demand per trade attempt |

### 2.2 Live Oracle Pipeline (`live_oracle.py`)

```
Binance REST API (50 x 1h OHLCV)
         |
         v
  engineer_features()
  ┌─────────────────────────────────┐
  │  Volatility State               │  → std_dev / 0.005, clamp [0.5, 2.0]
  │  Trend Strength                 │  → % distance from SMA-20 x 50, clamp [-1.0, 1.0]
  │  VWAP (cumulative, 50-bar)      │
  │  RSI-14 (ewm-smoothed)         │
  └─────────────────────────────────┘
         |
         v
  commit_to_database()
  DELETE FROM market_context         ← SINGLE-ROW MODEL (see §5.2)
  INSERT INTO market_context (...)
         |
  trading_ledger.db  ←──────── All agents read this
```

**Critical Detail:** The Oracle uses `DELETE + INSERT` (not `UPSERT`) to maintain a single-row context model. This means there is a brief window between DELETE and INSERT where `market_context` is **empty**. Any agent querying during this gap receives no data.

### 2.3 Regime Classifier Pipeline (`regime_classifier.py`)

```
Binance REST API (100 x 15m OHLCV, BTCUSDT only)
         |
         v
  classify_and_save()
  ┌─────────────────────────────────┐
  │  Hurst Exponent (R/S Analysis)  │  → H > 0.55: TRENDING, H < 0.45: CHOP
  │  ADX-14                         │  → ADX < 25.0 → hard CHOP override
  └─────────────────────────────────┘
         |
         v
  DELETE FROM market_regime          ← Same single-row model vulnerability
  INSERT INTO market_regime (...)
         |
  trading_data.db ←──────── Sniper + Berserker read this
```

**Note:** The Regime Classifier writes to `trading_data.db`, while the Oracle writes to `trading_ledger.db`. These are two separate databases, creating a **split-read model** in the trading agents.

### 2.4 WebSocket → Engine Flow (Berserker)

```
Binance WebSocket (wss://stream.binance.com:9443/stream)
  Streams: btcusdt@kline_5m, ethusdt@kline_5m, ...
         |
         v  (per message, every ~5s)
  monitor_position(symbol, close_price)
    └── Trail stop-loss, check TP/SL, manage exits
         |
         v  (on candle close: k['x'] == True)
  handle_candle_close(symbol, kline_data)
    └── Compute indicators → VARSI → 60m Warden → AI Audit → Boardroom → execute
```

---

## 3. Execution Hierarchy

The Berserker engine enforces a **6-layer veto hierarchy** before any trade fires. Each gate is independent and additive — all must pass.

```
CANDLE CLOSE EVENT
        |
        v  Gate 1: Veto Cooldown
  Is this symbol in veto_pause_until cooldown?
  (Set for 30 min after AI veto)
  FAIL: return immediately, no processing
        |
        v  Gate 2: Deterministic Entry Conditions
  is_volume_spike  = vol_ratio >= BERSERKER_VOL_GATE
  is_trending      = ADX > 25.0
  v17_long_valid   = (RSI < VARSI target) AND (price > VWAP x 0.98)
  BTC macro trend  = BULLISH (for longs)
  FAIL: HOLD, no signal
        |
        v  Gate 3: 60-Minute Macro Warden
  warden.determine_regime() → LONG_ONLY / SHORT_ONLY / CHOP_LOCK
  LONG signal requires: LONG_ONLY
  SHORT signal requires: SHORT_ONLY
  FAIL: signal vaporized, 60M-VETO logged
        |
        v  Gate 4: TP Cooldown (same-direction block)
  Has this direction been blocked for 3 candles post Take-Profit?
  FAIL: signal blocked
        |
        v  Gate 5: Fleet Alpha Strike + Circuit Breaker
  FleetState.register_opportunity() → 200ms race
  FleetState.is_alpha_leader() → highest vol wins
  FleetState.get_active_positions_count() >= 3?
  FAIL: ALPHA-FILTER or CIRCUIT-BREAKER logged
        |
        v  Gate 6: AI Boardroom (LLM + Conviction)
  ai_engine.get_risk_params() → quality score (0-1)
  ConvictionBoardroom.get_conviction() → score + bool
  required_score = 0.75 (raised to 0.90 in fear)
  SentimentSentinel.perform_disaster_audit() check
  RiskManager.get_position_size() (Kelly sizing)
  MacroCircuitBreaker.fetch() (SPY contagion)
  FAIL: trade aborted, reason logged
        |
        v PASS ALL
  LivePaperExchange.open_position()
  (Atomic DB write, asyncio.Lock protected)
```

### 3.1 VARSI Dynamic RSI Threshold (V24.0)

Instead of a static RSI gate, the system now queries live Oracle data before each entry decision:

```
dynamic_rsi_target = 30.0 + (trend_strength x 10.0) - ((volatility - 1.0) x 10.0)
Clamped: [22.0, 38.0]
```

In high-trend, low-volatility markets the bot becomes more aggressive (target rises toward 38). In choppy, high-volatility conditions it becomes highly selective (target drops toward 22).

---

## 4. AI / ML Integration

### 4.1 LLM Boardroom (`ai_engine.py` + `conviction_filter.py`)

**Model:** Qwen/Llama via Groq API (remote inference)  
**Purpose:** Pre-trade qualitative audit of the setup  
**Pattern:** Singleton `LLMClient` per engine (cached via `_llm_clients` dict)

```
Pre-Trade Trigger
        |
        ├── ai_engine.get_risk_params()       LLM Call #1: Risk quality score (0.0–1.0)
        │   └── Response cached by MD5 of {regime, signal, phase, ema_stack, patterns}
        │
        └── ConvictionBoardroom.get_conviction()   LLM Call #2: Bull vs Bear deliberation
            └── Returns {execute: bool, conviction_score: 0-1, final_rationale: str}

Both calls run in parallel via asyncio.gather() with a hard 5-second timeout.
Timeout fallback: quality=0.5, boardroom defaults to baseline approval (0.80 score).
```

**Cache Mechanism:** MD5 hash of coarse market signals. Identical market regime + signal + EMA stack combos return a cached response.

### 4.2 PyTorch RL Agents (`analyst_agent.py` + `strategist_agent.py`)

**Model:** Custom PyTorch neural network (`policy_model.pth`, 84KB)  
**Status:** DEAD CODE — as of V24.0, RL agents are instantiated but their outputs are never consumed. The VARSI upgrade replaced the RL threshold system with direct Oracle DB queries. 14 model loads occur across 7 fleet nodes for zero functional benefit.

### 4.3 Sentiment Sentinel (`sentinel_agent.py`)

**Pattern:** Background async RSS cache (non-blocking hot path)  
**Feeds:** CoinTelegraph RSS, CoinDesk RSS  
**Refresh:** Every 60 seconds

```
startup: prime_data()
  └── start_background_refresh(['BTC', 'ETH', 'SOL', ...])
        └── _refresh_loop() → every 60s: fetches RSS, keyword-matches disaster terms
              └── Updates in-memory cache: {ticker: (status, msg, timestamp)}

On trade attempt (hot path, <1ms):
  └── perform_disaster_audit(ticker) → reads from self._cache only
        └── BLOCK if any keyword matched, PASSTHROUGH otherwise
```

**Disaster Keywords:** hack, exploit, sec sues, arrested, bankrupt, crash, delist, rug pull, fraud, insolvent, shutdown, suspended, seized

### 4.4 Fractional Kelly Risk Manager (`risk_manager.py`)

**Formula:** Half-Kelly Criterion, scaled by LLM conviction score

```
full_kelly  = win_rate - ((1 - win_rate) / avg_rr)   [e.g., 0.65 - 0.233 = 0.417]
half_kelly  = full_kelly / 2.0                         [→ 0.208 = 20.8%]
adjusted    = half_kelly × (conviction / 100)          [scaled by AI score]
final_pct   = min(adjusted, 0.05)                      [hard cap: 5% max]
```

**Current Parameters:** WR=65%, RR=1.5, max_cap=5%  
**Effective range at full conviction:** ~$25 allocation per $500 balance (5% capped)

---

## 5. Vulnerabilities & Bottlenecks

### 5.1 P0 CRITICAL: Synchronous `determine_regime()` Blocking the Async Loop

**File:** `berserker_trader.py` L266  
**Code:** `current_macro_regime = self.macro_warden.determine_regime()`

`determine_regime()` is a **synchronous** function that makes a live HTTP call to Binance via ccxt. Called directly without `asyncio.to_thread()` inside the async event loop. This blocks the **entire event loop** for all 7 fleet nodes on every candle close that generates a signal. At 1-3s API latency this introduces a significant stall across the fleet.

**Fix:** `await asyncio.to_thread(self.macro_warden.determine_regime)`

---

### 5.2 P0 CRITICAL: DELETE+INSERT Race Window Creates Empty Table

**Files:** `live_oracle.py` L156, `regime_classifier.py` L165

Both agents use `DELETE FROM table` then `INSERT`. During the gap, concurrent reads return zero rows. VARSI silently falls back to `dynamic_rsi_target = 30.0` without any log warning, making the adaptive logic indistinguishable from its degraded fallback.

**Fix:** Replace `DELETE + INSERT` with `INSERT OR REPLACE` keyed on a fixed primary key (e.g., `id=1`).

---

### 5.3 P0 CRITICAL: FleetState Reads Wrong Database — Circuit Breaker Non-Functional

**File:** `main.py` L144  
**Code:** `FleetState(db_path="trading_ledger.db", max_trades=3)`

`FleetState.get_active_positions_count()` queries `berserker_open_positions` from `trading_ledger.db`. However, `LivePaperExchange` writes all position data to `trading_data.db`. The `berserker_open_positions` table does not exist in `trading_ledger.db`. The Circuit Breaker always reads 0 positions, meaning the `MAX_OPEN_TRADES = 3` cap is **completely inoperative**. The fleet could theoretically open all 7 positions simultaneously.

**Fix:** Change to `FleetState(db_path="trading_data.db", max_trades=3)`.

---

### 5.4 P1 HIGH: VARSI Query Uses 5s Timeout vs. System Standard of 30s

**File:** `berserker_trader.py` L214  
**Code:** `conn = sqlite3.connect("trading_ledger.db", timeout=5.0)`

The system-wide standard established in V23.4 is `timeout=30.0`. The shorter 5s window may cause spurious lock errors during fleet boot and silently degrade VARSI to its static fallback.

**Fix:** Change `timeout=5.0` to `timeout=30.0`.

---

### 5.5 P1 HIGH: Dead RL Code — PyTorch Model Loaded 14 Times, Used 0 Times

**File:** `berserker_trader.py` L78-79  
**Code:** `self.rl_analyst = AnalystAgent(); self.rl_strategist = StrategistAgent()`

Seven fleet nodes × 2 model instantiations = 14 PyTorch model loads per boot. Outputs are never read since V24.0.

**Fix:** Remove these lines from `BerserkerTrader.__init__`.

---

### 5.6 P1 HIGH: `_load_balance_sync` Has No Retry or Timeout

**File:** `live_paper_exchange.py` L102-128

Called during `__init__` with raw `sqlite3.connect()` and no `timeout=`. If a DB lock occurs during the 1.5s stagger window between fleet node launches, balance silently defaults to `initial_balance`, potentially resetting an active session's in-memory balance.

**Fix:** Add `timeout=30.0` and 5-retry backoff (matching the V23.4 standard).

---

### 5.7 P2 MEDIUM: Hardcoded Values in Warden CHOP_LOCK Log

**File:** `warden_60m.py` L71  
**Code:** `log.info("[WARDEN-INSIGHT] Trend Strength {-0.19} is below LONG_ONLY threshold {0.25}.")`

Appears to be a broken f-string. The values `-0.19` and `0.25` are static literals, not computed values. This produces misleading diagnostics regardless of actual market state.

**Fix:** Replace with actual computed EMA/price values.

---

### 5.8 P2 MEDIUM: Duplicate `_ipv4_session` Function Definition

**File:** `ai_engine.py` L16 and L24

The function is defined twice. The second silently overwrites the first.

**Fix:** Remove the first definition (L16-19).

---

### 5.9 P2 MEDIUM: Raw `print()` Debug Statement in Production Recovery Path

**File:** `live_paper_exchange.py` L141  
**Code:** `print(f"DEBUG [{self.account_name}]: Searching {positions_table}...")`

Bypasses the logging system, log rotation, and log-level filtering. Outputs to stdout unformatted on every boot.

**Fix:** Replace with `log.debug(...)`.

---

### 5.10 P3 LOW: Module Docstring References Obsolete Database Name

**File:** `main.py` L11  
**Code:** `Both engines write to the shared live_paper_trades.db`

The canonical database is `trading_data.db`. This comment is a relic from an earlier architecture.

**Fix:** Update the docstring.

---

## 6. Database Architecture Summary

| Database | Written By | Read By | Key Tables |
|---|---|---|---|
| `trading_data.db` | `LivePaperExchange`, `RegimeClassifier` | Dashboard, Watchdog, Fleet engines | `system_state`, `trades`, `berserker_trades`, `open_positions`, `berserker_open_positions`, `market_regime` |
| `trading_ledger.db` | `LiveOracle`, `FleetState` (alpha), `sentiment` | VARSI query, Boardroom SPY check, Warden | `market_context`, `sentiment_logs`, `macro_logs` |

> **Note on §5.3:** `FleetState` writes `fleet_alpha` key to `trading_ledger.db` — this part works. But it reads `berserker_open_positions` from the same DB, which doesn't exist there. The write logic is correct; the read logic is wrong.

---

## 7. Priority Fix Recommendations

| Priority | ID | Issue | File | Impact |
|---|---|---|---|---|
| P0 CRITICAL | §5.3 | FleetState reads wrong DB — Circuit Breaker broken | `main.py` L144 | MAX_OPEN_TRADES cap never enforced |
| P0 CRITICAL | §5.1 | Sync `determine_regime()` blocks async loop | `berserker_trader.py` L266 | 1-3s full fleet stall per signal |
| P0 CRITICAL | §5.2 | DELETE+INSERT race window empties Oracle/Regime | `live_oracle.py`, `regime_classifier.py` | Silent VARSI fallback, false regime data |
| P1 HIGH | §5.4 | VARSI timeout too short (5s vs 30s standard) | `berserker_trader.py` L214 | Adaptive RSI silently degrades |
| P1 HIGH | §5.5 | 14 PyTorch model loads, 0 model reads | `berserker_trader.py` L78-79 | Memory waste x7 fleet nodes |
| P1 HIGH | §5.6 | `_load_balance_sync` has no retry/timeout | `live_paper_exchange.py` L102 | Silent balance reset on DB lock |
| P2 MEDIUM | §5.7 | Hardcoded values in Warden CHOP_LOCK log | `warden_60m.py` L71 | Misleading diagnostics |
| P2 MEDIUM | §5.8 | Duplicate `_ipv4_session` function | `ai_engine.py` L16 | Technical debt |
| P2 MEDIUM | §5.9 | Raw `print()` in recovery code | `live_paper_exchange.py` L141 | Bypasses log system |
| P3 LOW | §5.10 | Obsolete DB name in docstring | `main.py` L11 | Misleading docs |

---

*End of V24.0 Architecture Audit — Fleet Manager Review Required*

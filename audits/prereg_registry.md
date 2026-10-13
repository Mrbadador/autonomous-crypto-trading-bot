# Pre-Registration Hypothesis Registry (Forward Validation)

This registry formalizes empirical hypothesis testing for live crypto futures engines under the research-grade **MinTRL (Minimum Track Record Length)** discipline.
Any strategy tuning or allocation adjustment must be registered here prior to evaluation.

**Hard Allocation Gate:** While rolling bootstrap 90% CI lower bound $\le 0.0$, **NO allocation increase and NO Kelly acceleration** is permitted.

---

## Registered Trials

### TRIAL-001: 5m Chop Maker Grid (Fleet-Capped Deep Chop ADX < 20)
- **Engine:** 5m Chop Maker Grid (`super_bot.py`)
- **Hypothesis:** Restricting grid limit entries strictly to deep-market-chop environments (BTC 15m ADX < 20.0) under a fleet cap of 2 concurrent longs yields positive net EV net of Bitget taker stop-loss exits and funding carry.
- **Pre-Registered Parameters:**
  - `market_adx_ceiling = 20.0` (BTC 15m ADX)
  - `max_long_cap = 2`
  - `trade_allocation_usd = 85.0` (leverage 3x)
  - `fill_model = "one_tick"`
- **Target Sample Size:** $N = 300$ trades
- **Success Criterion:** Bootstrap 90% CI lower bound $> 0.0$ USD/trade (Net of maker fees, taker SL fees, and 8h funding).
- **Status:** ACTIVE
- **Date Registered:** 2026-10-13

### TRIAL-002: 4H Swing Stack (Leader-Gate Filtered Ichimoku Breakouts)
- **Engine:** 4H Swing Architect (`swing_trader.py`)
- **Hypothesis:** 4H Ichimoku Kumo breakouts filtered by relative-strength momentum (Leader Gate) and trailing breakeven ratchets produce positive net expectancy on high-beta alts (ADA, SOL).
- **Pre-Registered Parameters:**
  - 4H timeframe only
  - Soft-SL ratchet at +0.5% MFE to breakeven
  - Leader Gate active (suppresses laggards vs BTC)
- **Target Sample Size:** $N = 100$ trades
- **Success Criterion:** Bootstrap 90% CI lower bound $> 0.0$ USD/trade.
- **Status:** ACTIVE
- **Date Registered:** 2026-10-13

### TRIAL-003: books15 Empirical L2 Queue Calibration
- **Engine:** Execution Infrastructure (`depth_recorder.py`)
- **Hypothesis:** Fitting LogProb/SquareProb queue models to empirical books15 depth snapshots resolves the optimistic bias of bar-level touch fills.
- **Target Sample Size:** $N \ge 50,000$ snapshots across $\ge 14$ consecutive days
- **Success Criterion:** Statistically significant logistic depletion parameters ($k > 0$).
- **Status:** ACCUMULATING
- **Date Registered:** 2026-10-13

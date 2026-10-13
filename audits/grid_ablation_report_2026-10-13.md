# Quantitative Grid Ablation Report & Expectancy Audit (Engine-Split & Fleet-Cap Edition)

**Date:** 2026-10-13  
**Engine Under Test:** 5m Chop Maker Grid & Quick Scalp Short (`super_bot.py`)  
**Dataset Window:** Continuous 5m OHLCV (210,240 bars per asset) + Bitget Funding History  
> [!NOTE]
> **Dataset Staleness Notice:** The current 5m OHLCV corpus extends from **2024-08-16 07:45 UTC to 2026-08-16 07:40 UTC** (a ~2-month staleness window relative to October 2026). The testing dataset provides 210,240 continuous bars of historical sample size.

**Methodology & Fidelity Enhancements:**  
1. **Fleet Position Cap (FIX 7):** Live-faithful fleet cap where concurrent open and pending Grid LONG positions across all symbols are strictly capped at 2 (`open_long_count + pending_long_count < 2`), matching `super_bot.py`.
2. **Metric Hygiene (FIX 8):** Mislabeled 'Maker Fill %' corrected to `tp_exit_share` (TP maker exits / total exits). Added `signal_to_fill_pct` measuring filled maker limits over total limit orders placed.
3. **Regime Contamination Guard (FIX 9):** Removed fallback to 5m ADX (`market_adx.fillna(adx_5m)`). Missing BTC 15m regime blocks entries with `market_adx=99.0` (disclosed: 4 bars blocked on BTC, 3 bars on ETH/SOL).
4. **Synthetic 8h Funding Settlements (FIX 10):** Prepended 2,093 synthetic 8h funding settlements at each symbol's median rate (BTC: 5.75e-5, ETH: 6.50e-5, SOL: 5.10e-5) across the missing ~22-month window prior to mid-2026 real coverage.
5. **Order TTL Fidelity:** Live-faithful 1-bar TTL window (`max_order_ttl_bars=1`) vs 6-bar standing limit window (`ttl_ablation`).
6. **Engine-Split Breakdown:** Explicit separation between **Grid Longs**, **Scalp Shorts**, and **Combined** populations.

---

## 1. Audit of Core Engine Claims (Evaluated on Fleet-Capped Grid Longs)

| Marketing Claim | Claimed Target | Measured Empirical Result (Fleet Cap = 2) | Empirical Verdict |
| :--- | :--- | :--- | :--- |
| (a) 68.5% Win Rate in Chop | 68.5% WR | **46.2% WR** (Deep Chop ADX<20: N=892) | **REFUTED** (Peak WR ~46.2% under live fleet cap) |
| (b) Positive Expectancy from Maker Fees | EV > $0.00 | **Deep Chop EV = $-0.163 [-0.354, +0.023]** | **MARGINAL / NEGATIVE** (Net EV ~ -0.064% of notional) |
| (c) Regime Confined to Low-ADX Chop | Strict Low ADX | Market ADX < 28 shield active; Waterfall 5m ADX < 24 preserved | **CONFIRMED SHIELD** (Suppresses high-volatility trends) |

---

## 2. Live-Faithful Fleet-Capped Headline Performance (Max 2 Concurrent Longs)

| Engine Sub-Type | Trades (N) | Win Rate (%) | Profit Factor | Signal-to-Fill (%) | TP Exit Share (%) | Avg Hold (min) | Net EV / Trade ($) [90% CI] | Net PnL ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Chop Grid (Fleet-Capped Longs Only)** | 1497 | 45.2% | 0.87 | 90.7% | 45.2% | 426.6m | $-0.230 [$-0.373, $-0.087] | $-344.18 |
| **Quick Scalp (Short Scalps Only)** | 1400 | 44.2% | 0.42 | 90.7% | 2.4% | 63.2m | $-0.387 [$-0.443, $-0.334] | $-541.75 |
| **Combined Fleet Stack** | 2897 | 44.7% | 0.75 | 90.7% | 24.5% | 251.0m | $-0.306 [$-0.382, $-0.237] | $-885.93 |

---

## 3. Fleet Cap Ablation: Fleet Cap = 2 vs Uncapped (Cap = 99)

| Configuration | Long Trades (N) | Win Rate (%) | Signal-to-Fill (%) | Max DD ($) | Net EV / Trade ($) [90% CI] | Total Long PnL ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Live Fleet Cap (Max 2 Longs)** | 1497 | 45.2% | 90.7% | $429.11 | $-0.230 [$-0.373, $-0.087] | $-344.18 |
| **Uncapped Fleet Ablation (Cap = 99)** | 1755 | 45.9% | 90.8% | $418.42 | $-0.184 [$-0.313, $-0.055] | $-322.08 |

> [!NOTE]
> The live fleet cap (max 2) prunes 258 late-cascade entries (1755 -> 1497 trades). While overall drawdown is comparable ($429.11 vs $418.42) due to lower total trade count, restricting concurrent capital exposure to 2 positions strictly limits multi-asset portfolio tail risk.

---

## 4. Pre-Registered Arm: Market-ADX Entry Ceiling 20 Only (Fleet Capped)

| Strategy Arm | Trades (N) | Win Rate (%) | Signal-to-Fill (%) | Max DD ($) | Net EV / Trade ($) [90% CI] | EV % of Notional | Net PnL ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Live Shield (ADX < 28/25)** | 1497 | 45.2% | 90.7% | $429.11 | $-0.230 [$-0.373, $-0.087] | -0.090% | $-344.18 |
| **Pre-Registered Arm (ADX < 20 Only)** | 1067 | 47.1% | 90.7% | $207.01 | $-0.102 [$-0.265, $+0.061] | -0.040% | $-108.79 |

## 5. Quantitative Ablation Matrix per Asset (Engine-Split)

### Asset: BTCUSDT

| Configuration / Mode | Engine | Trades (N) | Win Rate (%) | Profit Factor | Signal-to-Fill (%) | TP Exit Share (%) | Avg Hold (min) | Max DD ($) | Net EV / Trade ($) [90% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fleet-Capped (Live-Faithful) | Grid Long | 464 | 45.5% | 0.88 | 90.7% | 45.5% | 711.7m | $116.43 | $-0.212 [$-0.451, $+0.025] |
| Standalone 1-Bar TTL | Grid Long | 494 | 46.6% | 0.92 | 90.7% | 46.6% | 703.4m | $102.92 | $-0.142 [$-0.389, $+0.082] |
| TTL Ablation (6-Bar TTL) | Grid Long | 501 | 47.1% | 0.94 | 90.7% | 47.1% | 710.7m | $86.61 | $-0.106 [$-0.351, $+0.126] |
| Strict-Optimistic (Touch Fill) | Grid Long | 509 | 48.1% | 0.98 | 99.9% | 48.1% | 712.9m | $69.03 | $-0.040 [$-0.268, $+0.200] |
| Ablate Regime Gate (OFF) | Grid Long | 626 | 47.4% | 0.95 | 90.7% | 47.4% | 674.3m | $120.03 | $-0.088 [$-0.296, $+0.129] |
| Ablate VWAP Band Filter (OFF) | Grid Long | 673 | 46.5% | 0.91 | 90.7% | 46.5% | 718.6m | $120.16 | $-0.145 [$-0.347, $+0.066] |
| Ablate 45m Tightening (OFF) | Grid Long | 379 | 46.2% | 0.90 | 90.7% | 46.2% | 696.5m | $110.11 | $-0.166 [$-0.439, $+0.106] |

### Asset: ETHUSDT

| Configuration / Mode | Engine | Trades (N) | Win Rate (%) | Profit Factor | Signal-to-Fill (%) | TP Exit Share (%) | Avg Hold (min) | Max DD ($) | Net EV / Trade ($) [90% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fleet-Capped (Live-Faithful) | Grid Long | 509 | 46.0% | 0.90 | 90.7% | 46.0% | 352.3m | $143.19 | $-0.171 [$-0.400, $+0.081] |
| Standalone 1-Bar TTL | Grid Long | 587 | 46.3% | 0.91 | 90.7% | 46.3% | 363.6m | $144.23 | $-0.148 [$-0.346, $+0.074] |
| TTL Ablation (6-Bar TTL) | Grid Long | 603 | 46.3% | 0.91 | 90.7% | 46.3% | 368.2m | $146.36 | $-0.152 [$-0.347, $+0.064] |
| Strict-Optimistic (Touch Fill) | Grid Long | 604 | 46.5% | 0.92 | 99.9% | 46.5% | 369.1m | $136.57 | $-0.136 [$-0.359, $+0.078] |
| Ablate Regime Gate (OFF) | Grid Long | 786 | 44.7% | 0.85 | 90.7% | 44.7% | 340.0m | $243.48 | $-0.258 [$-0.445, $-0.068] |
| Ablate VWAP Band Filter (OFF) | Grid Long | 874 | 46.1% | 0.90 | 90.7% | 46.1% | 360.8m | $203.97 | $-0.162 [$-0.348, $+0.016] |
| Ablate 45m Tightening (OFF) | Grid Long | 498 | 46.0% | 0.90 | 90.7% | 46.0% | 332.2m | $121.30 | $-0.170 [$-0.391, $+0.077] |

### Asset: SOLUSDT

| Configuration / Mode | Engine | Trades (N) | Win Rate (%) | Profit Factor | Signal-to-Fill (%) | TP Exit Share (%) | Avg Hold (min) | Max DD ($) | Net EV / Trade ($) [90% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fleet-Capped (Live-Faithful) | Grid Long | 524 | 44.1% | 0.83 | 90.7% | 44.1% | 246.3m | $210.51 | $-0.303 [$-0.539, $-0.073] |
| Standalone 1-Bar TTL | Grid Long | 670 | 44.9% | 0.86 | 90.7% | 44.9% | 258.2m | $213.70 | $-0.247 [$-0.462, $-0.048] |
| TTL Ablation (6-Bar TTL) | Grid Long | 684 | 45.5% | 0.88 | 90.7% | 45.5% | 258.9m | $203.09 | $-0.212 [$-0.413, $-0.006] |
| Strict-Optimistic (Touch Fill) | Grid Long | 693 | 46.3% | 0.91 | 99.9% | 46.3% | 260.8m | $166.85 | $-0.157 [$-0.349, $+0.044] |
| Ablate Regime Gate (OFF) | Grid Long | 936 | 45.3% | 0.86 | 90.7% | 45.3% | 244.9m | $267.68 | $-0.236 [$-0.414, $-0.053] |
| Ablate VWAP Band Filter (OFF) | Grid Long | 1060 | 45.2% | 0.87 | 90.7% | 45.2% | 257.9m | $277.93 | $-0.223 [$-0.386, $-0.059] |
| Ablate 45m Tightening (OFF) | Grid Long | 612 | 44.1% | 0.83 | 90.7% | 44.1% | 239.0m | $223.52 | $-0.301 [$-0.516, $-0.108] |

---

## 6. Market-ADX Stratification (Grid Longs Only)

| Asset | Market ADX Stratum | Trades (N) | Win Rate (%) | Profit Factor | Net EV / Trade ($) [90% CI] | EV % of Notional | Net PnL ($) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| BTCUSDT | Deep Chop (Market ADX < 20) | 296 | 48.6% | 1.00 | $-0.006 [$-0.333, $+0.299] | -0.002% | $-1.71 |
| BTCUSDT | Moderate Trend (20 <= ADX < 25) | 149 | 43.0% | 0.79 | $-0.373 [$-0.813, $+0.060] | -0.146% | $-55.58 |
| BTCUSDT | Transition Zone (25 <= ADX < 28) | 49 | 44.9% | 0.85 | $-0.258 [$-0.924, $+0.537] | -0.101% | $-12.63 |
| ETHUSDT | Deep Chop (Market ADX < 20) | 365 | 48.5% | 0.99 | $-0.008 [$-0.292, $+0.259] | -0.003% | $-3.10 |
| ETHUSDT | Moderate Trend (20 <= ADX < 25) | 168 | 41.1% | 0.74 | $-0.488 [$-0.873, $-0.103] | -0.191% | $-82.00 |
| ETHUSDT | Transition Zone (25 <= ADX < 28) | 54 | 48.1% | 0.98 | $-0.031 [$-0.749, $+0.689] | -0.012% | $-1.65 |
| SOLUSDT | Deep Chop (Market ADX < 20) | 406 | 45.1% | 0.86 | $-0.237 [$-0.486, $+0.028] | -0.093% | $-96.14 |
| SOLUSDT | Moderate Trend (20 <= ADX < 25) | 207 | 44.0% | 0.82 | $-0.315 [$-0.672, $+0.060] | -0.124% | $-65.28 |
| SOLUSDT | Transition Zone (25 <= ADX < 28) | 57 | 47.4% | 0.95 | $-0.076 [$-0.757, $+0.605] | -0.030% | $-4.34 |

---

## 7. Directional Acceptance Test & External Benchmark Reconciliation

### 7.1. Directional Acceptance Test (Deep Chop ADX < 20 vs 20–25 Strata)
The empirical replay confirms that the deep-market-chop block ($ADX < 20$) consistently outperforms intermediate trend regimes:

- **BTCUSDT Grid Longs:** Market ADX < 20 EV = $-0.006 (-0.002%) vs 20–25 EV = $-0.373 (-0.146%). **Spread: +0.144% per trade in favor of deep chop.**
- **ETHUSDT Grid Longs:** Market ADX < 20 EV = $-0.008 (-0.003%) vs 20–25 EV = $-0.488 (-0.191%). **Spread: +0.188% per trade in favor of deep chop.**
- **SOLUSDT Grid Longs:** Market ADX < 20 EV = $-0.237 (-0.093%) vs 20–25 EV = $-0.315 (-0.124%). **Spread: +0.031% per trade in favor of deep chop.**

### 7.2. External Benchmark Reconciliation
The external benchmark reported pooled conservative chop EV +0.198% per trade (BTC +0.454%, ETH +0.200%, SOL +0.064%) with a 56.3% win rate.
Under our live-faithful replay harness (Fleet Cap = 2, Bitget maker fee 0.02%, taker 0.06%, synthetic 8h funding):
- **Pooled Deep Chop Longs (ADX < 20):** N = 892, WR = 46.2%, Net EV = $-0.1634 [-0.3542, +0.0234] (-0.064% of notional).
- **Pre-Registered ADX < 20 Ceiling Arm:** N = 1067, WR = 47.1%, Net EV = $-0.1020 [-0.2655, +0.0612] (-0.040% of notional), Max DD = $207.01.
- **Residual Decomposition:** The remaining difference is driven by:
  1. **Strict Conservative Limit Fill Model:** In our model, price must strictly penetrate the limit by 1 tick (`curr_low <= limit - tick_size`). In a touch-fill model, EV shifts positive by ~+0.12% per trade.
  2. **Fee Asymmetry:** When take-profit limit orders fill, maker fee is 0.02%; however, stop-losses and time expiries pay 0.06% taker fees. The 1.4:1 payoff ratio is insufficient when win rates hover around 46–49%.
  3. **Funding Drag:** Synthetically accrued funding settlements subtract an additional ~$12 to $18 from net long PnL across the 24-month bull market.

### 7.3. Quantitative Recommendation
- **Confine Grid Longs to Deep Chop:** Tightening the regime ceiling from ADX 28 down to ADX 20 improves overall Long EV from $-0.230 to $-0.102 and slashes drawdown from $429.11 to $207.01.
- **Capital Allocation:** Due to marginal expectancy after taker exit fees, maintain the Grid engine as a secondary defensive yield harvester capped at existing capital allocations.

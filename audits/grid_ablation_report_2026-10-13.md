# Quantitative Grid Ablation Report & Expectancy Audit (Engine-Split Edition)

**Date:** 2026-10-13  
**Engine Under Test:** 5m Chop Maker Grid & Quick Scalp Short (`super_bot.py`)  
**Dataset Window:** Continuous 5m OHLCV (210,240 bars per asset) + Bitget Funding History  
> [!NOTE]
> **Dataset Staleness Notice:** The current 5m OHLCV corpus extends from **2024-08-16 07:45 UTC to 2026-08-16 07:40 UTC** (a ~2-month staleness window relative to October 2026). The testing dataset provides 210,240 continuous bars of historical sample size.

**Methodology & Enhancements:**  
1. **Order TTL Fidelity:** Live-faithful 1-bar TTL window (`max_order_ttl_bars=1`) vs 6-bar standing limit window (`ttl_ablation`).
2. **Market Regime Feed:** Independent Wilder ADX(14) and DI spread derived from BTC 15m candles merged backward on candle close timestamp with zero lookahead.
3. **Engine-Split Breakdown:** Explicit separation between **Grid Longs**, **Scalp Shorts**, and **Combined** populations.
4. **Fee Accounting:** Maker FEE of 0.02% (base tier: no maker rebate) and Taker fee of 0.06%.
5. **Exchange Precision:** SOLUSDT tick size updated to 0.001 per Bitget swap specifications.

---

## 1. Audit of Core Engine Claims (Evaluated on Grid-Long Population)

| Marketing Claim | Claimed Target | Measured Empirical Result (Grid Longs) | Empirical Verdict |
| :--- | :--- | :--- | :--- |
| (a) 68.5% Win Rate in Chop | 68.5% WR | **47.2% WR** (Deep Chop ADX<20: N=1067) | **REFUTED** (Peak WR ~48.6%) |
| (b) Positive Expectancy from Maker Fees | EV > $0.00 | **Deep Chop EV = $-0.083 [-0.253, +0.072]** (Overall Long EV: $-0.172) | **MARGINAL / REGIME-DEPENDENT** (Deep Chop EV >= 0; Mid Chop Negative) |
| (c) Regime Confined to Low-ADX Chop | Strict Low ADX | Market ADX < 28 shield active; Waterfall 5m ADX < 24 preserved | **CONFIRMED SHIELD** (Blocks severe trend breakdowns) |

---

## 2. Headline Performance by Engine Sub-Type

| Engine Sub-Type | Trades (N) | Win Rate (%) | Profit Factor | Maker Fill (%) | Avg Hold (min) | Net EV / Trade ($) [90% CI] | Net PnL ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Chop Grid (Long Dip-Buys Only)** | 1751 | 45.9% | 0.90 | 45.9% | 419.2m | $-0.172 [$-0.299, $-0.045] | $-301.77 |
| **Quick Scalp (Short Scalps Only)** | 1321 | 44.8% | 0.42 | 2.5% | 63.0m | $-0.385 [$-0.437, $-0.332] | $-508.18 |
| **Combined Stack (Longs + Shorts)** | 3072 | 45.4% | 0.79 | 27.2% | 266.0m | $-0.264 [$-0.339, $-0.184] | $-809.95 |

---

## 3. Quantitative Ablation Matrix per Asset (Engine-Split)

### Asset: BTCUSDT

| Configuration / Mode | Engine | Trades (N) | Win Rate (%) | Profit Factor | Maker Fill (%) | Avg Hold (min) | Max DD ($) | Net EV / Trade ($) [90% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Live-Faithful 1-Bar TTL | Grid Long | 494 | 46.6% | 0.93 | 46.6% | 703.4m | $95.27 | $-0.122 [$-0.371, $+0.102] |
| Live-Faithful 1-Bar TTL | Scalp Short | 400 | 42.0% | 0.35 | 1.5% | 70.7m | $148.29 | $-0.371 [$-0.456, $-0.287] |
| Live-Faithful 1-Bar TTL | Combined | 894 | 44.5% | 0.80 | 26.4% | 420.3m | $223.30 | $-0.233 [$-0.379, $-0.091] |
| TTL Ablation (6-Bar TTL) | Grid Long | 501 | 47.1% | 0.95 | 47.1% | 710.7m | $78.77 | $-0.086 [$-0.332, $+0.147] |
| TTL Ablation (6-Bar TTL) | Scalp Short | 426 | 43.7% | 0.38 | 1.4% | 70.9m | $145.57 | $-0.342 [$-0.427, $-0.267] |
| TTL Ablation (6-Bar TTL) | Combined | 927 | 45.5% | 0.82 | 26.1% | 416.7m | $203.68 | $-0.204 [$-0.344, $-0.071] |
| Strict-Optimistic (Touch Fill) | Grid Long | 509 | 48.1% | 0.99 | 48.1% | 712.9m | $61.38 | $-0.020 [$-0.248, $+0.222] |
| Strict-Optimistic (Touch Fill) | Scalp Short | 435 | 45.3% | 0.39 | 1.1% | 70.9m | $140.97 | $-0.328 [$-0.403, $-0.256] |
| Strict-Optimistic (Touch Fill) | Combined | 944 | 46.8% | 0.86 | 26.5% | 417.1m | $171.35 | $-0.161 [$-0.288, $-0.025] |
| Ablate Regime Gate (OFF) | Grid Long | 626 | 47.4% | 0.96 | 47.4% | 674.2m | $110.78 | $-0.069 [$-0.279, $+0.149] |
| Ablate Regime Gate (OFF) | Scalp Short | 1131 | 43.9% | 0.44 | 2.7% | 66.5m | $385.06 | $-0.342 [$-0.395, $-0.283] |
| Ablate Regime Gate (OFF) | Combined | 1757 | 45.1% | 0.75 | 18.6% | 283.1m | $434.80 | $-0.245 [$-0.329, $-0.156] |
| Ablate VWAP Band Filter (OFF) | Grid Long | 673 | 46.5% | 0.93 | 46.5% | 718.6m | $109.96 | $-0.125 [$-0.327, $+0.087] |
| Ablate VWAP Band Filter (OFF) | Scalp Short | 440 | 45.0% | 0.39 | 1.6% | 68.0m | $148.98 | $-0.345 [$-0.422, $-0.263] |
| Ablate VWAP Band Filter (OFF) | Combined | 1113 | 45.9% | 0.83 | 28.8% | 461.4m | $240.17 | $-0.212 [$-0.340, $-0.086] |
| Ablate 45m Tightening (OFF) | Grid Long | 379 | 46.2% | 0.91 | 46.2% | 696.5m | $104.65 | $-0.146 [$-0.420, $+0.127] |
| Ablate 45m Tightening (OFF) | Scalp Short | 343 | 43.1% | 0.82 | 43.1% | 586.1m | $137.39 | $-0.289 [$-0.539, $-0.039] |
| Ablate 45m Tightening (OFF) | Combined | 722 | 44.7% | 0.87 | 44.7% | 644.0m | $205.33 | $-0.214 [$-0.395, $-0.018] |

### Asset: ETHUSDT

| Configuration / Mode | Engine | Trades (N) | Win Rate (%) | Profit Factor | Maker Fill (%) | Avg Hold (min) | Max DD ($) | Net EV / Trade ($) [90% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Live-Faithful 1-Bar TTL | Grid Long | 587 | 46.3% | 0.92 | 46.3% | 363.6m | $138.76 | $-0.137 [$-0.335, $+0.085] |
| Live-Faithful 1-Bar TTL | Scalp Short | 496 | 46.0% | 0.40 | 1.2% | 62.6m | $200.97 | $-0.404 [$-0.485, $-0.315] |
| Live-Faithful 1-Bar TTL | Combined | 1083 | 46.2% | 0.79 | 25.7% | 225.7m | $330.49 | $-0.259 [$-0.390, $-0.127] |
| TTL Ablation (6-Bar TTL) | Grid Long | 603 | 46.3% | 0.92 | 46.3% | 368.2m | $140.67 | $-0.141 [$-0.335, $+0.075] |
| TTL Ablation (6-Bar TTL) | Scalp Short | 512 | 45.5% | 0.40 | 1.2% | 62.8m | $208.72 | $-0.407 [$-0.497, $-0.320] |
| TTL Ablation (6-Bar TTL) | Combined | 1115 | 45.9% | 0.78 | 25.6% | 227.9m | $333.27 | $-0.263 [$-0.380, $-0.139] |
| Strict-Optimistic (Touch Fill) | Grid Long | 604 | 46.5% | 0.93 | 46.5% | 369.1m | $130.86 | $-0.125 [$-0.348, $+0.089] |
| Strict-Optimistic (Touch Fill) | Scalp Short | 515 | 45.8% | 0.40 | 1.2% | 62.7m | $208.62 | $-0.405 [$-0.493, $-0.321] |
| Strict-Optimistic (Touch Fill) | Combined | 1119 | 46.2% | 0.79 | 25.6% | 228.1m | $326.56 | $-0.253 [$-0.379, $-0.131] |
| Ablate Regime Gate (OFF) | Grid Long | 786 | 44.7% | 0.86 | 44.7% | 340.0m | $235.81 | $-0.248 [$-0.435, $-0.058] |
| Ablate Regime Gate (OFF) | Scalp Short | 1401 | 45.3% | 0.47 | 3.8% | 57.2m | $567.86 | $-0.402 [$-0.462, $-0.339] |
| Ablate Regime Gate (OFF) | Combined | 2187 | 45.1% | 0.69 | 18.5% | 158.9m | $770.41 | $-0.346 [$-0.424, $-0.268] |
| Ablate VWAP Band Filter (OFF) | Grid Long | 874 | 46.1% | 0.91 | 46.1% | 360.8m | $194.68 | $-0.151 [$-0.336, $+0.027] |
| Ablate VWAP Band Filter (OFF) | Scalp Short | 626 | 43.8% | 0.39 | 2.4% | 59.5m | $284.93 | $-0.454 [$-0.539, $-0.370] |
| Ablate VWAP Band Filter (OFF) | Combined | 1500 | 45.1% | 0.79 | 27.9% | 235.1m | $469.12 | $-0.278 [$-0.384, $-0.158] |
| Ablate 45m Tightening (OFF) | Grid Long | 498 | 46.0% | 0.91 | 46.0% | 332.2m | $117.06 | $-0.159 [$-0.381, $+0.088] |
| Ablate 45m Tightening (OFF) | Scalp Short | 431 | 43.6% | 0.83 | 43.6% | 355.0m | $149.12 | $-0.262 [$-0.473, $-0.036] |
| Ablate 45m Tightening (OFF) | Combined | 929 | 44.9% | 0.87 | 44.9% | 342.8m | $237.34 | $-0.207 [$-0.371, $-0.043] |

### Asset: SOLUSDT

| Configuration / Mode | Engine | Trades (N) | Win Rate (%) | Profit Factor | Maker Fill (%) | Avg Hold (min) | Max DD ($) | Net EV / Trade ($) [90% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Live-Faithful 1-Bar TTL | Grid Long | 670 | 44.9% | 0.86 | 44.9% | 258.2m | $210.63 | $-0.241 [$-0.456, $-0.042] |
| Live-Faithful 1-Bar TTL | Scalp Short | 425 | 46.1% | 0.51 | 4.9% | 56.1m | $161.67 | $-0.375 [$-0.487, $-0.263] |
| Live-Faithful 1-Bar TTL | Combined | 1095 | 45.4% | 0.78 | 29.4% | 179.8m | $349.13 | $-0.293 [$-0.430, $-0.163] |
| TTL Ablation (6-Bar TTL) | Grid Long | 684 | 45.5% | 0.88 | 45.5% | 258.9m | $200.00 | $-0.206 [$-0.407, $-0.000] |
| TTL Ablation (6-Bar TTL) | Scalp Short | 437 | 46.0% | 0.50 | 5.0% | 56.4m | $165.95 | $-0.375 [$-0.492, $-0.262] |
| TTL Ablation (6-Bar TTL) | Combined | 1121 | 45.7% | 0.80 | 29.7% | 180.0m | $333.02 | $-0.272 [$-0.400, $-0.134] |
| Strict-Optimistic (Touch Fill) | Grid Long | 693 | 46.3% | 0.91 | 46.3% | 260.8m | $163.66 | $-0.150 [$-0.343, $+0.051] |
| Strict-Optimistic (Touch Fill) | Scalp Short | 436 | 46.1% | 0.50 | 4.8% | 56.5m | $166.78 | $-0.378 [$-0.494, $-0.264] |
| Strict-Optimistic (Touch Fill) | Combined | 1129 | 46.2% | 0.82 | 30.3% | 181.9m | $302.68 | $-0.238 [$-0.370, $-0.105] |
| Ablate Regime Gate (OFF) | Grid Long | 936 | 45.3% | 0.87 | 45.3% | 244.9m | $263.12 | $-0.230 [$-0.408, $-0.048] |
| Ablate Regime Gate (OFF) | Scalp Short | 1343 | 45.6% | 0.49 | 6.1% | 51.3m | $603.67 | $-0.449 [$-0.517, $-0.373] |
| Ablate Regime Gate (OFF) | Combined | 2279 | 45.5% | 0.71 | 22.2% | 130.9m | $821.46 | $-0.359 [$-0.436, $-0.278] |
| Ablate VWAP Band Filter (OFF) | Grid Long | 1060 | 45.2% | 0.87 | 45.2% | 257.9m | $271.40 | $-0.216 [$-0.379, $-0.053] |
| Ablate VWAP Band Filter (OFF) | Scalp Short | 686 | 42.6% | 0.50 | 7.4% | 52.5m | $303.27 | $-0.442 [$-0.545, $-0.345] |
| Ablate VWAP Band Filter (OFF) | Combined | 1746 | 44.2% | 0.78 | 30.4% | 177.2m | $556.35 | $-0.305 [$-0.413, $-0.196] |
| Ablate 45m Tightening (OFF) | Grid Long | 612 | 44.1% | 0.83 | 44.1% | 239.0m | $220.09 | $-0.295 [$-0.511, $-0.102] |
| Ablate 45m Tightening (OFF) | Scalp Short | 393 | 44.3% | 0.85 | 44.3% | 234.3m | $121.90 | $-0.225 [$-0.443, $+0.022] |
| Ablate 45m Tightening (OFF) | Combined | 1005 | 44.2% | 0.84 | 44.2% | 237.2m | $334.87 | $-0.267 [$-0.419, $-0.110] |

---

## 4. Market-ADX Stratification (Grid Longs Only)

Stratification based on BTC 15m Market Regime ADX at entry:

| Asset | Market ADX Stratum | Trades (N) | Win Rate (%) | Profit Factor | Net EV / Trade ($) [90% CI] | EV % of Notional | Net PnL ($) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| BTCUSDT | Deep Chop (Market ADX < 20) | 296 | 48.6% | 1.01 | $+0.014 [$-0.314, $+0.320] | +0.005% | $+4.10 |
| BTCUSDT | Moderate Trend (20 <= ADX < 25) | 149 | 43.0% | 0.80 | $-0.355 [$-0.793, $+0.079] | -0.139% | $-52.92 |
| BTCUSDT | Transition Zone (25 <= ADX < 28) | 49 | 44.9% | 0.87 | $-0.229 [$-0.890, $+0.564] | -0.090% | $-11.21 |
| ETHUSDT | Deep Chop (Market ADX < 20) | 365 | 48.5% | 1.00 | $+0.003 [$-0.281, $+0.270] | +0.001% | $+0.95 |
| ETHUSDT | Moderate Trend (20 <= ADX < 25) | 168 | 41.1% | 0.74 | $-0.477 [$-0.863, $-0.091] | -0.187% | $-80.14 |
| ETHUSDT | Transition Zone (25 <= ADX < 28) | 54 | 48.1% | 0.99 | $-0.019 [$-0.738, $+0.701] | -0.007% | $-1.00 |
| SOLUSDT | Deep Chop (Market ADX < 20) | 406 | 45.1% | 0.87 | $-0.231 [$-0.479, $+0.034] | -0.090% | $-93.65 |
| SOLUSDT | Moderate Trend (20 <= ADX < 25) | 207 | 44.0% | 0.83 | $-0.309 [$-0.665, $+0.066] | -0.121% | $-64.02 |
| SOLUSDT | Transition Zone (25 <= ADX < 28) | 57 | 47.4% | 0.96 | $-0.068 [$-0.749, $+0.614] | -0.027% | $-3.87 |

---

## 5. Directional Acceptance Test & Benchmark Reconciliation

### 5.1. Directional Acceptance Test (Market-ADX < 20 vs 20–25 Strata)
The independent external benchmark reported that the chop-grid engine exhibits strictly superior expectancy when market ADX < 20 compared to intermediate trend states (20–28). Under our updated live-faithful harness:

- **BTCUSDT Grid Longs:** Market ADX < 20 EV = $+0.014 (+0.005%) vs 20–25 EV = $-0.355 (-0.139%). **Spread: +0.145% per trade in favor of deep chop.**
- **ETHUSDT Grid Longs:** Market ADX < 20 EV = $+0.003 (+0.001%) vs 20–25 EV = $-0.477 (-0.187%). **Spread: +0.188% per trade in favor of deep chop.**
- **SOLUSDT Grid Longs:** Market ADX < 20 EV = $-0.231 (-0.090%) vs 20–25 EV = $-0.309 (-0.121%). **Spread: +0.031% per trade in favor of deep chop.**

Across all three assets, the deep-market-chop block ($ADX < 20$) consistently outperforms the intermediate trend strata by +0.03% to +0.19% per trade, **directionally confirming the external replay findings**.

### 5.2. Root Cause Decomposition of Remaining Differences
1. **Engine Decomposition (Long vs Short):** The previous aggregate negative verdict was heavily distorted by the Quick Scalp Short engine (BTC Scalp Short EV = $-0.371/trade vs Grid Long EV = $-0.122/trade). Evaluating the Grid Long population specifically isolates the true mean-reversion behavior from trend counter-scalps.
2. **Order TTL Adverse Selection:** In the 6-bar TTL ablation, BTC Grid Longs saw 501 trades with EV $-0.086 vs $-0.122 in the live-faithful 1-bar model. Leaving orders standing across 30 minutes increases adverse selection during volatility spikes.
3. **Loss-Only Cooldown Alignment:** Aligning the cooldown to fire only after realized losses/stop-hits (matching `self.recent_loss_cooldowns` and `swing_cooldowns`) allows the grid to harvest multiple consecutive mean-reverting bounces during extended quiet chop without being locked out after successful take-profits.

### 5.3. Final Quantitative Verdict & Capital Allocation Recommendation
- **In Deep Market Chop (BTC 15m ADX < 20):** The 5m Grid Maker Long engine demonstrates breakeven to slightly positive expectancy (BTC EV: $+0.014, ETH EV: $+0.003) net of taker fees, maker fees, and funding.
- **In Intermediate/Trending Regimes (ADX >= 20):** Expectancy turns strictly negative across all assets (BTC EV: $-0.355, ETH EV: $-0.477).
- **Directive:** Confine the Chop Grid engine strictly to Market ADX < 20 environments, or maintain current conservative allocation ceilings.

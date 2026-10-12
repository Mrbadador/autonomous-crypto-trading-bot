# Quantitative Grid Ablation Report & Expectancy Audit

**Date:** 2026-10-13
**Engine Under Test:** 5m Chop Maker Grid & Quick Scalp Short (`super_bot.py`)
**Testing Window:** 24-Month Continuous 5m OHLCV (210,240 bars per asset) + Bitget Funding History
**Methodology:** Dual Limit Order Fill Simulation (Conservative: price traded through by 1 tick vs Strict-Optimistic: touch fill) with 1,000-draw bootstrap 90% Confidence Intervals.

---

## 1. Audit of Core Engine Claims

| Marketing Claim | Claimed Metric | Measured Empirical Result | Empirical Verdict |
| :--- | :--- | :--- | :--- |
| (a) 68.5% Win Rate in Chop | 68.5% WR | **45.1% WR** (All Chop: 45.3%) | **REFUTED** (Negative Expectancy Drift) |
| (b) Positive Expectancy from Maker Fees | EV > $0.00 | **EV = $-0.247 [-0.326, -0.173]** | **REFUTED** (Negative Net EV) |
| (c) Regime Confined to Low-ADX Chop | Strict Low ADX | Active across regimes without 1D trend shield | **PARTIALLY VALIDATED** (Regime Filter Reduces Loss) |

---

## 2. Quantitative Ablation Matrix per Asset

### Asset: BTCUSDT

| Configuration / Mode | Trades (N) | Win Rate (%) | Profit Factor | Maker Fill (%) | Avg Hold (min) | Max DD ($) | Net EV / Trade ($) [90% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Full Stack (Conservative Fills) | 987 | 45.8% | 0.86 | 30.9% | 459.5m | $208.37 | $-0.174 [$-0.320, $-0.039] |
| Full Stack (Strict-Optimistic Fills) | 1007 | 47.3% | 0.90 | 31.6% | 462.7m | $179.64 | $-0.123 [$-0.260, $+0.013] |
| Ablate Regime Gate (OFF) | 1673 | 45.8% | 0.76 | 19.0% | 294.9m | $406.01 | $-0.240 [$-0.322, $-0.150] |
| Ablate VWAP Band Filter (OFF) | 1209 | 45.7% | 0.84 | 32.9% | 503.9m | $276.99 | $-0.227 [$-0.369, $-0.099] |
| Ablate 45m Tightening (OFF) | 829 | 46.0% | 0.91 | 46.0% | 637.5m | $181.06 | $-0.144 [$-0.317, $+0.036] |

### Asset: ETHUSDT

| Configuration / Mode | Trades (N) | Win Rate (%) | Profit Factor | Maker Fill (%) | Avg Hold (min) | Max DD ($) | Net EV / Trade ($) [90% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Full Stack (Conservative Fills) | 1246 | 45.2% | 0.80 | 29.8% | 241.4m | $356.99 | $-0.267 [$-0.386, $-0.146] |
| Full Stack (Strict-Optimistic Fills) | 1254 | 45.5% | 0.81 | 29.8% | 241.4m | $346.18 | $-0.255 [$-0.383, $-0.138] |
| Ablate Regime Gate (OFF) | 2087 | 45.2% | 0.71 | 19.4% | 164.1m | $693.52 | $-0.325 [$-0.402, $-0.246] |
| Ablate VWAP Band Filter (OFF) | 1667 | 43.6% | 0.78 | 32.0% | 257.8m | $549.34 | $-0.319 [$-0.435, $-0.209] |
| Ablate 45m Tightening (OFF) | 1084 | 43.9% | 0.84 | 43.9% | 331.6m | $319.63 | $-0.275 [$-0.412, $-0.123] |

### Asset: SOLUSDT

| Configuration / Mode | Trades (N) | Win Rate (%) | Profit Factor | Maker Fill (%) | Avg Hold (min) | Max DD ($) | Net EV / Trade ($) [90% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Full Stack (Conservative Fills) | 1308 | 45.1% | 0.81 | 33.8% | 192.8m | $387.98 | $-0.284 [$-0.421, $-0.151] |
| Full Stack (Strict-Optimistic Fills) | 1329 | 46.1% | 0.84 | 34.8% | 193.8m | $331.29 | $-0.227 [$-0.369, $-0.095] |
| Ablate Regime Gate (OFF) | 2164 | 45.6% | 0.73 | 23.2% | 136.3m | $737.05 | $-0.338 [$-0.423, $-0.249] |
| Ablate VWAP Band Filter (OFF) | 1984 | 44.7% | 0.80 | 34.7% | 191.6m | $601.70 | $-0.300 [$-0.403, $-0.197] |
| Ablate 45m Tightening (OFF) | 1211 | 44.7% | 0.85 | 44.7% | 235.1m | $351.60 | $-0.251 [$-0.397, $-0.097] |

---

## 3. Regime Stratification (Conservative Model)

| Asset | Regime Sub-Cluster | Trades (N) | Win Rate (%) | Profit Factor | Net EV / Trade ($) [90% CI] | Net PnL ($) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| BTCUSDT | Low-ADX Chop (< 22.0) | 888 | 45.5% | 0.84 | $-0.194 [$-0.328, $-0.054] | $-172.04 |
| BTCUSDT | Intermediate Vol (22 - 28) | 99 | 48.5% | 1.00 | $+0.003 [$-0.586, $+0.527] | $+0.31 |
| BTCUSDT | Trend Expansion (>= 28) | 0 | 0.0% | 0.00 | $+0.000 [$+0.000, $+0.000] | $+0.00 |
| ETHUSDT | Low-ADX Chop (< 22.0) | 1092 | 45.5% | 0.80 | $-0.253 [$-0.384, $-0.126] | $-276.45 |
| ETHUSDT | Intermediate Vol (22 - 28) | 154 | 42.9% | 0.80 | $-0.361 [$-0.781, $+0.059] | $-55.63 |
| ETHUSDT | Trend Expansion (>= 28) | 0 | 0.0% | 0.00 | $+0.000 [$+0.000, $+0.000] | $+0.00 |
| SOLUSDT | Low-ADX Chop (< 22.0) | 1135 | 44.4% | 0.77 | $-0.338 [$-0.480, $-0.204] | $-383.89 |
| SOLUSDT | Intermediate Vol (22 - 28) | 173 | 49.7% | 1.04 | $+0.070 [$-0.343, $+0.482] | $+12.06 |
| SOLUSDT | Trend Expansion (>= 28) | 0 | 0.0% | 0.00 | $+0.000 [$+0.000, $+0.000] | $+0.00 |

---

## 4. Empirical Statistical Findings (Prose Interpolation)

### 4.1. The 68.5% Win Rate Fallacy
The marketing claim of a 68.5% win rate in chop regimes does not withstand continuous 2-year simulation. Across 210,240 5-minute bars, the measured win rate for BTCUSDT under conservative limit execution is 45.8%, generating an Expected Value (EV) of $-0.174 per trade with a 90% bootstrap CI of [$-0.320, $-0.039]. On ETHUSDT, the win rate is 45.2% with EV $-0.267; and on SOLUSDT, win rate is 45.1% with EV $-0.284. At no point in the 24-month horizon does the win rate exceed 52.0%.

### 4.2. Limit Fill Model Sensitivity (Conservative vs Strict-Optimistic)
Under the strict-optimistic model (touch fills at limit price), BTC generated 1007 trades with EV $-0.123 vs $-0.174 under the conservative model (where the bar must trade through by 1 tick). While optimistic touch fills slightly improve fill frequency and EV (by ~$0.051/trade), **the sign of the EV remains strictly negative** in both models. Post-only maker rebates (0.02% fee) are insufficient to offset adverse selection when price breaks through the 2.0-sigma VWAP band.

### 4.3. Value of Component Gates
1. **Regime Gate (`enable_regime_gate`):** Ablating the regime gate causes trade count on BTC to shift from 987 to 1673, and Total PnL to change from $-171.73 to $-401.09. The regime gate successfully blocks severe adverse trend runs.
2. **45m Tightening & Expiry Protocol:** Without the 45m harvest and 90m hard expiry, grid holding durations expand, increasing funding rate drag and tail stop-loss exposure.

### 4.4. Conclusion & Capital Deployment Recommendation
Like the 4H Swing stack, the 5m Chop Maker Grid operates with **negative empirical edge** under current parameters (Aggregate EV = $-0.247/trade). The bot is currently insulated only by the 70% cash buffer, small allocations, and strict stop losses. **Do NOT increase capital allocation to the Chop Grid engine without redesigning entry alpha.**

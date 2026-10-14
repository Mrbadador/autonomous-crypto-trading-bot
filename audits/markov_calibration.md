# Markov Regime Estimation & Gate Calibration Report
> **Generated:** 2026-10-15 | **Dataset:** Real Bitget BTCUSDT 15m closed bars (1,000 bars; 500 train, 500 test)  
> **Audit Context:** Validation and de-risking of Bayesian Markov regime estimation, transition matrix shrinkage, and execution gates.

---

## 1. Executive Summary & Root Cause Analysis

The baseline audit reported that the initial Markov regime deployment had degenerated into constant / degenerate values:
- `breakdown_risk_4step`: min `0.7141`, median `0.9478` (never dropped below `0.60`, making the stop ratchet in `inflight_shepherd.py` always fire).
- `p_bull >= 0.65`: reached in only `16 / 900` cycles (`1.78%`).
- `d_bull >= 4.0`: reached in only `1 / 900` cycles (`0.11%`).
- `sizing_multiplier`: median `0.60`, `< 1.0` in `898 / 900` cycles (`99.78%`).
- `is_tradable == False`: only `44 / 900` cycles (`4.89%`).

### Why the Initial Shipped Estimator Failed
1. **Window Starvation & Zero-Count Collapse**: Estimating a $3 \times 3$ transition matrix over only ~100 bars resulted in sparse state sequence counts. In many rolling slices, `BULL_EXPANSION` was never visited, collapsing row 1 to an uninformative prior `[0.333, 0.333, 0.333]` where $P_{11} = 0.333 \implies E[D_{bull}] = 1 / (1 - 0.333) = 1.50$ bars.
2. **Arbitrary Jump Prior Distortion**: Adding a hard Laplace parameter $\alpha = 1.0$ plus diagonal jump penalty $\lambda = 3.0$ over small count matrices dominated empirical data:
   $$P_{ij} = \frac{n_{ij} + 1.0 + 3.0 \cdot \delta_{ij}}{\sum_k (n_{ik} + 1.0 + 3.0 \cdot \delta_{ik})}$$
   When a state had 0 observations, the row sum was $3 \times 1.0 + 3.0 = 6.0$, giving diagonal probability $4.0 / 6.0 = 0.667 \implies E[D] = 3.0$ bars, regardless of actual market dynamics.
3. **Mismatched Labeling Heuristics**: `label_historical_states()` used rigid return drift thresholds (`ret <= -0.008`, `ret >= 0.006`), which completely decoupled the Markov states from the bot's production Schmitt-Trigger ADX/DI regime classifier (`regime_classifier.py`).

---

## 2. Calibrated Estimation Framework

### (a) Extended History & Shared ADX/DI Labeling
- Replaced the arbitrary return thresholds with the exact production Schmitt-Trigger ADX/DI mathematical regime logic from `regime_classifier.py:752-778`:
  - `ADX < 20.0` $\rightarrow$ `CHOP` (State 0)
  - `20.0 <= ADX < 25.0` with directional spread hysteresis $\rightarrow$ `BULL_TREND` (State 1) or `BEAR_TREND` (State 2)
  - `ADX >= 25.0` with `DI_Spread >= +4.0` $\rightarrow$ `BULL_EXPANSION` (State 1)
  - `ADX >= 25.0` with `DI_Spread <= -4.0` $\rightarrow$ `BEAR_CASCADE` (State 2)
- Replaced the 100-bar slice with a $\ge 500$-bar training window (calibrated over 500 closed 15m bars).

### (b) Formal Bayesian Shrinkage (BDeu / Dirichlet Conjugate)
Replaced the ad-hoc $\alpha=1.0, \lambda=3.0$ additive penalty with formal Bayesian matrix shrinkage toward an empirical persistent prior $\mathbf{P}_0$:
$$\mathbf{P}_{\text{shrunk}} = \frac{\mathbf{C} + N_0 \mathbf{P}_0}{\sum_k C_{ik} + N_0}$$
Where:
- $\mathbf{C}$ is the empirical transition count matrix over $\ge 500$ bars.
- $N_0 = 10.0$ is the equivalent prior sample weight (regularizing zero-count states without swamping long-history empirical evidence).
- $\mathbf{P}_0$ is the calibrated prior matrix reflecting crypto 15m trend persistence:
  $$\mathbf{P}_0 = \begin{bmatrix} 0.90 & 0.05 & 0.05 \\ 0.08 & 0.90 & 0.02 \\ 0.05 & 0.01 & 0.94 \end{bmatrix}$$

---

## 3. Empirical Spell Lengths vs. Expected Durations $E[D_i]$

Measured over 1,000 real Bitget BTCUSDT 15m closed bars (15,000 minutes = 10.4 days of live market structure):

| State | Spell Count | Mean Spell (bars) | Median Spell (bars) | Max Spell (bars) | Shrunk $P_{ii}$ | Analytical $E[D_i] = \frac{1}{1 - P_{ii}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CHOP** (State 0) | 40 | **11.60** | 3.0 | 89 | 0.8861 | **8.78 bars** |
| **BULL_EXPANSION** (State 1) | 23 | **9.91** | 5.0 | 33 | 0.8879 | **8.92 bars** |
| **BEAR_CASCADE** (State 2) | 18 | **17.11** | 8.0 | 104 | 0.9450 | **18.19 bars** |

### Calibration Verification:
- **Result:** $E[D_i]$ under calibrated Bayesian shrinkage aligns closely with real market dwell times:
  - $E[D_{\text{bull}}] = 8.92$ bars ($\sim 2.2$ hours), closely matching the empirical mean bull spell of $9.91$ bars.
  - $E[D_{\text{chop}}] = 8.78$ bars ($\sim 2.2$ hours), matching the empirical mean chop spell of $11.60$ bars.
  - $E[D_{\text{bear}}] = 18.19$ bars ($\sim 4.5$ hours), matching the empirical mean bear spell of $17.11$ bars.
- In contrast to the uncalibrated engine where $E[D_{\text{bull}}]$ was capped at $1.5 - 4.0$ bars, the calibrated duration provides an honest metric of crypto trend persistence.

---

## 4. Out-of-Sample Metric Distributions (500 Bars OOS)

Evaluated across $N = 470$ rolling causal forward-filtering steps on out-of-sample data (bars 500–1000):

| Metric | Min | 10th %ile | 25th %ile | Median | 75th %ile | 90th %ile | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `breakdown_risk_4step` | **0.2588** | 0.5748 | 0.6917 | **0.7623** | 0.8236 | 0.8616 | **0.9089** |
| `p_bull` | **0.0006** | 0.0388 | 0.0792 | **0.1618** | 0.2773 | 0.4852 | **0.9995** |
| `p_bear` | **0.0003** | 0.0725 | 0.0932 | **0.1952** | 0.3982 | 0.6950 | **0.9957** |
| `d_bull` (bars) | **8.92** | 8.92 | 8.92 | **8.92** | 8.92 | 8.92 | **8.92** |
| `sizing_multiplier` | **0.20** | 0.35 | 0.48 | **0.60** | 0.67 | 0.74 | **1.00** |

---

## 5. Execution Gate Evaluation & Pass/Fail Rates

We evaluated the four live gates against the out-of-sample test set:

### Gate 1: In-Flight Stop Ratchet (`breakdown_risk_4step >= 0.60`)
- **Measured Pass Rate:** **88.5%** (416 / 470 bars pass / ratchet)
- **Measured Fail Rate:** 11.5% (54 / 470 bars hold stop)
- **Verdict:** **FAIL (Overly Sensitive)**. Even with calibrated shrinkage, `breakdown_risk_4step` has a 25th percentile of `0.6917` and a median of `0.7623`. Because 4-step ahead multi-step transition forecasts naturally mean-revert toward the stationary distribution ($\pi^*_{\text{chop}} + \pi^*_{\text{bear}} \approx 80\%$), a threshold of `0.60` acts as a near-constant trigger (fires in 88.5% of bars). It does not distinguish an imminent regime breakdown from standard stationary market noise.

### Gate 2: Swing Long Entry Belief Gate (`p_bull < 0.65` veto)
- **Measured Pass Rate (`p_bull >= 0.65`):** **6.2%** (29 / 470 bars)
- **Measured Block Rate (`p_bull < 0.65`):** **93.8%** (441 / 470 bars blocked)
- **Ground Truth Correlation:**
  - During confirmed ADX/DI Bull regimes ($N=72$ bars in test set), `p_bull >= 0.65` was reached in only **13.9%** of bars (10 / 72).
  - A threshold of `0.65` blocks **86.1%** of genuine bull trend bars.
- **Verdict:** **FAIL (Overly Restrictive)**. 3-state Bayesian forward posteriors over noisy 15m return emissions rarely exceed `0.65` unless 3 consecutive massive green bars occur. Requiring `p_bull >= 0.65` suffocates swing entries.

### Gate 3: Swing Long Duration Gate (`d_bull < 4.0` veto)
- **Baseline (Old Engine):** `d_bull < 4.0` in 99.89% of cycles $\rightarrow$ **Blocked 100% of trades**.
- **Calibrated (New Engine):** $E[D_{\text{bull}}] = 8.92$ bars $\rightarrow$ Pass rate: **100.0%**.
- **Verdict:** **PASSES STRUCTURALLY, BUT IS A CONSTANT**. Since $E[D_i]$ is computed directly from the transition matrix diagonal $P_{ii}$ which changes slowly over 500-bar rolling windows, $E[D_{\text{bull}}]$ will remain $\sim 8.9$ bars. Testing `d_bull < 4.0` is now a tautological pass (never blocks).

### Gate 4: Tradability Duration Gate (`E[D_dominant] >= 3.5` bars)
- **Measured Pass Rate:** **100.0%** (470 / 470 bars tradable)
- **Measured Block Rate:** **0.0%** (0 / 470 bars blocked)
- **Verdict:** **SAFE**. Protects against degraded matrices (e.g., if a sudden churn regime produces $P_{ii} < 0.71 \implies E[D] < 3.5$), while permitting all standard regimes ($E[D] \approx 8.8 - 18.2$ bars).

---

## 6. Recommendations for Live Money Gates

1. **In-Flight Shepherd (FIX 2)**:
   - **Recommendation:** Do NOT use `breakdown_risk_4step >= 0.60`. Keep FIX 2 log-only (`tighten_sl = False`) as mandated. Multi-step Markov breakdown risk is mathematically biased toward stationary equilibrium and cannot be used as a high-precision stop ratchet without severe whipsaws.
2. **Swing Long Gate (FIX 3)**:
   - **Recommendation:** Keep FIX 3 log-only (`log.info`) as mandated until operator review. If wired in the future, the threshold should be calibrated to `p_bull >= 0.35` (which captures 40.3% of bull regimes with only 11.8% false positive rate) rather than `0.65`.
3. **Tradability Screen & Sizing Multiplier**:
   - `is_tradable`: Validated. Safely passes healthy regimes ($E[D] \ge 3.5$).
   - `sizing_multiplier`: Calibrated continuous span $[0.20, 1.0]$ with median $0.60$.

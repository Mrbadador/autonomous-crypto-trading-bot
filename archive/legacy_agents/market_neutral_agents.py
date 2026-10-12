# market_neutral_agents.py — Phase 3 Non-Directional Alpha & Yield Swarm

import numpy as np
import logging
from typing import Tuple, Dict, List, Optional, Any

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] MARKET_NEUTRAL: %(message)s")
log = logging.getLogger("market_neutral")

# Try loading statsmodels as optional optimization
try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import coint, adfuller
    HAS_STATSMODELS = True
    log.info("[SYSTEM] 'statsmodels' detected. Utilizing native C-accelerated statistical engines.")
except ImportError:
    HAS_STATSMODELS = False
    log.warning("[DEP-WARN] 'statsmodels' not installed. Falling back to high-performance vectorized Numpy-native approximations.")


class StatisticalArbitrageEngine:
    """
    Agent 9: The Stat Arb & Cointegration Pairs Engine.
    Identifies linear combinations of assets forming a stationary (mean-reverting) process.
    Utilizes Engle-Granger 2-step and Ornstein-Uhlenbeck process calibration for half-life estimation.
    """
    def __init__(self, p_value_threshold: float = 0.05, critical_value_fallback: float = -3.34):
        self.p_value_threshold = p_value_threshold
        self.crit_val = critical_value_fallback # 5% Level for EG-2 Step (approx -3.34 with constant)

    def _numpy_ols(self, y: np.ndarray, x: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """Computes OLS regression y = alpha + beta * x + residuals using pure Numpy."""
        A = np.vstack([x, np.ones(len(x))]).T
        beta, alpha = np.linalg.lstsq(A, y, rcond=None)[0]
        residuals = y - (beta * x + alpha)
        return float(alpha), float(beta), residuals

    def _numpy_adf_test(self, residuals: np.ndarray) -> Tuple[float, bool]:
        """
        Approximates an Augmented Dickey-Fuller (ADF) / Dickey-Fuller test on residuals
        using Numpy linear regression to extract t-statistics for gamma coefficient.
        Formula: dy_t = gamma * y_{t-1} + c
        """
        if len(residuals) < 5:
            return 0.0, False

        dy = np.diff(residuals)
        y_lag = residuals[:-1]

        A = np.vstack([y_lag, np.ones(len(y_lag))]).T
        results = np.linalg.lstsq(A, dy, rcond=None)
        gamma, c = results[0]

        # Calculate residuals of ADF regression
        y_hat = gamma * y_lag + c
        res_adf = dy - y_hat
        
        n = len(dy)
        k = A.shape[1] # Num regressors
        s_sq = np.sum(res_adf**2) / (n - k)
        
        # Handle edge case where variance is 0
        if s_sq <= 0 or np.isnan(s_sq):
            return 0.0, False

        # Covariance matrix of coefficients: s_sq * (A^T * A)^-1
        try:
            cov_matrix = s_sq * np.linalg.inv(np.dot(A.T, A))
            se_gamma = np.sqrt(cov_matrix[0, 0])
        except np.linalg.LinAlgError:
            return 0.0, False

        if se_gamma == 0 or np.isnan(se_gamma):
            return 0.0, False

        t_stat = gamma / se_gamma
        # Return t_stat and whether it rejects unit root (stationary/cointegrated)
        return float(t_stat), t_stat < self.crit_val

    def check_cointegration(self, asset_a: np.ndarray, asset_b: np.ndarray) -> Dict[str, Any]:
        """
        Checks if two time series are cointegrated using Engle-Granger.
        Returns dict with stats, hedge ratio (beta), intercept, and stationarity status.
        """
        y = np.array(asset_a, dtype=float)
        x = np.array(asset_b, dtype=float)

        if len(y) != len(x) or len(y) < 10:
            return {"cointegrated": False, "t_stat": 0.0, "half_life": np.inf}

        if HAS_STATSMODELS:
            try:
                score, pvalue, _ = coint(y, x)
                alpha, beta, residuals = self._numpy_ols(y, x)
                is_coint = pvalue < self.p_value_threshold
                t_stat = score
            except Exception as e:
                log.error(f"Statsmodels coint failed, falling back to Numpy: {e}")
                alpha, beta, residuals = self._numpy_ols(y, x)
                t_stat, is_coint = self._numpy_adf_test(residuals)
        else:
            alpha, beta, residuals = self._numpy_ols(y, x)
            t_stat, is_coint = self._numpy_adf_test(residuals)

        # Estimate spread half-life via Ornstein-Uhlenbeck fit
        half_life = self.estimate_half_life(residuals)

        # Calculate current Z-Score
        current_spread = residuals[-1]
        spread_mean = np.mean(residuals)
        spread_std = np.std(residuals)
        z_score = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0.0

        return {
            "cointegrated": bool(is_coint),
            "t_stat": float(t_stat),
            "beta": float(beta),
            "alpha": float(alpha),
            "half_life_bars": float(half_life),
            "z_score": float(z_score),
            "last_spread": float(current_spread)
        }

    def estimate_half_life(self, residuals: np.ndarray) -> float:
        """Estimates mean-reversion half life (tau) of residuals via OU process."""
        if len(residuals) < 5:
            return np.inf
        
        dy = np.diff(residuals)
        y_lag = residuals[:-1]
        
        A = np.vstack([y_lag, np.ones(len(y_lag))]).T
        try:
            gamma, c = np.linalg.lstsq(A, dy, rcond=None)[0]
        except np.linalg.LinAlgError:
            return np.inf

        lam = -gamma
        if lam <= 0:
            return np.inf # No mean reversion detected (divergent or zero)
        
        half_life = np.log(2.0) / lam
        return float(half_life)


class FundingArbitrageur:
    """
    Agent 3: The Funding Rate Arbitrageur.
    Constructs delta-neutral baskets exploiting perp/spot rate differentials.
    Allocates specified capital across high-yield pairs.
    """
    def __init__(self, capital: float = 500.0, min_annual_yield: float = 0.08):
        self.capital = capital
        self.min_annual_yield = min_annual_yield

    def evaluate_opportunities(self, funding_rates: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Ranks arrays of funding rates and returns precise delta-neutral allocations.
        Binance funding rates occur every 8 hours (3 times/day).
        """
        valid_opps = []
        
        for symbol, rate in funding_rates.items():
            # Rate is usually decimal, e.g., 0.0001 = 0.01% per 8h
            annualized_yield = float(rate) * 3.0 * 365.0
            
            abs_yield = abs(annualized_yield)
            if abs_yield >= self.min_annual_yield:
                direction = "SHORT_PERP_LONG_SPOT" if rate > 0 else "LONG_PERP_SHORT_SPOT"
                
                # Dynamic delta-neutral allocation sizing
                notional_per_leg = self.capital / 2.0
                
                valid_opps.append({
                    "symbol": symbol,
                    "funding_rate_8h": float(rate),
                    "annualized_yield": float(annualized_yield),
                    "direction": direction,
                    "spot_allocation": float(notional_per_leg),
                    "perp_allocation": float(notional_per_leg)
                })

        # Sort by absolute yield descending
        valid_opps.sort(key=lambda x: abs(x["annualized_yield"]), reverse=True)
        return valid_opps


class CorrelationMatrixMonitor:
    """
    Agent 11 & 18: Cross-Asset Correlation & Regime Breakdown Detector.
    Uses a fully vectorized, recursively updated EWMA for correlation matrices,
    coupled with Fisher Z-Transform distance boundaries for breakdown alerts.
    """
    def __init__(self, assets: List[str], window: int = 100, confidence_level: float = 1.96):
        self.assets = assets
        self.K = len(assets)
        self.window = window
        self.alpha = 2.0 / (window + 1.0) # EWMA decay factor
        self.confidence_level = confidence_level # 1.96 for 95% CI
        
        # Precompute standard error of Fisher Z-Transform: 1 / sqrt(N - 3)
        self.fisher_se = 1.0 / np.sqrt(max(4.0, float(window)) - 3.0)
        
        # In-memory recursive states
        self.mean = None
        self.covariance = None
        self.avg_fisher_z = None # Long-term expected Z
        
        # Tracker to stabilize baseline
        self.samples_seen = 0

    def update(self, latest_returns: np.ndarray) -> Dict[str, Any]:
        """
        Accepts a 1D array of returns representing latest step for self.assets.
        Returns current correlation matrix, Fisher Z bounds, and systemic risk alerts.
        """
        latest_returns = np.array(latest_returns, dtype=float).flatten()
        if len(latest_returns) != self.K:
            raise ValueError(f"Asset mismatch. Expected {self.K} returns, got {len(latest_returns)}")

        # 1. Bootstrapping initialization
        if self.mean is None:
            self.mean = latest_returns
            self.covariance = np.zeros((self.K, self.K))
            self.avg_fisher_z = np.zeros((self.K, self.K))
            self.samples_seen += 1
            return {"systemic_risk_warning": False, "correlations": np.eye(self.K)}

        self.samples_seen += 1
        
        # 2. Recursive EWMA Covariance Update
        self.mean = (1.0 - self.alpha) * self.mean + self.alpha * latest_returns
        diff = latest_returns - self.mean
        cov_outer = np.outer(diff, diff)
        self.covariance = (1.0 - self.alpha) * self.covariance + self.alpha * cov_outer

        # 3. Derive Correlation Matrix
        std_devs = np.sqrt(np.diag(self.covariance))
        std_devs[std_devs <= 1e-9] = 1e-9 # Strict DivZero guard
        outer_std = np.outer(std_devs, std_devs)
        
        # Safe matrix division
        correlation = self.covariance / outer_std
        # Clamp between almost -1 and 1 to prevent arctanh infinity
        correlation = np.clip(correlation, -1.0 + 1e-7, 1.0 - 1e-7)

        # 4. Fisher Z-Transform Breakdown Detection
        # Calculate arctanh(correlation) element-wise
        fisher_z = np.arctanh(correlation)
        
        # Initial warmup of the Fisher Z baseline
        if self.samples_seen <= self.window:
            # In stabilization phase, pull avg_fisher_z to current values
            weight = 1.0 / float(self.samples_seen)
            self.avg_fisher_z = (1.0 - weight) * self.avg_fisher_z + weight * fisher_z
            return {"systemic_risk_warning": False, "correlations": correlation}

        # Determine Z-score drop
        # Diff between historical Z and current Z. A breakdown is a drop in correlation.
        z_diff = self.avg_fisher_z - fisher_z # Positive if correlation dropped

        # Find any pair where correlation dropped significantly
        # We exclude diagonal elements (self-correlation is always 1, arctanh is inf, diff is 0)
        risk_detected = False
        triggered_pairs = []

        for i in range(self.K):
            for j in range(i + 1, self.K):
                # Normalized distance in standard errors
                dist_se = z_diff[i, j] / self.fisher_se
                
                # Check if drop exceeds confidence boundary
                if dist_se > self.confidence_level:
                    risk_detected = True
                    triggered_pairs.append({
                        "assets": (self.assets[i], self.assets[j]),
                        "current_corr": correlation[i, j],
                        "prior_corr": np.tanh(self.avg_fisher_z[i, j]),
                        "z_score_drop": dist_se
                    })

        # Slow adaptation for long-term baseline tracking
        adapt_alpha = 0.01 # Long-term baseline moves slower
        self.avg_fisher_z = (1.0 - adapt_alpha) * self.avg_fisher_z + adapt_alpha * fisher_z

        return {
            "systemic_risk_warning": bool(risk_detected),
            "triggered_pairs": triggered_pairs,
            "correlations": correlation,
            "sample_count": self.samples_seen
        }

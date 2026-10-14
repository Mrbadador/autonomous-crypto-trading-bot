"""
markov_regime.py — Bayesian Markov Chain & Regime Transition Engine

Models market regimes and volatility states as a Discrete-Time Markov Chain (DTMC)
with Forward Bayesian Filtering, Gaussian emissions, Laplace smoothing, and
expected regime duration forecasting.

3 Discrete Regimes:
    S0 = CHOP
    S1 = BULL_EXPANSION
    S2 = BEAR_CASCADE
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


class MarkovRegimeEngine:
    """
    Bayesian Markov Chain & Forward Filtering Engine for Regime Transitions.
    Eliminates rigid threshold whipsaws by computing real-time posterior probabilities
    over continuous feature vectors [log_return, range_volatility].
    """

    STATES = ["CHOP", "BULL_EXPANSION", "BEAR_CASCADE"]
    STATE_TO_IDX = {"CHOP": 0, "BULL_EXPANSION": 1, "BEAR_CASCADE": 2}
    IDX_TO_STATE = {0: "CHOP", 1: "BULL_EXPANSION", 2: "BEAR_CASCADE"}

    def __init__(self, alpha_smoothing: float = 1.0):
        self.alpha = float(alpha_smoothing)
        self.num_states = 3
        # Uniform initial prior belief: [1/3, 1/3, 1/3]
        self.pi = np.full(self.num_states, 1.0 / self.num_states, dtype=np.float64)
        # Default row-stochastic transition matrix with moderate persistence
        self.P = np.array([
            [0.70, 0.15, 0.15],
            [0.20, 0.70, 0.10],
            [0.20, 0.10, 0.70]
        ], dtype=np.float64)
        # Emission parameters per state: (mean_vec, var_vec) for [log_return, range_vol]
        self.emissions: Dict[int, Dict[str, np.ndarray]] = {
            0: {"mean": np.array([0.000, 0.006]), "var": np.array([1e-4, 1e-5])},     # CHOP
            1: {"mean": np.array([0.005, 0.012]), "var": np.array([3e-4, 4e-5])},     # BULL_EXPANSION
            2: {"mean": np.array([-0.007, 0.018]), "var": np.array([5e-4, 8e-5])}     # BEAR_CASCADE
        }

    def label_historical_states(self, df: pd.DataFrame) -> np.ndarray:
        """
        Labels historical bars into discrete states [0: CHOP, 1: BULL_EXPANSION, 2: BEAR_CASCADE]
        using rolling log-return drift, normalized high-low range volatility, and ADX.
        """
        c = df["close"].to_numpy(dtype=np.float64)
        h = df["high"].to_numpy(dtype=np.float64)
        l = df["low"].to_numpy(dtype=np.float64)
        n = len(c)
        if n < 2:
            return np.zeros(n, dtype=np.int32)

        # Rolling 3-bar log returns
        log_ret_3 = np.zeros(n, dtype=np.float64)
        for i in range(3, n):
            if c[i - 3] > 0 and c[i] > 0:
                log_ret_3[i] = math.log(c[i] / c[i - 3])

        # High-Low Range Volatility: (High - Low) / Close
        range_vol = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if c[i] > 0:
                range_vol[i] = (h[i] - l[i]) / c[i]

        # ADX or proxy
        adx_series = np.full(n, 20.0, dtype=np.float64)
        if "adx" in df.columns:
            adx_series = df["adx"].fillna(20.0).to_numpy(dtype=np.float64)

        labels = np.zeros(n, dtype=np.int32)
        for i in range(n):
            ret = log_ret_3[i]
            adx_val = adx_series[i]
            vol_val = range_vol[i]

            # Cascade: sharp downward move or expanding volatility to the downside
            if ret <= -0.008 or (ret < -0.003 and adx_val >= 24.0) or (ret < -0.002 and vol_val >= 0.020):
                labels[i] = 2  # BEAR_CASCADE
            # Bull Expansion: clear positive return with trend confirmation
            elif ret >= 0.006 or (ret > 0.002 and adx_val >= 22.0):
                labels[i] = 1  # BULL_EXPANSION
            else:
                labels[i] = 0  # CHOP

        return labels

    def estimate_transition_matrix(self, state_sequence: np.ndarray) -> np.ndarray:
        """
        Computes the 3x3 Laplace-smoothed (alpha=1.0) row-stochastic Transition Matrix:
            P_ij = (n_ij + alpha) / sum_k (n_ik + alpha)
        """
        K = self.num_states
        counts = np.zeros((K, K), dtype=np.float64)
        seq = np.asarray(state_sequence, dtype=np.int32)

        for t in range(len(seq) - 1):
            i = int(seq[t])
            j = int(seq[t + 1])
            if 0 <= i < K and 0 <= j < K:
                counts[i, j] += 1.0

        # Laplace smoothing
        smoothed = counts + self.alpha
        row_sums = smoothed.sum(axis=1, keepdims=True)
        self.P = smoothed / row_sums
        return self.P

    def fit_emissions(self, df: pd.DataFrame, state_sequence: np.ndarray):
        """
        Estimates diagonal Gaussian emission parameters (mean, var) for [log_return, range_vol]
        per discrete state.
        """
        c = df["close"].to_numpy(dtype=np.float64)
        h = df["high"].to_numpy(dtype=np.float64)
        l = df["low"].to_numpy(dtype=np.float64)
        n = len(c)
        if n < 5:
            return

        log_rets = np.zeros(n, dtype=np.float64)
        log_rets[1:] = np.diff(np.log(np.maximum(c, 1e-9)))
        range_vols = (h - l) / np.maximum(c, 1e-9)
        seq = np.asarray(state_sequence, dtype=np.int32)

        for s in range(self.num_states):
            mask = (seq == s)
            if np.sum(mask) >= 3:
                s_rets = log_rets[mask]
                s_vols = range_vols[mask]
                mean_vec = np.array([np.mean(s_rets), np.mean(s_vols)], dtype=np.float64)
                var_vec = np.array([
                    max(1e-6, float(np.var(s_rets))),
                    max(1e-6, float(np.var(s_vols)))
                ], dtype=np.float64)
                self.emissions[s] = {"mean": mean_vec, "var": var_vec}

    def _emission_density(self, state_idx: int, obs: np.ndarray) -> float:
        """Computes diagonal bivariate Gaussian probability density b_j(X_t)."""
        params = self.emissions[state_idx]
        mean = params["mean"]
        var = np.maximum(params["var"], 1e-8)

        # -0.5 * sum((x_d - mu_d)^2 / sigma_d^2)
        diff = obs - mean
        exponent = -0.5 * np.sum((diff ** 2) / var)
        # Avoid underflow/overflow
        exponent = max(-50.0, min(20.0, exponent))
        denom = 2.0 * math.pi * math.sqrt(var[0] * var[1])
        return max(1e-12, float(math.exp(exponent) / max(1e-12, denom)))

    def forward_bayesian_filter(self, observations: np.ndarray) -> np.ndarray:
        """
        Executes the Forward Bayesian Filter over a sequence of observations X_t:
            hat_pi_t = pi_{t-1} * P
            pi_{t,j} = (hat_pi_{t,j} * b_j(X_t)) / sum_k (hat_pi_{t,k} * b_k(X_t))
        Returns the final posterior belief vector pi_t = [pi_0, pi_1, pi_2].
        """
        obs = np.asarray(observations, dtype=np.float64)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        pi_curr = self.pi.copy()

        for t in range(len(obs)):
            x_t = obs[t]
            # Prior predictive step: hat_pi_t = pi_{t-1} @ P
            hat_pi = pi_curr @ self.P

            # Likelihood weighting
            b_vals = np.array([self._emission_density(j, x_t) for j in range(self.num_states)], dtype=np.float64)
            unnormalized = hat_pi * b_vals
            denom = np.sum(unnormalized)

            if denom > 0 and math.isfinite(denom):
                pi_curr = unnormalized / denom
            else:
                pi_curr = hat_pi / max(1e-12, np.sum(hat_pi))

        self.pi = np.clip(pi_curr, 1e-6, 1.0)
        self.pi = self.pi / np.sum(self.pi)
        return self.pi

    def get_expected_regime_durations(self) -> Dict[str, float]:
        """
        Expected remaining regime duration in bars: E[D_i] = 1.0 / (1.0 - P_ii).
        """
        durations = {}
        for idx, state in self.IDX_TO_STATE.items():
            p_ii = float(self.P[idx, idx])
            if p_ii >= 0.999:
                durations[state] = 100.0
            else:
                durations[state] = round(1.0 / max(1e-3, 1.0 - p_ii), 2)
        return durations

    def forecast_k_steps(self, k: int = 4) -> Dict[str, Any]:
        """
        Computes k-step ahead forecast: pi_{t+k} = pi_t * P^k.
        Includes breakdown_risk = pi_future[CHOP] + pi_future[BEAR_CASCADE].
        """
        P_k = np.linalg.matrix_power(self.P, k)
        pi_future = self.pi @ P_k
        pi_future = np.clip(pi_future, 0.0, 1.0)
        pi_future = pi_future / np.sum(pi_future)

        prob_chop = float(pi_future[0])
        prob_bull = float(pi_future[1])
        prob_bear = float(pi_future[2])
        breakdown_risk = float(prob_chop + prob_bear)

        return {
            "k_steps": k,
            "forecast_probabilities": {
                "CHOP": round(prob_chop, 4),
                "BULL_EXPANSION": round(prob_bull, 4),
                "BEAR_CASCADE": round(prob_bear, 4)
            },
            "breakdown_risk": round(breakdown_risk, 4),
            "P_k_matrix": P_k.tolist()
        }

    def get_stationary_distribution(self) -> np.ndarray:
        """
        Computes the stationary distribution pi* where pi* P = pi*.
        Left eigenvector corresponding to eigenvalue 1.0.
        """
        try:
            eigvals, eigvecs = np.linalg.eig(self.P.T)
            # Find eigenvector for eigenvalue closest to 1.0
            idx = int(np.argmin(np.abs(eigvals - 1.0)))
            stat_dist = np.real(eigvecs[:, idx])
            stat_dist = np.abs(stat_dist)
            total = np.sum(stat_dist)
            if total > 0:
                return stat_dist / total
        except Exception:
            pass
        return np.full(self.num_states, 1.0 / self.num_states, dtype=np.float64)

    def process_bars(self, df: pd.DataFrame, window: int = 30) -> Dict[str, Any]:
        """
        Full pipeline: labels historical bars, updates transition matrix P,
        runs Bayesian Forward Filtering over the recent window, and emits comprehensive
        telemetry.
        """
        if df is None or len(df) < 15:
            # Fallback telemetry
            dur = self.get_expected_regime_durations()
            fc = self.forecast_k_steps(k=4)
            return {
                "dominant_regime": self.IDX_TO_STATE[int(np.argmax(self.pi))],
                "state_probabilities": {
                    "CHOP": round(float(self.pi[0]), 4),
                    "BULL_EXPANSION": round(float(self.pi[1]), 4),
                    "BEAR_CASCADE": round(float(self.pi[2]), 4)
                },
                "expected_durations_bars": dur,
                "breakdown_risk_4step": fc["breakdown_risk"],
                "forecast_4step": fc["forecast_probabilities"],
                "transition_matrix": self.P.tolist(),
                "stationary_distribution": [round(float(x), 4) for x in self.get_stationary_distribution()]
            }

        labels = self.label_historical_states(df)
        self.estimate_transition_matrix(labels)
        self.fit_emissions(df, labels)

        # Extract recent feature window [log_return, range_vol]
        c = df["close"].to_numpy(dtype=np.float64)
        h = df["high"].to_numpy(dtype=np.float64)
        l = df["low"].to_numpy(dtype=np.float64)
        recent_c = c[-window:]
        recent_h = h[-window:]
        recent_l = l[-window:]

        n_r = len(recent_c)
        obs_rets = np.zeros(n_r, dtype=np.float64)
        obs_rets[1:] = np.diff(np.log(np.maximum(recent_c, 1e-9)))
        obs_vols = (recent_h - recent_l) / np.maximum(recent_c, 1e-9)
        obs_matrix = np.column_stack([obs_rets, obs_vols])

        self.forward_bayesian_filter(obs_matrix)

        dominant_idx = int(np.argmax(self.pi))
        dominant_regime = self.IDX_TO_STATE[dominant_idx]
        expected_durations = self.get_expected_regime_durations()
        forecast_info = self.forecast_k_steps(k=4)
        stat_dist = self.get_stationary_distribution()

        return {
            "dominant_regime": dominant_regime,
            "state_probabilities": {
                "CHOP": round(float(self.pi[0]), 4),
                "BULL_EXPANSION": round(float(self.pi[1]), 4),
                "BEAR_CASCADE": round(float(self.pi[2]), 4)
            },
            "expected_durations_bars": expected_durations,
            "breakdown_risk_4step": forecast_info["breakdown_risk"],
            "forecast_4step": forecast_info["forecast_probabilities"],
            "transition_matrix": self.P.tolist(),
            "stationary_distribution": [round(float(x), 4) for x in stat_dist]
        }

"""
etf_macro_engine.py - Causal Spot ETF Inflow & TradFi Macro Credit Engine
Vectorized, strictly causal computation of ETF net flow z-scores, stationary macro features
(SPY trend distance, UUP momentum, HYG/LQD credit spread), and market-hours aware confidence decay.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import math
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

log = logging.getLogger("etf_macro_engine")


@dataclass(frozen=True)
class FlowSignalParameters:
    half_life_hours: float = 18.0
    baseline_window_days: int = 30
    epsilon: float = 1e-8
    clamping_bound: float = 3.0


class CausalETFFlowEngine:
    """
    Computes running, continuous, causal z-scores from spot ETF net inflow data (e.g. IBIT, FBTC, ETHA).
    Ensures zero future-peeking and continuous exponential decay over missing intervals.
    """

    def __init__(self, params: FlowSignalParameters = FlowSignalParameters()):
        self.params = params
        self.lambda_decay = math.log(2.0) / (self.params.half_life_hours * 3600.0)

    def compute_causal_flow_signals(self, flow_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates recursive causal moments and decayed z-scores across historic flow entries.
        Index must be a monotonic ascending DatetimeIndex (UTC).
        """
        if flow_df.empty:
            return pd.DataFrame()
        if not flow_df.index.is_monotonic_increasing:
            flow_df = flow_df.sort_index()

        n_records = len(flow_df)
        flows = flow_df["net_flow_usd"].to_numpy(dtype=np.float64)
        timestamps = flow_df.index.astype(np.int64) // 10**9

        running_mean = np.zeros(n_records, dtype=np.float64)
        running_var = np.zeros(n_records, dtype=np.float64)
        causal_z = np.zeros(n_records, dtype=np.float64)

        window_sec = self.params.baseline_window_days * 86400.0

        running_mean[0] = flows[0]
        running_var[0] = max(1.0, abs(flows[0]) * 0.1)
        causal_z[0] = 0.0

        for i in range(1, n_records):
            dt = max(1.0, float(timestamps[i] - timestamps[i - 1]))
            alpha = 1.0 - math.exp(-dt / window_sec)
            alpha = max(1e-4, min(0.5, alpha))

            prior_mean = running_mean[i - 1]
            prior_var = running_var[i - 1]

            std_dev = math.sqrt(prior_var + self.params.epsilon)
            raw_z = (flows[i] - prior_mean) / std_dev
            causal_z[i] = max(-self.params.clamping_bound, min(self.params.clamping_bound, raw_z))

            new_mean = (1.0 - alpha) * prior_mean + alpha * flows[i]
            new_var = (1.0 - alpha) * prior_var + alpha * ((flows[i] - new_mean) ** 2)

            running_mean[i] = new_mean
            running_var[i] = new_var

        conviction_mod = np.tanh(causal_z / 1.5)

        return pd.DataFrame(
            index=flow_df.index,
            data={
                "causal_mean": running_mean,
                "causal_std": np.sqrt(running_var + self.params.epsilon),
                "causal_z_score": causal_z,
                "decayed_conviction_modifier": conviction_mod,
            },
        )

    def query_continuous_state(
        self,
        query_timestamp_utc: datetime,
        last_report_timestamp_utc: datetime,
        last_z_score: float,
    ) -> float:
        """
        Evaluates the current decayed conviction at high-frequency bar intervals.
        Returns a bounded value in [-1.0, 1.0].
        """
        if query_timestamp_utc.tzinfo is None:
            query_timestamp_utc = query_timestamp_utc.replace(tzinfo=timezone.utc)
        if last_report_timestamp_utc.tzinfo is None:
            last_report_timestamp_utc = last_report_timestamp_utc.replace(tzinfo=timezone.utc)

        dt_seconds = max(0.0, (query_timestamp_utc - last_report_timestamp_utc).total_seconds())
        decay = math.exp(-self.lambda_decay * dt_seconds)
        decayed_z = last_z_score * decay
        return float(math.tanh(decayed_z / 1.5))


@dataclass
class MacroGateThresholds:
    spy_trend_z_min: float = -1.50
    uup_mom_z_max: float = 1.96
    credit_spread_z_min: float = -1.65
    freshness_limit_seconds: float = 14400.0  # 4 Hours grace before decay
    decay_half_life_seconds: float = 43200.0  # 12 Hours half-life over weekend
    crypto_basis_stress_bps: float = -50.0   # -50 bps annualized basis signals stress


class MarketHoursAwareMacroGate:
    """
    Evaluates TradFi cross-market macro filters (SPY, UUP, HYG, LQD).
    Handles market closures without thread blocking or stale gate latching.
    """

    def __init__(self, thresholds: MacroGateThresholds = MacroGateThresholds()):
        self.cfg = thresholds
        self.last_update_ts: Optional[datetime] = None
        self.flow_engine = CausalETFFlowEngine()
        self.latest_state: Dict[str, Any] = {
            "status": "INITIALIZING",
            "tradfi_active": False,
            "confidence_weight": 0.0,
            "veto_long_entries": False,
            "metrics": {},
            "rationale": "Engine initialized. Awaiting market feed.",
        }

    @staticmethod
    def is_us_equity_market_open(dt_utc: datetime) -> bool:
        """
        Determines whether the New York Stock Exchange is in normal cash hours.
        Monday-Friday: ~13:30 UTC to 20:00 UTC (9:30 AM to 4:00 PM EST/EDT).
        """
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)

        # 0 = Monday, ..., 4 = Friday, 5 = Saturday, 6 = Sunday
        if dt_utc.weekday() >= 5:
            return False

        utc_minutes = dt_utc.hour * 60 + dt_utc.minute
        cash_open_utc = 13 * 60 + 30   # 13:30 UTC
        cash_close_utc = 20 * 60       # 20:00 UTC

        return cash_open_utc <= utc_minutes < cash_close_utc

    def compute_macro_metrics(
        self,
        spy_series: pd.Series,
        uup_series: pd.Series,
        hyg_series: pd.Series,
        lqd_series: pd.Series,
    ) -> Dict[str, float]:
        """
        Transforms raw non-stationary price series into stationary z-score metrics.
        All calculations are strictly backward-looking.
        """
        # 1. SPY 50-bar Trend Z-Score
        spy_close = spy_series.to_numpy(dtype=np.float64)
        if len(spy_close) < 20:
            raise ValueError(f"Insufficient SPY history ({len(spy_close)} bars).")
        lookback_spy = min(50, len(spy_close))
        sma_spy = float(np.mean(spy_close[-lookback_spy:]))
        vol_spy = float(np.std(spy_close[-lookback_spy:])) + 1e-8
        spy_trend_z = float((spy_close[-1] - sma_spy) / vol_spy)

        # 2. UUP (USD) Momentum Z-Score
        uup_close = uup_series.to_numpy(dtype=np.float64)
        if len(uup_close) < 5:
            raise ValueError("Insufficient UUP history.")
        log_ret_uup = math.log(uup_close[-1] / uup_close[-min(5, len(uup_close))])
        rolling_uup_rets = np.log(uup_close[1:] / uup_close[:-1])
        uup_mean = float(np.mean(rolling_uup_rets)) if len(rolling_uup_rets) > 0 else 0.0
        uup_std = float(np.std(rolling_uup_rets)) + 1e-8 if len(rolling_uup_rets) > 0 else 1.0
        uup_mom_z = float((log_ret_uup - uup_mean) / uup_std)

        # 3. Credit Stress Z-Score: ln(HYG / LQD)
        hyg_close = hyg_series.dropna().to_numpy(dtype=np.float64)
        lqd_close = lqd_series.dropna().to_numpy(dtype=np.float64)
        min_len = min(len(hyg_close), len(lqd_close))
        if min_len < 5:
            raise ValueError("Insufficient HYG/LQD history.")
        credit_ratio = np.log(hyg_close[-min_len:] / lqd_close[-min_len:])
        ema_span = min(60, len(credit_ratio))
        ema_credit = float(pd.Series(credit_ratio).ewm(span=ema_span).mean().iloc[-1])
        # Enforce a 100 bps (0.0100) minimum dispersion floor so quiet rate drift does not trigger false credit blowouts
        std_credit = max(float(np.std(credit_ratio[-ema_span:])), 0.0100) + 1e-8
        credit_z = float((credit_ratio[-1] - ema_credit) / std_credit)

        return {
            "spy_trend_z": 0.0 if math.isnan(spy_trend_z) else round(spy_trend_z, 4),
            "uup_mom_z": 0.0 if math.isnan(uup_mom_z) else round(uup_mom_z, 4),
            "credit_spread_z": 0.0 if math.isnan(credit_z) else round(credit_z, 4),
        }

    def update_market_feed(
        self,
        spy_series: pd.Series,
        uup_series: pd.Series,
        hyg_series: pd.Series,
        lqd_series: pd.Series,
        feed_timestamp_utc: datetime,
    ) -> Dict[str, Any]:
        """
        Synchronously updates state from external data feeds.
        Designed to run inside loop.run_in_executor to avoid blocking asyncio.
        """
        if feed_timestamp_utc.tzinfo is None:
            feed_timestamp_utc = feed_timestamp_utc.replace(tzinfo=timezone.utc)
        try:
            metrics = self.compute_macro_metrics(spy_series, uup_series, hyg_series, lqd_series)
            self.last_update_ts = feed_timestamp_utc
            self.latest_state["metrics"] = metrics
            self.latest_state["status"] = "OK"
            self.latest_state["last_update_utc"] = feed_timestamp_utc.isoformat()
            return self.latest_state
        except Exception as err:
            self.latest_state["status"] = "ERROR"
            self.latest_state["rationale"] = f"Error processing market feed: {err}"
            log.warning(f"[ETF-MACRO] Failed to update feed: {err}")
            return self.latest_state

    def evaluate_gate(
        self,
        current_time_utc: datetime,
        crypto_basis_bps: float = 0.0,
        crypto_funding_rate: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Continuous, non-blocking gate evaluation.
        Calculates time decay over weekends/holidays and yields a confidence-weighted decision.
        """
        if current_time_utc.tzinfo is None:
            current_time_utc = current_time_utc.replace(tzinfo=timezone.utc)

        if self.last_update_ts is None or self.latest_state.get("status") == "ERROR":
            return {
                "veto_long_entries": False,
                "confidence_weight": 0.0,
                "state": "UNINITIALIZED_FAIL_OPEN",
                "reason": "TradFi data feed uninitialized. Gate bypassed to prevent lockouts.",
                "metrics": {},
            }

        elapsed_sec = max(0.0, (current_time_utc - self.last_update_ts).total_seconds())
        market_active = self.is_us_equity_market_open(current_time_utc)

        # Confidence weight based on staleness
        if elapsed_sec <= self.cfg.freshness_limit_seconds:
            confidence_w = 1.0
        else:
            decay_time = elapsed_sec - self.cfg.freshness_limit_seconds
            confidence_w = float(math.exp(-decay_time * (math.log(2.0) / self.cfg.decay_half_life_seconds)))
            confidence_w = max(0.0, min(1.0, confidence_w))

        metrics = self.latest_state.get("metrics", {})
        spy_z = metrics.get("spy_trend_z", 0.0)
        uup_z = metrics.get("uup_mom_z", 0.0)
        cred_z = metrics.get("credit_spread_z", 0.0)

        # Core TradFi Defect Evaluation
        # Hard Veto Trigger: Credit spread blowout or extreme SPY breakdown
        is_tradfi_toxic = (
            cred_z < self.cfg.credit_spread_z_min
            or spy_z < self.cfg.spy_trend_z_min
            or uup_z > self.cfg.uup_mom_z_max
        )

        # Fallback Logic: When TradFi confidence decays below 0.30 (weekend closure),
        # substitute continuous crypto perpetual basis and funding rate proxies
        if confidence_w < 0.30:
            # Weekend / Extended Closure Fallback
            # Stress condition: Deep negative basis (< -50 bps) or heavy funding squeeze (< -0.02% per 8h)
            crypto_stress = (crypto_basis_bps < self.cfg.crypto_basis_stress_bps) or (crypto_funding_rate < -0.0002)
            effective_veto = bool(crypto_stress)
            regime_tag = "DECAYED_CRYPTO_BASIS_FALLBACK"
            rationale = (
                f"TradFi equity market closed/stale ({elapsed_sec/3600.0:.1f}h ago). Confidence={confidence_w:.2f}. "
                f"Falling back to continuous crypto basis ({crypto_basis_bps:.1f} bps, FR: {crypto_funding_rate:.4%}). "
                f"Crypto stress={'VETO' if crypto_stress else 'CLEAR'}."
            )
        else:
            effective_veto = bool(is_tradfi_toxic and (confidence_w >= 0.50))
            regime_tag = "ACTIVE_TRADFI_MONITORED"
            toxic_reasons = []
            if cred_z < self.cfg.credit_spread_z_min:
                toxic_reasons.append(f"Credit blowout (Z={cred_z:.2f} < {self.cfg.credit_spread_z_min})")
            if spy_z < self.cfg.spy_trend_z_min:
                toxic_reasons.append(f"SPY breakdown (Z={spy_z:.2f} < {self.cfg.spy_trend_z_min})")
            if uup_z > self.cfg.uup_mom_z_max:
                toxic_reasons.append(f"USD surge (Z={uup_z:.2f} > {self.cfg.uup_mom_z_max})")

            toxic_summary = ", ".join(toxic_reasons) if toxic_reasons else "Normal"
            rationale = (
                f"TradFi Active. Confidence={confidence_w:.2f}. "
                f"Status: {toxic_summary}."
            )

        return {
            "veto_long_entries": effective_veto,
            "confidence_weight": round(confidence_w, 4),
            "tradfi_market_active": market_active,
            "staleness_seconds": round(elapsed_sec, 1),
            "regime_tag": regime_tag,
            "metrics": metrics,
            "rationale": rationale,
        }

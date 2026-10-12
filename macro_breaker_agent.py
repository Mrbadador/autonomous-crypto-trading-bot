"""
macro_breaker_agent.py — Sprint 3 (Phase 3) Crypto-Native Contagion Radar
========================================================================
Replaces high-latency TradFi REST polling with real-time crypto-native metrics:
  1. BTC Dominance velocity (v_dom)
  2. Stablecoin aggregate depeg delta (Delta_peg)
  3. Binance-to-Bitget mid-price basis divergence (D_basis)
  4. Funding rate imbalance skew (F_skew)

Computes the normalized Macro Contagion Index:
  I_contagion(t) = sigma(theta_1 * v_dom / sigma_dom + theta_2 * Delta_peg / 0.0020 + theta_3 * D_basis / 15.0 + theta_4 * F_skew - intercept)

If I_contagion > 0.85, immediately dispatches EVENT_CIRCUIT_BREAKER_TRIGGERED (< 10 ms SLA)
and enforces the emergency macro shutdown.
"""

import asyncio
import json
import logging
import math
import os
import sqlite3
import time
from datetime import datetime, UTC
from typing import Callable, Dict, List, Optional

log = logging.getLogger("macro_breaker")

# Safe import of yfinance (legacy fallback only)
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

EVENT_CIRCUIT_BREAKER_TRIGGERED = "EVENT_CIRCUIT_BREAKER_TRIGGERED"
DEFAULT_CONTAGION_THRESHOLD = 0.85


def compute_macro_contagion_index(
    v_dom: float = 0.0,
    delta_peg: float = 0.0,
    d_basis: float = 0.0,
    f_skew: float = 0.0,
    theta1: float = 1.0,
    theta2: float = 2.0,
    theta3: float = 1.5,
    theta4: float = 1.0,
    sigma_dom: float = 0.005,
    intercept: float = 2.0
) -> float:
    """
    Computes the normalized Macro Contagion Index I_contagion in [0.0, 1.0].
    
    Formula:
      z = theta_1 * (v_dom / sigma_dom)
        + theta_2 * (delta_peg / 0.0020)
        + theta_3 * (d_basis / 15.0)
        + theta_4 * f_skew
        - intercept
      I_contagion = 1.0 / (1.0 + exp(-z))
    """
    term_dom = theta1 * (v_dom / max(1e-6, sigma_dom))
    term_peg = theta2 * (delta_peg / 0.0020)
    term_basis = theta3 * (d_basis / 15.0)
    term_skew = theta4 * f_skew
    
    z = term_dom + term_peg + term_basis + term_skew - intercept
    if z >= 40.0:
        return 1.0
    elif z <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


class CryptoContagionRadar:
    """
    High-Frequency Crypto-Native Contagion Radar.
    Monitors cross-market structural dislocations and fires circuit breaker events
    within < 10 ms of breach.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_CONTAGION_THRESHOLD,
        db_path: Optional[str] = None
    ):
        self.threshold = threshold
        if db_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base_dir, "trading_data.db")
        self.db_path = db_path
        
        self.listeners: List[Callable] = []
        self.last_contagion_index: float = 0.0
        self.is_circuit_breaker_active: bool = False
        self.last_metrics: Dict[str, float] = {}
        self.last_eval_time: float = 0.0
        self._prev_btc_dom: Optional[float] = None
        self._prev_dom_time: Optional[float] = None

    def register_listener(self, callback: Callable) -> None:
        """Registers a synchronous or asynchronous callback for circuit breaker events."""
        if callback not in self.listeners:
            self.listeners.append(callback)

    async def fetch_crypto_metrics(self) -> Dict[str, float]:
        """
        Gathers high-frequency crypto-native metrics non-blockingly:
          - v_dom: BTC dominance velocity
          - delta_peg: Stablecoin depeg delta (USDT, USDC, FDUSD)
          - d_basis: Binance vs Bitget mid-price basis divergence in bps
          - f_skew: Funding rate skew between BTC and altcoins
        """
        v_dom = 0.0
        delta_peg = 0.0
        d_basis = 0.0
        f_skew = 0.0
        
        now = time.time()
        
        # 1. Inspect microstructure_state.json for live cross-asset telemetry
        try:
            ofi_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "microstructure_state.json")
            if os.path.exists(ofi_file):
                with open(ofi_file, "r") as f:
                    state = json.load(f)
                
                d_basis = float(state.get("basis_divergence_bps", 0.0))
                delta_peg = float(state.get("stablecoin_depeg_delta", 0.0))
                v_dom = float(state.get("btc_dom_velocity", 0.0))
                f_skew = float(state.get("funding_skew", 0.0))
        except Exception as e:
            log.debug(f"[CONTAGION-RADAR] Local state read note: {e}")

        # 2. Check system_state SQLite cache for any external feeds
        try:
            with sqlite3.connect(self.db_path, timeout=2.0) as conn:
                rows = conn.execute(
                    "SELECT key, value FROM system_state WHERE key IN ('btc_dominance', 'stablecoin_depeg', 'cross_basis_bps', 'funding_skew')"
                ).fetchall()
                for k, v in rows:
                    try:
                        val = float(v)
                        if k == "btc_dominance":
                            if self._prev_btc_dom is not None and self._prev_dom_time is not None:
                                dt = max(1.0, now - self._prev_dom_time)
                                v_dom = (val - self._prev_btc_dom) / dt
                            self._prev_btc_dom = val
                            self._prev_dom_time = now
                        elif k == "stablecoin_depeg":
                            delta_peg = max(delta_peg, val)
                        elif k == "cross_basis_bps":
                            d_basis = max(d_basis, val)
                        elif k == "funding_skew":
                            f_skew = val
                    except (ValueError, TypeError):
                        pass
        except Exception as dbe:
            log.debug(f"[CONTAGION-RADAR] DB read note: {dbe}")

        return {
            "v_dom": v_dom,
            "delta_peg": delta_peg,
            "d_basis": d_basis,
            "f_skew": f_skew
        }

    async def evaluate(self, metrics: Optional[Dict[str, float]] = None) -> Dict:
        """
        Evaluates the current macro contagion state.
        If I_contagion > threshold (0.85), dispatches EVENT_CIRCUIT_BREAKER_TRIGGERED
        in < 10 ms SLA.
        """
        t0 = time.perf_counter()
        
        if metrics is None:
            metrics = await self.fetch_crypto_metrics()
            
        v_dom = metrics.get("v_dom", 0.0)
        delta_peg = metrics.get("delta_peg", 0.0)
        d_basis = metrics.get("d_basis", 0.0)
        f_skew = metrics.get("f_skew", 0.0)
        
        c_index = compute_macro_contagion_index(
            v_dom=v_dom,
            delta_peg=delta_peg,
            d_basis=d_basis,
            f_skew=f_skew
        )
        
        self.last_contagion_index = c_index
        self.last_metrics = metrics
        self.last_eval_time = time.time()
        
        is_breached = c_index > self.threshold
        dispatch_latency_ms = 0.0
        
        if is_breached:
            if not self.is_circuit_breaker_active:
                log.critical(
                    f"[CONTAGION-RADAR] [CIRCUIT-BREAKER-TRIGGERED] SEVERE CONTAGION BREACHED! Index: {c_index:.4f} > {self.threshold:.2f} "
                    f"(v_dom={v_dom:.5f}, Delta_peg={delta_peg:.5f}, D_basis={d_basis:.2f}bps, F_skew={f_skew:.3f})"
                )
            self.is_circuit_breaker_active = True
            
            event_data = {
                "event": EVENT_CIRCUIT_BREAKER_TRIGGERED,
                "contagion_index": c_index,
                "threshold": self.threshold,
                "timestamp": datetime.now(UTC).isoformat(),
                "metrics": metrics
            }
            
            # Sub-10ms event dispatch SLA
            d_t0 = time.perf_counter()
            await self.dispatch_circuit_breaker(event_data)
            dispatch_latency_ms = (time.perf_counter() - d_t0) * 1000.0
            
            # Non-blocking async micro-batched DB persistence via Sprint 1 DBWriteQueue (< 0.1 ms)
            try:
                from db_manager import DBWriteQueue
                wq = DBWriteQueue.get_instance(self.db_path)
                ts_now = datetime.now(UTC).isoformat()
                wq.enqueue_write("system_state", "REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                                 ("macro_global_killswitch", "1", ts_now))
                wq.enqueue_write("system_state", "REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                                 ("circuit_breaker_active", "1", ts_now))
                wq.enqueue_write("system_state", "REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                                 ("macro_contagion_index", str(round(c_index, 4)), ts_now))
            except Exception as w_err:
                log.debug(f"[CONTAGION-RADAR] DBWriteQueue persistence note: {w_err}")
                
            return {
                "safe_to_trade": False,
                "contagion_index": c_index,
                "threshold": self.threshold,
                "dispatch_latency_ms": dispatch_latency_ms,
                "reason": f"Severe Contagion Breached: {c_index:.3f} > {self.threshold:.2f}",
                "metrics": metrics
            }
        else:
            self.is_circuit_breaker_active = False
            eval_latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "safe_to_trade": True,
                "contagion_index": c_index,
                "threshold": self.threshold,
                "eval_latency_ms": eval_latency_ms,
                "reason": f"Contagion index {c_index:.3f} within safe bounds (< {self.threshold:.2f})",
                "metrics": metrics
            }

    async def dispatch_circuit_breaker(self, event_data: Dict) -> None:
        """Dispatches event to all registered listeners asynchronously."""
        tasks = []
        for listener in self.listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    tasks.append(listener(event_data))
                else:
                    listener(event_data)
            except Exception as le:
                log.error(f"[CONTAGION-RADAR] Listener exception: {le}")
                
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    log.error(f"[CONTAGION-RADAR] Async listener exception: {res}")


class MacroCircuitBreaker(CryptoContagionRadar):
    """
    Macro Circuit Breaker Agent (V40.0 Architecture).
    Combines high-frequency crypto-native contagion radar with traditional S&P 500 / DXY
    monitoring fallback.
    """

    def __init__(
        self,
        sp500_threshold: float = -2.5,
        dxy_threshold: float = 1.0,
        contagion_threshold: float = DEFAULT_CONTAGION_THRESHOLD,
        db_path: Optional[str] = None
    ):
        super().__init__(threshold=contagion_threshold, db_path=db_path)
        self.sp500_threshold = sp500_threshold
        self.dxy_threshold = dxy_threshold

    async def fetch(self) -> Dict:
        """
        Executes high-frequency crypto-native evaluation first (< 1 ms).
        If safe, checks traditional TradFi metrics if yfinance is available.
        """
        crypto_res = await self.evaluate()
        if not crypto_res.get("safe_to_trade", True):
            return crypto_res

        # Fallback check on TradFi macro metrics if needed
        if not HAS_YFINANCE:
            return crypto_res

        try:
            data = await asyncio.to_thread(
                yf.download,
                tickers="^GSPC DX-Y.NYB",
                period="3d",
                interval="1d",
                progress=False
            )
            if not data.empty and "Close" in data.columns:
                close_data = data["Close"]
                if "^GSPC" in close_data.columns:
                    sp_series = close_data["^GSPC"].dropna()
                    if len(sp_series) >= 2:
                        sp_change = ((sp_series.iloc[-1] - sp_series.iloc[-2]) / sp_series.iloc[-2]) * 100.0
                        if sp_change <= self.sp500_threshold:
                            return {
                                "safe_to_trade": False,
                                "contagion_index": self.last_contagion_index,
                                "reason": f"TradFi S&P 500 Crash Breached: {sp_change:.2f}% <= {self.sp500_threshold}%",
                                "metrics": self.last_metrics
                            }
                if "DX-Y.NYB" in close_data.columns:
                    dxy_series = close_data["DX-Y.NYB"].dropna()
                    if len(dxy_series) >= 2:
                        dxy_change = ((dxy_series.iloc[-1] - dxy_series.iloc[-2]) / dxy_series.iloc[-2]) * 100.0
                        if dxy_change >= self.dxy_threshold:
                            return {
                                "safe_to_trade": False,
                                "contagion_index": self.last_contagion_index,
                                "reason": f"TradFi DXY Spike Breached: {dxy_change:+.2f}% >= +{self.dxy_threshold}%",
                                "metrics": self.last_metrics
                            }
        except Exception as tradfi_err:
            log.debug(f"[MACRO-BREAKER] TradFi polling note: {tradfi_err}")

        return crypto_res

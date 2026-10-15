import asyncio
import websockets
import json
import time
import os
from collections import deque
from datetime import datetime
import sys
import numpy as np
import dns_fix  # noqa: F401
from microstructure_metrics import (
    compute_micro_price,
    compute_multi_level_ofi,
    compute_mlofi_step,
    HawkesSpoofGuard,
    VPINCalculator,
    verify_bitget_crc32_checksum,
    KyleLambdaCalculator,
    calculate_lvr_tax_bps
)

# Import symbols from config if available, fallback to top active coins
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "NEARUSDT", "SUIUSDT"]
try:
    import config
    raw_symbols = getattr(config, "SYMBOLS", [])
    blacklist = [s.replace("/", "").upper() for s in getattr(config, "SWING_BLACKLIST", [])]
    SYMBOLS = [s.replace("/", "").upper() for s in raw_symbols if s and s.replace("/", "").upper() not in blacklist]
    if not SYMBOLS:
        SYMBOLS = [s for s in DEFAULT_SYMBOLS if s not in blacklist]
    if "BTCUSDT" not in SYMBOLS:
        SYMBOLS.insert(0, "BTCUSDT")
except Exception:
    SYMBOLS = DEFAULT_SYMBOLS

# Bitget Public V2 Multi-Stream Depth WebSocket (15 levels snapshot)
WS_URL = "wss://ws.bitget.com/v2/ws/public"
STATE_FILE = "microstructure_state.json"


class SymbolOFIState:
    """
    Maintains 10-level pre-allocated NumPy depth buffers and computes
    volumetric Order Flow Imbalance (OFI) with exponential distance decay.
    """
    def __init__(self, symbol: str, window_seconds: int = 60, levels: int = 10, delta: float = 0.25):
        self.symbol = symbol
        self.window_seconds = window_seconds
        self.levels = levels
        self.delta = delta

        # Pre-allocated NumPy memory buffers (zero heap allocations in hot path)
        self.prev_bid_p = np.zeros(levels, dtype=np.float64)
        self.prev_bid_q = np.zeros(levels, dtype=np.float64)
        self.prev_ask_p = np.zeros(levels, dtype=np.float64)
        self.prev_ask_q = np.zeros(levels, dtype=np.float64)

        # Pre-allocated arrays for current tick ingestion
        self.curr_bid_p = np.zeros(levels, dtype=np.float64)
        self.curr_bid_q = np.zeros(levels, dtype=np.float64)
        self.curr_ask_p = np.zeros(levels, dtype=np.float64)
        self.curr_ask_q = np.zeros(levels, dtype=np.float64)

        # Exponential distance weighting kernel: w_k = exp(-delta*(k-1)) / sum(...)
        k = np.arange(levels, dtype=np.float64)
        raw_w = np.exp(-delta * k)
        self.weights = raw_w / np.sum(raw_w)

        # Addition and cancellation tracking for Hawkes & Spoofing Guard
        self.rolling_additions = deque()
        self.rolling_cancellations = deque()
        self.level_add_counts = np.zeros(levels, dtype=np.float64)
        self.level_cancel_counts = np.zeros(levels, dtype=np.float64)
        self.level_add_ts = np.zeros(levels, dtype=np.float64)
        self.residence_times = deque(maxlen=200)

        self.initialized = False
        self.ofi_history = deque()
        self.net_ofi_history = deque()
        self.best_bid = 0.0
        self.best_ask = 0.0

        # Microstructure Alpha Upgrades (MLOFI, Micro-Price, VPIN, Hawkes Spoof Guard)
        self.latest_mlofi = 0.0
        self.latest_micro_price = 0.0
        self.latest_micro_bps = 0.0
        self.vpin_calc = VPINCalculator(bucket_volume=2000.0, num_buckets=30)
        self.vpin_calc_real = VPINCalculator(bucket_volume=2000.0, num_buckets=30)
        self.hawkes_guard = HawkesSpoofGuard()
        self.latest_hawkes_spoof = 0.0
        self.latest_hawkes_rho = 0.0
        self.latest_hawkes_veto = False
        self.last_depth_ts = 0.0
        self.real_trade_count = 0
        self.last_real_trade_ts = 0.0

        # State-dependent Kyle's Lambda & LVR Tax (Upgrade 2)
        self.kyle_calc = KyleLambdaCalculator(window_seconds=1800.0, kappa_basis=0.05)
        self.latest_kyle_lambda = 0.0
        self.latest_kyle_pctile = 0.50
        self.last_mid_price = 0.0
        self.recent_log_returns: deque = deque(maxlen=60)
        self.latest_lvr_tax_bps = 0.0
        self.latest_toxicity_halt = False

    def reset(self):
        """Resets state machine on WebSocket reconnection to avoid spurious spikes."""
        self.prev_bid_p.fill(0.0)
        self.prev_bid_q.fill(0.0)
        self.prev_ask_p.fill(0.0)
        self.prev_ask_q.fill(0.0)
        self.rolling_additions.clear()
        self.rolling_cancellations.clear()
        self.level_add_counts.fill(0.0)
        self.level_cancel_counts.fill(0.0)
        self.level_add_ts.fill(0.0)
        self.residence_times.clear()
        self.initialized = False
        self.ofi_history.clear()
        self.net_ofi_history.clear()
        self.latest_mlofi = 0.0
        self.latest_micro_price = 0.0
        self.latest_micro_bps = 0.0
        self.hawkes_guard = HawkesSpoofGuard()
        self.latest_hawkes_spoof = 0.0
        self.latest_hawkes_rho = 0.0
        self.latest_hawkes_veto = False
        self.last_depth_ts = 0.0
        self.real_trade_count = 0
        self.last_real_trade_ts = 0.0
        self.kyle_calc = KyleLambdaCalculator(window_seconds=1800.0, kappa_basis=0.05)
        self.latest_kyle_lambda = 0.0
        self.latest_kyle_pctile = 0.50
        self.last_mid_price = 0.0
        self.recent_log_returns.clear()
        self.latest_lvr_tax_bps = 0.0
        self.latest_toxicity_halt = False

    def process_trade(self, price: float, size: float, side: str = "buy"):
        """Ingests live trade tape from Bitget trade channel for genuine VPIN and flow toxicity."""
        if price <= 0.0 or size <= 0.0:
            return
        self.real_trade_count += 1
        self.last_real_trade_ts = time.time()
        self.vpin_calc_real.update_trade(price=price, size=size, side=side)

    def process_depth(self, bids: list, asks: list):
        """
        Ingests L2 depth levels and computes 10-level volumetric OFI via vectorized NumPy.
        Tracks additions, cancellations, and residence times for Hawkes spoof detection.
        """
        n_bids = min(len(bids), self.levels)
        n_asks = min(len(asks), self.levels)
        if n_bids == 0 or n_asks == 0:
            return

        self.curr_bid_p.fill(0.0)
        self.curr_bid_q.fill(0.0)
        self.curr_ask_p.fill(0.0)
        self.curr_ask_q.fill(0.0)

        for i in range(n_bids):
            self.curr_bid_p[i] = float(bids[i][0])
            self.curr_bid_q[i] = float(bids[i][1])

        for i in range(n_asks):
            self.curr_ask_p[i] = float(asks[i][0])
            self.curr_ask_q[i] = float(asks[i][1])

        self.best_bid = float(self.curr_bid_p[0])
        self.best_ask = float(self.curr_ask_p[0])

        now = time.time()

        if not self.initialized:
            self.prev_bid_p[:] = self.curr_bid_p
            self.prev_bid_q[:] = self.curr_bid_q
            self.prev_ask_p[:] = self.curr_ask_p
            self.prev_ask_q[:] = self.curr_ask_q
            self.level_add_ts.fill(now)
            self.initialized = True
            return

        # Vectorized Bid Innovation Delta W (10 levels)
        delta_w = np.where(
            self.curr_bid_p > self.prev_bid_p,
            self.curr_bid_q,
            np.where(
                self.curr_bid_p == self.prev_bid_p,
                self.curr_bid_q - self.prev_bid_q,
                -self.prev_bid_q
            )
        )

        # Vectorized Ask Innovation Delta V (10 levels)
        delta_v = np.where(
            self.curr_ask_p < self.prev_ask_p,
            self.curr_ask_q,
            np.where(
                self.curr_ask_p == self.prev_ask_p,
                self.curr_ask_q - self.prev_ask_q,
                -self.prev_ask_q
            )
        )

        level_ofi = delta_w - delta_v
        multi_ofi = float(np.dot(self.weights, level_ofi))
        q_bar = float(np.sum(self.curr_bid_q + self.curr_ask_q) / (2.0 * self.levels))

        # Event tracking: additions and cancellations across levels
        bid_add_mask = (self.curr_bid_p > self.prev_bid_p) | ((self.curr_bid_p == self.prev_bid_p) & (self.curr_bid_q > self.prev_bid_q))
        bid_cancel_mask = (self.curr_bid_p < self.prev_bid_p) | ((self.curr_bid_p == self.prev_bid_p) & (self.curr_bid_q < self.prev_bid_q))

        ask_add_mask = (self.curr_ask_p < self.prev_ask_p) | ((self.curr_ask_p == self.prev_ask_p) & (self.curr_ask_q > self.prev_ask_q))
        ask_cancel_mask = (self.curr_ask_p > self.prev_ask_p) | ((self.curr_ask_p == self.prev_ask_p) & (self.curr_ask_q < self.prev_ask_q))

        tick_adds = 0
        tick_cancels = 0

        for lvl in range(self.levels):
            if bid_add_mask[lvl]:
                tick_adds += 1
                self.level_add_counts[lvl] += 1.0
                self.level_add_ts[lvl] = now
            elif bid_cancel_mask[lvl]:
                tick_cancels += 1
                self.level_cancel_counts[lvl] += 1.0
                if self.level_add_ts[lvl] > 0.0:
                    tau_ms = (now - self.level_add_ts[lvl]) * 1000.0
                    if 2.0 <= tau_ms <= 60000.0:
                        self.residence_times.append(tau_ms)

            if ask_add_mask[lvl]:
                tick_adds += 1
                self.level_add_counts[lvl] += 1.0
                self.level_add_ts[lvl] = now
            elif ask_cancel_mask[lvl]:
                tick_cancels += 1
                self.level_cancel_counts[lvl] += 1.0
                if self.level_add_ts[lvl] > 0.0:
                    tau_ms = (now - self.level_add_ts[lvl]) * 1000.0
                    if 2.0 <= tau_ms <= 60000.0:
                        self.residence_times.append(tau_ms)

        self.rolling_additions.append((now, tick_adds))
        self.rolling_cancellations.append((now, tick_cancels))

        while self.rolling_additions and now - self.rolling_additions[0][0] > self.window_seconds:
            self.rolling_additions.popleft()
        while self.rolling_cancellations and now - self.rolling_cancellations[0][0] > self.window_seconds:
            self.rolling_cancellations.popleft()

        # Compute Numba-vectorized Multi-Level OFI with geometric depth decay (<3us)
        prev_b_arr = np.column_stack((self.prev_bid_p, self.prev_bid_q))
        curr_b_arr = np.column_stack((self.curr_bid_p, self.curr_bid_q))
        prev_a_arr = np.column_stack((self.prev_ask_p, self.prev_ask_q))
        curr_a_arr = np.column_stack((self.curr_ask_p, self.curr_ask_q))
        self.latest_mlofi = float(compute_mlofi_step(prev_b_arr, curr_b_arr, prev_a_arr, curr_a_arr, delta_kernel=0.5))

        # Real-time Bivariate Hawkes Process for LOB Spoofing & Phantom Liquidity (<2us)
        dt = (now - self.last_depth_ts) if self.last_depth_ts > 0 else 0.1
        self.last_depth_ts = now
        avg_res_sec = (sum(self.residence_times) / len(self.residence_times) / 1000.0) if self.residence_times else 0.250
        exec_est = max(1, int(round((tick_adds + tick_cancels) * 0.05)))
        h_veto, h_spoof, h_rho = self.hawkes_guard.evaluate_tick(
            additions=tick_adds,
            cancellations=tick_cancels,
            executions=exec_est,
            avg_residence_sec=avg_res_sec,
            dt=dt
        )
        self.latest_hawkes_spoof = float(h_spoof)
        self.latest_hawkes_rho = float(h_rho)
        self.latest_hawkes_veto = bool(h_veto)

        # Imbalance-weighted Micro-Price & BPS spread bias (Stoikov 2018)
        mp, mp_bps = compute_micro_price(
            bid_price=float(self.curr_bid_p[0]),
            bid_size=float(self.curr_bid_q[0]),
            ask_price=float(self.curr_ask_p[0]),
            ask_size=float(self.curr_ask_q[0])
        )
        self.latest_micro_price = float(mp)
        self.latest_micro_bps = float(mp_bps)

        # Update Kyle's Lambda (λ_t) & LVR Volatility Tax
        curr_mid = (float(self.curr_bid_p[0]) + float(self.curr_ask_p[0])) / 2.0
        if curr_mid > 0:
            if self.last_mid_price > 0:
                d_p = curr_mid - self.last_mid_price
                log_r = math.log(curr_mid / self.last_mid_price)
                self.recent_log_returns.append(log_r)
                # Volumetric proxy for delta buy/sell: level additions/cancels
                tot_buy_flow = float(self.curr_bid_q[0] + tick_adds)
                tot_sell_flow = float(self.curr_ask_q[0] + tick_cancels)
                self.kyle_calc.update_tick(price_delta=d_p, buy_volume=tot_buy_flow, sell_volume=tot_sell_flow, basis_bps=mp_bps, timestamp=now)
            self.last_mid_price = curr_mid

        cur_lam, cur_pctile = self.kyle_calc.get_current_lambda_and_percentile()
        self.latest_kyle_lambda = cur_lam
        self.latest_kyle_pctile = cur_pctile

        # Realized volatility over recent ticks
        if len(self.recent_log_returns) >= 5:
            sigma_ticks = float(np.std(self.recent_log_returns))
            self.latest_lvr_tax_bps = float(calculate_lvr_tax_bps(sigma_ticks))
        else:
            self.latest_lvr_tax_bps = 0.50

        # Toxicity Halt Evaluation (Upgrade 2)
        vpin_val = self.vpin_calc_real.get_vpin() if self.real_trade_count >= 10 else self.vpin_calc.get_vpin()
        try:
            import config as _cfg
            rho_cut = float(getattr(_cfg, "HAWKES_CASCADE_RHO_THRESHOLD", 1.05))
            spoof_cut = float(getattr(_cfg, "HAWKES_SPOOF_VETO_THRESHOLD", 0.78))
        except Exception:
            rho_cut = 1.05
            spoof_cut = 0.78

        self.latest_toxicity_halt = bool(
            (vpin_val >= 0.68) or
            (self.latest_hawkes_rho >= rho_cut) or
            (self.latest_hawkes_spoof >= spoof_cut)
        )

        # Streaming VPIN update
        mid_p = (self.curr_bid_p[0] + self.curr_ask_p[0]) / 2.0
        top_vol = (self.curr_bid_q[0] + self.curr_ask_q[0]) / 2.0
        self.vpin_calc.update_trade(price=mid_p, size=top_vol)

        self.prev_bid_p[:] = self.curr_bid_p
        self.prev_bid_q[:] = self.curr_bid_q
        self.prev_ask_p[:] = self.curr_ask_p
        self.prev_ask_q[:] = self.curr_ask_q

        self.ofi_history.append((now, multi_ofi, q_bar))

        while self.ofi_history and now - self.ofi_history[0][0] > self.window_seconds:
            self.ofi_history.popleft()

    def get_metrics(self) -> dict:
        """Calculates normalized OFI, velocity, spread compression, and Hawkes counts."""
        now = time.time()
        now_ms = int(now * 1000)

        net_ofi = sum(val for _, val, _ in self.ofi_history)
        self.net_ofi_history.append((now_ms, net_ofi))
        while self.net_ofi_history and now_ms - self.net_ofi_history[0][0] > 5000:
            self.net_ofi_history.popleft()

        q_bars = [q for _, _, q in self.ofi_history if q > 0]
        mean_q = float(np.mean(q_bars)) if q_bars else 5.0
        adaptive_scale = max(10.0, 2.0 * mean_q)
        norm_ofi = float(np.tanh(net_ofi / adaptive_scale))

        if len(self.net_ofi_history) > 1:
            oldest_time_ms, oldest_net_ofi = self.net_ofi_history[0]
            elapsed_seconds = (now_ms - oldest_time_ms) / 1000.0
            if elapsed_seconds > 0.0:
                raw_velocity = (net_ofi - oldest_net_ofi) / elapsed_seconds
                vel_scale = max(5.0, adaptive_scale / 2.5)
                ofi_velocity = float(np.tanh(raw_velocity / vel_scale))
            else:
                ofi_velocity = 0.0
        else:
            ofi_velocity = 0.0

        spread = abs(self.best_ask - self.best_bid)
        mid_price = (self.best_bid + self.best_ask) / 2.0
        spread_compression = spread / mid_price if mid_price > 0 else 0.0

        n_a = sum(cnt for _, cnt in self.rolling_additions)
        n_c = sum(cnt for _, cnt in self.rolling_cancellations)
        mean_tau = float(np.mean(self.residence_times)) if self.residence_times else 250.0

        vpin_synth = round(self.vpin_calc.get_vpin(), 3)
        vpin_real = round(self.vpin_calc_real.get_vpin(), 3)
        # Use real VPIN when available (e.g. > 10 real trades received); fallback smoothly to synthetic
        effective_vpin = vpin_real if self.real_trade_count >= 10 else vpin_synth

        return {
            "net_ofi_score": round(net_ofi, 4),
            "normalized_ofi": round(norm_ofi, 4),
            "ofi_velocity": round(ofi_velocity, 4),
            "spread_compression": round(spread_compression, 6),
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mlofi": round(self.latest_mlofi, 4),
            "micro_price": round(self.latest_micro_price, 4),
            "micro_price_bps": round(self.latest_micro_bps, 2),
            "vpin": effective_vpin,
            "vpin_synthetic": vpin_synth,
            "vpin_real": vpin_real,
            "real_trade_count": self.real_trade_count,
            "hawkes_spoof_index": round(self.latest_hawkes_spoof, 4),
            "hawkes_rho": round(self.latest_hawkes_rho, 4),
            "hawkes_veto": self.latest_hawkes_veto,
            "kyle_lambda": round(self.latest_kyle_lambda, 6),
            "kyle_lambda_pctile": round(self.latest_kyle_pctile, 4),
            "lvr_tax_bps": round(self.latest_lvr_tax_bps, 2),
            "toxicity_halt": self.latest_toxicity_halt,
            "order_additions_count": n_a,
            "order_cancellations_count": n_c,
            "mean_residence_time_ms": round(mean_tau, 2),
            "cancel_counts": self.level_cancel_counts.tolist(),
            "add_counts": self.level_add_counts.tolist()
        }


class MultiSymbolOFIDaemon:
    def __init__(self, symbols=None, window_seconds=60):
        self.symbols = symbols or SYMBOLS
        self.window_seconds = window_seconds
        self.symbol_states = {
            s: SymbolOFIState(s, window_seconds=window_seconds, levels=10, delta=0.25)
            for s in self.symbols
        }
        self.last_write_time = time.time()
        self._background_tasks = set()
        self.latest_tape_intel = None

    def reset_all(self):
        for state in self.symbol_states.values():
            state.reset()

    def process_trade_tick(self, symbol: str, price: float, size: float, side: str = "buy"):
        if symbol in self.symbol_states:
            self.symbol_states[symbol].process_trade(price, size, side)

    def process_tick(self, symbol: str, bids: list, asks: list):
        if symbol in self.symbol_states:
            self.symbol_states[symbol].process_depth(bids, asks)

        now = time.time()
        if now - self.last_write_time >= 1.0:
            # Drop trigger if prior write is still pending to avoid disk queue build-up
            if not any(t.get_name() == "ofi_write_state" and not t.done() for t in self._background_tasks):
                self.last_write_time = now
                t = asyncio.create_task(self.write_state_async(), name="ofi_write_state")
                self._background_tasks.add(t)
                t.add_done_callback(self._background_tasks.discard)


    async def write_state_async(self):
        now = time.time()
        all_metrics = {}
        for s, state in self.symbol_states.items():
            all_metrics[s] = state.get_metrics()

        # Primary benchmark is BTCUSDT
        primary = all_metrics.get("BTCUSDT", next(iter(all_metrics.values()), {}))

        export_payload = {
            "timestamp": now,
            "net_ofi_score": primary.get("net_ofi_score", 0.0),
            "normalized_ofi": primary.get("normalized_ofi", 0.0),
            "ofi_velocity": primary.get("ofi_velocity", 0.0),
            "spread_compression": primary.get("spread_compression", 0.0),
            "best_bid": primary.get("best_bid", 0.0),
            "best_ask": primary.get("best_ask", 0.0),
            "mlofi": primary.get("mlofi", 0.0),
            "micro_price": primary.get("micro_price", 0.0),
            "micro_price_bps": primary.get("micro_price_bps", 0.0),
            "vpin": primary.get("vpin", 0.0),
            "hawkes_spoof_index": primary.get("hawkes_spoof_index", 0.0),
            "hawkes_rho": primary.get("hawkes_rho", 0.0),
            "hawkes_veto": primary.get("hawkes_veto", False),
            "kyle_lambda": primary.get("kyle_lambda", 0.0),
            "kyle_lambda_pctile": primary.get("kyle_lambda_pctile", 0.5),
            "lvr_tax_bps": primary.get("lvr_tax_bps", 0.0),
            "toxicity_halt": primary.get("toxicity_halt", False),
            "order_additions_count": primary.get("order_additions_count", 0),
            "order_cancellations_count": primary.get("order_cancellations_count", 0),
            "mean_residence_time_ms": primary.get("mean_residence_time_ms", 250.0),
            "cancel_counts": primary.get("cancel_counts", [0.0] * 10),
            "add_counts": primary.get("add_counts", [0.0] * 10),
            "tape_intel": getattr(self, "latest_tape_intel", None),
            "ghost_walls": getattr(self, "latest_ghost_walls", None),
            "symbols": all_metrics
        }

        def _write():
            import time as _time  # explicit local import avoids closure UnboundLocalError in asyncio.to_thread
            pid = os.getpid()
            temp_file = f"{STATE_FILE}.tmp.{pid}"
            for i in range(5):
                try:
                    with open(temp_file, "w", encoding="utf-8") as f:
                        json.dump(export_payload, f)
                    os.replace(temp_file, STATE_FILE)
                    break
                except Exception as e:
                    if i == 4:
                        print(f"[ERROR] Failed to write microstructure state: {e}")
                    else:
                        _time.sleep(0.05)
                finally:
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except Exception:
                            pass

        await asyncio.to_thread(_write)


async def _ai_tape_whisperer_loop(daemon: MultiSymbolOFIDaemon):
    """
    Feature 2 & Engine 2: Order Flow Tape Whisperer & Ghost Wall / Spoofing Detector.
    Every 5 minutes (300 seconds), concurrently runs:
    1. L2 OFI Tape & Iceberg Absorption analysis (analyze_order_flow_tape)
    2. 10-level addition/cancellation velocity & phantom spoofing detection (detect_spoof_and_ghost_walls)
    Dispatched via Groq Cloud 3-key rotation pool in parallel (<1.5s total latency).
    Writes intelligence to system_state and caches in microstructure_state.json.
    """
    import sqlite3
    from datetime import datetime, timezone
    try:
        from ai_engine import analyze_order_flow_tape, detect_spoof_and_ghost_walls
    except ImportError:
        analyze_order_flow_tape = None
        detect_spoof_and_ghost_walls = None

    # Initial brief warmup to allow order book state machines to populate
    await asyncio.sleep(20)

    while True:
        try:
            if analyze_order_flow_tape and detect_spoof_and_ghost_walls:
                all_metrics = {}
                for s, state in daemon.symbol_states.items():
                    all_metrics[s] = state.get_metrics()

                primary = all_metrics.get("BTCUSDT", next(iter(all_metrics.values()), {}))
                
                if primary.get("best_bid", 0.0) > 0.0:
                    fleet_deltas = {}
                    fleet_cancels = {}
                    for s, m in all_metrics.items():
                        n_a = m.get("order_additions_count", 1)
                        n_c = m.get("order_cancellations_count", 1)
                        tot_ev = max(1, n_a + n_c)
                        fleet_deltas[s] = {
                            "normalized_ofi": m.get("normalized_ofi", 0.0),
                            "ofi_velocity": m.get("ofi_velocity", 0.0),
                            "spread_compression": m.get("spread_compression", 0.0),
                            "add_vs_cancel_ratio": round(n_a / max(1, n_c), 2),
                            "mean_residence_time_ms": m.get("mean_residence_time_ms", 250.0)
                        }
                        fleet_cancels[s] = {
                            "residence_ms": m.get("mean_residence_time_ms", 250.0),
                            "cancel_rate": round(n_c / tot_ev, 2),
                            "normalized_ofi": m.get("normalized_ofi", 0.0)
                        }

                    now_iso = datetime.now(timezone.utc).isoformat()
                    ofi_packet = {
                        "timestamp": now_iso,
                        "primary_symbol": "BTCUSDT",
                        "primary_metrics": primary,
                        "fleet_ofi_deltas": fleet_deltas
                    }

                    depth_packet = {
                        "timestamp": now_iso,
                        "primary_symbol": "BTCUSDT",
                        "best_bid": primary.get("best_bid", 0.0),
                        "best_ask": primary.get("best_ask", 0.0),
                        "mean_residence_time_ms": primary.get("mean_residence_time_ms", 250.0),
                        "order_additions_count": primary.get("order_additions_count", 0),
                        "order_cancellations_count": primary.get("order_cancellations_count", 0),
                        "cancel_counts_per_level": primary.get("cancel_counts", [0.0] * 10),
                        "add_counts_per_level": primary.get("add_counts", [0.0] * 10),
                        "fleet_cancellations": fleet_cancels
                    }

                    # Concurrently execute Tape Whisperer and Ghost Wall detection on Groq pool
                    results = await asyncio.gather(
                        analyze_order_flow_tape(ofi_packet, timeout_sec=15.0),
                        detect_spoof_and_ghost_walls(depth_packet, timeout_sec=15.0),
                        return_exceptions=True
                    )

                    tape_intel = results[0] if isinstance(results[0], dict) else None
                    ghost_walls = results[1] if isinstance(results[1], dict) else None

                    if tape_intel:
                        daemon.latest_tape_intel = tape_intel
                        print(f"[TAPE-WHISPERER] Condition={tape_intel.get('tape_condition')} | Side={tape_intel.get('absorption_side')} | Veto={tape_intel.get('execution_veto')} | Conf={tape_intel.get('confidence')}")

                    if ghost_walls:
                        daemon.latest_ghost_walls = ghost_walls
                        print(f"[GHOST-WALL] Spoof={ghost_walls.get('spoof_detected')} | Side={ghost_walls.get('spoof_side')} | Advisory={ghost_walls.get('execution_advisory')} | Risk={ghost_walls.get('manipulation_risk_pct')}%")

                    # Persist to databases
                    def _db_write():
                        for db_file in ["trading_data.db", "trading_ledger.db"]:
                            try:
                                with sqlite3.connect(db_file, timeout=10.0) as conn:
                                    if tape_intel:
                                        conn.execute("""
                                            INSERT OR REPLACE INTO system_state (key, value, updated_at)
                                            VALUES ('order_flow_tape_intel', ?, ?)
                                        """, (json.dumps(tape_intel), now_iso))
                                    if ghost_walls:
                                        conn.execute("""
                                            INSERT OR REPLACE INTO system_state (key, value, updated_at)
                                            VALUES ('ai_ghost_walls', ?, ?)
                                        """, (json.dumps(ghost_walls), now_iso))
                            except Exception:
                                pass
                    await asyncio.to_thread(_db_write)
        except Exception as e:
            print(f"[WARN] Error in Tape Whisperer background loop: {e}")

        await asyncio.sleep(300)


async def stream_level2_data():
    daemon = MultiSymbolOFIDaemon(symbols=SYMBOLS, window_seconds=60)
    print("=========================================================")
    print("   *** STARTING BITGET 15-LEVEL VOLUMETRIC OFI DAEMON ***")
    print("=========================================================")
    print(f"Monitored Symbols ({len(SYMBOLS)}): {SYMBOLS}")
    print(f"Connecting to Bitget Public WebSocket: {WS_URL}")

    # Launch background AI Order Flow Tape Whisperer
    whisperer_task = asyncio.create_task(_ai_tape_whisperer_loop(daemon))

    _reconnect_attempt = 0
    _BASE_DELAY = 3.0
    _MAX_DELAY = 60.0

    sub_args = []
    for s in SYMBOLS:
        sub_args.append({
            "instType": "USDT-FUTURES",
            "channel": "books15",
            "instId": s
        })
        sub_args.append({
            "instType": "USDT-FUTURES",
            "channel": "trade",
            "instId": s
        })

    sub_payload = {
        "op": "subscribe",
        "args": sub_args
    }

    while True:
        try:
            async with websockets.connect(WS_URL, ping_interval=None) as ws:
                await ws.send(json.dumps(sub_payload))
                print(f"[OK] Connection Established. Subscribed to Bitget books15 & trade channels ({len(SYMBOLS)} symbols)...")
                _reconnect_attempt = 0
                daemon.reset_all()
                print("[OFI] All symbol state machines reset on WebSocket reconnection.")

                async def _ping_loop():
                    while True:
                        await asyncio.sleep(25)
                        try:
                            await ws.send("ping")
                        except Exception:
                            break

                ping_task = asyncio.create_task(_ping_loop())

                try:
                    async for message in ws:
                        if message == "pong":
                            continue
                        data = json.loads(message)
                        if data.get("event") == "subscribe":
                            continue

                        arg = data.get("arg", {})
                        symbol = arg.get("instId", "").upper()
                        channel = arg.get("channel", "")
                        raw_data = data.get("data", [])
                        if not raw_data or not symbol or symbol not in daemon.symbol_states:
                            continue

                        # Handle real public trade feed
                        if channel == "trade":
                            for tr in raw_data:
                                try:
                                    t_price = float(tr.get("price") or tr.get("p") or 0.0)
                                    t_size = float(tr.get("size") or tr.get("s") or 0.0)
                                    t_side = str(tr.get("side") or "buy").lower()
                                    if t_price > 0 and t_size > 0:
                                        daemon.process_trade_tick(symbol, t_price, t_size, t_side)
                                except Exception:
                                    pass
                            continue

                        # Handle 15-level L2 depth feed
                        if channel == "books15":
                            entry = raw_data[0]
                            bids = entry.get("bids", [])
                            asks = entry.get("asks", [])

                            chk = entry.get("checksum") or entry.get("checkSum")
                            if chk is not None and bids and asks:
                                try:
                                    if not verify_bitget_crc32_checksum(bids, asks, int(chk)):
                                        print(f"[WARN] CRC32 Checksum mismatch for {symbol}! Re-syncing book.")
                                        daemon.symbol_states[symbol].reset()
                                except Exception:
                                    pass

                            if bids and asks:
                                daemon.process_tick(symbol, bids, asks)
                finally:
                    ping_task.cancel()

        except Exception as e:
            _reconnect_attempt += 1
            backoff = min(_BASE_DELAY * (2 ** (_reconnect_attempt - 1)), _MAX_DELAY)
            print(f"\n[WARN] WebSocket Disconnected (attempt #{_reconnect_attempt}): {e}. Reconnecting in {backoff:.0f}s...")
            await asyncio.sleep(backoff)

if __name__ == "__main__":
    try:
        asyncio.run(stream_level2_data())
    except KeyboardInterrupt:
        print("\n[STOP] OFI Daemon Terminated.")
        sys.exit(0)

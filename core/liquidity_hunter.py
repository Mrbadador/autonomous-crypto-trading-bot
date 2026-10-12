# core/liquidity_hunter.py
import numpy as np
import logging

log = logging.getLogger("liquidity_hunter")

# =============================================================================
# MLOFI CONFIGURATION (V30.5 Deep-Book Whale Spoofing Defense)
# =============================================================================
MLOFI_DEPTH:   int   = 10    # D: number of levels to aggregate
MLOFI_KAPPA:   float = 0.5   # κ: exponential decay steepness
MLOFI_ZSCORE_WINDOW: int = 200  # rolling buffer for Z-score normalization

# Pre-compute level weights once at import time: w_d = exp(-κ * (d-1)) for d=1..D
_MLOFI_WEIGHTS = np.array(
    [np.exp(-MLOFI_KAPPA * d) for d in range(MLOFI_DEPTH)],
    dtype=np.float64
)


class MLOFICalculator:
    """
    Multi-Level Order Flow Imbalance (MLOFI) engine with rolling Z-score normalization.

    At each depth snapshot the signed delta-volume at every level is weighted by
    an exponential decay function that discounts deeper, less-reliable book levels:

        w_d  = exp(-κ * (d - 1))          (d = 1 … D, zero-indexed here)
        MLOFI_raw = Σ w_d * (ΔBid_d - ΔAsk_d)

    The raw value is then normalized via a rolling Z-score so that the PyTorch
    FlowClassifier always receives a signal on the same scale regardless of
    symbol notional size:

        MLOFI_z = (MLOFI_raw - μ_window) / (σ_window + ε)

    The rolling buffer is bounded (MLOFI_ZSCORE_WINDOW samples) to remain
    memory-safe during multi-day runs.
    """

    def __init__(self, depth: int = MLOFI_DEPTH, kappa: float = MLOFI_KAPPA,
                 window: int = MLOFI_ZSCORE_WINDOW):
        self.depth   = depth
        self.weights = np.array(
            [np.exp(-kappa * d) for d in range(depth)], dtype=np.float64
        )
        # Rolling buffer for Z-score normalisation
        self._buffer = np.zeros(window, dtype=np.float64)
        self._ptr    = 0
        self._filled = False
        self._window = window

    # ------------------------------------------------------------------
    def compute(self, bids: list, asks: list,
                prev_bids: list, prev_asks: list) -> float:
        """
        Compute the normalized MLOFI from two consecutive depth snapshots.

        Parameters
        ----------
        bids / asks       : current snapshot  [[price, qty], ...]
        prev_bids / prev_asks : previous snapshot [[price, qty], ...]

        Returns
        -------
        mlofi_z : float
            Z-score normalized MLOFI value (range typically ~[-4, +4]).
            Returns 0.0 on degenerate / missing data.
        """
        raw = self._raw_mlofi(bids, asks, prev_bids, prev_asks)
        return self._normalize(raw)

    # ------------------------------------------------------------------
    def _raw_mlofi(self, bids, asks, prev_bids, prev_asks) -> float:
        """
        Fallback-safe Multi-Level OFI calculation.
        Gracefully handles snapshots with fewer than D levels.
        """
        if not bids or not asks:
            return 0.0

        # Use as many levels as available in the smaller of current / prev
        n_bid = min(len(bids), len(prev_bids) if prev_bids else 0, self.depth)
        n_ask = min(len(asks), len(prev_asks) if prev_asks else 0, self.depth)

        if n_bid == 0 and n_ask == 0:
            return 0.0

        # Convert to numpy (price, qty) arrays
        b_cur  = np.array(bids[:n_bid],      dtype=np.float64) if n_bid > 0 else None
        b_prev = np.array(prev_bids[:n_bid], dtype=np.float64) if n_bid > 0 else None
        a_cur  = np.array(asks[:n_ask],      dtype=np.float64) if n_ask > 0 else None
        a_prev = np.array(prev_asks[:n_ask], dtype=np.float64) if n_ask > 0 else None

        bid_contribution = 0.0
        if b_cur is not None and b_prev is not None:
            for d in range(n_bid):
                w_d = self.weights[d]
                p_cur,  q_cur  = b_cur[d,  0], b_cur[d,  1]
                p_prev, q_prev = b_prev[d, 0], b_prev[d, 1]

                # Delta-bid: positive if price rose (aggressive bid) or size increased
                if p_cur >= p_prev:
                    delta_bid = q_cur
                elif p_cur < p_prev:
                    delta_bid = -q_cur
                else:
                    delta_bid = q_cur - q_prev

                bid_contribution += w_d * delta_bid

        ask_contribution = 0.0
        if a_cur is not None and a_prev is not None:
            for d in range(n_ask):
                w_d = self.weights[d]
                p_cur,  q_cur  = a_cur[d,  0], a_cur[d,  1]
                p_prev, q_prev = a_prev[d, 0], a_prev[d, 1]

                # Delta-ask: positive if price fell (aggressive ask) or size increased
                if p_cur <= p_prev:
                    delta_ask = q_cur
                elif p_cur > p_prev:
                    delta_ask = -q_cur
                else:
                    delta_ask = q_cur - q_prev

                ask_contribution += w_d * delta_ask

        return bid_contribution - ask_contribution

    # ------------------------------------------------------------------
    def _normalize(self, raw: float) -> float:
        """
        Rolling Z-score normalization over the last MLOFI_ZSCORE_WINDOW samples.
        Returns the raw value unchanged until the buffer has ≥2 observations.
        """
        self._buffer[self._ptr] = raw
        self._ptr = (self._ptr + 1) % self._window
        if self._ptr == 0:
            self._filled = True

        active = self._buffer if self._filled else self._buffer[:self._ptr]
        if len(active) < 2:
            # Cold-start guard: never feed unscaled raw magnitudes to the neural net.
            # Return 0.0 (neutral) until the ring buffer has enough samples to compute
            # a meaningful Z-score (prevents gradient blow-out in the sigmoid layer).
            return 0.0

        mu  = np.mean(active)
        sig = np.std(active)
        return float((raw - mu) / (sig + 1e-9))


# =============================================================================
# LIQUIDITY HUNTER (Upgraded to V30.5 with MLOFI)
# =============================================================================
class LiquidityHunter:
    def __init__(self, depth_levels: int = 20, volume_shock_threshold: float = 2.5):
        self.depth_levels = depth_levels
        self.volume_shock_threshold = volume_shock_threshold
        # MLOFI engine with its own state
        self.mlofi_calc = MLOFICalculator()
        # Previous snapshot for delta computation
        self._prev_bids: list = []
        self._prev_asks: list = []

    def analyze_order_book(self, bids: list, asks: list) -> dict:
        """
        Vectorized analysis of the order book arrays.
        bids/asks format: [[price, quantity], [price, quantity], ...]

        Returns
        -------
        dict with keys:
            liquidity_imbalance : float  [-1, +1]
            mlofi               : float  rolling-Z-score normalized MLOFI
            vacuum_detected     : bool
            target_walls        : dict
            signal_multiplier   : float
        """
        if not bids or not asks:
            self._prev_bids, self._prev_asks = [], []
            return {
                "liquidity_imbalance": 0.0,
                "mlofi": 0.0,
                "vacuum_detected": False,
                "target_walls": {},
                "signal_multiplier": 1.0,
            }

        # --- V30.5: MLOFI computation (uses prev snapshot for delta) ---------
        mlofi_z = self.mlofi_calc.compute(
            bids, asks,
            self._prev_bids, self._prev_asks
        )
        # Store current snapshot for next tick's delta
        self._prev_bids = bids
        self._prev_asks = asks

        # Convert to float64 numpy arrays for vectorized analysis
        b_arr = np.array(bids, dtype=np.float64)
        a_arr = np.array(asks, dtype=np.float64)

        b_prices, b_sizes = b_arr[:, 0], b_arr[:, 1]
        a_prices, a_sizes = a_arr[:, 0], a_arr[:, 1]

        # 1. Cumulative Depth Profiles
        cum_bid_depth = np.cumsum(b_sizes)
        cum_ask_depth = np.cumsum(a_sizes)

        total_bid_liquidity = cum_bid_depth[-1]
        total_ask_liquidity = cum_ask_depth[-1]

        # 2. Liquidity Imbalance (L1-style total ratio, preserved for legacy HUD)
        denom = total_bid_liquidity + total_ask_liquidity
        liquidity_imbalance = (
            (total_bid_liquidity - total_ask_liquidity) / denom
            if denom != 0 else 0.0
        )

        # 3. Vacuum Zone detection (top-5 levels vs median thickness)
        bid_median = np.median(b_sizes)
        ask_median = np.median(a_sizes)
        bid_vacuum  = bool(np.any(b_sizes[:5] < (bid_median * 0.15)))
        ask_vacuum  = bool(np.any(a_sizes[:5] < (ask_median * 0.15)))

        # 4. Liquidity Walls / Stop-Loss Clusters
        large_bid_idx = np.where(b_sizes > (bid_median * self.volume_shock_threshold))[0]
        large_ask_idx = np.where(a_sizes > (ask_median * self.volume_shock_threshold))[0]

        target_wall: dict = {}
        if large_bid_idx.size > 0:
            target_wall["bid_wall_price"] = float(b_prices[large_bid_idx[0]])
            target_wall["bid_wall_size"]  = float(b_sizes[large_bid_idx[0]])
        if large_ask_idx.size > 0:
            target_wall["ask_wall_price"] = float(a_prices[large_ask_idx[0]])
            target_wall["ask_wall_size"]  = float(a_sizes[large_ask_idx[0]])

        # Signal multiplier: amplify on bullish vacuum + imbalance
        signal_multiplier = (
            1.1 if (bid_vacuum and liquidity_imbalance > 0.3) else 0.9
        )

        return {
            "liquidity_imbalance": float(liquidity_imbalance),
            "mlofi":               float(mlofi_z),
            "vacuum_detected":     bool(bid_vacuum or ask_vacuum),
            "target_walls":        target_wall,
            "signal_multiplier":   signal_multiplier,
        }

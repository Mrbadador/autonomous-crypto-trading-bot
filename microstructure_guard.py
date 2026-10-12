import logging
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
import asyncio

log = logging.getLogger("microstructure_guard")

# =============================================================================
# DYNAMIC RESILIENCY WRAPPERS (Soft Fallbacks for missing dependencies)
# =============================================================================
try:
    from numba import njit
except ImportError:
    log.warning("[DEP-WARN] 'numba' not installed. Falling back to pure Python loops (Hawkes process latency may spike +5ms).")
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(f):
            return f
        return decorator

try:
    from scipy.signal import stft
except ImportError:
    log.warning("[DEP-WARN] 'scipy' not installed. Falling back to vectorized Numpy-native STFT approximation.")
    def stft(x, nperseg=64, noverlap=56, nfft=128, boundary=None):
        """Numpy sliding FFT providing mathematically equivalent Zxx output shapes for feature extractor."""
        step = nperseg - noverlap
        x_len = len(x)
        n_frames = 1 + (x_len - nperseg) // step
        if n_frames <= 0:
            # Minimal pad
            pad = np.zeros(max(0, nperseg - x_len))
            x = np.concatenate([x, pad])
            n_frames = 1
        
        frames = []
        for i in range(n_frames):
            start = i * step
            frame_seg = x[start:start + nperseg]
            windowed = frame_seg * np.hamming(len(frame_seg))
            frames.append(np.fft.rfft(windowed, n=nfft))
            
        Zxx = np.column_stack(frames)
        return None, None, Zxx


# =============================================================================
# EXACT PRODUCTION PARAMETERS (Spoofing Detector)
# =============================================================================
N_LOB_LEVELS: int = 10
HAWKES_MU: float = 0.1          # Base intensity (events/sec)
HAWKES_ALPHA: float = 0.8       # Jump magnitude
HAWKES_BETA: float = 2.5        # Exponential decay (1/sec)
VETO_THRESHOLD_STATIC: float = 0.65
ADAPTIVE_K: float = 2.5
ADAPTIVE_WINDOW: int = 100
GAMMA_DECAY: float = 0.3
EPS: float = 1e-9

# =============================================================================
# EXACT PRODUCTION PARAMETERS (VPIN Classifier)
# =============================================================================
VPIN_BUCKETS: int = 50
VPIN_WINDOW: int = 50          # rolling std window
BUCKET_VOL_DIVISOR: int = 50   # V_bar = V_24h / 50
STFT_NPERSEG: int = 64
STFT_NOVERLAP: int = 56
STFT_NFFT: int = 128
FUNDING_SCALE: float = 0.1     # κ_f
VPIN_HIGH: float = 0.85        # hard override threshold
INFORMED_THRESHOLD: float = 0.70
CAUTION_THRESHOLD: float = 0.45


# =============================================================================
# EXACT PRODUCTION PARAMETERS (Bivariate Hawkes & Spoofing Guard)
# =============================================================================
HAWKES_MU_A: float = 1.2
HAWKES_MU_C: float = 0.8
HAWKES_ALPHA_AA: float = 0.4
HAWKES_BETA_AA: float = 2.0
HAWKES_ALPHA_AC: float = 0.1
HAWKES_BETA_AC: float = 3.0
HAWKES_ALPHA_CA: float = 0.8
HAWKES_BETA_CA: float = 1.5
HAWKES_ALPHA_CC: float = 0.3
HAWKES_BETA_CC: float = 2.5
HAWKES_TAU_THRESHOLD: float = 150.0  # ms
try:
    import config
    SPOOF_VETO_THRESHOLD: float = float(getattr(config, "HAWKES_SPOOF_VETO_THRESHOLD", 0.78))
    CASCADE_RHO_THRESHOLD: float = float(getattr(config, "HAWKES_CASCADE_RHO_THRESHOLD", 1.05))
except Exception:
    SPOOF_VETO_THRESHOLD: float = 0.78
    CASCADE_RHO_THRESHOLD: float = 1.05
STATE_FILE: str = "microstructure_state.json"


# =============================================================================
# JIT COMPILATION ENGINE (Latency Optimized)
# =============================================================================
@njit(cache=True, fastmath=True)
def _bivariate_hawkes_step(
    lambda_a: float, lambda_c: float,
    S_aa: float, S_ac: float, S_ca: float, S_cc: float,
    dt: float, n_a: float, n_c: float,
    mu_a: float, mu_c: float,
    alpha_aa: float, alpha_ac: float, alpha_ca: float, alpha_cc: float,
    beta_aa: float, beta_ac: float, beta_ca: float, beta_cc: float
) -> Tuple[float, float, float, float, float, float]:
    """
    Discrete-time recursive update for Bivariate Marked Hawkes Process.
    Tracks cross-excitation between additions (lambda_a) and cancellations (lambda_c).
    """
    if dt <= 0.0:
        dt = 0.001

    decay_aa = np.exp(-beta_aa * dt)
    decay_ac = np.exp(-beta_ac * dt)
    decay_ca = np.exp(-beta_ca * dt)
    decay_cc = np.exp(-beta_cc * dt)

    new_S_aa = S_aa * decay_aa + alpha_aa * n_a
    new_S_ac = S_ac * decay_ac + alpha_ac * n_c
    new_S_ca = S_ca * decay_ca + alpha_ca * n_a
    new_S_cc = S_cc * decay_cc + alpha_cc * n_c

    new_lambda_a = mu_a + new_S_aa + new_S_ac
    new_lambda_c = mu_c + new_S_ca + new_S_cc

    if new_lambda_a < mu_a:
        new_lambda_a = mu_a
    elif new_lambda_a > 1000.0:
        new_lambda_a = 1000.0

    if new_lambda_c < mu_c:
        new_lambda_c = mu_c
    elif new_lambda_c > 1000.0:
        new_lambda_c = 1000.0

    return new_lambda_a, new_lambda_c, new_S_aa, new_S_ac, new_S_ca, new_S_cc


@njit(cache=True, fastmath=True)
def _compute_spectral_radius(
    alpha_aa: float, alpha_ac: float, alpha_ca: float, alpha_cc: float,
    beta_aa: float, beta_ac: float, beta_ca: float, beta_cc: float
) -> float:
    """
    Computes spectral radius rho(Gamma) of 2x2 Hawkes branching matrix:
    Gamma = [[alpha_aa/beta_aa, alpha_ac/beta_ac],
             [alpha_ca/beta_ca, alpha_cc/beta_cc]]
    """
    g11 = alpha_aa / beta_aa
    g12 = alpha_ac / beta_ac
    g21 = alpha_ca / beta_ca
    g22 = alpha_cc / beta_cc

    tr = g11 + g22
    det = g11 * g22 - g12 * g21
    disc = tr * tr - 4.0 * det
    if disc < 0.0:
        disc = 0.0
    rho = (tr + np.sqrt(disc)) / 2.0
    return rho


@njit(cache=True, fastmath=True)
def _compute_spoof_index(
    lambda_a: float, lambda_c: float, lambda_exec: float,
    tau_res: float, tau_thresh: float, eps: float
) -> float:
    """
    Computes Spoofing & Layering Index S_spoof(t).
    S_spoof(t) = tanh((lambda_c - lambda_exec)/(lambda_a + eps)) * exp(-tau_res / tau_thresh)
    """
    num = lambda_c - lambda_exec
    den = lambda_a + eps
    ratio = num / den
    factor_res = np.exp(-max(0.0, tau_res) / tau_thresh)
    spoof = np.tanh(ratio) * factor_res
    return spoof


@njit(cache=True, fastmath=True)
def _update_intensity(intensity: float, dt: float, n_events: int,
                      mu: float, alpha: float, beta: float) -> float:
    """
    Recursive exponential Hawkes intensity update.
    λ(t+Δt) = μ + (λ(t) - μ) * exp(-β * Δt) + α * ΔN
    """
    decay = np.exp(-beta * dt)
    new_intensity = mu + (intensity - mu) * decay + alpha * float(n_events)
    if new_intensity < 0.0:
        new_intensity = mu
    return new_intensity


@njit(cache=True, fastmath=True)
def _compute_toxicity(intensities_c: np.ndarray,
                      intensities_a: np.ndarray,
                      weights: np.ndarray) -> float:
    """
    Compute multi-scale toxicity score T(t) = Σ w_i * tanh((λ_c - λ_a)/(λ_c + λ_a + ε))
    """
    n = intensities_c.shape[0]
    score = 0.0
    for i in range(n):
        lam_c = intensities_c[i]
        lam_a = intensities_a[i]
        denom = lam_c + lam_a + EPS
        s_i = (lam_c - lam_a) / denom
        s_i = np.tanh(s_i)  # squash to [-1, 1]
        score += weights[i] * s_i
    return score


@njit(cache=True, fastmath=True)
def _compute_volume_buckets(trade_volumes: np.ndarray,
                            trade_signs: np.ndarray,
                            bucket_size: float) -> Tuple[np.ndarray, int]:
    """
    Aggregate tick-level signed volume into fixed-size volume buckets.
    Returns:
        deltas : np.ndarray, shape (VPIN_BUCKETS,)
        n_filled : int
    """
    n = trade_volumes.shape[0]
    deltas = np.empty(VPIN_BUCKETS, dtype=np.float64)
    current_delta = 0.0
    current_vol = 0.0
    bucket_idx = 0
    for i in range(n):
        if bucket_idx >= VPIN_BUCKETS:
            break
        current_delta += trade_signs[i] * trade_volumes[i]
        current_vol += trade_volumes[i]
        if current_vol >= bucket_size:
            deltas[bucket_idx] = current_delta
            bucket_idx += 1
            current_delta = 0.0
            current_vol = 0.0
    return deltas, bucket_idx


# =============================================================================
# MICROSTRUCTURE STATE CACHE (Zero-IO Hot-Path Reader)
# =============================================================================
class MicrostructureStateCache:
    """
    In-memory cache for microstructure_state.json with mtime tracking.
    Ensures pre-trade veto executes in < 1.0 ms without disk I/O bottlenecks.
    """
    _cached_data: dict = {}
    _last_mtime: float = 0.0
    _last_check_ts: float = 0.0

    @classmethod
    def get_state(cls, symbol: str = "BTCUSDT") -> dict:
        now = time.time()
        if now - cls._last_check_ts > 0.2:
            cls._last_check_ts = now
            if os.path.exists(STATE_FILE):
                try:
                    mtime = os.path.getmtime(STATE_FILE)
                    if mtime > cls._last_mtime:
                        with open(STATE_FILE, "r", encoding="utf-8") as f:
                            cls._cached_data = json.load(f)
                            cls._last_mtime = mtime
                except Exception:
                    pass
        symbols = cls._cached_data.get("symbols", {})
        if symbol in symbols:
            return symbols[symbol]
        return cls._cached_data


# =============================================================================
# BIVARIATE HAWKES POINT-PROCESS MODEL
# =============================================================================
class BivariateHawkesModel:
    """
    Recursive Bivariate Marked Hawkes Process tracking mutual excitation
    between order additions (Na) and cancellations (Nc).
    Computes spectral radius rho(Gamma) and Spoofing Index S_spoof.
    """
    def __init__(self,
                 mu_a: float = HAWKES_MU_A, mu_c: float = HAWKES_MU_C,
                 alpha_aa: float = HAWKES_ALPHA_AA, beta_aa: float = HAWKES_BETA_AA,
                 alpha_ac: float = HAWKES_ALPHA_AC, beta_ac: float = HAWKES_BETA_AC,
                 alpha_ca: float = HAWKES_ALPHA_CA, beta_ca: float = HAWKES_BETA_CA,
                 alpha_cc: float = HAWKES_ALPHA_CC, beta_cc: float = HAWKES_BETA_CC,
                 tau_thresh: float = HAWKES_TAU_THRESHOLD):
        self.mu_a = float(mu_a)
        self.mu_c = float(mu_c)
        self.alpha_aa = float(alpha_aa)
        self.beta_aa = float(beta_aa)
        self.alpha_ac = float(alpha_ac)
        self.beta_ac = float(beta_ac)
        self.alpha_ca = float(alpha_ca)
        self.beta_ca = float(beta_ca)
        self.alpha_cc = float(alpha_cc)
        self.beta_cc = float(beta_cc)
        self.tau_thresh = float(tau_thresh)

        # Precompute default branching matrix spectral radius
        self.rho = _compute_spectral_radius(
            self.alpha_aa, self.alpha_ac, self.alpha_ca, self.alpha_cc,
            self.beta_aa, self.beta_ac, self.beta_ca, self.beta_cc
        )

        # Per-symbol state tracking
        self.states = {}

    def _get_symbol_state(self, symbol: str) -> dict:
        if symbol not in self.states:
            self.states[symbol] = {
                "last_time": time.time(),
                "lambda_a": self.mu_a,
                "lambda_c": self.mu_c,
                "S_aa": 0.0,
                "S_ac": 0.0,
                "S_ca": 0.0,
                "S_cc": 0.0,
                "spoof_index": 0.0
            }
        return self.states[symbol]

    def update(self, symbol: str, dt: float, n_a: float, n_c: float,
               lambda_exec: float = 0.0, tau_res: float = 150.0) -> Tuple[float, float, float, float]:
        """
        Executes single recursive step for symbol in < 5 us.
        Returns:
            (lambda_a, lambda_c, spoof_index, rho)
        """
        st = self._get_symbol_state(symbol)

        la, lc, saa, sac, sca, scc = _bivariate_hawkes_step(
            st["lambda_a"], st["lambda_c"],
            st["S_aa"], st["S_ac"], st["S_ca"], st["S_cc"],
            float(dt), float(n_a), float(n_c),
            self.mu_a, self.mu_c,
            self.alpha_aa, self.alpha_ac, self.alpha_ca, self.alpha_cc,
            self.beta_aa, self.beta_ac, self.beta_ca, self.beta_cc
        )
        st["lambda_a"] = la
        st["lambda_c"] = lc
        st["S_aa"] = saa
        st["S_ac"] = sac
        st["S_ca"] = sca
        st["S_cc"] = scc

        spoof_idx = _compute_spoof_index(
            la, lc, float(lambda_exec), float(tau_res), self.tau_thresh, EPS
        )
        st["spoof_index"] = spoof_idx

        return float(la), float(lc), float(spoof_idx), float(self.rho)


# =============================================================================
# CORE AGENTS: SPOOFING DETECTOR
# =============================================================================
class SpoofingDetector:
    """
    Institutional-grade Spoofing and Layering Detector using multi-scale Hawkes processes.
    Assesses LOB toxicity on additions vs. cancellations.
    """
    def __init__(self, n_levels: int = N_LOB_LEVELS):
        self.n_levels = n_levels
        self.mu = HAWKES_MU
        self.alpha = HAWKES_ALPHA
        self.beta = HAWKES_BETA

        # Running intensities
        self.lambda_c = np.full(n_levels, self.mu, dtype=np.float64)
        self.lambda_a = np.full(n_levels, self.mu, dtype=np.float64)

        # Level weight allocation
        raw_weights = np.exp(-GAMMA_DECAY * np.arange(n_levels))
        self.weights = raw_weights / raw_weights.sum()

        # Rolling adaptive buffer
        self.tox_history = np.zeros(1000, dtype=np.float64)
        self.tox_idx = 0
        self.tox_count = 0

    def update(self, dt: float, cancel_counts: np.ndarray, add_counts: np.ndarray) -> Tuple[float, bool]:
        """
        Updates model intensities and emits Veto signal (< 5ms latency target).
        """
        if dt <= 0:
            dt = 1e-3 # avoid zero division or null decay

        # Recursive update loop
        for i in range(self.n_levels):
            self.lambda_c[i] = _update_intensity(
                self.lambda_c[i], dt, int(cancel_counts[i]),
                self.mu, self.alpha, self.beta
            )
            self.lambda_a[i] = _update_intensity(
                self.lambda_a[i], dt, int(add_counts[i]),
                self.mu, self.alpha, self.beta
            )

        # Global toxicity score T
        T = _compute_toxicity(self.lambda_c, self.lambda_a, self.weights)

        # Update ring buffer
        self.tox_history[self.tox_idx] = T
        self.tox_idx = (self.tox_idx + 1) % 1000
        if self.tox_count < 1000:
            self.tox_count += 1

        # Adaptive veto thresholding
        veto = False
        if self.tox_count >= ADAPTIVE_WINDOW:
            recent = self.tox_history[:self.tox_count]
            mu_t = np.mean(recent)
            sig_t = np.std(recent)
            adaptive_thresh = mu_t + ADAPTIVE_K * sig_t
            effective_thresh = max(VETO_THRESHOLD_STATIC, adaptive_thresh)
        else:
            effective_thresh = VETO_THRESHOLD_STATIC

        if T > effective_thresh:
            veto = True

        return float(T), veto


# =============================================================================
# CORE AGENTS: VPIN CLASSIFIER
# =============================================================================
class STFTFeatureExtractor:
    """
    Uses scipy STFT to analyze volume-velocity in frequency space.
    """
    def __init__(self, nperseg: int = STFT_NPERSEG,
                 noverlap: int = STFT_NOVERLAP,
                 nfft: int = STFT_NFFT):
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft

    def extract(self, volume_series: np.ndarray) -> np.ndarray:
        # Pad/truncate series for minimum length
        if len(volume_series) < self.nperseg:
            temp = np.zeros(self.nperseg, dtype=np.float64)
            temp[-len(volume_series):] = volume_series
            volume_series = temp

        f, t, Zxx = stft(
            volume_series,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft,
            boundary=None
        )
        mag = np.abs(Zxx)
        # Extract spectral bins
        low = np.mean(mag[0:2, :])
        mid = np.mean(mag[2:8, :])
        high = np.mean(mag[8:, :])
        return np.array([low, mid, high], dtype=np.float64)


class FlowClassifier(nn.Module):
    """Input(7) -> Hidden(32) -> Hidden(16) -> Output(2). [Noise, Informed]"""
    def __init__(self, input_dim: int = 7, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _VPINCore:
    """Ring buffer holding fixed delta buckets for standard deviation aggregation."""
    def __init__(self):
        self.deltas = np.zeros(VPIN_BUCKETS, dtype=np.float64)
        self.ptr = 0
        self.filled = False

    def update(self, trade_volumes: np.ndarray, trade_signs: np.ndarray, bucket_size: float) -> float:
        deltas, n_filled = _compute_volume_buckets(trade_volumes, trade_signs, bucket_size)
        if n_filled > 0:
            for i in range(n_filled):
                self.deltas[self.ptr] = deltas[i]
                self.ptr = (self.ptr + 1) % VPIN_BUCKETS
            if n_filled >= VPIN_BUCKETS:
                self.filled = True
        
        if self.filled or self.ptr > 1:
            active = self.deltas if self.filled else self.deltas[:self.ptr]
            sigma = np.std(active)
            vpin = sigma / (bucket_size + 1e-9)
        else:
            vpin = 0.0
        return vpin


class VPINClassifier:
    """
    Uses VPIN, STFT features, and a PyTorch Flow Classifier to detect informed flow velocity.
    """
    def __init__(self, device: str = "cpu"):
        import os
        self.device = device
        self.vpin_calc = _VPINCore()
        self.stft = STFTFeatureExtractor()
        self.classifier = FlowClassifier().to(device)
        
        self.has_weights = False
        if os.path.exists("flow_classifier.pth"):
            try:
                self.classifier.load_state_dict(torch.load("flow_classifier.pth", map_location=device, weights_only=True))
                self.has_weights = True
                log.info("[INIT] VPIN Flow Classifier weights successfully loaded.")
            except Exception as e:
                log.warning(f"[INIT-WARN] Could not load flow_classifier.pth: {e}")
        else:
            log.warning("[INIT-WARN] 'flow_classifier.pth' weights not found. VPIN Classifier will rely on deterministic math overrides until weights are trained/deployed.")
            
        self.classifier.eval() # Strictly for evaluation mode


    def evaluate(self,
                 trade_volumes: np.ndarray,
                 trade_signs: np.ndarray,
                 total_volume_24h: float,
                 funding_rate: float,
                 volume_series: np.ndarray,
                 ofi: float,
                 spread_bps: float) -> Tuple[int, float]:
        """
        Hot-path classifier step.
        Returns:
            action: 0 = PASS, 1 = BLOCK, 2 = CAUTION
            vpin_adj: float
        """
        # VPIN calc
        bucket_size = total_volume_24h / float(BUCKET_VOL_DIVISOR)
        vpin_raw = self.vpin_calc.update(trade_volumes, trade_signs, bucket_size)
        vpin_adj = vpin_raw * (1.0 + np.abs(funding_rate) * FUNDING_SCALE)

        # STFT
        stft_feats = self.stft.extract(volume_series)

        # Tensor formation
        if self.has_weights:
            x = np.array([
                vpin_adj,
                stft_feats[0], # Low
                stft_feats[1], # Mid
                stft_feats[2], # High
                funding_rate,
                ofi,
                spread_bps
            ], dtype=np.float32)
            
            x_t = torch.from_numpy(x).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.classifier(x_t)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            p_noise, p_informed = float(probs[0]), float(probs[1])
        else:
            p_noise, p_informed = 1.0, 0.0


        action = 0 # PASS default
        if vpin_adj > VPIN_HIGH or p_informed > INFORMED_THRESHOLD:
            action = 1 # BLOCK
        elif p_informed > CAUTION_THRESHOLD:
            action = 2 # CAUTION
            
        return action, float(vpin_adj)


# =============================================================================
# MICROSTRUCTURE GUARD COORDINATOR
# =============================================================================
class MicrostructureGuard:
    """
    High-performance asynchronous veto layer coordinating Hawkes Spoofing
    and VPIN Flow checks before a trade execution is permitted.

    Set MICROSTRUCTURE_ACTIVE_VETO=true in .env (default) to enable hard blocking.
    Set MICROSTRUCTURE_ACTIVE_VETO=false to revert to data-harvest Observation Mode
    where signals are recorded but trades are never blocked.
    """
    def __init__(self):
        import os
        self.bivariate_hawkes = BivariateHawkesModel()
        self.spoofing_detector = SpoofingDetector()
        self.vpin_classifier = VPINClassifier()
        active_veto_env = os.getenv("MICROSTRUCTURE_ACTIVE_VETO", "true").lower() == "true"
        observation_mode_env = os.getenv("OBSERVATION_MODE", "0") == "1"
        self.active_veto = active_veto_env and not observation_mode_env
        mode_label = "ACTIVE VETO" if self.active_veto else "OBSERVATION MODE"
        log.info(f"[INIT] Pre-Trade Microstructure Guard loaded. Mode: {mode_label}")

    async def evaluate_pre_trade_veto(self, context: Dict) -> Tuple[bool, str]:
        import os
        if os.getenv("OBSERVATION_MODE", "0") == "1":
            log.warning("[OBSERVATION MODE] Microstructure Guard bypassed — OBSERVATION_MODE=1 is set.")
            return True, "Observation Mode Override"
        try:
            raw_symbol = context.get("symbol", "BTCUSDT")
            symbol = str(raw_symbol).replace("/", "").upper() if raw_symbol else "BTCUSDT"

            # 1. Hawkes Bivariate Spoofing & Layering Verification
            # Retrieve order addition, cancellation, and residence time metrics
            cached_ms = MicrostructureStateCache.get_state(symbol)

            cancel_counts = context.get("cancel_counts")
            if cancel_counts is None:
                cached_cancels = cached_ms.get("cancel_counts")
                if cached_cancels and len(cached_cancels) == N_LOB_LEVELS:
                    cancel_counts = np.array(cached_cancels, dtype=np.float64)
                else:
                    cancel_counts = np.zeros(N_LOB_LEVELS, dtype=np.float64)
            elif not isinstance(cancel_counts, np.ndarray):
                cancel_counts = np.array(cancel_counts, dtype=np.float64)

            add_counts = context.get("add_counts")
            if add_counts is None:
                cached_adds = cached_ms.get("add_counts")
                if cached_adds and len(cached_adds) == N_LOB_LEVELS:
                    add_counts = np.array(cached_adds, dtype=np.float64)
                else:
                    add_counts = np.zeros(N_LOB_LEVELS, dtype=np.float64)
            elif not isinstance(add_counts, np.ndarray):
                add_counts = np.array(add_counts, dtype=np.float64)

            dt = float(context.get("dt", 1.0))
            if dt <= 0:
                dt = 1e-3

            # Calculate scalar additions and cancellations
            n_a = float(context.get("order_additions_count", context.get("n_a", np.sum(add_counts))))
            if n_a == 0.0:
                n_a = float(cached_ms.get("order_additions_count", np.sum(add_counts)))

            n_c = float(context.get("order_cancellations_count", context.get("n_c", np.sum(cancel_counts))))
            if n_c == 0.0:
                n_c = float(cached_ms.get("order_cancellations_count", np.sum(cancel_counts)))

            tau_res = float(context.get("mean_residence_time_ms", context.get("tau_res", 0.0)))
            if tau_res <= 0.0:
                tau_res = float(cached_ms.get("mean_residence_time_ms", 250.0))

            lambda_exec = float(context.get("lambda_exec", 0.0))

            # Execute Bivariate Hawkes JIT update (< 5 us SLA)
            la, lc, spoof_index, rho = self.bivariate_hawkes.update(
                symbol=symbol, dt=dt, n_a=n_a, n_c=n_c,
                lambda_exec=lambda_exec, tau_res=tau_res
            )

            # Check Hawkes Liquidity Cascade Veto (Spectral Radius rho >= 1.0)
            cached_spoof = float(cached_ms.get("hawkes_spoof_index", 0.0) or 0.0)
            cached_rho = float(cached_ms.get("hawkes_rho", 0.0) or 0.0)
            cached_h_veto = bool(cached_ms.get("hawkes_veto", False))

            effective_rho = max(rho, cached_rho)
            effective_spoof = max(spoof_index, cached_spoof)

            if effective_rho >= CASCADE_RHO_THRESHOLD:
                if self.active_veto:
                    log.warning(f"[VETO] Hawkes Liquidity Cascade Detected for {symbol} (Spectral Radius: {effective_rho:.2f} >= 1.00). Blocking trade.")
                    return False, f"Hawkes Liquidity Cascade Veto (Spectral Radius: {effective_rho:.2f})"
                else:
                    log.warning(f"[OBSERVATION MODE] Liquidity Cascade Detected ({effective_rho:.2f}), allowing execution for data harvest.")
                    return True, f"[OBSERVATION MODE] Hawkes Liquidity Cascade Veto (Spectral Radius: {effective_rho:.2f})"

            # Check Hawkes Spoofing & Layering Veto (Spoof Index > 0.72)
            if effective_spoof > SPOOF_VETO_THRESHOLD or cached_h_veto:
                if self.active_veto:
                    log.warning(f"[VETO] Hawkes Spoofing & Layering Detected for {symbol} (Spoof Index: {effective_spoof:.2f} > {SPOOF_VETO_THRESHOLD}). Blocking trade.")
                    return False, f"Hawkes Spoofing & Layering Veto (Spoof Index: {effective_spoof:.2f})"
                else:
                    log.warning(f"[OBSERVATION MODE] Spoofing Flow Detected ({effective_spoof:.2f}), allowing execution for data harvest.")
                    return True, f"[OBSERVATION MODE] Hawkes Spoofing & Layering Veto (Spoof Index: {effective_spoof:.2f})"

            # 2. Multi-Scale LOB Toxicity Update
            def run_spoof():
                return self.spoofing_detector.update(dt, cancel_counts, add_counts)

            tox, spoof_veto = await asyncio.to_thread(run_spoof)

            if spoof_veto:
                if self.active_veto:
                    log.warning(f"[VETO] Toxic Order Book Detected (Hawkes Score: {tox:.2f}). Blocking trade.")
                    return False, f"Hawkes Toxicity Veto (Score: {tox:.2f})"
                else:
                    log.warning(f"[OBSERVATION MODE] Toxic Flow Detected ({tox:.2f}), allowing execution for data harvest.")
                    return True, f"[OBSERVATION MODE] Order Book Toxicity Veto (Hawkes Score: {tox:.2f})"

            # 3. VPIN evaluation
            trade_volumes = context.get("trade_volumes", np.array([1.0], dtype=np.float64))
            trade_signs = context.get("trade_signs", np.array([1], dtype=np.float64))
            vol_24h = context.get("vol_24h", 1000.0)
            funding = context.get("funding_rate", 0.0)
            vol_series = context.get("volume_series", np.array([1.0] * STFT_NPERSEG, dtype=np.float64))
            ofi = context.get("ofi", 0.0)
            spread = context.get("spread", 1.0)

            def run_vpin():
                return self.vpin_classifier.evaluate(
                    trade_volumes, trade_signs, vol_24h, funding, vol_series, ofi, spread
                )

            vpin_action, vpin_adj = await asyncio.to_thread(run_vpin)

            # VPIN Toxicity Veto Gate (Easley-López de Prado)
            cached_vpin = float(cached_ms.get("vpin", 0.0))
            eff_vpin = max(cached_vpin, vpin_adj)
            if vpin_action == 1 or eff_vpin >= INFORMED_THRESHOLD:
                if self.active_veto:
                    log.warning(f"[VETO] High Informed Order Flow Toxicity ({symbol} VPIN: {eff_vpin:.2f} >= {INFORMED_THRESHOLD}). Blocking trade.")
                    return False, f"VPIN Toxicity Veto (VPIN: {eff_vpin:.2f})"
                else:
                    log.warning(f"[OBSERVATION MODE] High Toxicity Flow ({symbol} VPIN: {eff_vpin:.2f}), allowing execution for data harvest.")
                    return True, f"[OBSERVATION MODE] VPIN Toxicity Veto (VPIN: {eff_vpin:.2f})"

            # 4. Live Order Book Microstructure Table Inspection (oracle_microstructure)
            try:
                import sqlite3 as _sqlite3
                def _check_oracle_ms():
                    with _sqlite3.connect("trading_ledger.db", timeout=3.0) as conn:
                        return conn.execute("""
                            SELECT liquidity_imbalance, vacuum_detected, bid_wall_price, ask_wall_price, signal_multiplier
                            FROM oracle_microstructure
                            ORDER BY id DESC LIMIT 1
                        """).fetchone()
                ms_row = await asyncio.to_thread(_check_oracle_ms)
                if ms_row:
                    liq_imbalance, vac_detected, bid_w, ask_w, sig_mult = (
                        float(ms_row[0] or 0.0), int(ms_row[1] or 0),
                        float(ms_row[2] or 0.0), float(ms_row[3] or 0.0),
                        float(ms_row[4] or 1.0)
                    )
                    trade_side = str(context.get("side", "long")).lower()

                    # Veto if severe order book imbalance opposes trade direction
                    if trade_side in ("long", "buy") and liq_imbalance < -0.40:
                        if self.active_veto:
                            log.warning(f"[VETO] Severe Ask Liquidity Pressure (Imbalance: {liq_imbalance:.2f}). Blocking Long trade.")
                            return False, f"Order Book Imbalance Veto ({liq_imbalance:+.2f})"
                    elif trade_side in ("short", "sell") and liq_imbalance > 0.40:
                        if self.active_veto:
                            log.warning(f"[VETO] Severe Bid Liquidity Wall (Imbalance: {liq_imbalance:.2f}). Blocking Short trade.")
                            return False, f"Order Book Imbalance Veto ({liq_imbalance:+.2f})"

                    # Vacuum caution
                    if vac_detected == 1:
                        log.info(f"[MICROSTRUCTURE] Liquidity Vacuum Flag Active (Multiplier: {sig_mult:.2f}).")
            except Exception as ms_err:
                log.debug(f"Oracle microstructure table read note: {ms_err}")

            # 5. Live AI Order Flow Tape Whisperer & Ghost Wall Spoof Gate
            try:
                import sqlite3 as _sqlite3
                def _check_ai_tape():
                    with _sqlite3.connect("trading_data.db", timeout=3.0) as conn:
                        t_intel = conn.execute("SELECT value FROM system_state WHERE key = 'order_flow_tape_intel'").fetchone()
                        g_walls = conn.execute("SELECT value FROM system_state WHERE key = 'ai_ghost_walls'").fetchone()
                        return (t_intel[0] if t_intel else None, g_walls[0] if g_walls else None)
                tape_raw, ghost_raw = await asyncio.to_thread(_check_ai_tape)
                trade_side = str(context.get("side", "long")).lower()

                if tape_raw:
                    tape_data = json.loads(tape_raw)
                    veto_status = tape_data.get("execution_veto", "CLEAR")
                    tape_cond = tape_data.get("tape_condition", "NEUTRAL_BALANCED")
                    tape_conf = float(tape_data.get("confidence", 0.0) or 0.0)
                    headline = tape_data.get("headline", "")

                    if trade_side in ("long", "buy") and (veto_status == "BLOCK_LONGS" or (tape_cond == "BEARISH_ABSORPTION" and tape_conf >= 0.70)):
                        if self.active_veto:
                            log.warning(f"[VETO] AI Tape Whisperer VETO: Aggressive sell absorption detected! Blocking Long entry. ({headline})")
                            return False, f"Tape Whisperer Veto: {headline or 'Bearish absorption'}"

                if ghost_raw:
                    ghost_data = json.loads(ghost_raw)
                    is_spoof = bool(ghost_data.get("spoof_detected", False))
                    advisory = ghost_data.get("execution_advisory", "ALLOW")
                    g_headline = ghost_data.get("headline", "")

                    if trade_side in ("long", "buy") and is_spoof and advisory == "VETO_BUY_SPOOF":
                        if self.active_veto:
                            log.warning(f"[VETO] AI Ghost Wall Detector VETO: Deceptive bid spoofing wall detected! ({g_headline})")
                            return False, f"Ghost Wall Spoof Veto: {g_headline or 'Bid spoofing wall detected'}"
            except Exception as tape_err:
                log.debug(f"Tape whisperer check note: {tape_err}")

            return True, f"Microstructure Approved (Hawkes: {tox:.2f}, Spoof: {spoof_index:.2f}, VPIN: {vpin_adj:.2f})"

        except Exception as e:
            # Failsafe: guard errors must never crash the bot.
            # Capital Protection Invariant: When active_veto is True, fail CLOSED
            # to prevent executing into toxic order books during feed failure or corrupted state.
            if getattr(self, "active_veto", True):
                log.critical(f"MicrostructureGuard calculation failed: {e}. VETOING trade via fail-closed capital guard.", exc_info=True)
                return False, f"MicrostructureGuard Error Fail-Closed: {e}"
            else:
                log.critical(f"MicrostructureGuard calculation failed: {e}. Allowing execution via advisory-only failsafe.", exc_info=True)
                return True, f"MicrostructureGuard Error Failsafe: {e}"

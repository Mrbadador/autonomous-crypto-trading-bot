# core/entropy_engine.py
import numpy as np
from numba import njit

@njit(fastmath=True)
def approximate_entropy_fast(U, m=2, r=0.2):
    def _phi(m):
        N = len(U)
        x = np.zeros((N - m + 1, m))
        for i in range(N - m + 1):
            for j in range(m):
                x[i, j] = U[i + j]
        
        C = np.zeros(N - m + 1)
        for i in range(N - m + 1):
            match_count = 0
            for j in range(N - m + 1):
                max_dist = 0.0
                for k in range(m):
                    dist = abs(x[i, k] - x[j, k])
                    if dist > max_dist:
                        max_dist = dist
                if max_dist <= r:
                    match_count += 1
            C[i] = match_count / (N - m + 1)
        
        return np.sum(np.log(C + 1e-10)) / (N - m + 1)
    
    return np.abs(_phi(m) - _phi(m + 1))

@njit(fastmath=True)
def hurst_rs_fast(prices):
    """Calculates Hurst Exponent using Rescaled Range to determine trend persistence."""
    if len(prices) < 20: return 0.5
    returns = np.diff(prices)
    mean_ret = np.mean(returns)
    deviations = returns - mean_ret
    cum_dev = np.cumsum(deviations)
    R = np.max(cum_dev) - np.min(cum_dev)
    S = np.std(returns)
    if S == 0: return 0.5
    return np.log(R/S) / np.log(len(prices))

@njit(fastmath=True)
def higuchi_fd_fast(data, kmax=10):
    """Higuchi Fractal Dimension for measuring series roughness."""
    N = len(data)
    L = np.zeros(kmax)
    for k in range(1, kmax + 1):
        Lk = 0.0
        for m in range(k):
            indices = np.arange(1, int(np.floor((N - m) / k)))
            diff = np.sum(np.abs(data[m + indices * k] - data[m + (indices - 1) * k]))
            norm = (N - 1) / (int(np.floor((N - m) / k)) * k)
            Lk += diff * norm / k
        L[k - 1] = Lk / k
    # Slope of ln(L(k)) vs ln(1/k) approximation
    x = np.log(1.0 / np.arange(1, kmax + 1))
    y = np.log(L)
    # Using simple linear regression for slope
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sum((x - x_mean)**2)
    return num / den if den != 0 else 0.0

@njit(fastmath=True)
def shannon_entropy_flow(buy_vol, sell_vol, neutral_vol):
    """Calculates order flow entropy to detect institutional spoofing."""
    total = np.sum(buy_vol) + np.sum(sell_vol) + np.sum(neutral_vol)
    if total == 0: return 0.0
    probs = np.array([np.sum(buy_vol)/total, np.sum(sell_vol)/total, np.sum(neutral_vol)/total])
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def compute_signal_readiness(prices, buy_vol, sell_vol, neutral_vol):
    """
    Main entry point for LiveOracle. 
    Normalizes fractal dimensions and entropy into a continuous 0.0 - 1.0 readiness signal.
    """
    # Normalize price array
    p_min = np.min(prices)
    p_max = np.max(prices)
    p_norm = (prices - p_min) / (p_max - p_min + 1e-9)
    r_filter = 0.2 * np.std(p_norm)
    
    apen = approximate_entropy_fast(p_norm, m=2, r=r_filter)
    hurst = hurst_rs_fast(prices)
    hfd = higuchi_fd_fast(p_norm)
    shannon = shannon_entropy_flow(buy_vol, sell_vol, neutral_vol)
    
    # Readiness calculation: High Hurst (Trending) + Low ApEn (Predictable) = High Readiness
    readiness = 0.50
    if hurst > 0.65 and apen < 0.3:
        readiness = 0.85
    elif hurst < 0.4 and apen > 0.5:
        readiness = 0.20
    else:
        readiness = 0.50 + ((hurst - 0.5) * 0.4) - (apen * 0.1)
        
    return {
        "readiness": np.clip(readiness, 0.0, 1.0),
        "apen": apen,
        "hurst": hurst,
        "hfd": hfd,
        "shannon": shannon
    }

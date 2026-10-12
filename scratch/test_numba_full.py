import numpy as np
from numba import jit
from typing import Dict

@jit(nopython=True, cache=False, fastmath=True)
def approximate_entropy_fast(
    x: np.ndarray,
    m: int = 2,
    r: float = 0.2
) -> float:
    N = len(x)
    if N < m + 1:
        return 1.0
    
    sigma = np.std(x)
    if sigma < 1e-10:
        return 0.0
    
    r_scaled = r * sigma
    
    def phi(m_emb: int) -> float:
        N_m = N - m_emb + 1
        if N_m <= 0:
            return 0.0
        
        counts = 0.0
        for i in range(N_m):
            for j in range(N_m):
                d_max = 0.0
                for k in range(m_emb):
                    diff = abs(x[i + k] - x[j + k])
                    if diff > d_max:
                        d_max = diff
                
                if d_max <= r_scaled:
                    counts += 1.0
        
        C = counts / N_m
        return np.log(C + 1e-10)
    
    phi_m = phi(m)
    phi_m1 = phi(m + 1)
    
    return phi_m - phi_m1


@jit(nopython=True, cache=False, fastmath=True)
def hurst_rs_fast(x: np.ndarray) -> float:
    N = len(x)
    if N < 20:
        return 0.5
    
    lags = np.array([8, 16, 32, 64])
    lags = lags[lags < N // 4]
    
    if len(lags) == 0:
        return 0.5
    
    rs_values = np.zeros(len(lags))
    
    for idx in range(len(lags)):
        n = lags[idx]
        num_segments = N // n
        
        rs_segment = np.zeros(num_segments)
        
        for seg in range(num_segments):
            start = seg * n
            end = start + n
            segment = x[start:end]
            
            mean_seg = np.mean(segment)
            
            y = np.zeros(n)
            cumsum = 0.0
            for i in range(n):
                cumsum += segment[i] - mean_seg
                y[i] = cumsum
            
            R = np.max(y) - np.min(y)
            
            s = np.std(segment)
            if s < 1e-10:
                rs_segment[seg] = 0.0
            else:
                rs_segment[seg] = R / s
        
        rs_values[idx] = np.mean(rs_segment[rs_segment > 0])
    
    log_lags = np.log(lags.astype(np.float64))
    log_rs = np.log(rs_values + 1e-10)
    
    n_points = len(log_lags)
    x_mean = np.mean(log_lags)
    y_mean = np.mean(log_rs)
    
    num = 0.0
    den = 0.0
    for i in range(n_points):
        num += (log_lags[i] - x_mean) * (log_rs[i] - y_mean)
        den += (log_lags[i] - x_mean) ** 2
    
    if den < 1e-10:
        return 0.5
    
    H = num / den
    return np.clip(H, 0.0, 1.0)


@jit(nopython=True, cache=False, fastmath=True)
def dfa_fast(x: np.ndarray, max_box: int = 64) -> float:
    N = len(x)
    if N < max_box * 2:
        return 0.5
    
    mean_x = np.mean(x)
    y = np.zeros(N)
    cumsum = 0.0
    for i in range(N):
        cumsum += x[i] - mean_x
        y[i] = cumsum
    
    box_sizes = np.array([4, 8, 16, 32, 64])
    box_sizes = box_sizes[box_sizes <= N // 4]
    
    if len(box_sizes) == 0:
        return 0.5
    
    fluctuations = np.zeros(len(box_sizes))
    
    for idx in range(len(box_sizes)):
        n = box_sizes[idx]
        num_boxes = N // n
        
        f_n = 0.0
        for box in range(num_boxes):
            start = box * n
            end = start + n
            segment = y[start:end]
            
            x_vals = np.arange(n, dtype=np.float64)
            x_mean = np.mean(x_vals)
            y_mean = np.mean(segment)
            
            num = 0.0
            den = 0.0
            for i in range(n):
                num += (x_vals[i] - x_mean) * (segment[i] - y_mean)
                den += (x_vals[i] - x_mean) ** 2
            
            if den < 1e-10:
                slope = 0.0
                intercept = y_mean
            else:
                slope = num / den
                intercept = y_mean - slope * x_mean
            
            for i in range(n):
                trend = slope * x_vals[i] + intercept
                diff = segment[i] - trend
                f_n += diff * diff
        
        fluctuations[idx] = np.sqrt(f_n / (num_boxes * n))
    
    log_n = np.log(box_sizes.astype(np.float64))
    log_f = np.log(fluctuations + 1e-10)
    
    n_points = len(log_n)
    x_mean = np.mean(log_n)
    y_mean = np.mean(log_f)
    
    num = 0.0
    den = 0.0
    for i in range(n_points):
        num += (log_n[i] - x_mean) * (log_f[i] - y_mean)
        den += (log_n[i] - x_mean) ** 2
    
    if den < 1e-10:
        return 0.5
    
    alpha = num / den
    return np.clip(alpha, 0.0, 1.0)


@jit(nopython=True, cache=False, fastmath=True)
def higuchi_fd_fast(x: np.ndarray, k_max: int = 8) -> float:
    N = len(x)
    if N < k_max * 2:
        return 1.5
    
    k_values = np.arange(1, k_max + 1)
    L_k = np.zeros(k_max)
    
    for idx in range(k_max):
        k = k_values[idx]
        
        L_m_sum = 0.0
        for m in range(1, k + 1):
            n_max = int(np.floor((N - m) / k))
            if n_max < 1:
                continue
            
            L_m = 0.0
            for i in range(1, n_max):
                idx_curr = m + i * k - 1
                idx_prev = m + (i - 1) * k - 1
                L_m += abs(x[idx_curr] - x[idx_prev])
            
            if n_max > 0:
                norm = (N - 1) / (n_max * k)
                L_m_sum += (L_m * norm) / k
        
        L_k[idx] = L_m_sum / k if k > 0 else 0.0
    
    log_k = np.log(k_values.astype(np.float64))
    log_L = np.log(L_k + 1e-10)
    
    n_points = len(log_k)
    x_mean = np.mean(log_k)
    y_mean = np.mean(log_L)
    
    num = 0.0
    den = 0.0
    for i in range(n_points):
        num += (log_k[i] - x_mean) * (log_L[i] - y_mean)
        den += (log_k[i] - x_mean) ** 2
    
    if den < 1e-10:
        return 1.5
    
    slope = num / den
    D = -slope
    
    return np.clip(D, 1.0, 2.0)


@jit(nopython=True, cache=False, fastmath=True)
def kl_divergence_fast(
    p: np.ndarray,
    q: np.ndarray,
    bins: int = 20
) -> float:
    min_val = min(np.min(p), np.min(q))
    max_val = max(np.max(p), np.max(q))
    
    if max_val - min_val < 1e-10:
        return 0.0
    
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    
    p_hist = np.zeros(bins)
    q_hist = np.zeros(bins)
    
    for i in range(len(p)):
        bin_idx = int((p[i] - min_val) / (max_val - min_val) * bins)
        bin_idx = min(bin_idx, bins - 1)
        p_hist[bin_idx] += 1.0
    
    for i in range(len(q)):
        bin_idx = int((q[i] - min_val) / (max_val - min_val) * bins)
        bin_idx = min(bin_idx, bins - 1)
        q_hist[bin_idx] += 1.0
    
    p_hist = p_hist / (np.sum(p_hist) + 1e-10)
    q_hist = q_hist / (np.sum(q_hist) + 1e-10)
    
    kl = 0.0
    for i in range(bins):
        if p_hist[i] > 1e-10 and q_hist[i] > 1e-10:
            kl += p_hist[i] * np.log(p_hist[i] / q_hist[i])
    
    return kl


@jit(nopython=True, cache=False, fastmath=True)
def shannon_entropy_flow(
    buy_volume: float,
    sell_volume: float,
    neutral_volume: float
) -> float:
    total = buy_volume + sell_volume + neutral_volume
    if total < 1e-10:
        return 1.585
    
    p_buy = buy_volume / total
    p_sell = sell_volume / total
    p_neutral = neutral_volume / total
    
    entropy = 0.0
    for p in [p_buy, p_sell, p_neutral]:
        if p > 1e-10:
            entropy -= p * np.log2(p)
    
    return entropy


@jit(nopython=True, cache=False, fastmath=True)
def compute_signal_readiness(
    prices: np.ndarray,
    buy_vol: float,
    sell_vol: float,
    neutral_vol: float,
    baseline_prices: np.ndarray
) -> Dict[str, float]:
    N = len(prices)
    returns = np.zeros(N - 1)
    for i in range(N - 1):
        returns[i] = np.log(prices[i + 1] / prices[i]) if prices[i] > 1e-10 else 0.0
    
    apen = approximate_entropy_fast(returns, m=2, r=0.2)
    apen_norm = np.clip(apen / 1.5, 0.0, 1.0)
    
    hurst = hurst_rs_fast(returns)
    hurst_norm = 1.0 - abs(hurst - 0.5) * 2.0
    hurst_norm = np.clip(hurst_norm, 0.0, 1.0)
    
    dfa_alpha = dfa_fast(returns)
    dfa_norm = np.clip(dfa_alpha, 0.0, 1.0)
    
    fd = higuchi_fd_fast(prices)
    fd_norm = np.clip((2.0 - fd), 0.0, 1.0)
    
    kl_div = kl_divergence_fast(prices, baseline_prices)
    kl_norm = np.exp(-kl_div)
    
    shannon = shannon_entropy_flow(buy_vol, sell_vol, neutral_vol)
    shannon_norm = shannon / 1.585
    
    w1, w2, w3, w4 = 0.35, 0.25, 0.25, 0.15
    
    s_ready = (
        w1 * (1.0 - apen_norm) +
        w2 * hurst_norm +
        w3 * dfa_norm +
        w4 * kl_norm
    )
    
    return {
        'signal_readiness': np.clip(s_ready, 0.0, 1.0),
        'apen': apen,
        'hurst': hurst,
        'dfa_alpha': dfa_alpha,
        'fractal_dim': fd,
        'kl_divergence': kl_div,
        'shannon_entropy': shannon,
        'regime': 0 if s_ready > 0.7 else (2 if s_ready < 0.3 else 1)
    }

# Test data
p = np.random.rand(100) * 100 + 10000
b_p = p.copy()
try:
    res = compute_signal_readiness(p, 100.0, 50.0, 10.0, b_p)
    print("FULL COMPILATION SUCCESSFUL!")
    print(res)
except Exception as e:
    print("FULL COMPILATION FAILED!")
    import traceback
    traceback.print_exc()

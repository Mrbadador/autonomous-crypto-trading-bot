import pandas as pd
import numpy as np
import ta
import config


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or len(df) < 50:
        raise ValueError(f"Cannot compute indicators: insufficient history ({len(df) if df is not None else 0} bars, minimum 50 required)")
    
    df = df.copy()
    # Trend Cluster (calculated with adaptive min_periods for mathematical integrity)
    df["ema20"] = ta.trend.ema_indicator(df["close"], window=20, fillna=True)
    df["ema50"] = ta.trend.ema_indicator(df["close"], window=50, fillna=True)
    df["ema200"] = df["close"].ewm(span=200, adjust=False, min_periods=min(len(df), 50)).mean()
    
    # Trend Strength (ADX)
    adx_ind = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["adx"] = adx_ind.adx()
    df["adx_delta"] = df["adx"].diff(3) 
    df["adx_avg"] = df["adx"].rolling(config.ADX_WINDOW).mean() # Dynamic benchmark
    
    # Momentum & Volatility
    df["rsi14"] = ta.momentum.rsi(df["close"], window=14)
    df["rsi_avg"] = df["rsi14"].rolling(config.ADX_WINDOW).mean() # Dynamic benchmark
    df["atr14"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df["atr_diff"] = df["atr14"].diff(3)
    
    # V26.3 Sovereign Upgrade: Bollinger Bands & Mean Reversion Primitives
    df["sma20"] = df["close"].rolling(20).mean()
    df["std20"] = df["close"].rolling(20).std()
    df["bb_lower"] = df["sma20"] - (2.0 * df["std20"])
    df["bb_upper"] = df["sma20"] + (2.0 * df["std20"])
    
    # Market Structure (Swings)
    df["swing_high"] = df["high"].rolling(20).max().shift(1)
    df["swing_low"] = df["low"].rolling(20).min().shift(1)
    
    # Relative Volatility (ATR Percentile for UI)
    wl = min(len(df), 200)
    df["atr_pct"] = df["atr14"].rolling(wl).rank(pct=True) if wl > 1 else 0.5
    
    # Absolute Volatility (ATR as % of Price)
    df["atr_pct_price"] = (df["atr14"] / df["close"]) * 100
    
    # Structual Slopes
    ema200_shift = df["ema200"].shift(10).replace(0, 1e-9)
    ema_slope_raw = df["ema200"].diff(10) / ema200_shift * 100
    df["ema_slope"] = ema_slope_raw.fillna(0)
    df["trend_str"] = (df["ema_slope"].abs() / 0.5 * 100).clip(lower=0, upper=100)
    
    # EMA Divergence
    df["ema_div"] = (abs(df["ema20"] - df["ema50"]) / df["close"]) * 100

    # Volume Cluster
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    
    # V17.0 VWAP Matrix Tracking
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["vwap"] = (df["typical_price"] * df["volume"]).rolling(288, min_periods=1).sum() / df["volume"].rolling(288, min_periods=1).sum().replace(0, 1)
    
    return df

def check_ema_stack(price, ema20, ema50, ema200, side):
    """Deterministic validation of the full structural stack."""
    if side == "long":
        return (price > ema20 > ema50 > ema200)
    elif side == "short":
        return (price < ema20 < ema50 < ema200)
    return False

def classify_regime_hard(row):
    """Deterministic logic to classify regime without AI hallucinations."""
    adx = row["adx"]
    slope = row["ema_slope"]
    atr_p = row["atr_pct_price"]
    
    # EMA Stack Alignment (Strong trending indicator)
    ema20, ema50, ema200 = row["ema20"], row["ema50"], row["ema200"]
    is_stacked = (ema20 > ema50 > ema200) or (ema20 < ema50 < ema200)

    if adx > config.ADX_STRONG_THRESHOLD or abs(slope) > config.SLOPE_TREND_THRESHOLD or is_stacked:
        return "TRENDING"
    
    adx_bench = row.get("adx_avg", 25)
    if pd.isna(adx_bench): adx_bench = 25
    
    if adx < (adx_bench * config.ADX_RANGING_MULT) and abs(slope) < config.SLOPE_RANGING_THRESHOLD:
        return "RANGING"

    if adx < config.ADX_TREND_THRESHOLD and (atr_p > config.ATR_CHAOTIC_THRESHOLD or abs(slope) > config.SLOPE_CHAOTIC_THRESHOLD):
        return "CHAOTIC"

    if adx > config.ADX_TREND_THRESHOLD: return "TRENDING"
    return "RANGING"

# ANSI color codes inlined to avoid circular import with ui_render
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_BLUE   = "\033[94m"
_GREEN  = "\033[92m"
_WHITE  = "\033[97m"

def get_volatility_regime(atr_pct: float) -> tuple:
    if atr_pct >= 0.90: return ("Extreme", _RED, "!!!")
    if atr_pct >= 0.75: return ("High", _YELLOW, "!!")
    if atr_pct <= 0.15: return ("Compression", _BLUE, ">>")
    if atr_pct <= 0.35: return ("Low", _GREEN, "<<")
    return ("Normal", _WHITE, "--")

def is_market_ready(df: pd.DataFrame) -> tuple[bool, str]:
    if len(df) < 10:
        return False, "Insufficient data"
    
    recent = df.tail(10)
    avg_adx = recent["adx"].mean()
    
    trending_candles = (
        (recent["adx"] > config.ADX_TREND_THRESHOLD) | 
        (recent["ema_slope"].abs() > config.SLOPE_RANGING_THRESHOLD)
    ).sum()
    
    if trending_candles < config.MIN_TRENDING_CANDLES:
        return False, f"Market in chop (only {trending_candles}/10 trending candles)"
    
    if avg_adx < config.MIN_ADX_READY:
        return False, f"ADX too weak (avg={avg_adx:.1f})"
        
    return True, f"Market ready (ADX={avg_adx:.1f}, trending={trending_candles}/10)"

def get_candle_patterns(df: pd.DataFrame) -> list[str]:
    if len(df) < 5: return []
    patterns = []
    c1 = df.iloc[-1]; c2 = df.iloc[-2]; c3 = df.iloc[-3]; c4 = df.iloc[-4]; c5 = df.iloc[-5]

    def is_doji(c):
        body = abs(c['close'] - c['open']); total = c['high'] - c['low']
        return total > 0 and body / total < 0.1

    def is_spinning_top(c):
        body = abs(c['close'] - c['open']); total = c['high'] - c['low']
        return total > 0 and 0.1 <= body / total <= 0.3

    if is_doji(c1): patterns.append("Doji")
    elif is_spinning_top(c1): patterns.append("Spinning Top")

    body = abs(c1['close'] - c1['open'])
    lower_wick = min(c1['open'], c1['close']) - c1['low']
    upper_wick = c1['high'] - max(c1['open'], c1['close'])
    if lower_wick >= 2 * body and upper_wick < 0.2 * body:
        # FIX: Exclude the current candle (-1) from the recent mean calculation
        recent_mean = df['close'].iloc[-5:-1].mean()
        if c1['close'] < recent_mean: patterns.append("Hammer")
        else: patterns.append("Hanging Man")

    if c1['close'] > c1['open'] and c2['open'] > c2['close']:
        if c1['close'] > c2['open'] and c1['open'] < c2['close']: patterns.append("Bullish Engulfing")
    elif c1['open'] > c1['close'] and c2['close'] > c2['open']:
        if c1['open'] > c2['close'] and c1['close'] < c2['open']: patterns.append("Bearish Engulfing")

    if (c3['open'] > c3['close']) and (abs(c2['open']-c2['close']) < abs(c3['open']-c3['close'])*0.3) and (c1['close'] > c1['open']):
        if c1['close'] > (c3['open'] + c3['close'])/2: patterns.append("Morning Star")
    elif (c3['close'] > c3['open']) and (abs(c2['open']-c2['close']) < abs(c3['close']-c3['open'])*0.3) and (c1['open'] > c1['close']):
        if c1['close'] < (c3['open'] + c3['close'])/2: patterns.append("Evening Star")

    if all(df['close'].tail(3) > df['open'].tail(3)) and c1['close'] > c2['close'] > c3['close']:
        patterns.append("Three White Soldiers")
    elif all(df['open'].tail(3) > df['close'].tail(3)) and c1['close'] < c2['close'] < c3['close']:
        patterns.append("Three Black Crows")

    if (c5['close'] > c5['open']) and all(df['open'].iloc[-4:-1] > df['close'].iloc[-4:-1]) and (c1['close'] > c1['open']):
        if c1['close'] > c5['close']: patterns.append("Rising Three")
    elif (c5['open'] > c5['close']) and all(df['close'].iloc[-4:-1] > df['open'].iloc[-4:-1]) and (c1['open'] > c1['close']):
        if c1['close'] < c5['close']: patterns.append("Falling Three")

    return patterns


# ---------------------------------------------------------------------------
# Quantitative Alpha Gates (Inspired by go-stock Architecture)
# ---------------------------------------------------------------------------

def detect_momentum_divergence(df: pd.DataFrame, lookback: int = 24) -> dict:
    """
    Automated Momentum Divergence Engine (背离检测).
    Identifies regular and hidden divergences between price peaks/troughs and
    momentum indicators (RSI and MACD Histogram) over a rolling lookback window.
    
    Returns:
        dict: {
            "bearish_divergence": bool,
            "bullish_divergence": bool,
            "rsi_bearish_div": bool,
            "macd_bearish_div": bool,
            "rsi_bullish_div": bool,
            "macd_bullish_div": bool,
            "details": str
        }
    """
    if df is None or len(df) < lookback:
        return {
            "bearish_divergence": False, "bullish_divergence": False,
            "rsi_bearish_div": False, "macd_bearish_div": False,
            "rsi_bullish_div": False, "macd_bullish_div": False,
            "details": "insufficient_data"
        }

    df_sub = df.tail(lookback).copy().reset_index(drop=True)
    c = df_sub["close"]
    h = df_sub["high"]
    l = df_sub["low"]

    # 1. RSI(14) calculation
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=5).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_s = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi_s.bfill().fillna(50.0)

    # 2. MACD(12, 26, 9) calculation
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line

    # 3. Swing Peak & Trough Detection
    peaks = []   # (index, high, rsi, macd_hist)
    troughs = [] # (index, low, rsi, macd_hist)
    for i in range(1, len(df_sub) - 1):
        if h[i] > h[i-1] and h[i] >= h[i+1]:
            peaks.append((i, float(h[i]), float(rsi.iloc[i]), float(macd_hist.iloc[i])))
        if l[i] < l[i-1] and l[i] <= l[i+1]:
            troughs.append((i, float(l[i]), float(rsi.iloc[i]), float(macd_hist.iloc[i])))

    bearish_div = False
    bullish_div = False
    rsi_bearish = False
    macd_bearish = False
    rsi_bullish = False
    macd_bullish = False
    details = []

    # 4. Bearish Divergence: Price Higher High, but RSI or MACD Lower High
    if len(peaks) >= 2:
        p1, p2 = peaks[-2], peaks[-1]
        if p2[1] > p1[1]:  # Price made Higher High
            if p2[2] < (p1[2] - 1.0):  # RSI made Lower High
                rsi_bearish = True
                bearish_div = True
                details.append(f"RSI Bearish Div (Price {p1[1]:.4f}->{p2[1]:.4f}, RSI {p1[2]:.1f}->{p2[2]:.1f})")
            if p2[3] < p1[3] and p2[3] < 0:  # MACD histogram deteriorating below zero
                macd_bearish = True
                bearish_div = True
                details.append(f"MACD Bearish Div (Hist {p1[3]:.4f}->{p2[3]:.4f})")

    # 5. Bullish Divergence: Price Lower Low, but RSI or MACD Higher Low
    if len(troughs) >= 2:
        t1, t2 = troughs[-2], troughs[-1]
        if t2[1] < t1[1]:  # Price made Lower Low
            if t2[2] > (t1[2] + 1.0):  # RSI made Higher Low
                rsi_bullish = True
                bullish_div = True
                details.append(f"RSI Bullish Div (Price {t1[1]:.4f}->{t2[1]:.4f}, RSI {t1[2]:.1f}->{t2[2]:.1f})")
            if t2[3] > t1[3]:  # MACD histogram rising
                macd_bullish = True
                bullish_div = True
                details.append(f"MACD Bullish Div (Hist {t1[3]:.4f}->{t2[3]:.4f})")

    return {
        "bearish_divergence": bearish_div,
        "bullish_divergence": bullish_div,
        "rsi_bearish_div": rsi_bearish,
        "macd_bearish_div": macd_bearish,
        "rsi_bullish_div": rsi_bullish,
        "macd_bullish_div": macd_bullish,
        "details": "; ".join(details) if details else "No divergence"
    }


def compute_volume_profile(
    df: pd.DataFrame,
    lookback: int = 48,
    n_bins: int = 30,
    value_area_pct: float = 0.70
) -> dict:
    """
    Volume Profile & Value Area Engine (VPVR: POC, VAH, VAL).
    Computes volume distribution across price levels over rolling lookback candles.
    Identifies the Point of Control (POC), Value Area High (VAH), and Value Area Low (VAL).
    
    Returns:
        dict: {
            "poc": float,
            "vah": float,
            "val": float,
            "close": float,
            "in_value_area": bool,
            "above_vah": bool,
            "below_val": bool,
            "distance_to_vah_pct": float,
            "distance_to_val_pct": float,
            "details": str
        }
    """
    if df is None or len(df) < 10:
        return {
            "poc": 0.0, "vah": 0.0, "val": 0.0, "close": 0.0,
            "in_value_area": False, "above_vah": False, "below_val": False,
            "distance_to_vah_pct": 0.0, "distance_to_val_pct": 0.0,
            "details": "insufficient_data"
        }

    df_slice = df.tail(lookback).copy()
    min_price = float(df_slice["low"].min())
    max_price = float(df_slice["high"].max())
    curr_close = float(df_slice["close"].iloc[-1])

    if max_price <= min_price:
        return {
            "poc": curr_close, "vah": curr_close, "val": curr_close, "close": curr_close,
            "in_value_area": True, "above_vah": False, "below_val": False,
            "distance_to_vah_pct": 0.0, "distance_to_val_pct": 0.0,
            "details": "flat_range"
        }

    bins = np.linspace(min_price, max_price, n_bins + 1)
    bin_volumes = np.zeros(n_bins)

    for _, row in df_slice.iterrows():
        bar_low = float(row["low"])
        bar_high = float(row["high"])
        bar_vol = float(row["volume"])
        idx_low = max(0, min(n_bins - 1, int(np.searchsorted(bins, bar_low) - 1)))
        idx_high = max(0, min(n_bins - 1, int(np.searchsorted(bins, bar_high) - 1)))
        span = max(1, idx_high - idx_low + 1)
        vol_per_bin = bar_vol / span
        for b in range(idx_low, idx_high + 1):
            bin_volumes[b] += vol_per_bin

    total_vol = bin_volumes.sum()
    if total_vol <= 0:
        return {
            "poc": curr_close, "vah": curr_close, "val": curr_close, "close": curr_close,
            "in_value_area": True, "above_vah": False, "below_val": False,
            "distance_to_vah_pct": 0.0, "distance_to_val_pct": 0.0,
            "details": "zero_vol"
        }

    poc_idx = int(np.argmax(bin_volumes))
    poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2.0

    # Expand outward from POC until target volume (e.g. 70%) is accumulated
    va_target = total_vol * value_area_pct
    current_va_vol = bin_volumes[poc_idx]
    up_idx = poc_idx
    down_idx = poc_idx

    while current_va_vol < va_target and (up_idx < n_bins - 1 or down_idx > 0):
        up_next = bin_volumes[up_idx + 1] if up_idx < n_bins - 1 else -1.0
        down_next = bin_volumes[down_idx - 1] if down_idx > 0 else -1.0

        if up_next >= down_next and up_idx < n_bins - 1:
            up_idx += 1
            current_va_vol += bin_volumes[up_idx]
        elif down_idx > 0:
            down_idx -= 1
            current_va_vol += bin_volumes[down_idx]
        elif up_idx < n_bins - 1:
            up_idx += 1
            current_va_vol += bin_volumes[up_idx]
        else:
            break

    vah = float(bins[up_idx + 1])
    val = float(bins[down_idx])

    return {
        "poc": round(poc_price, 4),
        "vah": round(vah, 4),
        "val": round(val, 4),
        "close": round(curr_close, 4),
        "in_value_area": (val <= curr_close <= vah),
        "above_vah": (curr_close > vah),
        "below_val": (curr_close < val),
        "distance_to_vah_pct": round((curr_close - vah) / vah * 100, 2),
        "distance_to_val_pct": round((curr_close - val) / val * 100, 2),
        "details": f"POC: ${poc_price:.4f} | VAH: ${vah:.4f} | VAL: ${val:.4f} (Close: ${curr_close:.4f})"
    }


def compute_td_sequential(df: pd.DataFrame) -> dict:
    """
    Tom DeMark TD Sequential Exhaustion Gate (神奇九转 / TD-9).
    Counts consecutive closes relative to the close 4 bars prior.
    Triggers an exhaustion climax flag on setup count 9 (TD-9),
    identifying statistical turning points where momentum exhaustion pullbacks occur.
    
    Returns:
        dict: {
            "setup_direction": "UP" | "DOWN" | "NONE",
            "count": int,
            "up_count": int,
            "down_count": int,
            "is_td9_upward_exhaustion": bool,
            "is_td9_downward_exhaustion": bool,
            "is_td_exhaustion_risk": bool,
            "details": str
        }
    """
    if df is None or len(df) < 5:
        return {
            "setup_direction": "NONE", "count": 0, "up_count": 0, "down_count": 0,
            "is_td9_upward_exhaustion": False, "is_td9_downward_exhaustion": False,
            "is_td_exhaustion_risk": False, "details": "insufficient_data"
        }

    closes = df["close"].values
    up_count = 0
    down_count = 0

    for i in range(4, len(closes)):
        if closes[i] > closes[i - 4]:
            up_count += 1
            down_count = 0
        elif closes[i] < closes[i - 4]:
            down_count += 1
            up_count = 0
        else:
            up_count = 0
            down_count = 0

    is_td9_up = (up_count >= 9)
    is_td9_down = (down_count >= 9)
    direction = "UP" if up_count > 0 else ("DOWN" if down_count > 0 else "NONE")
    count = up_count if up_count > 0 else -down_count

    return {
        "setup_direction": direction,
        "count": count,
        "up_count": up_count,
        "down_count": down_count,
        "is_td9_upward_exhaustion": is_td9_up,
        "is_td9_downward_exhaustion": is_td9_down,
        "is_td_exhaustion_risk": (up_count in (8, 9) or down_count in (8, 9)),
        "details": f"TD Sequential: {direction} setup count = {abs(count)} (TD-9 Up: {is_td9_up}, TD-9 Down: {is_td9_down})"
    }


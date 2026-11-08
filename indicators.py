"""
indicators.py — Causal Quantitative Indicator Pipeline
Deterministic, causal, and mathematically sound technical indicators.
Ensures zero forward-looking bias, strict validation, robust warm-up handling,
causal TD-9 setups, and confirmed fractal momentum divergence.
"""

import numpy as np
import pandas as pd
import ta
import config

# ---------------------------------------------------------------------------
# Warm-up policy
# ---------------------------------------------------------------------------
# EMA(span=200, adjust=False) keeps weight (1 - 2/201)^n on the very first
# close after n bars: ~61% @50, ~37% @100, ~13% @200, ~1.8% @400, ~0.25% @600.
# Indicators are trusted once `warm` is True (>= WARMUP_BARS).
WARMUP_BARS = 600


def validate_ohlcv(df: pd.DataFrame) -> None:
    """Fail loudly on data that would silently corrupt every downstream indicator."""
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    if df[required].isna().any().any():
        raise ValueError("NaN in OHLCV input; refusing to compute indicators")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Non-positive price in OHLCV input")
    if (df["volume"] < 0).any():
        raise ValueError("Negative volume in OHLCV input")
    if (df["high"] < df[["open", "close", "low"]].max(axis=1)).any() or \
       (df["low"] > df[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Inconsistent OHLC (high < max(o,c,l) or low > min(o,c,h))")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("Index must be strictly ordered with no duplicates")


def td_setup_counts(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    TD Sequential *Setup* counts, causal (value at bar t uses closes <= t only).

    Sell setup (up count): close[t] > close[t-4] for up to 9 consecutive bars.
    Buy setup (down count): close[t] < close[t-4] for up to 9 consecutive bars.
    A setup can only START on a price flip (previous bar had the opposite
    relation to its own t-4 close). Counts run 1..9 and the setup then
    completes: further bars of the same relation output 0 until the relation
    breaks and a new flip occurs. A run already in progress at the start of
    the data (no observed flip) is ignored rather than counted from a
    truncated start, so results do not depend on where the window begins.

    Returns (up_count, down_count) as int8 Series aligned to `close`.
    """
    c = close.to_numpy(dtype=float)
    n = len(c)
    up_out = np.zeros(n, dtype=np.int8)
    dn_out = np.zeros(n, dtype=np.int8)
    up_run = 0   # >0 active count, 0 idle, -1 consumed/unflipped until relation breaks
    dn_run = 0
    for i in range(5, n):
        d_now = c[i] - c[i - 4]
        d_prev = c[i - 1] - c[i - 5]

        if d_now > 0:
            if up_run == 0:
                up_run = 1 if d_prev < 0 else -1
            elif up_run > 0:
                up_run += 1
            if up_run > 0:
                up_out[i] = up_run
                if up_run == 9:
                    up_run = -1
        else:
            up_run = 0

        if d_now < 0:
            if dn_run == 0:
                dn_run = 1 if d_prev > 0 else -1
            elif dn_run > 0:
                dn_run += 1
            if dn_run > 0:
                dn_out[i] = dn_run
                if dn_run == 9:
                    dn_run = -1
        else:
            dn_run = 0

    return pd.Series(up_out, index=close.index), pd.Series(dn_out, index=close.index)


def compute_indicators(
    df: pd.DataFrame,
    warmup_bars: int = WARMUP_BARS,
    strict: bool = False,
    vwap_window: int = 288,
) -> pd.DataFrame:
    """
    Causal indicator pipeline: row t depends only on rows <= t.

    - No fillna/bfill anywhere: anything not yet computable stays NaN.
    - `warm` is True once `warmup_bars` bars have been seen.
    - strict=False by default to allow graceful handling on live feeds while
      setting warm=False until warmup_bars is satisfied.
    - `vwap_window` is in BARS, not time.
    """
    if df is None or df.empty:
        raise ValueError("Cannot compute indicators: empty input")
    if strict and len(df) < warmup_bars:
        raise ValueError(
            f"Cannot compute indicators: {len(df)} bars < warm-up requirement of {warmup_bars}"
        )
    validate_ohlcv(df)

    df = df.copy()
    close, high, low = df["close"], df["high"], df["low"]

    # Trend cluster: NaN until each EMA has sufficient observations
    df["ema20"] = ta.trend.ema_indicator(close, window=20, fillna=False)
    df["ema50"] = ta.trend.ema_indicator(close, window=50, fillna=False)
    df["ema200"] = close.ewm(span=200, adjust=False, min_periods=min(len(df), 200)).mean()

    # Trend strength
    df["adx"] = ta.trend.ADXIndicator(high, low, close, window=14, fillna=False).adx()
    df["adx_delta"] = df["adx"].diff(3)
    adx_win = getattr(config, "ADX_WINDOW", 14)
    df["adx_avg"] = df["adx"].rolling(adx_win).mean()

    # Momentum & volatility
    df["rsi14"] = ta.momentum.rsi(close, window=14, fillna=False)
    df["rsi_avg"] = df["rsi14"].rolling(adx_win).mean()
    df["atr14"] = ta.volatility.average_true_range(high, low, close, window=14, fillna=False)
    df["atr_diff"] = df["atr14"].diff(3)

    # MACD on the FULL series (used by divergence logic; never recomputed on a slice)
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    df["macd_line"] = ema12 - ema26
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False, min_periods=9).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    # Bollinger primitives (ddof=0 standard convention)
    df["sma20"] = close.rolling(20).mean()
    df["std20"] = close.rolling(20).std(ddof=0)
    df["bb_lower"] = df["sma20"] - 2.0 * df["std20"]
    df["bb_upper"] = df["sma20"] + 2.0 * df["std20"]

    # Market structure (shifted: excludes current bar)
    df["swing_high"] = high.rolling(20).max().shift(1)
    df["swing_low"] = low.rolling(20).min().shift(1)

    # Volatility
    wl = min(len(df), 200)
    df["atr_pct"] = df["atr14"].rolling(wl, min_periods=min(wl, 50)).rank(pct=True)
    df["atr_pct_price"] = df["atr14"] / close * 100

    # Structural slope
    df["ema_slope"] = (df["ema200"] / df["ema200"].shift(10) - 1.0) * 100
    df["trend_str"] = (df["ema_slope"].abs() / 0.5 * 100).clip(lower=0, upper=100)
    df["ema_div"] = (df["ema20"] - df["ema50"]).abs() / close * 100

    # Volume
    df["vol_sma20"] = df["volume"].rolling(20).mean()

    # Rolling VWAP: NaN when window has no volume (never a fake 0)
    df["typical_price"] = (high + low + close) / 3.0
    pv = (df["typical_price"] * df["volume"]).rolling(vwap_window, min_periods=1).sum()
    vv = df["volume"].rolling(vwap_window, min_periods=1).sum()
    df["vwap"] = pv / vv.where(vv > 0)

    # TD Sequential setup counts (causal, per bar)
    df["td_sell_setup"], df["td_buy_setup"] = td_setup_counts(close)

    # Warm-up flag
    df["warm"] = (np.arange(1, len(df) + 1) >= warmup_bars)

    return df


def check_ema_stack(price, ema20, ema50, ema200, side):
    """Deterministic validation of the full structural stack."""
    if side == "long":
        return bool(price > ema20 > ema50 > ema200)
    elif side == "short":
        return bool(price < ema20 < ema50 < ema200)
    return False


def classify_regime_hard(row):
    """
    Deterministic regime classification. Returns "UNKNOWN" (never a tradable
    default) when inputs are NaN or warm-up is incomplete.
    """
    needed = ("adx", "ema_slope", "atr_pct_price", "ema20", "ema50", "ema200")
    if any(pd.isna(row.get(k, np.nan)) for k in needed):
        return "UNKNOWN"
    if "warm" in row and not bool(row["warm"]):
        return "UNKNOWN"

    adx = row["adx"]
    slope = row["ema_slope"]
    atr_p = row["atr_pct_price"]

    ema20, ema50, ema200 = row["ema20"], row["ema50"], row["ema200"]
    is_stacked = (ema20 > ema50 > ema200) or (ema20 < ema50 < ema200)

    strong_adx = getattr(config, "ADX_STRONG_THRESHOLD", 30)
    slope_trend = getattr(config, "SLOPE_TREND_THRESHOLD", 0.5)
    if adx > strong_adx or abs(slope) > slope_trend or is_stacked:
        return "TRENDING"

    adx_bench = row.get("adx_avg", 25)
    if pd.isna(adx_bench):
        adx_bench = 25

    adx_range_mult = getattr(config, "ADX_RANGING_MULT", 0.8)
    slope_range = getattr(config, "SLOPE_RANGING_THRESHOLD", 0.2)
    if adx < (adx_bench * adx_range_mult) and abs(slope) < slope_range:
        return "RANGING"

    adx_trend = getattr(config, "ADX_TREND_THRESHOLD", 22)
    atr_chaotic = getattr(config, "ATR_CHAOTIC_THRESHOLD", 3.0)
    slope_chaotic = getattr(config, "SLOPE_CHAOTIC_THRESHOLD", 1.0)
    if adx < adx_trend and (atr_p > atr_chaotic or abs(slope) > slope_chaotic):
        return "CHAOTIC"

    if adx > adx_trend:
        return "TRENDING"
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
    if "warm" in df.columns and not bool(df["warm"].iloc[-1]):
        return False, "Warm-up incomplete"

    recent = df.tail(10)
    if recent[["adx", "ema_slope"]].isna().any().any():
        return False, "Indicators not ready (NaN)"
    avg_adx = recent["adx"].mean()

    adx_trend = getattr(config, "ADX_TREND_THRESHOLD", 22)
    slope_range = getattr(config, "SLOPE_RANGING_THRESHOLD", 0.2)
    min_trending = getattr(config, "MIN_TRENDING_CANDLES", 5)
    min_adx_ready = getattr(config, "MIN_ADX_READY", 20)

    trending_candles = (
        (recent["adx"] > adx_trend) |
        (recent["ema_slope"].abs() > slope_range)
    ).sum()

    if trending_candles < min_trending:
        return False, f"Market in chop (only {trending_candles}/10 trending candles)"

    if avg_adx < min_adx_ready:
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
        recent_mean = df['close'].iloc[-5:-1].mean()
        if c1['close'] < recent_mean: patterns.append("Hammer")
        else: patterns.append("Hanging Man")

    if c1['close'] > c1['open'] and c2['open'] > c2['close']:
        if c1['close'] > c2['open'] and c1['open'] < c2['close']: patterns.append("Bullish Engulfing")
    elif c1['open'] > c1['close'] and c2['close'] > c2['open']:
        if c1['open'] > c2['close'] and c1['close'] < c2['open']: patterns.append("Bearish Engulfing")

    if (c3['open'] > c3['close']) and (abs(c2['open']-c2['close']) < abs(c3['open']-c3['close'])*0.3) and (c1['close'] > c1['open']):
        if c1['close'] > (c3['open'] + c3['close'])/2: patterns.append("Morning Star")
    elif (c3['close'] > c3['open']) and (abs(c2['open']-c2['close']) < abs(c3['open']-c3['close'])*0.3) and (c1['open'] > c1['close']):
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

def _fractal_pivots(x: np.ndarray, k: int, kind: str) -> list:
    """Indices i (k <= i < n-k) that are strict k-bar fractal highs/lows.
    A pivot at i is only knowable at bar i+k, so the last k bars never produce one."""
    out = []
    for i in range(k, len(x) - k):
        left = x[i - k:i]
        right = x[i + 1:i + k + 1]
        if kind == "high":
            if x[i] > left.max() and x[i] >= right.max():
                out.append(i)
        else:
            if x[i] < left.min() and x[i] <= right.min():
                out.append(i)
    return out


def _neutral_div(details: str) -> dict:
    return {
        "bearish_divergence": False, "bullish_divergence": False,
        "rsi_bearish_div": False, "macd_bearish_div": False,
        "rsi_bullish_div": False, "macd_bullish_div": False,
        "hidden_bearish_divergence": False, "hidden_bullish_divergence": False,
        "details": details,
    }


def detect_momentum_divergence(
    df: pd.DataFrame,
    lookback: int = 48,
    k: int = 3,
    min_sep: int = 5,
    min_atr_move: float = 0.5,
    rsi_buf: float = 3.0,
    macd_buf: float = 0.0,
    max_pivot_age: int = 12,
    require_both: bool = False,
) -> dict:
    """
    Regular and hidden RSI/MACD divergence on CONFIRMED pivots only.
    Requires columns (high, low, rsi14, macd_hist, atr14) from compute_indicators.
    If not already present in df, computes them causal-style on df.
    """
    if df is None or len(df) < lookback:
        return _neutral_div("insufficient_data")

    # If indicators are not precomputed on df, compute them
    need = {"high", "low", "rsi14", "macd_hist", "atr14"}
    if not need.issubset(df.columns):
        try:
            df = compute_indicators(df, warmup_bars=min(len(df), 200), strict=False)
        except Exception:
            return _neutral_div("indicators_failed")

    sub = df.iloc[-lookback:]
    if sub[list(need)].isna().any().any():
        return _neutral_div("indicators_not_ready")

    h = sub["high"].to_numpy(float)
    lo = sub["low"].to_numpy(float)
    rsi = sub["rsi14"].to_numpy(float)
    atr = sub["atr14"].to_numpy(float)
    hn = sub["macd_hist"].to_numpy(float) / np.where(atr > 0, atr, np.nan)
    if np.isnan(hn).any():
        return _neutral_div("zero_atr")
    n = len(sub)

    def pick(pivs):
        if len(pivs) < 2:
            return None
        i2 = pivs[-1]
        if (n - 1) - i2 > max_pivot_age:
            return None
        for i1 in reversed(pivs[:-1]):
            if i2 - i1 >= min_sep:
                return i1, i2
        return None

    res = _neutral_div("No divergence")
    details = []
    move = lambda i2: min_atr_move * atr[i2]

    pk = pick(_fractal_pivots(h, k, "high"))
    if pk:
        i1, i2 = pk
        r_lo = rsi[i2] < rsi[i1] - rsi_buf
        m_lo = hn[i2] < hn[i1] - macd_buf
        r_hi = rsi[i2] > rsi[i1] + rsi_buf
        m_hi = hn[i2] > hn[i1] + macd_buf
        if h[i2] > h[i1] + move(i2):  # higher high -> bearish div
            r, m = r_lo, m_lo
            if (r and m) if require_both else (r or m):
                res["rsi_bearish_div"], res["macd_bearish_div"] = bool(r), bool(m)
                res["bearish_divergence"] = True
                details.append(f"Bearish Div (high {h[i1]:.4f}->{h[i2]:.4f}, RSI {rsi[i1]:.1f}->{rsi[i2]:.1f}, age {n-1-i2})")
        elif h[i2] < h[i1] - move(i2):  # lower high -> hidden bearish
            if (r_hi and m_hi) if require_both else (r_hi or m_hi):
                res["hidden_bearish_divergence"] = True
                details.append(f"Hidden Bearish Div (high {h[i1]:.4f}->{h[i2]:.4f}, RSI {rsi[i1]:.1f}->{rsi[i2]:.1f}, age {n-1-i2})")

    tr = pick(_fractal_pivots(lo, k, "low"))
    if tr:
        i1, i2 = tr
        r_hi = rsi[i2] > rsi[i1] + rsi_buf
        m_hi = hn[i2] > hn[i1] + macd_buf
        r_lo = rsi[i2] < rsi[i1] - rsi_buf
        m_lo = hn[i2] < hn[i1] - macd_buf
        if lo[i2] < lo[i1] - move(i2):  # lower low -> bullish div
            r, m = r_hi, m_hi
            if (r and m) if require_both else (r or m):
                res["rsi_bullish_div"], res["macd_bullish_div"] = bool(r), bool(m)
                res["bullish_divergence"] = True
                details.append(f"Bullish Div (low {lo[i1]:.4f}->{lo[i2]:.4f}, RSI {rsi[i1]:.1f}->{rsi[i2]:.1f}, age {n-1-i2})")
        elif lo[i2] > lo[i1] + move(i2):  # higher low -> hidden bullish
            if (r_lo and m_lo) if require_both else (r_lo or m_lo):
                res["hidden_bullish_divergence"] = True
                details.append(f"Hidden Bullish Div (low {lo[i1]:.4f}->{lo[i2]:.4f}, RSI {rsi[i1]:.1f}->{rsi[i2]:.1f}, age {n-1-i2})")

    if details:
        res["details"] = "; ".join(details)
    return res


def compute_volume_profile(
    df: pd.DataFrame,
    lookback: int = 48,
    n_bins: int = 30,
    value_area_pct: float = 0.70
) -> dict:
    """
    Volume Profile & Value Area Engine (VPVR: POC, VAH, VAL).
    Computes volume distribution across prior candles (excluding the in-flight bar).
    """
    if df is None or len(df) < 10:
        return {
            "poc": 0.0, "vah": 0.0, "val": 0.0, "close": 0.0,
            "in_value_area": False, "above_vah": False, "below_val": False,
            "distance_to_vah_pct": 0.0, "distance_to_val_pct": 0.0,
            "details": "insufficient_data"
        }

    # Exclude current in-flight candle from profile distribution to prevent lookahead
    df_prior = df.iloc[:-1] if len(df) > 10 else df
    df_slice = df_prior.tail(lookback).copy()
    min_price = float(df_slice["low"].min())
    max_price = float(df_slice["high"].max())
    curr_close = float(df["close"].iloc[-1])

    if max_price <= min_price:
        return {
            "poc": curr_close, "vah": curr_close, "val": curr_close, "close": curr_close,
            "in_value_area": True, "above_vah": False, "below_val": False,
            "distance_to_vah_pct": 0.0, "distance_to_val_pct": 0.0,
            "details": "flat_range"
        }

    bins = np.linspace(min_price, max_price, n_bins + 1)
    bin_volumes = np.zeros(n_bins)

    lows = df_slice["low"].to_numpy(float)
    highs = df_slice["high"].to_numpy(float)
    vols = df_slice["volume"].to_numpy(float)

    for i in range(len(df_slice)):
        bar_low, bar_high, bar_vol = lows[i], highs[i], vols[i]
        idx_low = max(0, min(n_bins - 1, int(np.searchsorted(bins, bar_low) - 1)))
        idx_high = max(0, min(n_bins - 1, int(np.searchsorted(bins, bar_high) - 1)))
        span = max(1, idx_high - idx_low + 1)
        vol_per_bin = bar_vol / span
        bin_volumes[idx_low:idx_high + 1] += vol_per_bin

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
    Latest-bar view of the causal TD Setup counts.
    Counts run 1..9 on a confirmed price flip and complete at 9.
    """
    if df is None or len(df) < 6:
        return {
            "setup_direction": "NONE", "count": 0, "up_count": 0, "down_count": 0,
            "is_td9_upward_exhaustion": False, "is_td9_downward_exhaustion": False,
            "is_td_exhaustion_risk": False, "details": "insufficient_data",
        }
    if "td_sell_setup" in df.columns and "td_buy_setup" in df.columns:
        up = int(df["td_sell_setup"].iloc[-1])
        dn = int(df["td_buy_setup"].iloc[-1])
    else:
        u, d = td_setup_counts(df["close"])
        up, dn = int(u.iloc[-1]), int(d.iloc[-1])

    direction = "UP" if up > 0 else ("DOWN" if dn > 0 else "NONE")
    count = up if up > 0 else -dn
    return {
        "setup_direction": direction,
        "count": count,
        "up_count": up,
        "down_count": dn,
        "is_td9_upward_exhaustion": up == 9,
        "is_td9_downward_exhaustion": dn == 9,
        "is_td_exhaustion_risk": up in (8, 9) or dn in (8, 9),
        "details": f"TD Setup: {direction} count = {abs(count)} (sell9: {up == 9}, buy9: {dn == 9})",
    }

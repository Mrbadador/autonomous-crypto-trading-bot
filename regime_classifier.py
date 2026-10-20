# regime_classifier.py — Quantitative Regime Classifier Agent

import sqlite3
import requests
import pandas as pd
import numpy as np
import logging
import os
import re
import time
from dotenv import load_dotenv
from datetime import datetime, UTC

# ENV CUSTODIAN: Force-load .env from the project root before any I/O.
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] REGIME_CLASSIFIER: %(message)s")
log = logging.getLogger("regime_classifier")

# Optional accelerator for ADX. Not declared in requirements.txt, so its absence is the normal
# case; calculate_adx() then uses the Wilder fallback. Imported once here instead of per call so
# the ImportError is reported once rather than swallowed on every classification pass.
try:
    import pandas_ta as _pandas_ta
except ImportError:
    _pandas_ta = None
    log.debug("pandas_ta not installed - ADX will use the internal Wilder calculation.")
except Exception as _ta_import_err:  # pragma: no cover - broken install
    _pandas_ta = None
    log.warning(f"pandas_ta import failed ({_ta_import_err}) - ADX will use the internal Wilder calculation.")

# Row cap for the append-only market_regime history. At the ~5 minute classification cadence
# this keeps roughly 100 days of regime transitions for gate debugging without growing the
# live trading database without bound.
REGIME_HISTORY_ROWS = 30000

# Contract for the AI cognitive arbiter verdict. A verdict is cached only if it satisfies
# this contract, because whatever is cached is applied to the regime label and the leverage
# recommendation for the next 15 minutes, and a cached `null` used to raise float(None)
# inside classify_and_save() - which the __main__ loop did not catch, killing the daemon.
AI_REGIME_BIASES = ("BULL_EXPANSION", "BEAR_DUMP", "DEFENSIVE_CHOP", "LIQUIDITY_HUNT_TRAP")
AI_LEVERAGE_ALLOWLIST = (2, 3, 5)
AI_THESIS_MAX_CHARS = 400
AI_TRAP_RISK_DEFAULT = 0.50
AI_LEVERAGE_DEFAULT = 3
AI_OFFLINE_BIAS = "MATH_DEFAULT"

# Turbulence circuit breaker (Fix #4): quantiles are only trusted once the persisted
# history is large enough that the 99th percentile is not just interpolating the top
# two fresh values; tiers need consecutive passes before they flip.
TURB_POOL_TRUST_MIN = 100
TURB_HISTORY_RETENTION_DAYS = 30
TURB_ENGAGE_BREACHES = 2
TURB_DISENGAGE_CALMS = 3

# ENV INTEGRITY CHECK
_api_key = os.getenv("BITGET_API_KEY", "")
if _api_key:
    log.info(f"[BOOT] ENV OK — BITGET_API_KEY loaded (last 4: ...{_api_key[-4:]}).")
else:
    log.warning("[BOOT] BITGET_API_KEY is empty — this module uses public endpoints only.")

class RegimeClassifier:
    """
    Agent V14.5: Regime Classifier.
    Analyzes historical bar memory to determine whether the market is trending or sideways.
    Saves a global flag to 'trading_data.db' for engines to gate entry signals.
    """
    def __init__(self, db_path="trading_data.db"):
        self.db_path = db_path
        self.last_ai_eval_ts = 0.0
        self.ai_eval_interval_sec = 900  # 15 minutes = 96 calls/day
        self.latest_ai_arbiter = {
            "ai_regime_bias": "DEFENSIVE_CHOP",
            "trap_risk_score": AI_TRAP_RISK_DEFAULT,
            "recommended_leverage_cap": AI_LEVERAGE_DEFAULT,
            "structural_thesis": "Baseline cognitive regime initialization.",
            "override_math_regime": False,
            "evaluated_by": "baseline",
            "evaluated_at_utc": None,
            "evaluated_at_epoch": 0.0,
        }
        # Hysteresis & Stabilization for AI Regime Arbiter (prevents 15m ping-pong flip-flopping)
        self.active_ai_bias = "DEFENSIVE_CHOP"
        self.pending_ai_bias = None
        self.pending_ai_bias_count = 0
        self.ai_bias_dwell_ts = 0.0
        self.last_counted_ai_eval_epoch = None
        self.min_ai_bias_dwell_sec = 2700.0  # 45 minutes minimum dwell time or 2 consecutive matches
        # Turbulence tier hysteresis state (persisted across restarts via system_state)
        self.turbulence_tier = "CALM_NORMAL"
        self._turb_breach_streak = 0
        self._turb_calm_streak = 0
        try:
            from markov_regime import MarkovRegimeEngine
            self.markov_engine = MarkovRegimeEngine()
        except Exception as me_err:
            log.warning(f"Could not initialize MarkovRegimeEngine: {me_err}")
            self.markov_engine = None
        self._init_db()

    def _init_db(self):
        """Ensures the market_regime history table exists in trading_data.db with AI columns."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=60.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=60000;")
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS market_regime (
                        id INTEGER PRIMARY KEY DEFAULT 1,
                        timestamp TEXT,
                        regime TEXT,
                        hurst REAL,
                        adx REAL,
                        source_close_ts TEXT,
                        math_regime TEXT
                    )
                """)
                for col, col_type in [
                    ("plus_di", "REAL"), ("minus_di", "REAL"), ("macro_trend", "TEXT"),
                    ("ai_bias", "TEXT"), ("trap_risk", "REAL"), ("ai_thesis", "TEXT"),
                    ("turbulence_index", "REAL"), ("turbulence_threshold", "REAL"), ("turbulence_state", "TEXT"),
                    # `timestamp` is the publication time of the row; `source_close_ts` is the close
                    # time of the newest 15m bar it was computed from. Keeping them apart is the only
                    # way a reader can tell a live producer analysing old data from a stalled one.
                    ("source_close_ts", "TEXT"), ("math_regime", "TEXT")
                ]:
                    try:
                        conn.execute(f"ALTER TABLE market_regime ADD COLUMN {col} {col_type}")
                    except sqlite3.OperationalError:
                        pass
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS turbulence_history (
                        ts REAL NOT NULL,
                        value REAL NOT NULL
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_turbulence_history_ts ON turbulence_history (ts)")
                row_tier = None
                try:
                    row_tier = conn.execute(
                        "SELECT value FROM system_state WHERE key = 'turbulence_tier_state'"
                    ).fetchone()
                except sqlite3.OperationalError:
                    pass
                if row_tier and row_tier[0]:
                    try:
                        import json as _json
                        tstate = _json.loads(row_tier[0])
                        self.turbulence_tier = str(tstate.get("tier") or "CALM_NORMAL")
                        self._turb_breach_streak = int(tstate.get("breach_streak") or 0)
                        self._turb_calm_streak = int(tstate.get("calm_streak") or 0)
                    except Exception:
                        pass
        except Exception as e:
            log.critical(f"Failed to initialize database table: {e}")
            raise
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    @staticmethod
    def _interval_seconds(interval):
        """Seconds covered by a Bitget granularity string ('15m', '4h', '1d'); None if unparsable."""
        match = re.fullmatch(r"(\d+)\s*([mhd])", str(interval).strip().lower())
        if not match:
            return None
        return int(match.group(1)) * {"m": 60, "h": 3600, "d": 86400}[match.group(2)]

    def _normalize_bars(self, df, interval, symbol="BTCUSDT"):
        """Returns strictly closed, sorted, de-duplicated bars with no forward-dated row.

        Bitget's candle endpoint always appends the bar that is still forming; its close,
        high and low mutate between calls. Feeding it to ADX/Hurst/turbulence means the
        regime verdict is partly a function of an incomplete bar, so any bar whose close
        time has not passed yet is dropped here. Out-of-order and duplicate timestamps are
        repaired; genuine gaps in the history are reported, never fabricated.
        """
        if df is None or df.empty:
            return df
        bars = df.copy()
        try:
            bars["open_time"] = bars["open_time"].astype("int64")
            for col in ("open", "high", "low", "close", "volume"):
                if col in bars.columns:
                    bars[col] = bars[col].astype(float)
        except (TypeError, ValueError) as conv_err:
            log.warning(f"[BARS] {symbol} {interval}: non-numeric candle payload ({conv_err}); using rows as returned.")
            return bars

        bars = bars.dropna(subset=[c for c in ("open", "high", "low", "close") if c in bars.columns])
        bars = bars.sort_values("open_time", kind="mergesort")
        dupes = int(bars.duplicated(subset="open_time").sum())
        if dupes:
            log.warning(f"[BARS] {symbol} {interval}: dropped {dupes} duplicate open_time row(s).")
            bars = bars.drop_duplicates(subset="open_time", keep="last")

        step_sec = self._interval_seconds(interval)
        if step_sec is not None:
            cutoff_ms = int(time.time() * 1000)
            closed = bars[bars["open_time"] + step_sec * 1000 <= cutoff_ms]
            dropped = int(len(bars) - len(closed))
            if dropped:
                log.debug(f"[BARS] {symbol} {interval}: discarded {dropped} still-forming bar(s) from regime input.")
            bars = closed

        bars = bars.reset_index(drop=True)

        if step_sec is not None and len(bars) >= 2:
            steps = bars["open_time"].diff().dropna().astype("int64").to_numpy()
            expected = step_sec * 1000
            broken = steps[steps != expected]
            if broken.size:
                log.warning(
                    f"[BARS] {symbol} {interval}: {broken.size} discontinuous step(s) in closed history "
                    f"(expected {expected} ms, worst {int(max(broken))} ms). Indicator memory is shorter than nominal."
                )
        return bars

    def fetch_ohlcv(self, symbol="BTCUSDT", interval="15m", limit=105):
        """Pulls OHLCV bars directly from the public Bitget USDT-FUTURES API.

        The returned frame is normalised by `_normalize_bars`: closed bars only, sorted
        and de-duplicated by `open_time`.
        """
        gran_map = {
            "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1H", "1H": "1H", "2h": "2H", "2H": "2H", "4h": "4H", "4H": "4H",
            "6h": "6H", "6H": "6H", "12h": "12H", "12H": "12H",
            "1d": "1D", "1D": "1D", "1w": "1W", "1W": "1W"
        }
        api_gran = gran_map.get(interval, interval)
        clean_sym = symbol.replace("/", "").replace("-", "").upper()
        url = f"https://api.bitget.com/api/v2/mix/market/candles?symbol={clean_sym}&granularity={api_gran}&limit={limit}&productType=USDT-FUTURES"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json().get("data", [])
                if data:
                    df = pd.DataFrame(data, columns=[
                        "open_time", "open", "high", "low", "close", "volume", "quote_volume"
                    ])
                    for col in ["open", "high", "low", "close", "volume"]:
                        df[col] = df[col].astype(float)
                    return self._normalize_bars(df, interval, symbol=clean_sym)
            else:
                log.error(f"Bitget candles API returned HTTP {response.status_code} for {clean_sym} ({api_gran}): {response.text[:200]}")
                return None
        except Exception as e:
            log.error(f"Failed to fetch Bitget historical bars: {e}")
            return None

    @staticmethod
    def _bar_close_ts(df, interval):
        """ISO-8601 UTC close time of the newest bar in a Bitget OHLCV frame.

        Bitget returns `open_time` in milliseconds; the close time is open_time plus the
        bar duration. Because fetch_ohlcv() discards still-forming bars this is the close
        time of the last *completed* bar, i.e. a market timestamp the data already
        reflects. Returns None (stored as NULL) rather than a guessed value.
        """
        try:
            seconds = RegimeClassifier._interval_seconds(interval)
            if seconds is None:
                raise ValueError(f"unparsable granularity '{interval}'")
            last_open_ms = int(float(df["open_time"].iloc[-1]))
            return datetime.fromtimestamp((last_open_ms + seconds * 1000) / 1000.0, UTC).isoformat()
        except Exception as e:
            log.warning(f"Could not derive source bar close time from '{interval}' bars: {e}")
            return None

    def calculate_hurst(self, prices):
        """Estimates Hurst Exponent over prices using Rescaled Range (R/S) Analysis.

        R/S is computed over every overlapping window of each lag, so the newest bars are
        always inside the sample. The previous implementation sliced the series into
        num_subsets = N // lag *disjoint* blocks starting at index 0, which silently threw
        away the trailing N % lag returns: for the live 99-return window at lag 50 only the
        oldest 50 returns were used, so a violent move in the most recent 25 bars could not
        change the exponent at all.
        """
        if len(prices) < 50:
            return 0.50
        try:
            # Convert series to logarithmic returns
            returns = np.diff(np.log(prices))
            N = len(returns)

            lags = [10, 20, 30, 40, 50]
            valid_pairs = []

            for lag in lags:
                if N < lag:
                    continue
                rs_sub = []
                for start in range(0, N - lag + 1):
                    sub = returns[start:start + lag]
                    mean_adj = sub - np.mean(sub)
                    cum_sum = np.cumsum(mean_adj)
                    r = np.max(cum_sum) - np.min(cum_sum)
                    s = np.std(sub)
                    if s > 0:
                        rs_sub.append(r / s)
                if rs_sub:
                    avg_rs = float(np.mean(rs_sub))
                    if avg_rs > 0:
                        valid_pairs.append((lag, avg_rs))
                    
            if len(valid_pairs) < 2:
                return 0.50
                
            valid_lags = [p[0] for p in valid_pairs]
            valid_rs = [p[1] for p in valid_pairs]
            
            poly = np.polyfit(np.log(valid_lags), np.log(valid_rs), 1)
            hurst_val = float(poly[0])
            return max(0.0, min(1.0, hurst_val))
        except Exception as e:
            log.error(f"Error calculating Hurst Exponent: {e}")
            return 0.50

    def calculate_adx(self, df, length=14):
        """
        Calculates 14-period Average Directional Index (ADX) along with +DI and -DI.
        Returns: (adx_val, plus_di_val, minus_di_val)
        """
        if _pandas_ta is not None:
            try:
                adx_df = _pandas_ta.adx(df['high'], df['low'], df['close'], length=length)
                adx_val = float(adx_df[f'ADX_{length}'].iloc[-1])
                plus_di = float(adx_df[f'DMP_{length}'].iloc[-1])
                minus_di = float(adx_df[f'DMN_{length}'].iloc[-1])
                if not np.isnan(adx_val) and not np.isnan(plus_di) and not np.isnan(minus_di):
                    return adx_val, plus_di, minus_di
                log.warning("pandas_ta returned NaN ADX/DI; falling back to the Wilder calculation.")
            except Exception as ta_err:
                log.warning(f"pandas_ta ADX failed ({ta_err}); falling back to the Wilder calculation.")

        # Wilder fallback. TR and +/-DM must be seeded with the SUM of the first `length` values
        # before Wilder's recurrence is applied. The previous code seeded with a rolling mean, so
        # the smoothed series carried a seed error of (length-1)/length that only decays with
        # bars. Seeding at index `length` (excluding bar 0, whose TR has no prior close) matches
        # reference Wilder implementations such as ta.trend.ADXIndicator to machine precision.
        try:
            high = np.asarray(df['high'], dtype=float)
            low = np.asarray(df['low'], dtype=float)
            close = np.asarray(df['close'], dtype=float)
            n = len(high)

            if n < 2 * length + 1:
                log.warning(
                    f"ADX fallback needs at least {2 * length + 1} bars, got {n}; "
                    "returning neutral 20.0/0.0/0.0 instead of a partially smoothed value."
                )
                return 20.0, 0.0, 0.0

            up_move = np.full(n, np.nan)
            down_move = np.full(n, np.nan)
            if n > 1:
                up_move[1:] = high[1:] - high[:-1]
                down_move[1:] = low[:-1] - low[1:]

            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

            tr = np.empty(n)
            tr[0] = high[0] - low[0]
            if n > 1:
                prev_close = close[:-1]
                tr[1:] = np.maximum(
                    high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)),
                )

            def _wilder(values):
                out = np.full(n, np.nan)
                out[length] = values[1:length + 1].sum()
                for i in range(length + 1, n):
                    out[i] = out[i - 1] - (out[i - 1] / length) + values[i]
                return out

            tr_smooth = _wilder(tr)
            plus_dm_smooth = _wilder(plus_dm)
            minus_dm_smooth = _wilder(minus_dm)

            with np.errstate(divide="ignore", invalid="ignore"):
                active = tr_smooth > 0
                plus_di_arr = np.where(active, 100.0 * plus_dm_smooth / tr_smooth, 0.0)
                minus_di_arr = np.where(active, 100.0 * minus_dm_smooth / tr_smooth, 0.0)
                di_sum = plus_di_arr + minus_di_arr
                raw_dx = np.where(di_sum > 0, 100.0 * np.abs(plus_di_arr - minus_di_arr) / di_sum, 0.0)

            dx = np.full(n, np.nan)
            dx[active] = raw_dx[active]

            adx_series = np.full(n, np.nan)
            seed_idx = 2 * length - 1
            adx_series[seed_idx] = dx[length:seed_idx + 1].mean()
            for i in range(seed_idx + 1, n):
                adx_series[i] = (adx_series[i - 1] * (length - 1) + dx[i]) / length

            adx_val = float(adx_series[-1])
            plus_di_val = float(plus_di_arr[-1])
            minus_di_val = float(minus_di_arr[-1])

            if np.isnan(adx_val):
                log.warning("ADX fallback produced NaN; defaulting to 20.0 (Choppy).")
                adx_val = 20.0
            if np.isnan(plus_di_val):
                plus_di_val = 0.0
            if np.isnan(minus_di_val):
                minus_di_val = 0.0

            return adx_val, plus_di_val, minus_di_val
        except Exception as e:
            log.warning(f"ADX calculation error: {e}. Defaulting to 20.0 (Choppy).")
            return 20.0, 0.0, 0.0

    def calculate_mahalanobis_turbulence(
        self,
        basket=None,
        interval="15m",
        limit=105,
        window=48
    ) -> dict:
        """
        FinRL Mahalanobis Cross-Asset Turbulence Index & Two-Tier Quantile Circuit Breaker:
            d_t = (r_t - mu)^T * Sigma^+ * (r_t - mu)
        where Sigma^+ is the Moore-Penrose pseudoinverse of the rolling asset return covariance matrix.
        Implements FinRL DRLEnsembleAgent's dynamic threshold tightening (tau_90 vs tau_99)
        and bounded sigmoid_sign compression.
        """
        if basket is None:
            basket = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT", "NEARUSDT"]

        default_res = {
            "turbulence_index": 0.0,
            "turbulence_tau90": 12.0,
            "turbulence_tau99": 22.0,
            "active_threshold": 22.0,
            "trailing_mean": 0.0,
            "sigmoid_score": 0.0,
            "percentile_rank": 50.0,
            "turbulence_breach": False,
            "turbulence_state": "CALM_NORMAL",
            "assets_sampled": 0,
        }
        try:
            close_series_list = []
            for sym in basket:
                df_sym = self.fetch_ohlcv(symbol=sym, interval=interval, limit=limit)
                if df_sym is not None and len(df_sym) >= window + 15:
                    s = df_sym["close"].reset_index(drop=True)
                    close_series_list.append(s.rename(sym))

            if len(close_series_list) < 3:
                return default_res

            min_len = min(len(s) for s in close_series_list)
            aligned = pd.concat([s.iloc[-min_len:].reset_index(drop=True) for s in close_series_list], axis=1)
            returns_df = np.log(aligned / aligned.shift(1)).dropna()
            ret_matrix = returns_df.to_numpy(dtype=float)
            n_rows, n_assets = ret_matrix.shape

            if n_rows <= window + 5:
                return default_res

            turbulence_history = []
            for t in range(window, n_rows):
                hist_slice = ret_matrix[t - window : t, :]
                curr_ret = ret_matrix[t, :]
                mu = np.mean(hist_slice, axis=0)
                diff = curr_ret - mu
                cov = np.cov(hist_slice, rowvar=False)
                if cov.ndim == 0:
                    cov = np.array([[float(cov)]])
                pinv_cov = np.linalg.pinv(cov)
                d_t = float(diff @ pinv_cov @ diff.T)
                turbulence_history.append(max(0.0, d_t))

            if not turbulence_history:
                return default_res

            turb_arr = np.array(turbulence_history, dtype=float)
            current_d = float(turb_arr[-1])

            # Fix #4: quantiles come from a persisted 30-day pool. The freshly computed
            # window is ~55 points, where a 99th percentile is just interpolation of the
            # top two values - one spike could swing the active threshold 15 -> 146.
            pool_values, persisted_count, bootstrap_active = self._record_and_load_turbulence_pool(
                current_d, turb_arr
            )
            if bootstrap_active:
                log.info(
                    f"[TURB-BOOTSTRAP] Persisted d_t history {persisted_count}/{TURB_POOL_TRUST_MIN} "
                    f"- quantiles seeded from {len(pool_values)} computed points; breaker stays armed."
                )
            pool_arr = np.asarray(pool_values, dtype=float)

            tau_90 = float(np.percentile(pool_arr, 90))
            tau_99 = float(np.percentile(pool_arr, 99))
            # Median, not mean: one spike in the last 12 bars must not drop the tier.
            trailing_12 = float(np.median(turb_arr[-min(12, len(turb_arr)):]))

            # FinRL Two-Tier Dynamic Quantile Rule:
            # When trailing turbulence exceeds the 90th percentile, tighten active circuit-breaker to tau_90
            active_thresh = tau_90 if trailing_12 > tau_90 else tau_99
            active_thresh = max(active_thresh, 1e-4)

            # FinRL bounded sigmoid compression
            raw_sig = 1.0 / (1.0 + float(np.exp(-current_d * np.e / active_thresh)))
            sigmoid_score = round(max(0.0, min(1.0, (raw_sig - 0.5) * 2.0)), 4)
            pct_rank = round(float(np.mean(pool_arr <= current_d) * 100.0), 1)

            breach = bool(current_d > active_thresh)
            self.turbulence_tier = self._advance_turbulence_tier(breach)
            self._persist_turbulence_tier()
            if self.turbulence_tier != "EXTREME_SHOCK" and current_d > tau_90:
                state_str = "ELEVATED_STRESS"
            else:
                state_str = self.turbulence_tier

            return {
                "turbulence_index": round(current_d, 3),
                "turbulence_tau90": round(tau_90, 3),
                "turbulence_tau99": round(tau_99, 3),
                "active_threshold": round(active_thresh, 3),
                "trailing_mean": round(trailing_12, 3),
                "sigmoid_score": sigmoid_score,
                "percentile_rank": pct_rank,
                "turbulence_breach": breach,
                "turbulence_state": state_str,
                "assets_sampled": int(n_assets),
                "history_pool_size": int(persisted_count),
                "bootstrap_active": bool(bootstrap_active),
            }
        except Exception as e:
            log.warning(f"[FINRL-TURBULENCE-NOTE] Error calculating Mahalanobis turbulence: {e}")
            return default_res

    def _record_and_load_turbulence_pool(self, current_d: float, local_arr) -> tuple:
        """Persists this pass's d_t and returns (pool_values, persisted_count, bootstrap_active).

        Below TURB_POOL_TRUST_MIN persisted points the freshly computed window seeds the
        pool instead, so the breaker is armed (conservative) from the first pass and
        never defines its own ceiling from two interpolated points.
        """
        persisted = []
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            try:
                with conn:
                    conn.execute(
                        "INSERT INTO turbulence_history (ts, value) VALUES (?, ?)",
                        (time.time(), max(0.0, float(current_d))),
                    )
                    conn.execute(
                        "DELETE FROM turbulence_history WHERE ts < ?",
                        (time.time() - TURB_HISTORY_RETENTION_DAYS * 86400.0,),
                    )
                rows = conn.execute(
                    "SELECT value FROM turbulence_history WHERE ts >= ? ORDER BY ts DESC LIMIT 5000",
                    (time.time() - TURB_HISTORY_RETENTION_DAYS * 86400.0,),
                ).fetchall()
                persisted = [float(r[0]) for r in rows]
            finally:
                conn.close()
        except Exception as pool_err:
            log.warning(f"[TURB-POOL-NOTE] Persisted pool unavailable ({pool_err}); using seeded window.")
        if len(persisted) >= TURB_POOL_TRUST_MIN:
            return persisted, len(persisted), False
        seeded = list(persisted) + [float(v) for v in local_arr]
        return seeded, len(persisted), True

    def _advance_turbulence_tier(self, breach: bool) -> str:
        """Two-pass engage / three-pass disengage hysteresis for EXTREME_SHOCK (Fix #4).

        Once engaged, the tier stays EXTREME_SHOCK until TURB_DISENGAGE_CALMS consecutive
        calm passes clear it; a single breaching bar cannot flip the breaker on or off.
        """
        if breach:
            self._turb_calm_streak = 0
            if self.turbulence_tier == "EXTREME_SHOCK":
                self._turb_breach_streak = 0
            else:
                self._turb_breach_streak += 1
                if self._turb_breach_streak >= TURB_ENGAGE_BREACHES:
                    self.turbulence_tier = "EXTREME_SHOCK"
                    self._turb_breach_streak = 0
                    log.warning(
                        f"[FINRL-TURBULENCE-TIER] EXTREME_SHOCK engaged after "
                        f"{TURB_ENGAGE_BREACHES} consecutive breaches."
                    )
        else:
            self._turb_breach_streak = 0
            self._turb_calm_streak += 1
            if self.turbulence_tier == "EXTREME_SHOCK" and self._turb_calm_streak >= TURB_DISENGAGE_CALMS - 1:
                # TURB_DISENGAGE_CALMS calm observations total (the pass that engaged the
                # tier last, then this streak) must land before the breaker releases.
                self.turbulence_tier = "CALM_NORMAL"
                log.info(
                    f"[FINRL-TURBULENCE-TIER] EXTREME_SHOCK disengaged after "
                    f"{TURB_DISENGAGE_CALMS} consecutive calm passes."
                )
        return self.turbulence_tier

    def _persist_turbulence_tier(self):
        try:
            conn = None
            try:
                import json as _json
                conn = sqlite3.connect(self.db_path, timeout=10.0)
                payload = _json.dumps({
                    "tier": self.turbulence_tier,
                    "breach_streak": self._turb_breach_streak,
                    "calm_streak": self._turb_calm_streak,
                })
                with conn:
                    try:
                        conn.execute(
                            "UPDATE system_state SET value = ?, updated_at = ? WHERE key = 'turbulence_tier_state'",
                            (payload, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
                        )
                        if conn.execute("SELECT changes()").fetchone()[0] == 0:
                            conn.execute(
                                "INSERT INTO system_state (key, value, updated_at) VALUES ('turbulence_tier_state', ?, ?)",
                                (payload, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
                            )
                    except sqlite3.OperationalError:
                        pass
            finally:
                if conn:
                    conn.close()
        except Exception as tier_err:
            log.debug(f"[TURB-TIER-PERSIST-NOTE] {tier_err}")

    def fetch_funding_rate(self, symbol="BTCUSDT"):
        """Pulls current funding rate from Bitget public API."""
        clean_sym = symbol.replace("/", "").replace("-", "").upper()
        url = f"https://api.bitget.com/api/v2/mix/market/current-fund-rate?symbol={clean_sym}&productType=USDT-FUTURES"
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                data = res.json().get("data", [])
                if data and isinstance(data, list):
                    return float(data[0].get("fundingRate", 0.0))
        except Exception:
            pass
        return 0.0001

    @staticmethod
    def _bounded_float(value, default, lo, hi):
        """float() that can never raise: non-numeric, None, NaN and +-inf fall back to default."""
        try:
            as_float = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not np.isfinite(as_float):
            return float(default)
        return min(hi, max(lo, as_float))

    @staticmethod
    def _bounded_int(value, default, allowed):
        """Snaps a provider-supplied integer to the allowlist; anything else becomes default."""
        try:
            as_int = int(round(float(value)))
        except (TypeError, ValueError):
            return int(default)
        return as_int if as_int in allowed else int(default)

    def _validate_ai_payload(self, parsed, source: str):
        """Returns a normalised verdict dict, or None when the provider output is unusable.

        Runs before caching: an out-of-enum bias, a null/out-of-range trap score or a
        leverage cap outside {2, 3, 5} must not reach latest_ai_arbiter, the regime row or
        the leverage the dashboards quote.
        """
        if not isinstance(parsed, dict):
            log.warning(f"[AI-ARBITER-VALIDATION] {source}: payload is {type(parsed).__name__}, not an object - discarded.")
            return None

        raw_bias = parsed.get("ai_regime_bias")
        bias = str(raw_bias or "").strip().upper()
        if bias not in AI_REGIME_BIASES:
            log.warning(f"[AI-ARBITER-VALIDATION] {source}: ai_regime_bias {raw_bias!r} is not one of {AI_REGIME_BIASES} - discarded.")
            return None

        raw_score = parsed.get("trap_risk_score")
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = float("nan")
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            log.warning(f"[AI-ARBITER-VALIDATION] {source}: trap_risk_score {raw_score!r} is not a finite 0..1 value - discarded.")
            return None

        raw_cap = parsed.get("recommended_leverage_cap")
        try:
            cap = int(round(float(raw_cap)))
        except (TypeError, ValueError):
            cap = None
        if cap not in AI_LEVERAGE_ALLOWLIST:
            log.warning(f"[AI-ARBITER-VALIDATION] {source}: recommended_leverage_cap {raw_cap!r} not in {AI_LEVERAGE_ALLOWLIST} - discarded.")
            return None

        raw_thesis = parsed.get("structural_thesis")
        thesis = str(raw_thesis if raw_thesis is not None else "").strip()[:AI_THESIS_MAX_CHARS]
        raw_override = parsed.get("override_math_regime")
        if isinstance(raw_override, str):
            override = raw_override.strip().lower() in ("true", "1", "yes")
        else:
            override = bool(raw_override)

        return {
            "ai_regime_bias": bias,
            "trap_risk_score": score,
            "recommended_leverage_cap": cap,
            "structural_thesis": thesis,
            "override_math_regime": override,
            "evaluated_by": source,
        }

    def _offline_ai_fallback(self) -> dict:
        """Deterministic verdict used when every provider failed or was rejected.

        Deliberately carries no evaluated_at stamp: classify_and_save() treats an
        unstamped verdict as 'no fresh AI', so it can neither promote nor silently replace
        a last known protective veto with a permissive default.
        """
        return {
            "ai_regime_bias": AI_OFFLINE_BIAS,
            "trap_risk_score": 0.35,
            "recommended_leverage_cap": AI_LEVERAGE_DEFAULT,
            "structural_thesis": "AI offline — defaulting to deterministic mathematical regime rules.",
            "override_math_regime": False,
            "evaluated_by": "offline_fallback",
        }

    def _query_ai_cognitive_arbiter(self, telemetry: dict) -> dict:
        """
        Dispatches a compact prompt to Google Gemini (or Groq failover pool)
        to evaluate whether current momentum is genuine trend expansion or a liquidity trap.

        Only verdicts that pass _validate_ai_payload() are returned as fresh; every other
        path returns the deterministic offline fallback, so callers never cache garbage.
        """
        import json
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        system_prompt = (
            "You are the Chief Quantitative Market Regime Arbiter. "
            "You evaluate whether current cryptocurrency market structure represents "
            "a genuine institutional trend expansion (BULL_EXPANSION / BEAR_DUMP) "
            "or a retail stop-hunt / distribution trap / chop (LIQUIDITY_HUNT_TRAP / DEFENSIVE_CHOP).\n\n"
            "Decision Rules:\n"
            "1. If mathematical ADX is high but candles show long upper rejection wicks or funding is overheated (> 0.02%), "
            "classify as 'LIQUIDITY_HUNT_TRAP' with high trap_risk_score (> 0.65) and override_math_regime=true.\n"
            "2. If price is clean, candles have small wicks in the trend direction, and 4H trend aligns, classify as 'BULL_EXPANSION' "
            "with trap_risk_score < 0.30 and recommended_leverage_cap=5.\n"
            "3. If consolidating inside a range, classify as 'DEFENSIVE_CHOP' with recommended_leverage_cap=3.\n"
            "Return strictly valid JSON with no markdown."
        )
        user_prompt = (
            f"Market Telemetry Context:\n{json.dumps(telemetry, indent=2)}\n\n"
            "Return JSON format:\n"
            "{\n"
            '  "ai_regime_bias": "BULL_EXPANSION" | "BEAR_DUMP" | "DEFENSIVE_CHOP" | "LIQUIDITY_HUNT_TRAP",\n'
            '  "trap_risk_score": 0.0 to 1.0,\n'
            '  "recommended_leverage_cap": 2 | 3 | 5,\n'
            '  "structural_thesis": "1-2 concise institutional sentences",\n'
            '  "override_math_regime": true | false\n'
            "}"
        )

        import config
        gemini_models = getattr(config, "GEMINI_MODEL_POOL", [
            os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"),
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash",
        ])
        groq_models = getattr(config, "GROQ_MODEL_POOL", [
            os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "qwen/qwen3-32b",
        ])

        # Primary: Gemini Multi-Model Pool
        if gemini_key and "REPLACE" not in gemini_key:
            for model in gemini_models:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key.strip()}"
                payload = {
                    "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}],
                    "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1}
                }
                try:
                    res = requests.post(url, json=payload, timeout=8)
                    if res.status_code == 200:
                        text = res.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        if text:
                            validated = self._validate_ai_payload(json.loads(text), f"Gemini:{model}")
                            if validated:
                                validated["evaluated_at_utc"] = datetime.now(UTC).isoformat()
                                validated["evaluated_at_epoch"] = time.time()
                                log.info(f"[AI-REGIME-ARBITER] Gemini ({model}) synthesis: {validated['ai_regime_bias']} (Trap Risk: {validated['trap_risk_score']:.2f})")
                                return validated
                            continue
                    else:
                        log.warning(f"[AI-REGIME-ARBITER] Gemini ({model}) HTTP {res.status_code}")
                except Exception as ge:
                    log.debug(f"[AI-REGIME-ARBITER] Gemini ({model}) error: {ge}")

        # Secondary: Groq Multi-Key & Multi-Model Failover Pool
        groq_keys = [
            k.strip() for k in [
                os.getenv("GROQ_API_KEY", ""),
                os.getenv("GROQ_API_KEY_2", ""),
                os.getenv("GROQ_API_KEY_3", ""),
            ] if k and k.strip()
        ]
        for gk in groq_keys:
            for g_model in groq_models:
                url_groq = "https://api.groq.com/openai/v1/chat/completions"
                headers = {"Authorization": f"Bearer {gk}", "Content-Type": "application/json"}
                payload_groq = {
                    "model": g_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"}
                }
                try:
                    res_g = requests.post(url_groq, headers=headers, json=payload_groq, timeout=8)
                    if res_g.status_code == 200:
                        text_g = res_g.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                        if text_g:
                            validated_g = self._validate_ai_payload(json.loads(text_g), f"Groq:{g_model}")
                            if validated_g:
                                validated_g["evaluated_at_utc"] = datetime.now(UTC).isoformat()
                                validated_g["evaluated_at_epoch"] = time.time()
                                log.info(f"[AI-REGIME-ARBITER] Groq ({g_model}) synthesis: {validated_g['ai_regime_bias']} (Trap Risk: {validated_g['trap_risk_score']:.2f})")
                                return validated_g
                            continue
                    else:
                        log.warning(f"[AI-REGIME-ARBITER] Groq ({g_model}) HTTP {res_g.status_code}")
                except Exception as gr_err:
                    log.debug(f"[AI-REGIME-ARBITER] Groq ({g_model}) error: {gr_err}")

        log.warning("[AI-REGIME-ARBITER] ⚠️ No provider returned a contract-valid verdict; holding deterministic math rules.")
        return self._offline_ai_fallback()

    def classify_and_save(self):
        """Runs the directional multi-timeframe classifier pipeline with AI Cognitive Arbitration."""
        import time
        import json
        df = self.fetch_ohlcv(symbol="BTCUSDT", interval="15m", limit=100)
        if df is None or len(df) < 50:
            log.error("Insufficient 15m bar memory to execute classification. Aborting.")
            return False

        # Fetch 4H macro bars for higher-timeframe trend alignment
        df_4h = self.fetch_ohlcv(symbol="BTCUSDT", interval="4h", limit=50)
        macro_trend = "NEUTRAL"
        if df_4h is not None and len(df_4h) >= 20:
            try:
                c4 = df_4h["close"].iloc[-1]
                ema20_4h = df_4h["close"].ewm(span=20).mean().iloc[-1]
                ema50_4h = df_4h["close"].ewm(span=50).mean().iloc[-1]
                if c4 > ema20_4h and ema20_4h > ema50_4h:
                    macro_trend = "BULL"
                elif c4 < ema20_4h and ema20_4h < ema50_4h:
                    macro_trend = "BEAR"
                elif c4 > ema20_4h:
                    macro_trend = "MILD_BULL"
                elif c4 < ema20_4h:
                    macro_trend = "MILD_BEAR"
            except Exception as e:
                log.warning(f"Error computing 4H macro trend: {e}")

        prices = df["close"].values
        hurst = self.calculate_hurst(prices)
        adx, plus_di, minus_di = self.calculate_adx(df)
        di_spread = plus_di - minus_di

        # Query prior regime for Schmitt Trigger hysteresis, plus the prior *mathematical*
        # label. The published regime can be an AI override, so using it to detect a
        # transition makes the transition test self-referential: an override to
        # DEFENSIVE_CHOP looks like a change every following cycle, and a genuine math
        # flip masked by a persistent override looks like no change at all.
        prior_regime = "NEUTRAL"
        prior_math_regime = "NEUTRAL"
        conn_prev = None
        try:
            db_ro_uri = f"file:{os.path.abspath(self.db_path).replace(os.sep, '/')}?mode=ro"
            conn_prev = sqlite3.connect(db_ro_uri, timeout=10.0, uri=True)
            try:
                r_prev = conn_prev.execute(
                    "SELECT regime, math_regime FROM market_regime ORDER BY id DESC LIMIT 1"
                ).fetchone()
            except sqlite3.OperationalError:
                # Pre-migration database without the additive math_regime column.
                r_prev = list(conn_prev.execute(
                    "SELECT regime FROM market_regime ORDER BY id DESC LIMIT 1"
                ).fetchone() or ()) + [None]
            if r_prev and r_prev[0]:
                prior_regime = str(r_prev[0])
                prior_math_regime = str(r_prev[1]) if len(r_prev) > 1 and r_prev[1] else str(r_prev[0])
        except Exception as prev_err:
            log.debug(f"[REGIME-PRIOR-NOTE] Could not read prior regime: {prev_err}")
        finally:
            if conn_prev:
                try:
                    conn_prev.close()
                except Exception:
                    pass

        # Directional Polarity & Schmitt Trigger Decision Matrix
        # The 20-25 ADX dead zone carries the previous *mathematical* trend forward. Using
        # the published label here would let an AI override (BULL_TREND -> DEFENSIVE_CHOP,
        # or a CHOP -> BULL_TREND promotion) masquerade as indicator memory it never had.
        if adx < 20.0:
            math_regime = "CHOP"
        elif 20.0 <= adx < 25.0:
            if prior_math_regime in ("BULL_TREND", "BULLISH") and di_spread > 5.0:
                math_regime = "BULL_TREND"
            elif prior_math_regime in ("BEAR_TREND", "BEARISH") and di_spread < -5.0:
                math_regime = "BEAR_TREND"
            elif di_spread > 8.0 and macro_trend in ("BULL", "MILD_BULL"):
                math_regime = "BULL_TREND"
            elif di_spread < -8.0 and macro_trend in ("BEAR", "MILD_BEAR"):
                math_regime = "BEAR_TREND"
            else:
                math_regime = "NEUTRAL"
        else:
            if di_spread >= 4.0:
                if macro_trend == "BEAR" and di_spread < 8.0:
                    math_regime = "NEUTRAL"
                else:
                    math_regime = "BULL_TREND"
            elif di_spread <= -4.0:
                math_regime = "BEAR_TREND"
            else:
                math_regime = "CHOP"

            if adx < 32.0 and abs(di_spread) < 8.0 and hurst < 0.40 and math_regime != "BEAR_TREND":
                math_regime = "CHOP"

        # Baseline Mathematical Leverage
        if math_regime == "BULL_TREND":
            math_leverage = 5
        elif math_regime == "BEAR_TREND":
            math_leverage = 2
        else:
            math_leverage = 3

        # -----------------------------------------------------------------------
        # AI Cognitive Regime Arbiter (15-min cadence or on regime transition)
        # -----------------------------------------------------------------------
        now_epoch = time.time()
        should_run_ai = (
            (now_epoch - self.last_ai_eval_ts >= self.ai_eval_interval_sec) or
            (math_regime != prior_math_regime and prior_math_regime != "NEUTRAL")
        )
        ai_is_fresh = False

        if should_run_ai:
            try:
                funding = self.fetch_funding_rate("BTCUSDT")
                c_last = df.iloc[-3:].copy()
                c_high = c_last["high"].values
                c_low = c_last["low"].values
                c_open = c_last["open"].values
                c_close = c_last["close"].values
                wick_tops = [(c_high[i] - max(c_open[i], c_close[i])) / (c_high[i] - c_low[i] + 1e-6) for i in range(len(c_last))]
                avg_upper_wick = float(np.mean(wick_tops))

                telemetry = {
                    "symbol": "BTC/USDT",
                    # fetch_ohlcv() discards still-forming bars, so this is the last completed
                    # 15m close. The as-of stamp stops the arbiter mistaking it for a live tick.
                    "price": float(df["close"].iloc[-1]),
                    "price_as_of_utc": self._bar_close_ts(df, "15m"),
                    "adx": round(float(adx), 2),
                    "plus_di": round(float(plus_di), 2),
                    "minus_di": round(float(minus_di), 2),
                    "di_spread": round(float(di_spread), 2),
                    "hurst": round(float(hurst), 4),
                    "math_regime": math_regime,
                    "macro_trend_4h": macro_trend,
                    "funding_rate": funding,
                    "avg_upper_wick_ratio_15m": round(avg_upper_wick, 3)
                }

                ai_res = self._query_ai_cognitive_arbiter(telemetry)
                # Only a stamped, contract-valid verdict is cached. The offline fallback and
                # any rejected payload leave the previous protective verdict in place rather
                # than silently replacing it with a permissive default.
                if isinstance(ai_res, dict) and ai_res.get("evaluated_at_utc"):
                    self.latest_ai_arbiter = ai_res
                    self.last_ai_eval_ts = now_epoch
                    ai_is_fresh = True
                else:
                    self.last_ai_eval_ts = now_epoch
                    log.warning(
                        "[AI-REGIME-ARBITER] ⚠️ No validated verdict this pass - holding "
                        f"{self.latest_ai_arbiter.get('ai_regime_bias')} from "
                        f"{self.latest_ai_arbiter.get('evaluated_at_utc') or 'baseline'}."
                    )
            except Exception as ai_e:
                log.warning(f"[AI-REGIME-ARBITER-NOTE] Error querying AI arbiter: {ai_e}")

        # Apply Cognitive Arbitration Rules with Hysteresis & Symmetric Veto.
        cached = self.latest_ai_arbiter if isinstance(self.latest_ai_arbiter, dict) else {}
        raw_bias = str(cached.get("ai_regime_bias") or "").strip().upper()
        incoming_bias = raw_bias if raw_bias in AI_REGIME_BIASES else "DEFENSIVE_CHOP"
        trap_risk = self._bounded_float(cached.get("trap_risk_score"), AI_TRAP_RISK_DEFAULT, 0.0, 1.0)
        ai_thesis = str(cached.get("structural_thesis") or "")[:AI_THESIS_MAX_CHARS]
        ai_lev_cap = self._bounded_int(cached.get("recommended_leverage_cap"), AI_LEVERAGE_DEFAULT, AI_LEVERAGE_ALLOWLIST)
        ai_evaluated_at = cached.get("evaluated_at_utc")
        ai_eval_epoch = self._bounded_float(cached.get("evaluated_at_epoch"), 0.0, 0.0, now_epoch + 1.0)

        ai_verdict_age_sec = (now_epoch - ai_eval_epoch) if ai_eval_epoch > 0.0 else None
        ai_verdict_usable = (ai_verdict_age_sec is not None) and (ai_verdict_age_sec <= self.ai_eval_interval_sec)

        # -----------------------------------------------------------------------
        # Recommendation 1: AI Bias Hysteresis Filter
        # Prevents 15m ping-pong flips. Only change active_ai_bias if:
        # (a) 2 consecutive matching incoming verdicts from the AI arbiter, OR
        # (b) Minimum dwell time of 45 minutes (2700s) has elapsed, OR
        # (c) First initialization (dwell_ts == 0.0)
        # -----------------------------------------------------------------------
        if ai_verdict_usable or ai_is_fresh:
            # Only count a verdict the first time it is seen
            is_new_verdict = (ai_eval_epoch > 0.0 and ai_eval_epoch != self.last_counted_ai_eval_epoch) or (ai_eval_epoch == 0.0 and self.last_counted_ai_eval_epoch is None)
            if is_new_verdict:
                self.last_counted_ai_eval_epoch = ai_eval_epoch
                if incoming_bias == self.active_ai_bias:
                    self.pending_ai_bias = None
                    self.pending_ai_bias_count = 0
                else:
                    if incoming_bias == self.pending_ai_bias:
                        self.pending_ai_bias_count += 1
                    else:
                        self.pending_ai_bias = incoming_bias
                        self.pending_ai_bias_count = 1

                    dwell_elapsed = now_epoch - self.ai_bias_dwell_ts
                    if self.pending_ai_bias_count >= 2 or dwell_elapsed >= self.min_ai_bias_dwell_sec or self.ai_bias_dwell_ts == 0.0:
                        old_b = self.active_ai_bias
                        self.active_ai_bias = incoming_bias
                        self.ai_bias_dwell_ts = now_epoch
                        self.pending_ai_bias = None
                        self.pending_ai_bias_count = 0
                        log.info(f"[AI-REGIME-HYSTERESIS] 🔄 AI Bias switched: {old_b} -> {incoming_bias} (dwell: {dwell_elapsed:.0f}s, confirmation: {self.pending_ai_bias_count})")
                    else:
                        log.info(f"[AI-REGIME-HYSTERESIS] ⏳ Pending switch {self.active_ai_bias} -> {incoming_bias} buffered (count: {self.pending_ai_bias_count}/2, dwell: {dwell_elapsed:.0f}s/{self.min_ai_bias_dwell_sec:.0f}s)")
            else:
                log.debug(f"[AI-REGIME-HYSTERESIS] Skipping counting step: verdict epoch {ai_eval_epoch} already counted.")
        elif not ai_verdict_usable and not self.latest_ai_arbiter.get("evaluated_at_utc"):
            # Unstamped/offline/poisoned baseline: reflect default directly
            self.active_ai_bias = incoming_bias

        ai_bias = self.active_ai_bias

        regime = math_regime
        recommended_leverage = math_leverage

        # -----------------------------------------------------------------------
        # Recommendation 2: Symmetric & Validated Veto
        # Only override mathematical BULL_TREND when:
        # - AI verdict is fresh and usable (non-stale)
        # - trap_risk >= 0.70 AND active AI bias confirms an active liquidity trap
        # Offline defaults or held/stale verdicts CANNOT veto a valid mathematical trend.
        # -----------------------------------------------------------------------
        if math_regime == "BULL_TREND":
            if ai_verdict_usable and trap_risk >= 0.70 and ai_bias in ("LIQUIDITY_HUNT_TRAP", "BEAR_DUMP"):
                regime = "DEFENSIVE_CHOP"
                recommended_leverage = min(recommended_leverage, ai_lev_cap, 3)
                log.warning(f"[AI-REGIME-ARBITER] 🛡️ VETO: Math signaled BULL_TREND, but fresh AI detected {ai_bias} (Trap Risk: {trap_risk:.2f} >= 0.70). Overriding to DEFENSIVE_CHOP ({recommended_leverage}x cap). Thesis: {ai_thesis}")
            else:
                recommended_leverage = min(5, ai_lev_cap)
        elif math_regime == "BEAR_TREND":
            regime = "BEAR_TREND"
            recommended_leverage = 2
            log.info(f"[AI-REGIME-ARBITER] 🛡️ BEAR_TREND confirmed: Preserving 100% USDT shield in super_bot (no counter-trend dip-buys).")
        elif math_regime == "CHOP":
            if ai_bias == "BULL_EXPANSION" and trap_risk < 0.25 and di_spread > 6.0 and macro_trend in ("BULL", "MILD_BULL"):
                if not ai_verdict_usable:
                    log.warning(
                        f"[AI-REGIME-ARBITER] 🚫 PROMOTION SUPPRESSED: BULL_EXPANSION verdict is "
                        f"{'untimestamped' if ai_verdict_age_sec is None else f'{ai_verdict_age_sec:.0f}s old'} "
                        f"(cadence {self.ai_eval_interval_sec:.0f}s) - staying on mathematical CHOP."
                    )
                else:
                    regime = "BULL_TREND"
                    recommended_leverage = ai_lev_cap
                    log.info(f"[AI-REGIME-ARBITER] 🚀 PROMOTION: Math indicated CHOP, but AI detected early institutional BULL_EXPANSION (Trap Risk: {trap_risk:.2f}, verdict {ai_verdict_age_sec:.0f}s old). Promoting to BULL_TREND.")

        # -----------------------------------------------------------------------
        # FinRL Mahalanobis Cross-Asset Turbulence Index & Two-Tier Circuit Breaker
        # -----------------------------------------------------------------------
        turb_info = self.calculate_mahalanobis_turbulence()
        turb_idx = self._bounded_float(turb_info.get("turbulence_index"), 0.0, 0.0, 1e9)
        turb_thresh = self._bounded_float(turb_info.get("active_threshold"), 22.0, 0.01, 1e9)
        turb_state = str(turb_info.get("turbulence_state") or "CALM_NORMAL")
        turb_breach = bool(turb_info.get("turbulence_breach", False))

        if turb_breach:
            recommended_leverage = min(recommended_leverage, 2)
            if regime == "BULL_TREND":
                regime = "DEFENSIVE_CHOP"
            log.warning(
                f"[FINRL-TURBULENCE-BREACH] 🚨 Cross-Asset Turbulence d_t={turb_idx:.2f} > Active Threshold={turb_thresh:.2f} "
                f"(State: {turb_state}). Capping leverage to {recommended_leverage}x and activating Turbulence Shield!"
            )

        # -----------------------------------------------------------------------
        # Bayesian Markov Chain & Regime Transition Engine
        # -----------------------------------------------------------------------
        markov_telemetry = {}
        if getattr(self, "markov_engine", None):
            try:
                markov_telemetry = self.markov_engine.process_bars(df)
            except Exception as me_err:
                log.warning(f"[MARKOV-REGIME-ERROR] Failed to process bars: {me_err}")

        timestamp = datetime.now(UTC).isoformat()
        source_close_ts = self._bar_close_ts(df, "15m")
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=60.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=60000;")
            with conn:
                # Append-only history: every classification is a new row. Readers that want the
                # current gate use ORDER BY id/rowid DESC LIMIT 1, which is unchanged by this.
                conn.execute("""
                    INSERT INTO market_regime (
                        timestamp, source_close_ts, regime, math_regime, hurst, adx, plus_di, minus_di,
                        macro_trend, ai_bias, trap_risk, ai_thesis,
                        turbulence_index, turbulence_threshold, turbulence_state
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp, source_close_ts, regime, math_regime, hurst, adx, plus_di, minus_di,
                    macro_trend, ai_bias, trap_risk, ai_thesis,
                    turb_idx, turb_thresh, turb_state
                ))
                conn.execute(
                    "DELETE FROM market_regime WHERE id <= (SELECT MAX(id) - ? FROM market_regime)",
                    (REGIME_HISTORY_ROWS,),
                )
                conn.execute("""
                    INSERT OR REPLACE INTO system_state (key, value, updated_at)
                    VALUES ('recommended_leverage', ?, ?)
                """, (str(recommended_leverage), timestamp))
                conn.execute("""
                    INSERT OR REPLACE INTO system_state (key, value, updated_at)
                    VALUES ('market_regime_label', ?, ?)
                """, (regime, timestamp))
                conn.execute("""
                    INSERT OR REPLACE INTO system_state (key, value, updated_at)
                    VALUES ('regime_adx', ?, ?)
                """, (f"{adx:.2f}", timestamp))
                conn.execute("""
                    INSERT OR REPLACE INTO system_state (key, value, updated_at)
                    VALUES ('regime_hurst', ?, ?)
                """, (f"{hurst:.2f}", timestamp))
                conn.execute("""
                    INSERT OR REPLACE INTO system_state (key, value, updated_at)
                    VALUES ('market_regime_di_spread', ?, ?)
                """, (f"{di_spread:+.2f}", timestamp))
                conn.execute("""
                    INSERT OR REPLACE INTO system_state (key, value, updated_at)
                    VALUES ('macro_trend_4h', ?, ?)
                """, (macro_trend, timestamp))
                conn.execute("""
                    INSERT OR REPLACE INTO system_state (key, value, updated_at)
                    VALUES ('ai_regime_bias', ?, ?)
                """, (ai_bias, timestamp))
                conn.execute("""
                    INSERT OR REPLACE INTO system_state (key, value, updated_at)
                    VALUES ('ai_regime_trap_risk', ?, ?)
                """, (f"{trap_risk:.2f}", timestamp))
                conn.execute("""
                    INSERT OR REPLACE INTO system_state (key, value, updated_at)
                    VALUES ('ai_regime_thesis', ?, ?)
                """, (ai_thesis, timestamp))
                conn.execute("""
                    INSERT OR REPLACE INTO system_state (key, value, updated_at)
                    VALUES ('ai_regime_updated_at', ?, ?)
                """, (timestamp, timestamp))
                # The publication stamp above says when this row was written; these two say
                # when a model actually produced the verdict being applied, so a cached
                # arbiter answer can never be mistaken for a live one (or survive a restart).
                conn.execute("""
                    INSERT OR REPLACE INTO system_state (key, value, updated_at)
                    VALUES ('ai_regime_last_eval_utc', ?, ?)
                """, (ai_evaluated_at or "", timestamp))
                conn.execute("""
                    INSERT OR REPLACE INTO system_state (key, value, updated_at)
                    VALUES ('ai_regime_eval_age_sec', ?, ?)
                """, (f"{ai_verdict_age_sec:.0f}" if ai_verdict_age_sec is not None else "-1", timestamp))
                conn.execute("""
                    INSERT OR REPLACE INTO system_state (key, value, updated_at)
                    VALUES ('finrl_turbulence_json', ?, ?)
                """, (json.dumps(turb_info), timestamp))
                if markov_telemetry:
                    conn.execute("""
                        INSERT OR REPLACE INTO system_state (key, value, updated_at)
                        VALUES ('markov_regime_telemetry', ?, ?)
                    """, (json.dumps(markov_telemetry), timestamp))
            
            log.info(
                f"[REGIME] Final Regime: {regime} (Math: {math_regime}, AI Bias: {ai_bias}, Trap Risk: {trap_risk:.2f}). "
                f"ADX={adx:.2f} | Spread={di_spread:+.1f} | Turbulence={turb_idx:.2f}/{turb_thresh:.2f} ({turb_state}) | Lev={recommended_leverage}x"
            )
            return True
        except Exception as e:
            log.error(f"Failed to commit regime to database: {e}")
            return False
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

if __name__ == "__main__":
    import time
    classifier = RegimeClassifier(db_path="trading_data.db")
    while True:
        log.info("--- REGIME CLASSIFIER starting new update cycle ---")
        try:
            classifier.classify_and_save()
        except Exception as cycle_err:
            # The bot reads market_regime to gate entries. Letting one bad cycle kill this
            # process freezes that row at whatever it last was; retrying keeps it moving and
            # any genuinely broken input still surfaces in the log every 5 minutes.
            log.error(f"[REGIME-CYCLE] Classification cycle failed, will retry next cycle: {cycle_err}", exc_info=True)
        log.info("--- REGIME CLASSIFIER update complete. Sleeping for 5 minutes... ---")
        time.sleep(300)

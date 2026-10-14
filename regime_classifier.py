# regime_classifier.py — Quantitative Regime Classifier Agent

import sqlite3
import requests
import pandas as pd
import numpy as np
import logging
import os
import re
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
            "trap_risk_score": 0.50,
            "recommended_leverage_cap": 3,
            "structural_thesis": "Baseline cognitive regime initialization.",
            "override_math_regime": False
        }
        self._init_db()

    def _init_db(self):
        """Ensures the market_regime history table exists in trading_data.db with AI columns."""
        try:
            conn = sqlite3.connect(self.db_path)
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
            conn.close()
        except Exception as e:
            log.critical(f"Failed to initialize database table: {e}")
            raise

    def fetch_ohlcv(self, symbol="BTCUSDT", interval="15m", limit=105):
        """Pulls OHLCV bars directly from the public Bitget USDT-FUTURES API."""
        clean_sym = symbol.replace("/", "").replace("-", "").upper()
        url = f"https://api.bitget.com/api/v2/mix/market/candles?symbol={clean_sym}&granularity={interval}&limit={limit}&productType=USDT-FUTURES"
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
                    return df
        except Exception as e:
            log.error(f"Failed to fetch Bitget historical bars: {e}")
            return None

    @staticmethod
    def _bar_close_ts(df, interval):
        """ISO-8601 UTC close time of the newest bar in a Bitget OHLCV frame.

        Bitget returns `open_time` in milliseconds; the close time is open_time plus the
        bar duration. Returns None (stored as NULL) rather than a guessed value.
        """
        try:
            match = re.fullmatch(r"(\d+)\s*([mhd])", str(interval).strip().lower())
            if not match:
                raise ValueError(f"unparsable granularity '{interval}'")
            seconds = int(match.group(1)) * {"m": 60, "h": 3600, "d": 86400}[match.group(2)]
            last_open_ms = int(float(df["open_time"].iloc[-1]))
            return datetime.fromtimestamp((last_open_ms + seconds * 1000) / 1000.0, UTC).isoformat()
        except Exception as e:
            log.warning(f"Could not derive source bar close time from '{interval}' bars: {e}")
            return None

    def calculate_hurst(self, prices):
        """Estimates Hurst Exponent over prices using Rescaled Range (R/S) Analysis."""
        if len(prices) < 50:
            return 0.50
        try:
            # Convert series to logarithmic returns
            returns = np.diff(np.log(prices))
            N = len(returns)
            
            lags = [10, 20, 30, 40, 50]
            valid_pairs = []
            
            for lag in lags:
                num_subsets = N // lag
                if num_subsets == 0:
                    continue
                rs_sub = []
                for i in range(num_subsets):
                    sub = returns[i*lag : (i+1)*lag]
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
            hist_pool = turb_arr[:-1] if len(turb_arr) > 5 else turb_arr

            tau_90 = float(np.percentile(hist_pool, 90))
            tau_99 = float(np.percentile(hist_pool, 99))
            trailing_12 = float(np.mean(turb_arr[-min(12, len(turb_arr)):]))

            # FinRL Two-Tier Dynamic Quantile Rule:
            # When trailing turbulence exceeds the 90th percentile, tighten active circuit-breaker to tau_90
            active_thresh = tau_90 if trailing_12 > tau_90 else tau_99
            active_thresh = max(active_thresh, 1e-4)

            # FinRL bounded sigmoid compression
            raw_sig = 1.0 / (1.0 + float(np.exp(-current_d * np.e / active_thresh)))
            sigmoid_score = round(max(0.0, min(1.0, (raw_sig - 0.5) * 2.0)), 4)
            pct_rank = round(float(np.mean(hist_pool <= current_d) * 100.0), 1)

            breach = bool(current_d > active_thresh)
            if breach:
                state_str = "EXTREME_SHOCK"
            elif current_d > tau_90:
                state_str = "ELEVATED_STRESS"
            else:
                state_str = "CALM_NORMAL"

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
            }
        except Exception as e:
            log.warning(f"[FINRL-TURBULENCE-NOTE] Error calculating Mahalanobis turbulence: {e}")
            return default_res

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

    def _query_ai_cognitive_arbiter(self, telemetry: dict) -> dict:
        """
        Dispatches a compact prompt to Google Gemini (or Groq failover pool)
        to evaluate whether current momentum is genuine trend expansion or a liquidity trap.
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
                            parsed = json.loads(text)
                            log.info(f"[AI-REGIME-ARBITER] Gemini ({model}) synthesis: {parsed.get('ai_regime_bias')} (Trap Risk: {parsed.get('trap_risk_score')})")
                            return parsed
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
                            parsed_g = json.loads(text_g)
                            log.info(f"[AI-REGIME-ARBITER] Groq ({g_model}) synthesis: {parsed_g.get('ai_regime_bias')} (Trap Risk: {parsed_g.get('trap_risk_score')})")
                            return parsed_g
                except Exception as gr_err:
                    log.debug(f"[AI-REGIME-ARBITER] Groq ({g_model}) error: {gr_err}")

        return {
            "ai_regime_bias": "MATH_DEFAULT",
            "trap_risk_score": 0.35,
            "recommended_leverage_cap": 3,
            "structural_thesis": "AI offline — defaulting to deterministic mathematical regime rules.",
            "override_math_regime": False
        }

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

        # Query prior regime for Schmitt Trigger hysteresis
        prior_regime = "NEUTRAL"
        try:
            conn_prev = sqlite3.connect(self.db_path)
            r_prev = conn_prev.execute("SELECT regime FROM market_regime ORDER BY id DESC LIMIT 1").fetchone()
            conn_prev.close()
            if r_prev and r_prev[0]:
                prior_regime = str(r_prev[0])
        except Exception:
            pass

        # Directional Polarity & Schmitt Trigger Decision Matrix
        if adx < 20.0:
            math_regime = "CHOP"
        elif 20.0 <= adx < 25.0:
            if prior_regime in ("BULL_TREND", "BULLISH") and di_spread > 5.0:
                math_regime = "BULL_TREND"
            elif prior_regime in ("BEAR_TREND", "BEARISH") and di_spread < -5.0:
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
            (math_regime != prior_regime and prior_regime != "NEUTRAL")
        )

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
                    "price": float(df["close"].iloc[-1]),
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
                if ai_res and isinstance(ai_res, dict):
                    self.latest_ai_arbiter = ai_res
                    self.last_ai_eval_ts = now_epoch
            except Exception as ai_e:
                log.warning(f"[AI-REGIME-ARBITER-NOTE] Error querying AI arbiter: {ai_e}")

        # Apply Cognitive Arbitration Rules
        ai_bias = self.latest_ai_arbiter.get("ai_regime_bias", "DEFENSIVE_CHOP")
        trap_risk = float(self.latest_ai_arbiter.get("trap_risk_score", 0.5))
        ai_thesis = str(self.latest_ai_arbiter.get("structural_thesis", ""))
        ai_lev_cap = int(self.latest_ai_arbiter.get("recommended_leverage_cap", 3))

        regime = math_regime
        recommended_leverage = math_leverage

        if math_regime == "BULL_TREND":
            if trap_risk >= 0.60 or ai_bias in ("LIQUIDITY_HUNT_TRAP", "DEFENSIVE_CHOP"):
                regime = "DEFENSIVE_CHOP"
                recommended_leverage = min(recommended_leverage, ai_lev_cap, 3)
                log.warning(f"[AI-REGIME-ARBITER] 🛡️ VETO: Math signaled BULL_TREND, but AI detected {ai_bias} (Trap Risk: {trap_risk:.2f}). Overriding to DEFENSIVE_CHOP ({recommended_leverage}x cap). Thesis: {ai_thesis}")
            else:
                recommended_leverage = min(5, ai_lev_cap)
        elif math_regime == "BEAR_TREND":
            regime = "BEAR_TREND"
            recommended_leverage = 2
            log.info(f"[AI-REGIME-ARBITER] 🛡️ BEAR_TREND confirmed: Preserving 100% USDT shield in super_bot (no counter-trend dip-buys).")
        elif math_regime == "CHOP":
            if ai_bias == "BULL_EXPANSION" and trap_risk < 0.25 and di_spread > 6.0 and macro_trend in ("BULL", "MILD_BULL"):
                regime = "BULL_TREND"
                recommended_leverage = ai_lev_cap
                log.info(f"[AI-REGIME-ARBITER] 🚀 PROMOTION: Math indicated CHOP, but AI detected early institutional BULL_EXPANSION (Trap Risk: {trap_risk:.2f}). Promoting to BULL_TREND.")

        # -----------------------------------------------------------------------
        # FinRL Mahalanobis Cross-Asset Turbulence Index & Two-Tier Circuit Breaker
        # -----------------------------------------------------------------------
        turb_info = self.calculate_mahalanobis_turbulence()
        turb_idx = float(turb_info.get("turbulence_index", 0.0))
        turb_thresh = float(turb_info.get("active_threshold", 22.0))
        turb_state = str(turb_info.get("turbulence_state", "CALM_NORMAL"))
        turb_breach = bool(turb_info.get("turbulence_breach", False))

        if turb_breach:
            recommended_leverage = min(recommended_leverage, 2)
            if regime == "BULL_TREND":
                regime = "DEFENSIVE_CHOP"
            log.warning(
                f"[FINRL-TURBULENCE-BREACH] 🚨 Cross-Asset Turbulence d_t={turb_idx:.2f} > Active Threshold={turb_thresh:.2f} "
                f"(State: {turb_state}). Capping leverage to {recommended_leverage}x and activating Turbulence Shield!"
            )

        timestamp = datetime.now(UTC).isoformat()
        source_close_ts = self._bar_close_ts(df, "15m")
        try:
            conn = sqlite3.connect(self.db_path)
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
                conn.execute("""
                    INSERT OR REPLACE INTO system_state (key, value, updated_at)
                    VALUES ('finrl_turbulence_json', ?, ?)
                """, (json.dumps(turb_info), timestamp))
            conn.close()
            
            log.info(
                f"[REGIME] Final Regime: {regime} (Math: {math_regime}, AI Bias: {ai_bias}, Trap Risk: {trap_risk:.2f}). "
                f"ADX={adx:.2f} | Spread={di_spread:+.1f} | Turbulence={turb_idx:.2f}/{turb_thresh:.2f} ({turb_state}) | Lev={recommended_leverage}x"
            )
            return True
        except Exception as e:
            log.error(f"Failed to commit regime to database: {e}")
            return False

if __name__ == "__main__":
    import time
    classifier = RegimeClassifier(db_path="trading_data.db")
    while True:
        log.info("--- REGIME CLASSIFIER starting new update cycle ---")
        classifier.classify_and_save()
        log.info("--- REGIME CLASSIFIER update complete. Sleeping for 5 minutes... ---")
        time.sleep(300)

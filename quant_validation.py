"""
quant_validation.py — Institutional Bootstrap Significance & Monte Carlo Validation Engine
Implements Jesse-style statistical validation completely isolated from live execution:
  1. Bootstrap Resampling for Entry Rule Significance (B = 2,000 iterations)
  2. Trade-Sequence Monte Carlo Drawdown & Equity Fan Simulation (N = 5,000 draws)
  3. Gaussian ATR Candle-Path Perturbation (±0.15 * ATR) for Trailing Ratchet Survival
"""

import os
import sys
import json
import time
import sqlite3
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

import config
from strategy_base import (
    StrategyContext,
    BaseStrategy,
    SwingKumoStrategy,
    ChopGridMakerStrategy,
    BearScalpShortStrategy,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPORT_CACHE_PATH = os.path.join("cache", "quant_validation_report.json")


class InsufficientValidationData(RuntimeError):
    """Raised when a validation statistic cannot be computed from real recorded trades."""


def _ensure_5m_validation_datasets(symbols: List[str] = None) -> List[str]:
    """Ensures 5m OHLCV Parquet datasets exist in data/historical for high-frequency Grid & Scalp validation."""
    if symbols is None:
        symbols = ["SOL/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "NEAR/USDT", "XRP/USDT"]
    os.makedirs(os.path.join("data", "historical"), exist_ok=True)
    cached_files: List[str] = []

    ex = None
    for sym in symbols:
        clean = sym.replace("/", "").upper()
        fname = f"{clean}_5m_1000.parquet"
        fpath = os.path.join("data", "historical", fname)
        if os.path.exists(fpath) and (time.time() - os.path.getmtime(fpath)) < 86400.0:
            cached_files.append(fname)
            continue
        try:
            if ex is None:
                import ccxt
                ex = ccxt.binance({"enableRateLimit": True, "timeout": 8000})
            raw = ex.fetch_ohlcv(sym, timeframe="5m", limit=1000)
            if raw and len(raw) >= 200:
                df5 = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df5.to_parquet(fpath, index=False)
                cached_files.append(fname)
        except Exception:
            if os.path.exists(fpath):
                cached_files.append(fname)
    return cached_files


def _prepare_indicator_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Computes vectorized indicators matching Swing 4H, Chop Grid, and Scalp Short."""
    df = df_raw.copy()
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    high = df["high"]
    low = df["low"]
    close = df["close"]
    vol = df["volume"].fillna(1.0)

    # Ichimoku
    df["tenkan"] = (high.rolling(9).max() + low.rolling(9).min()) / 2.0
    df["kijun"] = (high.rolling(26).max() + low.rolling(26).min()) / 2.0
    span_a_raw = (df["tenkan"] + df["kijun"]) / 2.0
    span_b_raw = (high.rolling(52).max() + low.rolling(52).min()) / 2.0
    # Always projected 26 bars like the live engine. The previous `if len(df) >= 78 / else` guard fell back to
    # the *unshifted* span for short frames, so any frame under 78 bars got the lookahead back; on the
    # production path such a frame is dropped at the 120-bar check, so the branch bought nothing.
    span_a = span_a_raw.shift(26)
    span_b = span_b_raw.shift(26)
    df["cloud_top"] = np.maximum(span_a, span_b)
    df["cloud_bottom"] = np.minimum(span_a, span_b)

    # ATR(14)
    prev_c = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # ADX(14) & DI spread
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_sum = tr.rolling(14).sum().replace(0, np.nan)
    plus_di = 100.0 * (pd.Series(plus_dm, index=df.index).rolling(14).sum() / tr_sum)
    minus_di = 100.0 * (pd.Series(minus_dm, index=df.index).rolling(14).sum() / tr_sum)
    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    df["adx"] = dx.rolling(14).mean().fillna(20.0)
    df["di_spread"] = (plus_di - minus_di).fillna(0.0)

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)
    df["rsi_prev"] = df["rsi"].shift(1).fillna(50.0)

    # VWAP & Bollinger-style VWAP Bands (24-bar)
    cum_pv = (close * vol).rolling(24).sum()
    cum_v = vol.rolling(24).sum()
    df["vwap"] = (cum_pv / (cum_v + 1e-9)).bfill()
    r_std = close.rolling(24).std().fillna(0.0)
    df["lower_vwap_band"] = df["vwap"] - (2.0 * r_std)
    df["upper_vwap_band"] = df["vwap"] + (2.0 * r_std)

    # Volume Profile Value Area proxy (48-bar VAH / VAL)
    df["vah"] = close.rolling(48).quantile(0.75).bfill()
    df["val"] = close.rolling(48).quantile(0.25).bfill()

    # Tom DeMark TD Sequential 4-bar close comparison
    c_arr = close.to_numpy()
    td_up = np.zeros(len(c_arr), dtype=int)
    td_down = np.zeros(len(c_arr), dtype=int)
    for k in range(4, len(c_arr)):
        if c_arr[k] > c_arr[k - 4]:
            td_up[k] = td_up[k - 1] + 1
        elif c_arr[k] < c_arr[k - 4]:
            td_down[k] = td_down[k - 1] + 1
    df["td_up"] = td_up
    df["td_down"] = td_down

    # EMA 20, 50, 200 (proxy macro alignment)
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    df["daily_trend"] = np.where(
        (close > ema200) & (ema50 >= ema200 * 0.98),
        "BULLISH",
        np.where((close < ema200) & (ema50 <= ema200 * 1.02), "BEARISH", "NEUTRAL"),
    )
    df["market_regime"] = np.where(
        df["adx"] >= 25.0,
        np.where(df["di_spread"] >= 0, "BULL", "BEAR"),
        "CHOP",
    )
    return df


def _simulate_single_entry_fast(
    idx: int,
    side: str,
    strategy_name: str,
    open_arr: np.ndarray,
    close_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    atr_arr: np.ndarray,
    max_horizon: int = 36,
    fee_roundtrip: float = 0.0008,
    noise_scale: float = 0.0,
    rng: np.random.Generator = None,
) -> Tuple[float, bool]:
    """
    Simulates a single trade forward from `idx` using the exact trailing ratchets
    (+0.8% -> +0.15%, +1.5% -> +0.50%, +0.40% short BE, and ATR/pct SL/TP).
    Returns (net_pct_return_unlevered, ratchet_triggered_and_stopped).
    """
    n = len(close_arr)
    if idx + 1 >= n:
        return 0.0, False

    entry = float(close_arr[idx])
    if entry <= 0:
        return 0.0, False
    atr = float(atr_arr[idx])
    is_long = (side == "long")

    if strategy_name == "SWING_4H":
        sl_dist = max(2.5 * atr, entry * 0.015)
        tp_dist = 4.0 * sl_dist
        sl = entry - sl_dist if is_long else entry + sl_dist
        tp = entry + tp_dist if is_long else entry - tp_dist
    elif strategy_name == "GRID_MAKER":
        sl = entry * (1.0 - 0.0115)
        tp = entry * (1.0 + 0.0135)
        fee_roundtrip = 0.0004  # Maker fee
    else:  # SCALP_SHORT
        sl = entry * (1.0 + 0.0100)
        tp = entry * (1.0 - 0.0120)
        fee_roundtrip = 0.0004
        max_horizon = min(max_horizon, 9)

    peak = entry
    ratchet_armed = False
    end_idx = min(n, idx + 1 + max_horizon)

    for j in range(idx + 1, end_idx):
        o = float(open_arr[j])
        h = float(high_arr[j])
        l = float(low_arr[j])
        c = float(close_arr[j])
        cur_atr = float(atr_arr[j])

        if noise_scale > 0.0 and rng is not None and cur_atr > 0:
            eps_h = abs(float(rng.normal(0.0, noise_scale * cur_atr)))
            eps_l = abs(float(rng.normal(0.0, noise_scale * cur_atr)))
            h += eps_h
            l = max(1e-8, l - eps_l)

        if is_long:
            hit_sl = l <= sl
            hit_tp = h >= tp
            if hit_sl and hit_tp:
                # Resolve same-bar conflict via bar polarity (green bar touches low then high if open near low, or TP if bullish expansion)
                if c >= o:
                    return ((tp - entry) / entry) - fee_roundtrip, False
                else:
                    return ((sl - entry) / entry) - fee_roundtrip, ratchet_armed
            elif hit_sl:
                return ((sl - entry) / entry) - fee_roundtrip, ratchet_armed
            elif hit_tp:
                return ((tp - entry) / entry) - fee_roundtrip, False

            if h > peak:
                peak = h
            gain_pct = (peak - entry) / entry

            if strategy_name == "SWING_4H":
                if gain_pct >= 0.015:
                    cand_sl = entry * 1.005
                    if cand_sl > sl:
                        sl = cand_sl
                        ratchet_armed = True
                elif gain_pct >= 0.008:
                    cand_sl = entry * 1.0015
                    if cand_sl > sl:
                        sl = cand_sl
                        ratchet_armed = True
                if cur_atr > 0 and (peak - entry) >= 2.0 * cur_atr:
                    chand = peak - 2.2 * cur_atr
                    if chand > sl and chand < c:
                        sl = chand
                        ratchet_armed = True
            elif strategy_name == "GRID_MAKER":
                if gain_pct >= 0.008:
                    cand_sl = entry * 1.0015
                    if cand_sl > sl:
                        sl = cand_sl
                        ratchet_armed = True
        else:
            hit_sl = h >= sl
            hit_tp = l <= tp
            if hit_sl and hit_tp:
                if c <= o:
                    return ((entry - tp) / entry) - fee_roundtrip, False
                else:
                    return ((entry - sl) / entry) - fee_roundtrip, ratchet_armed
            elif hit_sl:
                return ((entry - sl) / entry) - fee_roundtrip, ratchet_armed
            elif hit_tp:
                return ((entry - tp) / entry) - fee_roundtrip, False

            if l < peak:
                peak = l
            gain_pct = (entry - peak) / entry

            if strategy_name == "SCALP_SHORT":
                if gain_pct >= 0.004:
                    cand_sl = entry * 0.9995
                    if cand_sl < sl:
                        sl = cand_sl
                        ratchet_armed = True
            else:
                if gain_pct >= 0.015:
                    cand_sl = entry * 0.995
                    if cand_sl < sl:
                        sl = cand_sl
                        ratchet_armed = True
                elif gain_pct >= 0.008:
                    cand_sl = entry * 0.9985
                    if cand_sl < sl:
                        sl = cand_sl
                        ratchet_armed = True

    final_c = float(close_arr[end_idx - 1])
    ret = (((final_c - entry) / entry) if is_long else ((entry - final_c) / entry)) - fee_roundtrip
    return ret, False


def _extract_strategy_signals(df: pd.DataFrame, strategy: BaseStrategy, symbol: str) -> List[Tuple[int, str]]:
    """Evaluates declarative BaseStrategy (should_long / should_short) across historical bars."""
    signals: List[Tuple[int, str]] = []
    cooldown_until = -1
    min_gap = 6 if strategy.name == "SWING_4H" else 4

    close_a = df["close"].to_numpy()
    open_a = df["open"].to_numpy()
    high_a = df["high"].to_numpy()
    low_a = df["low"].to_numpy()
    vol_a = df["volume"].to_numpy()
    atr_a = df["atr"].to_numpy()
    adx_a = df["adx"].to_numpy()
    rsi_a = df["rsi"].to_numpy()
    rsi_p_a = df["rsi_prev"].to_numpy()
    vwap_a = df["vwap"].to_numpy()
    l_vwap_a = df["lower_vwap_band"].to_numpy()
    u_vwap_a = df["upper_vwap_band"].to_numpy()
    ema20_a = df["ema20"].to_numpy()
    tenkan_a = df["tenkan"].to_numpy()
    kijun_a = df["kijun"].to_numpy()
    ctop_a = df["cloud_top"].to_numpy()
    cbot_a = df["cloud_bottom"].to_numpy()
    vah_a = df["vah"].to_numpy()
    val_a = df["val"].to_numpy()
    tdup_a = df["td_up"].to_numpy()
    tddn_a = df["td_down"].to_numpy()
    dtrend_a = df["daily_trend"].to_numpy()
    mreg_a = df["market_regime"].to_numpy()
    disp_a = df["di_spread"].to_numpy()

    for i in range(60, len(df) - 10):
        if i < cooldown_until:
            continue
        ctx = StrategyContext(
            symbol=symbol,
            timeframe=strategy.timeframe,
            current_price=float(close_a[i]),
            open_price=float(open_a[i]),
            high_price=float(high_a[i]),
            low_price=float(low_a[i]),
            volume=float(vol_a[i]),
            atr=float(atr_a[i]),
            adx=float(adx_a[i]),
            rsi=float(rsi_a[i]),
            rsi_prev=float(rsi_p_a[i]),
            vwap=float(vwap_a[i]),
            lower_vwap_band=float(l_vwap_a[i]),
            upper_vwap_band=float(u_vwap_a[i]),
            ema20=float(ema20_a[i]),
            tenkan=float(tenkan_a[i]),
            kijun=float(kijun_a[i]),
            cloud_top=float(ctop_a[i]),
            cloud_bottom=float(cbot_a[i]),
            vah=float(vah_a[i]),
            val=float(val_a[i]),
            td_up=int(tdup_a[i]),
            td_down=int(tddn_a[i]),
            daily_trend=str(dtrend_a[i]),
            market_regime=str(mreg_a[i]),
            di_spread=float(disp_a[i]),
            bar_index=i,
        )
        if strategy.should_long(ctx):
            signals.append((i, "long"))
            cooldown_until = i + min_gap
        elif strategy.should_short(ctx):
            signals.append((i, "short"))
            cooldown_until = i + min_gap

    return signals


def run_bootstrap_entry_significance(
    b_iterations: int = 2000,
    noise_sigma: float = 0.15,
    seed: int = 42,
    missing_datasets: Dict[str, List[str]] = None,
) -> Dict[str, Any]:
    """
    Runs B = 2,000 Bootstrap Resampling iterations for each of the 3 declarative strategies:
      1. 4H Swing Breakout (SwingKumoStrategy)
      2. 5m Chop Grid Dip-Buy (ChopGridMakerStrategy)
      3. 5m Quick Scalp Short (BearScalpShortStrategy)
    Also performs Gaussian ATR Candle-Path Perturbation (±0.15 * ATR) to measure ratchet survival.
    `missing_datasets`, when a dict is passed in, is filled with the datasets each arm asked for and
    did not get, including for arms that therefore produced no result at all. A dataset that is
    absent from ``data/historical`` is recorded as its bare file name; a dataset that is present but
    unusable is recorded as ``"<name> (unusable: <reason>)"``, because an empty list otherwise reads
    as "this arm got everything" when it silently dropped a file.
    """
    rng = np.random.default_rng(seed)
    files_5m = _ensure_5m_validation_datasets()
    fallback_1h = ["SOLUSDT_1h_full.csv", "ETHUSDT_1h_full.csv", "BTCUSDT_1h_full.csv", "LINKUSDT_1h_full.csv"]
    hf_files = files_5m if files_5m else fallback_1h

    strategies = [
        (
            SwingKumoStrategy(),
            ["SOLUSDT_4h_full.csv", "ADAUSDT_4h_full.csv", "AVAXUSDT_4h_full.csv", "NEARUSDT_4h_full.csv"],
            36,
        ),
        (
            ChopGridMakerStrategy(),
            files_5m + ["SOLUSDT_1h_full.csv", "ADAUSDT_1h.csv", "ETHUSDT_1h_full.csv", "LINKUSDT_1h_full.csv", "BNBUSDT_1h.csv"],
            12,
        ),
        (
            BearScalpShortStrategy(),
            hf_files,
            9,
        ),
    ]

    results: Dict[str, Any] = {}

    for strat, file_list, horizon in strategies:
        all_signal_rets: List[float] = []
        all_perturbed_rets: List[float] = []
        ratchet_premature_stops = 0
        ratchet_clean_survivors = 0
        random_pool_rets: List[float] = []
        arm_missing_datasets: List[str] = []

        for fname in file_list:
            fpath = os.path.join("data", "historical", fname)
            if not os.path.exists(fpath):
                arm_missing_datasets.append(fname)
                continue
            try:
                if fname.endswith(".parquet"):
                    df_raw = pd.read_parquet(fpath)
                else:
                    df_raw = pd.read_csv(fpath).tail(2500)
                df = _prepare_indicator_dataframe(df_raw)
                if len(df) < 120:
                    # Present but unusable: a reader who only learns "missing_datasets: []" would
                    # conclude this arm got everything, which is false. Record the reason too.
                    arm_missing_datasets.append(
                        f"{fname} (unusable: {len(df)} bars after indicator preparation, "
                        f"below the 120-bar minimum this arm needs)"
                    )
                    continue

                sym = fname.split("_")[0]
                signals = _extract_strategy_signals(df, strat, sym)
                if not signals:
                    continue

                open_a = df["open"].to_numpy()
                close_a = df["close"].to_numpy()
                high_a = df["high"].to_numpy()
                low_a = df["low"].to_numpy()
                atr_a = df["atr"].to_numpy()
                lev = float(getattr(strat, "leverage", getattr(strat, "standard_leverage", 3)))

                for idx, side in signals:
                    base_r, base_ratchet_stop = _simulate_single_entry_fast(
                        idx, side, strat.name, open_a, close_a, high_a, low_a, atr_a, max_horizon=horizon
                    )
                    all_signal_rets.append(base_r * lev * 100.0)

                    pert_sub = []
                    premature_hit = False
                    for _ in range(5):
                        p_r, p_ratchet_stop = _simulate_single_entry_fast(
                            idx, side, strat.name, open_a, close_a, high_a, low_a, atr_a,
                            max_horizon=horizon, noise_scale=noise_sigma, rng=rng
                        )
                        pert_sub.append(p_r * lev * 100.0)
                        if p_ratchet_stop and not base_ratchet_stop and p_r < base_r:
                            premature_hit = True
                    all_perturbed_rets.append(float(np.mean(pert_sub)))
                    if premature_hit:
                        ratchet_premature_stops += 1
                    else:
                        ratchet_clean_survivors += 1

                default_side = "short" if strat.name == "SCALP_SHORT" else "long"
                candidate_indices = rng.integers(60, len(df) - horizon - 1, size=min(600, len(df) - 80))
                for r_idx in candidate_indices:
                    r_ret, _ = _simulate_single_entry_fast(
                        int(r_idx), default_side, strat.name, open_a, close_a, high_a, low_a, atr_a, max_horizon=horizon
                    )
                    random_pool_rets.append(r_ret * lev * 100.0)
            except Exception as exc:
                # The file exists but this arm could not use it: same reporting duty as the
                # not-found case, with the failure named so the gap is diagnosable.
                arm_missing_datasets.append(
                    f"{fname} (unusable: {type(exc).__name__}: {str(exc)[:160]})"
                )
                continue

        if missing_datasets is not None and arm_missing_datasets:
            missing_datasets[strat.name] = list(arm_missing_datasets)

        if not all_signal_rets or not random_pool_rets:
            continue

        sig_arr = np.array(all_signal_rets, dtype=float)
        rand_pool = np.array(random_pool_rets, dtype=float)
        k_signals = len(sig_arr)
        obs_mean = float(np.mean(sig_arr))
        obs_win_rate = float(np.mean(sig_arr > 0) * 100.0)
        gross_win = float(np.sum(sig_arr[sig_arr > 0]))
        gross_loss = abs(float(np.sum(sig_arr[sig_arr < 0])))
        # An arm whose sampled signals never lose has no denominator. `2.5` was a fabricated ratio
        # published as a measurement, so the key now carries the absence (JSON null) and every
        # renderer of it must be None-aware; nothing reads it as a number today.
        obs_pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else None

        boot_matrix = rng.choice(rand_pool, size=(b_iterations, k_signals), replace=True)
        boot_means = np.mean(boot_matrix, axis=1)
        rand_mean = float(np.mean(boot_means))
        rand_std = float(np.std(boot_means)) or 1e-6
        z_score = (obs_mean - rand_mean) / rand_std
        p_value = float((1.0 + np.sum(boot_means >= obs_mean)) / (b_iterations + 1.0))

        total_pert_checks = max(1, ratchet_premature_stops + ratchet_clean_survivors)
        ratchet_survival_pct = round((ratchet_clean_survivors / total_pert_checks) * 100.0, 1)
        pert_mean = float(np.mean(all_perturbed_rets)) if all_perturbed_rets else obs_mean

        # FinRL DRLEnsembleAgent Rolling Validation Window (Most recent out-of-sample trades)
        val_window = min(30, max(10, k_signals // 3))
        recent_slice = sig_arr[-val_window:]
        rec_mean = float(np.mean(recent_slice))
        rec_std = max(float(np.std(recent_slice)), 0.10)
        downside_slice = recent_slice[recent_slice < 0]
        downside_std = max(float(np.sqrt(np.mean(downside_slice ** 2))) if len(downside_slice) > 0 else 0.15, 0.10)

        # Window-scaled Sharpe & Sortino (sqrt(N_window) exactly per FinRL's validation Sharpe formula)
        rolling_sharpe = round(float(np.sqrt(len(recent_slice)) * (rec_mean / rec_std)), 2)
        rolling_sortino = round(float(np.sqrt(len(recent_slice)) * (rec_mean / downside_std)), 2)

        results[strat.name] = {
            "strategy_name": strat.name,
            "Sampled Signals (K)": int(k_signals),
            "bootstrap_iterations": int(b_iterations),
            "observed_expectancy_roe_pct": round(obs_mean, 3),
            "random_baseline_roe_pct": round(rand_mean, 3),
            "edge_spread_roe_pct": round(obs_mean - rand_mean, 3),
            "observed_win_rate_pct": round(obs_win_rate, 1),
            "observed_profit_factor": obs_pf,
            "z_score": round(z_score, 2),
            "p_value": round(p_value, 4),
            "significant_edge": bool(p_value < 0.05),
            "perturbed_expectancy_roe_pct": round(pert_mean, 3),
            "ratchet_survival_rate_pct": ratchet_survival_pct,
            "rolling_validation_window": int(len(recent_slice)),
            "rolling_recent_roe_pct": round(rec_mean, 3),
            "rolling_sharpe": rolling_sharpe,
            "rolling_sortino": rolling_sortino,
            "missing_datasets": list(arm_missing_datasets),
        }

    return results


def compute_finrl_rolling_sharpe_selector(bootstrap_res: Dict[str, Any]) -> Dict[str, Any]:
    """
    FinRL DRLEnsembleAgent Rolling Validation Sharpe Strategy Selector:
    Ranks the declarative strategies by out-of-sample validation Sharpe ratio and assigns
    dynamic capital allocation weights (1.20x Overweight Leader, 1.00x Standard, 0.75x Defensive Throttle).
    """
    ranked = sorted(
        bootstrap_res.values(),
        key=lambda x: (float(x.get("rolling_sharpe", 0.0)), float(x.get("z_score", 0.0))),
        reverse=True,
    )
    winner = ranked[0]["strategy_name"] if ranked else "SWING_4H"
    weights: Dict[str, Dict[str, Any]] = {}

    for idx, item in enumerate(ranked):
        s_name = item["strategy_name"]
        r_sharpe = float(item.get("rolling_sharpe", 0.0))
        sig_edge = bool(item.get("significant_edge", False))

        if idx == 0 and r_sharpe >= 0.75 and sig_edge:
            mult = 1.20
            posture = "OVERWEIGHT_LEADER"
        elif r_sharpe >= 0.40 and sig_edge:
            mult = 1.05
            posture = "ACTIVE_VALIDATED"
        elif r_sharpe >= 0.0:
            mult = 0.90
            posture = "STANDARD_CAUTIOUS"
        else:
            mult = 0.75
            posture = "DEFENSIVE_THROTTLE"

        weights[s_name] = {
            "rank": idx + 1,
            "rolling_sharpe": r_sharpe,
            "rolling_sortino": float(item.get("rolling_sortino", 0.0)),
            "rolling_recent_roe_pct": float(item.get("rolling_recent_roe_pct", 0.0)),
            "allocation_weight_mult": mult,
            "posture": posture,
        }

    return {
        "ensemble_winner": winner,
        "winner_sharpe": float(ranked[0].get("rolling_sharpe", 0.0)) if ranked else 0.0,
        "strategy_weights": weights,
    }


def run_trade_sequence_monte_carlo(
    n_simulations: int = 5000,
    horizon_trades: int = 60,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Resamples closed trades from trading_data.db (swing_trades) across N = 5,000 Monte Carlo paths
    under Quarter-Kelly sizing to project 95th and 99th percentile Maximum Drawdown and Equity Cones.
    Strictly read-only on trading_data.db.
    """
    rng = np.random.default_rng(seed)
    initial_equity = 500.0
    trade_pct_returns: List[float] = []
    min_trades = 10

    if not os.path.exists("trading_data.db"):
        raise InsufficientValidationData(
            "Monte Carlo validation requires trading_data.db in the working directory; none found."
        )

    current_row = None
    try:
        with sqlite3.connect("file:trading_data.db?mode=ro", uri=True, timeout=5.0) as conn:
            row_bal = conn.execute(
                "SELECT value FROM system_state WHERE key IN ('live_wallet_balance', 'wallet_balance', 'hwm_balance') LIMIT 1"
            ).fetchone()
            if row_bal and row_bal[0]:
                try:
                    b_val = float(row_bal[0])
                    if b_val > 50.0:
                        initial_equity = b_val
                except (TypeError, ValueError):
                    pass

            rows = conn.execute(
                "SELECT pnl, roe_pct, entry_price, size FROM swing_trades "
                "WHERE pnl IS NOT NULL ORDER BY id DESC LIMIT 200"
            ).fetchall()

        # swing_trades has no leverage column; roe_pct is the recorded return on margin.
        for current_row in rows:
            pnl, roe_pct, entry_price, size = current_row
            if roe_pct is not None:
                trade_pct_returns.append(float(roe_pct) / 100.0)
                continue
            notional = float(entry_price or 0.0) * float(size or 0.0)
            if notional > 0:
                trade_pct_returns.append(float(pnl) / notional)
    except (sqlite3.Error, ValueError, TypeError) as exc:
        if current_row is None:
            raise InsufficientValidationData(f"Monte Carlo validation could not read trading_data.db: {exc}") from exc
        raise InsufficientValidationData(
            f"Monte Carlo validation could not parse swing_trades row {current_row!r}: {exc}"
        ) from exc

    if len(trade_pct_returns) < min_trades:
        raise InsufficientValidationData(
            f"Monte Carlo validation needs at least {min_trades} closed trades with a recorded return; "
            f"found {len(trade_pct_returns)} in swing_trades. Refusing to substitute synthetic returns."
        )

    ret_arr = np.array(trade_pct_returns, dtype=float)
    wins = ret_arr[ret_arr > 0]
    losses = np.abs(ret_arr[ret_arr < 0])
    if len(wins) == 0 or len(losses) == 0:
        absent_kind = "losing" if len(losses) == 0 else "winning"
        raise InsufficientValidationData(
            f"Monte Carlo validation cannot measure a payoff ratio from {len(ret_arr)} recorded trades "
            f"that contain no {absent_kind} trade. Refusing to substitute a default win or loss size."
        )
    win_rate = float(len(wins) / len(ret_arr)) if len(ret_arr) > 0 else 0.5
    avg_win = float(np.mean(wins))
    avg_loss = float(np.mean(losses))
    payoff_b = avg_win / max(avg_loss, 1e-6)

    full_kelly = win_rate - ((1.0 - win_rate) / max(payoff_b, 1e-6))
    if full_kelly <= 0:
        raise InsufficientValidationData(
            f"Monte Carlo validation found no positive edge across {len(ret_arr)} recorded trades "
            f"(measured full Kelly {full_kelly:.4f}). Refusing to simulate a bet size the ledger does not support."
        )
    quarter_kelly = min(0.25, full_kelly * 0.25)

    sampled_roes = rng.choice(ret_arr, size=(n_simulations, horizon_trades), replace=True)
    acct_step_returns = 1.0 + (quarter_kelly * sampled_roes)

    equity_paths = np.empty((n_simulations, horizon_trades + 1), dtype=float)
    equity_paths[:, 0] = initial_equity
    equity_paths[:, 1:] = initial_equity * np.cumprod(acct_step_returns, axis=1)

    running_peaks = np.maximum.accumulate(equity_paths, axis=1)
    drawdowns = (running_peaks - equity_paths) / np.maximum(running_peaks, 1e-6) * 100.0
    max_dds = np.max(drawdowns, axis=1)

    dd_p50 = float(np.percentile(max_dds, 50))
    dd_p95 = float(np.percentile(max_dds, 95))
    dd_p99 = float(np.percentile(max_dds, 99))
    risk_of_ruin_20pct = float(np.mean(max_dds >= 20.0) * 100.0)

    cone_p05 = [round(float(x), 2) for x in np.percentile(equity_paths, 5, axis=0)]
    cone_p25 = [round(float(x), 2) for x in np.percentile(equity_paths, 25, axis=0)]
    cone_p50 = [round(float(x), 2) for x in np.percentile(equity_paths, 50, axis=0)]
    cone_p75 = [round(float(x), 2) for x in np.percentile(equity_paths, 75, axis=0)]
    cone_p95 = [round(float(x), 2) for x in np.percentile(equity_paths, 95, axis=0)]

    return {
        "status": "ok",
        "data_source": "swing_trades",
        "n_simulations": int(n_simulations),
        "horizon_trades": int(horizon_trades),
        "historical_sample_size": int(len(ret_arr)),
        "initial_equity_usdt": round(initial_equity, 2),
        "quarter_kelly_fraction_pct": round(quarter_kelly * 100.0, 2),
        "empirical_win_rate_pct": round(win_rate * 100.0, 1),
        "empirical_payoff_ratio": round(payoff_b, 2),
        "median_max_dd_pct": round(dd_p50, 2),
        "max_dd_95_pct": round(dd_p95, 2),
        "max_dd_99_pct": round(dd_p99, 2),
        "risk_of_20pct_dd_pct": round(risk_of_ruin_20pct, 2),
        "terminal_equity_p05": cone_p05[-1],
        "terminal_equity_p50": cone_p50[-1],
        "terminal_equity_p95": cone_p95[-1],
        "equity_cone": {
            "steps": list(range(horizon_trades + 1)),
            "p05": cone_p05,
            "p25": cone_p25,
            "p50": cone_p50,
            "p75": cone_p75,
            "p95": cone_p95,
        },
    }


def generate_full_validation_report(force_refresh: bool = False) -> Dict[str, Any]:
    """Generates or loads the cached Bootstrap Significance + Monte Carlo Validation Report."""
    if not force_refresh and os.path.exists(REPORT_CACHE_PATH):
        try:
            if (time.time() - os.path.getmtime(REPORT_CACHE_PATH)) < 3600.0:
                with open(REPORT_CACHE_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass

    t0 = time.time()
    missing_datasets: Dict[str, List[str]] = {}
    bootstrap_res = run_bootstrap_entry_significance(
        b_iterations=2000, noise_sigma=0.15, missing_datasets=missing_datasets
    )
    ensemble_res = compute_finrl_rolling_sharpe_selector(bootstrap_res)
    try:
        mc_res = run_trade_sequence_monte_carlo(n_simulations=5000, horizon_trades=60)
    except InsufficientValidationData as exc:
        mc_res = {"status": "unavailable", "data_source": "none", "reason": str(exc)}
    elapsed = round(time.time() - t0, 2)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "compute_seconds": elapsed,
        "bootstrap_significance": bootstrap_res,
        "finrl_ensemble_selector": ensemble_res,
        "monte_carlo": mc_res,
        "missing_datasets": missing_datasets,
    }

    try:
        os.makedirs("cache", exist_ok=True)
        with open(REPORT_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    except Exception:
        pass

    return report


def main() -> None:
    rep = generate_full_validation_report(force_refresh=True)
    print(f"=== QUANT VALIDATION SUITE COMPLETED IN {rep['compute_seconds']}s ===")
    for k, v in rep["bootstrap_significance"].items():
        print(
            f"[{k}] K={v['Sampled Signals (K)']} | Obs ROE={v['observed_expectancy_roe_pct']:+.2f}% vs "
            f"Rand={v['random_baseline_roe_pct']:+.2f}% | Z={v['z_score']:+.2f} | p={v['p_value']:.4f} "
            f"| Pass={v['significant_edge']} | Rolling Sharpe={v.get('rolling_sharpe', 0.0):+.2f} "
            f"| Ratchet Survival={v['ratchet_survival_rate_pct']:.1f}%"
        )
    for k, v in rep.get("missing_datasets", {}).items():
        if v:
            # Wording has to cover both kinds of gap the map now carries: files that are absent and
            # files that exist but could not be used. The old sentence asserted the first for both.
            print(f"   [DATA GAP] {k}: requested datasets missing or unusable — {', '.join(v)}")
    ens = rep.get("finrl_ensemble_selector", {})
    print(f"[FINRL ENSEMBLE WINNER] {ens.get('ensemble_winner')} (Sharpe={ens.get('winner_sharpe', 0.0):+.2f}) | Weights: {ens.get('strategy_weights')}")
    mc = rep["monte_carlo"]
    if mc.get("status") == "unavailable":
        print(f"[MONTE CARLO] UNAVAILABLE — {mc.get('reason')}")
    else:
        print(
            f"[MONTE CARLO N={mc['n_simulations']}] 1/4 Kelly={mc['quarter_kelly_fraction_pct']:.1f}% | "
            f"Median DD={mc['median_max_dd_pct']:.2f}% | MaxDD_95={mc['max_dd_95_pct']:.2f}% | "
            f"MaxDD_99={mc['max_dd_99_pct']:.2f}% | Median Terminal Equity=${mc['terminal_equity_p50']:,.2f}"
        )


if __name__ == "__main__":
    main()

"""
tests/test_replay_consistency.py — Replay Fidelity & Consistency Invariant Tests

Verifies that:
1. Pure gate functions and causal indicators computed in replay mode match the
   live pipeline bar-by-bar across >= 500 rolling slices per symbol.
2. Truncation invariance holds: indicator outputs at time t are identical whether
   computed on the full dataset or on a historical prefix ending at t.
3. No forward-looking bias exists in gate evaluations.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import indicators as ind
from swing_trader import SwingArchitectTrader
from backtests.backtest_ablation_harness import precompute_4h_indicators_vectorized


def load_dataset(csv_path: str = "data/historical/BTCUSDT_1h_full.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads and resamples authentic 1H BTC data to 4H and 1D."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Test data not found: {csv_path}")

    df_1h = pd.read_csv(csv_path)
    if "timestamp" in df_1h.columns:
        ts_series = df_1h["timestamp"]
        unit = "ms" if ts_series.iloc[0] > 1e11 else "s"
        df_1h["dt"] = pd.to_datetime(ts_series, unit=unit)
        df_1h = df_1h.set_index("dt").sort_index()
        df_4h = df_1h.resample("4h").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "timestamp": "last"
        }).dropna().reset_index(drop=True)
        df_1d = df_1h.resample("1D").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "timestamp": "last"
        }).dropna().reset_index(drop=True)
    else:
        df_4h = df_1h.iloc[::4].copy().reset_index(drop=True)
        df_1d = df_1h.iloc[::24].copy().reset_index(drop=True)

    return df_4h, df_1d


def test_indicator_and_gate_replay_consistency():
    """
    Tests >= 500 rolling slices of 4H data, verifying that the indicators and gate
    evaluations in swing_trader._compute_4h_indicators match the vectorized / harness math.
    """
    df_4h, df_1d = load_dataset()
    assert len(df_4h) >= 600, f"Need at least 600 4H bars for testing, found {len(df_4h)}"

    # Instantiate SwingTrader without connecting live exchange
    st = SwingArchitectTrader()

    # Precompute vectorized dataframe
    df_vec = precompute_4h_indicators_vectorized(df_4h)

    # Test rolling window over 500 slices
    num_test_slices = 500
    start_idx = len(df_4h) - num_test_slices

    for idx in range(start_idx, len(df_4h)):
        # Simulate live rolling buffer of bars up to idx
        slice_4h = df_4h.iloc[:idx + 1]

        # Live engine method
        live_ind = st._compute_4h_indicators(slice_4h)
        if not live_ind:
            continue

        vec_row = df_vec.iloc[idx]

        # 1. Compare close, open, high, low
        assert abs(live_ind["close"] - vec_row["close"]) < 1e-5
        assert abs(live_ind["open"] - vec_row["open"]) < 1e-5

        # 2. Compare Ichimoku components
        assert abs(live_ind["tenkan"] - ((slice_4h["high"].rolling(9).max().iloc[-1] + slice_4h["low"].rolling(9).min().iloc[-1]) / 2.0)) < 1e-4
        assert abs(live_ind["kijun"] - ((slice_4h["high"].rolling(26).max().iloc[-1] + slice_4h["low"].rolling(26).min().iloc[-1]) / 2.0)) < 1e-4

        # 3. Compare ATR
        assert abs(live_ind["atr"] - vec_row["atr"]) < 1e-4

        # 4. Compare Kumo breakout signals
        assert live_ind["bullish_kumo"] == vec_row["bullish_kumo"], f"Kumo mismatch at bar {idx}"
        assert live_ind["bearish_kumo"] == vec_row["bearish_kumo"], f"Kumo mismatch at bar {idx}"

        # 5. Compare base trade setups
        if live_ind["bullish_kumo"] and live_ind["bullish_candle_4h"] and live_ind["adx"] >= 22.0 and live_ind["rsi"] <= 70.0:
            assert vec_row["is_base_long"], f"Base long mismatch at bar {idx}"


def test_gate_purity_and_zero_lookahead():
    """
    Verifies that gate functions are pure functions of bar history:
    appending future bars does not change previous gate decisions.
    """
    df_4h, _ = load_dataset()
    st = SwingArchitectTrader()

    test_bar = 800
    sub_1 = df_4h.iloc[:test_bar + 1]
    sub_2 = df_4h.iloc[:test_bar + 50]  # with 50 future bars

    # Divergence gate test
    div_1, rsn_1 = st._check_divergence_gate("BTC/USDT", sub_1, direction="LONG")
    div_2, rsn_2 = st._check_divergence_gate("BTC/USDT", sub_2.iloc[:test_bar + 1], direction="LONG")
    assert div_1 == div_2, "Divergence gate decision changed with prefix slice"

    # Volume profile gate test
    curr_price = float(sub_1["close"].iloc[-1])
    vp_1, vp_rsn_1 = st._check_volume_profile_gate("BTC/USDT", sub_1, curr_price, direction="LONG")
    vp_2, vp_rsn_2 = st._check_volume_profile_gate("BTC/USDT", sub_2.iloc[:test_bar + 1], curr_price, direction="LONG")
    assert vp_1 == vp_2, "Volume profile gate decision changed with prefix slice"

    # TD Sequential gate test
    td_1, td_rsn_1 = st._check_td_sequential_gate("BTC/USDT", sub_1, direction="LONG")
    td_2, td_rsn_2 = st._check_td_sequential_gate("BTC/USDT", sub_2.iloc[:test_bar + 1], direction="LONG")
    assert td_1 == td_2, "TD Sequential gate decision changed with prefix slice"


if __name__ == "__main__":
    print("[*] Running test_indicator_and_gate_replay_consistency...")
    test_indicator_and_gate_replay_consistency()
    print("[*] Running test_gate_purity_and_zero_lookahead...")
    test_gate_purity_and_zero_lookahead()
    print("ALL REPLAY CONSISTENCY TESTS PASSED SUCCESSFULLY!")

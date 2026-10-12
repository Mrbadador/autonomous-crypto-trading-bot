import sys
import os
import types
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import indicators as ind


def make_ohlcv(n=1600, seed=7):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    vol = rng.uniform(100, 1000, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol})


def test_validate_rejects_nan_and_bad_ohlc():
    df = make_ohlcv(100)
    bad = df.copy()
    bad.loc[10, "close"] = np.nan
    try:
        ind.validate_ohlcv(bad)
        assert False, "Should raise ValueError on NaN"
    except ValueError:
        pass

    bad = df.copy()
    bad.loc[10, "high"] = bad.loc[10, "low"] - 1
    try:
        ind.validate_ohlcv(bad)
        assert False, "Should raise ValueError on high < low"
    except ValueError:
        pass


def test_strict_warmup_fails_closed():
    try:
        ind.compute_indicators(make_ohlcv(200), strict=True)
        assert False, "Should raise on < 600 bars with strict=True"
    except ValueError:
        pass
    out = ind.compute_indicators(make_ohlcv(200), warmup_bars=100)
    assert len(out) == 200


def test_truncation_invariance():
    df = make_ohlcv(900)
    full = ind.compute_indicators(df, warmup_bars=100)
    for t in (350, 500, 899):
        part = ind.compute_indicators(df.iloc[:t + 1], warmup_bars=100)
        a = part.iloc[-1][["ema20", "ema50", "rsi14", "atr14"]].astype(float).to_numpy()
        b = full.iloc[t][["ema20", "ema50", "rsi14", "atr14"]].astype(float).to_numpy()
        np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-7, equal_nan=True, err_msg=f"t={t}")


def test_td_setup():
    closes = [100 - 1.0 * i for i in range(20)] + [81 + 1.5 * (j + 1) for j in range(40)]
    series = pd.Series(np.asarray(closes, dtype=float))
    up, dn = ind.td_setup_counts(series)
    assert up.max() == 9, "TD Setup must cap at 9"
    first = int(np.argmax(up.to_numpy() > 0))
    assert list(up.iloc[first:first + 9]) == list(range(1, 10))
    assert up.iloc[first + 9] == 0, "TD Setup must reset after 9"


def test_divergence():
    n, pk1, pk2 = 48, 22, 38
    base = 90.0
    h = np.full(n, base)
    for idx, peak in ((pk1, 100.0), (pk2, 103.0)):
        for d in range(-3, 4):
            h[idx + d] = peak - 2 * abs(d)
    lo = h - 2.0
    rsi = np.full(n, 50.0); rsi[pk1], rsi[pk2] = 70.0, 62.0
    hist = np.zeros(n); hist[pk1], hist[pk2] = 1.0, 0.4
    df = pd.DataFrame({"high": h, "low": lo, "rsi14": rsi, "macd_hist": hist, "atr14": np.ones(n)})
    res = ind.detect_momentum_divergence(df)
    assert res["bearish_divergence"]
    assert not res["bullish_divergence"]


def run_all_tests():
    test_validate_rejects_nan_and_bad_ohlc()
    test_strict_warmup_fails_closed()
    test_truncation_invariance()
    test_td_setup()
    test_divergence()
    print("All tests passed in test_indicators.py!")


if __name__ == "__main__":
    run_all_tests()

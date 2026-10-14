"""
tests/test_regime_adx_parity.py

TASK 4 regression tests for RegimeClassifier.calculate_adx — the ADX/+DI/-DI producer that
gates swing and grid routing in super_bot.

The Wilder fallback used to seed TR/+DM/-DM with rolling MEANS and then apply Wilder's
sum-form recurrence (`x[i] = x[i-1] - x[i-1]/length + v[i]`). That recurrence expects the seed
to be the SUM of the first `length` values. Only the seed was wrong and its influence decays,
so the practical error is modest (0.01-0.48 ADX points on a 105-bar window) — but the producer
was not computing the indicator the super_bot ADX gates assume it computes.

Invariant under test: after warm-up the fallback agrees with an independent reference
implementation (ta.trend.ADXIndicator, the same library the backtests use) to 1e-6.

Safety: pure in-memory arithmetic on deterministic synthetic OHLC. No network, no exchange,
no live DB — RegimeClassifier is constructed against a throwaway temp database.
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

import regime_classifier  # noqa: E402
from regime_classifier import RegimeClassifier  # noqa: E402

try:
    from ta.trend import ADXIndicator  # declared dependency (ta==0.11.0), used by backtests
    _HAS_TA = True
except ImportError:
    _HAS_TA = False

LENGTH = 14
TOL = 1e-6


def synthetic_ohlcv(n=600, seed=7, drift=0.0006, vol=0.004):
    """Deterministic OHLC: geometric random walk with a same-bar wick on both sides."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[100.0], close[:-1]])
    span = np.abs(rng.normal(0.0, vol, n)) * close
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


class RegimeAdxParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="regime_adx_")
        cls.db_path = os.path.join(cls.tmpdir, "regime_test.db")
        cls.rc = RegimeClassifier(db_path=cls.db_path)

    @staticmethod
    def _reference(df):
        ind = ADXIndicator(df["high"], df["low"], df["close"], window=LENGTH, fillna=True)
        return float(ind.adx().iloc[-1]), float(ind.adx_pos().iloc[-1]), float(ind.adx_neg().iloc[-1])

    @unittest.skipUnless(_HAS_TA, "ta (declared dependency) not importable")
    def test_fallback_matches_reference_wilder_adx(self):
        """Core parity: three seeds x two lengths agree with ta.trend.ADXIndicator to 1e-6."""
        cases = [
            (105, 7, 0.0006, 0.004),    # the live 15m window size (fetch limit 105)
            (600, 7, 0.0006, 0.004),
            (600, 21, 0.0020, 0.006),   # strong uptrend
            (600, 42, -0.0025, 0.006),  # strong downtrend
            (600, 99, 0.0000, 0.0015),  # dead chop
        ]
        for n, seed, drift, vol in cases:
            df = synthetic_ohlcv(n=n, seed=seed, drift=drift, vol=vol)
            with self.subTest(n=n, seed=seed, drift=drift):
                got = self.rc.calculate_adx(df, length=LENGTH)
                want = self._reference(df)
                self.assertTrue(all(np.isfinite(v) for v in got), got)
                for label, g, w in zip(("adx", "+di", "-di"), got, want):
                    self.assertLess(abs(g - w), TOL, f"{label} n={n} seed={seed}: got {g!r} want {w!r}")

    @unittest.skipUnless(_HAS_TA, "ta (declared dependency) not importable")
    def test_legacy_mean_seed_did_not_reach_reference_parity(self):
        """Records the size of the bug rather than overstating it.

        The removed fallback seeded Wilder's sum-form recurrence with a rolling MEAN, so only
        the seed was wrong (short by a factor of `length`) and its influence decays. Measured
        on deterministic 105-bar windows the legacy ADX sat 0.01-0.48 points from a reference
        Wilder ADX and +DI/-DI within ~0.02 — a real correctness defect, but far smaller than
        a regime-gate rewrite. This test asserts the old code failed parity and the new code
        achieves it; the printed delta is the before/after evidence for the verification report.
        """
        df = synthetic_ohlcv(n=105, seed=21, drift=0.002, vol=0.006)
        adx_new, plus_new, minus_new = self.rc.calculate_adx(df, length=LENGTH)

        # Re-implementation of the removed mean-seeded fallback, for the before/after delta.
        def legacy(df, length=LENGTH):
            d = df.copy()
            up, down = d["high"].diff(), -d["low"].diff()
            d["plus_dm"] = np.where((up > down) & (up > 0), up, 0.0)
            d["minus_dm"] = np.where((down > up) & (down > 0), down, 0.0)
            d["tr"] = np.maximum(d["high"] - d["low"],
                                 np.maximum(abs(d["high"] - d["close"].shift(1)),
                                            abs(d["low"] - d["close"].shift(1))))
            d["tr"] = d["tr"].fillna(d["high"] - d["low"])
            sm = {k: d[k].rolling(length).mean().values.copy() for k in ("tr", "plus_dm", "minus_dm")}
            raw = {k: d[k].values for k in ("tr", "plus_dm", "minus_dm")}
            for i in range(length, len(d)):
                if np.isnan(sm["tr"][i - 1]):
                    continue
                for k in sm:
                    sm[k][i] = sm[k][i - 1] - sm[k][i - 1] / length + raw[k][i]
            pdi = 100.0 * sm["plus_dm"] / (sm["tr"] + 1e-9)
            mdi = 100.0 * sm["minus_dm"] / (sm["tr"] + 1e-9)
            dx = 100.0 * abs(pdi - mdi) / (pdi + mdi + 1e-9)
            adx = pd.Series(dx).rolling(length).mean().values.copy()
            for i in range(2 * length - 1, len(d)):
                if np.isnan(adx[i - 1]):
                    continue
                adx[i] = (adx[i - 1] * (length - 1) + dx[i]) / length
            return float(adx[-1]), float(pdi[-1]), float(mdi[-1])

        adx_old, plus_old, minus_old = legacy(df)
        ref_adx, ref_p, ref_m = self._reference(df)
        self.assertLess(abs(adx_new - ref_adx), TOL, "corrected fallback must match reference Wilder")
        self.assertLess(abs(plus_new - ref_p), TOL)
        self.assertLess(abs(minus_new - ref_m), TOL)
        self.assertGreater(abs(adx_old - ref_adx), 0.05, "legacy seed is expected to miss parity")
        self.assertLess(abs(adx_old - ref_adx), 5.0, "legacy error stays small; do not overstate it")
        print(f"\n[before/after] n=105 trending: legacy ADX={adx_old:.4f}  corrected={adx_new:.4f}  "
              f"reference={ref_adx:.4f}  (legacy +DI err={abs(plus_old - ref_p):.4f}, "
              f"-DI err={abs(minus_old - ref_m):.4f})")

    def test_pandas_ta_failure_is_logged_not_swallowed(self):
        """TASK 4: the old `except Exception: pass` hid the missing dependency on every call."""
        original = regime_classifier._pandas_ta

        class _Broken:
            @staticmethod
            def adx(*args, **kwargs):
                raise RuntimeError("simulated pandas_ta failure")

        regime_classifier._pandas_ta = _Broken()
        df = synthetic_ohlcv(n=200, seed=5, drift=0.001, vol=0.005)
        try:
            with self.assertLogs("regime_classifier", level="WARNING") as captured:
                adx_val, plus_di, minus_di = self.rc.calculate_adx(df, length=LENGTH)
        finally:
            regime_classifier._pandas_ta = original

        self.assertTrue(any("pandas_ta ADX failed" in m for m in captured.output), captured.output)
        self.assertTrue(np.isfinite(adx_val) and np.isfinite(plus_di) and np.isfinite(minus_di))
        if _HAS_TA:
            ref = self._reference(df)
            self.assertLess(abs(adx_val - ref[0]), TOL)

    def test_too_few_bars_warns_instead_of_returning_quiet_default(self):
        """Below 2*length+1 bars the smoothing cannot be seeded; the old code returned a quiet
        20.0 with no trace. The value contract is unchanged (fail-closed routing is TASK 2) but
        the condition is now visible in the log."""
        df = synthetic_ohlcv(n=20, seed=3)
        with self.assertLogs("regime_classifier", level="WARNING") as captured:
            got = self.rc.calculate_adx(df, length=LENGTH)
        self.assertEqual(got, (20.0, 0.0, 0.0))
        self.assertTrue(any("ADX fallback needs at least" in m for m in captured.output), captured.output)

    def test_classifier_does_not_write_the_repository_database(self):
        """calculate_adx is pure arithmetic; the only database this object owns is the temp one."""
        self.assertIn(os.path.realpath(self.tmpdir), os.path.realpath(self.db_path))
        repo_db = os.path.join(REPO_DIR, "trading_data.db")
        if not os.path.exists(repo_db):
            self.skipTest("no trading_data.db in the repo")
        before = os.stat(repo_db).st_mtime_ns
        self.rc.calculate_adx(synthetic_ohlcv(n=200), length=LENGTH)
        self.assertEqual(os.stat(repo_db).st_mtime_ns, before, "live DB must be untouched by this test")


if __name__ == "__main__":
    unittest.main()

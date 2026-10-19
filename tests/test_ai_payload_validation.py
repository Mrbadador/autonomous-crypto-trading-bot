"""Regime P0 / TASK 5 — the AI cognitive arbiter payload is validated before it is cached,
and a rejected or garbage verdict can never abort the classifier loop.

Pinned behaviour:
  * _validate_ai_payload() enforces the enum, finite 0..1 trap score and the {2,3,5}
    leverage allowlist, and returns None instead of a half-shaped dict.
  * A provider that answers with invalid JSON, an out-of-enum bias or a leverage cap of 7
    is discarded; the model pool keeps failing over and, if nothing validates, the
    deterministic offline fallback is returned WITHOUT an evaluation stamp, so it cannot
    be cached as if a model had spoken.
  * classify_and_save() coerces every cached field through bounded parsers. Before this
    fix a cached {"trap_risk_score": None} raised float(None) outside the local try, which
    exited the process and froze market_regime for every entry gate downstream.
  * Loosening (CHOP -> BULL_TREND promotion, leverage straight from the provider) requires
    a verdict stamped inside the AI cadence; tightening may still act on a held verdict.
  * The AI transition trigger and the 20-25 ADX Schmitt hysteresis compare against the
    previous MATHEMATICAL regime, not the published label which may be an AI override.
  * The true evaluation time is persisted (ai_regime_last_eval_utc /
    ai_regime_eval_age_sec), separately from the row publication timestamp.

Isolation: temp-directory database, stubbed HTTP, no orders, no network.
"""

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import unittest
from datetime import datetime, UTC
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

import config  # noqa: E402
from regime_classifier import (  # noqa: E402
    AI_LEVERAGE_ALLOWLIST,
    AI_REGIME_BIASES,
    RegimeClassifier,
)

CLASSIFIER_LOGGER = "regime_classifier"
STEP_15M = 15 * 60
STEP_4H = 4 * 3600

VALID_VERDICT = {
    "ai_regime_bias": "LIQUIDITY_HUNT_TRAP",
    "trap_risk_score": 0.72,
    "recommended_leverage_cap": 2,
    "structural_thesis": "Upper wicks into resistance; distribution suspected.",
    "override_math_regime": True,
}


class FakeResponse:
    def __init__(self, payload_text, status_code=200):
        self.status_code = status_code
        self._payload_text = payload_text

    def json(self):
        return json.loads(self._payload_text) if isinstance(self._payload_text, str) else self._payload_text


def gemini_text(verdict):
    return json.dumps({
        "candidates": [{"content": {"parts": [{"text": json.dumps(verdict)}]}}]
    })


def bars(n, step_sec, slope, end_offset_sec=0.0):
    """Closed-looking OHLCV frame; slope>0 rises, the newest bar ends end_offset_sec ago."""
    epoch = int(time.time())
    slot = epoch - epoch % step_sec
    rows = []
    close = 100.0
    for i in range(n):
        open_ms = (slot - (n - 1 - i) * step_sec - int(end_offset_sec)) * 1000
        rows.append({"open_time": open_ms, "open": close,
                     "high": close * (1.002 + abs(slope)), "low": close * (0.998 - abs(slope)),
                     "close": close, "volume": 10.0, "quote_volume": close * 10.0})
        close *= (1.0 + slope)
    return pd.DataFrame(rows)


class TestAiPayloadValidation(unittest.TestCase):
    clf: RegimeClassifier

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="ai_payload_")
        cls.db_path = os.path.join(cls.tmp_dir, "fixture_regime.db")
        cls.clf = RegimeClassifier(db_path=cls.db_path)
        # The classifier only owns market_regime; system_state is created by other daemons
        # in production, so the fixture supplies the production-shaped table.
        with sqlite3.connect(cls.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS "system_state" (
                key TEXT PRIMARY KEY, value TEXT, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
            conn.commit()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def setUp(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM market_regime")
            conn.execute("DELETE FROM system_state")
            conn.commit()
        self.clf.last_ai_eval_ts = 0.0
        self.clf.latest_ai_arbiter = {
            "ai_regime_bias": "DEFENSIVE_CHOP",
            "trap_risk_score": 0.50,
            "recommended_leverage_cap": 3,
            "structural_thesis": "Baseline cognitive regime initialization.",
            "override_math_regime": False,
            "evaluated_by": "baseline",
            "evaluated_at_utc": None,
            "evaluated_at_epoch": 0.0,
        }
        self.clf.active_ai_bias = "DEFENSIVE_CHOP"
        self.clf.pending_ai_bias = None
        self.clf.pending_ai_bias_count = 0
        self.clf.ai_bias_dwell_ts = 0.0

    # ------------------------------------------------------- contract validation
    def test_valid_verdict_is_normalised(self):
        out = self.clf._validate_ai_payload(VALID_VERDICT, "test")
        self.assertEqual(out["ai_regime_bias"], "LIQUIDITY_HUNT_TRAP")
        self.assertEqual(out["trap_risk_score"], 0.72)
        self.assertEqual(out["recommended_leverage_cap"], 2)
        self.assertIs(out["override_math_regime"], True)
        self.assertEqual(out["evaluated_by"], "test")

    def test_lenient_but_safe_shapes_are_accepted(self):
        out = self.clf._validate_ai_payload({
            "ai_regime_bias": "  bear_dump  ",
            "trap_risk_score": "0.10",
            "recommended_leverage_cap": "3",
            "structural_thesis": None,
            "override_math_regime": "true",
        }, "test")
        self.assertEqual(out["ai_regime_bias"], "BEAR_DUMP")
        self.assertEqual(out["trap_risk_score"], 0.10)
        self.assertEqual(out["recommended_leverage_cap"], 3)
        self.assertEqual(out["structural_thesis"], "")
        self.assertIs(out["override_math_regime"], True)

    def test_rejected_payloads(self):
        bad = [
            ("bias out of enum", {"ai_regime_bias": "MOON_SHOT", "trap_risk_score": 0.2, "recommended_leverage_cap": 3}),
            ("bias missing", {"trap_risk_score": 0.2, "recommended_leverage_cap": 3}),
            ("bias null", {"ai_regime_bias": None, "trap_risk_score": 0.2, "recommended_leverage_cap": 3}),
            ("score null", {"ai_regime_bias": "BEAR_DUMP", "trap_risk_score": None, "recommended_leverage_cap": 3}),
            ("score not numeric", {"ai_regime_bias": "BEAR_DUMP", "trap_risk_score": "high", "recommended_leverage_cap": 3}),
            ("score above range", {"ai_regime_bias": "BEAR_DUMP", "trap_risk_score": 1.5, "recommended_leverage_cap": 3}),
            ("score below range", {"ai_regime_bias": "BEAR_DUMP", "trap_risk_score": -0.2, "recommended_leverage_cap": 3}),
            ("score NaN", {"ai_regime_bias": "BEAR_DUMP", "trap_risk_score": float("nan"), "recommended_leverage_cap": 3}),
            ("leverage off allowlist", {"ai_regime_bias": "BEAR_DUMP", "trap_risk_score": 0.2, "recommended_leverage_cap": 7}),
            ("leverage 20", {"ai_regime_bias": "BULL_EXPANSION", "trap_risk_score": 0.1, "recommended_leverage_cap": 20}),
            ("leverage missing", {"ai_regime_bias": "BEAR_DUMP", "trap_risk_score": 0.2}),
            ("leverage null", {"ai_regime_bias": "BEAR_DUMP", "trap_risk_score": 0.2, "recommended_leverage_cap": None}),
        ]
        for label, payload in bad:
            with self.subTest(case=label):
                self.assertIsNone(self.clf._validate_ai_payload(payload, "test"))
        for label, payload in (("list", [1, 2]), ("string", "BEAR_DUMP"), ("none", None), ("number", 7)):
            with self.subTest(non_object=label):
                self.assertIsNone(self.clf._validate_ai_payload(payload, "test"))

    def test_allowlist_and_enum_constants_are_what_the_gate_uses(self):
        self.assertEqual(AI_LEVERAGE_ALLOWLIST, (2, 3, 5))
        self.assertEqual(set(AI_REGIME_BIASES),
                         {"BULL_EXPANSION", "BEAR_DUMP", "DEFENSIVE_CHOP", "LIQUIDITY_HUNT_TRAP"})

    def test_thesis_is_truncated_before_it_reaches_the_database(self):
        out = self.clf._validate_ai_payload({**VALID_VERDICT, "structural_thesis": "x" * 5000}, "test")
        self.assertLessEqual(len(out["structural_thesis"]), 400)

    # ------------------------------------------------------------- provider pool
    def _post(self, *responses):
        it = iter(responses)
        return MagicMock(side_effect=lambda *a, **k: next(it))

    def test_invalid_json_is_not_cached_and_pool_fails_over(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyntheticTestKey", "GROQ_API_KEY": ""}), \
             patch.object(config, "GEMINI_MODEL_POOL", ["model-a", "model-b"], create=True), \
             patch("regime_classifier.requests.post",
                   self._post(FakeResponse('{"candidates":[{"content":{"parts":[{"text":"not json"}]}}]}'),
                              FakeResponse(gemini_text(VALID_VERDICT)))):
            out = self.clf._query_ai_cognitive_arbiter({"symbol": "BTC/USDT"})
        self.assertEqual(out["ai_regime_bias"], "LIQUIDITY_HUNT_TRAP")
        self.assertEqual(out["evaluated_by"], "Gemini:model-b")
        self.assertTrue(out.get("evaluated_at_utc"))

    def test_contract_violation_from_every_provider_yields_unstamped_fallback(self):
        garbage = FakeResponse(gemini_text({"ai_regime_bias": "MOON_SHOT", "trap_risk_score": 0.1,
                                            "recommended_leverage_cap": 9}))
        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyntheticTestKey",
                                     "GROQ_API_KEY": "", "GROQ_API_KEY_2": "", "GROQ_API_KEY_3": ""}), \
             patch.object(config, "GEMINI_MODEL_POOL", ["model-a", "model-b"], create=True), \
             patch("regime_classifier.requests.post", self._post(garbage, garbage)) as post, \
             self.assertLogs(CLASSIFIER_LOGGER, level="WARNING") as captured:
            out = self.clf._query_ai_cognitive_arbiter({"symbol": "BTC/USDT"})
        self.assertEqual(post.call_count, 2, "each rejected answer must move on to the next model")
        self.assertNotIn("evaluated_at_utc", out)
        self.assertEqual(out["evaluated_by"], "offline_fallback")
        self.assertTrue(any("contract-valid" in line for line in captured.output), captured.output)

    def test_http_error_responses_do_not_raise(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyntheticTestKey", "GROQ_API_KEY": ""}), \
             patch.object(config, "GEMINI_MODEL_POOL", ["model-a"], create=True), \
             patch("regime_classifier.requests.post", self._post(FakeResponse("{}", status_code=503))):
            out = self.clf._query_ai_cognitive_arbiter({"symbol": "BTC/USDT"})
        self.assertEqual(out["evaluated_by"], "offline_fallback")

    # ------------------------------------------------------- classify_and_save
    def _stub_market(self, adx=18.0, plus_di=30.0, minus_di=20.0, prior_math="CHOP", prior_label="CHOP"):
        ts = datetime.now(UTC).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO market_regime (timestamp, regime, math_regime, hurst, adx,
                   plus_di, minus_di, macro_trend, ai_bias, trap_risk, ai_thesis,
                   turbulence_index, turbulence_threshold, turbulence_state)
                   VALUES (?, ?, ?, 0.50, ?, ?, ?, 'BULL', 'DEFENSIVE_CHOP', 0.4, 'seed', 1.0, 22.0, 'CALM_NORMAL')""",
                (ts, prior_label, prior_math, adx, plus_di, minus_di),
            )
            conn.commit()

    def _run_classify(self, *, ai_mock=None, arbiter=None, adx=18.0, plus_di=30.0, minus_di=20.0,
                      cadence="fresh"):
        """Runs one classify_and_save() with all I/O stubbed; returns (result, row_written).

        cadence="fresh"   -> inside the 15-min AI cadence, so only a mathematical
                             transition can arm a provider call.
        cadence="expired" -> the cadence has elapsed, so the provider is called anyway.
        """
        clf = self.clf
        if arbiter is not None:
            clf.latest_ai_arbiter = arbiter
        if cadence == "fresh":
            clf.last_ai_eval_ts = time.time()
        elif cadence == "expired":
            clf.last_ai_eval_ts = time.time() - (clf.ai_eval_interval_sec + 5.0)
        else:
            raise ValueError(f"unknown cadence mode {cadence!r}")
        query = ai_mock if ai_mock is not None else MagicMock(return_value=self.clf._offline_ai_fallback())
        with patch.object(RegimeClassifier, "fetch_ohlcv",
                          side_effect=lambda symbol, interval, limit: (
                              bars(99, STEP_15M, 0.0002) if interval == "15m" else bars(49, STEP_4H, 0.004))), \
             patch.object(RegimeClassifier, "calculate_adx", return_value=(adx, plus_di, minus_di)), \
             patch.object(RegimeClassifier, "fetch_funding_rate", return_value=0.0001), \
             patch.object(RegimeClassifier, "calculate_mahalanobis_turbulence",
                          return_value={"turbulence_index": 1.0, "active_threshold": 22.0,
                                        "turbulence_state": "CALM_NORMAL", "turbulence_breach": False}), \
             patch.object(RegimeClassifier, "_query_ai_cognitive_arbiter", query):
            ok = clf.classify_and_save()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM market_regime ORDER BY id DESC LIMIT 1").fetchone()
            state = {r[0]: r[1] for r in conn.execute("SELECT key, value FROM system_state")}
        return ok, dict(row) if row else None, state, query

    def test_poisoned_cache_no_longer_aborts_the_loop(self):
        """The exact float(None) shape that used to kill the daemon."""
        self._stub_market()
        poisoned = {
            "ai_regime_bias": None,
            "trap_risk_score": None,
            "recommended_leverage_cap": None,
            "structural_thesis": None,
            "override_math_regime": None,
        }
        ok, row, state, _ = self._run_classify(arbiter=poisoned)
        self.assertTrue(ok)
        self.assertIsNotNone(row)
        self.assertEqual(row["ai_bias"], "DEFENSIVE_CHOP")
        self.assertEqual(row["trap_risk"], 0.5)
        self.assertEqual(row["regime"], row["math_regime"])          # no phantom override
        self.assertEqual(state["ai_regime_eval_age_sec"], "-1")      # never evaluated
        self.assertEqual(state["ai_regime_last_eval_utc"], "")

    def test_leverage_cap_is_clamped_to_the_allowlist(self):
        self._stub_market()
        wild = {
            "ai_regime_bias": "BULL_EXPANSION",
            "trap_risk_score": 0.05,
            "recommended_leverage_cap": 25,
            "structural_thesis": "give me twenty five times",
            "override_math_regime": True,
            "evaluated_by": "Gemini:rogue",
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
            "evaluated_at_epoch": time.time() - 30,
        }
        ok, row, state, _ = self._run_classify(arbiter=wild)
        self.assertTrue(ok)
        written = int(state["recommended_leverage"])
        self.assertIn(written, AI_LEVERAGE_ALLOWLIST)
        self.assertLessEqual(written, 3)   # CHOP baseline; the off-allowlist cap cannot loosen it

    def test_stale_verdict_cannot_promote_chop_to_bull_trend(self):
        self._stub_market(prior_math="CHOP", prior_label="CHOP")
        self.clf.active_ai_bias = "BULL_EXPANSION"
        stale = {
            "ai_regime_bias": "BULL_EXPANSION",
            "trap_risk_score": 0.10,
            "recommended_leverage_cap": 5,
            "structural_thesis": "hours old",
            "override_math_regime": True,
            "evaluated_by": "Gemini:model-a",
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
            "evaluated_at_epoch": time.time() - 7200,
        }
        with self.assertLogs(CLASSIFIER_LOGGER, level="WARNING") as captured:
            ok, row, state, _ = self._run_classify(arbiter=stale)
        self.assertTrue(ok)
        self.assertEqual(row["regime"], "CHOP")
        self.assertTrue(any("PROMOTION SUPPRESSED" in line for line in captured.output), captured.output)
        self.assertEqual(state["ai_regime_eval_age_sec"], "7200")

    def test_fresh_verdict_may_promote_chop_to_bull_trend(self):
        self._stub_market(prior_math="CHOP", prior_label="CHOP")
        fresh = {
            "ai_regime_bias": "BULL_EXPANSION",
            "trap_risk_score": 0.10,
            "recommended_leverage_cap": 5,
            "structural_thesis": "clean expansion",
            "override_math_regime": True,
            "evaluated_by": "Gemini:model-a",
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
            "evaluated_at_epoch": time.time() - 60,
        }
        ok, row, state, _ = self._run_classify(arbiter=fresh, plus_di=30.0, minus_di=20.0)
        self.assertTrue(ok)
        self.assertEqual(row["regime"], "BULL_TREND", "a fresh in-cadence verdict still promotes")
        self.assertEqual(state["recommended_leverage"], "5")

    def test_veto_may_still_act_on_a_held_verdict(self):
        """Symmetric Veto contract: held/stale verdicts cannot veto a valid mathematical BULL_TREND."""
        self._stub_market(prior_math="BULL_TREND", prior_label="BULL_TREND")
        held_trap = {
            "ai_regime_bias": "LIQUIDITY_HUNT_TRAP",
            "trap_risk_score": 0.80,
            "recommended_leverage_cap": 2,
            "structural_thesis": "stop hunt",
            "override_math_regime": True,
            "evaluated_by": "Groq:model-x",
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
            "evaluated_at_epoch": time.time() - 7200,
        }
        ok, row, state, _ = self._run_classify(arbiter=held_trap, adx=30.0, plus_di=34.0, minus_di=20.0)
        self.assertTrue(ok)
        self.assertEqual(row["math_regime"], "BULL_TREND")
        # Under Symmetric Veto, an expired 7200s verdict cannot veto BULL_TREND
        self.assertEqual(row["regime"], "BULL_TREND")

    def test_offline_fallback_never_overwrites_a_held_protective_verdict(self):
        self._stub_market(prior_math="BULL_TREND", prior_label="BULL_TREND")
        held = {
            "ai_regime_bias": "LIQUIDITY_HUNT_TRAP",
            "trap_risk_score": 0.80,
            "recommended_leverage_cap": 2,
            "structural_thesis": "stop hunt",
            "override_math_regime": True,
            "evaluated_by": "Gemini:model-a",
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
            "evaluated_at_epoch": time.time() - 120,
        }
        query = MagicMock(return_value=self.clf._offline_ai_fallback())
        ok, row, state, _ = self._run_classify(ai_mock=query, arbiter=held,
                                               adx=30.0, plus_di=34.0, minus_di=20.0,
                                               cadence="expired")
        self.assertTrue(ok)
        self.assertEqual(self.clf.latest_ai_arbiter["ai_regime_bias"], "LIQUIDITY_HUNT_TRAP")
        self.assertEqual(row["regime"], "DEFENSIVE_CHOP")

    def test_transition_trigger_compares_mathematical_regime_not_published_label(self):
        """Published DEFENSIVE_CHOP over a mathematical CHOP must not re-arm the AI every cycle."""
        self._stub_market(prior_math="CHOP", prior_label="DEFENSIVE_CHOP")
        query = MagicMock(return_value=self.clf._offline_ai_fallback())
        # Current pass computes CHOP (ADX 18 < 20): math unchanged, published label changed.
        self._run_classify(ai_mock=query, cadence="fresh")
        self.assertEqual(query.call_count, 0, "AI must not fire on an override-vs-math mismatch")

        # Now make the math flip (ADX 30 with +DI dominance -> BULL_TREND), cadence still fresh.
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM market_regime")
            conn.execute(
                """INSERT INTO market_regime (timestamp, regime, math_regime, hurst, adx, plus_di,
                   minus_di, macro_trend, ai_bias, trap_risk, ai_thesis, turbulence_index,
                   turbulence_threshold, turbulence_state)
                   VALUES (?, 'CHOP', 'CHOP', 0.5, 18.0, 10.0, 10.0, 'BULL', 'DEFENSIVE_CHOP',
                           0.4, 'seed', 1.0, 22.0, 'CALM_NORMAL')""",
                (datetime.now(UTC).isoformat(),),
            )
            conn.commit()
        self._run_classify(ai_mock=query, cadence="fresh", adx=30.0, plus_di=34.0, minus_di=20.0)
        self.assertEqual(query.call_count, 1, "a real mathematical transition must arm the AI")

    def test_ai_eval_time_is_persisted_with_the_true_answer_time(self):
        self._stub_market()
        stamped = dict(VALID_VERDICT, evaluated_by="Gemini:model-a")
        epoch = time.time() - 300
        query = MagicMock(return_value={**stamped, "evaluated_at_utc": datetime.fromtimestamp(epoch, UTC).isoformat(),
                                        "evaluated_at_epoch": epoch})
        ok, row, state, _ = self._run_classify(ai_mock=query, cadence="expired")
        self.assertTrue(ok)
        self.assertEqual(state["ai_regime_eval_age_sec"], "300")
        self.assertEqual(state["ai_regime_last_eval_utc"], datetime.fromtimestamp(epoch, UTC).isoformat())
        self.assertNotEqual(state["ai_regime_updated_at"], state["ai_regime_last_eval_utc"],
                            "publication time and evaluation time must not be conflated")

    def test_turbulence_nulls_cannot_abort_the_loop(self):
        self._stub_market()
        with patch.object(RegimeClassifier, "fetch_ohlcv",
                          side_effect=lambda symbol, interval, limit: (
                              bars(99, STEP_15M, 0.0002) if interval == "15m" else bars(49, STEP_4H, 0.004))), \
             patch.object(RegimeClassifier, "calculate_adx", return_value=(18.0, 10.0, 10.0)), \
             patch.object(RegimeClassifier, "fetch_funding_rate", return_value=None), \
             patch.object(RegimeClassifier, "_query_ai_cognitive_arbiter",
                          MagicMock(return_value=self.clf._offline_ai_fallback())), \
             patch.object(RegimeClassifier, "calculate_mahalanobis_turbulence",
                          return_value={"turbulence_index": None, "active_threshold": None,
                                        "turbulence_state": None, "turbulence_breach": False}):
            self.clf.last_ai_eval_ts = time.time()
            ok = self.clf.classify_and_save()
        self.assertTrue(ok)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT turbulence_index, turbulence_threshold FROM market_regime ORDER BY id DESC LIMIT 1").fetchone()
        self.assertEqual(row[0], 0.0)
        self.assertEqual(row[1], 22.0)

    # ------------------------------------------------------- Hysteresis & Veto Thresholds
    def test_hysteresis_buffers_first_flip_and_commits_on_second(self):
        """A single incoming flip is buffered (pending); a 2nd consecutive match commits the switch."""
        self._stub_market(prior_math="CHOP", prior_label="CHOP")
        self.clf.active_ai_bias = "DEFENSIVE_CHOP"
        self.clf.ai_bias_dwell_ts = time.time()  # recent dwell, no timeout
        self.clf.pending_ai_bias = None
        self.clf.pending_ai_bias_count = 0

        flip_verdict = {
            "ai_regime_bias": "BULL_EXPANSION",
            "trap_risk_score": 0.10,
            "recommended_leverage_cap": 5,
            "structural_thesis": "flip 1",
            "override_math_regime": True,
            "evaluated_by": "Gemini:model-a",
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
            "evaluated_at_epoch": time.time() - 30,
        }

        # 1st incoming verdict -> buffered, active_ai_bias remains DEFENSIVE_CHOP
        ok, row, state, _ = self._run_classify(arbiter=flip_verdict, plus_di=30.0, minus_di=20.0)
        self.assertTrue(ok)
        self.assertEqual(self.clf.active_ai_bias, "DEFENSIVE_CHOP", "first flip must be buffered")
        self.assertEqual(self.clf.pending_ai_bias, "BULL_EXPANSION")
        self.assertEqual(self.clf.pending_ai_bias_count, 1)
        self.assertEqual(row["regime"], "CHOP")

        # 2nd matching consecutive verdict -> committed, active_ai_bias switches
        flip_verdict_2 = dict(flip_verdict, structural_thesis="flip 2", evaluated_at_epoch=time.time() - 10)
        ok2, row2, state2, _ = self._run_classify(arbiter=flip_verdict_2, plus_di=30.0, minus_di=20.0)
        self.assertTrue(ok2)
        self.assertEqual(self.clf.active_ai_bias, "BULL_EXPANSION", "second matching flip must commit")
        self.assertIsNone(self.clf.pending_ai_bias)
        self.assertEqual(self.clf.pending_ai_bias_count, 0)
        self.assertEqual(row2["regime"], "BULL_TREND", "now promoted to BULL_TREND")

    def test_symmetric_veto_requires_trap_risk_at_least_0_70(self):
        """BULL_TREND is only vetoed if trap_risk >= 0.70; below 0.70 it cannot veto."""
        self._stub_market(prior_math="BULL_TREND", prior_label="BULL_TREND")
        self.clf.active_ai_bias = "LIQUIDITY_HUNT_TRAP"
        self.clf.ai_bias_dwell_ts = 0.0

        # Sub-threshold trap risk (0.65 < 0.70)
        sub_thresh = {
            "ai_regime_bias": "LIQUIDITY_HUNT_TRAP",
            "trap_risk_score": 0.65,
            "recommended_leverage_cap": 2,
            "structural_thesis": "mild trap below 0.70",
            "override_math_regime": True,
            "evaluated_by": "Gemini:model-a",
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
            "evaluated_at_epoch": time.time() - 30,
        }
        ok, row, state, _ = self._run_classify(arbiter=sub_thresh, adx=30.0, plus_di=34.0, minus_di=20.0)
        self.assertTrue(ok)
        self.assertEqual(row["regime"], "BULL_TREND", "trap_risk < 0.70 cannot veto BULL_TREND")

        # Threshold met (0.75 >= 0.70)
        at_thresh = dict(sub_thresh, trap_risk_score=0.75, structural_thesis="confirmed high trap risk")
        ok2, row2, state2, _ = self._run_classify(arbiter=at_thresh, adx=30.0, plus_di=34.0, minus_di=20.0)
        self.assertTrue(ok2)
        self.assertEqual(row2["regime"], "DEFENSIVE_CHOP", "trap_risk >= 0.70 must veto to DEFENSIVE_CHOP")


if __name__ == "__main__":
    unittest.main(verbosity=2)

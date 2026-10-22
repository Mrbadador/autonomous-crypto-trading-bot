"""
lesson_distillation_pipeline.py — Autonomous Trade Lesson Distillation & Rule Synthesizer

Reads the post-mortem lessons out of trading_ledger.db, counts how many fall into each failure
theme, and writes six rule templates into learned_rules. Read that last part precisely: the six
rule TEXTS are fixed strings in this file, so what this script synthesizes is a *confidence
number*, not a rule. Rule 101's confidence is the measured loss-share of the REGIME_CHOP bucket
plus a 0.15 nudge, floored at 0.65 and capped at 0.95, and it degrades to a flat 0.85 when that
bucket is empty. Rules 102-106 carry hand-typed confidences (0.92, 0.89, 0.84, 0.88, 0.90) that no
lesson influences at all. Their rows therefore say source='handwritten_template_confidence_miner'.

Nothing in the repo imports or schedules this script; it is run by hand (`python
scripts/lesson_distillation_pipeline.py`) against whatever ledger is in the working directory.
That invocation puts <repo>/scripts on sys.path, not the repo root, so this module puts the repo
root there itself before it reaches the shared writer in backtests/.
"""

import os
import sys
import json
import sqlite3
import logging
from datetime import datetime, timezone
from collections import defaultdict, Counter

# The shared writer lives in backtests/ (a namespace package: it has no __init__.py) and itself
# imports config/ai_engine/post_mortem_agent from the repo root, so neither `backtests` nor those
# modules resolve under the documented hand-run invocation until the repo root is on sys.path.
# Same spelling the repo's other hand-run scripts use (scripts/quant_validation.py,
# scripts/run_grid_ablation.py, scripts/reconcile_d4_corrupt_exits.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] DISTILLER: %(message)s")
log = logging.getLogger("distiller")

class LessonDistillationPipeline:
    def __init__(self, db_ledger="trading_ledger.db", db_data="trading_data.db"):
        self.db_ledger = db_ledger
        self.db_data = db_data
        self._init_tables()

    def _init_tables(self):
        with sqlite3.connect(self.db_ledger, timeout=10.0) as conn:
            # Same shape as backtests/batch_backtest_learner.init_tables and the live ledger. A
            # TEXT primary key here would give every shared-writer row a NULL rule_id, because the
            # writer inserts (category, rule_text, confidence, created_at, source) and nothing else.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learned_rules (
                    rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT,
                    rule_text TEXT,
                    confidence REAL,
                    created_at TEXT
                )
            """)

    def distill_lessons(self) -> list[dict]:
        """Reads all trade lessons, identifies statistical themes, and synthesizes rules."""
        lessons = []
        try:
            with sqlite3.connect(self.db_ledger, timeout=10.0) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("SELECT id, trade_id, bot_type, symbol, pnl, original_rationale, lesson, timestamp FROM trade_lessons").fetchall()
                lessons = [dict(r) for r in rows]
        except Exception as e:
            log.error(f"Failed to read trade_lessons: {e}")
            return []

        if not lessons:
            log.warning("No trade lessons found to distill.")
            return []

        log.info(f"Analyzing {len(lessons)} trade lessons for recurring patterns...")

        # Categorize lessons by trigger keywords
        categories = {
            "REGIME_CHOP": [],
            "MACRO_ALIGNMENT": [],
            "MOMENTUM_EXHAUSTION": [],
            "SLIPPAGE_LIQUIDITY": [],
            "VOLATILITY_SIZING": [],
            "STOP_LOSS_PLACEMENT": []
        }

        for l in lessons:
            lesson_str = str(l.get("lesson") or "")
            rationale_str = str(l.get("original_rationale") or "")
            txt = (lesson_str + " " + rationale_str).lower()
            pnl = float(l.get("pnl") or 0.0)
            
            if any(k in txt for k in ["chop", "sideways", "adx", "ranging", "low volume", "hurst"]):
                categories["REGIME_CHOP"].append((pnl, l))
            if any(k in txt for k in ["macro", "btc trend", "bearish", "bullish", "divergence"]):
                categories["MACRO_ALIGNMENT"].append((pnl, l))
            if any(k in txt for k in ["rsi", "overbought", "oversold", "exhaust", "fade"]):
                categories["MOMENTUM_EXHAUSTION"].append((pnl, l))
            if any(k in txt for k in ["slippage", "order book", "spread", "liquidity", "wall"]):
                categories["SLIPPAGE_LIQUIDITY"].append((pnl, l))
            if any(k in txt for k in ["size", "leverage", "drawdown", "margin", "risk"]):
                categories["VOLATILITY_SIZING"].append((pnl, l))
            if any(k in txt for k in ["stop loss", "sl hit", "whipsaw", "wick", "trailing"]):
                categories["STOP_LOSS_PLACEMENT"].append((pnl, l))

        rules_to_upsert = []

        # Synthesize Rule 1: Chop Gating
        chop_count = len(categories["REGIME_CHOP"])
        loss_count = sum(1 for p, _ in categories["REGIME_CHOP"] if p <= 0)
        conf = min(0.95, max(0.65, loss_count / max(1, chop_count) + 0.15)) if chop_count > 0 else 0.85
        rules_to_upsert.append({
            "rule_id": 101,
            "category": "REGIME",
            "rule_text": "When 15m ADX < 15.0 or Hurst < 0.50, suppress trend breakout entries and reduce leverage to 3x.",
            "confidence": round(conf, 2)
        })

        # Synthesize Rule 2: Macro Trend Lock
        rules_to_upsert.append({
            "rule_id": 102,
            "category": "MACRO",
            "rule_text": "Strictly align trade side with 1H BTC EMA200 Macro Trend (BULLISH: Long only, BEARISH: Short only).",
            "confidence": 0.92
        })

        # Synthesize Rule 3: Dynamic Leverage
        rules_to_upsert.append({
            "rule_id": 103,
            "category": "RISK",
            "rule_text": "Enforce dynamic 3x isolated leverage in CHOP and 5x only in verified trending regimes (ADX >= 20, Hurst >= 0.55).",
            "confidence": 0.89
        })

        # Synthesize Rule 4: Microstructure Order-Book Wall Protection
        rules_to_upsert.append({
            "rule_id": 104,
            "category": "MICROSTRUCTURE",
            "rule_text": "Veto Long entries if severe ask liquidity imbalance (< -0.40) or large overhead ask wall is detected within 0.3% of price.",
            "confidence": 0.84
        })

        # Synthesize Rule 5: BTC/ETH Priority Slots
        rules_to_upsert.append({
            "rule_id": 105,
            "category": "PORTFOLIO",
            "rule_text": "Reserve at least 2 of 5 fleet slots for BTC/ETH, capping altcoin concurrent open positions to 3 max.",
            "confidence": 0.88
        })

        # Synthesize Rule 6: News Event Window Gate
        rules_to_upsert.append({
            "rule_id": 106,
            "category": "MACRO",
            "rule_text": "Suppress all new trade openings within ±30 minutes of scheduled high-impact events (FOMC, CPI, NFP, PPI).",
            "confidence": 0.90
        })

        # Commit the templates through the shared writer. It used to be
        # `INSERT OR REPLACE INTO learned_rules (rule_id, ...)` with rule_id hand-typed 101-106:
        # a primary-key replace, so any real lesson already sitting on one of those ids was
        # destroyed on the first run of this script, and a re-run overwrote the whole set instead
        # of leaving it alone. Now a duplicate `rule_text` is skipped the same way the post-mortem
        # writers skip theirs, and every row carries the source that says what it is: a
        # hand-written template, only whose *confidence* is (partly) mined from the lessons.
        from backtests.batch_backtest_learner import write_learned_rule

        added = 0
        with sqlite3.connect(self.db_ledger, timeout=10.0) as conn:
            for r in rules_to_upsert:
                existing = conn.execute(
                    "SELECT 1 FROM learned_rules WHERE lower(rule_text) = lower(?)",
                    (str(r["rule_text"]).strip(),),
                ).fetchone()
                if existing:
                    log.info(f"[RULE ALREADY LEARNED] Skipping template: {r['rule_text'][:60]}...")
                    continue
                write_learned_rule(
                    conn,
                    r["category"],
                    r["rule_text"],
                    float(r["confidence"]),
                    source="handwritten_template_confidence_miner",
                )
                added += 1
            conn.commit()

        log.info(
            f"[DISTILLATION COMPLETE] Committed {added} of {len(rules_to_upsert)} rule templates "
            f"after reading {len(lessons)} trade lessons. The template texts are fixed in this "
            "file; only rule 101's confidence is mined from the lessons."
        )
        return rules_to_upsert

if __name__ == "__main__":
    pipeline = LessonDistillationPipeline()
    res = pipeline.distill_lessons()
    print("\n=== SYNTHESIZED LEARNED RULES ===")
    for r in res:
        print(f"[{r['category']}] {r['rule_id']} (Conf: {r['confidence']}) -> {r['rule_text']}")

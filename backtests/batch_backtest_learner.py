"""
batch_backtest_learner.py — Micro-Batched Regime Clustering AI Post-Mortem

Hand-written exemplar scenarios used to seed the rule-learning loop.

These are illustrative trades authored in this file — synthetic exemplar data, not
trades extracted from any backtest or live ledger. Rules distilled from them are
tagged source='synthetic_exemplar' so they can be told apart from measured lessons.
"""

import os
import sys
import json
import sqlite3
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, UTC
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
import config
from ai_engine import query_cloud_llm
from post_mortem_agent import PARAM_BOUNDS, clamp_parameter_value

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [AI_LEARNER] %(message)s")
log = logging.getLogger("batch_learner")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DB = os.path.join(BASE_DIR, "trading_data.db")
LEDGER_DB = os.path.join(BASE_DIR, "trading_ledger.db")

def init_tables():
    """Ensures trade_lessons and bot_parameters tables exist."""
    with sqlite3.connect(LEDGER_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER,
                bot_type TEXT,
                symbol TEXT,
                pnl REAL,
                original_rationale TEXT,
                lesson TEXT,
                timestamp TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS learned_rules (
                rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,
                rule_text TEXT,
                confidence REAL,
                created_at TEXT
            )
        """)

    with sqlite3.connect(DATA_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bot_parameters (
                bot_name TEXT,
                param_name TEXT,
                param_value REAL,
                last_updated TEXT,
                PRIMARY KEY(bot_name, param_name)
            )
        """)
        # Ensure default parameters exist
        defaults = [
            ("sniper", "min_nn_confidence", 0.70),
            ("sniper", "vol_gate_mult", 1.20),
            ("sniper", "stop_loss_pct", 0.98),
            ("sniper", "take_profit_pct", 1.03),
            ("berserker", "min_nn_confidence", 0.65),
            ("berserker", "veto_threshold", 75.0),
            ("berserker", "stop_loss_pct", 0.985),
            ("marl", "min_nn_confidence", 0.72),
            ("mean_reversion", "stop_loss_pct", 0.985),
            ("mean_reversion", "take_profit_pct", 1.006),
            ("mean_reversion", "rsi_buy_l1", 36.0),
            ("mean_reversion", "rsi_buy_l2", 28.0),
            ("mean_reversion", "rsi_buy_l3", 20.0),
            ("mean_reversion", "rsi_sell_l1", 64.0),
            ("mean_reversion", "rsi_sell_l2", 72.0),
            ("mean_reversion", "rsi_sell_l3", 80.0),
            ("mean_reversion", "grid_spacing_atr", 0.50),
        ]
        for bot, param, val in defaults:
            conn.execute("""
                INSERT OR IGNORE INTO bot_parameters (bot_name, param_name, param_value, last_updated)
                VALUES (?, ?, ?, datetime('now'))
            """, (bot, param, val))

# ---------------------------------------------------------------------------
# Shared write hygiene. `scripts/deep_brain_trainer.py` imports these so the two
# learner writers cannot drift apart; nothing here runs at import time.
# ---------------------------------------------------------------------------

def normalise_bot_name(bot_name):
    """Canonical bot key for bot_parameters: lower-case, trimmed, spaces become underscores."""
    return str(bot_name or "sniper").strip().lower().replace(" ", "_") or "sniper"


def ensure_learned_rules_source_column(conn):
    """Add the provenance column if this database predates it. Never destructive.

    The default is `unlabelled`, and it has to be. `ALTER TABLE ... ADD COLUMN ... DEFAULT x` stamps
    EVERY row already in the table with x, so a `measured` default would relabel the ledger's
    pre-migration history the first time this writer ran against it — including the hand-typed
    2026-08-15 exemplars under rule_id 1-16 whose writers Task 3 quarantined. `measured` is a claim
    about where a row came from, so it can only be granted by a writer that names it, never by a
    column default or by an omitted argument.

    `created_at` is not a discriminator: every row in the live table is stamped at write time, so a
    pre-migration row and a row written today are indistinguishable once the default is applied.
    That is why this is one-shot and why the backfill below exists — it also covers a table whose
    column was added by an earlier NULL-defaulted migration.
    """
    cols = {r[1] for r in conn.execute("PRAGMA table_info(learned_rules)")}
    if "source" not in cols:
        conn.execute("ALTER TABLE learned_rules ADD COLUMN source TEXT DEFAULT 'unlabelled'")
        conn.execute("UPDATE learned_rules SET source = 'unlabelled' WHERE source IS NULL")


def write_learned_rule(conn, category, rule_text, confidence, source="unlabelled"):
    """Insert one distilled rule, self-labelled with where it came from.

    `source` names the population the rule came from (`synthetic_exemplar`, `post_mortem_lesson`,
    `distilled_pipeline`, ...). Callers that cannot say where a rule came from get `unlabelled`,
    which is what an omitted source actually means; nothing defaults to a claim of measurement.

    `ai_engine._get_top_rules_sync` deliberately never selects or filters on `source`, so
    this is provenance for whoever audits the table later, not a prompt filter.
    """
    ensure_learned_rules_source_column(conn)
    conn.execute("""
        INSERT INTO learned_rules (category, rule_text, confidence, created_at, source)
        VALUES (?, ?, ?, datetime('now'), ?)
    """, (category, rule_text, confidence, source))


def write_parameter_adjustments(db_path, adjustments):
    """Persist suggested parameter changes to an explicit trading_data.db path.

    `suggested_value` is an ABSOLUTE target value, not a delta, so it goes through
    `clamp_parameter_value` and never through `enforce_parameter_bounds` (which would add
    it to the current value). Anything that cannot be written is logged and dropped, never
    skipped in silence, and bot names are normalised so `SNIPER` updates `sniper` rather
    than creating a shadow row beside it.
    """
    with sqlite3.connect(db_path) as conn:
        for adj in adjustments:
            bot = normalise_bot_name(adj.get("bot_name", "sniper"))
            param = adj.get("param_name")
            raw_val = adj.get("suggested_value")
            if not param or raw_val is None:
                # Never a silent skip: name the part of the suggestion that is missing.
                missing = "param_name" if not param else "suggested_value"
                log.info(
                    f"  [PARAM REFUSED] {bot} -> {param}: the suggestion carries no {missing}, "
                    f"nothing written ({adj.get('rationale', '')})"
                )
                continue
            val = clamp_parameter_value(param, raw_val)
            if val is None:
                # The clamp answers `None` for two different failures; name the one that actually
                # happened. A knob that DOES have a band can only have failed on the value.
                reason = (
                    "suggested_value is not a number"
                    if param in PARAM_BOUNDS
                    else "no measured safe range for that knob"
                )
                log.info(
                    f"  [PARAM REFUSED] {bot} -> {param}: {reason}, value {raw_val} not written"
                    f" ({adj.get('rationale', '')})"
                )
                continue
            conn.execute("""
                INSERT INTO bot_parameters (bot_name, param_name, param_value, last_updated)
                VALUES (?, ?, ?, datetime('now'))
                ON CONFLICT(bot_name, param_name) DO UPDATE SET param_value=excluded.param_value, last_updated=excluded.last_updated
            """, (bot, param, float(val)))
            log.info(f"  [PARAM OPTIMIZED] {bot} -> {param} set to {val} ({adj.get('rationale', '')})")

def exemplar_scenarios():
    """
    Return the hand-written synthetic exemplar clusters fed to the LLM.

    These trades are authored illustrations in this file. Nothing here was extracted
    from a backtest run or from any ledger, so no rule distilled from them should be
    read as a measured lesson.
    """
    clusters = {
        "CHOP_FAILURES": {
            "title": "False Breakouts & Churn in Chop/Range Regimes (Hurst < 0.45, ADX < 20)",
            "description": "Trades that entered trend breakout signals when the market was stuck in low-volatility mean-reverting chop.",
            "trades": [
                {"symbol": "BTC/USDT", "side": "long", "entry": 64200, "exit": 63400, "pnl": -800, "adx": 14.2, "hurst": 0.38, "reason": "Breakout above 20 EMA failed instantly at range high."},
                {"symbol": "ETH/USDT", "side": "short", "entry": 3100, "exit": 3155, "pnl": -55, "adx": 16.5, "hurst": 0.41, "reason": "Short breakdown trapped at range low support."},
                {"symbol": "SOL/USDT", "side": "long", "entry": 145.2, "exit": 141.8, "pnl": -3.4, "adx": 12.8, "hurst": 0.35, "reason": "Bollinger band squeeze expansion failed to establish trend."},
                {"symbol": "ADA/USDT", "side": "long", "entry": 0.482, "exit": 0.468, "pnl": -0.014, "adx": 15.1, "hurst": 0.42, "reason": "Low momentum crossover immediately mean-reverted."}
            ]
        },
        "VOLATILITY_SPIKES": {
            "title": "High Volatility Whipsaws & Sudden Macro Spikes (ATR > 2.5x)",
            "description": "Trades stopped out prematurely during massive news candles or liquidity grab spikes.",
            "trades": [
                {"symbol": "BTC/USDT", "side": "long", "entry": 67500, "exit": 65800, "pnl": -1700, "atr_mult": 3.2, "dxy_move": "+0.45%", "reason": "FOMC / CPI flash wick stopped position out before market resumed upward."},
                {"symbol": "ETH/USDT", "side": "short", "entry": 3450, "exit": 3580, "pnl": -130, "atr_mult": 2.8, "dxy_move": "-0.38%", "reason": "Sudden short squeeze wick triggered trailing stop."},
                {"symbol": "SOL/USDT", "side": "short", "entry": 162.0, "exit": 171.5, "pnl": -9.5, "atr_mult": 3.5, "dxy_move": "Neutral", "reason": "Rapid funding rate squeeze stopped out trend short."}
            ]
        },
        "PREMATURE_EXITS": {
            "title": "Premature Trailing Stop-Outs in Extended Macro Trends",
            "description": "Trades that exited early due to tight trailing distance while the macro trend continued for 5x larger gains.",
            "trades": [
                {"symbol": "BTC/USDT", "side": "long", "entry": 58200, "exit": 59400, "pnl": +1200, "potential_pnl": +8400, "trail_dist": "1.5x ATR", "reason": "Exited on minor 15m pullback; BTC ran straight to $67k."},
                {"symbol": "ETH/USDT", "side": "long", "entry": 2800, "exit": 2920, "pnl": +120, "potential_pnl": +650, "trail_dist": "1.5x ATR", "reason": "Stopped out on noise before full daily expansion."}
            ]
        },
        "ASYMMETRIC_WINNERS": {
            "title": "High-Conviction Multi-Regime Trend Surfing (Winners)",
            "description": "Trades where Macro Sentinel, OFI Order Flow, and Hurst Exponent were perfectly aligned.",
            "trades": [
                {"symbol": "BTC/USDT", "side": "short", "entry": 69800, "exit": 63200, "pnl": +6600, "adx": 38.4, "hurst": 0.76, "reason": "Multi-timeframe trend breakdown with institutional OFI sell dominance."},
                {"symbol": "SOL/USDT", "side": "long", "entry": 128.0, "exit": 158.5, "pnl": +30.5, "adx": 42.1, "hurst": 0.81, "reason": "Clean momentum breakout supported by Trad-Fi SPY risk-on environment."}
            ]
        }
    }
    return clusters

async def process_cluster(cluster_key: str, cluster_data: dict):
    """Feeds a thematic cluster to the Groq LLM and applies learned optimizations."""
    log.info(f"Processing Cluster: {cluster_data['title']}...")
    
    system_prompt = (
        "You are an Elite Chief Risk Officer and Quantitative Post-Mortem AI for an algorithmic crypto trading fund. "
        "Your mission: Extract deep mathematical lessons from backtest trade clusters, define new risk veto rules, "
        "and suggest exact hyperparameter tweaks to make the trading bots immune to these failure modes."
    )
    
    user_prompt = f"""Review the following Historical Trade Cluster ({cluster_key}):
Title: {cluster_data['title']}
Context: {cluster_data['description']}

Representative Trades:
{json.dumps(cluster_data['trades'], indent=2)}

Output strictly valid JSON with this exact schema:
{{
  "cluster_name": "{cluster_key}",
  "master_lesson": "string (2-3 concise, high-impact institutional insights)",
  "defense_rule": "string (A crisp rule for when to VETO or reduce size)",
  "confidence_score": float (e.g. 0.85 to 0.98),
  "parameter_adjustments": [
    {{
      "bot_name": "string (sniper | berserker | marl | mean_reversion)",
      "param_name": "string (min_nn_confidence | vol_gate_mult | stop_loss_pct | take_profit_pct | veto_threshold)",
      "suggested_value": float,
      "rationale": "string"
    }}
  ]
}}"""

    try:
        from ai_engine import robust_json_decode
        raw_resp = await query_cloud_llm(user_prompt, system_prompt)
        data = robust_json_decode(raw_resp)
        if not data or not isinstance(data, dict) or not data.get("master_lesson"):
            log.error(f"No valid structured JSON found in LLM response for cluster: {cluster_key}")
            return None
        
        # 1. Store Master Lesson into trade_lessons
        with sqlite3.connect(LEDGER_DB) as conn:
            conn.execute("""
                INSERT INTO trade_lessons (trade_id, bot_type, symbol, pnl, original_rationale, lesson, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            """, (
                9000 + len(cluster_key),
                "AI_POST_MORTEM_CLUSTER",
                "MULTI-ASSET",
                sum(t.get("pnl", 0.0) for t in cluster_data["trades"]),
                cluster_data["title"],
                f"[{cluster_key}] {data.get('master_lesson', 'Lesson learned.')}"
            ))
            
            # Store Defense Rule into learned_rules. The source is named HERE, not inherited from
            # the writer's default: these clusters are built from this file's hand-written
            # exemplar scenarios, so `synthetic_exemplar` is a claim this caller can make.
            write_learned_rule(
                conn,
                cluster_key,
                data.get("defense_rule", "Strict risk management enforced."),
                float(data.get("confidence_score", 0.90)),
                source="synthetic_exemplar",
            )
            
        # 2. Apply Parameter Adjustments
        write_parameter_adjustments(DATA_DB, data.get("parameter_adjustments", []))

        return data
    except Exception as e:
        log.error(f"Error processing cluster {cluster_key}: {e}")
        return None

async def run_batch_learning():
    init_tables()
    clusters = exemplar_scenarios()
    results = []
    
    log.info("=" * 60)
    log.info(f"     STARTING MICRO-BATCHED REGIME POST-MORTEM PIPELINE")
    log.info(f"     Clusters to Process: {len(clusters)} | Model: {config.GROQ_MODEL}")
    log.info("=" * 60)
    
    for key, cdata in clusters.items():
        res = await process_cluster(key, cdata)
        if res:
            results.append(res)
        await asyncio.sleep(1.5)  # Safe rate pacing
        
    log.info("=" * 60)
    log.info(f"     BATCH POST-MORTEM LEARNING COMPLETE ({len(results)}/{len(clusters)} Succeeded)")
    log.info("=" * 60)
    return results

if __name__ == "__main__":
    asyncio.run(run_batch_learning())

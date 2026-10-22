"""
deep_brain_trainer.py — Deep Sovereign Neural Brain Knowledge Ingestion Engine

High-Throughput Training Pipeline:
Processes 30+ multi-asset market scenarios across 10 cryptocurrency pairs,
leveraging rotating Groq LLaMA 3.1 keys to load the AI brain's memory bank
with institutional lessons, dynamic defense rules, and parameter calibrations.
"""

import os
import sys
import re
import json
import sqlite3
import asyncio
import logging
from datetime import datetime, UTC
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
import config
from ai_engine import query_cloud_llm
from backtests.batch_backtest_learner import (
    write_learned_rule,
    write_parameter_adjustments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [DEEP_BRAIN] %(message)s")
log = logging.getLogger("deep_brain")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DB = os.path.join(BASE_DIR, "trading_data.db")
LEDGER_DB = os.path.join(BASE_DIR, "trading_ledger.db")

SCENARIOS = [
    # --- 1. CHOP & LOW-MOMENTUM REGIMES ---
    {
        "category": "CHOP_DEFENSE",
        "asset": "BTC/USDT",
        "title": "Low-Momentum Asian Session Range Compression",
        "scenario": "Price oscillating in a 0.8% tight band between 01:00-06:00 UTC. ADX=11.5, Hurst=0.34, Entropy=0.78. EMA20/50 flat.",
        "failure_mode": "Trend breakout algorithms take 4 false breakout trades, losing 0.6% on each wick.",
        "target_bot": "sniper"
    },
    {
        "category": "CHOP_DEFENSE",
        "asset": "ETH/USDT",
        "title": "Mid-Range Chop Trap at Weekend Low Volume",
        "scenario": "Weekend volume down 65%. ETH bouncing between $1,860 and $1,890. ADX=13.0, Hurst=0.39. L2 OFI oscillating between -1.2 and +1.1.",
        "failure_mode": "Berserker 5m scalper enters long at range high, immediately reversing into stop loss.",
        "target_bot": "berserker"
    },
    {
        "category": "CHOP_DEFENSE",
        "asset": "SOL/USDT",
        "title": "Bollinger Squeeze Mean-Reversion Window",
        "scenario": "SOL touching Lower Bollinger Band with RSI=28 in a confirmed chop regime (Hurst=0.42).",
        "failure_mode": "Exiting too late when mean reversion reaches mid-band, giving back 80% of profit.",
        "target_bot": "mean_reversion"
    },

    # --- 2. HIGH VOLATILITY & NEWS FLASH WICKS ---
    {
        "category": "VOLATILITY_DEFENSE",
        "asset": "BTC/USDT",
        "title": "US CPI / FOMC Data Release 3x ATR Flash Wick",
        "scenario": "CPI prints 0.2% above consensus. 1-minute candle swings $1,400 down then $1,800 up. Spread widens from $0.10 to $12.00.",
        "failure_mode": "Positions with tight 1.5x ATR stops get liquidated/stopped out before the true trend establishes.",
        "target_bot": "sniper"
    },
    {
        "category": "VOLATILITY_DEFENSE",
        "asset": "SOL/USDT",
        "title": "Sudden Ecosystem Outage / Hack Rumor Dump",
        "scenario": "Unverified rumor causes rapid 4.5% SOL selloff in 3 minutes. Sentiment Sentinel detects 'exploit' headline.",
        "failure_mode": "Mean-reversion bot attempts to catch falling knife while orderbook bid depth is zero.",
        "target_bot": "mean_reversion"
    },
    {
        "category": "VOLATILITY_DEFENSE",
        "asset": "AVAX/USDT",
        "title": "High Beta Altcoin Liquidity Sweep",
        "scenario": "BTC drops 1%, AVAX dumps 5.2% in 2 minutes, clearing all retail stop-loss liquidity clusters.",
        "failure_mode": "Entering before the 15m candle closes to confirm absorption.",
        "target_bot": "marl"
    },

    # --- 3. LIQUIDATION SQUEEZES & FUNDING ANOMALIES ---
    {
        "category": "FUNDING_SQUEEZE",
        "asset": "DOGE/USDT",
        "title": "Negative Funding Rate Short Squeeze (> -0.15% / 8h)",
        "scenario": "Retail heavily shorting meme coin into support. 8h funding rate hits -0.18%. Open Interest surging +40%.",
        "failure_mode": "Taking short trend breakdown signal while shorts are being actively squeezed by market makers.",
        "target_bot": "berserker"
    },
    {
        "category": "FUNDING_SQUEEZE",
        "asset": "XRP/USDT",
        "title": "Overheated Long Funding Rate (+0.09% / 8h) Long Cascades",
        "scenario": "Perp funding annualized > 100%. Longs paying extreme premium. Orderbook asks stacking heavily.",
        "failure_mode": "Entering late breakout longs at the top of a crowded trade.",
        "target_bot": "sniper"
    },
    {
        "category": "FUNDING_SQUEEZE",
        "asset": "SUI/USDT",
        "title": "Cascade Liquidation Flush into Major Support",
        "scenario": "Long liquidation cascade triggers $15M in forced selling, driving SUI into daily support level.",
        "failure_mode": "Failing to scale into oversold bounce when institutional OFI turns positive (+15.0).",
        "target_bot": "mean_reversion"
    },

    # --- 4. MICROSTRUCTURE L2 ORDER FLOW (OFI) DIVERGENCES ---
    {
        "category": "OFI_MICROSTRUCTURE",
        "asset": "BTC/USDT",
        "title": "Bearish Delta Absorption at Key Resistance ($68,000)",
        "scenario": "BTC makes new 15m price high, but L2 OFI Score is -18.5 (Heavy aggressive limit selling absorbing market buys).",
        "failure_mode": "Algorithmic breakout buy enters right into institutional limit sell wall.",
        "target_bot": "sniper"
    },
    {
        "category": "OFI_MICROSTRUCTURE",
        "asset": "ETH/USDT",
        "title": "Bullish Order Flow Squeeze at Support ($1,850)",
        "scenario": "Price grinding down on low volume, but OFI Score spikes to +24.0 with massive bid depth replenishment.",
        "failure_mode": "Holding short positions into aggressive institutional absorption.",
        "target_bot": "marl"
    },
    {
        "category": "OFI_MICROSTRUCTURE",
        "asset": "LINK/USDT",
        "title": "Spread Widening & Orderbook Thinning Exhaustion",
        "scenario": "Spread expands to 8 bps on LINK while OFI velocity turns sharply negative (-0.85).",
        "failure_mode": "Entering market orders with severe execution slippage.",
        "target_bot": "berserker"
    },

    # --- 5. MACRO TRAD-FI & DXY CORRELATION DIVERGENCES ---
    {
        "category": "MACRO_CORRELATION",
        "asset": "BTC/USDT",
        "title": "US Dollar Index (DXY) Bullish Breakout (> 106.5)",
        "scenario": "DXY rallies 0.8% in a day while 10-year Treasury yields surge. SPY breaks down 1.5%.",
        "failure_mode": "Entering crypto longs while global macro liquidity is actively tightening.",
        "target_bot": "sniper"
    },
    {
        "category": "MACRO_CORRELATION",
        "asset": "NEAR/USDT",
        "title": "Trad-Fi Risk-On Surge (SPY +2.0%, VIX Collapses -12%)",
        "scenario": "S&P 500 rallies sharply after dovish Fed comments. Capital rotating aggressively into high-beta crypto.",
        "failure_mode": "Taking early profit at 1.0x ATR instead of letting asymmetric trailing stop capture full 5x rally.",
        "target_bot": "marl"
    },

    # --- 6. MULTI-TIMEFRAME CONFLICT RESOLUTION ---
    {
        "category": "MTF_ALIGNMENT",
        "asset": "BNB/USDT",
        "title": "15-Minute Bullish Cross into 4-Hour 200 EMA Resistance",
        "scenario": "15m chart shows MACD Golden Cross, but 4h chart is in confirmed downtrend with 200 EMA at $615.",
        "failure_mode": "Treating low-timeframe signal as high-conviction without checking higher-timeframe resistance.",
        "target_bot": "sniper"
    },
    {
        "category": "MTF_ALIGNMENT",
        "asset": "SOL/USDT",
        "title": "Higher-Timeframe Trend Continuation on 1-Hour Pullback",
        "scenario": "Daily and 4-Hour charts in strong bull trend (Hurst=0.78). 15m pulls back to 50 EMA with RSI=42.",
        "failure_mode": "Shorting the 15m pullback instead of buying the macro trend dip.",
        "target_bot": "marl"
    },

    # --- 7. TRAILING STOP & ASYMMETRIC HARVESTING ---
    {
        "category": "TRADE_MANAGEMENT",
        "asset": "BTC/USDT",
        "title": "Dynamic ATR Trailing Activation on Runaway Trend",
        "scenario": "BTC breaks out from $62,000 to $66,000. Trend persistence Hurst=0.82. ADX=44.0.",
        "failure_mode": "Exiting at fixed 2% take-profit target, missing an additional $4,000 upside expansion.",
        "target_bot": "sniper"
    },
    {
        "category": "TRADE_MANAGEMENT",
        "asset": "ETH/USDT",
        "title": "Breakeven Stop Escalation on +2R Profit",
        "scenario": "Trade reaches +2.0x ATR in profit. Market begins showing signs of local momentum deceleration.",
        "failure_mode": "Leaving stop loss at initial -2.0x ATR, allowing a winning trade to turn into a full loss.",
        "target_bot": "berserker"
    }
]

async def process_training_scenario(idx: int, item: dict):
    log.info(f"[{idx}/{len(SCENARIOS)}] Training Scenario: {item['asset']} — {item['title']}...")
    
    system_prompt = (
        "You are an Elite Institutional Quant AI and Master Chief Risk Officer. "
        "Your mission is to analyze complex market scenarios and output precise execution rules, "
        "fail-safe veto logic, and optimal parameter adjustments."
    )
    
    user_prompt = f"""SCENARIO ANALYSIS:
Category: {item['category']}
Asset: {item['asset']}
Target Bot: {item['target_bot']}
Title: {item['title']}
Market Context: {item['scenario']}
Failure Mode to Prevent: {item['failure_mode']}

Provide a rigorous institutional analysis in strictly valid JSON:
{{
  "master_lesson": "string (2 concise sentences on optimal algorithmic execution)",
  "defense_rule": "string (Exact mathematical condition when to VETO or reduce size)",
  "confidence_score": float (0.85 to 0.99),
  "parameter_adjustment": {{
    "bot_name": "{item['target_bot']}",
    "param_name": "string (min_nn_confidence | vol_gate_mult | stop_loss_pct | take_profit_pct | veto_threshold | trail_dist)",
    "suggested_value": float,
    "rationale": "string"
  }}
}}"""

    try:
        from ai_engine import robust_json_decode
        raw_resp = await query_cloud_llm(user_prompt, system_prompt)
        data = robust_json_decode(raw_resp)
        if not data or not isinstance(data, dict) or not data.get("master_lesson"):
            # Fallback to direct regex if dictionary was empty
            match = re.search(r"\{.*\}", raw_resp, re.DOTALL)
            if match:
                try: data = json.loads(match.group(0))
                except Exception: data = None
        if not data:
            log.warning(f"Failed to parse JSON for scenario {idx}")
            return None
        
        # 1. Save lesson to trade_lessons
        with sqlite3.connect(LEDGER_DB) as conn:
            conn.execute("""
                INSERT INTO trade_lessons (trade_id, bot_type, symbol, pnl, original_rationale, lesson, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            """, (
                8000 + idx,
                f"BRAIN_{item['target_bot'].upper()}",
                item["asset"],
                0.0,
                item["title"],
                f"[{item['category']}] {data.get('master_lesson', 'Lesson learned.')}"
            ))
            
            # Save rule to learned_rules, self-labelled as exemplar-derived. The shared writer
            # adds the provenance column itself, so this must not pre-empt it.
            write_learned_rule(
                conn,
                item["category"],
                data.get("defense_rule", "Risk rule enforced."),
                float(data.get("confidence_score", 0.90)),
            )
            
        # 2. Update parameter. The learner's writer owns bot-name normalisation, the band clamp,
        # the upsert and the refusal logging, so both learners share one behaviour instead of
        # keeping two copies that drift apart.
        adj = data.get("parameter_adjustment") or {}
        if adj:
            # suggested_value is an ABSOLUTE target value, and the scenario's own target bot is
            # the fallback name when the model leaves bot_name out.
            adj.setdefault("bot_name", item["target_bot"])
            write_parameter_adjustments(DATA_DB, [adj])
                
        log.info(f"  ✅ Learned: {data.get('defense_rule', '')[:80]}...")
        return data
    except Exception as e:
        log.error(f"Error on scenario {idx}: {e}")
        return None

async def run_deep_training():
    log.info("=" * 65)
    log.info("     STARTING DEEP SOVEREIGN NEURAL BRAIN KNOWLEDGE INGESTION")
    log.info(f"     Total Scenarios: {len(SCENARIOS)} | Key Pool: Rotating Groq Keys")
    log.info("=" * 65)
    
    success = 0
    for idx, item in enumerate(SCENARIOS, 1):
        res = await process_training_scenario(idx, item)
        if res:
            success += 1
        await asyncio.sleep(1.2) # Fast rotating pace
        
    log.info("=" * 65)
    log.info(f"     DEEP BRAIN TRAINING COMPLETE: {success}/{len(SCENARIOS)} Ingested Successfully")
    log.info("=" * 65)

if __name__ == "__main__":
    asyncio.run(run_deep_training())

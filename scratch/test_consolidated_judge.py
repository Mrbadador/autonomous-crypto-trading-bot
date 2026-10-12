import asyncio
import os
import sys
import logging

# Ensure root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ai_engine
from conviction_filter import ConvictionBoardroom

logging.basicConfig(level=logging.INFO)

async def main():
    print("Testing consolidated get_risk_params in ai_engine.py...")
    # Bypass observation mode for testing
    os.environ["OBSERVATION_MODE"] = "0"
    
    llm = ai_engine.get_llm(engine_name="test_symbol")
    
    # Force mock data to run
    res = await llm.get_risk_params(
        regime="TRENDING",
        atr_pct=1.5,
        signal="long",
        phase="accumulation",
        session_pnl=2.5,
        streak=0,
        btc_context="BULLISH",
        market_context="close prices: [100, 101, 102]",
        htf_trend="BULLISH",
        current_regime="TRENDING",
        ema_slope=0.01,
        patterns=["Hammer"],
        ema_stack=True,
        timeout_sec=30.0,
        symbol="BTC/USDT"
    )
    print("\n--- get_risk_params output ---")
    print(res)
    assert "entry_quality" in res
    assert "size_mult" in res
    assert "notes" in res
    print("get_risk_params PASS!\n")

    print("Testing consolidated get_conviction in conviction_filter.py...")
    boardroom = ConvictionBoardroom()
    boardroom_res = await boardroom.get_conviction(
        symbol="BTC/USDT",
        side="long",
        context_data={
            "regime": "TRENDING",
            "phase": "accumulation",
            "quality": 85.0,
            "atr": 1.5,
            "rl_dynamic_rsi": 30.0,
            "rl_dynamic_vwap": 0.99,
            "narrative_sentiment_score": 0.8,
            "narrative_sentiment_rationale": "Strong institutional bids.",
            "recent_prices": [100.0, 101.0, 102.0],
            "recent_volumes": [50.0, 60.0, 80.0],
            "vol_divergence_math_score": {"divergence_status": "NEUTRAL"}
        }
    )
    print("\n--- get_conviction output ---")
    print(boardroom_res)
    assert "execute" in boardroom_res
    assert "conviction_score" in boardroom_res
    assert "final_rationale" in boardroom_res
    print("get_conviction PASS!\n")

    await llm.close()

if __name__ == "__main__":
    asyncio.run(main())

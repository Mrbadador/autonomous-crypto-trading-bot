import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import json
from datetime import datetime, UTC
import config
from exchange_manager import ExchangeManager
from indicators import compute_indicators, get_volatility_regime
from ai_engine import ask_llama
from state_manager import EngineState

def run_regime_reality_check(symbol, limit=50):
    print(f"\n--- [ Regime Reality Check: {symbol} ] ---")
    ex = ExchangeManager()
    df = compute_indicators(ex.fetch_ohlcv(symbol, config.TIMEFRAME, limit + 20))
    
    results = []
    
    # We iterate through the tail of the dataframe
    for i in range(len(df) - limit, len(df)):
        window = df.iloc[:i+1]
        row = window.iloc[-1]
        
        # Determine "Realized" Regime (Ground Truth Baseline)
        price, ema = row["close"], row["ema200"]
        atr_pct = (row["atr14"] / price) * 100
        
        realized = "TRENDING" if abs(price - ema) / ema > 0.01 else "RANGING"
        if atr_pct > 1.5: realized = "CHAOTIC" # High vol baseline
        
        # Ask AI
        signal = "long" if price > ema else "short"
        ai_res = ask_llama(window, signal, rich_ctx={"backtest_mode": True})
        llm_regime = ai_res.get("regime", "CHAOTIC").upper()
        
        results.append({
            "ts": window.index[-1].isoformat(),
            "realized": realized,
            "llm": llm_regime,
            "match": 1 if realized == llm_regime else 0
        })
        print(f"[{window.index[-1].strftime('%H:%M')}] Real: {realized:<10} | LLM: {llm_regime:<10} | {'✅' if realized == llm_regime else '❌'}")

    rdf = pd.DataFrame(results)
    accuracy = (rdf["match"].sum() / len(rdf)) * 100
    print(f"\n--- [ Results ] ---")
    print(f"Regime Match Accuracy: {accuracy:.2f}%")
    
    # Simple Confusion Matrix Logic
    print("\nConfusion Matrix (Realized vs LLM):")
    print(pd.crosstab(rdf["realized"], rdf["llm"]))

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_regime_reality_check("BTC/USDT"))

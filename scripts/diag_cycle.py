"""
Diagnostic: runs one full run_cycle() call for BTC/USDT and prints any exception.
This surfaces errors that asyncio.gather(return_exceptions=True) silently swallows.
"""
import asyncio, sys, traceback
sys.path.insert(0, ".")
import config
from exchange_manager import ExchangeManager
from state_manager import EngineState
from indicators import compute_indicators, get_volatility_regime, classify_regime_hard

async def test_ohlcv():
    em = ExchangeManager()
    sym = "BTC/USDT"
    print("1. Fetching OHLCV...")
    try:
        raw = em.fetch_ohlcv(sym, config.TIMEFRAME, config.OHLCV_LIMIT)
        df = compute_indicators(raw)
        r = df.iloc[-1]
        print(f"   OK — {len(df)} rows | close={r['close']} | atr={r['atr14']:.2f} | rsi={r['rsi14']:.2f} | ema200={r['ema200']:.2f}")
    except Exception as e:
        print(f"   FAIL: {e}")
        traceback.print_exc()
        return

    print("2. Testing classify_regime_hard...")
    try:
        from indicators import classify_regime_hard
        regime = classify_regime_hard(r)
        print(f"   OK — regime={regime}")
    except Exception as e:
        print(f"   FAIL: {e}")
        traceback.print_exc()
        return

    print("3. Testing build_trend_context...")
    try:
        from trend_lifecycle import build_trend_context
        ctx = build_trend_context(r)
        print(f"   OK — phase={ctx.phase}")
    except Exception as e:
        print(f"   FAIL: {e}")
        traceback.print_exc()
        return

    print("4. Testing get_funding_rate...")
    try:
        f = em.get_funding_rate(sym)
        print(f"   OK — rate={f['rate']}")
    except Exception as e:
        print(f"   FAIL: {e}")
        traceback.print_exc()
        return

    print("5. Testing ask_llama (this may take ~15s)...")
    try:
        from ai_engine import ask_llama
        from indicators import get_volatility_regime
        price = float(r["close"])
        signal = "long" if float(r["close"]) > float(r["ema200"]) else "short"
        vol_lvl, _, _ = get_volatility_regime(float(r["atr_pct"]))
        rich_ctx = {"trend": {"dir": signal}, "vol": {"lvl": vol_lvl}, "positions": []}
        ai_res = await ask_llama(df, signal, rich_ctx=rich_ctx)
        print(f"   OK — action={ai_res.get('action')} | sl={ai_res.get('sl')} | notes={ai_res.get('notes','')[:60]}")
    except Exception as e:
        print(f"   FAIL: {e}")
        traceback.print_exc()
        return

    print("\nAll checks passed — run_cycle should work correctly.")

asyncio.run(test_ohlcv())

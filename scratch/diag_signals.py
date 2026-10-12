"""
Diagnostic: Check what's blocking signals specifically in the decision tree.
"""
import sys, os, asyncio, logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

import config
config.MODE = "replay"
config.IS_REPLAY = True
config.BYPASS_AI_IN_REPLAY = True
config.SYMBOLS = ["BTC/USDT"]

from exchange_manager import ExchangeManager
from state_manager import EngineState
from indicators import compute_indicators, classify_regime_hard, is_market_ready
from trend_lifecycle import build_trend_context
from bot import get_deterministic_action

async def main():
    em = ExchangeManager()
    state = EngineState(sim_instance=em.sim)
    em.init_replay(config.SYMBOLS, config.TIMEFRAME)
    for b in em.replay.buffers.values():
        b.cursor = 500

    sym = "BTC/USDT"
    results = []

    for _ in range(100):
        ohlcv = em.fetch_ohlcv(sym, config.TIMEFRAME, config.OHLCV_LIMIT)
        htf_ohlcv = em.fetch_ohlcv(sym, config.HTF_TIMEFRAME, config.OHLCV_LIMIT)

        if ohlcv is None or len(ohlcv) < 10:
            em.replay.advance_all(config.TIMEFRAME)
            continue

        df = compute_indicators(ohlcv)
        r = df.iloc[-1]

        price   = float(r["close"])
        ema200  = float(r.get("ema200", 0)) if not (r.get("ema200") != r.get("ema200")) else 0
        adx_val = float(r.get("adx", 0))
        adx_avg = float(r.get("adx_avg", 0))
        rsi     = float(r.get("rsi14", 50))
        vol     = float(r.get("volume", 0))
        vol_sma = float(r.get("vol_sma20", vol))
        atr     = float(r.get("atr14", 0))
        swing_h = float(r.get("swing_high", price))
        swing_l = float(r.get("swing_low", price))

        # HTF trend
        htf_trend = "unknown"
        if htf_ohlcv is not None and len(htf_ohlcv) > 200:
            htf_df = compute_indicators(htf_ohlcv)
            hr = htf_df.iloc[-1]
            htf_ema200 = float(hr.get("ema200", 0)) if not (hr.get("ema200") != hr.get("ema200")) else 0
            htf_close  = float(hr["close"])
            htf_trend  = "bullish" if htf_close > htf_ema200 else "bearish"

        hard_regime = classify_regime_hard(r)
        trend_ctx   = build_trend_context(r)

        action, note, q = get_deterministic_action(
            hard_regime, trend_ctx, adx_val, adx_avg, price, ema200,
            htf_trend, rsi, vol, vol_sma, swing_h, swing_l, atr
        )

        results.append({
            "price": round(price), "ema200": round(ema200),
            "htf_trend": htf_trend, "regime": hard_regime,
            "phase": trend_ctx.phase.value, "action": action,
            "note": note[:60]
        })

        ok = em.replay.advance_all(config.TIMEFRAME)
        if not ok.get(sym, False):
            break

    # Summarize
    from collections import Counter
    actions = Counter(r["action"] for r in results)
    notes   = Counter(r["note"] for r in results)
    print(f"\n=== ACTION DISTRIBUTION ({len(results)} cycles) ===")
    for a, c in actions.most_common():
        print(f"  {a:20s} : {c}")

    print(f"\n=== TOP HOLD REASONS ===")
    for n, c in notes.most_common(10):
        print(f"  [{c:3d}x] {n}")

    print(f"\n=== SAMPLE SNAPSHOTS (last 5) ===")
    for r in results[-5:]:
        print(f"  {r}")

asyncio.run(main())

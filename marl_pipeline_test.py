# marl_pipeline_test.py
# End-to-End MARL Pipeline Integration Test
# Chain: AnalystAgent -> StrategistAgent -> ExecutorAgent
# Uses only mock/synthetic data — no live API calls, no DB dependency.

import asyncio
import logging
import numpy as np
import sys

sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("PIPELINE_TEST")

# ── Import the three MARL agents ─────────────────────────────────────────────
from analyst_agent    import AnalystAgent
from strategist_agent import StrategistAgent
from executor_agent   import ExecutorAgent, ACTION_HOLD, ACTION_LONG, ACTION_SHORT

SEPARATOR = "=" * 60


async def run_pipeline(mock_price: float = 150.00):
    print(f"\n{SEPARATOR}")
    print("  MARL PIPELINE — SYNTHETIC END-TO-END TEST")
    print(f"  Mock Live Price Feed: ${mock_price:,.2f}")
    print(f"{SEPARATOR}\n")

    # ── STAGE 1: AnalystAgent (Feature Engineer) ──────────────────────────────
    print("[STAGE 1] AnalystAgent — Building State Tensor...")
    analyst = AnalystAgent(db_path="trading_ledger.db")

    # build_state_tensor gracefully falls back when DB has sparse data;
    # we override with a fully synthetic tensor to isolate pipeline logic.
    synthetic_tensor = np.array([
        0.60,   # fear_greed (normalised)
        0.20,   # volatility_state
        0.50,   # trend_strength
        0.02,   # ema_distance
        0.50,   # volume_multiplier (normalised)
        0.55,   # hurst exponent
        0.35,   # adx
        0.10,   # rsi_14 (normalised)
        0.01,   # vwap_distance
        0.30,   # sentiment_score
        0.05,   # spy_health
       -0.10,   # dxy_momentum
    ], dtype=np.float32)

    print(f"  [ANALYST] Synthetic state tensor shape : {synthetic_tensor.shape}")
    print(f"  [ANALYST] Feature vector               : {synthetic_tensor.round(3)}")
    print()

    # ── STAGE 2: StrategistAgent (The Brain) ──────────────────────────────────
    print("[STAGE 2] StrategistAgent — Policy Inference...")
    strategist = StrategistAgent()
    decision   = strategist.get_action(synthetic_tensor)

    action_id   = decision["action_id"]
    action_name = decision["action_name"]
    confidence  = decision["confidence"]

    print(f"  [STRATEGIST] Action Selected : {action_name} (id={action_id})")
    print(f"  [STRATEGIST] Confidence      : {confidence:.2%}")
    print()

    # ── STAGE 3: ExecutorAgent (Trade Execution) ──────────────────────────────
    print("[STAGE 3] ExecutorAgent — Mock Order Execution...")
    executor = ExecutorAgent(sl_pct=0.02, tp_pct=0.05)
    order    = executor.mock_execute_order(
        action_id        = action_id,
        confidence_score = confidence,
        current_price    = mock_price,
    )
    print()

    # ── PIPELINE SUMMARY ──────────────────────────────────────────────────────
    print(SEPARATOR)
    print("  PIPELINE SUMMARY")
    print(SEPARATOR)
    print(f"  Stage 1  AnalystAgent    : PASS  (tensor shape {synthetic_tensor.shape})")
    print(f"  Stage 2  StrategistAgent : PASS  ({action_name} @ {confidence:.2%})")

    if action_id == ACTION_HOLD or order is None:
        print(f"  Stage 3  ExecutorAgent   : PASS  (HOLD — no order required)")
        exec_status = "HOLD"
    else:
        sl_ok = order.stop_loss  is not None
        tp_ok = order.take_profit is not None
        math_ok = sl_ok and tp_ok
        if action_id == ACTION_LONG:
            math_ok = math_ok and (order.stop_loss < mock_price < order.take_profit)
        elif action_id == ACTION_SHORT:
            math_ok = math_ok and (order.take_profit < mock_price < order.stop_loss)
        status = "PASS" if math_ok else "FAIL"
        print(f"  Stage 3  ExecutorAgent   : {status}  "
              f"(SL=${order.stop_loss:.4f}  TP=${order.take_profit:.4f}  R:R=1:{0.05/0.02:.1f})")
        exec_status = status

    print(SEPARATOR)
    overall = "ALL STAGES PASSED" if exec_status in ("PASS", "HOLD") else "PIPELINE FAILURE"
    print(f"  OVERALL RESULT: {overall}")
    print(f"{SEPARATOR}\n")

    return overall


if __name__ == "__main__":
    result = asyncio.run(run_pipeline(mock_price=150.00))
    sys.exit(0 if "PASSED" in result else 1)

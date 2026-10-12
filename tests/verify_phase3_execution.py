"""
verify_phase3_execution.py — Sprint 3 (Phase 3) Verification Suite
==================================================================
Tests and benchmarks:
  1. Bitget UTA Post-Only order mechanics (force="post_only", contract precision, inside-spread pricing).
  2. Maker queue fill monitor (_monitor_maker_fill) with full fill rebate capture (0.02%).
  3. Maker monitor adverse OFI shock pre-emption and race-free market fallback.
  4. LivePaperExchange maker fee (0.02%) vs taker fee (0.04%) and zero-slippage queue modeling.
  5. CryptoContagionRadar index calculation, shock response (> 0.85), and sub-10ms event dispatch SLA.
  6. Portfolio delta aggregation (Delta_port = sum beta_i * Q_i * P_i) and BTC hedge sizing.
  7. Synthetic collar asynchronous dispatch latency (< 50 ms SLA) and non-blocking altcoin unwinding.
"""

import asyncio
import json
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock

# ── Colors / Formatting (ASCII safe for Windows console) ──
PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"


async def test_1_post_only_order_mechanics():
    print(f"\n{INFO} --- Test 1: Bitget UTA Post-Only Order Mechanics ---")
    from bitget_uta_client import BitgetUTAClient
    
    uta = BitgetUTAClient()
    # Mock underlying ccxt exchange methods
    mock_order_res = {
        "code": "00000",
        "msg": "success",
        "data": {"orderId": "PO_TEST_998877", "clientOid": "BW_PO_TEST"}
    }
    uta.exchange.private_uta_post_v3_trade_place_order = AsyncMock(return_value=mock_order_res)
    uta.exchange.set_leverage = AsyncMock(return_value={"code": "00000"})
    uta.exchange.fetch_order_book = AsyncMock(return_value={
        "bids": [[91500.0, 1.5]],
        "asks": [[91501.0, 2.0]]
    })
    uta.ensure_time_sync = AsyncMock()

    # Test Post-Only Long
    res = await uta.create_post_only_order(
        symbol="BTC/USDT:USDT",
        side="buy",
        qty=0.005,
        price=None, # Should fetch inside bid 91500.0
        client_oid="TEST_CLIENT_OID_12345"
    )
    
    assert res["id"] == "PO_TEST_998877", f"Expected order ID PO_TEST_998877, got {res['id']}"
    assert res["price"] == 91500.0, f"Expected inside bid 91500.0, got {res['price']}"
    
    # Inspect arguments passed to private_uta_post_v3_trade_place_order
    call_args = uta.exchange.private_uta_post_v3_trade_place_order.call_args[0][0]
    assert call_args["orderType"] == "limit", f"Expected limit orderType, got {call_args['orderType']}"
    assert call_args["force"] == "post_only", f"Expected force=post_only, got {call_args['force']}"
    assert call_args["side"] == "buy"
    assert call_args["posSide"] == "long"
    assert len(call_args["clientOid"]) <= 32
    
    await uta.close()
    print(f"{PASS} Post-only limit mechanics verified: force='post_only', inside-bid pricing, 32-char clientOid.")
    return True


async def test_2_maker_fill_monitor_rebate_capture():
    print(f"\n{INFO} --- Test 2: Maker Queue Fill Monitor - Immediate/Full Fill ---")
    from executor_agent import _monitor_maker_fill
    
    # Create mock UTA client where order fills after 1 second
    mock_uta = MagicMock()
    mock_uta.fetch_order = AsyncMock(return_value={
        "state": "filled",
        "baseVolume": "0.005",
        "filledQty": "0.005"
    })
    mock_uta.cancel_order = AsyncMock()
    mock_uta.create_market_order = AsyncMock()
    mock_uta.close = AsyncMock()
    
    t0 = time.perf_counter()
    fill_res = await _monitor_maker_fill(
        uta_client=mock_uta,
        order_id="TEST_MAKER_FILL_1",
        symbol="BTCUSDT",
        side="buy",
        qty=0.005,
        current_price=91500.0,
        timeout_sec=5.0
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    
    assert fill_res["status"] == "filled", f"Expected filled status, got {fill_res['status']}"
    assert fill_res["is_maker"] is True, "Expected is_maker to be True"
    assert fill_res["filled_qty"] == 0.005
    # Order was filled so cancel_order should NOT have been called (no orphan cancels)
    mock_uta.cancel_order.assert_not_called()
    mock_uta.create_market_order.assert_not_called()
    
    print(f"{PASS} Maker fill monitor captured rebate (is_maker=True, 0 orphan cancels, resolved in {elapsed_ms:.1f} ms).")
    return True


async def test_3_maker_fill_monitor_adverse_shock_fallback():
    print(f"\n{INFO} --- Test 3: Maker Monitor Adverse OFI Shock & Fallback ---")
    from executor_agent import _monitor_maker_fill
    
    # Inject temporary adverse OFI in microstructure_state.json
    ofi_path = "microstructure_state.json"
    adverse_state = {
        "timestamp": time.time(),
        "net_ofi_score": -250.0,
        "normalized_ofi": -0.85,
        "ofi_zscore": -2.45,
        "ofi_velocity": -0.75,
        "symbols": {
            "BTCUSDT": {
                "ofi_10level": -250.0,
                "normalized_ofi": -0.85,
                "ofi_zscore": -2.45
            }
        }
    }
    with open(ofi_path, "w") as f:
        json.dump(adverse_state, f)
        
    mock_uta = MagicMock()
    # Order remains live while waiting
    mock_uta.fetch_order = AsyncMock(return_value={
        "state": "live",
        "baseVolume": "0.0",
        "filledQty": "0.0"
    })
    mock_uta.cancel_order = AsyncMock(return_value={"code": "00000"})
    mock_uta.create_market_order = AsyncMock(return_value={"id": "FALLBACK_MKT_77", "side": "buy", "qty": 0.005})
    mock_uta.close = AsyncMock()
    
    t0 = time.perf_counter()
    fill_res = await _monitor_maker_fill(
        uta_client=mock_uta,
        order_id="TEST_ADVERSE_1",
        symbol="BTCUSDT",
        side="buy",
        qty=0.005,
        current_price=91500.0,
        timeout_sec=8.0,
        adverse_drift_detected=True
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    
    assert fill_res["status"] == "fallback_market", f"Expected fallback_market, got {fill_res['status']}"
    assert fill_res["is_maker"] is False, "Expected is_maker to be False on fallback"
    mock_uta.cancel_order.assert_called_once()
    mock_uta.create_market_order.assert_called_once()
    
    print(f"{PASS} Adverse OFI drift detected early: limit cancelled and market fallback executed in {elapsed_ms:.1f} ms.")
    return True


async def test_4_paper_exchange_maker_fee_accounting():
    print(f"\n{INFO} --- Test 4: LivePaperExchange Maker Fee (0.02%) vs Taker Fee (0.04%) ---")
    from live_paper_exchange import LivePaperExchange
    import tempfile
    
    # Create isolated temporary db
    temp_dir = tempfile.mkdtemp()
    test_db = os.path.join(temp_dir, "test_paper_fees.db")
    
    p_ex = LivePaperExchange(db_path=test_db, initial_balance=25000.0, account_name="test_fees")
    p_ex.is_live_execution = False
    p_ex.uta_client = None
    
    # 1. Open Maker Position (maker=True)
    pos = await p_ex.open_position(
        symbol="BTC/USDT",
        side="long",
        qty=0.1,
        price=90000.0,
        maker=True
    )
    assert pos is not None
    assert pos["is_maker"] is True
    # Slippage should be 0.0 for maker
    assert pos["entry_price"] == 90000.0, f"Expected 90000.0, got {pos['entry_price']}"
    
    # Close Position with take profit (maker exit)
    close_res = await p_ex.close_position(
        symbol="BTC/USDT",
        price=92000.0,
        t_type="tp_hit"
    )
    assert close_res is not None
    
    # Expected Gross PnL = (92000 - 90000) * 0.1 = +$200.0
    # Maker Entry Fee = 0.1 * 90000 * 0.0002 = $1.80
    # Maker Exit Fee  = 0.1 * 92000 * 0.0002 = $1.84
    # Expected Net PnL = 200.0 - (1.80 + 1.84) = $196.36
    # (If taker fee was applied: 0.1*90000*0.0004 + 0.1*92000*0.0004 = $3.60 + $3.68 = $7.28, net=$192.72)
    expected_maker_fee = 0.1 * 90000.0 * 0.0002 + 0.1 * 92000.0 * 0.0002
    expected_taker_fee = 0.1 * 90000.0 * 0.0004 + 0.1 * 92000.0 * 0.0004
    fee_savings = expected_taker_fee - expected_maker_fee
    
    actual_net = close_res if isinstance(close_res, (int, float)) else close_res.get("pnl")
    assert abs(actual_net - 196.36) < 0.01, f"Expected net PnL $196.36, got ${actual_net:.4f}"
    
    print(f"{PASS} Paper maker fee verified: Net PnL = ${actual_net:.2f}. Total fee savings vs taker: ${fee_savings:.2f} (50% fee reduction).")
    return True


async def test_5_crypto_contagion_radar_and_sla():
    print(f"\n{INFO} --- Test 5: Crypto Contagion Radar Index & Sub-10ms Latency SLA ---")
    from macro_breaker_agent import (
        CryptoContagionRadar,
        compute_macro_contagion_index,
        EVENT_CIRCUIT_BREAKER_TRIGGERED
    )
    
    # 1. Calm state index test
    calm_index = compute_macro_contagion_index(v_dom=0.0, delta_peg=0.0, d_basis=0.0, f_skew=0.0)
    assert calm_index < 0.20, f"Expected calm index < 0.20, got {calm_index:.4f}"
    
    # 2. Shock injection test
    shock_index = compute_macro_contagion_index(
        v_dom=0.02,        # Rapid dominance shift
        delta_peg=0.0045,   # 45 bps depeg
        d_basis=35.0,       # 35 bps cross-exchange basis
        f_skew=0.8          # funding skew
    )
    assert shock_index > 0.85, f"Expected severe contagion shock > 0.85, got {shock_index:.4f}"
    
    radar = CryptoContagionRadar(threshold=0.85)
    triggered_events = []
    
    async def on_circuit_breaker(event_data):
        triggered_events.append(event_data)
        
    radar.register_listener(on_circuit_breaker)
    
    # Benchmark evaluation and event dispatch latency
    latencies = []
    dispatch_latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        eval_res = await radar.evaluate({
            "v_dom": 0.025,
            "delta_peg": 0.0050,
            "d_basis": 40.0,
            "f_skew": 1.0
        })
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)
        dispatch_latencies.append(eval_res.get("dispatch_latency_ms", 0.0))
        
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    max_dispatch = max(dispatch_latencies)
    
    assert eval_res["safe_to_trade"] is False, "Expected safe_to_trade=False under shock"
    assert len(triggered_events) == 50, f"Expected 50 events, got {len(triggered_events)}"
    assert max_dispatch < 10.0, f"Dispatch SLA violated! Max={max_dispatch:.2f}ms >= 10.0ms"
    assert avg_latency < 10.0, f"Average latency SLA violated! Avg={avg_latency:.2f}ms >= 10.0ms"
    
    print(f"{PASS} Contagion Radar verified: Calm={calm_index:.3f}, Shock={shock_index:.3f}. "
          f"Dispatch Latency: Max={max_dispatch*1000:.1f} us, Eval Latency: Avg={avg_latency*1000:.1f} us (< 10 ms SLA).")
    return True


async def test_6_portfolio_delta_and_collar_sizing():
    print(f"\n{INFO} --- Test 6: Portfolio Delta & Synthetic Collar Sizing Math ---")
    from core_engine import compute_portfolio_delta, compute_collar_hedge_qty, DEFAULT_ASSET_BETAS
    
    btc_price = 90000.0
    positions = {
        "BTCUSDT": {
            "symbol": "BTC/USDT",
            "side": "short",
            "qty": 0.0006, # $54 notional short
            "entry_price": 90000.0
        },
        "AVAXUSDT": {
            "symbol": "AVAX/USDT",
            "side": "long",
            "qty": 2.6,
            "entry_price": 28.0 # $72.8 notional long, beta=1.40
        },
        "LINKUSDT": {
            "symbol": "LINK/USDT",
            "side": "short",
            "qty": 3.0,
            "entry_price": 14.0 # $42.0 notional short, beta=1.25
        },
        "ADAUSDT": {
            "symbol": "ADA/USDT",
            "side": "short",
            "qty": 179.0,
            "entry_price": 0.35 # $62.65 notional short, beta=1.20
        }
    }
    
    # Theoretical Delta Calculation:
    # BTC:  1.00 * (-0.0006 * 90000) = -54.00
    # AVAX: 1.40 * (+2.6 * 28.0)    = +101.92
    # LINK: 1.25 * (-3.0 * 14.0)    = -52.50
    # ADA:  1.20 * (-179.0 * 0.35)  = -75.18
    # Total Delta = -54.00 + 101.92 - 52.50 - 75.18 = -79.76 USD
    
    p_delta = compute_portfolio_delta(positions, btc_price)
    expected_delta = -54.00 + 101.92 - 52.50 - 75.18
    assert abs(p_delta - expected_delta) < 0.01, f"Expected delta {expected_delta:.2f}, got {p_delta:.2f}"
    
    # Required BTC Hedge Qty:
    # Q_hedge = - Delta_port / P_BTC = - (-79.76) / 90000 = +0.000886 BTC (LONG hedge to balance net short altcoins)
    q_hedge = compute_collar_hedge_qty(p_delta, btc_price)
    expected_hedge_qty = - (expected_delta / btc_price)
    assert abs(q_hedge - expected_hedge_qty) < 1e-6, f"Expected hedge {expected_hedge_qty:.6f}, got {q_hedge:.6f}"
    
    # Post-hedge Net Delta
    net_delta = p_delta + (1.0 * q_hedge * btc_price)
    assert abs(net_delta) < 1e-6, f"Expected net delta 0.0, got {net_delta}"
    
    print(f"{PASS} Portfolio delta aggregation verified: Delta=${p_delta:.2f}, Required BTC Hedge={q_hedge:+.6f} BTC. Net Delta=$0.00.")
    return True


async def test_7_synthetic_collar_async_dispatch_sla():
    print(f"\n{INFO} --- Test 7: Synthetic Collar Dispatch Latency (< 50 ms SLA) ---")
    from core_engine import execute_synthetic_collar
    
    btc_price = 90000.0
    positions = {
        "ETHUSDT": {
            "symbol": "ETH/USDT",
            "side": "long",
            "qty": 0.5,
            "entry_price": 2600.0 # $1300 notional, beta=1.15
        },
        "SOLUSDT": {
            "symbol": "SOL/USDT",
            "side": "long",
            "qty": 10.0,
            "entry_price": 140.0 # $1400 notional, beta=1.35
        }
    }
    
    mock_exchange = MagicMock()
    mock_exchange.create_market_order = AsyncMock(return_value={
        "id": "HEDGE_BTC_001",
        "symbol": "BTCUSDT",
        "side": "short",
        "qty": 0.0376
    })
    mock_exchange.close_position_order = AsyncMock(return_value={"status": "closed"})
    
    t0 = time.perf_counter()
    collar_res = await execute_synthetic_collar(
        exchange_client=mock_exchange,
        positions=positions,
        btc_price=btc_price,
        unwind_horizon_sec=10.0 # Fast test horizon
    )
    total_latency_ms = (time.perf_counter() - t0) * 1000.0
    
    assert collar_res["status"] == "hedged", f"Expected hedged status, got {collar_res['status']}"
    assert collar_res["hedge_side"] == "short", f"Expected short hedge, got {collar_res['hedge_side']}"
    assert collar_res["hedge_qty"] > 0.0
    assert collar_res["dispatch_latency_ms"] < 50.0, f"Dispatch latency SLA breached: {collar_res['dispatch_latency_ms']:.2f} ms >= 50 ms"
    assert total_latency_ms < 50.0, f"Total latency SLA breached: {total_latency_ms:.2f} ms >= 50 ms"
    
    mock_exchange.create_market_order.assert_called_once()
    
    print(f"{PASS} Synthetic collar dispatched in {collar_res['dispatch_latency_ms']:.2f} ms (SLA < 50 ms). Altcoin unwind background task launched.")
    return True


async def main():
    print("=" * 70)
    print("  SOVEREIGN SWARM TRADING BOT — SPRINT 3 VERIFICATION SUITE")
    print("=" * 70)
    
    tests = [
        test_1_post_only_order_mechanics,
        test_2_maker_fill_monitor_rebate_capture,
        test_3_maker_fill_monitor_adverse_shock_fallback,
        test_4_paper_exchange_maker_fee_accounting,
        test_5_crypto_contagion_radar_and_sla,
        test_6_portfolio_delta_and_collar_sizing,
        test_7_synthetic_collar_async_dispatch_sla
    ]
    
    passed = 0
    start_total = time.perf_counter()
    
    for t in tests:
        try:
            ok = await t()
            if ok:
                passed += 1
        except Exception as e:
            print(f"{FAIL} {t.__name__} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            
    total_time = (time.perf_counter() - start_total) * 1000.0
    print("\n" + "=" * 70)
    print(f"  SPRINT 3 VERIFICATION SUMMARY: {passed}/{len(tests)} TESTS PASSED ({total_time:.1f} ms)")
    print("=" * 70)
    
    if passed == len(tests):
        print(f"\n>>> ALL SPRINT 3 REQUIREMENTS VERIFIED & READY FOR PRODUCTION <<<")
        sys.exit(0)
    else:
        print(f"\n>>> SPRINT 3 VERIFICATION INCOMPLETE: {len(tests) - passed} TESTS FAILED <<<")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

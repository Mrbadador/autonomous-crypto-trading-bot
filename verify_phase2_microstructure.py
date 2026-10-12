"""
verify_phase2_microstructure.py — Verification Suite for Sprint 2 (Phase 2 Microstructure Alpha)

Benchmarks and validates:
1. Numba @njit Bivariate Hawkes process execution latency (< 1.0 ms SLA).
2. Synthetic predatory spoof burst detection and pre-trade veto enforcement (S_spoof > 0.72).
3. Stationary normal market flow pass (S_spoof < 0.30).
4. Hawkes branching matrix spectral radius cascade veto (rho >= 1.00).
5. SymbolOFIState addition/cancellation/residence time tracking in ofi_daemon.py.
6. End-to-end MicrostructureGuard backward compatibility and failsafe resilience.
"""

import sys
import os
import time
import asyncio
import numpy as np

# Ensure workspace root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import ofi_daemon
from microstructure_guard import (
    MicrostructureGuard,
    BivariateHawkesModel,
    _bivariate_hawkes_step,
    _compute_spectral_radius,
    _compute_spoof_index,
    HAWKES_TAU_THRESHOLD,
    SPOOF_VETO_THRESHOLD,
    CASCADE_RHO_THRESHOLD
)


def benchmark_numba_latency():
    print("=" * 70)
    print("TEST 1: Benchmarking Numba @njit Bivariate Hawkes Latency (<1.0 ms SLA)")
    print("=" * 70)

    model = BivariateHawkesModel()

    # JIT Warmup pass
    for _ in range(20):
        model.update("BTCUSDT", dt=0.1, n_a=2.0, n_c=1.0, lambda_exec=0.5, tau_res=200.0)

    # Benchmark 2,000 iterations
    iterations = 2000
    start = time.perf_counter()
    for _ in range(iterations):
        model.update("BTCUSDT", dt=0.1, n_a=3.0, n_c=2.0, lambda_exec=1.0, tau_res=180.0)
    elapsed = time.perf_counter() - start

    mean_sec = elapsed / iterations
    mean_ms = mean_sec * 1000.0
    mean_us = mean_sec * 1_000_000.0

    print(f"Total time for {iterations} ticks: {elapsed * 1000.0:.2f} ms")
    print(f"Mean latency per tick: {mean_us:.2f} us ({mean_ms:.4f} ms)")

    assert mean_ms < 1.0, f"FAILED SLA: Mean latency {mean_ms:.4f} ms exceeds 1.0 ms threshold!"
    print(f"[PASS] Latency SLA passed! ({mean_us:.2f} us << 1000.0 us SLA)\n")


async def test_synthetic_spoof_burst_veto():
    print("=" * 70)
    print("TEST 2: Verifying Synthetic Predatory Spoof Burst Veto (S_spoof > 0.72)")
    print("=" * 70)

    guard = MicrostructureGuard()
    guard.active_veto = True  # Ensure active veto enforcement

    # Synthetic spoofing scenario:
    # Predatory trader posts large walls (n_a = 40) then instantly cancels almost all of them (n_c = 140)
    # with ultra-fast residence time (tau_res = 20 ms << 150 ms threshold) and 0 fills (lambda_exec = 0).
    spoof_context = {
        "symbol": "BTC/USDT",
        "dt": 0.1,
        "n_a": 40.0,
        "n_c": 140.0,
        "lambda_exec": 0.0,
        "mean_residence_time_ms": 20.0,
        "trade_volumes": np.array([0.0, 0.0]),
        "trade_signs": np.array([1.0, -1.0]),
        "ofi": -5.0,
        "spread": 1.0
    }

    # First warmup model with a few burst ticks
    for _ in range(5):
        await guard.evaluate_pre_trade_veto(spoof_context)

    passed, reason = await guard.evaluate_pre_trade_veto(spoof_context)
    print(f"Spoof Burst Evaluation Result: Passed = {passed} | Reason = {reason}")

    assert passed is False, "FAILED: Spoofing burst should have triggered pre-trade veto!"
    assert "Hawkes Spoofing & Layering Veto" in reason, f"FAILED: Expected Hawkes Spoofing Veto in reason, got: {reason}"
    print("[PASS] Predatory spoof burst correctly identified and blocked!\n")


async def test_clean_stationary_flow_pass():
    print("=" * 70)
    print("TEST 3: Verifying Clean Stationary Normal Flow Pass (S_spoof < 0.30)")
    print("=" * 70)

    guard = MicrostructureGuard()
    guard.active_veto = True

    # Organic market making flow:
    # Additions and cancellations are balanced (n_a = 20, n_c = 18)
    # Residence time is healthy (tau_res = 850 ms >> 150 ms)
    # Active executions occurring (lambda_exec = 12.0)
    normal_context = {
        "symbol": "ETH/USDT",
        "side": "neutral",
        "dt": 0.5,
        "n_a": 20.0,
        "n_c": 18.0,
        "lambda_exec": 12.0,
        "mean_residence_time_ms": 850.0,
        "trade_volumes": np.array([25.0, 22.0]),
        "trade_signs": np.array([1.0, -1.0]),
        "ofi": 0.5,
        "spread": 0.5
    }

    for _ in range(5):
        await guard.evaluate_pre_trade_veto(normal_context)

    passed, reason = await guard.evaluate_pre_trade_veto(normal_context)
    print(f"Normal Flow Evaluation Result: Passed = {passed} | Reason = {reason}")

    assert passed is True, f"FAILED: Normal stationary flow should pass pre-trade veto, got {reason}"
    assert "Microstructure Approved" in reason, f"FAILED: Expected approval reason, got: {reason}"
    print("[PASS] Clean stationary flow cleanly approved!\n")


async def test_spectral_radius_cascade_veto():
    print("=" * 70)
    print("TEST 4: Verifying Hawkes Liquidity Cascade Veto (rho >= 1.00)")
    print("=" * 70)

    # Calculate baseline stable spectral radius
    rho_stable = _compute_spectral_radius(
        0.4, 0.1, 0.8, 0.3,
        2.0, 3.0, 1.5, 2.5
    )
    print(f"Baseline calibrated spectral radius: rho = {rho_stable:.4f}")
    assert rho_stable < 1.0, f"Baseline spectral radius should be stable (< 1.0), got {rho_stable}"

    # Calculate unstable cascade branching matrix (high cross-excitation)
    rho_unstable = _compute_spectral_radius(
        0.9, 0.8, 1.5, 0.8,
        1.0, 1.2, 1.0, 1.1
    )
    print(f"Unstable cascade spectral radius: rho = {rho_unstable:.4f}")
    assert rho_unstable >= 1.0, f"Unstable matrix should yield rho >= 1.0, got {rho_unstable}"

    # Initialize guard with cascade parameters to verify veto
    cascade_model = BivariateHawkesModel(
        alpha_aa=0.9, alpha_ac=0.8, alpha_ca=1.5, alpha_cc=0.8,
        beta_aa=1.0, beta_ac=1.2, beta_ca=1.0, beta_cc=1.1
    )
    guard = MicrostructureGuard()
    guard.bivariate_hawkes = cascade_model
    guard.active_veto = True

    context = {
        "symbol": "SOL/USDT",
        "dt": 0.1,
        "n_a": 10.0,
        "n_c": 10.0,
        "mean_residence_time_ms": 500.0
    }

    passed, reason = await guard.evaluate_pre_trade_veto(context)
    print(f"Cascade Evaluation Result: Passed = {passed} | Reason = {reason}")

    assert passed is False, "FAILED: Unstable branching matrix should trigger liquidity cascade veto!"
    assert "Hawkes Liquidity Cascade Veto" in reason, f"FAILED: Expected cascade veto reason, got {reason}"
    print("[PASS] Liquidity cascade veto correctly enforced on rho >= 1.00!\n")


def test_ofi_daemon_tracking():
    print("=" * 70)
    print("TEST 5: Verifying SymbolOFIState Hawkes Metrics Tracking in ofi_daemon.py")
    print("=" * 70)

    state = ofi_daemon.SymbolOFIState("BTCUSDT", window_seconds=60, levels=10, delta=0.25)
    state.reset()

    # Initial snapshot
    bids_init = [[70000 - i * 10, 1.0] for i in range(10)]
    asks_init = [[70010 + i * 10, 1.0] for i in range(10)]
    state.process_depth(bids_init, asks_init)

    # Second tick with order additions (size increased at level 0 and level 1)
    time.sleep(0.01)
    bids_adds = [[70000, 2.5], [69990, 3.0]] + [[70000 - i * 10, 1.0] for i in range(2, 10)]
    asks_adds = [[70010, 1.0]] + [[70010 + i * 10, 1.0] for i in range(1, 10)]
    state.process_depth(bids_adds, asks_adds)

    # Third tick with cancellations (size decreased at level 0, new better ask)
    time.sleep(0.01)
    bids_cancels = [[70000, 0.5], [69990, 3.0]] + [[70000 - i * 10, 1.0] for i in range(2, 10)]
    asks_cancels = [[70005, 1.5]] + [[70010 + i * 10, 1.0] for i in range(1, 10)]
    state.process_depth(bids_cancels, asks_cancels)

    metrics = state.get_metrics()
    print("Computed SymbolOFIState metrics:")
    for k in ["order_additions_count", "order_cancellations_count", "mean_residence_time_ms", "normalized_ofi"]:
        print(f"  -> {k}: {metrics.get(k)}")

    assert "order_additions_count" in metrics, "Missing order_additions_count in get_metrics"
    assert "order_cancellations_count" in metrics, "Missing order_cancellations_count in get_metrics"
    assert "mean_residence_time_ms" in metrics, "Missing mean_residence_time_ms in get_metrics"
    assert "cancel_counts" in metrics, "Missing cancel_counts in get_metrics"
    assert "add_counts" in metrics, "Missing add_counts in get_metrics"

    assert metrics["order_additions_count"] > 0, "Expected non-zero order additions count"
    assert metrics["order_cancellations_count"] > 0, "Expected non-zero order cancellations count"
    assert len(metrics["cancel_counts"]) == 10, "Expected 10 levels in cancel_counts"
    assert len(metrics["add_counts"]) == 10, "Expected 10 levels in add_counts"

    print("[PASS] ofi_daemon Hawkes metrics tracking validated!\n")


async def test_backward_compatibility_and_failsafe():
    print("=" * 70)
    print("TEST 6: Verifying Backward Compatibility and Safety Failsafe")
    print("=" * 70)

    guard = MicrostructureGuard()

    # 1. Test minimal context (as sent from BerserkerTrader without Hawkes fields)
    berserker_ctx = {
        "ofi": 0.12,
        "trade_volumes": np.array([10.0, 12.0]),
        "trade_signs": np.array([1.0, -1.0]),
        "vol_24h": 50000.0,
        "volume_series": np.array([1.0] * 64),
        "funding_rate": 0.0001,
        "spread": 1.0,
        "dt": 300.0
    }
    passed, reason = await guard.evaluate_pre_trade_veto(berserker_ctx)
    print(f"Berserker minimal context call: Passed = {passed} | Reason = {reason}")
    assert passed is True, f"Minimal context should pass by default, got: {reason}"

    # 2. Test invalid context (triggering failsafe exception handler)
    invalid_ctx = {
        "ofi": "invalid_string_type",
        "trade_volumes": np.array([1.0]),
        "trade_signs": np.array([1.0])
    }
    passed, reason = await guard.evaluate_pre_trade_veto(invalid_ctx)
    print(f"Invalid context failsafe call: Passed = {passed} | Reason = {reason}")
    assert passed is True, "Failsafe must return passed=True to avoid freezing fleet"
    assert "MicrostructureGuard Error Failsafe" in reason, f"Expected failsafe message, got: {reason}"

    print("[PASS] Backward compatibility and failsafe resilience verified!\n")


async def main():
    print("\n" + "#" * 70)
    print("   SOVEREIGN SWARM — SPRINT 2 (PHASE 2) MICROSTRUCTURE ALPHA SUITE")
    print("#" * 70 + "\n")

    benchmark_numba_latency()
    await test_synthetic_spoof_burst_veto()
    await test_clean_stationary_flow_pass()
    await test_spectral_radius_cascade_veto()
    test_ofi_daemon_tracking()
    await test_backward_compatibility_and_failsafe()

    print("#" * 70)
    print("   ALL SPRINT 2 MICROSTRUCTURE ALPHA VERIFICATIONS PASSED (6/6)")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

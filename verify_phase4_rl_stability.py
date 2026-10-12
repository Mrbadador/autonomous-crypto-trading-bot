"""
verify_phase4_rl_stability.py — Verification Suite for Sprint 4 (Phase 4 RL Stability)

Validates:
1. Generalized Advantage Estimation (GAE, gamma=0.99, lambda=0.95) against analytical trajectories.
2. Adaptive KL divergence tracking and beta_kl scaling.
3. Dynamic entropy annealing schedule over 50,000 steps.
4. Potential-Based Reward Shaping (PBRS) & Policy Invariance telescoping sum property.
5. Policy network inference latency SLA (< 3.0 ms).
6. Safe weight promotion to strategist_weights_v28.pth while strictly preserving strategist_weights_v27.pth.
"""

import os
import sys
import time
import hashlib
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from strategist_agent import (
    StrategistAgent,
    DuelingDQNPolicyNetwork,
    compute_gae,
    AdaptiveKLTracker,
    DynamicEntropyScheduler
)
from core.overlord import PotentialBasedRewardShaper, OverlordMetaController, RewardComposer
from nightly_trainer import NightlyTrainer, PPOTrainer


def test_gae_analytical_benchmark():
    print("=" * 70)
    print("TEST 1: Generalized Advantage Estimation (GAE) Analytical Benchmark")
    print("=" * 70)

    # Deterministic trajectory
    # T = 4 steps
    rewards = np.array([1.0, 2.0, -1.0, 0.5], dtype=np.float64)
    values = np.array([0.5, 1.0, 0.2, 0.8], dtype=np.float64)
    next_values = np.array([1.0, 0.2, 0.8, 0.0], dtype=np.float64)
    dones = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    gamma = 0.99
    lambda_ = 0.95

    # Hand-calculated analytical steps:
    # t=3: non_terminal = 0. delta_3 = 0.5 + 0 - 0.8 = -0.3. A_3 = -0.3
    # t=2: non_terminal = 1. delta_2 = -1.0 + 0.99*0.8*1 - 0.2 = -0.408.
    #      A_2 = -0.408 + (0.99*0.95)*1*(-0.3) = -0.408 - 0.28215 = -0.69015
    # t=1: non_terminal = 1. delta_1 = 2.0 + 0.99*0.2*1 - 1.0 = 1.198.
    #      A_1 = 1.198 + 0.9405 * (-0.69015) = 1.198 - 0.649086075 = 0.548913925
    # t=0: non_terminal = 1. delta_0 = 1.0 + 0.99*1.0*1 - 0.5 = 1.49.
    #      A_0 = 1.49 + 0.9405 * (0.548913925) = 1.49 + 0.516253546 = 2.006253546
    expected_advantages = np.array([2.006253546, 0.548913925, -0.69015, -0.3], dtype=np.float64)

    # 1. NumPy path without standardization
    adv_np, ret_np = compute_gae(
        rewards, values, next_values, dones,
        gamma=gamma, lambda_=lambda_, standardize=False
    )
    np_err = np.max(np.abs(adv_np - expected_advantages))
    print(f"NumPy GAE Max Error vs Analytical: {np_err:.8e}")
    assert np_err < 1e-6, f"NumPy GAE divergence: {np_err}"

    # 2. PyTorch tensor path without standardization
    r_t = torch.FloatTensor(rewards)
    v_t = torch.FloatTensor(values)
    nv_t = torch.FloatTensor(next_values)
    d_t = torch.FloatTensor(dones)
    adv_torch, ret_torch = compute_gae(
        r_t, v_t, nv_t, d_t,
        gamma=gamma, lambda_=lambda_, standardize=False
    )
    torch_err = float(torch.max(torch.abs(adv_torch - torch.FloatTensor(expected_advantages))).item())
    print(f"PyTorch GAE Max Error vs Analytical: {torch_err:.8e}")
    assert torch_err < 1e-5, f"PyTorch GAE divergence: {torch_err}"

    # 3. Standardized GAE assertions
    adv_std, _ = compute_gae(
        r_t, v_t, nv_t, d_t,
        gamma=gamma, lambda_=lambda_, standardize=True
    )
    adv_mean = float(adv_std.mean().item())
    adv_std_dev = float(adv_std.std().item())
    print(f"Standardized GAE: mean={adv_mean:.6f} (target ~0.0), std={adv_std_dev:.6f} (target ~1.0)")
    assert abs(adv_mean) < 1e-4, f"Advantage mean not ~0.0: {adv_mean}"
    assert abs(adv_std_dev - 1.0) < 1e-4, f"Advantage std not ~1.0: {adv_std_dev}"

    print("[PASS] Generalized Advantage Estimation mathematically verified!\n")


def test_adaptive_kl_tracking():
    print("=" * 70)
    print("TEST 2: Adaptive KL Divergence Tracking")
    print("=" * 70)

    tracker = AdaptiveKLTracker(target_kl=0.015, beta_init=1.0, min_beta=0.1, max_beta=20.0)
    print(f"Initial beta_kl: {tracker.beta_kl:.2f}")

    # Case 1: Excessive divergence D_kl > 1.5 * target_kl (0.025 > 0.0225)
    b1 = tracker.update(0.025)
    print(f"Update with D_kl=0.025 (>1.5*0.015) -> beta_kl: {b1:.2f}")
    assert b1 == 2.0, f"Expected beta_kl=2.0, got {b1}"

    # Case 2: Consecutive high divergence
    b2 = tracker.update(0.030)
    print(f"Update with D_kl=0.030 (>1.5*0.015) -> beta_kl: {b2:.2f}")
    assert b2 == 4.0, f"Expected beta_kl=4.0, got {b2}"

    # Case 3: Very small divergence D_kl < target_kl / 1.5 (0.005 < 0.010)
    b3 = tracker.update(0.005)
    print(f"Update with D_kl=0.005 (<0.015/1.5) -> beta_kl: {b3:.2f}")
    assert b3 == 2.0, f"Expected beta_kl=2.0, got {b3}"

    b4 = tracker.update(0.002)
    print(f"Update with D_kl=0.002 (<0.015/1.5) -> beta_kl: {b4:.2f}")
    assert b4 == 1.0, f"Expected beta_kl=1.0, got {b4}"

    # Test clamping limits
    for _ in range(10):
        tracker.update(0.10)
    print(f"Max clamped beta_kl: {tracker.beta_kl:.2f} (cap: {tracker.max_beta})")
    assert tracker.beta_kl == 20.0, f"Expected max beta=20.0, got {tracker.beta_kl}"

    for _ in range(15):
        tracker.update(0.0001)
    print(f"Min clamped beta_kl: {tracker.beta_kl:.2f} (floor: {tracker.min_beta})")
    assert tracker.beta_kl == 0.1, f"Expected min beta=0.1, got {tracker.beta_kl}"

    print("[PASS] Adaptive KL tracking and bounds verified!\n")


def test_dynamic_entropy_annealing():
    print("=" * 70)
    print("TEST 3: Dynamic Entropy Annealing Schedule")
    print("=" * 70)

    scheduler = DynamicEntropyScheduler(init_coef=0.05, min_coef=0.005, total_steps=50000)

    c_0 = scheduler.get_entropy_coef(0)
    c_12k = scheduler.get_entropy_coef(12500)
    c_25k = scheduler.get_entropy_coef(25000)
    c_50k = scheduler.get_entropy_coef(50000)
    c_75k = scheduler.get_entropy_coef(75000)

    print(f"Step 0:     c2 = {c_0:.6f} (target = 0.0500)")
    print(f"Step 12500: c2 = {c_12k:.6f}")
    print(f"Step 25000: c2 = {c_25k:.6f} (target = 0.0158)")
    print(f"Step 50000: c2 = {c_50k:.6f} (target = 0.0050)")
    print(f"Step 75000: c2 = {c_75k:.6f} (target = 0.0050 clamped)")

    assert abs(c_0 - 0.05) < 1e-5, f"Expected c_0=0.05, got {c_0}"
    assert abs(c_25k - 0.015811) < 1e-4, f"Expected c_25k ~ 0.0158, got {c_25k}"
    assert abs(c_50k - 0.005) < 1e-5, f"Expected c_50k=0.005, got {c_50k}"
    assert abs(c_75k - 0.005) < 1e-5, f"Expected c_75k clamped to 0.005, got {c_75k}"
    assert c_0 > c_12k > c_25k > c_50k == c_75k, "Decay must be strictly monotonically non-increasing"

    print("[PASS] Dynamic entropy annealing schedule verified!\n")


def test_pbrs_policy_invariance():
    print("=" * 70)
    print("TEST 4: Potential-Based Reward Shaping (PBRS) & Policy Invariance")
    print("=" * 70)

    shaper = PotentialBasedRewardShaper(w1=1.0, w2=0.5, w3=0.25, gamma=0.99)

    # 1. Verify inventory penalty
    phi_low_inv = shaper.compute_potential(sortino=1.0, inventory_qty=10.0, max_margin=100.0, side="long", z_ofi=0.0)
    phi_high_inv = shaper.compute_potential(sortino=1.0, inventory_qty=90.0, max_margin=100.0, side="long", z_ofi=0.0)
    print(f"Potential low inventory (10%):  Phi = {phi_low_inv:.4f}")
    print(f"Potential high inventory (90%): Phi = {phi_high_inv:.4f}")
    assert phi_low_inv > phi_high_inv, "Higher inventory risk must lower state potential"

    # 2. Verify OFI flow alignment bonus
    phi_ofi_favorable = shaper.compute_potential(sortino=1.0, inventory_qty=20.0, max_margin=100.0, side="long", z_ofi=1.5)
    phi_ofi_adverse = shaper.compute_potential(sortino=1.0, inventory_qty=20.0, max_margin=100.0, side="long", z_ofi=-1.5)
    print(f"Potential favorable OFI (Long with Z=+1.5): Phi = {phi_ofi_favorable:.4f}")
    print(f"Potential adverse OFI (Long with Z=-1.5):   Phi = {phi_ofi_adverse:.4f}")
    assert phi_ofi_favorable > phi_ofi_adverse, "Order flow aligned with side must increase state potential"

    # 3. Policy Invariance Telescoping Sum Property (Ng et al., 1999)
    # Sum_{t=0}^{T-1} gamma^t * F(s_t, a_t, s_{t+1}) == gamma^T * Phi(s_T) - Phi(s_0)
    gamma = 0.99
    T = 20
    np.random.seed(42)
    # Generate arbitrary sequence of state potentials
    potentials = [shaper.compute_potential(
        sortino=np.random.uniform(-1, 3),
        inventory_qty=np.random.uniform(0, 100),
        max_margin=100.0,
        side="long" if np.random.random() > 0.5 else "short",
        z_ofi=np.random.uniform(-2, 2)
    ) for _ in range(T + 1)]

    discounted_f_sum = 0.0
    for t in range(T):
        f_t = shaper.shape_reward(0.0, potentials[t], potentials[t + 1], gamma=gamma)
        discounted_f_sum += (gamma ** t) * f_t

    analytical_telescoping_sum = (gamma ** T) * potentials[T] - potentials[0]
    diff = abs(discounted_f_sum - analytical_telescoping_sum)
    print(f"Empirical Discounted Shaping Sum: {discounted_f_sum:.8f}")
    print(f"Analytical Telescoping Boundary:  {analytical_telescoping_sum:.8f}")
    print(f"Difference (Policy Invariance Error): {diff:.8e}")

    assert diff < 1e-10, f"Policy invariance telescoping sum violated! Error: {diff}"
    print("[PASS] PBRS policy invariance mathematically proven!\n")


def test_inference_latency_sla():
    print("=" * 70)
    print("TEST 5: Policy Network Inference Latency SLA (< 3.0 ms)")
    print("=" * 70)

    import logging
    logging.getLogger("strategist_agent").setLevel(logging.WARNING)

    agent = StrategistAgent()
    mock_15d = np.array([0.5, 0.2, 0.5, 0.2, 0.5, 0.1, 0.2, 0.1, 0.5, 0.2, 0.5, 0.1, 0.5, 0.5, 0.5], dtype=np.float32)

    # JIT / GPU / CPU warmup
    for _ in range(20):
        _ = agent.get_action(mock_15d)

    # Benchmark 1,000 passes
    N = 1000
    start = time.perf_counter()
    for _ in range(N):
        _ = agent.get_action(mock_15d)
    elapsed = time.perf_counter() - start

    mean_ms = (elapsed / N) * 1000.0
    mean_us = (elapsed / N) * 1_000_000.0
    print(f"Executed {N} forward passes in {elapsed * 1000.0:.2f} ms")
    print(f"Mean inference latency: {mean_us:.2f} us ({mean_ms:.4f} ms)")

    assert mean_ms < 3.0, f"FAILED SLA: Inference latency {mean_ms:.4f} ms exceeds 3.0 ms threshold!"
    print(f"[PASS] Latency SLA passed! ({mean_us:.2f} us << 3000.0 us SLA)\n")


def test_offline_training_pass_and_weight_promotion():
    print("=" * 70)
    print("TEST 6: Offline Training Pass & Safe Weight Promotion to v28")
    print("=" * 70)

    v27_path = os.path.join(BASE_DIR, "strategist_weights_v27.pth")
    v28_path = os.path.join(BASE_DIR, "strategist_weights_v28.pth")

    assert os.path.exists(v27_path), f"Baseline weights '{v27_path}' missing!"

    # Calculate SHA256 of v27 to verify it remains 100% UNTOUCHED
    def get_sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    v27_hash_before = get_sha256(v27_path)
    print(f"Baseline strategist_weights_v27.pth SHA256: {v27_hash_before[:16]}...")

    # Instantiate trainer targeting v28
    trainer = NightlyTrainer(
        memory_file="experience_replay.json",
        model_path=v28_path,
        base_model_path=v27_path
    )

    # Execute training pass
    success = trainer.train_network()
    print(f"NightlyTrainer.train_network() return status: {success}")
    assert success is True, "Training pass failed to execute"

    # Save to v28
    save_ok = trainer.save_model()
    print(f"NightlyTrainer.save_model() return status: {save_ok}")
    assert save_ok is True, "Model save failed"
    assert os.path.exists(v28_path), f"Expected '{v28_path}' to be created!"

    # CRITICAL: Verify v27 was NOT overwritten
    v27_hash_after = get_sha256(v27_path)
    assert v27_hash_before == v27_hash_after, "FATAL: strategist_weights_v27.pth was modified! Promotion rule violated!"
    print(f"[PASS] Rule 1 Verified: strategist_weights_v27.pth is strictly unchanged (SHA256 match).")

    # Verify v28 weights structure and forward pass
    v28_state = torch.load(v28_path, map_location="cpu", weights_only=True)
    v27_state = torch.load(v27_path, map_location="cpu", weights_only=True)

    print(f"v28 state dict keys: {list(v28_state.keys())}")
    assert set(v28_state.keys()) == set(v27_state.keys()), "Weight keys mismatch between v27 and v28!"

    for k in v28_state.keys():
        assert v28_state[k].shape == v27_state[k].shape, f"Shape mismatch for key {k}: {v28_state[k].shape} vs {v27_state[k].shape}"

    print("[PASS] Tensor shapes and layer architecture match v27 baseline 100%.")

    # Load fresh test network with v28 weights and verify continuous inference
    test_net = DuelingDQNPolicyNetwork()
    test_net.load_state_dict(v28_state)
    test_net.eval()

    test_input = torch.randn(1, 10, 15)
    with torch.no_grad():
        q_vals, param_logits = test_net(test_input)

    assert q_vals.shape == (1, 3), f"Expected Q-values shape (1, 3), got {q_vals.shape}"
    assert param_logits.shape == (1, 2), f"Expected param shape (1, 2), got {param_logits.shape}"
    print(f"v28 Forward Pass Output: Q-values={q_vals.numpy()}, params={param_logits.numpy()}")
    print("[PASS] Forward pass and 15D state inference compatibility confirmed!\n")


def main():
    print("\n" + "#" * 70)
    print("   SOVEREIGN SWARM — SPRINT 4 (PHASE 4) RL STABILITY SUITE")
    print("#" * 70 + "\n")

    test_gae_analytical_benchmark()
    test_adaptive_kl_tracking()
    test_dynamic_entropy_annealing()
    test_pbrs_policy_invariance()
    test_inference_latency_sla()
    test_offline_training_pass_and_weight_promotion()

    print("#" * 70)
    print("   ALL SPRINT 4 RL STABILITY VERIFICATIONS PASSED (6/6)")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()

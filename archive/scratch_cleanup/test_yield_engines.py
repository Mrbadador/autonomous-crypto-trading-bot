# test_yield_engines.py — QA Audit and Mathematical Verification

import numpy as np
import sys
from market_neutral_agents import StatisticalArbitrageEngine, FundingArbitrageur, CorrelationMatrixMonitor

def print_test_header(name):
    print("\n" + "="*60)
    print(f" RUNNING TEST: {name}")
    print("="*60)

def test_statistical_arbitrage():
    print_test_header("Statistical Arbitrage Engine")
    
    engine = StatisticalArbitrageEngine()
    
    # Scenario A: Perfectly Cointegrated Series
    np.random.seed(42)
    n = 200
    x = np.cumsum(np.random.normal(0, 1, n)) # Random walk
    noise = np.random.normal(0, 0.5, n)       # Stationary process
    y = 1.5 * x + 2.0 + noise                # Cointegrated: y = 1.5x + 2 + noise
    
    print("[INFO] Evaluating Cointegrated series scenario...")
    res_coint = engine.check_cointegration(y, x)
    
    print(f"  - Cointegrated: {res_coint['cointegrated']}")
    print(f"  - Calculated Beta: {res_coint['beta']:.4f} (Expected ~1.5)")
    print(f"  - Calculated Alpha: {res_coint['alpha']:.4f} (Expected ~2.0)")
    print(f"  - Cointegration T-Stat: {res_coint['t_stat']:.4f}")
    print(f"  - Half-Life (bars): {res_coint['half_life_bars']:.2f}")
    print(f"  - Current Z-Score: {res_coint['z_score']:.4f}")
    
    assert abs(res_coint['beta'] - 1.5) < 0.1, "Beta estimation error"
    assert res_coint['cointegrated'] is True, "Should detect cointegration"
    assert res_coint['half_life_bars'] < 50, "Should detect fast mean reversion half life"
    print("[OK] Cointegrated series verification passed!")

    # Scenario B: Uncorrelated Random Walks (Spurious)
    print("\n[INFO] Evaluating Spurious (uncorrelated) series scenario...")
    x_rand = np.cumsum(np.random.normal(0, 1, n))
    y_rand = np.cumsum(np.random.normal(0, 1, n))
    
    res_spurious = engine.check_cointegration(y_rand, x_rand)
    print(f"  - Cointegrated: {res_spurious['cointegrated']}")
    print(f"  - Cointegration T-Stat: {res_spurious['t_stat']:.4f}")
    
    # Typically non-cointegrated t-stats are greater than -3.34 (i.e. closer to zero)
    assert res_spurious['cointegrated'] is False, "Should NOT detect cointegration for independent random walks"
    print("[OK] Spurious series rejection verification passed!")


def test_funding_arbitrageur():
    print_test_header("Funding Arbitrageur")
    
    hunter = FundingArbitrageur(capital=1000.0, min_annual_yield=0.10) # 10% min
    
    # Feed a mix of rates
    rates = {
        "BTCUSDT": 0.0001,   # 0.01% * 3 * 365 = 10.95%  (VALID)
        "ETHUSDT": 0.00005,  # 0.005% * 3 * 365 = 5.47%  (INVALID)
        "SOLUSDT": 0.0004,   # 0.04% * 3 * 365 = 43.8%   (VALID)
        "DOGEUSDT": -0.0002, # -0.02% * 3 * 365 = -21.9% (VALID)
    }
    
    print("[INFO] Evaluating opportunities array...")
    opps = hunter.evaluate_opportunities(rates)
    
    for idx, opp in enumerate(opps):
        print(f"  {idx+1}. {opp['symbol']}: Yield={opp['annualized_yield']*100:.2f}% | Dir={opp['direction']} | Spot Allocation=${opp['spot_allocation']:.1f} | Perp Allocation=${opp['perp_allocation']:.1f}")
        
    assert len(opps) == 3, "Should have filtered out ETHUSDT"
    assert opps[0]["symbol"] == "SOLUSDT", "SOLUSDT must be ranked #1"
    assert opps[1]["symbol"] == "DOGEUSDT", "DOGEUSDT must be ranked #2 by absolute yield"
    assert opps[2]["symbol"] == "BTCUSDT", "BTCUSDT must be ranked #3"
    
    for opp in opps:
        assert opp["spot_allocation"] == 500.0, "Spot allocation must be 50% of capital"
        assert opp["perp_allocation"] == 500.0, "Perp allocation must be 50% of capital"
        
    print("[OK] Funding Arbitrageur verification passed!")


def test_correlation_monitor():
    print_test_header("Correlation Matrix Monitor & Fisher Z Breakdown")
    
    # Initialize with small window to speed up test warmup
    assets = ["BTC", "ETH", "SOL"]
    monitor = CorrelationMatrixMonitor(assets=assets, window=50)
    
    # 1. Inject highly correlated data to warm up (60 periods)
    print("[INFO] Feeding 60 periods of highly correlated data...")
    np.random.seed(10)
    for i in range(60):
        common_factor = np.random.normal(0, 0.01)
        # All assets follow common factor + very small noise
        ret = np.array([
            common_factor + np.random.normal(0, 0.001),
            common_factor + np.random.normal(0, 0.001),
            common_factor + np.random.normal(0, 0.001)
        ])
        status = monitor.update(ret)
        
    # Correlation matrix should be very high (~0.95+)
    initial_corr = status["correlations"]
    print(f"  - Initial BTC-ETH Correlation: {initial_corr[0,1]:.4f}")
    print(f"  - Initial BTC-SOL Correlation: {initial_corr[0,2]:.4f}")
    assert initial_corr[0,1] > 0.85, "Should detect high correlation"
    
    # 2. Inject sudden breakdown for SOL (decorrelates)
    print("\n[INFO] Simulating abrupt decorrelation breakdown for SOL...")
    
    breakdown_triggered = False
    for i in range(10):
        # BTC & ETH still correlated, but SOL completely reverses or becomes uncorrelated
        common_factor = np.random.normal(0, 0.01)
        ret = np.array([
            common_factor,
            common_factor,
            -common_factor * 2.5 # Strong negative divergence
        ])
        status = monitor.update(ret)
        if status["systemic_risk_warning"]:
            breakdown_triggered = True
            print(f"  *** BREAKDOWN DETECTED AT ITERATION {i+1}! ***")
            for p in status["triggered_pairs"]:
                print(f"    - Pair: {p['assets']} | Prior Mean: {p['prior_corr']:.3f} -> Current: {p['current_corr']:.3f} (Z-Drop SE: {p['z_score_drop']:.2f})")
            break
            
    assert breakdown_triggered is True, "Correlation Breakdown warning must fire"
    print("[OK] Correlation Matrix Monitor breakdown detection passed!")


if __name__ == "__main__":
    print(">>> Starting Phase 3 Yield Engines Verification Suite <<<")
    try:
        test_statistical_arbitrage()
        test_funding_arbitrageur()
        test_correlation_monitor()
        print("\n" + "="*60)
        print("SUCCESS: ALL MATHEMATICAL AND SYSTEM VERIFICATION TESTS PASSED!")
        print("="*60)
    except AssertionError as e:
        print(f"\n[ERROR] Assertion Failure during test: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected Error during test: {e}")
        sys.exit(1)

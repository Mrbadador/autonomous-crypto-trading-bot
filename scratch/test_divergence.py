import sys
import os
sys.path.append(os.getcwd())

from analyst_agent import detect_volume_divergence

print("Test 1: Normal Uptrend (Matching)")
p = [100, 101, 102, 103, 104]
v = [10, 15, 20, 25, 30]
print(detect_volume_divergence(p, v))

print("\nTest 2: Weak Momentum Divergence (Price up, Vol down)")
p = [100, 101, 102, 103, 104]
v = [50, 40, 30, 20, 10]
print(detect_volume_divergence(p, v))

print("\nTest 3: Hidden Bullish Setup (Price down but holding, Vol down faster)")
p = [100, 99, 98, 97.5, 97]
v = [50, 40, 30, 20, 10]
print(detect_volume_divergence(p, v))

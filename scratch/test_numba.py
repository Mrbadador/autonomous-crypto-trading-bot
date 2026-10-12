import numpy as np
from numba import jit
from typing import Dict

@jit(nopython=True, cache=False, fastmath=True)
def test_func(x: float) -> Dict[str, float]:
    return {
        'val': x,
        'regime': 1 if x > 0.5 else 0
    }

try:
    print(test_func(0.8))
except Exception as e:
    print("Compilation failed!")
    print(e)

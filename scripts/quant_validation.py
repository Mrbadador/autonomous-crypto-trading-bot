"""Compatibility entry point. The validation engine lives at the repository root."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_validation import *  # noqa: F401,F403
from quant_validation import (  # noqa: F401
    InsufficientValidationData,
    generate_full_validation_report,
    run_trade_sequence_monte_carlo,
)

if __name__ == "__main__":
    import quant_validation

    quant_validation.main()

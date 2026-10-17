# Quarantine package initializer for tests/ (D-11 / P2-9).
# Automatically activates tests.conftest whenever any test in the `tests` package
# is discovered or imported via `python -m unittest`.
from . import conftest as _conftest  # noqa: F401

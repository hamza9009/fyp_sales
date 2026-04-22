"""
Root conftest.py — ensures ``backend/`` is on sys.path for all pytest runs
regardless of the working directory.  This lets both ``app.*`` and ``etl.*``
imports resolve correctly in every test module under ``tests/``.
"""

import sys
from pathlib import Path

# Insert backend/ at the front so our code takes precedence over any
# installed packages with the same name.
_BACKEND = str(Path(__file__).parent / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

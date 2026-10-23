"""Test configuration for this repository."""

import os
import sys
from pathlib import Path

QUICK_MODE = os.environ.get("QUICK_MODE", "0").lower() in {"1", "true", "yes"}

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

"""Puts the repository root on ``sys.path`` so tests can import ``headtracker``."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for entry in (ROOT, ROOT / "tests"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

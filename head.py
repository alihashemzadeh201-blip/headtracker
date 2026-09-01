#!/usr/bin/env python3
"""HeadTracker entry point.

Kept at the repository root so ``python head.py`` still works.  The
implementation lives in the :mod:`headtracker` package.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The sys.path entry above has to exist first, so this import cannot move up.
from headtracker.app import main  # pylint: disable=wrong-import-position

if __name__ == "__main__":
    sys.exit(main())

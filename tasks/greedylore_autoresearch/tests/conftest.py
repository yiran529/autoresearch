"""Expose task-local packages without installing into CORAL's environment."""

from __future__ import annotations

import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
for source in (TASK_ROOT / "grader" / "src", TASK_ROOT / "autoresearch" / "src"):
    value = str(source)
    if value not in sys.path:
        sys.path.insert(0, value)

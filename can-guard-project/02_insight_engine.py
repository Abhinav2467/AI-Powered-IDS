"""Shim: canonical implementation lives in the repository root as `insight_engine.py`."""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from insight_engine import LLMInsightEngine, build_distilled_dashboard_context

__all__ = ["LLMInsightEngine", "build_distilled_dashboard_context"]

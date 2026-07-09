"""
Unified repair orchestration for ARCS failure queues.

Sorts NEGATIVE feedback into per-component piles and dispatches the
documented repair path for each (router export/retrain, DSPy sidecars,
ambiguous review). Does not invent new ML logic or auto-apply prompts.
"""

from arcs.repair.orchestrator import COMPONENTS, repair_all

__all__ = ["COMPONENTS", "repair_all"]

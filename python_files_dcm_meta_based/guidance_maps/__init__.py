"""Guidance-map public surfaces without importing planning for config consumers."""

from importlib import import_module
from typing import Any

from .config import GuidanceMapPlanningConfig

__all__ = [
    "GuidanceMapPlanningConfig",
    "GuidanceMapPatientPrecomputeResult",
    "GuidanceMapPlanningResult",
    "precompute_guidance_map_firing_depth_recommendations_for_patient",
    "precompute_guidance_map_firing_depth_recommendations_for_run",
]


def __getattr__(name: str) -> Any:
    """Resolve the existing planning exports only when callers request them."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module('.planning', __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

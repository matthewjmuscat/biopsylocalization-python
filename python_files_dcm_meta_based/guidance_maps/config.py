from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuidanceMapPlanningConfig:
    """Configuration for non-plotting guidance-map planning tables."""

    candidate_holes_k: int = 1
    candidate_axis_line_length_mm: float = 1000.0
    downcast_threshold: float = 0.25
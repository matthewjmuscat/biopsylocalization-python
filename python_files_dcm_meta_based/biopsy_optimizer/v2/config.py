"""Typed runtime policy for optimizer v2.

This module is intentionally limited to optimizer-v2-specific configuration.
Shared geometry, transform, and containment helpers belong outside
``biopsy_optimizer/v2`` so non-v2 callers can reuse them.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class OptimizerV2StageConfig:
    """Control one staged ranking pass."""

    stage_name: str
    num_trials: int
    survivor_fraction: Optional[float] = None
    survivor_limit: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_trials <= 0:
            raise ValueError("num_trials must be positive")
        if self.survivor_fraction is None and self.survivor_limit is None:
            raise ValueError("at least one survivor control must be provided")
        if self.survivor_fraction is not None and not (0.0 < self.survivor_fraction <= 1.0):
            raise ValueError("survivor_fraction must be in (0, 1]")
        if self.survivor_limit is not None and self.survivor_limit <= 0:
            raise ValueError("survivor_limit must be positive")

    def resolve_survivor_count(self, num_candidates: int) -> int:
        """Resolve the number of candidates that survive this stage."""
        if num_candidates <= 0:
            return 0

        resolved_counts = []
        if self.survivor_fraction is not None:
            resolved_counts.append(max(1, int(math.ceil(num_candidates * self.survivor_fraction))))
        if self.survivor_limit is not None:
            resolved_counts.append(min(num_candidates, self.survivor_limit))

        return min(resolved_counts)


DEFAULT_OPTIMIZER_V2_STAGE_CONFIGS = (
    OptimizerV2StageConfig("stage_a", 16, survivor_fraction=0.10, survivor_limit=256),
    OptimizerV2StageConfig("stage_b", 64, survivor_fraction=0.20, survivor_limit=64),
    OptimizerV2StageConfig("stage_c", 256, survivor_limit=16),
)


@dataclass(frozen=True)
class OptimizerV2SearchConfig:
    """Search-space and stage policy for optimizer v2."""

    lattice_spacing_mm: float = 1.0
    stage_configs: Tuple[OptimizerV2StageConfig, ...] = DEFAULT_OPTIMIZER_V2_STAGE_CONFIGS

    def __post_init__(self) -> None:
        if self.lattice_spacing_mm <= 0.0:
            raise ValueError("lattice_spacing_mm must be positive")
        if not self.stage_configs:
            raise ValueError("stage_configs cannot be empty")

        trial_counts = [stage_config.num_trials for stage_config in self.stage_configs]
        if any(next_count <= current_count for current_count, next_count in zip(trial_counts, trial_counts[1:])):
            raise ValueError("stage trial counts must increase strictly")


@dataclass(frozen=True)
class OptimizerV2VisualizationConfig:
    """Selector-based debug visualization policy for optimizer v2."""

    plot_candidate_lattice_bool: bool = False
    plot_candidate_containment_bool: bool = False
    plot_selected_candidate_points_bool: bool = False
    candidate_indices_to_plot: Tuple[int, ...] = ()
    num_random_candidates_to_plot: int = 0
    trial_indices_to_plot: Tuple[int, ...] = ()
    num_random_trials_to_plot: int = 0
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.num_random_candidates_to_plot < 0:
            raise ValueError("num_random_candidates_to_plot cannot be negative")
        if self.num_random_trials_to_plot < 0:
            raise ValueError("num_random_trials_to_plot cannot be negative")

    def resolve_candidate_indices(self, num_candidates: int) -> np.ndarray:
        """Resolve explicit and random candidate indices into one sorted array."""
        return _resolve_visualization_indices(
            total_count=num_candidates,
            explicit_indices=self.candidate_indices_to_plot,
            num_random_indices=self.num_random_candidates_to_plot,
            random_seed=self.random_seed,
        )

    def resolve_trial_indices(self, num_trials: int) -> np.ndarray:
        """Resolve explicit and random trial indices into one sorted array."""
        return _resolve_visualization_indices(
            total_count=num_trials,
            explicit_indices=self.trial_indices_to_plot,
            num_random_indices=self.num_random_trials_to_plot,
            random_seed=self.random_seed,
        )


def build_default_optimizer_v2_search_config() -> OptimizerV2SearchConfig:
    """Return the default search policy used by optimizer v2."""
    return OptimizerV2SearchConfig()


def build_optimizer_v2_search_config_with_trial_counts(
    stage_trial_counts: Sequence[int],
    lattice_spacing_mm: float = 1.0,
    template_stage_configs: Sequence[OptimizerV2StageConfig] = DEFAULT_OPTIMIZER_V2_STAGE_CONFIGS,
) -> OptimizerV2SearchConfig:
    """Build a search config by replacing only the stage trial counts."""
    if len(stage_trial_counts) != len(template_stage_configs):
        raise ValueError("stage_trial_counts must match the number of template stage configs")

    stage_configs = tuple(
        OptimizerV2StageConfig(
            template_stage_config.stage_name,
            int(stage_trial_count),
            survivor_fraction=template_stage_config.survivor_fraction,
            survivor_limit=template_stage_config.survivor_limit,
        )
        for template_stage_config, stage_trial_count in zip(template_stage_configs, stage_trial_counts)
    )

    return OptimizerV2SearchConfig(
        lattice_spacing_mm=lattice_spacing_mm,
        stage_configs=stage_configs,
    )


def build_default_optimizer_v2_visualization_config() -> OptimizerV2VisualizationConfig:
    """Return the default visualization policy used by optimizer v2."""
    return OptimizerV2VisualizationConfig()


def _resolve_visualization_indices(
    total_count: int,
    explicit_indices: Sequence[int],
    num_random_indices: int,
    random_seed: int,
) -> np.ndarray:
    """Resolve explicit and random indices into one unique sorted integer array."""
    if total_count < 0:
        raise ValueError("total_count cannot be negative")

    resolved_indices = []
    for explicit_index in explicit_indices:
        if explicit_index < 0 or explicit_index >= total_count:
            raise ValueError(
                "visualization index {} is out of range for total_count {}".format(explicit_index, total_count)
            )
        resolved_indices.append(int(explicit_index))

    if num_random_indices > 0 and total_count > 0:
        remaining_indices = np.setdiff1d(
            np.arange(total_count, dtype=np.int32),
            np.array(resolved_indices, dtype=np.int32),
            assume_unique=False,
        )
        if remaining_indices.size > 0:
            random_generator = np.random.default_rng(random_seed)
            num_random_indices_resolved = min(num_random_indices, remaining_indices.size)
            resolved_indices.extend(
                random_generator.choice(
                    remaining_indices,
                    size=num_random_indices_resolved,
                    replace=False,
                ).tolist()
            )

    if not resolved_indices:
        return np.empty(0, dtype=np.int32)

    return np.array(sorted(set(resolved_indices)), dtype=np.int32)


__all__ = [
    "DEFAULT_OPTIMIZER_V2_STAGE_CONFIGS",
    "OptimizerV2SearchConfig",
    "OptimizerV2StageConfig",
    "OptimizerV2VisualizationConfig",
    "build_default_optimizer_v2_search_config",
    "build_default_optimizer_v2_visualization_config",
    "build_optimizer_v2_search_config_with_trial_counts",
]
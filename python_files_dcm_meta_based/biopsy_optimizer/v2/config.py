"""Typed runtime policy for optimizer v2.

This module is intentionally limited to optimizer-v2-specific configuration.
Shared geometry, transform, and containment helpers belong outside
``biopsy_optimizer/v2`` so non-v2 callers can reuse them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
        if self.survivor_fraction is not None and not (0.0 < self.survivor_fraction <= 1.0):
            raise ValueError("survivor_fraction must be in (0, 1]")
        if self.survivor_limit is not None and self.survivor_limit <= 0:
            raise ValueError("survivor_limit must be positive")

    def resolve_survivor_count(self, num_candidates: int) -> int:
        """Resolve the number of candidates that survive this stage."""
        if num_candidates <= 0:
            return 0
        if self.survivor_fraction is None and self.survivor_limit is None:
            return int(num_candidates)

        resolved_counts = []
        if self.survivor_fraction is not None:
            resolved_counts.append(max(1, int(math.ceil(num_candidates * self.survivor_fraction))))
        if self.survivor_limit is not None:
            resolved_counts.append(min(num_candidates, self.survivor_limit))

        return min(resolved_counts)


DEFAULT_OPTIMIZER_V2_STAGE_CONFIGS = (
    OptimizerV2StageConfig("stage_a", 16, survivor_fraction=0.10, survivor_limit=256),
    OptimizerV2StageConfig("stage_b", 64, survivor_fraction=0.20, survivor_limit=64),
    OptimizerV2StageConfig("stage_c", 256, survivor_limit=1),
)


@dataclass(frozen=True)
class OptimizerV2AdaptiveBlockConfig:
    """Adaptive block-round policy for mean_pd-first pruning.

    `initial_trial_prefix` and `trial_block_size` are minimum floors.
    If `max_test_structures_per_call` is provided, the runner may choose a
    larger cumulative trial prefix for a round when the current chunk size can
    still fit within that per-call structure budget.
    """

    initial_trial_prefix: int
    trial_block_size: int
    max_total_trials: int
    round_name_prefix: str = "round"
    max_test_structures_per_call: Optional[int] = None

    def __post_init__(self) -> None:
        if self.initial_trial_prefix <= 0:
            raise ValueError("initial_trial_prefix must be positive")
        if self.trial_block_size <= 0:
            raise ValueError("trial_block_size must be positive")
        if self.max_total_trials < self.initial_trial_prefix:
            raise ValueError("max_total_trials must be greater than or equal to initial_trial_prefix")
        if str(self.round_name_prefix).strip() == "":
            raise ValueError("round_name_prefix cannot be empty")
        if self.max_test_structures_per_call is not None and self.max_test_structures_per_call <= 0:
            raise ValueError("max_test_structures_per_call must be positive when provided")

    def resolve_trial_prefixes(self) -> Tuple[int, ...]:
        """Return the minimum-floor cumulative trial-prefix schedule."""
        resolved_trial_prefixes = [int(self.initial_trial_prefix)]
        current_trial_prefix = int(self.initial_trial_prefix)
        while current_trial_prefix < int(self.max_total_trials):
            current_trial_prefix = min(
                int(self.max_total_trials),
                current_trial_prefix + int(self.trial_block_size),
            )
            resolved_trial_prefixes.append(current_trial_prefix)
        return tuple(resolved_trial_prefixes)

    def build_stage_configs(self) -> Tuple[OptimizerV2StageConfig, ...]:
        return tuple(
            OptimizerV2StageConfig(
                "{}_{:03d}_n{:04d}".format(str(self.round_name_prefix).strip(), round_index, trial_prefix),
                int(trial_prefix),
            )
            for round_index, trial_prefix in enumerate(self.resolve_trial_prefixes(), start=1)
        )

    def build_round_name(self, round_index: int, cumulative_trial_prefix: int) -> str:
        return "{}_{:03d}_n{:04d}".format(
            str(self.round_name_prefix).strip(),
            int(round_index),
            int(cumulative_trial_prefix),
        )

    def resolve_minimum_next_trial_prefix(self, current_trial_prefix: int) -> int:
        if current_trial_prefix < 0:
            raise ValueError("current_trial_prefix cannot be negative")
        if current_trial_prefix == 0:
            return min(int(self.max_total_trials), int(self.initial_trial_prefix))
        return min(
            int(self.max_total_trials),
            int(current_trial_prefix) + int(self.trial_block_size),
        )

    def resolve_capacity_packed_trial_prefix(
        self,
        current_trial_prefix: int,
        active_candidate_count: int,
        max_candidates_per_chunk: int,
        include_nominal: bool,
        max_test_structures_per_call: Optional[int] = None,
    ) -> Optional[int]:
        if current_trial_prefix < 0:
            raise ValueError("current_trial_prefix cannot be negative")
        if active_candidate_count <= 0:
            return None
        if max_candidates_per_chunk <= 0:
            raise ValueError("max_candidates_per_chunk must be positive")

        resolved_structure_budget = self.max_test_structures_per_call
        if max_test_structures_per_call is not None:
            resolved_structure_budget = int(max_test_structures_per_call)
        if resolved_structure_budget is None:
            return None
        if resolved_structure_budget <= 0:
            raise ValueError("max_test_structures_per_call must be positive when provided")

        effective_chunk_candidate_count = min(int(active_candidate_count), int(max_candidates_per_chunk))
        nominal_rows_per_candidate = int(bool(include_nominal))
        max_cumulative_trial_prefix = (
            int(resolved_structure_budget) // int(effective_chunk_candidate_count)
        ) - nominal_rows_per_candidate
        if max_cumulative_trial_prefix <= int(current_trial_prefix):
            return int(current_trial_prefix)
        return min(int(self.max_total_trials), int(max_cumulative_trial_prefix))


@dataclass(frozen=True)
class OptimizerV2TieBreakConfig:
    """Score-first winner-resolution policy for optimizer v2.

    The optimizer should prefer breaking ties by increasing the shared trial
    prefix before falling back to a geometric heuristic.
    """

    score_tolerance: float = 1e-12
    max_additional_rescore_attempts: int = 2
    rescore_trial_count_multiplier: float = 2.0
    fallback_policy: str = "nearest_target_centroid"

    def __post_init__(self) -> None:
        if self.score_tolerance < 0.0:
            raise ValueError("score_tolerance must be non-negative")
        if self.max_additional_rescore_attempts < 0:
            raise ValueError("max_additional_rescore_attempts cannot be negative")
        if self.rescore_trial_count_multiplier <= 1.0:
            raise ValueError("rescore_trial_count_multiplier must be greater than 1.0")
        if self.fallback_policy != "nearest_target_centroid":
            raise ValueError("unsupported fallback_policy: {}".format(self.fallback_policy))

    def resolve_max_tie_break_trial_count(self, base_trial_count: int) -> int:
        """Return the largest trial prefix needed by tie-break rescoring.

        This is the highest prefix the optimizer may need if the final stage
        remains tied through every configured score-based rescore attempt.
        """
        if base_trial_count <= 0:
            raise ValueError("base_trial_count must be positive")

        resolved_trial_count = int(base_trial_count)
        for _ in range(self.max_additional_rescore_attempts):
            resolved_trial_count = int(math.ceil(resolved_trial_count * self.rescore_trial_count_multiplier))
        return resolved_trial_count


@dataclass(frozen=True)
class OptimizerV2SearchConfig:
    """Search-space and stage policy for optimizer v2."""

    lattice_spacing_mm: float = 1.0
    stage_configs: Tuple[OptimizerV2StageConfig, ...] = DEFAULT_OPTIMIZER_V2_STAGE_CONFIGS
    adaptive_block_config: Optional[OptimizerV2AdaptiveBlockConfig] = None
    tie_break_config: OptimizerV2TieBreakConfig = field(default_factory=OptimizerV2TieBreakConfig)
    mean_pd_stage_prune_std_dev_threshold: Optional[float] = 1.0

    def __post_init__(self) -> None:
        if self.lattice_spacing_mm <= 0.0:
            raise ValueError("lattice_spacing_mm must be positive")
        if self.adaptive_block_config is None and not self.stage_configs:
            raise ValueError("either stage_configs or adaptive_block_config must be provided")
        if self.adaptive_block_config is not None and self.stage_configs:
            raise ValueError("stage_configs and adaptive_block_config are mutually exclusive")
        if (
            self.mean_pd_stage_prune_std_dev_threshold is not None
            and self.mean_pd_stage_prune_std_dev_threshold < 0.0
        ):
            raise ValueError("mean_pd_stage_prune_std_dev_threshold must be non-negative when provided")

        trial_counts = [stage_config.num_trials for stage_config in self.resolve_pruning_round_configs()]
        if any(next_count <= current_count for current_count, next_count in zip(trial_counts, trial_counts[1:])):
            raise ValueError("stage trial counts must increase strictly")

    def uses_adaptive_block_rounds(self) -> bool:
        return self.adaptive_block_config is not None

    def resolve_pruning_round_configs(self) -> Tuple[OptimizerV2StageConfig, ...]:
        if self.adaptive_block_config is not None:
            return self.adaptive_block_config.build_stage_configs()
        return tuple(self.stage_configs)

    def resolve_max_optimizer_trial_prefix(self) -> int:
        """Return the largest optimizer-side prefix that may actually be used.

        This includes the final stage's score-based tie-break escalation budget,
        not just the declared stage trial counts.
        """
        if self.adaptive_block_config is not None:
            final_stage_trial_count = int(self.adaptive_block_config.max_total_trials)
        else:
            final_stage_trial_count = self.resolve_pruning_round_configs()[-1].num_trials
        return self.tie_break_config.resolve_max_tie_break_trial_count(final_stage_trial_count)

    def resolve_required_transform_bank_size(self, downstream_trial_count: Optional[int] = None) -> int:
        """Return the minimum shared transform-bank size required by policy.

        The bank must be large enough for:

        1. the largest optimizer-side prefix that may actually be used,
        2. any explicit downstream-comparable winner rescore.
        """
        required_sizes = [self.resolve_max_optimizer_trial_prefix()]
        if downstream_trial_count is not None:
            if downstream_trial_count <= 0:
                raise ValueError("downstream_trial_count must be positive when provided")
            required_sizes.append(int(downstream_trial_count))
        return max(required_sizes)


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
    mean_pd_stage_prune_std_dev_threshold: Optional[float] = 1.0,
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
        mean_pd_stage_prune_std_dev_threshold=mean_pd_stage_prune_std_dev_threshold,
    )


def build_optimizer_v2_adaptive_block_search_config(
    initial_trial_prefix: int,
    trial_block_size: int,
    max_total_trials: int,
    lattice_spacing_mm: float = 1.0,
    mean_pd_stage_prune_std_dev_threshold: Optional[float] = 1.0,
    round_name_prefix: str = "round",
    max_test_structures_per_call: Optional[int] = None,
) -> OptimizerV2SearchConfig:
    """Build a search config that prunes after each appended shared trial block floor."""
    return OptimizerV2SearchConfig(
        lattice_spacing_mm=lattice_spacing_mm,
        stage_configs=tuple(),
        adaptive_block_config=OptimizerV2AdaptiveBlockConfig(
            initial_trial_prefix=int(initial_trial_prefix),
            trial_block_size=int(trial_block_size),
            max_total_trials=int(max_total_trials),
            round_name_prefix=round_name_prefix,
            max_test_structures_per_call=(
                int(max_test_structures_per_call) if max_test_structures_per_call is not None else None
            ),
        ),
        mean_pd_stage_prune_std_dev_threshold=mean_pd_stage_prune_std_dev_threshold,
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
    "OptimizerV2AdaptiveBlockConfig",
    "OptimizerV2SearchConfig",
    "OptimizerV2StageConfig",
    "OptimizerV2TieBreakConfig",
    "OptimizerV2VisualizationConfig",
    "build_optimizer_v2_adaptive_block_search_config",
    "build_default_optimizer_v2_search_config",
    "build_default_optimizer_v2_visualization_config",
    "build_optimizer_v2_search_config_with_trial_counts",
]
"""Data contracts for optimizer v2 candidate pools and chunk layouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np


@dataclass
class OptimizerV2CandidatePool:
    """Store the lattice and immediate target-interior prune results."""

    lattice_spacing_mm: float
    lattice_origin: np.ndarray
    lattice_shape_xyz: Tuple[int, int, int]
    full_lattice_points: np.ndarray
    contained_mask: np.ndarray
    contained_point_indices: np.ndarray
    candidate_points: np.ndarray
    nearest_zslice_index_and_values_3d_arr: np.ndarray
    containment_results_dataframe: Optional[Any] = None


@dataclass(frozen=True)
class OptimizerV2ChunkLayout:
    """Describe one candidate chunk expanded across a trial prefix."""

    candidate_indices_global: Tuple[int, ...]
    num_trials: int
    include_nominal: bool = True
    nominal_relative_structure_index: int = 0
    trial_relative_structure_start_index: int = 1

    def __post_init__(self) -> None:
        if self.num_trials < 0:
            raise ValueError("num_trials cannot be negative")
        if self.nominal_relative_structure_index < 0:
            raise ValueError("nominal_relative_structure_index cannot be negative")
        if self.trial_relative_structure_start_index < 0:
            raise ValueError("trial_relative_structure_start_index cannot be negative")
        for candidate_index in self.candidate_indices_global:
            if candidate_index < 0:
                raise ValueError("candidate indices must be non-negative")

    @property
    def num_candidates(self) -> int:
        return len(self.candidate_indices_global)

    @property
    def num_test_structures_per_candidate(self) -> int:
        return self.num_trials + int(self.include_nominal)

    @property
    def num_test_structures(self) -> int:
        return self.num_candidates * self.num_test_structures_per_candidate

    def build_candidate_metadata_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return row-aligned metadata for the expanded candidate-trial batch."""
        candidate_indices_global = np.empty(self.num_test_structures, dtype=np.int32)
        trial_indices = np.empty(self.num_test_structures, dtype=np.int32)
        is_nominal = np.zeros(self.num_test_structures, dtype=bool)

        write_index = 0
        for candidate_index_global in self.candidate_indices_global:
            if self.include_nominal:
                # Nominal rows use trial index -1 so downstream code can distinguish
                # them from stochastic trial rows without a second lookup table.
                candidate_indices_global[write_index] = candidate_index_global
                trial_indices[write_index] = -1
                is_nominal[write_index] = True
                write_index += 1

            for trial_index in range(self.num_trials):
                candidate_indices_global[write_index] = candidate_index_global
                trial_indices[write_index] = trial_index
                write_index += 1

        return candidate_indices_global, trial_indices, is_nominal

    def build_test_struct_to_relative_struct_mapping(self) -> np.ndarray:
        """Map each expanded test row to its matching nominal or trial structure."""
        test_struct_to_relative_struct = np.empty(self.num_test_structures, dtype=np.int32)
        write_index = 0
        for _ in self.candidate_indices_global:
            if self.include_nominal:
                test_struct_to_relative_struct[write_index] = self.nominal_relative_structure_index
                write_index += 1

            for trial_index in range(self.num_trials):
                test_struct_to_relative_struct[write_index] = self.trial_relative_structure_start_index + trial_index
                write_index += 1

        return test_struct_to_relative_struct

    def build_metadata_dataframe(self):
        """Build a dataframe version of the expanded candidate-trial metadata."""
        import pandas

        candidate_indices_global, trial_indices, is_nominal = self.build_candidate_metadata_arrays()
        return pandas.DataFrame(
            {
                "Test struct input index": np.arange(self.num_test_structures, dtype=np.int32),
                "Candidate global index": candidate_indices_global,
                "Trial index": trial_indices,
                "Is nominal": is_nominal,
            }
        )


@dataclass
class OptimizerV2ChunkScoreResult:
    """Target-only scoring outputs for one candidate chunk."""

    chunk_layout: OptimizerV2ChunkLayout
    candidate_indices_global: np.ndarray
    candidate_centroids: np.ndarray
    objective_reducer_name: str
    structured_containment_result: Any
    stochastic_success_counts: np.ndarray
    point_probabilities: np.ndarray
    candidate_trial_mean_point_scores: Optional[np.ndarray] = None
    candidate_scores: np.ndarray
    candidate_nominal_scores: np.ndarray
    distance_to_target_centroid_mm: np.ndarray
    relative_structure_localized_points: Optional[Any] = None
    tested_candidate_dataframe: Optional[Any] = None


@dataclass
class OptimizerV2StageRunResult:
    """Outputs for one staged ranking pass."""

    stage_name: str
    num_trials: int
    input_candidate_indices_global: np.ndarray
    survivor_candidate_indices_global: np.ndarray
    chunk_score_results: Tuple[OptimizerV2ChunkScoreResult, ...]
    tested_candidate_dataframe: Any
    ranked_candidate_dataframe: Any


@dataclass
class OptimizerV2SearchRunResult:
    """Top-level staged search outputs prior to winner tie-break validation."""

    stage_results: Tuple[OptimizerV2StageRunResult, ...]
    tested_candidate_dataframe: Any
    ranked_candidate_dataframe: Any
    operational_winner_candidate_index_global: Optional[int] = None
    winner_resolution_result: Optional[Any] = None
    winner_validation_result: Optional[Any] = None


@dataclass
class OptimizerV2WinnerResolutionResult:
    """Final winner-resolution metadata for optimizer v2."""

    candidate_index_global: int
    objective_reducer_name: str
    resolution_method: str
    tie_warning_flag: bool
    tie_break_fallback_flag: bool
    num_tied_candidates_at_stage_c: int
    num_additional_rescore_attempts_used: int
    final_resolution_trial_count: int
    resolved_objective_value: float
    resolved_nominal_objective_value: float
    chunk_score_result: Optional[OptimizerV2ChunkScoreResult] = None
    tied_candidate_dataframe: Optional[Any] = None


@dataclass
class OptimizerV2WinnerValidationResult:
    """Winner-only downstream-comparable rescore metadata."""

    candidate_index_global: int
    objective_reducer_name: str
    optimizer_selection_score: float
    optimizer_selection_trial_count: int
    downstream_comparable_target_score: float
    downstream_comparable_trial_count: int
    downstream_comparable_nominal_target_score: float
    used_additional_rescore: bool
    chunk_score_result: Optional[OptimizerV2ChunkScoreResult] = None


__all__ = [
    "OptimizerV2CandidatePool",
    "OptimizerV2ChunkLayout",
    "OptimizerV2ChunkScoreResult",
    "OptimizerV2StageRunResult",
    "OptimizerV2SearchRunResult",
    "OptimizerV2WinnerResolutionResult",
    "OptimizerV2WinnerValidationResult",
]
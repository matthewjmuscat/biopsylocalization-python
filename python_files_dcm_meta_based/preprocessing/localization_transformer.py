"""Shared array-oriented localization helpers.

These helpers sit outside optimizer-v2 because the same uncertainty-localization
logic should be reusable by optimizer code and downstream MC consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cupy as cp
import numpy as np

import MC_prepper_funcs
import pca
from preprocessing.transform_bank import SharedTransformBankPrefix


@dataclass(frozen=True)
class CandidateBiopsySelfTransformBatch:
    """Candidate-localized biopsy self-transform batch.

    The transformed batch is shaped `(num_candidates, num_trial_slices,
    num_points_per_biopsy, 3)`. When `include_nominal` is true, slice 0 along
    the trial axis is the nominal candidate-localized biopsy before stochastic
    self transforms.
    """

    candidate_centroids: np.ndarray
    nominal_points_per_candidate: Any
    transformed_points: Any
    include_nominal: bool
    requested_num_trials: int

    @property
    def num_candidates(self) -> int:
        return int(self.candidate_centroids.shape[0])

    @property
    def num_trial_slices_per_candidate(self) -> int:
        return int(self.transformed_points.shape[1])


@dataclass(frozen=True)
class RelativeStructureLocalizedBiopsyBatch:
    """Biopsy batch after transforming into one relative-structure frame.

    The transformed batch remains shaped `(num_candidates, num_trial_slices,
    num_points_per_biopsy, 3)` so callers can still decide how and when to
    flatten it for batched containment execution.
    """

    candidate_centroids: np.ndarray
    relative_structure_centroid: np.ndarray
    transformed_points: Any
    include_nominal: bool
    requested_num_trials: int

    @property
    def num_candidates(self) -> int:
        return int(self.candidate_centroids.shape[0])

    @property
    def num_trial_slices_per_candidate(self) -> int:
        return int(self.transformed_points.shape[1])


@dataclass(frozen=True)
class AlignedContainmentTestBatch:
    """Minimal aligned containment input surface for the mother function.

    This intentionally stops at the mother-function boundary: a 3D array of test
    structures and the 1D structure-mapping array that selects the corresponding
    relative structure for each test row.
    """

    test_structures: Any
    test_struct_to_relative_struct_mapping: np.ndarray
    include_nominal: bool
    requested_num_trials: int
    num_instances: int

    @property
    def num_test_structures(self) -> int:
        return int(self.test_structures.shape[0])

    @property
    def num_points_per_structure(self) -> int:
        return int(self.test_structures.shape[1])

    @property
    def num_trial_slices_per_instance(self) -> int:
        return int(self.test_structures.shape[0] // self.num_instances)


def build_candidate_biopsy_self_transform_batch(
    nominal_biopsy_points: np.ndarray,
    nominal_biopsy_centroid: np.ndarray,
    nominal_biopsy_centroid_line: np.ndarray,
    candidate_centroids: np.ndarray,
    biopsy_transform_bank_prefix: SharedTransformBankPrefix,
    include_nominal: bool = True,
    return_array_as: str = "cupy",
) -> CandidateBiopsySelfTransformBatch:
    """Build candidate-localized biopsy self-transform batches.

    This reuses the same self-transform sequence already used downstream:
    dilation, rotation, optional needle-compartment shift, then rigid
    translation.
    """
    normalized_nominal_biopsy_points = _validate_xyz_points_array(nominal_biopsy_points, "nominal_biopsy_points")
    normalized_nominal_biopsy_centroid = _validate_xyz_vector(nominal_biopsy_centroid, "nominal_biopsy_centroid")
    normalized_nominal_biopsy_centroid_line = _validate_centroid_line(
        nominal_biopsy_centroid_line,
        "nominal_biopsy_centroid_line",
    )
    normalized_candidate_centroids = _validate_xyz_points_array(candidate_centroids, "candidate_centroids")
    _validate_return_array_as(return_array_as)

    candidate_nominal_points_cp_arr, candidate_centroid_lines_cp_arr = translate_nominal_biopsy_to_candidate_centroids(
        normalized_nominal_biopsy_points,
        normalized_nominal_biopsy_centroid,
        normalized_nominal_biopsy_centroid_line,
        normalized_candidate_centroids,
        return_array_as="cupy",
    )

    dilation_samples_cp_arr = cp.asarray(biopsy_transform_bank_prefix.dilation_samples)
    rotation_samples_cp_arr = cp.asarray(biopsy_transform_bank_prefix.rotation_samples)
    translation_samples_cp_arr = cp.asarray(biopsy_transform_bank_prefix.translation_samples)
    needle_compartment_distance_samples_cp_arr = None
    if biopsy_transform_bank_prefix.needle_compartment_distance_samples is not None:
        needle_compartment_distance_samples_cp_arr = cp.asarray(
            biopsy_transform_bank_prefix.needle_compartment_distance_samples
        )

    candidate_trial_batches = []
    for candidate_index in range(candidate_nominal_points_cp_arr.shape[0]):
        candidate_nominal_points_cp_arr_slice = candidate_nominal_points_cp_arr[candidate_index]
        candidate_global_centroid_cp_arr = cp.asarray(normalized_candidate_centroids[candidate_index]).reshape(1, 3)
        candidate_centroid_line_cp_arr = candidate_centroid_lines_cp_arr[candidate_index]

        dilated_points_cp_arr = MC_prepper_funcs.biopsy_dilator_step_1(
            candidate_nominal_points_cp_arr_slice,
            dilation_samples_cp_arr,
            candidate_centroid_line_cp_arr,
            candidate_global_centroid_cp_arr,
            biopsy_transform_bank_prefix.requested_num_trials,
        )
        rotated_points_cp_arr = MC_prepper_funcs.biopsy_rotator_step_2_vectorized_version(
            dilated_points_cp_arr,
            rotation_samples_cp_arr,
            candidate_global_centroid_cp_arr,
            biopsy_transform_bank_prefix.requested_num_trials,
        )

        total_translation_vectors_cp_arr = translation_samples_cp_arr
        if needle_compartment_distance_samples_cp_arr is not None:
            total_translation_vectors_cp_arr = (
                total_translation_vectors_cp_arr
                + _build_needle_compartment_shift_vectors(
                    rotated_points_cp_arr,
                    needle_compartment_distance_samples_cp_arr,
                )
            )

        self_transformed_points_cp_arr = MC_prepper_funcs.biopsy_translator_step_3(
            rotated_points_cp_arr,
            total_translation_vectors_cp_arr,
        )

        if include_nominal:
            candidate_trial_batch_cp_arr = cp.concatenate(
                [candidate_nominal_points_cp_arr_slice[cp.newaxis, :, :], self_transformed_points_cp_arr],
                axis=0,
            )
        else:
            candidate_trial_batch_cp_arr = self_transformed_points_cp_arr

        candidate_trial_batches.append(candidate_trial_batch_cp_arr)

    transformed_points_cp_arr = cp.stack(candidate_trial_batches, axis=0)
    transformed_points = _coerce_output_array(transformed_points_cp_arr, return_array_as)
    nominal_points_per_candidate = _coerce_output_array(candidate_nominal_points_cp_arr, return_array_as)

    return CandidateBiopsySelfTransformBatch(
        candidate_centroids=normalized_candidate_centroids,
        nominal_points_per_candidate=nominal_points_per_candidate,
        transformed_points=transformed_points,
        include_nominal=include_nominal,
        requested_num_trials=biopsy_transform_bank_prefix.requested_num_trials,
    )


def translate_nominal_biopsy_to_candidate_centroids(
    nominal_biopsy_points: np.ndarray,
    nominal_biopsy_centroid: np.ndarray,
    nominal_biopsy_centroid_line: np.ndarray,
    candidate_centroids: np.ndarray,
    return_array_as: str = "cupy",
):
    """Translate one nominal biopsy model to multiple candidate centroids."""
    normalized_nominal_biopsy_points = _validate_xyz_points_array(nominal_biopsy_points, "nominal_biopsy_points")
    normalized_nominal_biopsy_centroid = _validate_xyz_vector(nominal_biopsy_centroid, "nominal_biopsy_centroid")
    normalized_nominal_biopsy_centroid_line = _validate_centroid_line(
        nominal_biopsy_centroid_line,
        "nominal_biopsy_centroid_line",
    )
    normalized_candidate_centroids = _validate_xyz_points_array(candidate_centroids, "candidate_centroids")
    _validate_return_array_as(return_array_as)

    nominal_biopsy_points_cp_arr = cp.asarray(normalized_nominal_biopsy_points)
    nominal_biopsy_centroid_cp_arr = cp.asarray(normalized_nominal_biopsy_centroid).reshape(1, 1, 3)
    nominal_biopsy_centroid_line_cp_arr = cp.asarray(normalized_nominal_biopsy_centroid_line)
    candidate_centroids_cp_arr = cp.asarray(normalized_candidate_centroids)

    candidate_offsets_cp_arr = candidate_centroids_cp_arr[:, cp.newaxis, :] - nominal_biopsy_centroid_cp_arr
    candidate_nominal_points_cp_arr = nominal_biopsy_points_cp_arr[cp.newaxis, :, :] + candidate_offsets_cp_arr
    candidate_centroid_lines_cp_arr = nominal_biopsy_centroid_line_cp_arr[cp.newaxis, :, :] + candidate_offsets_cp_arr

    return (
        _coerce_output_array(candidate_nominal_points_cp_arr, return_array_as),
        _coerce_output_array(candidate_centroid_lines_cp_arr, return_array_as),
    )


def build_relative_structure_localized_biopsy_batch(
    candidate_biopsy_self_transform_batch: CandidateBiopsySelfTransformBatch,
    relative_structure_centroid: np.ndarray,
    relative_structure_transform_bank_prefix: SharedTransformBankPrefix,
    return_array_as: str = "cupy",
) -> RelativeStructureLocalizedBiopsyBatch:
    """Apply inverse relative-structure rigid motion to the biopsy batch.

    This returns the same candidate/trial layout as the self-transform batch so
    later batching code can flatten it without losing the shared structure of the
    work.
    """
    normalized_relative_structure_centroid = _validate_xyz_vector(
        relative_structure_centroid,
        "relative_structure_centroid",
    )
    _validate_return_array_as(return_array_as)

    if relative_structure_transform_bank_prefix.requested_num_trials != candidate_biopsy_self_transform_batch.requested_num_trials:
        raise ValueError(
            "relative-structure and biopsy transform-bank prefixes must use the same requested_num_trials"
        )

    transformed_points_cp_arr = cp.asarray(candidate_biopsy_self_transform_batch.transformed_points)
    relative_structure_rotation_samples_cp_arr = cp.asarray(relative_structure_transform_bank_prefix.rotation_samples)
    relative_structure_translation_samples_cp_arr = cp.asarray(relative_structure_transform_bank_prefix.translation_samples)

    if candidate_biopsy_self_transform_batch.include_nominal:
        nominal_points_cp_arr = transformed_points_cp_arr[:, :1, :, :]
        stochastic_points_cp_arr = transformed_points_cp_arr[:, 1:, :, :]
    else:
        nominal_points_cp_arr = None
        stochastic_points_cp_arr = transformed_points_cp_arr

    relative_structure_localized_candidate_batches = []
    for candidate_index in range(transformed_points_cp_arr.shape[0]):
        candidate_stochastic_points_cp_arr = stochastic_points_cp_arr[candidate_index]
        candidate_rotated_points_cp_arr = MC_prepper_funcs.rotate_biopsy_to_relative_structure_points_vectorized(
            candidate_stochastic_points_cp_arr,
            relative_structure_rotation_samples_cp_arr,
            normalized_relative_structure_centroid.reshape(1, 3),
            candidate_biopsy_self_transform_batch.requested_num_trials,
        )
        candidate_localized_points_cp_arr = MC_prepper_funcs.translate_biopsy_to_relative_structure_points(
            candidate_rotated_points_cp_arr,
            relative_structure_translation_samples_cp_arr,
            candidate_biopsy_self_transform_batch.requested_num_trials,
        )

        if candidate_biopsy_self_transform_batch.include_nominal:
            candidate_trial_batch_cp_arr = cp.concatenate(
                [nominal_points_cp_arr[candidate_index], candidate_localized_points_cp_arr],
                axis=0,
            )
        else:
            candidate_trial_batch_cp_arr = candidate_localized_points_cp_arr

        relative_structure_localized_candidate_batches.append(candidate_trial_batch_cp_arr)

    relative_structure_localized_points_cp_arr = cp.stack(relative_structure_localized_candidate_batches, axis=0)

    return RelativeStructureLocalizedBiopsyBatch(
        candidate_centroids=candidate_biopsy_self_transform_batch.candidate_centroids,
        relative_structure_centroid=normalized_relative_structure_centroid,
        transformed_points=_coerce_output_array(relative_structure_localized_points_cp_arr, return_array_as),
        include_nominal=candidate_biopsy_self_transform_batch.include_nominal,
        requested_num_trials=candidate_biopsy_self_transform_batch.requested_num_trials,
    )


def flatten_relative_structure_localized_batch_for_containment(
    relative_structure_localized_biopsy_batch: RelativeStructureLocalizedBiopsyBatch,
    nominal_relative_structure_index: int = 0,
    trial_relative_structure_start_index: int = 1,
    return_array_as: str = "cupy",
) -> AlignedContainmentTestBatch:
    """Flatten a localized biopsy batch to the mother-function input surface."""
    _validate_non_negative_index(nominal_relative_structure_index, "nominal_relative_structure_index")
    _validate_non_negative_index(trial_relative_structure_start_index, "trial_relative_structure_start_index")
    _validate_return_array_as(return_array_as)

    transformed_points_cp_arr = cp.asarray(relative_structure_localized_biopsy_batch.transformed_points)
    flattened_test_structures_cp_arr = transformed_points_cp_arr.reshape(
        transformed_points_cp_arr.shape[0] * transformed_points_cp_arr.shape[1],
        transformed_points_cp_arr.shape[2],
        transformed_points_cp_arr.shape[3],
    )

    test_struct_to_relative_struct_mapping = _build_test_struct_to_relative_struct_mapping(
        num_instances=relative_structure_localized_biopsy_batch.num_candidates,
        requested_num_trials=relative_structure_localized_biopsy_batch.requested_num_trials,
        include_nominal=relative_structure_localized_biopsy_batch.include_nominal,
        nominal_relative_structure_index=nominal_relative_structure_index,
        trial_relative_structure_start_index=trial_relative_structure_start_index,
    )

    return AlignedContainmentTestBatch(
        test_structures=_coerce_output_array(flattened_test_structures_cp_arr, return_array_as),
        test_struct_to_relative_struct_mapping=test_struct_to_relative_struct_mapping,
        include_nominal=relative_structure_localized_biopsy_batch.include_nominal,
        requested_num_trials=relative_structure_localized_biopsy_batch.requested_num_trials,
        num_instances=relative_structure_localized_biopsy_batch.num_candidates,
    )


def build_candidate_relative_structure_containment_batch(
    candidate_biopsy_self_transform_batch: CandidateBiopsySelfTransformBatch,
    relative_structure_centroid: np.ndarray,
    relative_structure_transform_bank_prefix: SharedTransformBankPrefix,
    nominal_relative_structure_index: int = 0,
    trial_relative_structure_start_index: int = 1,
    return_array_as: str = "cupy",
) -> AlignedContainmentTestBatch:
    """Compose relative-structure localization with mother-function flattening."""
    relative_structure_localized_biopsy_batch = build_relative_structure_localized_biopsy_batch(
        candidate_biopsy_self_transform_batch,
        relative_structure_centroid,
        relative_structure_transform_bank_prefix,
        return_array_as="cupy",
    )
    return flatten_relative_structure_localized_batch_for_containment(
        relative_structure_localized_biopsy_batch,
        nominal_relative_structure_index=nominal_relative_structure_index,
        trial_relative_structure_start_index=trial_relative_structure_start_index,
        return_array_as=return_array_as,
    )


def _build_needle_compartment_shift_vectors(
    rotated_points_cp_arr: cp.ndarray,
    needle_compartment_distance_samples_cp_arr: cp.ndarray,
) -> cp.ndarray:
    """Convert stored needle-compartment distances into aligned translation vectors."""
    if cp.all(needle_compartment_distance_samples_cp_arr == 0):
        return cp.zeros((rotated_points_cp_arr.shape[0], 3), dtype=rotated_points_cp_arr.dtype)

    lines = pca.vectorized_linear_fitter(rotated_points_cp_arr)
    point_1 = lines[:, 0, :]
    point_2 = lines[:, 1, :]

    # The existing downstream path defines the tip-to-handle direction as the
    # negative of the superior-to-inferior line orientation for each trial.
    superior_point_mask = point_1[:, 2] > point_2[:, 2]
    point_sup = cp.where(superior_point_mask[:, None], point_1, point_2)
    point_inf = cp.where(superior_point_mask[:, None], point_2, point_1)
    biopsy_vec_handle_to_tip = point_sup - point_inf
    biopsy_vec_handle_to_tip_unit = biopsy_vec_handle_to_tip / cp.linalg.norm(
        biopsy_vec_handle_to_tip,
        axis=1,
        keepdims=True,
    )
    biopsy_vec_tip_to_handle_unit = -biopsy_vec_handle_to_tip_unit

    return cp.multiply(
        biopsy_vec_tip_to_handle_unit,
        needle_compartment_distance_samples_cp_arr[:, None],
    )


def _coerce_output_array(output_array, return_array_as: str):
    if return_array_as == "cupy":
        return output_array
    return cp.asnumpy(output_array)


def _validate_xyz_points_array(points_array: np.ndarray, array_name: str) -> np.ndarray:
    normalized_points_array = np.asarray(points_array, dtype=float)
    if normalized_points_array.ndim != 2 or normalized_points_array.shape[1] != 3:
        raise ValueError("{} must have shape (num_points, 3)".format(array_name))
    if normalized_points_array.shape[0] == 0:
        raise ValueError("{} cannot be empty".format(array_name))
    return normalized_points_array


def _validate_xyz_vector(vector: np.ndarray, vector_name: str) -> np.ndarray:
    normalized_vector = np.asarray(vector, dtype=float).reshape(-1)
    if normalized_vector.shape != (3,):
        raise ValueError("{} must contain exactly three coordinates".format(vector_name))
    return normalized_vector


def _validate_centroid_line(centroid_line: np.ndarray, line_name: str) -> np.ndarray:
    normalized_centroid_line = np.asarray(centroid_line, dtype=float)
    if normalized_centroid_line.shape != (2, 3):
        raise ValueError("{} must have shape (2, 3)".format(line_name))
    return normalized_centroid_line


def _validate_return_array_as(return_array_as: str) -> None:
    if return_array_as not in {"cupy", "numpy"}:
        raise ValueError("return_array_as must be 'cupy' or 'numpy'")


def _build_test_struct_to_relative_struct_mapping(
    num_instances: int,
    requested_num_trials: int,
    include_nominal: bool,
    nominal_relative_structure_index: int,
    trial_relative_structure_start_index: int,
) -> np.ndarray:
    if num_instances <= 0:
        raise ValueError("num_instances must be positive")

    per_instance_mapping = []
    if include_nominal:
        per_instance_mapping.append(nominal_relative_structure_index)
    per_instance_mapping.extend(
        range(
            trial_relative_structure_start_index,
            trial_relative_structure_start_index + requested_num_trials,
        )
    )

    return np.tile(np.array(per_instance_mapping, dtype=np.int32), num_instances)


def _validate_non_negative_index(index: int, index_name: str) -> None:
    if index < 0:
        raise ValueError("{} cannot be negative".format(index_name))


__all__ = [
    "AlignedContainmentTestBatch",
    "CandidateBiopsySelfTransformBatch",
    "RelativeStructureLocalizedBiopsyBatch",
    "build_candidate_biopsy_self_transform_batch",
    "build_candidate_relative_structure_containment_batch",
    "build_relative_structure_localized_biopsy_batch",
    "flatten_relative_structure_localized_batch_for_containment",
    "translate_nominal_biopsy_to_candidate_centroids",
]
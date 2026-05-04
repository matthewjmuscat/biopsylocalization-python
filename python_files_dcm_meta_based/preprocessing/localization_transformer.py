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


__all__ = [
    "CandidateBiopsySelfTransformBatch",
    "build_candidate_biopsy_self_transform_batch",
    "translate_nominal_biopsy_to_candidate_centroids",
]
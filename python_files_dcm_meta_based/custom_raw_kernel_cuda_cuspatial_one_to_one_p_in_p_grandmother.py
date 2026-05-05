"""General-use batching helpers for the custom CUDA point-in-polygon stack.

This module sits above `custom_point_containment_mother_function(...)` and
below any preprocessing or optimizer-specific adapters.

It currently exposes two batching layers:

1. `custom_point_containment_grandmother_function(...)`
    Shared-first-argument convenience layer that preserves the mother-function
    contract while chunking only the test-structure side.
2. `custom_point_containment_grandfather_function(...)`
    Explicit chunk-plan executor where each chunk provides its own first
    argument, second argument, and chunk-local mapping array.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import cupy as cp
import numpy as np

import custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p


@dataclass(frozen=True)
class ContainmentChunkSpec:
     """Explicit inputs for one chunk-local mother-function call.

     Each chunk may carry its own relative-structure universe, its own test
     structures, and its own mapping array. Repeating either argument across
     chunk specs is valid as long as the mapping is local to that chunk.
     """

     list_of_relative_structures_containting_list_of_constant_zslices_arrays: Sequence[Sequence[np.ndarray]]
     points_to_test_3d_arr_or_list_of_2d_arrays: Any
     test_struct_to_relative_struct_1d_mapping_array: np.ndarray


def custom_point_containment_grandmother_function(
    list_of_relative_structures_containting_list_of_constant_zslices_arrays: Sequence[Sequence[np.ndarray]],
    points_to_test_3d_arr_or_list_of_2d_arrays: Any,
    test_struct_to_relative_struct_1d_mapping_array: np.ndarray,
    max_test_structures_per_call: Optional[int] = None,
    constant_z_slice_polygons_handler_option: str = "auto-close-if-open",
    remove_consecutive_duplicate_points_in_polygons: bool = False,
    log_sub_dirs_list: Optional[Sequence[str]] = None,
    log_file_name: str = "cuda_log.txt",
    include_edges_in_log: bool = False,
    kernel_type: str = "one_to_one_pip_kernel_advanced_reparameterized_version_gpu_memory_performance_optimized",
):
    """Run the mother function over one or more chunks.

    The return value intentionally matches the mother-function contract:
    `(result_cp_arr, prepper_output_tuple)`. The only added behavior is that
    the job may be split across multiple mother-function calls and then merged
    back into the same logical output surface.

    This helper assumes the first argument is shared across every chunk. Only
    the second argument and its aligned mapping array are sliced per chunk.
    If different chunks need different first arguments, use
    `custom_point_containment_grandfather_function(...)` instead.
    """
    input_mode = _resolve_test_structure_input_mode(points_to_test_3d_arr_or_list_of_2d_arrays)
    num_test_structures = _resolve_num_test_structures(points_to_test_3d_arr_or_list_of_2d_arrays, input_mode)
    normalized_mapping = np.asarray(test_struct_to_relative_struct_1d_mapping_array, dtype=np.int32)
    if normalized_mapping.ndim != 1 or normalized_mapping.shape[0] != num_test_structures:
        raise ValueError("test_struct_to_relative_struct_1d_mapping_array must be 1D and match the number of test structures")

    resolved_chunk_size = _resolve_chunk_size(max_test_structures_per_call, num_test_structures)
    if resolved_chunk_size >= num_test_structures:
        return _run_single_containment_call(
            list_of_relative_structures_containting_list_of_constant_zslices_arrays,
            points_to_test_3d_arr_or_list_of_2d_arrays,
            normalized_mapping,
            constant_z_slice_polygons_handler_option=constant_z_slice_polygons_handler_option,
            remove_consecutive_duplicate_points_in_polygons=remove_consecutive_duplicate_points_in_polygons,
            log_sub_dirs_list=log_sub_dirs_list,
            log_file_name=log_file_name,
            include_edges_in_log=include_edges_in_log,
            kernel_type=kernel_type,
        )

    chunk_outputs = []
    for chunk_start_index in range(0, num_test_structures, resolved_chunk_size):
        chunk_end_index = min(chunk_start_index + resolved_chunk_size, num_test_structures)
        chunk_outputs.append(
            _run_single_containment_call(
                list_of_relative_structures_containting_list_of_constant_zslices_arrays,
                _slice_test_structures(
                    points_to_test_3d_arr_or_list_of_2d_arrays,
                    chunk_start_index,
                    chunk_end_index,
                    input_mode,
                ),
                normalized_mapping[chunk_start_index:chunk_end_index],
                constant_z_slice_polygons_handler_option=constant_z_slice_polygons_handler_option,
                remove_consecutive_duplicate_points_in_polygons=remove_consecutive_duplicate_points_in_polygons,
                log_sub_dirs_list=log_sub_dirs_list,
                log_file_name=log_file_name,
                include_edges_in_log=include_edges_in_log,
                kernel_type=kernel_type,
            )
        )

    return _concatenate_chunk_outputs(chunk_outputs, input_mode)


def custom_point_containment_grandfather_function(
    chunk_specs: Sequence[ContainmentChunkSpec],
    constant_z_slice_polygons_handler_option: str = "auto-close-if-open",
    remove_consecutive_duplicate_points_in_polygons: bool = False,
    log_sub_dirs_list: Optional[Sequence[str]] = None,
    log_file_name: str = "cuda_log.txt",
    include_edges_in_log: bool = False,
    kernel_type: str = "one_to_one_pip_kernel_advanced_reparameterized_version_gpu_memory_performance_optimized",
):
    """Run an explicit sequence of mother-function calls.

    Unlike grandmother, this surface does not assume a shared first argument.
    Each chunk spec carries its own first argument, second argument, and
    chunk-local mapping array. Repeating either argument across chunk specs is
    allowed.

    Returns a tuple of:

    1. `combined_result_cp_arr_or_none`
       Concatenated raw result array when the chunk result shapes are
       compatible along axis 0. Otherwise `None`.
    2. `chunk_output_tuples`
       List of `(result_cp_arr, prepper_output_tuple)` pairs in chunk order.

    The per-chunk outputs are the authoritative full-fidelity surface. The
    combined raw result is a convenience only when the chunk result shapes are
    mergeable.
    """
    normalized_chunk_specs = list(chunk_specs)
    if not normalized_chunk_specs:
        raise ValueError("chunk_specs cannot be empty")

    chunk_outputs = []
    for chunk_spec in normalized_chunk_specs:
        input_mode = _resolve_test_structure_input_mode(
            chunk_spec.points_to_test_3d_arr_or_list_of_2d_arrays
        )
        num_test_structures = _resolve_num_test_structures(
            chunk_spec.points_to_test_3d_arr_or_list_of_2d_arrays,
            input_mode,
        )
        normalized_mapping = np.asarray(
            chunk_spec.test_struct_to_relative_struct_1d_mapping_array,
            dtype=np.int32,
        )
        if normalized_mapping.ndim != 1 or normalized_mapping.shape[0] != num_test_structures:
            raise ValueError(
                "Each chunk spec must provide a 1D mapping array matching that chunk's number of test structures"
            )

        chunk_outputs.append(
            _run_single_containment_call(
                chunk_spec.list_of_relative_structures_containting_list_of_constant_zslices_arrays,
                chunk_spec.points_to_test_3d_arr_or_list_of_2d_arrays,
                normalized_mapping,
                constant_z_slice_polygons_handler_option=constant_z_slice_polygons_handler_option,
                remove_consecutive_duplicate_points_in_polygons=remove_consecutive_duplicate_points_in_polygons,
                log_sub_dirs_list=log_sub_dirs_list,
                log_file_name=log_file_name,
                include_edges_in_log=include_edges_in_log,
                kernel_type=kernel_type,
            )
        )

    return _maybe_concatenate_chunk_results(chunk_outputs), chunk_outputs


def _run_single_containment_call(
    list_of_relative_structures_containting_list_of_constant_zslices_arrays: Sequence[Sequence[np.ndarray]],
    points_to_test_3d_arr_or_list_of_2d_arrays: Any,
    test_struct_to_relative_struct_1d_mapping_array: np.ndarray,
    constant_z_slice_polygons_handler_option: str,
    remove_consecutive_duplicate_points_in_polygons: bool,
    log_sub_dirs_list: Optional[Sequence[str]],
    log_file_name: str,
    include_edges_in_log: bool,
    kernel_type: str,
):
    return custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.custom_point_containment_mother_function(
        list_of_relative_structures_containting_list_of_constant_zslices_arrays,
        points_to_test_3d_arr_or_list_of_2d_arrays,
        test_struct_to_relative_struct_1d_mapping_array,
        constant_z_slice_polygons_handler_option=constant_z_slice_polygons_handler_option,
        remove_consecutive_duplicate_points_in_polygons=remove_consecutive_duplicate_points_in_polygons,
        log_sub_dirs_list=list(log_sub_dirs_list or []),
        log_file_name=log_file_name,
        include_edges_in_log=include_edges_in_log,
        kernel_type=kernel_type,
    )


def _concatenate_chunk_outputs(chunk_outputs: Sequence[tuple[Any, tuple]], input_mode: str):
    if not chunk_outputs:
        raise ValueError("chunk_outputs cannot be empty")

    raw_containment_result_cp_arr = cp.concatenate(
        [cp.asarray(chunk_output[0]) for chunk_output in chunk_outputs],
        axis=0,
    )
    prepper_output_tuple = _concatenate_prepper_output_tuples(
        [chunk_output[1] for chunk_output in chunk_outputs],
        input_mode,
    )
    return raw_containment_result_cp_arr, prepper_output_tuple


def _maybe_concatenate_chunk_results(chunk_outputs: Sequence[tuple[Any, tuple]]):
    if not chunk_outputs:
        raise ValueError("chunk_outputs cannot be empty")

    normalized_results = [cp.asarray(chunk_output[0]) for chunk_output in chunk_outputs]
    reference_result = normalized_results[0]
    reference_trailing_shape = tuple(reference_result.shape[1:])

    for normalized_result in normalized_results[1:]:
        if normalized_result.ndim != reference_result.ndim:
            return None
        if tuple(normalized_result.shape[1:]) != reference_trailing_shape:
            return None

    return cp.concatenate(normalized_results, axis=0)


def _concatenate_prepper_output_tuples(prepper_output_tuples: Sequence[tuple], input_mode: str):
    first_prepper_output_tuple = prepper_output_tuples[0]
    nearest_zslice_index_and_values = np.concatenate(
        [np.asarray(prepper_output_tuple[0]) for prepper_output_tuple in prepper_output_tuples],
        axis=0,
    )

    if input_mode == "array_3d":
        return (
            nearest_zslice_index_and_values,
            first_prepper_output_tuple[1],
            first_prepper_output_tuple[2],
            first_prepper_output_tuple[3],
            first_prepper_output_tuple[4],
            first_prepper_output_tuple[5],
            first_prepper_output_tuple[6],
        )

    points_to_test_2d_arr = np.concatenate(
        [np.asarray(prepper_output_tuple[3]) for prepper_output_tuple in prepper_output_tuples],
        axis=0,
    )
    points_to_test_indices_arr = _concatenate_points_to_test_indices(
        [prepper_output_tuple[4] for prepper_output_tuple in prepper_output_tuples],
        [int(np.asarray(prepper_output_tuple[3]).shape[0]) for prepper_output_tuple in prepper_output_tuples],
    )
    return (
        nearest_zslice_index_and_values,
        first_prepper_output_tuple[1],
        first_prepper_output_tuple[2],
        points_to_test_2d_arr,
        points_to_test_indices_arr,
    )


def _concatenate_points_to_test_indices(
    points_to_test_indices_arrays: Sequence[np.ndarray],
    point_counts_per_chunk: Sequence[int],
) -> np.ndarray:
    concatenated_indices = []
    point_offset = 0

    for points_to_test_indices_arr, point_count in zip(points_to_test_indices_arrays, point_counts_per_chunk):
        chunk_points_to_test_indices_arr = np.asarray(points_to_test_indices_arr, dtype=np.int64).copy()
        chunk_points_to_test_indices_arr += point_offset
        concatenated_indices.append(chunk_points_to_test_indices_arr)
        point_offset += point_count

    return np.concatenate(concatenated_indices, axis=0)


def _resolve_test_structure_input_mode(points_to_test_3d_arr_or_list_of_2d_arrays: Any) -> str:
    if isinstance(points_to_test_3d_arr_or_list_of_2d_arrays, list):
        return "list_2d_arrays"
    if hasattr(points_to_test_3d_arr_or_list_of_2d_arrays, "ndim") and points_to_test_3d_arr_or_list_of_2d_arrays.ndim == 3:
        return "array_3d"
    raise ValueError("points_to_test_3d_arr_or_list_of_2d_arrays must be either a list of 2D arrays or a 3D array")


def _resolve_num_test_structures(points_to_test_3d_arr_or_list_of_2d_arrays: Any, input_mode: str) -> int:
    if input_mode == "list_2d_arrays":
        return len(points_to_test_3d_arr_or_list_of_2d_arrays)
    return int(points_to_test_3d_arr_or_list_of_2d_arrays.shape[0])


def _resolve_chunk_size(max_test_structures_per_call: Optional[int], num_test_structures: int) -> int:
    if max_test_structures_per_call is None:
        return num_test_structures
    if max_test_structures_per_call <= 0:
        raise ValueError("max_test_structures_per_call must be positive when provided")
    return min(int(max_test_structures_per_call), num_test_structures)


def _slice_test_structures(
    points_to_test_3d_arr_or_list_of_2d_arrays: Any,
    chunk_start_index: int,
    chunk_end_index: int,
    input_mode: str,
):
    if input_mode == "list_2d_arrays":
        return points_to_test_3d_arr_or_list_of_2d_arrays[chunk_start_index:chunk_end_index]
    return points_to_test_3d_arr_or_list_of_2d_arrays[chunk_start_index:chunk_end_index]


__all__ = [
    "ContainmentChunkSpec",
    "custom_point_containment_grandfather_function",
    "custom_point_containment_grandmother_function",
]
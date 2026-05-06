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

from dataclasses import dataclass, replace
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


@dataclass(frozen=True)
class ContainmentCallCapacitySignature:
    """Geometry and device signature for one calibrated containment-call budget."""

    device_id: int
    device_name: str
    total_device_memory_bytes: int
    kernel_type: str
    constant_z_slice_polygons_handler_option: str
    remove_consecutive_duplicate_points_in_polygons: bool
    include_edges_in_log: bool
    num_relative_structures: int
    total_relative_structure_slices: int
    total_relative_structure_points: int
    max_relative_structure_slice_count: int
    num_points_per_test_structure: int


@dataclass(frozen=True)
class ContainmentCallCapacityCalibrationResult:
    """Resolved safe package-level `max_test_structures_per_call` budget."""

    safe_max_test_structures_per_call: int
    estimated_max_test_structures_per_call: int
    verified_max_test_structures_per_call: int
    verification_attempt_count: int
    used_binary_search: bool
    from_cache: bool
    safety_factor: float
    verification_expansion_factor: float
    signature: ContainmentCallCapacitySignature


_CONTAINMENT_CALL_CAPACITY_CACHE: dict[tuple[Any, ...], ContainmentCallCapacityCalibrationResult] = {}


def calibrate_max_test_structures_per_call(
    list_of_relative_structures_containting_list_of_constant_zslices_arrays: Sequence[Sequence[np.ndarray]],
    prototype_test_structure_points_2d_arr: np.ndarray,
    constant_z_slice_polygons_handler_option: str = "auto-close-if-open",
    remove_consecutive_duplicate_points_in_polygons: bool = False,
    log_sub_dirs_list: Optional[Sequence[str]] = None,
    log_file_name: Optional[str] = None,
    include_edges_in_log: bool = False,
    kernel_type: str = "one_to_one_pip_kernel_advanced_reparameterized_version_gpu_memory_performance_optimized",
    safety_factor: float = 0.7,
    verification_expansion_factor: float = 1.25,
    max_verification_expansion_rounds: int = 2,
    max_binary_search_rounds: int = 6,
) -> ContainmentCallCapacityCalibrationResult:
    """Calibrate a safe `max_test_structures_per_call` for the current device.

    The calibration is package-owned and intentionally exercises the same
    mother-function call surface that production batching uses. The returned
    budget is the largest verified number of aligned test structures that can
    be processed in one unchunked call for the supplied relative-structure pack
    and prototype test-structure geometry.
    """
    if safety_factor <= 0.0 or safety_factor > 1.0:
        raise ValueError("safety_factor must be in the interval (0, 1]")
    if verification_expansion_factor < 1.0:
        raise ValueError("verification_expansion_factor must be >= 1.0")
    if max_verification_expansion_rounds < 0:
        raise ValueError("max_verification_expansion_rounds must be >= 0")
    if max_binary_search_rounds < 0:
        raise ValueError("max_binary_search_rounds must be >= 0")

    normalized_prototype = np.ascontiguousarray(
        np.asarray(prototype_test_structure_points_2d_arr, dtype=float)
    )
    if normalized_prototype.ndim != 2 or normalized_prototype.shape[1] < 3:
        raise ValueError(
            "prototype_test_structure_points_2d_arr must be a 2D array with at least three columns"
        )
    if normalized_prototype.shape[0] == 0:
        raise ValueError("prototype_test_structure_points_2d_arr cannot be empty")
    normalized_prototype = normalized_prototype[:, :3]

    signature = _build_call_capacity_signature(
        list_of_relative_structures_containting_list_of_constant_zslices_arrays,
        normalized_prototype,
        kernel_type,
        constant_z_slice_polygons_handler_option,
        remove_consecutive_duplicate_points_in_polygons,
        include_edges_in_log,
    )
    cache_key = _build_call_capacity_cache_key(signature)
    cached_result = _CONTAINMENT_CALL_CAPACITY_CACHE.get(cache_key)
    if cached_result is not None:
        return replace(cached_result, from_cache=True)

    estimated_max_test_structures_per_call = max(
        1,
        _estimate_safe_max_test_structures_per_call(signature, safety_factor),
    )

    verification_attempt_count = 0
    used_binary_search = False

    def verify_candidate(num_test_structures: int) -> bool:
        nonlocal verification_attempt_count
        verification_attempt_count += 1
        return _verify_max_test_structures_candidate(
            num_test_structures=num_test_structures,
            list_of_relative_structures_containting_list_of_constant_zslices_arrays=list_of_relative_structures_containting_list_of_constant_zslices_arrays,
            prototype_test_structure_points_2d_arr=normalized_prototype,
            constant_z_slice_polygons_handler_option=constant_z_slice_polygons_handler_option,
            remove_consecutive_duplicate_points_in_polygons=remove_consecutive_duplicate_points_in_polygons,
            log_sub_dirs_list=log_sub_dirs_list,
            log_file_name=log_file_name,
            include_edges_in_log=include_edges_in_log,
            kernel_type=kernel_type,
        )

    best_verified = 0
    expansion_failure_upper_bound = None

    if verify_candidate(estimated_max_test_structures_per_call):
        best_verified = estimated_max_test_structures_per_call
        current_verified = estimated_max_test_structures_per_call
        for _ in range(max_verification_expansion_rounds):
            expanded_candidate = max(
                current_verified + 1,
                int(np.ceil(current_verified * verification_expansion_factor)),
            )
            if expanded_candidate == current_verified:
                expanded_candidate = current_verified + 1
            if verify_candidate(expanded_candidate):
                best_verified = expanded_candidate
                current_verified = expanded_candidate
                continue
            expansion_failure_upper_bound = expanded_candidate
            used_binary_search = True
            break
    else:
        used_binary_search = True
        expansion_failure_upper_bound = estimated_max_test_structures_per_call

    if expansion_failure_upper_bound is not None:
        low = best_verified + 1
        high = expansion_failure_upper_bound - 1
        binary_search_rounds_used = 0
        while low <= high and binary_search_rounds_used < max_binary_search_rounds:
            binary_search_rounds_used += 1
            mid = (low + high) // 2
            if verify_candidate(mid):
                best_verified = mid
                low = mid + 1
            else:
                high = mid - 1

    if best_verified <= 0:
        raise RuntimeError(
            "Could not verify even a single unchunked containment call for the supplied geometry on the active CUDA device"
        )

    calibration_result = ContainmentCallCapacityCalibrationResult(
        safe_max_test_structures_per_call=int(best_verified),
        estimated_max_test_structures_per_call=int(estimated_max_test_structures_per_call),
        verified_max_test_structures_per_call=int(best_verified),
        verification_attempt_count=int(verification_attempt_count),
        used_binary_search=bool(used_binary_search),
        from_cache=False,
        safety_factor=float(safety_factor),
        verification_expansion_factor=float(verification_expansion_factor),
        signature=signature,
    )
    _CONTAINMENT_CALL_CAPACITY_CACHE[cache_key] = calibration_result
    return calibration_result


def _build_call_capacity_signature(
    list_of_relative_structures_containting_list_of_constant_zslices_arrays: Sequence[Sequence[np.ndarray]],
    prototype_test_structure_points_2d_arr: np.ndarray,
    kernel_type: str,
    constant_z_slice_polygons_handler_option: str,
    remove_consecutive_duplicate_points_in_polygons: bool,
    include_edges_in_log: bool,
) -> ContainmentCallCapacitySignature:
    (
        num_relative_structures,
        total_relative_structure_slices,
        total_relative_structure_points,
        max_relative_structure_slice_count,
    ) = _summarize_relative_structure_geometry(
        list_of_relative_structures_containting_list_of_constant_zslices_arrays,
        constant_z_slice_polygons_handler_option,
    )
    device_id, device_name, total_device_memory_bytes = _resolve_active_cuda_device_properties()
    return ContainmentCallCapacitySignature(
        device_id=device_id,
        device_name=device_name,
        total_device_memory_bytes=total_device_memory_bytes,
        kernel_type=str(kernel_type),
        constant_z_slice_polygons_handler_option=str(constant_z_slice_polygons_handler_option),
        remove_consecutive_duplicate_points_in_polygons=bool(remove_consecutive_duplicate_points_in_polygons),
        include_edges_in_log=bool(include_edges_in_log),
        num_relative_structures=int(num_relative_structures),
        total_relative_structure_slices=int(total_relative_structure_slices),
        total_relative_structure_points=int(total_relative_structure_points),
        max_relative_structure_slice_count=int(max_relative_structure_slice_count),
        num_points_per_test_structure=int(prototype_test_structure_points_2d_arr.shape[0]),
    )


def _build_call_capacity_cache_key(
    signature: ContainmentCallCapacitySignature,
) -> tuple[Any, ...]:
    return (
        signature.device_id,
        signature.device_name,
        signature.total_device_memory_bytes,
        signature.kernel_type,
        signature.constant_z_slice_polygons_handler_option,
        signature.remove_consecutive_duplicate_points_in_polygons,
        signature.include_edges_in_log,
        signature.num_relative_structures,
        signature.total_relative_structure_slices,
        signature.total_relative_structure_points,
        signature.max_relative_structure_slice_count,
        signature.num_points_per_test_structure,
    )


def _estimate_safe_max_test_structures_per_call(
    signature: ContainmentCallCapacitySignature,
    safety_factor: float,
) -> int:
    free_device_memory_bytes, _ = cp.cuda.Device().mem_info
    safe_free_device_memory_bytes = int(free_device_memory_bytes * safety_factor)

    fixed_geometry_bytes = (
        signature.total_relative_structure_points * 24
        + signature.total_relative_structure_slices * 16
        + signature.num_relative_structures * 32
    )
    per_point_device_bytes = 24 + 24 + 16 + 1 + 16
    if signature.include_edges_in_log:
        per_point_device_bytes += 128
    per_test_structure_bytes = max(
        1,
        signature.num_points_per_test_structure * per_point_device_bytes,
    )
    prep_peak_bytes = (
        max(
            signature.num_points_per_test_structure,
            signature.max_relative_structure_slice_count,
        )
        * 48
    )

    usable_device_memory_bytes = safe_free_device_memory_bytes - fixed_geometry_bytes - prep_peak_bytes
    if usable_device_memory_bytes <= 0:
        return 1
    return max(1, int(usable_device_memory_bytes // per_test_structure_bytes))


def _verify_max_test_structures_candidate(
    num_test_structures: int,
    list_of_relative_structures_containting_list_of_constant_zslices_arrays: Sequence[Sequence[np.ndarray]],
    prototype_test_structure_points_2d_arr: np.ndarray,
    constant_z_slice_polygons_handler_option: str,
    remove_consecutive_duplicate_points_in_polygons: bool,
    log_sub_dirs_list: Optional[Sequence[str]],
    log_file_name: Optional[str],
    include_edges_in_log: bool,
    kernel_type: str,
) -> bool:
    if num_test_structures <= 0:
        return False

    num_relative_structures = len(
        list_of_relative_structures_containting_list_of_constant_zslices_arrays
    )
    if num_relative_structures <= 0:
        raise ValueError("At least one relative structure is required for calibration")

    batched_points = np.broadcast_to(
        prototype_test_structure_points_2d_arr[np.newaxis, :, :],
        (
            int(num_test_structures),
            prototype_test_structure_points_2d_arr.shape[0],
            prototype_test_structure_points_2d_arr.shape[1],
        ),
    ).copy()
    test_struct_to_relative_struct_1d_mapping_array = (
        np.arange(int(num_test_structures), dtype=np.int32) % num_relative_structures
    )

    try:
        raw_result_cp_arr, prepper_output_tuple = _run_single_containment_call(
            list_of_relative_structures_containting_list_of_constant_zslices_arrays,
            batched_points,
            test_struct_to_relative_struct_1d_mapping_array,
            constant_z_slice_polygons_handler_option=constant_z_slice_polygons_handler_option,
            remove_consecutive_duplicate_points_in_polygons=remove_consecutive_duplicate_points_in_polygons,
            log_sub_dirs_list=log_sub_dirs_list,
            log_file_name=log_file_name,
            include_edges_in_log=include_edges_in_log,
            kernel_type=kernel_type,
        )
        cp.cuda.runtime.deviceSynchronize()
        del raw_result_cp_arr
        del prepper_output_tuple
        _free_cupy_memory_pools()
        return True
    except Exception as exc:
        _free_cupy_memory_pools()
        if _is_gpu_memory_capacity_exception(exc):
            return False
        raise
    finally:
        del batched_points
        del test_struct_to_relative_struct_1d_mapping_array


def _summarize_relative_structure_geometry(
    list_of_relative_structures_containting_list_of_constant_zslices_arrays: Sequence[Sequence[np.ndarray]],
    constant_z_slice_polygons_handler_option: str,
) -> tuple[int, int, int, int]:
    num_relative_structures = len(list_of_relative_structures_containting_list_of_constant_zslices_arrays)
    if num_relative_structures <= 0:
        raise ValueError("list_of_relative_structures_containting_list_of_constant_zslices_arrays cannot be empty")

    total_relative_structure_slices = 0
    total_relative_structure_points = 0
    max_relative_structure_slice_count = 0
    auto_close_open_polygons = constant_z_slice_polygons_handler_option == "auto-close-if-open"

    for relative_structure in list_of_relative_structures_containting_list_of_constant_zslices_arrays:
        current_slice_count = len(relative_structure)
        total_relative_structure_slices += current_slice_count
        max_relative_structure_slice_count = max(max_relative_structure_slice_count, current_slice_count)

        for constant_zslice_array in relative_structure:
            normalized_constant_zslice_array = np.asarray(constant_zslice_array)
            if normalized_constant_zslice_array.ndim != 2 or normalized_constant_zslice_array.shape[1] < 3:
                raise ValueError("Each constant-z slice must be a 2D array with at least three columns")
            num_points_in_slice = int(normalized_constant_zslice_array.shape[0])
            if (
                auto_close_open_polygons
                and num_points_in_slice > 1
                and not np.allclose(
                    normalized_constant_zslice_array[0, :3],
                    normalized_constant_zslice_array[-1, :3],
                )
            ):
                num_points_in_slice += 1
            total_relative_structure_points += num_points_in_slice

    return (
        int(num_relative_structures),
        int(total_relative_structure_slices),
        int(total_relative_structure_points),
        int(max_relative_structure_slice_count),
    )


def _resolve_active_cuda_device_properties() -> tuple[int, str, int]:
    device_id = int(cp.cuda.runtime.getDevice())
    device_properties = cp.cuda.runtime.getDeviceProperties(device_id)
    raw_device_name = device_properties.get("name", "unknown")
    if isinstance(raw_device_name, bytes):
        device_name = raw_device_name.decode("utf-8", errors="replace")
    else:
        device_name = str(raw_device_name)
    total_device_memory_bytes = int(device_properties["totalGlobalMem"])
    return device_id, device_name, total_device_memory_bytes


def _is_gpu_memory_capacity_exception(exc: Exception) -> bool:
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, cp.cuda.memory.OutOfMemoryError):
        return True
    normalized_message = str(exc).lower()
    return (
        "out of memory" in normalized_message
        or "cuda_error_out_of_memory" in normalized_message
        or "memory allocation" in normalized_message
        or "unable to allocate" in normalized_message
    )


def _free_cupy_memory_pools() -> None:
    try:
        cp.cuda.runtime.deviceSynchronize()
    except Exception:
        pass
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    try:
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


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
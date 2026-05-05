"""Shared containment execution helpers.

This module is a thin preprocessing-facing adapter on top of the general-use
custom-PIP containment surfaces. It reshapes aligned 3D batches for scoring
consumers without owning the generalized grandmother execution logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import cupy as cp
import numpy as np

import custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p
import custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p_grandparents
from preprocessing.localization_transformer import AlignedContainmentTestBatch


@dataclass(frozen=True)
class AlignedContainmentRunResult:
    """Containment results for one aligned batch execution."""

    aligned_containment_test_batch: AlignedContainmentTestBatch
    raw_containment_result: Any
    structured_containment_result: Any
    nearest_zslice_index_and_values_3d_arr: Any
    containment_results_dataframe: Optional[Any] = None


def run_aligned_containment_batch(
    list_of_relative_structures_containting_list_of_constant_zslices_arrays: Sequence[Sequence[np.ndarray]],
    aligned_containment_test_batch: AlignedContainmentTestBatch,
    constant_z_slice_polygons_handler_option: str = "auto-close-if-open",
    remove_consecutive_duplicate_points_in_polygons: bool = False,
    log_sub_dirs_list: Optional[Sequence[str]] = None,
    log_file_name: str = "cuda_log.txt",
    include_edges_in_log: bool = False,
    kernel_type: str = "one_to_one_pip_kernel_advanced_reparameterized_version_gpu_memory_performance_optimized",
    structure_info: Optional[dict] = None,
    create_containment_results_dataframe: bool = False,
    return_array_as: str = "cupy",
    max_test_structures_per_call: Optional[int] = None,
) -> AlignedContainmentRunResult:
    """Run an aligned batch through the general-use custom-PIP grandmother surface."""
    _validate_return_array_as(return_array_as)
    _validate_aligned_containment_test_batch(aligned_containment_test_batch)

    if create_containment_results_dataframe and structure_info is None:
        raise ValueError("structure_info is required when create_containment_results_dataframe is True")

    raw_containment_result_cp_arr, prepper_output_tuple = (
        custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p_grandparents.custom_point_containment_grandmother_function(
        list_of_relative_structures_containting_list_of_constant_zslices_arrays,
        aligned_containment_test_batch.test_structures,
        aligned_containment_test_batch.test_struct_to_relative_struct_mapping,
        max_test_structures_per_call=max_test_structures_per_call,
        constant_z_slice_polygons_handler_option=constant_z_slice_polygons_handler_option,
        remove_consecutive_duplicate_points_in_polygons=remove_consecutive_duplicate_points_in_polygons,
        log_sub_dirs_list=log_sub_dirs_list,
        log_file_name=log_file_name,
        include_edges_in_log=include_edges_in_log,
        kernel_type=kernel_type,
        )
    )

    structured_containment_result = reshape_aligned_containment_result(
        raw_containment_result_cp_arr,
        aligned_containment_test_batch,
        return_array_as=return_array_as,
    )

    containment_results_dataframe = None
    if create_containment_results_dataframe:
        containment_results_dataframe = (
            custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.create_containment_results_dataframe_type_2II(
                structure_info,
                prepper_output_tuple[0],
                aligned_containment_test_batch.test_structures,
                raw_containment_result_cp_arr,
                aligned_containment_test_batch.test_struct_to_relative_struct_mapping,
                do_not_convert_column_names_to_categorical=["Pt contained bool"],
                float_dtype=np.float32,
                int_dtype=np.int32,
            )
        )

    return AlignedContainmentRunResult(
        aligned_containment_test_batch=aligned_containment_test_batch,
        raw_containment_result=_coerce_output_array(raw_containment_result_cp_arr, return_array_as),
        structured_containment_result=structured_containment_result,
        nearest_zslice_index_and_values_3d_arr=_coerce_output_array(prepper_output_tuple[0], return_array_as),
        containment_results_dataframe=containment_results_dataframe,
    )


def reshape_aligned_containment_result(
    raw_containment_result: Any,
    aligned_containment_test_batch: AlignedContainmentTestBatch,
    return_array_as: str = "cupy",
):
    """Reshape a flat mother-function result back to instance/trial/point form."""
    _validate_return_array_as(return_array_as)
    _validate_aligned_containment_test_batch(aligned_containment_test_batch)

    raw_containment_result_cp_arr = cp.asarray(raw_containment_result)
    if raw_containment_result_cp_arr.ndim != 2:
        raise ValueError("raw_containment_result must have shape (num_test_structures, num_points_per_structure)")

    expected_shape = (
        aligned_containment_test_batch.num_test_structures,
        aligned_containment_test_batch.num_points_per_structure,
    )
    if tuple(raw_containment_result_cp_arr.shape) != expected_shape:
        raise ValueError(
            "raw_containment_result shape {} does not match aligned batch shape {}".format(
                tuple(raw_containment_result_cp_arr.shape),
                expected_shape,
            )
        )

    structured_containment_result_cp_arr = raw_containment_result_cp_arr.reshape(
        aligned_containment_test_batch.num_instances,
        aligned_containment_test_batch.num_trial_slices_per_instance,
        aligned_containment_test_batch.num_points_per_structure,
    )
    return _coerce_output_array(structured_containment_result_cp_arr, return_array_as)


def _coerce_output_array(output_array: Any, return_array_as: str):
    if return_array_as == "cupy":
        return output_array
    return cp.asnumpy(output_array)


def _validate_aligned_containment_test_batch(aligned_containment_test_batch: AlignedContainmentTestBatch) -> None:
    if aligned_containment_test_batch.num_instances <= 0:
        raise ValueError("aligned_containment_test_batch.num_instances must be positive")
    if aligned_containment_test_batch.test_structures.ndim != 3:
        raise ValueError("aligned_containment_test_batch.test_structures must have shape (num_test_structures, num_points, 3)")
    if aligned_containment_test_batch.test_structures.shape[2] != 3:
        raise ValueError("aligned_containment_test_batch.test_structures must store XYZ coordinates")
    if aligned_containment_test_batch.test_struct_to_relative_struct_mapping.ndim != 1:
        raise ValueError("aligned_containment_test_batch.test_struct_to_relative_struct_mapping must be 1D")
    if aligned_containment_test_batch.test_struct_to_relative_struct_mapping.shape[0] != aligned_containment_test_batch.num_test_structures:
        raise ValueError("aligned_containment_test_batch mapping length must match num_test_structures")


def _validate_return_array_as(return_array_as: str) -> None:
    if return_array_as not in {"cupy", "numpy"}:
        raise ValueError("return_array_as must be 'cupy' or 'numpy'")


__all__ = [
    "AlignedContainmentRunResult",
    "reshape_aligned_containment_result",
    "run_aligned_containment_batch",
]
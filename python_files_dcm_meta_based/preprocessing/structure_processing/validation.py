import copy

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from rich_preambles import NullLiveDisplay

from preprocessing.structure_processing.non_biopsy_structure_processing import preprocess_non_biopsy_structure


STRUCTURE_PREPROCESSING_VALIDATION_DF_KEY = "Structure preprocessing validation"


class NullProgress:
    def add_task(self, *args, **kwargs):
        return 0

    def update(self, *args, **kwargs):
        return None

    def advance(self, *args, **kwargs):
        return None

    def remove_task(self, *args, **kwargs):
        return None

    def start_task(self, *args, **kwargs):
        return None

    def stop_task(self, *args, **kwargs):
        return None

    def refresh(self, *args, **kwargs):
        return None


class NullImportantInfo:
    def add_text_line(self, *args, **kwargs):
        return None


class NullRuntimeLogger:
    def phase_start(self, *args, **kwargs):
        return None

    def phase_end(self, *args, **kwargs):
        return None

    def checkpoint(self, *args, **kwargs):
        return None


def _build_sparse_structure_list(structures, indices_to_copy):
    if len(indices_to_copy) == 0:
        return []

    max_index = max(indices_to_copy)
    sparse_list = [None] * (max_index + 1)
    for structure_index in indices_to_copy:
        sparse_list[structure_index] = copy.deepcopy(structures[structure_index])
    return sparse_list


def _resolve_selected_prostate_index(config, sp_patient_selected_structure_info_dataframe):
    if sp_patient_selected_structure_info_dataframe is None:
        return None

    struct_type_column = None
    if "Struct ref type" in sp_patient_selected_structure_info_dataframe.columns:
        struct_type_column = "Struct ref type"
    elif "Structure type" in sp_patient_selected_structure_info_dataframe.columns:
        struct_type_column = "Structure type"
    else:
        return None

    structure_index_column = None
    if "Index number" in sp_patient_selected_structure_info_dataframe.columns:
        structure_index_column = "Index number"
    elif "Structure index" in sp_patient_selected_structure_info_dataframe.columns:
        structure_index_column = "Structure index"
    else:
        return None

    selected_prostate_df = sp_patient_selected_structure_info_dataframe[
        sp_patient_selected_structure_info_dataframe[struct_type_column] == config.oar_ref
    ]
    if len(selected_prostate_df) == 0:
        return None

    selected_prostate_info = selected_prostate_df.iloc[0]
    selected_prostate_index = selected_prostate_info[structure_index_column]
    if pd.isna(selected_prostate_index):
        return None
    return int(selected_prostate_index)


def build_non_biopsy_structure_modular_snapshot(
    patient_uid,
    pydicom_item,
    master_structure_reference_dict,
    struct_ref_type,
    specific_structure_index,
    structs_referenced_dict,
    config,
    parallel_pool,
    layout_groups,
    sp_patient_selected_structure_info_dataframe=None,
):
    """Run the modular helper on an isolated clone and return a comparable snapshot.

    This validation path is intentionally expensive and is meant for focused
    equivalence checks, not standard cohort processing.
    """

    validation_pydicom_item = dict(pydicom_item)
    validation_pydicom_item[config.all_ref_key] = copy.deepcopy(pydicom_item[config.all_ref_key])
    validation_pydicom_item[struct_ref_type] = _build_sparse_structure_list(
        pydicom_item[struct_ref_type],
        [specific_structure_index],
    )

    if struct_ref_type == config.dil_ref and config.oar_ref in pydicom_item:
        selected_prostate_index = _resolve_selected_prostate_index(
            config,
            sp_patient_selected_structure_info_dataframe,
        )
        if selected_prostate_index is not None:
            validation_pydicom_item[config.oar_ref] = _build_sparse_structure_list(
                pydicom_item[config.oar_ref],
                [selected_prostate_index],
            )

    validation_master_structure_reference_dict = {patient_uid: validation_pydicom_item}

    preprocess_non_biopsy_structure(
        patient_uid=patient_uid,
        pydicom_item=validation_pydicom_item,
        master_structure_reference_dict=validation_master_structure_reference_dict,
        struct_ref_type=struct_ref_type,
        specific_structure_index=specific_structure_index,
        structs_referenced_dict=structs_referenced_dict,
        config=config,
        parallel_pool=parallel_pool,
        layout_groups=layout_groups,
        structures_progress=NullProgress(),
        indeterminate_progress_sub=NullProgress(),
        important_info=NullImportantInfo(),
        live_display=NullLiveDisplay(),
        runtime_logger=NullRuntimeLogger(),
        sp_patient_selected_structure_info_dataframe=sp_patient_selected_structure_info_dataframe,
    )

    return capture_non_biopsy_structure_processing_snapshot(
        master_structure_reference_dict=validation_master_structure_reference_dict,
        patient_uid=patient_uid,
        struct_ref_type=struct_ref_type,
        specific_structure_index=specific_structure_index,
        all_ref_key=config.all_ref_key,
    )


def _copy_numpy_array(value):
    if value is None:
        return None
    return np.asarray(value).copy()


def _copy_array_list(values):
    if values is None:
        return None
    return [_copy_numpy_array(value) for value in values]


def _snapshot_line_segment(segment_obj):
    if segment_obj is None:
        return None

    return {
        "segment_zslices": copy.deepcopy(getattr(segment_obj, "segment_zslices", None)),
        "segment_pt_indices": copy.deepcopy(getattr(segment_obj, "segment_pt_indices", None)),
        "segment_vector": _copy_numpy_array(getattr(segment_obj, "segment_vector", None)),
        "segment_end_points": _copy_numpy_array(getattr(segment_obj, "segment_end_points", None)),
        "segment_length": copy.deepcopy(getattr(segment_obj, "segment_length", None)),
        "num_interpolations_on_segment": copy.deepcopy(
            getattr(segment_obj, "num_interpolations_on_segment", None)
        ),
        "longest_segment_in_adjacent_slices": copy.deepcopy(
            getattr(segment_obj, "longest_segment_in_adjacent_slices", None)
        ),
    }


def _snapshot_line_segment_dict(segment_dict):
    if segment_dict is None:
        return None

    return {
        zslice_key: [_snapshot_line_segment(segment_obj) for segment_obj in segment_list]
        for zslice_key, segment_list in segment_dict.items()
    }


def _snapshot_interpolation_information(interpolation_information):
    if interpolation_information is None:
        return None

    return {
        "interpolate_distance": copy.deepcopy(
            getattr(interpolation_information, "interpolate_distance", None)
        ),
        "num_z_slices_raw": copy.deepcopy(
            getattr(interpolation_information, "num_z_slices_raw", None)
        ),
        "zslice_vals_after_interpolation_list": copy.deepcopy(
            getattr(interpolation_information, "zslice_vals_after_interpolation_list", None)
        ),
        "numpoints_after_interpolation_per_zslice_dict": copy.deepcopy(
            getattr(interpolation_information, "numpoints_after_interpolation_per_zslice_dict", None)
        ),
        "numpoints_raw_per_zslice_dict": copy.deepcopy(
            getattr(interpolation_information, "numpoints_raw_per_zslice_dict", None)
        ),
        "interpolated_pts_list": _copy_array_list(
            getattr(interpolation_information, "interpolated_pts_list", None)
        ),
        "interpolated_pts_np_arr": _copy_numpy_array(
            getattr(interpolation_information, "interpolated_pts_np_arr", None)
        ),
        "endcaps_points": _copy_array_list(
            getattr(interpolation_information, "endcaps_points", None)
        ),
        "interpolated_pts_with_end_caps_np_arr": _copy_numpy_array(
            getattr(interpolation_information, "interpolated_pts_with_end_caps_np_arr", None)
        ),
        "scipylinesegments_by_zslice_keys_dict": _snapshot_line_segment_dict(
            getattr(interpolation_information, "scipylinesegments_by_zslice_keys_dict", None)
        ),
    }


def _snapshot_point_cloud(point_cloud):
    if point_cloud is None:
        return None

    point_cloud_snapshot = {
        "points": _copy_numpy_array(point_cloud.points),
        "colors": _copy_numpy_array(point_cloud.colors),
    }

    if point_cloud.has_normals():
        point_cloud_snapshot["normals"] = _copy_numpy_array(point_cloud.normals)
    else:
        point_cloud_snapshot["normals"] = None

    return point_cloud_snapshot


def _snapshot_triangle_mesh(triangle_mesh):
    if triangle_mesh is None:
        return None

    triangle_mesh_snapshot = {
        "vertices": _copy_numpy_array(triangle_mesh.vertices),
        "triangles": _copy_numpy_array(triangle_mesh.triangles),
    }

    if triangle_mesh.has_vertex_normals():
        triangle_mesh_snapshot["vertex_normals"] = _copy_numpy_array(triangle_mesh.vertex_normals)
    else:
        triangle_mesh_snapshot["vertex_normals"] = None

    return triangle_mesh_snapshot


def capture_non_biopsy_structure_processing_snapshot(
    master_structure_reference_dict,
    patient_uid,
    struct_ref_type,
    specific_structure_index,
    all_ref_key,
):
    specific_structure = master_structure_reference_dict[patient_uid][struct_ref_type][specific_structure_index]
    patient_output_dataframes_dict = master_structure_reference_dict[patient_uid][all_ref_key][
        "Multi-structure pre-processing output dataframes dict"
    ]

    interpolated_pcd_dict = specific_structure.get("Interpolated structure point cloud dict")

    return {
        "metadata": {
            "patient_uid": patient_uid,
            "structure_id": specific_structure.get("ROI"),
            "structure_type": struct_ref_type,
            "structure_index": specific_structure_index,
            "structure_refnum": specific_structure.get("Ref #"),
        },
        "structure": {
            "Raw contour pts": _copy_numpy_array(specific_structure.get("Raw contour pts")),
            "Equal num zslice contour pts": _copy_array_list(
                specific_structure.get("Equal num zslice contour pts")
            ),
            "Inter-slice interpolation information": _snapshot_interpolation_information(
                specific_structure.get("Inter-slice interpolation information")
            ),
            "Intra-slice interpolation information": _snapshot_interpolation_information(
                specific_structure.get("Intra-slice interpolation information")
            ),
            "Maximum pairwise distance": copy.deepcopy(
                specific_structure.get("Maximum pairwise distance")
            ),
            "Structure volume": copy.deepcopy(specific_structure.get("Structure volume")),
            "Voxel size for structure volume calc": copy.deepcopy(
                specific_structure.get("Voxel size for structure volume calc")
            ),
            "Structure dimension at centroid dict": copy.deepcopy(
                specific_structure.get("Structure dimension at centroid dict")
            ),
            "Voxel size for structure dimension calc": copy.deepcopy(
                specific_structure.get("Voxel size for structure dimension calc")
            ),
            "Structure surface area": copy.deepcopy(
                specific_structure.get("Structure surface area")
            ),
            "Structure features dataframe": copy.deepcopy(
                specific_structure.get("Structure features dataframe")
            ),
            "Point cloud raw": _snapshot_point_cloud(specific_structure.get("Point cloud raw")),
            "Interpolated structure point cloud dict": {
                pcd_name: _snapshot_point_cloud(point_cloud)
                for pcd_name, point_cloud in (interpolated_pcd_dict or {}).items()
            },
            "Structure OPEN3D triangle mesh object": _snapshot_triangle_mesh(
                specific_structure.get("Structure OPEN3D triangle mesh object")
            ),
        },
        "patient_output_dataframes": {
            "MR - ADC - summary statistics by structure dataframe": copy.deepcopy(
                patient_output_dataframes_dict.get(
                    "MR - ADC - summary statistics by structure dataframe"
                )
            ),
            "Prostate only points MR ADC dataframe (temporary for pre-processing)": copy.deepcopy(
                patient_output_dataframes_dict.get(
                    "Prostate only points MR ADC dataframe (temporary for pre-processing)"
                )
            ),
        },
    }


def _record_mismatch(mismatches, path, reason):
    mismatches.append({"path": path, "reason": reason})


def _compare_numpy_arrays(path, left_value, right_value, mismatches, float_rtol, float_atol):
    if left_value.shape != right_value.shape:
        _record_mismatch(
            mismatches,
            path,
            f"array shape mismatch: modular {left_value.shape}, legacy {right_value.shape}",
        )
        return

    if np.issubdtype(left_value.dtype, np.number) and np.issubdtype(right_value.dtype, np.number):
        if not np.allclose(left_value, right_value, rtol=float_rtol, atol=float_atol, equal_nan=True):
            _record_mismatch(mismatches, path, "numeric array values differ")
        return

    if not np.array_equal(left_value, right_value):
        _record_mismatch(mismatches, path, "array values differ")


def _compare_values(path, left_value, right_value, mismatches, float_rtol, float_atol):
    if left_value is None or right_value is None:
        if left_value is not right_value:
            _record_mismatch(mismatches, path, f"value mismatch: modular={left_value}, legacy={right_value}")
        return

    if isinstance(left_value, np.ndarray) and isinstance(right_value, np.ndarray):
        _compare_numpy_arrays(path, left_value, right_value, mismatches, float_rtol, float_atol)
        return

    if isinstance(left_value, pd.DataFrame) and isinstance(right_value, pd.DataFrame):
        try:
            assert_frame_equal(
                left_value,
                right_value,
                check_dtype=True,
                check_like=False,
                rtol=float_rtol,
                atol=float_atol,
            )
        except AssertionError as exc:
            _record_mismatch(mismatches, path, str(exc).splitlines()[0])
        return

    if isinstance(left_value, dict) and isinstance(right_value, dict):
        left_keys = set(left_value.keys())
        right_keys = set(right_value.keys())
        if left_keys != right_keys:
            _record_mismatch(
                mismatches,
                path,
                f"dict keys mismatch: modular={sorted(left_keys)}, legacy={sorted(right_keys)}",
            )
        for shared_key in sorted(left_keys.intersection(right_keys), key=str):
            _compare_values(
                f"{path}.{shared_key}",
                left_value[shared_key],
                right_value[shared_key],
                mismatches,
                float_rtol,
                float_atol,
            )
        return

    if isinstance(left_value, (list, tuple)) and isinstance(right_value, (list, tuple)):
        if len(left_value) != len(right_value):
            _record_mismatch(
                mismatches,
                path,
                f"sequence length mismatch: modular={len(left_value)}, legacy={len(right_value)}",
            )
            return
        for index, (left_item, right_item) in enumerate(zip(left_value, right_value)):
            _compare_values(
                f"{path}[{index}]",
                left_item,
                right_item,
                mismatches,
                float_rtol,
                float_atol,
            )
        return

    if isinstance(left_value, (float, np.floating)) or isinstance(right_value, (float, np.floating)):
        if not np.isclose(left_value, right_value, rtol=float_rtol, atol=float_atol, equal_nan=True):
            _record_mismatch(
                mismatches,
                path,
                f"float mismatch: modular={left_value}, legacy={right_value}",
            )
        return

    if left_value != right_value:
        _record_mismatch(mismatches, path, f"value mismatch: modular={left_value}, legacy={right_value}")


def compare_non_biopsy_structure_processing_snapshots(
    modular_snapshot,
    legacy_snapshot,
    float_rtol=1e-7,
    float_atol=1e-9,
):
    mismatches = []
    _compare_values(
        path="snapshot",
        left_value=modular_snapshot,
        right_value=legacy_snapshot,
        mismatches=mismatches,
        float_rtol=float_rtol,
        float_atol=float_atol,
    )

    mismatch_fields = [mismatch["path"] for mismatch in mismatches]
    mismatch_details = [f"{mismatch['path']}: {mismatch['reason']}" for mismatch in mismatches[:10]]

    return {
        "patient_uid": legacy_snapshot["metadata"]["patient_uid"],
        "structure_id": legacy_snapshot["metadata"]["structure_id"],
        "structure_type": legacy_snapshot["metadata"]["structure_type"],
        "structure_index": legacy_snapshot["metadata"]["structure_index"],
        "structure_refnum": legacy_snapshot["metadata"]["structure_refnum"],
        "overall_match_bool": len(mismatches) == 0,
        "mismatch_count": len(mismatches),
        "mismatch_fields": mismatch_fields,
        "mismatch_summary": " | ".join(mismatch_details) if len(mismatch_details) > 0 else "",
    }


def append_non_biopsy_structure_validation_result(
    master_structure_reference_dict,
    patient_uid,
    all_ref_key,
    validation_result,
):
    patient_output_dataframes_dict = master_structure_reference_dict[patient_uid][all_ref_key][
        "Multi-structure pre-processing output dataframes dict"
    ]

    validation_result_row = pd.DataFrame(
        {
            "Patient ID": [validation_result["patient_uid"]],
            "Structure ID": [validation_result["structure_id"]],
            "Structure type": [validation_result["structure_type"]],
            "Structure index": [validation_result["structure_index"]],
            "Structure refnum": [validation_result["structure_refnum"]],
            "Overall match bool": [validation_result["overall_match_bool"]],
            "Mismatch count": [validation_result["mismatch_count"]],
            "Mismatch fields": ["; ".join(validation_result["mismatch_fields"])],
            "Mismatch summary": [validation_result["mismatch_summary"]],
        }
    )

    existing_validation_dataframe = patient_output_dataframes_dict.get(
        STRUCTURE_PREPROCESSING_VALIDATION_DF_KEY
    )
    if existing_validation_dataframe is None:
        patient_output_dataframes_dict[STRUCTURE_PREPROCESSING_VALIDATION_DF_KEY] = validation_result_row
        return

    patient_output_dataframes_dict[STRUCTURE_PREPROCESSING_VALIDATION_DF_KEY] = pd.concat(
        [existing_validation_dataframe, validation_result_row],
        ignore_index=True,
    )
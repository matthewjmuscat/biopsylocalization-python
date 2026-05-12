import copy
import pickle
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

import dataframe_builders
import lattice_reconstruction_tools
import misc_tools
import plotting_funcs
import point_containment_tools


PREPROCESSED_EXPORT_MODE = "preprocessed"
RESULTS_EXPORT_MODE = "results"
EXPORT_METADATA_KEY = "Dataset export metadata"
PICKLE_SANITIZER_VERSION = 3

EXPORT_BOUNDARY_STAGE_BY_MODE = {
    PREPROCESSED_EXPORT_MODE: "post_preprocessing",
    RESULTS_EXPORT_MODE: "post_results",
}

EXPORT_EXCLUDED_KEYS_BY_MODE = {
    PREPROCESSED_EXPORT_MODE: {
        "Biopsy optimization: Optimal biopsy location (entire cubic lattice) dataframe",
        "Simulated biopsy transport request dict",
    },
    RESULTS_EXPORT_MODE: {
        "Biopsy optimization - Target DIL optimizer v2 stage boundary render jobs",
    },
}

EXPORT_EXCLUDED_KEY_PREFIXES_BY_MODE = {
    PREPROCESSED_EXPORT_MODE: (
        "Biopsy optimization - ",
        "MC data:",
        "FANOVA:",
    ),
    RESULTS_EXPORT_MODE: (),
}


GENERAL_STRUCTURE_RUNTIME_KEYS = {
    "Point cloud raw",
    "Interpolated structure point cloud dict",
    "Structure OPEN3D triangle mesh object",
}

RESULTS_BX_RUNTIME_KEYS = {
    "Random uniformly sampled volume pts pcd",
    "Random uniformly sampled volume pts bx coord sys pcd",
    "Bounding box for random uniformly sampled volume pts",
    "MC data: bx and structure shifted dict",
    "MC data: bx to dose NN search objects list",
    "FANOVA: sobol indices (containment)",
    "FANOVA: sobol indices (dose)",
    "FANOVA: sobol indices (DIL tissue)",
}

DOSE_RUNTIME_KEYS = {
    "Dose grid point cloud",
    "Dose grid point cloud thresholded",
    "Dose grid gradient point cloud",
    "Dose grid gradient point cloud thresholded",
    "KDtree",
    "KDtree gradient",
}

MR_RUNTIME_KEYS = {
    "MR ADC grid point cloud",
    "MR ADC grid point cloud thresholded",
    "KDtree",
}

NON_PICKLABLE_RUNTIME_MODULE_FRAGMENTS = (
    "open3d",
    "scipy.spatial",
    "sklearn.neighbors",
    "rich.",
)

NON_PICKLABLE_RUNTIME_CLASS_FRAGMENTS = (
    "PointCloud",
    "TriangleMesh",
    "LineSet",
    "KDTree",
    "KDTreeFlann",
)


class _DropValueType:
    pass


DROP_VALUE = _DropValueType()


def _pop_keys(mapping_obj, key_names):
    for key_name in key_names:
        mapping_obj.pop(key_name, None)


def _should_drop_key_for_export_mode(key, export_mode):
    if not isinstance(key, str):
        return False

    if key in EXPORT_EXCLUDED_KEYS_BY_MODE.get(export_mode, set()):
        return True

    return any(
        key.startswith(key_prefix)
        for key_prefix in EXPORT_EXCLUDED_KEY_PREFIXES_BY_MODE.get(export_mode, ())
    )


def _prune_value_for_export_mode(value, export_mode):
    if isinstance(value, dict):
        keys_to_drop = [
            key for key in value.keys() if _should_drop_key_for_export_mode(key, export_mode)
        ]
        _pop_keys(value, keys_to_drop)
        for nested_value in value.values():
            _prune_value_for_export_mode(nested_value, export_mode)
        return

    if isinstance(value, (list, tuple, set)):
        for nested_value in value:
            _prune_value_for_export_mode(nested_value, export_mode)


def _is_known_non_picklable_runtime_object(value):
    value_type = type(value)
    module_name = getattr(value_type, "__module__", "")
    class_name = getattr(value_type, "__name__", "")

    if any(module_fragment in module_name for module_fragment in NON_PICKLABLE_RUNTIME_MODULE_FRAGMENTS):
        return True

    return any(class_fragment in class_name for class_fragment in NON_PICKLABLE_RUNTIME_CLASS_FRAGMENTS)


def _clone_value_for_pickle(value):
    if isinstance(value, dict):
        value_safe = {}
        for key, nested_value in value.items():
            nested_value_safe = _clone_value_for_pickle(nested_value)
            if nested_value_safe is DROP_VALUE:
                continue
            value_safe[key] = nested_value_safe
        return value_safe

    if isinstance(value, list):
        value_safe = []
        for nested_value in value:
            nested_value_safe = _clone_value_for_pickle(nested_value)
            if nested_value_safe is DROP_VALUE:
                continue
            value_safe.append(nested_value_safe)
        return value_safe

    if isinstance(value, tuple):
        value_safe = []
        for nested_value in value:
            nested_value_safe = _clone_value_for_pickle(nested_value)
            if nested_value_safe is DROP_VALUE:
                continue
            value_safe.append(nested_value_safe)
        return tuple(value_safe)

    if isinstance(value, set):
        value_safe = []
        for nested_value in value:
            nested_value_safe = _clone_value_for_pickle(nested_value)
            if nested_value_safe is DROP_VALUE:
                continue
            value_safe.append(nested_value_safe)
        try:
            return type(value)(value_safe)
        except TypeError:
            return value_safe

    if _is_known_non_picklable_runtime_object(value):
        return DROP_VALUE

    return value


def _clone_delaunay_without_lineset(delaunay_obj):
    if delaunay_obj is None:
        return None

    try:
        delaunay_obj_safe = copy.copy(delaunay_obj)
    except Exception:
        return None

    if hasattr(delaunay_obj_safe, "delaunay_line_set"):
        delaunay_obj_safe.delaunay_line_set = None

    return delaunay_obj_safe


def _clone_reconstructed_biopsy_model_dict_for_pickle(reconstructed_biopsy_model_dict):
    if reconstructed_biopsy_model_dict is None:
        return None

    reconstructed_biopsy_model_dict_safe = {}
    for key, value in reconstructed_biopsy_model_dict.items():
        if key == "Reconstructed structure point cloud":
            reconstructed_biopsy_model_dict_safe[key] = None
            continue

        if key == "Reconstructed structure delaunay global":
            reconstructed_biopsy_model_dict_safe[key] = _clone_delaunay_without_lineset(value)
            continue

        value_safe = _clone_value_for_pickle(value)
        if value_safe is DROP_VALUE:
            continue
        reconstructed_biopsy_model_dict_safe[key] = value_safe

    return reconstructed_biopsy_model_dict_safe


def _clone_simulated_biopsy_planning_dict_for_pickle(simulated_biopsy_planning_dict):
    if simulated_biopsy_planning_dict is None:
        return None

    simulated_biopsy_planning_dict_safe = {}
    for key, value in simulated_biopsy_planning_dict.items():
        if key == "Planned reconstructed biopsy model dict":
            simulated_biopsy_planning_dict_safe[key] = _clone_reconstructed_biopsy_model_dict_for_pickle(value)
            continue

        value_safe = _clone_value_for_pickle(value)
        if value_safe is DROP_VALUE:
            continue
        simulated_biopsy_planning_dict_safe[key] = value_safe

    return simulated_biopsy_planning_dict_safe


def _clone_general_structure_for_pickle(specific_structure,
                                        export_mode):
    specific_structure_safe = {}
    remove_uncertainty_data = export_mode == RESULTS_EXPORT_MODE

    for key, value in specific_structure.items():
        if key in GENERAL_STRUCTURE_RUNTIME_KEYS:
            continue

        if remove_uncertainty_data and key == "Uncertainty data":
            continue

        value_safe = _clone_value_for_pickle(value)
        if value_safe is DROP_VALUE:
            continue
        specific_structure_safe[key] = value_safe

    _prune_value_for_export_mode(specific_structure_safe, export_mode)

    return specific_structure_safe


def _clone_bx_structure_for_pickle(specific_bx_structure,
                                   export_mode):
    specific_bx_structure_safe = {}
    remove_uncertainty_data = export_mode == RESULTS_EXPORT_MODE

    for key, value in specific_bx_structure.items():
        if key in GENERAL_STRUCTURE_RUNTIME_KEYS:
            continue

        if remove_uncertainty_data and key == "Uncertainty data":
            continue

        if export_mode == RESULTS_EXPORT_MODE and key in RESULTS_BX_RUNTIME_KEYS:
            continue

        if key == "Reconstructed structure point cloud":
            specific_bx_structure_safe[key] = None
            continue

        if key == "Reconstructed structure delaunay global":
            specific_bx_structure_safe[key] = _clone_delaunay_without_lineset(value)
            continue

        if key == "Simulated biopsy planning dict":
            specific_bx_structure_safe[key] = _clone_simulated_biopsy_planning_dict_for_pickle(value)
            continue

        value_safe = _clone_value_for_pickle(value)
        if value_safe is DROP_VALUE:
            continue
        specific_bx_structure_safe[key] = value_safe

    _prune_value_for_export_mode(specific_bx_structure_safe, export_mode)

    return specific_bx_structure_safe


def _clone_dose_dict_for_pickle(dose_ref_dict,
                                export_mode):
    dose_ref_dict_safe = {}

    for key, value in dose_ref_dict.items():
        if key in DOSE_RUNTIME_KEYS:
            continue

        value_safe = _clone_value_for_pickle(value)
        if value_safe is DROP_VALUE:
            continue
        dose_ref_dict_safe[key] = value_safe

    _prune_value_for_export_mode(dose_ref_dict_safe, export_mode)

    return dose_ref_dict_safe


def _clone_mr_dict_for_pickle(mr_adc_subdict,
                              export_mode):
    mr_adc_subdict_safe = {}

    for key, value in mr_adc_subdict.items():
        if key in MR_RUNTIME_KEYS:
            continue

        value_safe = _clone_value_for_pickle(value)
        if value_safe is DROP_VALUE:
            continue
        mr_adc_subdict_safe[key] = value_safe

    _prune_value_for_export_mode(mr_adc_subdict_safe, export_mode)

    return mr_adc_subdict_safe


def _git_command_output(working_dir, args):
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=str(working_dir),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _build_git_metadata(export_dir):
    repo_root = _git_command_output(export_dir, ["rev-parse", "--show-toplevel"])
    if repo_root is None:
        return None

    return {
        "repo root": repo_root,
        "commit": _git_command_output(export_dir, ["rev-parse", "HEAD"]),
        "short commit": _git_command_output(export_dir, ["rev-parse", "--short", "HEAD"]),
        "branch": _git_command_output(export_dir, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(_git_command_output(export_dir, ["status", "--porcelain"])),
    }


def _build_export_metadata(export_mode,
                           export_dir,
                           reference_dict_filename,
                           info_dict_filename,
                           summary_filename=None):
    export_metadata = {
        "export mode": export_mode,
        "export boundary": EXPORT_BOUNDARY_STAGE_BY_MODE.get(export_mode, export_mode),
        "generated at utc": datetime.now(timezone.utc).isoformat(),
        "sanitizer version": PICKLE_SANITIZER_VERSION,
        "python version": sys.version.split()[0],
        "reference dict filename": reference_dict_filename,
        "info dict filename": info_dict_filename,
    }

    excluded_keys = sorted(EXPORT_EXCLUDED_KEYS_BY_MODE.get(export_mode, set()))
    if excluded_keys:
        export_metadata["excluded keys"] = excluded_keys

    excluded_key_prefixes = list(EXPORT_EXCLUDED_KEY_PREFIXES_BY_MODE.get(export_mode, ()))
    if excluded_key_prefixes:
        export_metadata["excluded key prefixes"] = excluded_key_prefixes

    if summary_filename is not None:
        export_metadata["summary filename"] = summary_filename

    git_metadata = _build_git_metadata(export_dir)
    if git_metadata is not None:
        export_metadata["git"] = git_metadata

    return export_metadata


def _build_exportable_master_structure_info_dict(master_structure_info_dict,
                                                 export_metadata):
    master_structure_info_dict_safe = copy.deepcopy(master_structure_info_dict)
    master_structure_info_dict_safe.setdefault("Global", {})[EXPORT_METADATA_KEY] = export_metadata
    return master_structure_info_dict_safe


def build_pickle_safe_master_structure_reference_dict(master_structure_reference_dict,
                                                      export_mode,
                                                      bx_ref,
                                                      oar_ref,
                                                      dil_ref,
                                                      rectum_ref_key,
                                                      urethra_ref_key,
                                                      dose_ref,
                                                      mr_adc_ref):
    master_structure_reference_dict_safe = {}

    for patient_uid, pydicom_item in master_structure_reference_dict.items():
        pydicom_item_safe = {}

        for key, value in pydicom_item.items():
            if key == bx_ref:
                pydicom_item_safe[key] = [_clone_bx_structure_for_pickle(specific_bx_structure, export_mode) for specific_bx_structure in value]
                continue

            if key in {oar_ref, dil_ref, rectum_ref_key, urethra_ref_key}:
                pydicom_item_safe[key] = [
                    _clone_general_structure_for_pickle(specific_structure, export_mode)
                    for specific_structure in value
                ]
                continue

            if key == dose_ref:
                pydicom_item_safe[key] = _clone_dose_dict_for_pickle(value, export_mode)
                continue

            if key == mr_adc_ref:
                pydicom_item_safe[key] = _clone_mr_dict_for_pickle(value, export_mode)
                continue

            value_safe = _clone_value_for_pickle(value)
            if value_safe is DROP_VALUE:
                continue
            pydicom_item_safe[key] = value_safe

        _prune_value_for_export_mode(pydicom_item_safe, export_mode)

        master_structure_reference_dict_safe[patient_uid] = pydicom_item_safe

    return master_structure_reference_dict_safe


def load_pickle_bundle(master_structure_reference_dict_path,
                       master_structure_info_dict_path):
    with open(master_structure_reference_dict_path, "rb") as master_structure_reference_dict_file:
        master_structure_reference_dict = pickle.load(master_structure_reference_dict_file)

    with open(master_structure_info_dict_path, "rb") as master_structure_info_dict_file:
        master_structure_info_dict = pickle.load(master_structure_info_dict_file)

    return master_structure_reference_dict, master_structure_info_dict


def export_preprocessed_pickle_bundle(master_structure_reference_dict,
                                      master_structure_info_dict,
                                      export_dir,
                                      reference_dict_filename,
                                      info_dict_filename,
                                      summary_filename,
                                      structs_referenced_list,
                                      bx_ref,
                                      oar_ref,
                                      dil_ref,
                                      rectum_ref_key,
                                      urethra_ref_key,
                                      dose_ref,
                                      mr_adc_ref):
    master_structure_reference_dict_safe = build_pickle_safe_master_structure_reference_dict(
        master_structure_reference_dict,
        PREPROCESSED_EXPORT_MODE,
        bx_ref,
        oar_ref,
        dil_ref,
        rectum_ref_key,
        urethra_ref_key,
        dose_ref,
        mr_adc_ref,
    )
    export_metadata = _build_export_metadata(
        PREPROCESSED_EXPORT_MODE,
        export_dir,
        reference_dict_filename,
        info_dict_filename,
        summary_filename=summary_filename,
    )
    master_structure_info_dict_safe = _build_exportable_master_structure_info_dict(
        master_structure_info_dict,
        export_metadata,
    )

    reference_dict_path = export_dir.joinpath(reference_dict_filename)
    with open(reference_dict_path, "wb") as master_structure_reference_dict_file:
        pickle.dump(master_structure_reference_dict_safe, master_structure_reference_dict_file)

    info_dict_path = export_dir.joinpath(info_dict_filename)
    with open(info_dict_path, "wb") as master_structure_info_dict_file:
        pickle.dump(master_structure_info_dict_safe, master_structure_info_dict_file)

    summary_path = export_dir.joinpath(summary_filename)
    preprocessed_info_dataframe = dataframe_builders.preprocessed_dataset_summary_dataframe_builder(
        master_structure_reference_dict,
        master_structure_info_dict,
        structs_referenced_list,
    )
    preprocessed_info_dataframe.to_csv(summary_path, index=False)

    return {
        "reference_dict_path": reference_dict_path,
        "info_dict_path": info_dict_path,
        "summary_path": summary_path,
    }


def export_results_pickle_bundle(master_structure_reference_dict,
                                 master_structure_info_dict,
                                 export_dir,
                                 reference_dict_filename,
                                 info_dict_filename,
                                 bx_ref,
                                 oar_ref,
                                 dil_ref,
                                 rectum_ref_key,
                                 urethra_ref_key,
                                 dose_ref,
                                 mr_adc_ref):
    master_structure_reference_dict_safe = build_pickle_safe_master_structure_reference_dict(
        master_structure_reference_dict,
        RESULTS_EXPORT_MODE,
        bx_ref,
        oar_ref,
        dil_ref,
        rectum_ref_key,
        urethra_ref_key,
        dose_ref,
        mr_adc_ref,
    )
    export_metadata = _build_export_metadata(
        RESULTS_EXPORT_MODE,
        export_dir,
        reference_dict_filename,
        info_dict_filename,
    )
    master_structure_info_dict_safe = _build_exportable_master_structure_info_dict(
        master_structure_info_dict,
        export_metadata,
    )

    reference_dict_path = export_dir.joinpath(reference_dict_filename)
    with open(reference_dict_path, "wb") as master_structure_reference_dict_file:
        pickle.dump(master_structure_reference_dict_safe, master_structure_reference_dict_file)

    info_dict_path = export_dir.joinpath(info_dict_filename)
    with open(info_dict_path, "wb") as master_structure_info_dict_file:
        pickle.dump(master_structure_info_dict_safe, master_structure_info_dict_file)

    return {
        "reference_dict_path": reference_dict_path,
        "info_dict_path": info_dict_path,
    }


def _structure_pcd_color(structs_referenced_dict,
                         structure_type,
                         specific_structure,
                         bx_ref):
    if structure_type == bx_ref:
        return structs_referenced_dict[structure_type]["PCD color dict"][specific_structure["Simulated type"]]

    return structs_referenced_dict[structure_type]["PCD color"]


def _rebuild_planned_simulated_biopsy_runtime_objects(specific_bx_structure,
                                                      pcd_struct_color):
    simulated_biopsy_planning_dict = specific_bx_structure.get("Simulated biopsy planning dict")
    if simulated_biopsy_planning_dict is None:
        return

    planned_reconstructed_biopsy_model_dict = simulated_biopsy_planning_dict.get("Planned reconstructed biopsy model dict")
    if planned_reconstructed_biopsy_model_dict is None:
        return

    drawn_biopsy_array = planned_reconstructed_biopsy_model_dict.get("Reconstructed structure pts arr")
    if drawn_biopsy_array is not None:
        planned_reconstructed_biopsy_model_dict["Reconstructed structure point cloud"] = point_containment_tools.create_point_cloud(
            drawn_biopsy_array,
            pcd_struct_color,
        )

    planned_reconstructed_bx_delaunay_global_convex_structure_obj = planned_reconstructed_biopsy_model_dict.get("Reconstructed structure delaunay global")
    if planned_reconstructed_bx_delaunay_global_convex_structure_obj is not None:
        planned_reconstructed_bx_delaunay_global_convex_structure_obj.generate_lineset()
        planned_reconstructed_biopsy_model_dict["Reconstructed structure delaunay global"] = planned_reconstructed_bx_delaunay_global_convex_structure_obj


def rebuild_loaded_preprocessed_runtime_objects(master_structure_reference_dict,
                                                master_structure_info_dict,
                                                structs_referenced_list_generalized,
                                                structs_referenced_dict,
                                                bx_ref,
                                                dose_ref,
                                                mr_adc_ref,
                                                interp_inter_slice_dist,
                                                interp_intra_slice_dist,
                                                radius_for_normals_estimation,
                                                max_nn_for_normals_estimation,
                                                lower_bound_dose_value,
                                                lower_bound_dose_gradient_value,
                                                lower_bound_mr_adc_value,
                                                upper_bound_mr_adc_value,
                                                color_flattening_deg_MR,
                                                patients_progress,
                                                completed_progress,
                                                indeterminate_progress_sub,
                                                live_display):
    patient_uid_default = "Initializing"
    pickling_dose_patients_task_main_description = "[red]Rebuilding non-picklable dose data [{}]...".format(patient_uid_default)
    pickling_dose_patients_task_completed_main_description = "[green]Rebuilding non-picklable dose data"
    pickling_dose_patients_task = patients_progress.add_task(pickling_dose_patients_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
    pickling_dose_patients_task_completed = completed_progress.add_task(pickling_dose_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible=False)

    for patient_uid, pydicom_item in master_structure_reference_dict.items():
        pickling_dose_patients_task_main_description = "[red]Rebuilding non-picklable dose data [{}]...".format(patient_uid)
        patients_progress.update(pickling_dose_patients_task, description=pickling_dose_patients_task_main_description)

        if dose_ref not in pydicom_item:
            patients_progress.update(pickling_dose_patients_task, advance=1)
            completed_progress.update(pickling_dose_patients_task_completed, advance=1)
            continue

        dose_ref_dict = pydicom_item[dose_ref]
        phys_space_dose_map_and_gradient_map_3d_arr = dose_ref_dict["Dose and gradient phys space and pixel 3d arr"]

        dose_point_cloud, dose_gradient_arrows_point_cloud = plotting_funcs.create_dose_point_cloud_with_gradients(
            phys_space_dose_map_and_gradient_map_3d_arr,
            paint_dose_color=True,
            arrow_scale=1.0,
            truncate_below_dose=None,
            truncate_below_gradient_norm=None,
        )
        thresholded_dose_point_cloud, thresholded_dose_gradient_arrows_point_cloud = plotting_funcs.create_dose_point_cloud_with_gradients(
            phys_space_dose_map_and_gradient_map_3d_arr,
            paint_dose_color=True,
            arrow_scale=1.0,
            truncate_below_dose=lower_bound_dose_value,
            truncate_below_gradient_norm=lower_bound_dose_gradient_value,
        )

        dose_ref_dict["Dose grid point cloud"] = dose_point_cloud
        dose_ref_dict["Dose grid point cloud thresholded"] = thresholded_dose_point_cloud
        dose_ref_dict["Dose grid gradient point cloud"] = dose_gradient_arrows_point_cloud
        dose_ref_dict["Dose grid gradient point cloud thresholded"] = thresholded_dose_gradient_arrows_point_cloud
        master_structure_reference_dict[patient_uid][dose_ref] = dose_ref_dict

        patients_progress.update(pickling_dose_patients_task, advance=1)
        completed_progress.update(pickling_dose_patients_task_completed, advance=1)

    patients_progress.update(pickling_dose_patients_task, visible=False)
    completed_progress.update(pickling_dose_patients_task_completed, visible=True)

    patient_uid_default = "Initializing"
    pickling_mr_patients_task_main_description = "[red]Rebuilding non-picklable MR data [{}]...".format(patient_uid_default)
    pickling_mr_patients_task_completed_main_description = "[green]Rebuilding non-picklable MR data"
    pickling_mr_patients_task = patients_progress.add_task(pickling_mr_patients_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
    pickling_mr_patients_task_completed = completed_progress.add_task(pickling_mr_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible=False)

    for patient_uid, pydicom_item in master_structure_reference_dict.items():
        pickling_mr_patients_task_main_description = "[red]Rebuilding non-picklable MR data [{}]...".format(patient_uid)
        patients_progress.update(pickling_mr_patients_task, description=pickling_mr_patients_task_main_description)

        if mr_adc_ref not in pydicom_item:
            patients_progress.update(pickling_mr_patients_task, advance=1)
            completed_progress.update(pickling_mr_patients_task_completed, advance=1)
            continue

        mr_adc_subdict = pydicom_item[mr_adc_ref]
        filtered_non_negative_adc_mr_phys_space_arr = lattice_reconstruction_tools.reconstruct_mr_lattice_with_coordinates_from_dict_v2(
            mr_adc_subdict,
            filter_out_negatives=True,
        )

        mr_adc_point_cloud = plotting_funcs.create_MR_point_cloud(
            filtered_non_negative_adc_mr_phys_space_arr,
            color_flattening_deg_MR,
            paint_mr_color=True,
        )
        thresholded_mr_adc_point_cloud = plotting_funcs.create_thresholded_MR_ADC_point_cloud(
            filtered_non_negative_adc_mr_phys_space_arr,
            color_flattening_deg_MR,
            paint_mr_color=True,
            lower_bound=lower_bound_mr_adc_value,
            upper_bound=upper_bound_mr_adc_value,
        )

        mr_adc_subdict["MR ADC grid point cloud"] = mr_adc_point_cloud
        mr_adc_subdict["MR ADC grid point cloud thresholded"] = thresholded_mr_adc_point_cloud

        patients_progress.update(pickling_mr_patients_task, advance=1)
        completed_progress.update(pickling_mr_patients_task_completed, advance=1)

    patients_progress.update(pickling_mr_patients_task, visible=False)
    completed_progress.update(pickling_mr_patients_task_completed, visible=True)

    patient_uid_default = "Initializing"
    pickling_structure_patients_task_main_description = "[red]Rebuilding non-picklable structure data [{}]...".format(patient_uid_default)
    pickling_structure_patients_task_completed_main_description = "[green]Rebuilding non-picklable structure data"
    pickling_structure_patients_task = patients_progress.add_task(pickling_structure_patients_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
    pickling_structure_patients_task_completed = completed_progress.add_task(pickling_structure_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible=False)

    for patient_uid, pydicom_item in master_structure_reference_dict.items():
        pickling_structure_patients_task_main_description = "[red]Rebuilding non-picklable structure data [{}]...".format(patient_uid)
        patients_progress.update(pickling_structure_patients_task, description=pickling_structure_patients_task_main_description)

        for structure_type in structs_referenced_list_generalized:
            for specific_structure_index, specific_structure in enumerate(pydicom_item[structure_type]):
                specific_structure_roi = specific_structure["ROI"]
                pcd_struct_color = _structure_pcd_color(structs_referenced_dict, structure_type, specific_structure, bx_ref)

                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Creating pcds of interp structures [{}]".format(specific_structure_roi), total=None)
                interslice_interpolation_information = pydicom_item[structure_type][specific_structure_index]["Inter-slice interpolation information"]
                interpolation_information = pydicom_item[structure_type][specific_structure_index]["Intra-slice interpolation information"]
                three_d_data_array_fully_interpolated = interpolation_information.interpolated_pts_np_arr
                three_d_data_array_fully_interpolated_with_end_caps = interpolation_information.interpolated_pts_with_end_caps_np_arr
                three_d_data_array_interslice_interpolation = np.vstack(interslice_interpolation_information.interpolated_pts_list)

                interslice_interp_pcd = point_containment_tools.create_point_cloud(three_d_data_array_interslice_interpolation, pcd_struct_color)
                inter_and_intra_interp_pcd = point_containment_tools.create_point_cloud(three_d_data_array_fully_interpolated, pcd_struct_color)
                inter_and_intra_and_end_caps_interp_pcd = point_containment_tools.create_point_cloud(three_d_data_array_fully_interpolated_with_end_caps, pcd_struct_color)
                interpolated_pcd_dict = {"Interslice": interslice_interp_pcd, "Full": inter_and_intra_interp_pcd, "Full with end caps": inter_and_intra_and_end_caps_interp_pcd}
                master_structure_reference_dict[patient_uid][structure_type][specific_structure_index]["Interpolated structure point cloud dict"] = interpolated_pcd_dict
                indeterminate_progress_sub.update(indeterminate_task, visible=False)

                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Creating pcds of raw structures [{}]".format(specific_structure_roi), total=None)
                three_d_data_array = pydicom_item[structure_type][specific_structure_index]["Raw contour pts"]
                three_d_data_point_cloud = point_containment_tools.create_point_cloud(three_d_data_array, pcd_struct_color)
                master_structure_reference_dict[patient_uid][structure_type][specific_structure_index]["Point cloud raw"] = three_d_data_point_cloud
                indeterminate_progress_sub.update(indeterminate_task, visible=False)

                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Creating trimesh [{}]".format(specific_structure_roi), total=None)
                live_display.refresh()
                fully_interp_with_end_caps_structure_triangle_mesh, _ = misc_tools.compute_structure_triangle_mesh(
                    interp_inter_slice_dist,
                    interp_intra_slice_dist,
                    three_d_data_array_fully_interpolated_with_end_caps,
                    radius_for_normals_estimation,
                    max_nn_for_normals_estimation,
                )
                master_structure_reference_dict[patient_uid][structure_type][specific_structure_index]["Structure OPEN3D triangle mesh object"] = fully_interp_with_end_caps_structure_triangle_mesh
                indeterminate_progress_sub.update(indeterminate_task, visible=False)

                if structure_type == bx_ref:
                    _rebuild_planned_simulated_biopsy_runtime_objects(specific_structure, pcd_struct_color)

                    indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Creating pcd of rcn bpsy [{}]".format(specific_structure_roi), total=None)
                    drawn_biopsy_array = pydicom_item[structure_type][specific_structure_index]["Reconstructed structure pts arr"]
                    reconstructed_biopsy_point_cloud = point_containment_tools.create_point_cloud(drawn_biopsy_array, pcd_struct_color)
                    master_structure_reference_dict[patient_uid][structure_type][specific_structure_index]["Reconstructed structure point cloud"] = reconstructed_biopsy_point_cloud
                    indeterminate_progress_sub.update(indeterminate_task, visible=False)

                    indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Creating delaunay lineset of rcn bpsy [{}]".format(specific_structure_roi), total=None)
                    reconstructed_bx_delaunay_global_convex_structure_obj = pydicom_item[structure_type][specific_structure_index]["Reconstructed structure delaunay global"]
                    if reconstructed_bx_delaunay_global_convex_structure_obj is not None:
                        reconstructed_bx_delaunay_global_convex_structure_obj.generate_lineset()
                        master_structure_reference_dict[patient_uid][structure_type][specific_structure_index]["Reconstructed structure delaunay global"] = reconstructed_bx_delaunay_global_convex_structure_obj
                    indeterminate_progress_sub.update(indeterminate_task, visible=False)

        patients_progress.update(pickling_structure_patients_task, advance=1)
        completed_progress.update(pickling_structure_patients_task_completed, advance=1)

    patients_progress.update(pickling_structure_patients_task, visible=False)
    completed_progress.update(pickling_structure_patients_task_completed, visible=True)

    return live_display
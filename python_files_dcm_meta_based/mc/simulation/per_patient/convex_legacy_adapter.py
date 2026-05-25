"""Patient-level adapter for the current convex MC simulator oracle.

This module lets a future patient runner call the validated
``MC_simulator_convex.simulator_parallel`` surface for a singleton patient while
the deeper containment and dose loop bodies are migrated into this package.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from presentation import LegacyNullProgress, LegacyPresentationContext

from .containment import PatientContainmentOutputs, collect_patient_containment_outputs
from .contracts import MCConvexPatientRunResult, MCConvexSimulationConfig
from .dose import PatientDoseOutputs, collect_patient_dose_outputs


class NullStopwatch:
    """No-op stopwatch for headless legacy-adapter execution."""

    def start(self, *args: Any, **kwargs: Any) -> None:
        return None

    def stop(self, *args: Any, **kwargs: Any) -> None:
        return None


def _build_legacy_mc_layout_groups(context: LegacyPresentationContext) -> tuple[Any, list[Any], Any, Any]:
    if context.layout_groups is not None:
        return context.layout_groups
    mc_trial_progress = LegacyNullProgress()
    progress_group_info_list = [
        context.completed_progress,
        context.completed_sections_progress,
        context.patients_progress,
        context.structures_progress,
        context.biopsies_progress,
        mc_trial_progress,
        context.indeterminate_progress_main,
        context.indeterminate_progress_sub,
        None,
    ]
    return (None, progress_group_info_list, context.important_info, None)


def build_single_patient_mc_master_info(patient_uid: str,
                                        patient_info_dict: Mapping[str, Any] | None,
                                        *,
                                        global_info: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build the legacy master-info shape for one MC patient."""
    patient_uid = str(patient_uid)
    if patient_info_dict is not None and "Global" in patient_info_dict and "By patient" in patient_info_dict:
        master_info = copy.deepcopy(dict(patient_info_dict))
        by_patient = copy.deepcopy(dict(master_info.get("By patient", {})))
        if patient_uid in by_patient:
            master_info["By patient"] = {patient_uid: by_patient[patient_uid]}
        master_info.setdefault("Global", {})["Num cases"] = 1
        return master_info

    resolved_global_info = copy.deepcopy(dict(global_info or {}))
    resolved_global_info["Num cases"] = 1
    return {
        "Global": resolved_global_info,
        "By patient": {
            patient_uid: copy.deepcopy(dict(patient_info_dict or {})),
        },
    }


def collect_mc_patient_outputs(patient_uid: str,
                               patient_reference_dict: Mapping[str, Any],
                               *,
                               bx_ref: str) -> tuple[PatientContainmentOutputs, PatientDoseOutputs]:
    """Collect patient MC containment and dose outputs from legacy storage."""
    containment_outputs = collect_patient_containment_outputs(
        patient_uid,
        patient_reference_dict,
        bx_ref=bx_ref,
    )
    dose_outputs = collect_patient_dose_outputs(
        patient_uid,
        patient_reference_dict,
        bx_ref=bx_ref,
    )
    return containment_outputs, dose_outputs


def run_patient_mc_convex_legacy_adapter(
    *,
    patient_uid: str,
    patient_reference_dict: dict[str, Any],
    patient_info_dict: Mapping[str, Any] | None,
    config: MCConvexSimulationConfig,
    parallel_pool: Any,
    presentation_context: LegacyPresentationContext | None = None,
    stopwatch: Any = None,
    global_info: Mapping[str, Any] | None = None,
    mutate_input: bool = True,
) -> MCConvexPatientRunResult:
    """Run the current convex MC simulator against a singleton patient cohort."""
    from MC_simulator_convex import simulator_parallel

    patient_uid = str(patient_uid)
    working_patient_reference_dict = patient_reference_dict if mutate_input else copy.deepcopy(patient_reference_dict)
    master_structure_reference_dict = {patient_uid: working_patient_reference_dict}
    master_structure_info_dict = build_single_patient_mc_master_info(
        patient_uid,
        patient_info_dict,
        global_info=global_info,
    )
    context = presentation_context or LegacyPresentationContext.null()
    layout_groups = _build_legacy_mc_layout_groups(context)
    resolved_stopwatch = stopwatch or NullStopwatch()
    keys = config.keys
    runtime = config.runtime
    containment = config.containment
    dose = config.dose

    live_display = simulator_parallel(
        parallel_pool,
        context.live_display,
        resolved_stopwatch,
        layout_groups,
        master_structure_reference_dict,
        keys.structs_referenced_list,
        keys.structs_referenced_dict,
        keys.bx_ref,
        keys.oar_ref,
        keys.dil_ref,
        keys.rectum_ref,
        keys.urethra_ref,
        keys.dose_ref,
        keys.plan_ref,
        keys.all_ref_key,
        master_structure_info_dict,
        dose.biopsy_z_voxel_length,
        dose.num_dose_calc_NN,
        dose.num_dose_NN_to_show_for_animation_plotting,
        dose.dose_views_jsons_paths_list,
        containment.containment_views_jsons_paths_list,
        dose.show_NN_dose_demonstration_plots,
        dose.show_NN_dose_demonstration_plots_all_trials_at_once,
        containment.show_num_containment_demonstration_plots,
        containment.containment_results_structure_types_to_show_per_trial,
        containment.show_num_nearest_neighbour_surface_boundary_demonstration,
        containment.show_num_relative_structure_centroid_demonstration,
        runtime.biopsy_needle_compartment_length,
        runtime.simulate_uniform_bx_shifts_due_to_bx_needle_compartment,
        runtime.plot_uniform_shifts_to_check_plotly,
        dose.differential_dvh_resolution,
        dose.cumulative_dvh_resolution,
        dose.v_percent_DVH_to_calc_list,
        dose.volume_DVH_quantiles_to_calculate,
        runtime.plot_translation_vectors_pointclouds,
        containment.plot_cupy_containment_distribution_results,
        runtime.plot_shifted_biopsies,
        containment.structure_miss_probability_roi,
        containment.cancer_tissue_label,
        containment.default_exterior_tissue,
        containment.miss_structure_complement_label,
        containment.tissue_length_above_probability_threshold_list,
        containment.n_bootstraps_for_tissue_length_above_threshold,
        containment.perform_mc_containment_sim,
        dose.perform_mc_dose_sim,
        runtime.spinner_type,
        runtime.cupy_array_upper_limit_NxN_size_input,
        runtime.nearest_zslice_vals_and_indices_cupy_generic_max_size,
        dose.idw_power,
        dose.raw_data_mc_dosimetry_dump_bool,
        containment.raw_data_mc_containment_dump_bool,
        containment.keep_light_containment_and_distances_to_relative_structures_dataframe_bool,
        containment.show_non_bx_relative_structure_z_dilation_bool,
        containment.show_non_bx_relative_structure_xy_dilation_bool,
        containment.generate_cuda_log_files_MC_containment_sim,
        runtime.custom_cuda_kernel_type,
        containment.constant_z_slice_polygons_handler_option,
        containment.remove_consecutive_duplicate_points_in_polygons,
        containment.interp_dist_caps,
        containment.cuml_NN_algo,
        containment.check_if_end_caps_filled_proper_NN_num,
        containment.nn_search_end_cap_grid_factor,
        containment.tissue_volume_operator_dictionary,
    )
    containment_outputs, dose_outputs = collect_mc_patient_outputs(
        patient_uid,
        working_patient_reference_dict,
        bx_ref=keys.bx_ref,
    )
    mc_info = master_structure_info_dict.get("Global", {}).get("MC info", {})
    return MCConvexPatientRunResult(
        patient_uid=patient_uid,
        patient_reference_dict=working_patient_reference_dict,
        master_structure_reference_dict=master_structure_reference_dict,
        master_structure_info_dict=master_structure_info_dict,
        containment_outputs=containment_outputs,
        dose_outputs=dose_outputs,
        presentation_context=context,
        live_display=live_display,
        performed_flags={
            "MC containment sim performed": mc_info.get("MC containment sim performed"),
            "MC dose sim performed": mc_info.get("MC dose sim performed"),
            "MC sim performed": mc_info.get("MC sim performed"),
        },
        metadata={"mutated_input": bool(mutate_input)},
    )

"""Opt-in patient-runner adapters for patient-local scientific stages."""

from __future__ import annotations

from functools import partial
from typing import Any, Mapping, Sequence

from .contracts import LegacyPatientRuntimeState
from .contracts import PatientRunConfig
from .contracts import PatientStageName
from .contracts import PatientStageResult
from .runner import PatientStage
from .scientific_config import PatientRunnerScientificConfig
from .stages import write_patient_artifacts_stage


DEFAULT_SCIENTIFIC_STAGE_ORDER = (
    PatientStageName.GRID_PREPROCESSING,
    PatientStageName.PREPROCESSING,
    PatientStageName.MC_PREP,
    PatientStageName.MC_SIMULATION,
    PatientStageName.OPTIMIZATION,
    PatientStageName.GUIDANCE,
)


def build_patient_scientific_stages(
    scientific_config: PatientRunnerScientificConfig,
    *,
    include_artifact_writing: bool = True,
    stage_order: Sequence[PatientStageName | str] = DEFAULT_SCIENTIFIC_STAGE_ORDER,
) -> tuple[PatientStage, ...]:
    """Build an opt-in stage sequence for independently run patient science."""
    if not isinstance(scientific_config, PatientRunnerScientificConfig):
        raise TypeError("scientific_config must be a PatientRunnerScientificConfig instance")

    stage_builders = {
        PatientStageName.GRID_PREPROCESSING.value: _grid_preprocessing_stage(scientific_config),
        PatientStageName.PREPROCESSING.value: _preprocessing_stage(scientific_config),
        PatientStageName.MC_PREP.value: _mc_prep_stage(scientific_config),
        PatientStageName.MC_SIMULATION.value: _mc_simulation_stage(scientific_config),
        PatientStageName.OPTIMIZATION.value: _optimization_stage(scientific_config),
        PatientStageName.GUIDANCE.value: _guidance_stage(scientific_config),
    }
    stages: list[PatientStage] = []
    for stage_name in stage_order:
        resolved_stage_name = _stage_name_value(stage_name)
        if resolved_stage_name not in stage_builders:
            raise ValueError(f"Unsupported patient scientific stage name: {resolved_stage_name}")
        stage = stage_builders[resolved_stage_name]
        if stage is not None:
            stages.append(stage)

    if include_artifact_writing:
        stages.append(PatientStage(PatientStageName.PATIENT_ARTIFACT_WRITING, write_patient_artifacts_stage))
    return tuple(stages)


def run_patient_grid_preprocessing_scientific_stage(
    runtime_state: LegacyPatientRuntimeState,
    config: PatientRunConfig,
    *,
    scientific_config: PatientRunnerScientificConfig,
) -> PatientStageResult:
    """Run configured patient-local grid preprocessing slices."""
    del config
    stage_config = scientific_config.grid_preprocessing
    if stage_config is None or not stage_config.enabled:
        return PatientStageResult.skipped(PatientStageName.GRID_PREPROCESSING, reason="grid_preprocessing_not_configured")

    metadata: dict[str, Any] = {"patient_uid": runtime_state.patient_uid, "steps": []}
    pydicom_item = runtime_state.pydicom_item

    if stage_config.dose_grid_config is not None:
        from preprocessing.dose_grid_processing import build_dose_grid_runtime_objects_for_patient

        dose_config = stage_config.dose_grid_config
        dose_ref = str(dose_config.dose_ref)
        metadata["dose_reference_available"] = dose_ref in pydicom_item
        if dose_ref in pydicom_item:
            lower_bound_dose_value = build_dose_grid_runtime_objects_for_patient(
                pydicom_item,
                dose_config,
                None,
                None,
                None,
                None,
                None,
            )
            metadata["steps"].append("dose_grid_runtime_objects")
            metadata["lower_bound_dose_value"] = lower_bound_dose_value

    if stage_config.mr_adc_input_normalization is not None:
        from preprocessing.mr_adc_input_checking import normalize_patient_mr_adc_input

        normalization_config = stage_config.mr_adc_input_normalization
        mr_input_result = normalize_patient_mr_adc_input(
            patient_uid=runtime_state.patient_uid,
            pydicom_item=pydicom_item,
            mr_adc_ref=normalization_config.mr_adc_ref,
            previous_mr_adc_units=normalization_config.previous_mr_adc_units,
            important_info=_NullImportantInfo(),
            live_display=None,
        )
        metadata["steps"].append("mr_adc_input_normalization")
        metadata.update(
            {
                "mr_adc_input_has_mr_adc": mr_input_result.has_mr_adc,
                "mr_adc_input_selected_series_uid": mr_input_result.selected_series_uid,
                "mr_adc_input_selected_units": mr_input_result.selected_units,
                "mr_adc_input_num_series": mr_input_result.num_mr_adc_series,
                "mr_adc_input_multiple_series_found": mr_input_result.multiple_series_found,
                "mr_adc_input_units_match_previous": mr_input_result.units_match_previous,
            }
        )

    if stage_config.mr_adc_grid_config is not None:
        from preprocessing.mr_adc_grid_processing import build_mr_adc_grid_runtime_objects_for_patient

        mr_grid_config = stage_config.mr_adc_grid_config
        mr_adc_ref = str(mr_grid_config.mr_adc_ref)
        metadata["mr_adc_reference_available"] = mr_adc_ref in pydicom_item
        if mr_adc_ref in pydicom_item:
            build_mr_adc_grid_runtime_objects_for_patient(
                runtime_state.patient_uid,
                pydicom_item,
                mr_grid_config,
                None,
                None,
                None,
                None,
                None,
            )
            metadata["steps"].append("mr_adc_grid_runtime_objects")

    return PatientStageResult.success(PatientStageName.GRID_PREPROCESSING, metadata=metadata)


def run_patient_preprocessing_scientific_stage(
    runtime_state: LegacyPatientRuntimeState,
    config: PatientRunConfig,
    *,
    scientific_config: PatientRunnerScientificConfig,
) -> PatientStageResult:
    """Run configured patient-local preprocessing slices."""
    del config
    stage_config = scientific_config.preprocessing
    if stage_config is None or not stage_config.enabled:
        return PatientStageResult.skipped(PatientStageName.PREPROCESSING, reason="preprocessing_not_configured")

    metadata: dict[str, Any] = {"patient_uid": runtime_state.patient_uid, "steps": []}
    pydicom_item = runtime_state.pydicom_item

    if stage_config.real_biopsy_processing is not None:
        from preprocessing.biopsy_processing.per_patient import process_patient_real_biopsies

        real_biopsy_config = stage_config.real_biopsy_processing
        real_biopsy_count = _count_biopsies_by_simulated_flag(pydicom_item, runtime_state.bx_ref, simulated=False)
        process_patient_real_biopsies(
            patient_uid=runtime_state.patient_uid,
            pydicom_item=pydicom_item,
            master_structure_reference_dict=runtime_state.master_structure_reference_dict,
            structs_referenced_dict=real_biopsy_config.structs_referenced_dict,
            bx_ref=runtime_state.bx_ref,
            parallel_pool=_require_parallel_pool(scientific_config, PatientStageName.PREPROCESSING),
            interp_inter_slice_dist=real_biopsy_config.interp_inter_slice_dist,
            interp_intra_slice_dist=real_biopsy_config.interp_intra_slice_dist,
            interp_dist_caps=real_biopsy_config.interp_dist_caps,
            biopsy_radius=real_biopsy_config.biopsy_radius,
            display_pca_fit_variation_for_biopsies_bool=(
                real_biopsy_config.display_pca_fit_variation_for_biopsies_bool
            ),
            voxel_size_for_structure_volume_calc_non_bx=(
                real_biopsy_config.voxel_size_for_structure_volume_calc_non_bx
            ),
            factor_for_voxel_size=real_biopsy_config.factor_for_voxel_size,
            cupy_array_upper_limit_NxN_size_input=real_biopsy_config.cupy_array_upper_limit_nxn_size_input,
            nearest_zslice_vals_and_indices_cupy_generic_max_size=(
                real_biopsy_config.nearest_zslice_vals_and_indices_cupy_generic_max_size
            ),
            generate_cuda_log_files_volume_calculation=(
                real_biopsy_config.generate_cuda_log_files_volume_calculation
            ),
            constant_z_slice_polygons_handler_option=real_biopsy_config.constant_z_slice_polygons_handler_option,
            remove_consecutive_duplicate_points_in_polygons=(
                real_biopsy_config.remove_consecutive_duplicate_points_in_polygons
            ),
            include_edges_in_log_files=real_biopsy_config.include_edges_in_log_files,
            custom_cuda_kernel_type=real_biopsy_config.custom_cuda_kernel_type,
            demonstrate_volume_calculation_correctness_bool_1=(
                real_biopsy_config.demonstrate_volume_calculation_correctness_bool_1
            ),
            plot_volume_calculation_containment_result_bool_1_old=(
                real_biopsy_config.plot_volume_calculation_containment_result_bool_1_old
            ),
            plot_binary_mask_bool=real_biopsy_config.plot_binary_mask_bool,
        )
        metadata["steps"].append("real_biopsy_processing")
        metadata["real_biopsy_count"] = int(real_biopsy_count)

    if stage_config.simulated_biopsy_preparation is not None:
        from preprocessing.biopsy_processing.per_patient import prepare_patient_simulated_biopsies

        preparation_config = stage_config.simulated_biopsy_preparation
        length_results, preparation_dataframe = prepare_patient_simulated_biopsies(
            patient_uid=runtime_state.patient_uid,
            pydicom_item=pydicom_item,
            bx_ref=runtime_state.bx_ref,
            dil_ref=preparation_config.dil_ref,
            all_ref_key=runtime_state.all_ref_key,
            simulated_biopsy_length_method=preparation_config.simulated_biopsy_length_method,
            biopsy_needle_compartment_length=preparation_config.biopsy_needle_compartment_length,
            master_structure_info_dict=runtime_state.master_structure_info_dict,
        )
        metadata["steps"].append("simulated_biopsy_preparation")
        metadata["simulated_biopsy_preparation_rows"] = int(len(preparation_dataframe))
        metadata["simulated_biopsy_length_result_count"] = len(length_results or {})

    if stage_config.simulated_biopsy_planning is not None:
        from preprocessing.biopsy_processing.per_patient import plan_patient_simulated_biopsies

        planning_config = stage_config.simulated_biopsy_planning
        simulated_biopsy_count = _count_biopsies_by_simulated_flag(pydicom_item, runtime_state.bx_ref, simulated=True)
        plan_patient_simulated_biopsies(
            patient_uid=runtime_state.patient_uid,
            pydicom_item=pydicom_item,
            bx_ref=runtime_state.bx_ref,
            bx_sample_pts_lattice_spacing=planning_config.bx_sample_pts_lattice_spacing,
            parallel_pool=_require_parallel_pool(scientific_config, PatientStageName.PREPROCESSING),
            centroid_line_vec_sim_list=planning_config.centroid_line_vec_sim_list,
            centroid_first_pos_sim_list=planning_config.centroid_first_pos_sim_list,
            num_centroids_for_sim_bxs=planning_config.num_centroids_for_sim_bxs,
            simulated_bx_rad=planning_config.simulated_bx_rad,
            plot_simulated_cores_immediately=planning_config.plot_simulated_cores_immediately,
        )
        metadata["steps"].append("simulated_biopsy_planning")
        metadata["simulated_biopsy_planning_count"] = int(simulated_biopsy_count)

    if stage_config.uncertainty_attachment is not None:
        from preprocessing.uncertainty_attachment import attach_patient_uncertainty_data_from_dataframe

        count = attach_patient_uncertainty_data_from_dataframe(
            patient_uid=runtime_state.patient_uid,
            pydicom_item=pydicom_item,
            read_uncertainties_dataframe=stage_config.uncertainty_attachment.read_uncertainties_dataframe,
            uncertainty_data_cls=stage_config.uncertainty_attachment.uncertainty_data_cls,
        )
        metadata["steps"].append("uncertainty_attachment")
        metadata["uncertainty_attached_count"] = int(count)

    if stage_config.realized_biopsy_targeting is not None:
        from preprocessing.biopsy_processing.per_patient.realized_biopsy_targeting import (
            determine_patient_realized_biopsy_targeting,
        )

        targeting_config = stage_config.realized_biopsy_targeting
        determine_patient_realized_biopsy_targeting(
            patient_uid=runtime_state.patient_uid,
            pydicom_item=pydicom_item,
            all_ref_key=runtime_state.all_ref_key,
            bx_ref=runtime_state.bx_ref,
            oar_ref=targeting_config.oar_ref,
            dil_ref=targeting_config.dil_ref,
        )
        metadata["steps"].append("realized_biopsy_targeting")
        metadata["realized_biopsy_count"] = len(pydicom_item.get(runtime_state.bx_ref, ()))

    if stage_config.sampled_biopsy_processing is not None:
        from preprocessing.biopsy_processing.sampled_biopsy_processing import process_patient_sampled_biopsies

        sampled_config = stage_config.sampled_biopsy_processing
        sampled_result = process_patient_sampled_biopsies(
            patient_uid=runtime_state.patient_uid,
            pydicom_item=pydicom_item,
            bx_ref=runtime_state.bx_ref,
            bx_sample_pts_lattice_spacing=sampled_config.bx_sample_pts_lattice_spacing,
            parallel_pool=_require_parallel_pool(scientific_config, PatientStageName.PREPROCESSING),
            show_reconstructed_biopsy_in_biopsy_coord_sys_tr_and_rot=(
                sampled_config.show_reconstructed_biopsy_in_biopsy_coord_sys_tr_and_rot
            ),
        )
        metadata["steps"].append("sampled_biopsy_processing")
        metadata.update(_prefix_mapping(sampled_result, "sampled_biopsy_"))

    return PatientStageResult.success(PatientStageName.PREPROCESSING, metadata=metadata)


def run_patient_mc_prep_scientific_stage(
    runtime_state: LegacyPatientRuntimeState,
    config: PatientRunConfig,
    *,
    scientific_config: PatientRunnerScientificConfig,
) -> PatientStageResult:
    """Run configured patient-local MC prep transform stages."""
    del config
    stage_config = scientific_config.mc_prep
    if stage_config is None or not stage_config.enabled:
        return PatientStageResult.skipped(PatientStageName.MC_PREP, reason="mc_prep_not_configured")

    from mc.prep.per_patient import apply_patient_biopsy_self_transforms
    from mc.prep.per_patient import apply_patient_relative_structure_transforms
    from mc.prep.per_patient import generate_transformations_for_patient

    pydicom_item = runtime_state.pydicom_item
    max_simulations = _resolve_mc_prep_max_simulations(runtime_state.master_structure_info_dict, stage_config)
    num_generated_transform_samples = stage_config.num_generated_transform_samples
    if stage_config.run_transform_generation and num_generated_transform_samples is None:
        num_generated_transform_samples = max_simulations
    num_mc_containment_simulations = None
    if stage_config.run_relative_structure_transforms:
        num_mc_containment_simulations = _resolve_optional_mc_info_int(
            runtime_state.master_structure_info_dict,
            stage_config.num_mc_containment_simulations,
            "Num MC containment simulations",
        )
    metadata: dict[str, Any] = {
        "patient_uid": runtime_state.patient_uid,
        "steps": [],
    }
    if num_generated_transform_samples is not None:
        metadata["num_generated_transform_samples"] = int(num_generated_transform_samples)
    if max_simulations is not None:
        metadata["max_simulations"] = int(max_simulations)
    if num_mc_containment_simulations is not None:
        metadata["num_mc_containment_simulations"] = int(num_mc_containment_simulations)

    if stage_config.run_transform_generation:
        generate_transformations_for_patient(
            patient_uid=runtime_state.patient_uid,
            pydicom_item=pydicom_item,
            simulate_uniform_bx_shifts_due_to_bx_needle_compartment=(
                stage_config.simulate_uniform_bx_shifts_due_to_bx_needle_compartment
            ),
            bx_ref=runtime_state.bx_ref,
            biopsy_needle_compartment_length=stage_config.biopsy_needle_compartment_length,
            num_generated_transform_samples=num_generated_transform_samples,
            structs_referenced_list=stage_config.structs_referenced_list,
            rng=scientific_config.resources.rng,
        )
        metadata["steps"].append("transform_generation")

    if stage_config.run_biopsy_self_transforms:
        if max_simulations is None:
            raise ValueError("max_simulations is required for biopsy self transforms")
        biopsy_transform_result = apply_patient_biopsy_self_transforms(
            patient_uid=runtime_state.patient_uid,
            pydicom_item=pydicom_item,
            bx_ref=runtime_state.bx_ref,
            max_simulations=max_simulations,
            simulate_uniform_bx_shifts_due_to_bx_needle_compartment=(
                stage_config.simulate_uniform_bx_shifts_due_to_bx_needle_compartment
            ),
            inspect_self_biopsy_dilate_bool=stage_config.inspect_self_biopsy_dilate_bool,
            inspect_self_biopsy_dilate_and_rotate_bool=stage_config.inspect_self_biopsy_dilate_and_rotate_bool,
            inspect_self_biopsy_dilate_and_rotate_and_translate_bool=(
                stage_config.inspect_self_biopsy_dilate_and_rotate_and_translate_bool
            ),
        )
        metadata["steps"].append("biopsy_self_transforms")
        metadata.update(_prefix_mapping(biopsy_transform_result, "biopsy_self_"))

    if stage_config.run_relative_structure_transforms:
        if num_mc_containment_simulations is None:
            raise ValueError("num_mc_containment_simulations is required for relative structure transforms")
        relative_transform_result = apply_patient_relative_structure_transforms(
            patient_uid=runtime_state.patient_uid,
            pydicom_item=pydicom_item,
            structs_referenced_list=stage_config.structs_referenced_list,
            bx_ref=runtime_state.bx_ref,
            num_MC_containment_simulations=num_mc_containment_simulations,
            inspect_relative_structure_rotate_and_shift_number=(
                stage_config.inspect_relative_structure_rotate_and_shift_number
            ),
        )
        metadata["steps"].append("relative_structure_transforms")
        metadata.update(_prefix_mapping(relative_transform_result, "relative_structure_"))

    return PatientStageResult.success(PatientStageName.MC_PREP, metadata=metadata)


def run_patient_mc_simulation_scientific_stage(
    runtime_state: LegacyPatientRuntimeState,
    config: PatientRunConfig,
    *,
    scientific_config: PatientRunnerScientificConfig,
) -> PatientStageResult:
    """Run configured patient-local MC simulation stages."""
    del config
    stage_config = scientific_config.mc_simulation
    if stage_config is None or not stage_config.enabled:
        return PatientStageResult.skipped(PatientStageName.MC_SIMULATION, reason="mc_simulation_not_configured")

    metadata: dict[str, Any] = {"patient_uid": runtime_state.patient_uid, "steps": []}
    if stage_config.convex_config is not None:
        from mc.simulation.per_patient import run_patient_mc_convex_stage

        parallel_pool = scientific_config.resources.parallel_pool
        if stage_config.convex_config.containment.perform_mc_containment_sim:
            parallel_pool = _require_parallel_pool(scientific_config, PatientStageName.MC_SIMULATION)
        convex_result = run_patient_mc_convex_stage(
            patient_uid=runtime_state.patient_uid,
            patient_reference_dict=runtime_state.pydicom_item,
            patient_info_dict=runtime_state.master_structure_info_dict,
            config=stage_config.convex_config,
            parallel_pool=parallel_pool,
            mutate_input=True,
        )
        _merge_mc_performed_flags(runtime_state.master_structure_info_dict, convex_result.performed_flags)
        metadata["steps"].append("convex")
        metadata.update(
            {
                "convex_containment_biopsy_count": convex_result.containment_biopsy_count,
                "convex_dose_biopsy_count": convex_result.dose_biopsy_count,
                "convex_dose_reference_available": convex_result.dose_reference_available,
                "convex_plan_reference_available": convex_result.plan_reference_available,
            }
        )
        metadata.update(_prefix_mapping(convex_result.metadata, "convex_"))

    if stage_config.mr_config is not None:
        from mc.simulation.per_patient import run_patient_mr_adc_localization_stage

        mr_result = run_patient_mr_adc_localization_stage(
            patient_uid=runtime_state.patient_uid,
            patient_reference_dict=runtime_state.pydicom_item,
            patient_info_dict=runtime_state.master_structure_info_dict,
            bx_ref=runtime_state.bx_ref,
            mr_adc_ref=stage_config.mr_adc_ref,
            config=stage_config.mr_config,
            mutate_input=True,
        )
        _merge_mc_performed_flags(runtime_state.master_structure_info_dict, mr_result.performed_flags)
        metadata["steps"].append("mr_adc")
        metadata.update(
            {
                "mr_adc_biopsy_count": mr_result.biopsy_count,
                "mr_adc_reference_available": mr_result.mr_reference_available,
            }
        )
        metadata.update(_prefix_mapping(mr_result.metadata, "mr_adc_"))

    return PatientStageResult.success(PatientStageName.MC_SIMULATION, metadata=metadata)


def run_patient_optimization_scientific_stage(
    runtime_state: LegacyPatientRuntimeState,
    config: PatientRunConfig,
    *,
    scientific_config: PatientRunnerScientificConfig,
) -> PatientStageResult:
    """Run configured patient-local optimizer stages."""
    del config
    stage_config = scientific_config.optimization
    if stage_config is None or not stage_config.enabled:
        return PatientStageResult.skipped(PatientStageName.OPTIMIZATION, reason="optimization_not_configured")

    metadata: dict[str, Any] = {"patient_uid": runtime_state.patient_uid, "steps": []}
    if stage_config.optimizer_v1_config is not None:
        from biopsy_optimizer.v1.per_patient import run_patient_optimizer_v1_stage

        optimizer_v1_result = run_patient_optimizer_v1_stage(
            patient_uid=runtime_state.patient_uid,
            patient_reference_dict=runtime_state.pydicom_item,
            patient_info_dict=runtime_state.master_structure_info_dict,
            config=stage_config.optimizer_v1_config,
            mutate_input=True,
        )
        metadata["steps"].append("optimizer_v1")
        metadata["optimizer_v1_dil_count"] = optimizer_v1_result.dil_count
        metadata.update(_prefix_mapping(optimizer_v1_result.metadata, "optimizer_v1_"))

    if stage_config.optimizer_v2_config is not None:
        from biopsy_optimizer.v2.per_patient import run_patient_target_dil_optimizer_v2_stage

        optimizer_v2_result = run_patient_target_dil_optimizer_v2_stage(
            patient_uid=runtime_state.patient_uid,
            patient_reference_dict=runtime_state.pydicom_item,
            patient_info_dict=runtime_state.master_structure_info_dict,
            config=stage_config.optimizer_v2_config,
            parallel_pool=_require_parallel_pool(scientific_config, PatientStageName.OPTIMIZATION),
            resolved_max_test_structures_per_call=(
                stage_config.optimizer_v2_resolved_max_test_structures_per_call
            ),
            resolved_max_candidates_per_chunk=stage_config.optimizer_v2_resolved_max_candidates_per_chunk,
        )
        metadata["steps"].append("optimizer_v2")
        metadata["optimizer_v2_target_structure_count"] = optimizer_v2_result.target_structure_count
        metadata["optimizer_v2_resolved_max_test_structures_per_call"] = (
            optimizer_v2_result.resolved_max_test_structures_per_call
        )
        metadata["optimizer_v2_resolved_max_candidates_per_chunk"] = (
            optimizer_v2_result.resolved_max_candidates_per_chunk
        )
        metadata.update(_prefix_mapping(optimizer_v2_result.metadata, "optimizer_v2_"))

    return PatientStageResult.success(PatientStageName.OPTIMIZATION, metadata=metadata)


def run_patient_guidance_scientific_stage(
    runtime_state: LegacyPatientRuntimeState,
    config: PatientRunConfig,
    *,
    scientific_config: PatientRunnerScientificConfig,
) -> PatientStageResult:
    """Run configured patient-local guidance-map precompute."""
    del config
    stage_config = scientific_config.guidance
    if stage_config is None or not stage_config.enabled:
        return PatientStageResult.skipped(PatientStageName.GUIDANCE, reason="guidance_not_configured")

    from guidance_maps.planning import precompute_guidance_map_firing_depth_recommendations_for_patient

    guidance_result = precompute_guidance_map_firing_depth_recommendations_for_patient(
        patient_uid=runtime_state.patient_uid,
        pydicom_item=runtime_state.pydicom_item,
        dil_ref=stage_config.dil_ref,
        all_ref_key=runtime_state.all_ref_key,
        oar_ref=stage_config.oar_ref,
        rectum_ref=stage_config.rectum_ref,
        biopsy_fire_travel_distances=stage_config.biopsy_fire_travel_distances,
        biopsy_needle_compartment_length=stage_config.biopsy_needle_compartment_length,
        interp_inter_slice_dist=stage_config.interp_inter_slice_dist,
        interp_intra_slice_dist=stage_config.interp_intra_slice_dist,
        radius_for_normals_estimation=stage_config.radius_for_normals_estimation,
        max_nn_for_normals_estimation=stage_config.max_nn_for_normals_estimation,
        biopsy_needle_tip_length=stage_config.biopsy_needle_tip_length,
        planning_config=stage_config.planning_config,
        runtime_logger=scientific_config.resources.runtime_logger,
    )
    return PatientStageResult.success(
        PatientStageName.GUIDANCE,
        metadata={
            "patient_uid": runtime_state.patient_uid,
            "guidance_dataframe_key": guidance_result.patient_dataframe_key,
            "guidance_row_count": guidance_result.patient_row_count,
            "guidance_elapsed_seconds": guidance_result.elapsed_seconds,
        },
    )


def _preprocessing_stage(scientific_config: PatientRunnerScientificConfig) -> PatientStage | None:
    if scientific_config.preprocessing is None or not scientific_config.preprocessing.enabled:
        return None
    return PatientStage(
        PatientStageName.PREPROCESSING,
        partial(run_patient_preprocessing_scientific_stage, scientific_config=scientific_config),
    )


def _grid_preprocessing_stage(scientific_config: PatientRunnerScientificConfig) -> PatientStage | None:
    if scientific_config.grid_preprocessing is None or not scientific_config.grid_preprocessing.enabled:
        return None
    return PatientStage(
        PatientStageName.GRID_PREPROCESSING,
        partial(run_patient_grid_preprocessing_scientific_stage, scientific_config=scientific_config),
    )


def _mc_prep_stage(scientific_config: PatientRunnerScientificConfig) -> PatientStage | None:
    if scientific_config.mc_prep is None or not scientific_config.mc_prep.enabled:
        return None
    return PatientStage(
        PatientStageName.MC_PREP,
        partial(run_patient_mc_prep_scientific_stage, scientific_config=scientific_config),
    )


def _mc_simulation_stage(scientific_config: PatientRunnerScientificConfig) -> PatientStage | None:
    if scientific_config.mc_simulation is None or not scientific_config.mc_simulation.enabled:
        return None
    return PatientStage(
        PatientStageName.MC_SIMULATION,
        partial(run_patient_mc_simulation_scientific_stage, scientific_config=scientific_config),
    )


def _optimization_stage(scientific_config: PatientRunnerScientificConfig) -> PatientStage | None:
    if scientific_config.optimization is None or not scientific_config.optimization.enabled:
        return None
    return PatientStage(
        PatientStageName.OPTIMIZATION,
        partial(run_patient_optimization_scientific_stage, scientific_config=scientific_config),
    )


def _guidance_stage(scientific_config: PatientRunnerScientificConfig) -> PatientStage | None:
    if scientific_config.guidance is None or not scientific_config.guidance.enabled:
        return None
    return PatientStage(
        PatientStageName.GUIDANCE,
        partial(run_patient_guidance_scientific_stage, scientific_config=scientific_config),
    )


def _stage_name_value(stage_name: PatientStageName | str) -> str:
    if isinstance(stage_name, PatientStageName):
        return stage_name.value
    return str(stage_name)


def _count_biopsies_by_simulated_flag(pydicom_item: Mapping[str, Any],
                                      bx_ref: str,
                                      *,
                                      simulated: bool) -> int:
    return sum(
        1
        for specific_structure in pydicom_item.get(bx_ref, ())
        if bool(specific_structure.get("Simulated bool")) is simulated
    )


def _require_parallel_pool(scientific_config: PatientRunnerScientificConfig,
                           stage_name: PatientStageName) -> Any:
    parallel_pool = scientific_config.resources.parallel_pool
    if parallel_pool is None:
        raise ValueError(f"{stage_name.value} requires scientific_config.resources.parallel_pool")
    return parallel_pool


def _resolve_optional_mc_info_int(master_structure_info_dict: Mapping[str, Any],
                                  explicit_value: int | None,
                                  mc_info_key: str) -> int:
    if explicit_value is not None:
        return int(explicit_value)
    try:
        return int(master_structure_info_dict["Global"]["MC info"][mc_info_key])
    except KeyError as exc:
        raise KeyError(
            "master_structure_info_dict['Global']['MC info']"
            f"[{mc_info_key!r}] is required for patient scientific MC prep"
        ) from exc


def _resolve_mc_prep_max_simulations(master_structure_info_dict: Mapping[str, Any],
                                     stage_config: Any) -> int | None:
    if stage_config.max_simulations is not None:
        return int(stage_config.max_simulations)
    if stage_config.run_biopsy_self_transforms:
        return _resolve_optional_mc_info_int(master_structure_info_dict, None, "Max of num MC simulations")
    if stage_config.run_transform_generation and stage_config.num_generated_transform_samples is None:
        return _resolve_optional_mc_info_int(master_structure_info_dict, None, "Max of num MC simulations")
    return None


def _merge_mc_performed_flags(master_structure_info_dict: Mapping[str, Any],
                              performed_flags: Mapping[str, Any]) -> None:
    global_info = master_structure_info_dict.get("Global")
    if not isinstance(global_info, dict):
        return
    mc_info = global_info.get("MC info")
    if not isinstance(mc_info, dict):
        return
    for key, value in performed_flags.items():
        mc_info[str(key)] = value


def _prefix_mapping(values: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    prefixed_values: dict[str, Any] = {}
    for key, value in dict(values or {}).items():
        if _json_safe_scalar(value):
            prefixed_values[f"{prefix}{key}"] = value
    return prefixed_values


def _json_safe_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


class _NullImportantInfo:
    def add_text_line(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
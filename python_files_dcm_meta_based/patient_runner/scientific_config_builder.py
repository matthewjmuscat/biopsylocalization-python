"""Bridge from root pipeline config to patient-runner scientific config."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from biopsy_optimizer.v1.per_patient import OptimizerV1LegacyConfig
from biopsy_optimizer.v2.per_patient import OptimizerV2LiveConfig
from mc.simulation.per_patient import MCContainmentSimulationConfig
from mc.simulation.per_patient import MCConvexSimulationConfig
from mc.simulation.per_patient import MCDoseSimulationConfig
from mc.simulation.per_patient import MCMRSimulationConfig
from mc.simulation.per_patient import MCReferenceKeys
from mc.simulation.per_patient import MCSimulationRuntimeConfig
from preprocessing.dose_grid_processing import DoseGridProcessingConfig
from preprocessing.mr_adc_grid_processing import MRADCGridProcessingConfig
from preprocessing.transform_bank import OPTIMIZER_V2_TRANSFORM_SAMPLE_COUNT_KEY
from preprocessing.transform_bank import STOCHASTIC_TARGETING_TRANSFORM_SAMPLE_COUNT_KEY
from preprocessing.transform_bank import resolve_required_generated_transform_samples

from .scientific_config import PatientAnatomicalPreprocessingScientificConfig
from .scientific_config import PatientGridPreprocessingScientificConfig
from .scientific_config import PatientGuidanceScientificConfig
from .scientific_config import PatientMCPrepScientificConfig
from .scientific_config import PatientMCSimulationScientificConfig
from .scientific_config import PatientMRADCInputNormalizationStageConfig
from .scientific_config import PatientOptimizationScientificConfig
from .scientific_config import PatientPreprocessingScientificConfig
from .scientific_config import PatientProstateOnlyMRADCStageConfig
from .scientific_config import PatientRawContourPullingStageConfig
from .scientific_config import PatientRealBiopsyProcessingStageConfig
from .scientific_config import PatientRealizedBiopsyTargetingStageConfig
from .scientific_config import PatientRunnerScientificConfig
from .scientific_config import PatientSampledBiopsyProcessingStageConfig
from .scientific_config import PatientSamplingClassificationScientificConfig
from .scientific_config import PatientScientificStageResources
from .scientific_config import PatientSimulatedBiopsyFinalizationStageConfig
from .scientific_config import PatientSimulatedBiopsyPlanningStageConfig
from .scientific_config import PatientSimulatedBiopsyPreparationStageConfig
from .scientific_config import PatientStandardNonBiopsyStructureProcessingStageConfig
from .scientific_config import PatientStructureSelectionStageConfig
from .scientific_config import PatientUncertaintyAttachmentStageConfig
from .scientific_shadow import PatientScientificShadowConfig


@dataclass(frozen=True, slots=True)
class PatientRunnerScientificConfigBuildContext:
    """Runtime/discovered values needed when building patient scientific config."""

    rtstruct_dicom_paths_by_patient_uid: Mapping[str, Any] | None = None
    previous_mr_adc_units: Any = None
    read_uncertainties_dataframe: Any = None
    uncertainty_data_cls: type[Any] | None = None
    dose_views_jsons_paths_list: Sequence[Any] = ()
    containment_views_jsons_paths_list: Sequence[Any] = ()
    mr_views_jsons_paths_list: Sequence[Any] = ()
    parallel_pool: Any = None
    rng: Any = None
    runtime_logger: Any = None
    optimizer_v2_resolved_max_test_structures_per_call: int | None = None
    optimizer_v2_resolved_max_candidates_per_chunk: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rtstruct_dicom_paths_by_patient_uid is not None:
            object.__setattr__(
                self,
                "rtstruct_dicom_paths_by_patient_uid",
                dict(self.rtstruct_dicom_paths_by_patient_uid),
            )
        for field_name in (
            "dose_views_jsons_paths_list",
            "containment_views_jsons_paths_list",
            "mr_views_jsons_paths_list",
        ):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name) or ()))
        for field_name in (
            "optimizer_v2_resolved_max_test_structures_per_call",
            "optimizer_v2_resolved_max_candidates_per_chunk",
        ):
            field_value = getattr(self, field_name)
            if field_value is not None:
                resolved_value = int(field_value)
                if resolved_value < 1:
                    raise ValueError(f"{field_name} must be positive when provided")
                object.__setattr__(self, field_name, resolved_value)
        object.__setattr__(self, "metadata", dict(self.metadata))


def build_patient_scientific_shadow_config(
    pipeline_config: Any,
    context: PatientRunnerScientificConfigBuildContext | None = None,
    **shadow_config_kwargs: Any,
) -> PatientScientificShadowConfig:
    """Build the scientific-shadow wrapper from root config and runtime context."""
    return PatientScientificShadowConfig(
        scientific_config=build_patient_runner_scientific_config(pipeline_config, context),
        **shadow_config_kwargs,
    )


def build_patient_runner_scientific_config(
    pipeline_config: Any,
    context: PatientRunnerScientificConfigBuildContext | None = None,
) -> PatientRunnerScientificConfig:
    """Build the executable patient-runner scientific config from PipelineConfig."""
    context = PatientRunnerScientificConfigBuildContext() if context is None else context
    registry = pipeline_config.structure_registry
    _require_structure_registry(registry)

    return PatientRunnerScientificConfig(
        resources=PatientScientificStageResources(
            parallel_pool=context.parallel_pool,
            rng=context.rng,
            runtime_logger=context.runtime_logger,
            metadata={"source": "PatientRunnerScientificConfigBuildContext", **context.metadata},
        ),
        grid_preprocessing=_build_grid_preprocessing_config(pipeline_config, context),
        anatomical_preprocessing=_build_anatomical_preprocessing_config(pipeline_config, context),
        preprocessing=_build_preprocessing_config(pipeline_config, context),
        mc_prep=_build_mc_prep_config(pipeline_config),
        mc_simulation=_build_mc_simulation_config(pipeline_config, context),
        optimization=_build_optimization_config(pipeline_config, context),
        simulated_biopsy_finalization=_build_simulated_biopsy_finalization_config(pipeline_config),
        sampling_classification=_build_sampling_classification_config(pipeline_config),
        guidance=_build_guidance_config(pipeline_config),
        metadata={"source": "PipelineConfig"},
    )


def _build_grid_preprocessing_config(
    pipeline_config: Any,
    context: PatientRunnerScientificConfigBuildContext,
) -> PatientGridPreprocessingScientificConfig:
    refs = pipeline_config.legacy_refs
    replay = pipeline_config.replay
    grid = pipeline_config.grid_preprocessing
    return PatientGridPreprocessingScientificConfig(
        dose_grid_config=DoseGridProcessingConfig(
            dose_ref=refs.dose_ref,
            plan_ref=refs.plan_ref,
            lower_bound_dose_value=replay.lower_bound_dose_value,
            lower_bound_dose_gradient_value=replay.lower_bound_dose_gradient_value,
            show_3d_dose_renderings=grid.show_3d_dose_renderings,
            show_3d_dose_renderings_thresholded=grid.show_3d_dose_renderings_thresholded,
        ),
        mr_adc_input_normalization=PatientMRADCInputNormalizationStageConfig(
            mr_adc_ref=refs.mr_adc_ref,
            previous_mr_adc_units=context.previous_mr_adc_units,
        ),
        mr_adc_grid_config=MRADCGridProcessingConfig(
            mr_adc_ref=refs.mr_adc_ref,
            color_flattening_deg_mr=replay.color_flattening_deg_mr,
            lower_bound_mr_adc_value=replay.lower_bound_mr_adc_value,
            upper_bound_mr_adc_value=replay.upper_bound_mr_adc_value,
            show_3d_mr_adc_renderings=grid.show_3d_mr_adc_renderings,
            show_3d_mr_adc_renderings_thresholded=grid.show_3d_mr_adc_renderings_thresholded,
        ),
    )


def _build_anatomical_preprocessing_config(
    pipeline_config: Any,
    context: PatientRunnerScientificConfigBuildContext,
) -> PatientAnatomicalPreprocessingScientificConfig:
    refs = pipeline_config.legacy_refs
    registry = pipeline_config.structure_registry
    raw_contour_pulling = None
    if context.rtstruct_dicom_paths_by_patient_uid:
        raw_contour_pulling = PatientRawContourPullingStageConfig(
            rtstruct_dicom_paths_by_patient_uid=context.rtstruct_dicom_paths_by_patient_uid,
            structs_referenced_list_generalized=registry.structs_referenced_list_generalized,
        )

    return PatientAnatomicalPreprocessingScientificConfig(
        raw_contour_pulling=raw_contour_pulling,
        structure_selection=PatientStructureSelectionStageConfig(
            structs_referenced_dict=registry.structs_referenced_dict,
            structs_referenced_list_generalized=registry.structs_referenced_list_generalized,
            structs_referenced_list_generalized_unique_structs=(
                registry.structs_referenced_list_generalized_unique_structs
            ),
        ),
        standard_non_biopsy_structure_processing=PatientStandardNonBiopsyStructureProcessingStageConfig(
            oar_ref=refs.oar_ref,
            rectum_ref_key=refs.rectum_ref_key,
            urethra_ref_key=refs.urethra_ref_key,
            dil_ref=refs.dil_ref,
            structs_referenced_dict=registry.structs_referenced_dict,
            preprocessing_config=pipeline_config.preprocessing.build_non_biopsy_structure_preprocessing_config(
                all_ref_key=refs.all_ref_key,
                oar_ref=refs.oar_ref,
                dil_ref=refs.dil_ref,
                mr_adc_ref=refs.mr_adc_ref,
            ),
        ),
        prostate_only_mr_adc=PatientProstateOnlyMRADCStageConfig(
            dil_ref=refs.dil_ref,
            mr_adc_ref=refs.mr_adc_ref,
            demonstrate_mr_adc_pcd_containment_correctness_prostate_only_all_other_structures_removed_bool=(
                pipeline_config.preprocessing.demonstrate_mr_adc_pcd_containment_correctness_prostate_only_all_other_structures_removed_bool
            ),
        ),
    )


def _build_preprocessing_config(
    pipeline_config: Any,
    context: PatientRunnerScientificConfigBuildContext,
) -> PatientPreprocessingScientificConfig:
    refs = pipeline_config.legacy_refs
    registry = pipeline_config.structure_registry
    biopsy = pipeline_config.biopsy
    mc_prep = pipeline_config.mc.prep
    uncertainty_attachment = None
    if context.read_uncertainties_dataframe is not None and context.uncertainty_data_cls is not None:
        uncertainty_attachment = PatientUncertaintyAttachmentStageConfig(
            read_uncertainties_dataframe=context.read_uncertainties_dataframe,
            uncertainty_data_cls=context.uncertainty_data_cls,
        )

    return PatientPreprocessingScientificConfig(
        real_biopsy_processing=PatientRealBiopsyProcessingStageConfig.from_preprocessing_config(
            pipeline_config.preprocessing,
            structs_referenced_dict=registry.structs_referenced_dict,
            biopsy_radius=biopsy.geometry.biopsy_radius,
            display_pca_fit_variation_for_biopsies_bool=(
                biopsy.geometry.display_pca_fit_variation_for_biopsies_bool
            ),
        ),
        simulated_biopsy_preparation=PatientSimulatedBiopsyPreparationStageConfig(
            dil_ref=refs.dil_ref,
            simulated_biopsy_length_method=biopsy.simulated.simulated_biopsy_length_method,
            biopsy_needle_compartment_length=mc_prep.biopsy_needle_compartment_length,
        ),
        simulated_biopsy_planning=PatientSimulatedBiopsyPlanningStageConfig(
            bx_sample_pts_lattice_spacing=mc_prep.bx_sample_pts_lattice_spacing,
            simulated_bx_rad=biopsy.geometry.simulated_biopsy_planning_radius_mm,
            centroid_line_vec_sim_list=biopsy.simulated.centroid_line_vec_sim_list,
            centroid_first_pos_sim_list=biopsy.simulated.centroid_first_pos_sim_list,
            num_centroids_for_sim_bxs=biopsy.simulated.num_centroids_for_sim_bxs,
            plot_simulated_cores_immediately=biopsy.simulated.plot_simulated_cores_immediately,
        ),
        realized_biopsy_targeting=PatientRealizedBiopsyTargetingStageConfig(
            oar_ref=refs.oar_ref,
            dil_ref=refs.dil_ref,
        ),
        uncertainty_attachment=uncertainty_attachment,
    )


def _build_mc_prep_config(pipeline_config: Any) -> PatientMCPrepScientificConfig:
    registry = pipeline_config.structure_registry
    mc = pipeline_config.mc
    max_simulations, max_generated_transform_samples = _resolve_explicit_transform_counts(pipeline_config)
    return PatientMCPrepScientificConfig(
        structs_referenced_list=registry.structs_referenced_list,
        simulate_uniform_bx_shifts_due_to_bx_needle_compartment=(
            mc.prep.simulate_uniform_bx_shifts_due_to_bx_needle_compartment
        ),
        biopsy_needle_compartment_length=mc.prep.biopsy_needle_compartment_length,
        run_transform_generation=mc.prep.run_transform_generation,
        run_biopsy_self_transforms=mc.prep.run_biopsy_self_transforms,
        run_relative_structure_transforms=mc.prep.run_relative_structure_transforms,
        num_generated_transform_samples=max_generated_transform_samples,
        max_simulations=max_simulations,
        num_mc_containment_simulations=mc.counts.num_mc_containment_simulations_input,
        inspect_self_biopsy_dilate_bool=mc.debug.inspect_self_biopsy_dilate_bool,
        inspect_self_biopsy_dilate_and_rotate_bool=mc.debug.inspect_self_biopsy_dilate_and_rotate_bool,
        inspect_self_biopsy_dilate_and_rotate_and_translate_bool=(
            mc.debug.inspect_self_biopsy_dilate_and_rotate_and_translate_bool
        ),
        inspect_relative_structure_rotate_and_shift_number=(
            mc.debug.inspect_relative_structure_rotate_and_shift_number
        ),
    )


def _build_mc_simulation_config(
    pipeline_config: Any,
    context: PatientRunnerScientificConfigBuildContext,
) -> PatientMCSimulationScientificConfig:
    mc = pipeline_config.mc
    convex_config = None
    if mc.counts.perform_mc_containment_sim or mc.counts.perform_mc_dose_sim:
        convex_config = MCConvexSimulationConfig(
            keys=_build_mc_reference_keys(pipeline_config),
            runtime=_build_mc_runtime_config(pipeline_config),
            containment=_build_mc_containment_config(pipeline_config, context),
            dose=_build_mc_dose_config(pipeline_config, context),
        )
    mr_config = None
    if mc.counts.perform_mc_mr_sim:
        mr_views_jsons_paths_list = context.mr_views_jsons_paths_list or context.dose_views_jsons_paths_list
        mr_config = MCMRSimulationConfig(
            num_mr_calc_NN=mc.simulation.num_mr_calc_nn,
            mr_views_jsons_paths_list=mr_views_jsons_paths_list,
            show_NN_mr_adc_demonstration_plots=mc.debug.show_nn_mr_adc_demonstration_plots,
            show_NN_mr_adc_demonstration_plots_all_trials_at_once=(
                mc.debug.show_nn_mr_adc_demonstration_plots_all_trials_at_once
            ),
            perform_mc_mr_sim=mc.counts.perform_mc_mr_sim,
            idw_power=mc.simulation.idw_power,
            raw_data_mc_mr_dump_bool=mc.output_dumps.raw_data_mc_mr_dump_bool,
        )
    return PatientMCSimulationScientificConfig(
        convex_config=convex_config,
        mr_config=mr_config,
        mr_adc_ref=pipeline_config.legacy_refs.mr_adc_ref,
    )


def _build_optimization_config(
    pipeline_config: Any,
    context: PatientRunnerScientificConfigBuildContext,
) -> PatientOptimizationScientificConfig:
    refs = pipeline_config.legacy_refs
    registry = pipeline_config.structure_registry
    preprocessing = pipeline_config.preprocessing
    optimizer = pipeline_config.optimizer
    optimizer_v1 = optimizer.optimizer_v1
    optimizer_v2 = optimizer.optimizer_v2
    optimizer_v2_rendering = optimizer_v2.rendering
    optimizer_v2_plotly = optimizer_v2_rendering.plotly_export
    return PatientOptimizationScientificConfig(
        optimizer_v1_config=OptimizerV1LegacyConfig(
            structs_referenced_dict=registry.structs_referenced_dict,
            bx_ref=refs.bx_ref,
            dil_ref=refs.dil_ref,
            oar_ref=refs.oar_ref,
            all_ref_key=refs.all_ref_key,
            voxel_size_for_dil_optimizer_grid=optimizer_v1.voxel_size_for_dil_optimizer_grid,
            optimal_normal_dist_option=optimizer_v1.optimal_normal_dist_option,
            bias_LR_multiplier=optimizer_v1.bias_LR_multiplier,
            bias_AP_multiplier=optimizer_v1.bias_AP_multiplier,
            bias_SI_multiplier=optimizer_v1.bias_SI_multiplier,
            num_normal_dist_points_for_biopsy_optimizer=(
                optimizer_v1.num_normal_dist_points_for_biopsy_optimizer
            ),
            normal_dist_sigma_factor_biopsy_optimizer=(
                optimizer_v1.normal_dist_sigma_factor_biopsy_optimizer
            ),
            plot_each_normal_dist_containment_result_bool=(
                optimizer_v1.plot_each_normal_dist_containment_result_bool
            ),
            plot_optimization_point_lattice_bool=optimizer_v1.plot_optimization_point_lattice_bool,
            show_optimization_point_bool=optimizer_v1.show_optimization_point_bool,
            cupy_array_upper_limit_NxN_size_input=optimizer_v1.cupy_array_upper_limit_nxn_size_input,
            numpy_array_upper_limit_NxN_size_input=optimizer_v1.numpy_array_upper_limit_nxn_size_input,
            nearest_zslice_vals_and_indices_cupy_generic_max_size=(
                optimizer_v1.nearest_zslice_vals_and_indices_cupy_generic_max_size
            ),
            nearest_zslice_vals_and_indices_numpy_generic_max_size=(
                optimizer_v1.nearest_zslice_vals_and_indices_numpy_generic_max_size
            ),
            constant_z_slice_polygons_handler_option=optimizer_v1.constant_z_slice_polygons_handler_option,
            remove_consecutive_duplicate_points_in_polygons=(
                optimizer_v1.remove_consecutive_duplicate_points_in_polygons
            ),
            include_edges_in_log_files=optimizer_v1.include_edges_in_log_files,
            custom_cuda_kernel_type=optimizer_v1.custom_cuda_kernel_type,
            demonstrate_dil_optimization_points_inside_correctness_bool_1=(
                optimizer_v1.demonstrate_dil_optimization_points_inside_correctness_bool_1
            ),
            demonstrate_dil_optimization_points_inside_correctness_bool_2=(
                optimizer_v1.demonstrate_dil_optimization_points_inside_correctness_bool_2
            ),
            demonstrate_dil_optimization_points_inside_correctness_num_3=(
                optimizer_v1.demonstrate_dil_optimization_points_inside_correctness_num_3
            ),
            generate_cuda_log_files_biopsy_optimizer=optimizer_v1.generate_cuda_log_files_biopsy_optimizer,
            display_optimization_contour_plots_bool=optimizer_v1.display_optimization_contour_plots_bool,
        ),
        optimizer_v2_config=OptimizerV2LiveConfig(
            structs_referenced_dict=registry.structs_referenced_dict,
            bx_ref=refs.bx_ref,
            dil_ref=refs.dil_ref,
            all_ref_key=refs.all_ref_key,
            optimizer_simulated_type=pipeline_config.biopsy.simulated.optimizer_simulated_type,
            search_config=optimizer_v2.search_config,
            constant_z_slice_polygons_handler_option=preprocessing.constant_z_slice_polygons_handler_option,
            remove_consecutive_duplicate_points_in_polygons=(
                preprocessing.remove_consecutive_duplicate_points_in_polygons
            ),
            include_edges_in_log=preprocessing.include_edges_in_log_files,
            kernel_type=preprocessing.custom_cuda_kernel_type,
            max_candidates_per_chunk=optimizer_v2.capacity.max_candidates_per_chunk,
            max_test_structures_per_call=optimizer_v2.capacity.max_test_structures_per_call,
            fallback_max_test_structures_per_call=optimizer_v2.capacity.fallback_max_test_structures_per_call,
            auto_calibrate_max_test_structures_per_call=(
                optimizer_v2.capacity.auto_calibrate_max_test_structures_per_call
            ),
            verify_calibrated_max_test_structures_per_call=(
                optimizer_v2.capacity.verify_calibrated_max_test_structures_per_call
            ),
            validate_nearest_z_helper_against_ver5=(
                optimizer_v2.diagnostics.validate_nearest_z_helper_against_ver5
            ),
            downstream_comparable_trial_count=(
                int(pipeline_config.mc.counts.num_mc_containment_simulations_input)
                if pipeline_config.mc.counts.num_mc_containment_simulations_input > 0
                else None
            ),
            benchmark_isolated_winner_validation_bool=(
                optimizer_v2.diagnostics.benchmark_isolated_winner_validation_bool
            ),
            render_stage_boundary_candidate_clouds_bool=(
                optimizer_v2_rendering.render_stage_boundary_candidate_clouds_bool
            ),
            render_stage_names_to_render=optimizer_v2_rendering.render_stage_names_to_render,
            render_backend=optimizer_v2_rendering.render_backend,
            render_layer_style_by_name=optimizer_v2_rendering.render_layer_style_by_name,
            render_plotly_export_bool=optimizer_v2_plotly.enabled,
            render_plotly_export_formats=optimizer_v2_plotly.formats,
            render_plotly_export_width=optimizer_v2_plotly.width,
            render_plotly_export_height=optimizer_v2_plotly.height,
            render_plotly_export_scale=optimizer_v2_plotly.scale,
            render_plotly_export_camera_eye=optimizer_v2_plotly.camera_eye,
            render_plotly_export_camera_center=optimizer_v2_plotly.camera_center,
            render_plotly_export_camera_up=optimizer_v2_plotly.camera_up,
            render_dialog_timeout_seconds=optimizer_v2_rendering.render_dialog_timeout_seconds,
            render_dialog_timeout_extend_seconds=optimizer_v2_rendering.render_dialog_timeout_extend_seconds,
            render_winner_containment_debug_bool=optimizer_v2_rendering.render_winner_containment_debug_bool,
            render_winner_containment_backend=optimizer_v2_rendering.render_winner_containment_backend,
            render_include_target_points_bool=optimizer_v2_rendering.render_include_target_points_bool,
            render_patient_whitelist=optimizer_v2_rendering.render_patient_whitelist,
            render_roi_whitelist=optimizer_v2_rendering.render_roi_whitelist,
            render_include_planned_sampled_points_bool=(
                optimizer_v2_rendering.render_include_planned_sampled_points_bool
            ),
            render_include_planned_core_structure_bool=(
                optimizer_v2_rendering.render_include_planned_core_structure_bool
            ),
            render_include_planned_centroid_line_bool=(
                optimizer_v2_rendering.render_include_planned_centroid_line_bool
            ),
            render_include_target_surface_bool=optimizer_v2_rendering.render_include_target_surface_bool,
            render_include_selected_anatomy_bool=optimizer_v2_rendering.render_include_selected_anatomy_bool,
            oar_ref=refs.oar_ref,
            rectum_ref=refs.rectum_ref_key,
            urethra_ref=refs.urethra_ref_key,
        ),
        optimizer_v2_resolved_max_test_structures_per_call=(
            context.optimizer_v2_resolved_max_test_structures_per_call
        ),
        optimizer_v2_resolved_max_candidates_per_chunk=context.optimizer_v2_resolved_max_candidates_per_chunk,
    )


def _build_simulated_biopsy_finalization_config(pipeline_config: Any) -> PatientSimulatedBiopsyFinalizationStageConfig:
    return PatientSimulatedBiopsyFinalizationStageConfig.from_preprocessing_config(
        pipeline_config.preprocessing,
        structs_referenced_dict=pipeline_config.structure_registry.structs_referenced_dict,
        biopsy_radius=pipeline_config.biopsy.geometry.biopsy_radius,
    )


def _build_sampling_classification_config(pipeline_config: Any) -> PatientSamplingClassificationScientificConfig:
    return PatientSamplingClassificationScientificConfig(
        sampled_biopsy_processing=PatientSampledBiopsyProcessingStageConfig(
            bx_sample_pts_lattice_spacing=pipeline_config.mc.prep.bx_sample_pts_lattice_spacing,
            show_reconstructed_biopsy_in_biopsy_coord_sys_tr_and_rot=(
                pipeline_config.biopsy.sampling.show_reconstructed_biopsy_in_biopsy_coord_sys_tr_and_rot
            ),
        ),
    )


def _build_guidance_config(pipeline_config: Any) -> PatientGuidanceScientificConfig:
    refs = pipeline_config.legacy_refs
    return PatientGuidanceScientificConfig(
        dil_ref=refs.dil_ref,
        oar_ref=refs.oar_ref,
        rectum_ref=refs.rectum_ref_key,
        biopsy_fire_travel_distances=pipeline_config.biopsy.geometry.biopsy_fire_travel_distances,
        biopsy_needle_compartment_length=pipeline_config.mc.prep.biopsy_needle_compartment_length,
        interp_inter_slice_dist=pipeline_config.preprocessing.interp_inter_slice_dist,
        interp_intra_slice_dist=pipeline_config.preprocessing.interp_intra_slice_dist,
        radius_for_normals_estimation=pipeline_config.preprocessing.radius_for_normals_estimation,
        max_nn_for_normals_estimation=pipeline_config.preprocessing.max_nn_for_normals_estimation,
        biopsy_needle_tip_length=pipeline_config.biopsy.geometry.biopsy_needle_tip_length,
        planning_config=pipeline_config.guidance_maps.planning_config,
    )


def _build_mc_reference_keys(pipeline_config: Any) -> MCReferenceKeys:
    refs = pipeline_config.legacy_refs
    registry = pipeline_config.structure_registry
    return MCReferenceKeys(
        structs_referenced_list=registry.structs_referenced_list,
        structs_referenced_dict=registry.structs_referenced_dict,
        bx_ref=refs.bx_ref,
        oar_ref=refs.oar_ref,
        dil_ref=refs.dil_ref,
        rectum_ref=refs.rectum_ref_key,
        urethra_ref=refs.urethra_ref_key,
        dose_ref=refs.dose_ref,
        plan_ref=refs.plan_ref,
        all_ref_key=refs.all_ref_key,
    )


def _build_mc_runtime_config(pipeline_config: Any) -> MCSimulationRuntimeConfig:
    mc = pipeline_config.mc
    preprocessing = pipeline_config.preprocessing
    return MCSimulationRuntimeConfig(
        biopsy_needle_compartment_length=mc.prep.biopsy_needle_compartment_length,
        simulate_uniform_bx_shifts_due_to_bx_needle_compartment=(
            mc.prep.simulate_uniform_bx_shifts_due_to_bx_needle_compartment
        ),
        plot_uniform_shifts_to_check_plotly=mc.debug.plot_uniform_shifts_to_check_plotly,
        plot_translation_vectors_pointclouds=mc.debug.plot_translation_vectors_pointclouds,
        plot_shifted_biopsies=mc.debug.plot_shifted_biopsies,
        spinner_type=pipeline_config.ui.spinner_type,
        cupy_array_upper_limit_NxN_size_input=preprocessing.cupy_array_upper_limit_nxn_size_input,
        nearest_zslice_vals_and_indices_cupy_generic_max_size=(
            preprocessing.nearest_zslice_vals_and_indices_cupy_generic_max_size
        ),
        custom_cuda_kernel_type=preprocessing.custom_cuda_kernel_type,
    )


def _build_mc_containment_config(
    pipeline_config: Any,
    context: PatientRunnerScientificConfigBuildContext,
) -> MCContainmentSimulationConfig:
    mc = pipeline_config.mc
    preprocessing = pipeline_config.preprocessing
    return MCContainmentSimulationConfig(
        containment_views_jsons_paths_list=context.containment_views_jsons_paths_list,
        show_num_containment_demonstration_plots=mc.debug.show_num_containment_demonstration_plots,
        containment_results_structure_types_to_show_per_trial=(
            mc.visualization.containment_results_structure_types_to_show_per_trial
        ),
        show_num_nearest_neighbour_surface_boundary_demonstration=(
            mc.debug.show_num_nearest_neighbour_surface_boundary_demonstration
        ),
        show_num_relative_structure_centroid_demonstration=(
            mc.debug.show_num_relative_structure_centroid_demonstration
        ),
        plot_cupy_containment_distribution_results=mc.debug.plot_cupy_containment_distribution_results,
        structure_miss_probability_roi=mc.tissue.structure_miss_probability_roi,
        cancer_tissue_label=mc.tissue.cancer_tissue_label,
        default_exterior_tissue=mc.tissue.default_exterior_tissue,
        miss_structure_complement_label=mc.tissue.miss_structure_complement_label,
        tissue_length_above_probability_threshold_list=(
            mc.simulation.tissue_length_above_probability_threshold_list
        ),
        n_bootstraps_for_tissue_length_above_threshold=(
            mc.simulation.n_bootstraps_for_tissue_length_above_threshold
        ),
        perform_mc_containment_sim=mc.counts.perform_mc_containment_sim,
        raw_data_mc_containment_dump_bool=mc.output_dumps.raw_data_mc_containment_dump_bool,
        keep_light_containment_and_distances_to_relative_structures_dataframe_bool=(
            mc.output_dumps.keep_light_containment_and_distances_to_relative_structures_dataframe_bool
        ),
        show_non_bx_relative_structure_z_dilation_bool=(
            mc.visualization.show_non_bx_relative_structure_z_dilation_bool
        ),
        show_non_bx_relative_structure_xy_dilation_bool=(
            mc.visualization.show_non_bx_relative_structure_xy_dilation_bool
        ),
        generate_cuda_log_files_MC_containment_sim=mc.debug.generate_cuda_log_files_mc_containment_sim,
        constant_z_slice_polygons_handler_option=preprocessing.constant_z_slice_polygons_handler_option,
        remove_consecutive_duplicate_points_in_polygons=(
            preprocessing.remove_consecutive_duplicate_points_in_polygons
        ),
        interp_dist_caps=preprocessing.interp_dist_caps,
        cuml_NN_algo=mc.simulation.cuml_nn_algo,
        check_if_end_caps_filled_proper_NN_num=mc.visualization.check_if_end_caps_filled_proper_nn_num,
        nn_search_end_cap_grid_factor=mc.simulation.nn_search_end_cap_grid_factor,
        tissue_volume_operator_dictionary=mc.tissue.tissue_volume_operator_dictionary,
    )


def _build_mc_dose_config(
    pipeline_config: Any,
    context: PatientRunnerScientificConfigBuildContext,
) -> MCDoseSimulationConfig:
    mc = pipeline_config.mc
    return MCDoseSimulationConfig(
        biopsy_z_voxel_length=mc.simulation.biopsy_z_voxel_length,
        num_dose_calc_NN=mc.simulation.num_dose_calc_nn,
        num_dose_NN_to_show_for_animation_plotting=(
            mc.visualization.num_dose_nn_to_show_for_animation_plotting
        ),
        dose_views_jsons_paths_list=context.dose_views_jsons_paths_list,
        show_NN_dose_demonstration_plots=mc.debug.show_nn_dose_demonstration_plots,
        show_NN_dose_demonstration_plots_all_trials_at_once=(
            mc.debug.show_nn_dose_demonstration_plots_all_trials_at_once
        ),
        differential_dvh_resolution=mc.simulation.differential_dvh_resolution,
        cumulative_dvh_resolution=mc.simulation.cumulative_dvh_resolution,
        v_percent_DVH_to_calc_list=mc.simulation.v_percent_dvh_to_calc_list,
        volume_DVH_quantiles_to_calculate=mc.simulation.volume_dvh_quantiles_to_calculate,
        perform_mc_dose_sim=mc.counts.perform_mc_dose_sim,
        idw_power=mc.simulation.idw_power,
        raw_data_mc_dosimetry_dump_bool=mc.output_dumps.raw_data_mc_dosimetry_dump_bool,
    )


def _resolve_explicit_transform_counts(pipeline_config: Any) -> tuple[int, int]:
    optimizer_v2_transform_count = pipeline_config.optimizer.optimizer_v2.search_config.resolve_required_transform_bank_size()
    transform_info = {
        OPTIMIZER_V2_TRANSFORM_SAMPLE_COUNT_KEY: optimizer_v2_transform_count,
        STOCHASTIC_TARGETING_TRANSFORM_SAMPLE_COUNT_KEY: (
            pipeline_config.optimizer.num_stochastic_targeting_transform_samples_input
        ),
    }
    return resolve_required_generated_transform_samples(
        transform_info,
        pipeline_config.mc.counts.num_mc_containment_simulations_input,
        pipeline_config.mc.counts.num_mc_dose_simulations_input,
        pipeline_config.mc.counts.num_mc_mr_simulations_input,
    )


def _require_structure_registry(registry: Any) -> None:
    missing_fields = []
    if not registry.structs_referenced_dict:
        missing_fields.append("structs_referenced_dict")
    if not registry.structs_referenced_list:
        missing_fields.append("structs_referenced_list")
    if not registry.structs_referenced_list_generalized:
        missing_fields.append("structs_referenced_list_generalized")
    if not registry.structs_referenced_list_generalized_unique_structs:
        missing_fields.append("structs_referenced_list_generalized_unique_structs")
    if missing_fields:
        raise ValueError(
            "PipelineConfig.structure_registry is missing patient-runner inputs: "
            + ", ".join(missing_fields)
        )

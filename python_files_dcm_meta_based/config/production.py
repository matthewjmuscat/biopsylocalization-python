"""Production choices and assembly for the existing typed PipelineConfig tree.

Dataclass defaults remain authoritative wherever they exactly match production.
This module owns only the remaining production choices and assembly relationships;
it reads no files, environment, legacy main, patient state, or execution modules.
JSON snapshots are evidence of the result, never construction input here.
"""

from __future__ import annotations

from dataclasses import replace

from biopsy_optimizer.v2.config import build_optimizer_v2_adaptive_block_search_config
from guidance_maps.config import GuidanceMapPlanningConfig, GuidanceMapRenderConfig

from .bootstrap import (
    PatientBootstrapConfig,
    SimulatedBiopsyBootstrapPolicy,
    StructureContourPolicy,
    StructureDataRemovalPolicy,
)
from .pipeline import (
    ArtifactConfig,
    BiopsyRuntimeConfig,
    GuidanceMapConfig,
    LegacyReferenceConfig,
    MCCountsConfig,
    MCPrepConfig,
    MCSimulationCoreConfig,
    MonteCarloConfig,
    OptimizerRuntimeConfig,
    OptimizerV1RuntimeConfig,
    OptimizerV2DiagnosticsConfig,
    OptimizerV2RenderConfig,
    OptimizerV2RuntimeConfig,
    PatientRunnerValidationHookConfig,
    PipelineConfig,
    PreprocessingConfig,
    PreprocessingGeometryConfig,
    PreprocessingInterpolationConfig,
    PreprocessingKernelExecutionConfig,
    RandomSeedConfig,
    RuntimeReplayConfig,
    RuntimeUIConfig,
    StructureRegistryConfig,
)

# Legacy simulation-family labels also consumed by main's presentation controls.
CENTROID_DIL_SIMULATED_TYPE = "Centroid DIL"
OPTIMAL_DIL_SIMULATED_TYPE = "Optimal DIL"


def build_production_pipeline_config() -> PipelineConfig:
    """Return a fresh pure-data config with the established production choices.

    Callers can use dataclasses.replace on this tree and its existing domain
    slices for explicit session/scientific overrides, then pass the resolved
    PipelineConfig to main or snapshot it for standalone workers. Domain
    constructors retain their validation; no loose override dictionary or second
    config model is introduced. Paths and derived patient state are not inferred.

    Some explicit integer values intentionally differ from generic float defaults:
    changing 1 to 1.0 would change the canonical scientific snapshot identity.
    """
    refs = LegacyReferenceConfig()
    contours = StructureContourPolicy()
    biopsy = BiopsyRuntimeConfig()
    # Retain production's shared real/planned radius and dose/MR trial budgets.
    biopsy = replace(biopsy, geometry=replace(
        biopsy.geometry,
        simulated_biopsy_planning_radius_mm=biopsy.geometry.biopsy_radius,
    ))
    counts = MCCountsConfig()
    counts = replace(counts, num_mc_mr_simulations_input=counts.num_mc_dose_simulations_input)
    optimizer_v1 = OptimizerV1RuntimeConfig()
    search = build_optimizer_v2_adaptive_block_search_config(
        initial_trial_prefix=16,
        trial_block_size=16,
        max_total_trials=256,
    )
    return PipelineConfig(
        ui=RuntimeUIConfig(),
        artifacts=ArtifactConfig(),
        legacy_refs=refs,
        biopsy=biopsy,
        bootstrap=PatientBootstrapConfig(
            contours=contours,
            removals=_build_structure_removals(),
            simulated_biopsies=SimulatedBiopsyBootstrapPolicy(
                locations={
                    CENTROID_DIL_SIMULATED_TYPE: {
                        "Create": True,
                        "Relative to struct type": refs.dil_ref,
                        "Transport family": "centroid",
                        "Identifier string": "sim_centroid_dil",
                    },
                    OPTIMAL_DIL_SIMULATED_TYPE: {
                        "Create": True,
                        "Relative to struct type": refs.dil_ref,
                        "Transport family": "optimal",
                        "Identifier string": "sim_optimal_dil",
                    },
                    biopsy.simulated.optimizer_simulated_type: {
                        "Create": True,
                        "Relative to struct type": refs.dil_ref,
                        "Transport family": "identity",
                        "Identifier string": "sim_target_dil_v2",
                    },
                },
            ),
        ),
        structure_registry=_build_structure_registry(refs, contours, biopsy),
        preprocessing=PreprocessingConfig(
            interpolation=PreprocessingInterpolationConfig(
                interp_inter_slice_dist=0.5,
                interp_intra_slice_dist=0.5,
                interp_dist_caps=0.25,
            ),
            geometry=PreprocessingGeometryConfig(
                radius_for_normals_estimation=1,
                max_nn_for_normals_estimation=30,
                voxel_size_for_structure_volume_calc_non_bx=1,
                voxel_size_for_structure_dimension_calc=0.1,
                factor_for_voxel_size=100,
            ),
            # These kernel choices were shared main locals. Reuse their existing
            # typed defaults until a separate shared-kernel contract is warranted.
            kernel_execution=PreprocessingKernelExecutionConfig(
                cupy_array_upper_limit_nxn_size_input=optimizer_v1.cupy_array_upper_limit_nxn_size_input,
                nearest_zslice_vals_and_indices_cupy_generic_max_size=(
                    optimizer_v1.nearest_zslice_vals_and_indices_cupy_generic_max_size
                ),
                constant_z_slice_polygons_handler_option=optimizer_v1.constant_z_slice_polygons_handler_option,
                remove_consecutive_duplicate_points_in_polygons=optimizer_v1.remove_consecutive_duplicate_points_in_polygons,
                include_edges_in_log_files=optimizer_v1.include_edges_in_log_files,
                custom_cuda_kernel_type=optimizer_v1.custom_cuda_kernel_type,
            ),
        ),
        replay=RuntimeReplayConfig(
            lower_bound_dose_value=None,
            lower_bound_dose_gradient_value=0,
            lower_bound_mr_adc_value=500,
            upper_bound_mr_adc_value=900,
            color_flattening_deg_mr=1,
        ),
        guidance_maps=GuidanceMapConfig(
            planning_config=GuidanceMapPlanningConfig(
                candidate_holes_k=3,
                candidate_axis_line_length_mm=1000,
            ),
            render_config=GuidanceMapRenderConfig(
                image_scale=1,
                show_euler_annotation_box=False,
                candidate_plot_rank="all",
            ),
        ),
        optimizer=OptimizerRuntimeConfig(
            optimizer_v1=optimizer_v1,
            optimizer_v2_search_config=search,
            optimizer_v2=OptimizerV2RuntimeConfig(
                search_config=search,
                diagnostics=OptimizerV2DiagnosticsConfig(
                    validate_nearest_z_helper_against_ver5=False,
                    benchmark_isolated_winner_validation_bool=False,
                ),
                rendering=OptimizerV2RenderConfig(
                    render_layer_style_by_name=_build_optimizer_render_styles(),
                ),
            ),
        ),
        random_seeds=RandomSeedConfig(
            transform_generation_random_seed=51,
            optimizer_v1_random_seed=51,
        ),
        patient_runner_validation=PatientRunnerValidationHookConfig(
            write_outputs=True,
            write_assembled_tables=True,
            scientific_shadow_pathway_name="full_current_pipeline_shadow",
            scientific_shadow_include_artifact_writing=True,
        ),
        mc=MonteCarloConfig(
            counts=counts,
            prep=MCPrepConfig(
                biopsy_needle_compartment_length=19,
                bx_sample_pts_lattice_spacing=1,
            ),
            simulation=MCSimulationCoreConfig(biopsy_z_voxel_length=1, idw_power=1),
        ),
    )


def _build_structure_removals() -> StructureDataRemovalPolicy:
    """Retain the production cohort's explicit ROI exclusion policy."""
    return StructureDataRemovalPolicy(
        biopsy={
            '189 (F2)': ('Bx_Tr LM1 blood',),
            '192 (F2)': ('Bx_trk LM blood',),
            '200 (F1)': ('Bx_LTapex_needle',),
            '201 (F2)': ('Bx_LTpost_air',),
            '203 (F1)': ('Bx_LTapex_air',),
        },
        prostate={
            '194 (F1)': ('Prostate pre',),
            '194 (F2)': ('Prostate_pre',),
            '195 (F1)': ('Prostate biop',),
            '195 (F2)': ('Prostate pre',),
            '196 (F1)': ('Prostate_pre',),
            '196 (F2)': ('Prostate_pre',),
            '199 (F1)': ('Prostate_pre',),
            '199 (F2)': ('Prostate_pre',),
            '198 (F2)': ('Prostate_pre', 'Prostate_biop'),
            '200 (F1)': ('Prostate_pre',),
            '200 (F2)': ('Prostate_pre',),
            '201 (F1)': ('Prostate_pre',),
            '201 (F2)': ('Prostate pre',),
            '203 (F1)': ('Prostate_pre',),
            '203 (F2)': ('Prostate-pre',),
        },
        dil={
            '194 (F1)': ('DIL 2',),
            '194 (F2)': ('DIL 2',),
            '195 (F2)': ('DIL 1 MIN', 'DIL 2 MIN'),
            '196 (F1)': ('DIL 1 MIN',),
            '196 (F2)': ('DIL 1 MIN',),
            '199 (F1)': ('DIL 1 MIN', 'DIL 2 MIN'),
            '199 (F2)': ('DIL 1 MIN', 'DIL 2 MIN'),
        },
        urethra={
            '194 (F1)': ('Opti Urethra',),
            '194 (F2)': ('Opti Urethra',),
            '195 (F1)': ('Opti Urethra', 'Urethra_pre'),
            '195 (F2)': ('Opti Urethra', 'Urethra_pre'),
            '196 (F1)': ('Opti Urethra',),
            '196 (F2)': ('Opti Urethra',),
            '199 (F1)': ('Opti Urethra',),
            '199 (F2)': ('Opti Urethra',),
            '198 (F2)': ('Opti Urethra',),
            '200 (F1)': ('Opti Urethra',),
            '200 (F2)': ('Opti Urethra',),
            '201 (F1)': ('Opti Urethra',),
            '201 (F2)': ('Opti Urethra',),
            '203 (F1)': ('Opti Urethra',),
            '203 (F2)': ('Opti Urethra',),
        },
    )


def _build_structure_registry(
    refs: LegacyReferenceConfig,
    contours: StructureContourPolicy,
    biopsy: BiopsyRuntimeConfig,
) -> StructureRegistryConfig:
    """Build fresh compatibility records in their established scientific order.

    Uncertainty values retain their existing mm/radian conventions and per-family
    multiplicities. The old main cited Liu et al., "Comparison of prostate volume,
    shape, and contouring variability determined from preimplant magnetic resonance
    and transrectal ultrasound images", Fig. 3, as prostate contouring context.
    These values are preserved, not re-derived or newly validated here. This registry remains transitional typed-config storage;
    replacing its dictionary schema is a separate migration.
    """
    records = {
        refs.bx_ref: {
            'Contour names': list(contours.biopsy),
            'Default mu X': [0],
            'Default mu Y': [0],
            'Default mu Z': [0],
            'Default sigma X': [2.5],
            'Default sigma Y': [2.5],
            'Default sigma Z': [2.5],
            'Dilations mu (xy)': [0],
            'Dilations mu (z)': [0],
            'Dilations sigma (xy)': [0],
            'Dilations sigma (z)': [0],
            'Rotations mu X': [0],
            'Rotations mu Y': [0],
            'Rotations mu Z': [0],
            'Rotations sigma X': [0],
            'Rotations sigma Y': [0],
            'Rotations sigma Z': [0],
            'Test tissue class': None,
            'Tissue heirarchy': None,
            'Tissue class name': None,
            'PCD color dict': {
                'Real': (0.5, 0.0, 0.5),
                CENTROID_DIL_SIMULATED_TYPE: (1.0, 0.55, 0.0),
                OPTIMAL_DIL_SIMULATED_TYPE: (0.0, 0.8, 0.6),
                biopsy.simulated.optimizer_simulated_type: (0.1, 0.65, 0.2),
            },
        },
        refs.oar_ref: {
            'Contour names': list(contours.oar),
            'Default mu X': [0],
            'Default mu Y': [0],
            'Default mu Z': [0],
            'Default sigma X': [2.5],
            'Default sigma Y': [2.5],
            'Default sigma Z': [2.5],
            'Dilations mu (xy)': [0],
            'Dilations mu (z)': [0],
            'Dilations sigma (xy)': [0],
            'Dilations sigma (z)': [0],
            'Rotations mu X': [0],
            'Rotations mu Y': [0],
            'Rotations mu Z': [0],
            'Rotations sigma X': [0],
            'Rotations sigma Y': [0],
            'Rotations sigma Z': [0],
            'Test tissue class': True,
            'Tissue heirarchy': 3,
            'Tissue class name': 'Prostatic',
            'PCD color': (0.86, 0.08, 0.24),
        },
        refs.dil_ref: {
            'Contour names': list(contours.dil),
            'Default mu X': [0],
            'Default mu Y': [0],
            'Default mu Z': [0],
            'Default sigma X': [2.5, 2.5, 2.5],
            'Default sigma Y': [2.5, 2.5, 2.5],
            'Default sigma Z': [2.5, 2.5, 2.5],
            'Dilations mu (xy)': [0],
            'Dilations mu (z)': [0],
            'Dilations sigma (xy)': [0],
            'Dilations sigma (z)': [0],
            'Rotations mu X': [0],
            'Rotations mu Y': [0],
            'Rotations mu Z': [0],
            'Rotations sigma X': [0],
            'Rotations sigma Y': [0],
            'Rotations sigma Z': [0],
            'Test tissue class': True,
            'Tissue heirarchy': 0,
            'Tissue class name': 'DIL',
            'PCD color': (0.13, 0.55, 0.13),
        },
        refs.rectum_ref_key: {
            'Contour names': list(contours.rectum),
            'Default mu X': [0],
            'Default mu Y': [0],
            'Default mu Z': [0],
            'Default sigma X': [2.5],
            'Default sigma Y': [2.5],
            'Default sigma Z': [2.5],
            'Dilations mu (xy)': [0],
            'Dilations mu (z)': [0],
            'Dilations sigma (xy)': [0],
            'Dilations sigma (z)': [0],
            'Rotations mu X': [0],
            'Rotations mu Y': [0],
            'Rotations mu Z': [0],
            'Rotations sigma X': [0],
            'Rotations sigma Y': [0],
            'Rotations sigma Z': [0],
            'Test tissue class': True,
            'Tissue heirarchy': 2,
            'Tissue class name': 'Rectal',
            'PCD color': (1.0, 0.84, 0.0),
        },
        refs.urethra_ref_key: {
            'Contour names': list(contours.urethra),
            'Default mu X': [0],
            'Default mu Y': [0],
            'Default mu Z': [0],
            'Default sigma X': [2.5],
            'Default sigma Y': [2.5],
            'Default sigma Z': [2.5],
            'Dilations mu (xy)': [0],
            'Dilations mu (z)': [0],
            'Dilations sigma (xy)': [0],
            'Dilations sigma (z)': [0],
            'Rotations mu X': [0],
            'Rotations mu Y': [0],
            'Rotations mu Z': [0],
            'Rotations sigma X': [0],
            'Rotations sigma Y': [0],
            'Rotations sigma Z': [0],
            'Test tissue class': True,
            'Tissue heirarchy': 1,
            'Tissue class name': 'Urethral',
            'PCD color': (0.0, 0.75, 1.0),
        },
    }
    tested = [key for key, record in records.items() if record.get("Test tissue class", False)]
    tested.insert(0, refs.bx_ref)
    generalized = list(records)
    unique = [key for key in generalized if key not in (refs.bx_ref, refs.dil_ref)]
    return StructureRegistryConfig(
        structs_referenced_dict=records,
        structs_referenced_list=tested,
        structs_referenced_list_generalized=generalized,
        structs_referenced_list_generalized_unique_structs=unique,
    )


def _build_optimizer_render_styles() -> dict:
    """Return plain style data; render consumers own native array conversion."""
    return {
        'stage_input_candidates': {
            'color': (0.88, 0.53, 0.1),
            'marker_size': 2.0,
            'opacity': 0.28,
        },
        'stage_survivors': {
            'color': (0.14, 0.68, 0.24),
            'marker_size': 3.2,
            'opacity': 0.88,
        },
        'target_points': {
            'color': (0.33, 0.63, 0.33),
            'marker_size': 0.7,
            'opacity': 0.1,
        },
        'target_structure_centroid': {
            'marker_size': 8.0,
            'opacity': 1.0,
        },
        'nominal_biopsy_centroid': {
            'color': (0.85, 0.2, 0.2),
            'marker_size': 7.0,
            'opacity': 1.0,
        },
        'operational_winner': {
            'color': (0.86, 0.12, 0.68),
            'marker_size': 8.0,
            'opacity': 1.0,
        },
        'planned_sampled_points': {
            'marker_size': 1.8,
            'opacity': 0.4,
        },
        'planned_core_structure': {
            'line_width': 5.0,
            'opacity': 0.98,
        },
        'planned_centroid_line': {
            'line_width': 6.0,
            'opacity': 1.0,
        },
        'target_structure_surface': {
            'line_width': 4.8,
            'opacity': 0.96,
        },
        'prostate_structure': {
            'line_width': 4.0,
            'opacity': 0.9,
        },
        'urethra_structure': {
            'line_width': 4.5,
            'opacity': 1.0,
        },
        'rectum_structure': {
            'line_width': 3.8,
            'opacity': 0.88,
        },
    }

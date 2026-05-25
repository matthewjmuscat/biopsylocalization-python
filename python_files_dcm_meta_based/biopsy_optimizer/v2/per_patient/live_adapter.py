"""Patient-level adapter for the optimizer-v2 live integration oracle.

This module keeps optimizer-v2 callable for a single patient without changing
the validated live integration function. The heavy search/scoring body still
lives in ``biopsy_optimizer.v2.live_integration`` during this validation phase.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from legacy_data_keys import legacy_data_keys
from presentation import LegacyPresentationContext


LEGACY_MASTER_INFO_KEYS = legacy_data_keys.master_info
LEGACY_STRUCTURE_RECORD_KEYS = legacy_data_keys.structure_record
LEGACY_PATIENT_ALL_REFERENCE_KEYS = legacy_data_keys.patient_all_reference
LEGACY_BIOPSY_RUNTIME_KEYS = legacy_data_keys.biopsy_runtime

OPTIMIZER_V2_OUTPUT_KEYS = (
    "Biopsy optimization - Target DIL optimizer v2 summary dataframe",
    "Biopsy optimization - Target DIL optimizer v2 ranked candidates dataframe",
    "Biopsy optimization - Target DIL optimizer v2 tested candidates dataframe",
)


@dataclass(frozen=True, slots=True)
class OptimizerV2LiveConfig:
    """Configuration required by the optimizer-v2 live integration wrapper."""

    structs_referenced_dict: Mapping[str, Any]
    bx_ref: str
    dil_ref: str
    all_ref_key: str
    optimizer_simulated_type: str
    search_config: Any
    constant_z_slice_polygons_handler_option: str
    remove_consecutive_duplicate_points_in_polygons: bool
    include_edges_in_log: bool
    kernel_type: str
    max_candidates_per_chunk: int | None = None
    max_test_structures_per_call: int | None = None
    fallback_max_test_structures_per_call: int | None = None
    auto_calibrate_max_test_structures_per_call: bool = True
    verify_calibrated_max_test_structures_per_call: bool = True
    validate_nearest_z_helper_against_ver5: bool = True
    downstream_comparable_trial_count: int | None = None
    benchmark_isolated_winner_validation_bool: bool = False
    render_stage_boundary_candidate_clouds_bool: bool = False
    render_stage_names_to_render: Sequence[str] | None = None
    render_backend: str = "open3d"
    render_layer_style_by_name: Mapping[str, Any] | None = None
    render_plotly_export_bool: bool = False
    render_plotly_export_formats: Sequence[str] = ("svg", "pdf")
    render_plotly_export_width: int = 1920
    render_plotly_export_height: int = 1080
    render_plotly_export_scale: float = 1.0
    render_plotly_export_camera_eye: Sequence[float] = (1.45, -1.45, 2.25)
    render_plotly_export_camera_center: Sequence[float] = (0.0, 0.0, 0.0)
    render_plotly_export_camera_up: Sequence[float] = (0.0, 0.0, 1.0)
    render_dialog_timeout_seconds: float | None = None
    render_dialog_timeout_extend_seconds: float = 300.0
    render_winner_containment_debug_bool: bool = False
    render_winner_containment_backend: str | None = None
    render_include_target_points_bool: bool = True
    render_patient_whitelist: Sequence[str] | None = None
    render_roi_whitelist: Sequence[str] | None = None
    render_include_planned_sampled_points_bool: bool = True
    render_include_planned_core_structure_bool: bool = True
    render_include_planned_centroid_line_bool: bool = True
    render_include_target_surface_bool: bool = True
    render_include_selected_anatomy_bool: bool = True
    oar_ref: str | None = None
    rectum_ref: str | None = None
    urethra_ref: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "bx_ref",
            "dil_ref",
            "all_ref_key",
            "optimizer_simulated_type",
            "constant_z_slice_polygons_handler_option",
            "kernel_type",
        ):
            field_value = str(getattr(self, field_name)).strip()
            if field_value == "":
                raise ValueError(f"{field_name} cannot be empty")
            object.__setattr__(self, field_name, field_value)
        object.__setattr__(self, "structs_referenced_dict", dict(self.structs_referenced_dict))
        object.__setattr__(self, "render_plotly_export_formats", tuple(self.render_plotly_export_formats))
        object.__setattr__(self, "render_plotly_export_camera_eye", tuple(self.render_plotly_export_camera_eye))
        object.__setattr__(self, "render_plotly_export_camera_center", tuple(self.render_plotly_export_camera_center))
        object.__setattr__(self, "render_plotly_export_camera_up", tuple(self.render_plotly_export_camera_up))
        if self.render_stage_names_to_render is not None:
            object.__setattr__(self, "render_stage_names_to_render", tuple(self.render_stage_names_to_render))
        if self.render_patient_whitelist is not None:
            object.__setattr__(self, "render_patient_whitelist", tuple(self.render_patient_whitelist))
        if self.render_roi_whitelist is not None:
            object.__setattr__(self, "render_roi_whitelist", tuple(self.render_roi_whitelist))
        if self.render_layer_style_by_name is not None:
            object.__setattr__(self, "render_layer_style_by_name", dict(self.render_layer_style_by_name))


@dataclass(slots=True)
class OptimizerV2PatientRunResult:
    """Output bundle from running optimizer-v2 against one patient."""

    patient_uid: str
    patient_reference_dict: dict[str, Any]
    master_structure_reference_dict: dict[str, dict[str, Any]]
    master_structure_info_dict: dict[str, Any]
    optimizer_outputs: dict[str, Any]
    presentation_context: LegacyPresentationContext
    live_display: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.patient_uid = str(self.patient_uid)
        self.metadata = dict(self.metadata or {})


def build_single_patient_optimizer_v2_master_info(patient_uid: str,
                                                  patient_info_dict: Mapping[str, Any] | None,
                                                  *,
                                                  global_info: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build the minimal legacy master-info shape for one optimizer-v2 patient."""
    if (
        patient_info_dict is not None
        and LEGACY_MASTER_INFO_KEYS.global_key in patient_info_dict
        and LEGACY_MASTER_INFO_KEYS.by_patient_key in patient_info_dict
    ):
        master_info = copy.deepcopy(dict(patient_info_dict))
        master_info.setdefault(LEGACY_MASTER_INFO_KEYS.global_key, {})[LEGACY_MASTER_INFO_KEYS.num_cases_key] = 1
        return master_info

    resolved_global_info = dict(global_info or {})
    resolved_global_info[LEGACY_MASTER_INFO_KEYS.num_cases_key] = 1
    return {
        LEGACY_MASTER_INFO_KEYS.global_key: resolved_global_info,
        LEGACY_MASTER_INFO_KEYS.by_patient_key: {
            str(patient_uid): copy.deepcopy(dict(patient_info_dict or {})),
        },
    }


def collect_optimizer_v2_patient_outputs(patient_reference_dict: Mapping[str, Any],
                                         *,
                                         bx_ref: str,
                                         all_ref_key: str) -> dict[str, Any]:
    """Collect optimizer-v2 outputs written into one patient's legacy dictionary."""
    pre_processing_dataframe_dict = patient_reference_dict[all_ref_key][
        LEGACY_PATIENT_ALL_REFERENCE_KEYS.preprocessing_output_dataframes_key
    ]
    biopsy_transport_requests = []
    for biopsy_structure in patient_reference_dict.get(bx_ref, ()):
        transport_request = biopsy_structure.get(LEGACY_BIOPSY_RUNTIME_KEYS.simulated_biopsy_transport_request_key)
        if transport_request is not None:
            biopsy_transport_requests.append(
                {
                    LEGACY_STRUCTURE_RECORD_KEYS.roi_key: biopsy_structure.get(LEGACY_STRUCTURE_RECORD_KEYS.roi_key),
                    LEGACY_STRUCTURE_RECORD_KEYS.ref_number_key: biopsy_structure.get(
                        LEGACY_STRUCTURE_RECORD_KEYS.ref_number_key
                    ),
                    LEGACY_STRUCTURE_RECORD_KEYS.index_number_key: biopsy_structure.get(
                        LEGACY_STRUCTURE_RECORD_KEYS.index_number_key
                    ),
                    LEGACY_STRUCTURE_RECORD_KEYS.simulated_type_key: biopsy_structure.get(
                        LEGACY_STRUCTURE_RECORD_KEYS.simulated_type_key
                    ),
                    "transport_request": transport_request,
                }
            )
    return {
        "dataframes": {
            output_key: pre_processing_dataframe_dict.get(output_key)
            for output_key in OPTIMIZER_V2_OUTPUT_KEYS
        },
        "biopsy_transport_requests": biopsy_transport_requests,
    }


def run_patient_target_dil_optimizer_v2_live_adapter(
    *,
    patient_uid: str,
    patient_reference_dict: dict[str, Any],
    patient_info_dict: Mapping[str, Any] | None,
    config: OptimizerV2LiveConfig,
    parallel_pool: Any,
    presentation_context: LegacyPresentationContext | None = None,
    global_info: Mapping[str, Any] | None = None,
    mutate_input: bool = True,
) -> OptimizerV2PatientRunResult:
    """Run the optimizer-v2 live integration surface against one patient."""
    from biopsy_optimizer.v2.live_integration import run_target_dil_optimizer_v2_for_live_simulated_family

    patient_uid = str(patient_uid)
    working_patient_reference_dict = patient_reference_dict if mutate_input else copy.deepcopy(patient_reference_dict)
    master_structure_reference_dict = {patient_uid: working_patient_reference_dict}
    master_structure_info_dict = build_single_patient_optimizer_v2_master_info(
        patient_uid,
        patient_info_dict,
        global_info=global_info,
    )
    context = presentation_context or LegacyPresentationContext.null()
    live_display = run_target_dil_optimizer_v2_for_live_simulated_family(
        master_structure_reference_dict,
        master_structure_info_dict,
        config.structs_referenced_dict,
        config.bx_ref,
        config.dil_ref,
        config.all_ref_key,
        config.optimizer_simulated_type,
        config.search_config,
        parallel_pool,
        config.constant_z_slice_polygons_handler_option,
        config.remove_consecutive_duplicate_points_in_polygons,
        config.include_edges_in_log,
        config.kernel_type,
        context.patients_progress,
        context.structures_progress,
        context.completed_progress,
        context.live_display,
        max_candidates_per_chunk=config.max_candidates_per_chunk,
        max_test_structures_per_call=config.max_test_structures_per_call,
        fallback_max_test_structures_per_call=config.fallback_max_test_structures_per_call,
        auto_calibrate_max_test_structures_per_call=config.auto_calibrate_max_test_structures_per_call,
        verify_calibrated_max_test_structures_per_call=config.verify_calibrated_max_test_structures_per_call,
        validate_nearest_z_helper_against_ver5=config.validate_nearest_z_helper_against_ver5,
        downstream_comparable_trial_count=config.downstream_comparable_trial_count,
        benchmark_isolated_winner_validation_bool=config.benchmark_isolated_winner_validation_bool,
        render_stage_boundary_candidate_clouds_bool=config.render_stage_boundary_candidate_clouds_bool,
        render_stage_names_to_render=config.render_stage_names_to_render,
        render_backend=config.render_backend,
        render_layer_style_by_name=config.render_layer_style_by_name,
        render_plotly_export_bool=config.render_plotly_export_bool,
        render_plotly_export_formats=config.render_plotly_export_formats,
        render_plotly_export_width=config.render_plotly_export_width,
        render_plotly_export_height=config.render_plotly_export_height,
        render_plotly_export_scale=config.render_plotly_export_scale,
        render_plotly_export_camera_eye=config.render_plotly_export_camera_eye,
        render_plotly_export_camera_center=config.render_plotly_export_camera_center,
        render_plotly_export_camera_up=config.render_plotly_export_camera_up,
        render_dialog_timeout_seconds=config.render_dialog_timeout_seconds,
        render_dialog_timeout_extend_seconds=config.render_dialog_timeout_extend_seconds,
        render_winner_containment_debug_bool=config.render_winner_containment_debug_bool,
        render_winner_containment_backend=config.render_winner_containment_backend,
        render_include_target_points_bool=config.render_include_target_points_bool,
        render_patient_whitelist=config.render_patient_whitelist,
        render_roi_whitelist=config.render_roi_whitelist,
        render_include_planned_sampled_points_bool=config.render_include_planned_sampled_points_bool,
        render_include_planned_core_structure_bool=config.render_include_planned_core_structure_bool,
        render_include_planned_centroid_line_bool=config.render_include_planned_centroid_line_bool,
        render_include_target_surface_bool=config.render_include_target_surface_bool,
        render_include_selected_anatomy_bool=config.render_include_selected_anatomy_bool,
        oar_ref=config.oar_ref,
        rectum_ref=config.rectum_ref,
        urethra_ref=config.urethra_ref,
    )
    optimizer_outputs = collect_optimizer_v2_patient_outputs(
        working_patient_reference_dict,
        bx_ref=config.bx_ref,
        all_ref_key=config.all_ref_key,
    )
    return OptimizerV2PatientRunResult(
        patient_uid=patient_uid,
        patient_reference_dict=working_patient_reference_dict,
        master_structure_reference_dict=master_structure_reference_dict,
        master_structure_info_dict=master_structure_info_dict,
        optimizer_outputs=optimizer_outputs,
        presentation_context=context,
        live_display=live_display,
        metadata={
            "optimizer_simulated_type": config.optimizer_simulated_type,
            "mutate_input": bool(mutate_input),
        },
    )
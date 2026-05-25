"""Patient-level validation adapter for the legacy optimizer-v1 oracle.

Optimizer-v1 is retained here as an additive one-patient bridge. It lets a
future patient runner execute the validated cohort oracle against a singleton
patient dictionary, without changing the legacy cohort entrypoint or duplicating
the optimizer math.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from presentation import LegacyPresentationContext


OPTIMIZER_V1_DIL_OUTPUT_KEYS = (
    "Biopsy optimization: DIL centroid optimal biopsy location dataframe",
    "Biopsy optimization: Optimal biopsy location dataframe",
    "Biopsy optimization: Optimal biopsy location (all tested lattice points) dataframe",
    "Biopsy optimization: Optimal biopsy location (zero lattice) dataframe",
    "Biopsy optimization: cubic lattice of optimization points only in dil",
    "Biopsy optimization: guidance map max-planes dataframe",
)

OPTIMIZER_V1_MULTI_STRUCTURE_INFORMATION_KEYS = (
    "Biopsy optimization: All points outside of DILs (zero points) dataframe",
    "Biopsy optimization: All points within DILs (tested points) dataframe",
    "Biopsy optimization: Optimal biopsy location (entire cubic lattice) dataframe",
)

OPTIMIZER_V1_MULTI_STRUCTURE_PREPROCESSING_KEYS = (
    "Biopsy optimization - Cumulative projection (all points within prostate) dataframe",
)


@dataclass(frozen=True, slots=True)
class OptimizerV1LegacyConfig:
    """Configuration required by the legacy optimizer-v1 function."""

    structs_referenced_dict: Mapping[str, Any]
    bx_ref: str
    dil_ref: str
    oar_ref: str
    all_ref_key: str
    voxel_size_for_dil_optimizer_grid: float
    optimal_normal_dist_option: str
    bias_LR_multiplier: float
    bias_AP_multiplier: float
    bias_SI_multiplier: float
    num_normal_dist_points_for_biopsy_optimizer: int
    normal_dist_sigma_factor_biopsy_optimizer: float
    plot_each_normal_dist_containment_result_bool: bool
    plot_optimization_point_lattice_bool: bool
    show_optimization_point_bool: bool
    cupy_array_upper_limit_NxN_size_input: int
    numpy_array_upper_limit_NxN_size_input: int
    nearest_zslice_vals_and_indices_cupy_generic_max_size: int
    nearest_zslice_vals_and_indices_numpy_generic_max_size: int
    constant_z_slice_polygons_handler_option: str
    remove_consecutive_duplicate_points_in_polygons: bool
    include_edges_in_log_files: bool
    custom_cuda_kernel_type: str
    demonstrate_dil_optimization_points_inside_correctness_bool_1: bool
    demonstrate_dil_optimization_points_inside_correctness_bool_2: bool
    demonstrate_dil_optimization_points_inside_correctness_num_3: int
    generate_cuda_log_files_biopsy_optimizer: bool
    display_optimization_contour_plots_bool: bool

    def as_legacy_args(self) -> tuple[Any, ...]:
        return (
            self.structs_referenced_dict,
            self.bx_ref,
            self.dil_ref,
            self.oar_ref,
            self.all_ref_key,
            self.voxel_size_for_dil_optimizer_grid,
            self.optimal_normal_dist_option,
            self.bias_LR_multiplier,
            self.bias_AP_multiplier,
            self.bias_SI_multiplier,
            self.num_normal_dist_points_for_biopsy_optimizer,
            self.normal_dist_sigma_factor_biopsy_optimizer,
            self.plot_each_normal_dist_containment_result_bool,
            self.plot_optimization_point_lattice_bool,
            self.show_optimization_point_bool,
            self.cupy_array_upper_limit_NxN_size_input,
            self.numpy_array_upper_limit_NxN_size_input,
            self.nearest_zslice_vals_and_indices_cupy_generic_max_size,
            self.nearest_zslice_vals_and_indices_numpy_generic_max_size,
            self.constant_z_slice_polygons_handler_option,
            self.remove_consecutive_duplicate_points_in_polygons,
            self.include_edges_in_log_files,
            self.custom_cuda_kernel_type,
            self.demonstrate_dil_optimization_points_inside_correctness_bool_1,
            self.demonstrate_dil_optimization_points_inside_correctness_bool_2,
            self.demonstrate_dil_optimization_points_inside_correctness_num_3,
            self.generate_cuda_log_files_biopsy_optimizer,
            self.display_optimization_contour_plots_bool,
        )


@dataclass(slots=True)
class OptimizerV1PatientRunResult:
    """Output bundle from running optimizer-v1 against one patient."""

    patient_uid: str
    patient_reference_dict: dict[str, Any]
    master_structure_reference_dict: dict[str, dict[str, Any]]
    master_structure_info_dict: dict[str, Any]
    optimizer_outputs: dict[str, Any]
    presentation_context: LegacyPresentationContext
    live_display: Any = None


def build_patient_info_from_reference(patient_uid: str,
                                      patient_reference_dict: Mapping[str, Any],
                                      *,
                                      bx_ref: str,
                                      dil_ref: str,
                                      oar_ref: str,
                                      all_ref_key: str,
                                      rectum_ref: str | None = None,
                                      urethra_ref: str | None = None) -> dict[str, Any]:
    """Derive the minimal patient info dict needed by the singleton adapter."""
    biopsies = list(patient_reference_dict.get(bx_ref, ()))
    biopsy_type_counts: dict[str, int] = {}
    for biopsy in biopsies:
        biopsy_type = str(biopsy.get("Simulated type", "Real"))
        biopsy_type_counts[biopsy_type] = biopsy_type_counts.get(biopsy_type, 0) + 1

    rectum_count = len(patient_reference_dict.get(rectum_ref, ())) if rectum_ref else 0
    urethra_count = len(patient_reference_dict.get(urethra_ref, ())) if urethra_ref else 0
    total_num_structs = (
        len(biopsies)
        + len(patient_reference_dict.get(oar_ref, ()))
        + len(patient_reference_dict.get(dil_ref, ()))
        + rectum_count
        + urethra_count
    )
    return {
        "Patient UID (generated)": str(patient_uid),
        "Patient ID (from dicom)": patient_reference_dict.get("Patient ID (from dicom)"),
        "Patient Name": patient_reference_dict.get("Patient Name"),
        "Fraction number": patient_reference_dict.get("Fraction number"),
        bx_ref: {
            "Num structs": len(biopsies),
            "Num sim structs": sum(1 for biopsy in biopsies if biopsy.get("Simulated bool", False)),
            "Num real structs": sum(1 for biopsy in biopsies if not biopsy.get("Simulated bool", False)),
            "Biopsy type counts": biopsy_type_counts,
        },
        oar_ref: {"Num structs": len(patient_reference_dict.get(oar_ref, ()))},
        dil_ref: {"Num structs": len(patient_reference_dict.get(dil_ref, ()))},
        all_ref_key: {"Total num structs": total_num_structs},
    }


def build_single_patient_master_structure_info(patient_uid: str,
                                               patient_info_dict: Mapping[str, Any],
                                               *,
                                               bx_ref: str,
                                               dil_ref: str,
                                               all_ref_key: str,
                                               bx_types_list: list[str] | None = None) -> dict[str, Any]:
    """Wrap one patient info dict in the legacy master-info shape."""
    if "Global" in patient_info_dict and "By patient" in patient_info_dict:
        return copy.deepcopy(dict(patient_info_dict))

    biopsy_info = patient_info_dict.get(bx_ref, {})
    dil_info = patient_info_dict.get(dil_ref, {})
    all_info = patient_info_dict.get(all_ref_key, {})
    if bx_types_list is None:
        bx_types_list = list(biopsy_info.get("Biopsy type counts", {"Real": 0}).keys())

    return {
        "Global": {
            "Num cases": 1,
            "Num structures": all_info.get("Total num structs"),
            "Num biopsies": biopsy_info.get("Num structs"),
            "Num biopsies by bx type dict": dict(biopsy_info.get("Biopsy type counts", {})),
            "Num DILs": dil_info.get("Num structs"),
            "Bx types list": bx_types_list,
        },
        "By patient": {
            str(patient_uid): copy.deepcopy(dict(patient_info_dict)),
        },
    }


def collect_optimizer_v1_patient_outputs(patient_reference_dict: Mapping[str, Any],
                                         *,
                                         dil_ref: str,
                                         all_ref_key: str) -> dict[str, Any]:
    """Collect the optimizer-v1 outputs written into the patient dictionary."""
    multi_structure_info = patient_reference_dict[all_ref_key]["Multi-structure information dict (not for csv output)"]
    preprocessing_outputs = patient_reference_dict[all_ref_key]["Multi-structure pre-processing output dataframes dict"]
    return {
        "per_dil": [
            {
                "ROI": dil_structure.get("ROI"),
                "Ref #": dil_structure.get("Ref #"),
                "Index number": dil_structure.get("Index number"),
                "outputs": {
                    output_key: dil_structure.get(output_key)
                    for output_key in OPTIMIZER_V1_DIL_OUTPUT_KEYS
                    if output_key in dil_structure
                },
            }
            for dil_structure in patient_reference_dict.get(dil_ref, ())
        ],
        "multi_structure_information": {
            output_key: multi_structure_info.get(output_key)
            for output_key in OPTIMIZER_V1_MULTI_STRUCTURE_INFORMATION_KEYS
            if output_key in multi_structure_info
        },
        "multi_structure_preprocessing": {
            output_key: preprocessing_outputs.get(output_key)
            for output_key in OPTIMIZER_V1_MULTI_STRUCTURE_PREPROCESSING_KEYS
            if output_key in preprocessing_outputs
        },
    }


def run_patient_optimizer_v1_legacy_adapter(
    *,
    patient_uid: str,
    patient_reference_dict: dict[str, Any],
    patient_info_dict: Mapping[str, Any] | None,
    config: OptimizerV1LegacyConfig,
    presentation_context: LegacyPresentationContext | None = None,
    mutate_input: bool = True,
) -> OptimizerV1PatientRunResult:
    """Run the legacy optimizer-v1 oracle against a singleton patient cohort."""
    from biopsy_optimizer.v1.biopsy_optimizer_module_v1 import biopsy_optimizer_module_v1

    patient_uid = str(patient_uid)
    working_patient_reference_dict = patient_reference_dict if mutate_input else copy.deepcopy(patient_reference_dict)
    if patient_info_dict is None:
        working_patient_info_dict = build_patient_info_from_reference(
            patient_uid,
            working_patient_reference_dict,
            bx_ref=config.bx_ref,
            dil_ref=config.dil_ref,
            oar_ref=config.oar_ref,
            all_ref_key=config.all_ref_key,
        )
    else:
        working_patient_info_dict = copy.deepcopy(dict(patient_info_dict))
    master_structure_reference_dict = {patient_uid: working_patient_reference_dict}
    master_structure_info_dict = build_single_patient_master_structure_info(
        patient_uid,
        working_patient_info_dict,
        bx_ref=config.bx_ref,
        dil_ref=config.dil_ref,
        all_ref_key=config.all_ref_key,
    )
    context = presentation_context or LegacyPresentationContext.null()
    live_display = biopsy_optimizer_module_v1(
        master_structure_reference_dict,
        master_structure_info_dict,
        *config.as_legacy_args(),
        context.layout_groups,
        context.patients_progress,
        context.structures_progress,
        context.indeterminate_progress_sub,
        context.important_info,
        context.completed_progress,
        context.live_display,
    )
    optimizer_outputs = collect_optimizer_v1_patient_outputs(
        working_patient_reference_dict,
        dil_ref=config.dil_ref,
        all_ref_key=config.all_ref_key,
    )
    return OptimizerV1PatientRunResult(
        patient_uid=patient_uid,
        patient_reference_dict=working_patient_reference_dict,
        master_structure_reference_dict=master_structure_reference_dict,
        master_structure_info_dict=master_structure_info_dict,
        optimizer_outputs=optimizer_outputs,
        presentation_context=context,
        live_display=live_display,
    )
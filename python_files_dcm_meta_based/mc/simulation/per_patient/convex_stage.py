"""Patient-local convex MC containment and dose stage."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping

from .containment import (
    PatientContainmentOutputs,
    build_patient_containment_biopsy_context,
    build_patient_containment_dilated_structure_bank,
    compile_patient_containment_biopsy_statistics,
    iter_patient_containment_relative_structure_inputs,
    run_patient_relative_structure_containment_core,
    write_patient_containment_biopsy_statistics_to_legacy_record,
)
from .contracts import MCConvexSimulationConfig
from .convex_legacy_adapter import build_single_patient_mc_master_info, collect_mc_patient_outputs
from .dose import (
    MC_DOSE_LOCALIZATION_KIND_DOSE,
    MC_DOSE_LOCALIZATION_KIND_GRADIENT,
    PatientDoseOutputs,
    build_patient_dose_biopsy_context,
    build_patient_dose_lattice_context,
    compile_patient_dose_dvh_outputs_for_biopsy,
    patient_has_dose_reference,
    resolve_patient_target_prescription_dose,
    run_patient_dose_localization_for_biopsy,
    write_patient_dose_dvh_outputs_to_legacy_record,
    write_patient_dose_localization_outputs_to_legacy_record,
)
from .legacy_keys import legacy_mc_keys
from .relative_structure_inventory import (
    PatientRelativeStructureInventory,
    RelativeStructureInfo,
    build_patient_relative_structure_inventory,
)

MC_BX_SAMPLE_PT_VOLUME_ELEMENT_KEY = "BX sample pt volume element (mm^3)"


@dataclass(slots=True)
class PatientMCConvexStageResult:
    """Output bundle from running patient-local convex MC containment/dose."""

    patient_uid: str
    patient_reference_dict: dict[str, Any]
    master_structure_reference_dict: dict[str, dict[str, Any]]
    master_structure_info_dict: dict[str, Any]
    containment_outputs: PatientContainmentOutputs
    dose_outputs: PatientDoseOutputs
    containment_biopsy_count: int
    dose_biopsy_count: int
    dose_reference_available: bool
    plan_reference_available: bool
    performed_flags: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.patient_uid = str(self.patient_uid)
        self.containment_biopsy_count = int(self.containment_biopsy_count)
        self.dose_biopsy_count = int(self.dose_biopsy_count)
        self.dose_reference_available = bool(self.dose_reference_available)
        self.plan_reference_available = bool(self.plan_reference_available)
        self.performed_flags = dict(self.performed_flags or {})
        self.metadata = dict(self.metadata or {})


def _reject_patient_mc_convex_stage_side_effect_options(config: MCConvexSimulationConfig) -> None:
    containment_config = config.containment
    dose_config = config.dose
    runtime_config = config.runtime

    side_effect_options = {
        "raw_data_mc_containment_dump_bool": containment_config.raw_data_mc_containment_dump_bool,
        "raw_data_mc_dosimetry_dump_bool": dose_config.raw_data_mc_dosimetry_dump_bool,
        "show_num_containment_demonstration_plots": (
            int(containment_config.show_num_containment_demonstration_plots) > 0
        ),
        "show_num_nearest_neighbour_surface_boundary_demonstration": (
            int(containment_config.show_num_nearest_neighbour_surface_boundary_demonstration) > 0
        ),
        "show_num_relative_structure_centroid_demonstration": (
            int(containment_config.show_num_relative_structure_centroid_demonstration) > 0
        ),
        "plot_cupy_containment_distribution_results": (
            containment_config.plot_cupy_containment_distribution_results
        ),
        "show_non_bx_relative_structure_z_dilation_bool": (
            containment_config.show_non_bx_relative_structure_z_dilation_bool
        ),
        "show_non_bx_relative_structure_xy_dilation_bool": (
            containment_config.show_non_bx_relative_structure_xy_dilation_bool
        ),
        "generate_cuda_log_files_MC_containment_sim": (
            containment_config.generate_cuda_log_files_MC_containment_sim
        ),
        "show_NN_dose_demonstration_plots": dose_config.show_NN_dose_demonstration_plots,
        "show_NN_dose_demonstration_plots_all_trials_at_once": (
            dose_config.show_NN_dose_demonstration_plots_all_trials_at_once
        ),
        "plot_uniform_shifts_to_check_plotly": runtime_config.plot_uniform_shifts_to_check_plotly,
        "plot_translation_vectors_pointclouds": runtime_config.plot_translation_vectors_pointclouds,
        "plot_shifted_biopsies": runtime_config.plot_shifted_biopsies,
    }
    unsupported_options = [option for option, requested in side_effect_options.items() if bool(requested)]
    if unsupported_options:
        raise ValueError(
            "Patient convex MC stage does not perform raw CSV dumps, plotting, or CUDA log side effects; "
            "use the convex legacy adapter for oracle/debug validation. Unsupported options: "
            + ", ".join(unsupported_options)
        )


def _resolve_mc_info_value(master_structure_info_dict: Mapping[str, Any], key: str) -> Any:
    master_keys = legacy_mc_keys.master_info
    try:
        return master_structure_info_dict[master_keys.global_key][master_keys.mc_info_key][key]
    except KeyError as exc:
        raise KeyError(
            f"master_structure_info_dict['Global']['MC info'][{key!r}] is required for patient convex MC"
        ) from exc


def _resolve_num_mc_containment_simulations(master_structure_info_dict: Mapping[str, Any]) -> int:
    master_keys = legacy_mc_keys.master_info
    return int(_resolve_mc_info_value(master_structure_info_dict, master_keys.num_containment_simulations_key))


def _resolve_num_mc_dose_simulations(master_structure_info_dict: Mapping[str, Any]) -> int:
    master_keys = legacy_mc_keys.master_info
    return int(_resolve_mc_info_value(master_structure_info_dict, master_keys.num_dose_simulations_key))


def _resolve_bx_sample_pts_volume_element(master_structure_info_dict: Mapping[str, Any]) -> float:
    return float(_resolve_mc_info_value(master_structure_info_dict, MC_BX_SAMPLE_PT_VOLUME_ELEMENT_KEY))


def _set_patient_mc_convex_performed_flags(master_structure_info_dict: dict[str, Any],
                                           *,
                                           containment_performed: Any,
                                           dose_performed: Any) -> dict[str, Any]:
    master_keys = legacy_mc_keys.master_info
    global_info = master_structure_info_dict.setdefault(master_keys.global_key, {})
    mc_info = global_info.setdefault(master_keys.mc_info_key, {})
    mc_info[master_keys.containment_performed_key] = bool(containment_performed)
    mc_info[master_keys.dose_performed_key] = bool(dose_performed)
    mc_info[master_keys.sim_performed_key] = any(
        [
            bool(mc_info.get(master_keys.sim_performed_key, False)),
            bool(mc_info.get(master_keys.mr_performed_key, False)),
            bool(mc_info[master_keys.containment_performed_key]),
            bool(mc_info[master_keys.dose_performed_key]),
        ]
    )
    return {
        master_keys.containment_performed_key: mc_info[master_keys.containment_performed_key],
        master_keys.dose_performed_key: mc_info[master_keys.dose_performed_key],
        master_keys.sim_performed_key: mc_info[master_keys.sim_performed_key],
    }


def _build_relative_structure_template_from_reference(
    patient_reference_dict: Mapping[str, Any],
    *,
    structs_referenced_list: tuple[str, ...],
    bx_ref: str,
) -> dict[RelativeStructureInfo, None]:
    identity_keys = legacy_mc_keys.biopsy_identity
    relative_structure_template: dict[RelativeStructureInfo, None] = {}
    for structure_type in structs_referenced_list:
        if structure_type == bx_ref:
            continue
        for structure_index, structure_record in enumerate(patient_reference_dict.get(structure_type, ())):
            structure_info = (
                structure_record[identity_keys.roi_key],
                structure_type,
                structure_record[identity_keys.ref_number_key],
                structure_index,
            )
            relative_structure_template[structure_info] = None
    return relative_structure_template


def _build_patient_relative_structure_inventory_for_stage(
    patient_uid: str,
    patient_reference_dict: Mapping[str, Any],
    patient_info_dict: Mapping[str, Any] | None,
    config: MCConvexSimulationConfig,
) -> PatientRelativeStructureInventory:
    keys = config.keys
    if patient_info_dict is not None:
        return build_patient_relative_structure_inventory(
            patient_uid,
            patient_reference_dict,
            patient_info_dict,
            structs_referenced_list=keys.structs_referenced_list,
            bx_ref=keys.bx_ref,
            all_ref_key=keys.all_ref_key,
        )

    structs_referenced_list = tuple(keys.structs_referenced_list)
    relative_structure_template = _build_relative_structure_template_from_reference(
        patient_reference_dict,
        structs_referenced_list=structs_referenced_list,
        bx_ref=keys.bx_ref,
    )
    total_num_biopsies = len(patient_reference_dict.get(keys.bx_ref, ()))
    total_num_non_biopsies = len(relative_structure_template)
    return PatientRelativeStructureInventory(
        patient_uid=str(patient_uid),
        relative_structure_template=relative_structure_template,
        total_num_structures=total_num_biopsies + total_num_non_biopsies,
        total_num_biopsies=total_num_biopsies,
        total_num_non_biopsies=total_num_non_biopsies,
    )


def _run_patient_mc_containment_stage(
    *,
    patient_uid: str,
    patient_reference_dict: dict[str, Any],
    patient_info_dict: Mapping[str, Any] | None,
    config: MCConvexSimulationConfig,
    parallel_pool: Any,
    num_mc_containment_simulations: int,
) -> tuple[int, int, int]:
    import pandas

    keys = config.keys
    inventory = _build_patient_relative_structure_inventory_for_stage(
        patient_uid,
        patient_reference_dict,
        patient_info_dict,
        config,
    )
    dilated_structure_bank = build_patient_containment_dilated_structure_bank(
        patient_uid,
        patient_reference_dict,
        inventory,
        num_mc_containment_simulations=num_mc_containment_simulations,
        oar_ref=keys.oar_ref,
        rectum_ref=keys.rectum_ref,
        urethra_ref=keys.urethra_ref,
        containment_config=config.containment,
        parallel_pool=parallel_pool,
    )

    biopsy_count = 0
    relative_comparison_count = 0
    for biopsy_index, biopsy_structure in enumerate(patient_reference_dict.get(keys.bx_ref, ())):
        biopsy_context = build_patient_containment_biopsy_context(
            patient_uid,
            biopsy_index,
            biopsy_structure,
        )
        containment_dataframes = []
        for relative_structure_input in iter_patient_containment_relative_structure_inputs(
            biopsy_context,
            dilated_structure_bank,
            num_mc_containment_simulations=num_mc_containment_simulations,
        ):
            containment_dataframes.append(
                run_patient_relative_structure_containment_core(
                    relative_structure_input,
                    biopsy_context,
                    num_mc_containment_simulations=num_mc_containment_simulations,
                    containment_config=config.containment,
                    runtime_config=config.runtime,
                )
            )

        if containment_dataframes:
            containment_dataframe = pandas.concat(containment_dataframes, ignore_index=True)
            statistics_outputs = compile_patient_containment_biopsy_statistics(
                containment_dataframe,
                biopsy_context,
                inventory,
                keys.structs_referenced_dict,
                num_mc_containment_simulations=num_mc_containment_simulations,
                biopsy_z_voxel_length=config.dose.biopsy_z_voxel_length,
                default_exterior_tissue=config.containment.default_exterior_tissue,
                keep_light_containment_and_distances_dataframe=(
                    config.containment.keep_light_containment_and_distances_to_relative_structures_dataframe_bool
                ),
                parallel_pool=parallel_pool,
            )
            write_patient_containment_biopsy_statistics_to_legacy_record(
                biopsy_structure,
                statistics_outputs,
            )
            relative_comparison_count += len(containment_dataframes)
        biopsy_count += 1

    return biopsy_count, relative_comparison_count, len(inventory.relative_structure_infos)


def _patient_has_plan_reference(patient_reference_dict: Mapping[str, Any], *, plan_ref: str) -> bool:
    return str(plan_ref) in patient_reference_dict


def _run_patient_dose_localization_kind(
    *,
    patient_uid: str,
    patient_reference_dict: dict[str, Any],
    config: MCConvexSimulationConfig,
    num_mc_dose_simulations: int,
    localization_kind: str,
) -> int:
    keys = config.keys
    lattice_context = build_patient_dose_lattice_context(
        patient_uid,
        patient_reference_dict,
        dose_ref=keys.dose_ref,
        localization_kind=localization_kind,
        mutate_reference=True,
    )
    biopsy_count = 0
    for biopsy_index, biopsy_structure in enumerate(patient_reference_dict.get(keys.bx_ref, ())):
        biopsy_context = build_patient_dose_biopsy_context(
            patient_uid,
            biopsy_index,
            biopsy_structure,
            num_mc_dose_simulations=num_mc_dose_simulations,
        )
        localization_outputs = run_patient_dose_localization_for_biopsy(
            biopsy_context,
            lattice_context,
            dose_config=config.dose,
            num_mc_dose_simulations=num_mc_dose_simulations,
        )
        write_patient_dose_localization_outputs_to_legacy_record(
            biopsy_structure,
            localization_outputs,
        )
        biopsy_count += 1
    return biopsy_count


def _run_patient_dose_dvh_stage(
    *,
    patient_reference_dict: dict[str, Any],
    config: MCConvexSimulationConfig,
    bx_sample_pts_volume_element: float,
) -> int:
    keys = config.keys
    ctv_dose = resolve_patient_target_prescription_dose(
        patient_reference_dict,
        plan_ref=keys.plan_ref,
    )
    biopsy_count = 0
    for biopsy_structure in patient_reference_dict.get(keys.bx_ref, ()):
        dvh_outputs = compile_patient_dose_dvh_outputs_for_biopsy(
            biopsy_structure,
            dose_config=config.dose,
            ctv_dose=ctv_dose,
            bx_sample_pts_volume_element=bx_sample_pts_volume_element,
        )
        write_patient_dose_dvh_outputs_to_legacy_record(
            biopsy_structure,
            dvh_outputs,
        )
        biopsy_count += 1
    return biopsy_count


def run_patient_mc_convex_stage(
    *,
    patient_uid: str,
    patient_reference_dict: dict[str, Any],
    patient_info_dict: Mapping[str, Any] | None,
    config: MCConvexSimulationConfig,
    parallel_pool: Any,
    global_info: Mapping[str, Any] | None = None,
    mutate_input: bool = True,
) -> PatientMCConvexStageResult:
    """Run convex MC containment/dose for one patient without oracle UI/file side effects."""
    _reject_patient_mc_convex_stage_side_effect_options(config)
    patient_uid = str(patient_uid)
    working_patient_reference_dict = patient_reference_dict if mutate_input else copy.deepcopy(patient_reference_dict)
    master_structure_reference_dict = {patient_uid: working_patient_reference_dict}
    master_structure_info_dict = build_single_patient_mc_master_info(
        patient_uid,
        patient_info_dict,
        global_info=global_info,
    )
    performed_flags = _set_patient_mc_convex_performed_flags(
        master_structure_info_dict,
        containment_performed=config.containment.perform_mc_containment_sim,
        dose_performed=config.dose.perform_mc_dose_sim,
    )

    metadata: dict[str, Any] = {"mutated_input": bool(mutate_input)}
    containment_biopsy_count = 0
    containment_relative_comparison_count = 0
    containment_relative_structure_count = 0
    if config.containment.perform_mc_containment_sim:
        num_mc_containment_simulations = _resolve_num_mc_containment_simulations(master_structure_info_dict)
        (
            containment_biopsy_count,
            containment_relative_comparison_count,
            containment_relative_structure_count,
        ) = _run_patient_mc_containment_stage(
            patient_uid=patient_uid,
            patient_reference_dict=working_patient_reference_dict,
            patient_info_dict=master_structure_info_dict,
            config=config,
            parallel_pool=parallel_pool,
            num_mc_containment_simulations=num_mc_containment_simulations,
        )
        metadata["num_mc_containment_simulations"] = num_mc_containment_simulations
    else:
        metadata["containment_skip_reason"] = "disabled"

    dose_reference_available = patient_has_dose_reference(
        working_patient_reference_dict,
        dose_ref=config.keys.dose_ref,
    )
    plan_reference_available = _patient_has_plan_reference(
        working_patient_reference_dict,
        plan_ref=config.keys.plan_ref,
    )
    dose_biopsy_count = 0
    dose_gradient_biopsy_count = 0
    dose_dvh_biopsy_count = 0
    if not config.dose.perform_mc_dose_sim:
        metadata["dose_skip_reason"] = "disabled"
    elif not dose_reference_available:
        metadata["dose_skip_reason"] = "patient_missing_dose_reference"
    elif not plan_reference_available:
        metadata["dose_skip_reason"] = "patient_missing_plan_reference"
    else:
        num_mc_dose_simulations = _resolve_num_mc_dose_simulations(master_structure_info_dict)
        bx_sample_pts_volume_element = _resolve_bx_sample_pts_volume_element(master_structure_info_dict)
        dose_biopsy_count = _run_patient_dose_localization_kind(
            patient_uid=patient_uid,
            patient_reference_dict=working_patient_reference_dict,
            config=config,
            num_mc_dose_simulations=num_mc_dose_simulations,
            localization_kind=MC_DOSE_LOCALIZATION_KIND_DOSE,
        )
        dose_gradient_biopsy_count = _run_patient_dose_localization_kind(
            patient_uid=patient_uid,
            patient_reference_dict=working_patient_reference_dict,
            config=config,
            num_mc_dose_simulations=num_mc_dose_simulations,
            localization_kind=MC_DOSE_LOCALIZATION_KIND_GRADIENT,
        )
        dose_dvh_biopsy_count = _run_patient_dose_dvh_stage(
            patient_reference_dict=working_patient_reference_dict,
            config=config,
            bx_sample_pts_volume_element=bx_sample_pts_volume_element,
        )
        metadata["num_mc_dose_simulations"] = num_mc_dose_simulations
        metadata["bx_sample_pts_volume_element"] = bx_sample_pts_volume_element

    containment_outputs, dose_outputs = collect_mc_patient_outputs(
        patient_uid,
        working_patient_reference_dict,
        bx_ref=config.keys.bx_ref,
    )
    metadata.update(
        {
            "containment_relative_structure_count": containment_relative_structure_count,
            "containment_relative_comparison_count": containment_relative_comparison_count,
            "dose_gradient_biopsy_count": dose_gradient_biopsy_count,
            "dose_dvh_biopsy_count": dose_dvh_biopsy_count,
        }
    )
    return PatientMCConvexStageResult(
        patient_uid=patient_uid,
        patient_reference_dict=working_patient_reference_dict,
        master_structure_reference_dict=master_structure_reference_dict,
        master_structure_info_dict=master_structure_info_dict,
        containment_outputs=containment_outputs,
        dose_outputs=dose_outputs,
        containment_biopsy_count=containment_biopsy_count,
        dose_biopsy_count=dose_biopsy_count,
        dose_reference_available=dose_reference_available,
        plan_reference_available=plan_reference_available,
        performed_flags=performed_flags,
        metadata=metadata,
    )
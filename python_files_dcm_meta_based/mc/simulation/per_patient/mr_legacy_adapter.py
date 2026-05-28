"""Patient-level adapter for the current MR MC simulator oracle."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from presentation import LegacyPresentationContext

from .contracts import MCMRPatientRunResult, MCMRSimulationConfig
from .convex_legacy_adapter import NullStopwatch
from .convex_legacy_adapter import _build_legacy_mc_layout_groups
from .convex_legacy_adapter import build_single_patient_mc_master_info
from .legacy_keys import legacy_mc_keys
from .mr import PatientMROutputs, collect_patient_mr_outputs


def collect_mc_mr_patient_outputs(patient_uid: str,
                                  patient_reference_dict: Mapping[str, Any],
                                  *,
                                  bx_ref: str) -> PatientMROutputs:
    """Collect patient MC MR outputs from legacy storage."""
    return collect_patient_mr_outputs(
        patient_uid,
        patient_reference_dict,
        bx_ref=bx_ref,
    )


def run_patient_mc_mr_legacy_adapter(
    *,
    patient_uid: str,
    patient_reference_dict: dict[str, Any],
    patient_info_dict: Mapping[str, Any] | None,
    structs_referenced_list: Sequence[str],
    bx_ref: str,
    mr_adc_ref: str,
    config: MCMRSimulationConfig,
    presentation_context: LegacyPresentationContext | None = None,
    stopwatch: Any = None,
    global_info: Mapping[str, Any] | None = None,
    mutate_input: bool = True,
) -> MCMRPatientRunResult:
    """Run the current MR MC simulator against a singleton patient cohort."""
    from MC_simulator_MR import simulator_parallel

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

    master_structure_reference_dict, master_structure_info_dict, live_display = simulator_parallel(
        context.live_display,
        layout_groups,
        master_structure_reference_dict,
        master_structure_info_dict,
        tuple(structs_referenced_list),
        mr_adc_ref,
        bx_ref,
        config.num_mr_calc_NN,
        config.idw_power,
        config.raw_data_mc_mr_dump_bool,
        config.show_NN_mr_adc_demonstration_plots,
        resolved_stopwatch,
        config.mr_views_jsons_paths_list,
        config.perform_mc_mr_sim,
        config.show_NN_mr_adc_demonstration_plots_all_trials_at_once,
    )
    mr_outputs = collect_mc_mr_patient_outputs(
        patient_uid,
        working_patient_reference_dict,
        bx_ref=bx_ref,
    )
    master_keys = legacy_mc_keys.master_info
    mc_info = master_structure_info_dict.get(master_keys.global_key, {}).get(master_keys.mc_info_key, {})
    return MCMRPatientRunResult(
        patient_uid=patient_uid,
        patient_reference_dict=working_patient_reference_dict,
        master_structure_reference_dict=master_structure_reference_dict,
        master_structure_info_dict=master_structure_info_dict,
        mr_outputs=mr_outputs,
        presentation_context=context,
        live_display=live_display,
        performed_flags={
            master_keys.mr_performed_key: mc_info.get(master_keys.mr_performed_key),
        },
        metadata={"mutated_input": bool(mutate_input)},
    )
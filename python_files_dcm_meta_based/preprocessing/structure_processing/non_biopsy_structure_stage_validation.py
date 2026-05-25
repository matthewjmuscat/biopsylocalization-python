import copy

from preprocessing.structure_processing.non_biopsy_structure_loop import build_standard_non_biopsy_structure_family_configs
from preprocessing.structure_processing.validation import append_non_biopsy_structure_validation_result
from preprocessing.structure_processing.validation import capture_non_biopsy_structure_processing_snapshot
from preprocessing.structure_processing.validation import compare_non_biopsy_structure_processing_snapshots


_OUTPUT_DATAFRAMES_KEY = "Multi-structure pre-processing output dataframes dict"


def _iter_stage_structure_ref_types(family_configs):
    seen_ref_types = set()
    for family_config in family_configs:
        struct_ref_type = family_config["struct_ref_type"]
        if struct_ref_type in seen_ref_types:
            continue
        seen_ref_types.add(struct_ref_type)
        yield struct_ref_type


def capture_standard_non_biopsy_stage_live_state(
        *,
        master_structure_reference_dict,
        family_configs,
        all_ref_key):
    patient_states = {}
    for patient_uid, pydicom_item in master_structure_reference_dict.items():
        patient_states[patient_uid] = {
            "structures": {
                struct_ref_type: copy.deepcopy(pydicom_item[struct_ref_type])
                for struct_ref_type in _iter_stage_structure_ref_types(family_configs)
            },
            "patient_output_dataframes_dict": copy.deepcopy(
                pydicom_item[all_ref_key][_OUTPUT_DATAFRAMES_KEY]
            ),
        }
    return patient_states


def restore_standard_non_biopsy_stage_live_state(
        *,
        master_structure_reference_dict,
        all_ref_key,
        saved_state):
    if saved_state is None:
        return

    for patient_uid, patient_state in saved_state.items():
        for struct_ref_type, structures_state in patient_state["structures"].items():
            master_structure_reference_dict[patient_uid][struct_ref_type] = structures_state
        master_structure_reference_dict[patient_uid][all_ref_key][_OUTPUT_DATAFRAMES_KEY] = (
            patient_state["patient_output_dataframes_dict"]
        )


def capture_standard_non_biopsy_stage_processing_snapshots(
        *,
        master_structure_reference_dict,
        family_configs,
        all_ref_key):
    snapshots = {}
    for patient_uid, pydicom_item in master_structure_reference_dict.items():
        for struct_ref_type in _iter_stage_structure_ref_types(family_configs):
            for specific_structure_index, _specific_structure in enumerate(pydicom_item[struct_ref_type]):
                snapshot_key = (patient_uid, struct_ref_type, specific_structure_index)
                snapshots[snapshot_key] = capture_non_biopsy_structure_processing_snapshot(
                    master_structure_reference_dict=master_structure_reference_dict,
                    patient_uid=patient_uid,
                    struct_ref_type=struct_ref_type,
                    specific_structure_index=specific_structure_index,
                    all_ref_key=all_ref_key,
                )
    return snapshots


def begin_standard_non_biopsy_structure_stage_legacy_validation(
        *,
        oar_ref,
        rectum_ref_key,
        urethra_ref_key,
        dil_ref,
        master_structure_reference_dict,
        config):
    family_configs = build_standard_non_biopsy_structure_family_configs(
        oar_ref=oar_ref,
        rectum_ref_key=rectum_ref_key,
        urethra_ref_key=urethra_ref_key,
        dil_ref=dil_ref,
    )
    pre_modular_live_state = capture_standard_non_biopsy_stage_live_state(
        master_structure_reference_dict=master_structure_reference_dict,
        family_configs=family_configs,
        all_ref_key=config.all_ref_key,
    )
    return {
        "family_configs": family_configs,
        "pre_modular_live_state": pre_modular_live_state,
    }


def prepare_standard_non_biopsy_structure_stage_legacy_validation(
        *,
        master_structure_reference_dict,
        all_ref_key,
        validation_context,
        live_display,
        runtime_logger=None):
    if validation_context is None:
        return live_display, None

    family_configs = validation_context["family_configs"]
    modular_validation_snapshots = capture_standard_non_biopsy_stage_processing_snapshots(
        master_structure_reference_dict=master_structure_reference_dict,
        family_configs=family_configs,
        all_ref_key=all_ref_key,
    )
    modular_live_state = capture_standard_non_biopsy_stage_live_state(
        master_structure_reference_dict=master_structure_reference_dict,
        family_configs=family_configs,
        all_ref_key=all_ref_key,
    )
    validation_context["modular_validation_snapshots"] = modular_validation_snapshots
    validation_context["modular_live_state"] = modular_live_state
    restore_standard_non_biopsy_stage_live_state(
        master_structure_reference_dict=master_structure_reference_dict,
        all_ref_key=all_ref_key,
        saved_state=validation_context["pre_modular_live_state"],
    )
    if runtime_logger is not None:
        runtime_logger.checkpoint(
            "preprocessing.structure.validation.prepare",
            "Captured modular non-biopsy stage output and restored pre-stage state for legacy sidecar validation.",
            details={"structure_family_count": len(family_configs)},
        )
    return live_display, validation_context


def finalize_standard_non_biopsy_structure_stage_legacy_validation(
        *,
        master_structure_reference_dict,
        all_ref_key,
        validation_context,
        important_info,
        live_display,
        runtime_logger):
    if validation_context is None:
        return live_display

    family_configs = validation_context["family_configs"]
    legacy_validation_snapshots = capture_standard_non_biopsy_stage_processing_snapshots(
        master_structure_reference_dict=master_structure_reference_dict,
        family_configs=family_configs,
        all_ref_key=all_ref_key,
    )
    validation_results = []
    for snapshot_key, modular_snapshot in validation_context[
        "modular_validation_snapshots"
    ].items():
        legacy_snapshot = legacy_validation_snapshots[snapshot_key]
        validation_results.append(
            compare_non_biopsy_structure_processing_snapshots(
                modular_snapshot,
                legacy_snapshot,
            )
        )

    restore_standard_non_biopsy_stage_live_state(
        master_structure_reference_dict=master_structure_reference_dict,
        all_ref_key=all_ref_key,
        saved_state=validation_context["modular_live_state"],
    )

    for validation_result in validation_results:
        append_non_biopsy_structure_validation_result(
            master_structure_reference_dict=master_structure_reference_dict,
            patient_uid=validation_result["patient_uid"],
            all_ref_key=all_ref_key,
            validation_result=validation_result,
        )
        if runtime_logger is not None:
            runtime_logger.checkpoint(
                "preprocessing.structure.validation",
                "Validated modular non-biopsy preprocessing stage against legacy inline path.",
                details={
                    "patient_uid": validation_result["patient_uid"],
                    "structure_id": validation_result["structure_id"],
                    "structure_type": validation_result["structure_type"],
                    "structure_index": validation_result["structure_index"],
                    "overall_match_bool": validation_result["overall_match_bool"],
                    "mismatch_count": validation_result["mismatch_count"],
                    "mismatch_fields": validation_result["mismatch_fields"][:10],
                },
            )
        if validation_result["overall_match_bool"] == False:
            mismatch_summary = validation_result["mismatch_summary"]
            if len(mismatch_summary) > 400:
                mismatch_summary = mismatch_summary[:400] + "..."
            important_info.add_text_line(
                f"WARNING! Modular non-biopsy preprocessing mismatch for patient {validation_result['patient_uid']}, structure {validation_result['structure_id']} ({validation_result['structure_type']}). {mismatch_summary}",
                live_display,
            )
    return live_display

"""Compatibility bridge from all-patient legacy dictionaries to one patient."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping

from .contracts import LegacyPatientRuntimeState
from .contracts import LegacyRuntimeKeys
from .contracts import PatientCase
from .contracts import _validate_patient_uids


def build_patient_case_from_legacy(patient_uid: str,
                                   master_structure_reference_dict: Mapping[str, Any],
                                   *,
                                   patient_label: str = "",
                                   source_run_id: str = "",
                                   input_manifest_id: str = "",
                                   metadata: Mapping[str, Any] | None = None) -> PatientCase:
    """Build a patient identity contract from the legacy patient registry."""
    resolved_patient_uid = _validate_patient_uids((patient_uid,), "patient_uid")[0]
    if resolved_patient_uid not in master_structure_reference_dict:
        raise KeyError(f"patient_uid not found in master_structure_reference_dict: {resolved_patient_uid}")
    return PatientCase(
        patient_uid=resolved_patient_uid,
        patient_label=patient_label or resolved_patient_uid,
        source_run_id=source_run_id,
        input_manifest_id=input_manifest_id,
        metadata=dict(metadata or {}),
    )


def carve_patient_runtime_state_by_uid(patient_uid: str,
                                       master_structure_reference_dict: MutableMapping[str, Any],
                                       master_structure_info_dict: MutableMapping[str, Any],
                                       *,
                                       legacy_keys: LegacyRuntimeKeys,
                                       patient_label: str = "",
                                       source_run_id: str = "",
                                       input_manifest_id: str = "",
                                       metadata: Mapping[str, Any] | None = None) -> LegacyPatientRuntimeState:
    """Create a patient-local runtime state directly from a legacy patient UID."""
    patient_case = build_patient_case_from_legacy(
        patient_uid,
        master_structure_reference_dict,
        patient_label=patient_label,
        source_run_id=source_run_id,
        input_manifest_id=input_manifest_id,
        metadata=metadata,
    )
    return carve_patient_runtime_state(
        patient_case,
        master_structure_reference_dict,
        master_structure_info_dict,
        legacy_keys=legacy_keys,
        metadata=metadata,
    )


def carve_patient_runtime_state(patient_case: PatientCase,
                                master_structure_reference_dict: MutableMapping[str, Any],
                                master_structure_info_dict: MutableMapping[str, Any],
                                *,
                                legacy_keys: LegacyRuntimeKeys,
                                metadata: Mapping[str, Any] | None = None) -> LegacyPatientRuntimeState:
    """Return a one-patient view of the current legacy runtime dictionaries.

    The bridge intentionally performs shallow slicing only. Heavy arrays,
    point-cloud objects, and dataframes remain shared with the caller until a
    stage-specific migration defines explicit copy/serialization semantics.
    """
    patient_uid = patient_case.patient_uid
    if patient_uid not in master_structure_reference_dict:
        raise KeyError(f"patient_uid not found in master_structure_reference_dict: {patient_uid}")

    patient_reference_view: dict[str, Any] = {
        patient_uid: master_structure_reference_dict[patient_uid],
    }
    patient_info_view = _build_patient_info_view(master_structure_info_dict, patient_uid, legacy_keys)
    return LegacyPatientRuntimeState(
        patient_case=patient_case,
        master_structure_reference_dict=patient_reference_view,
        master_structure_info_dict=patient_info_view,
        legacy_keys=legacy_keys,
        metadata=dict(metadata or {}),
    )


def _build_patient_info_view(master_structure_info_dict: Mapping[str, Any],
                             patient_uid: str,
                             legacy_keys: LegacyRuntimeKeys) -> dict[str, Any]:
    info_view = dict(master_structure_info_dict)

    by_patient = master_structure_info_dict.get(legacy_keys.by_patient_key)
    if isinstance(by_patient, Mapping):
        if patient_uid not in by_patient:
            raise KeyError(
                "patient_uid not found in master_structure_info_dict"
                f"[{legacy_keys.by_patient_key!r}]: {patient_uid}"
            )
        info_view[legacy_keys.by_patient_key] = {patient_uid: by_patient[patient_uid]}

    global_info = master_structure_info_dict.get(legacy_keys.global_key)
    if isinstance(global_info, Mapping):
        global_view = dict(global_info)
        if legacy_keys.global_num_cases_key in global_view:
            global_view[legacy_keys.global_num_cases_key] = 1
        info_view[legacy_keys.global_key] = global_view

    return info_view

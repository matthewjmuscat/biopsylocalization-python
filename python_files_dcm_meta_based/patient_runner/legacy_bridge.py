"""Compatibility bridge from all-patient legacy dictionaries to one patient."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping

from .contracts import LegacyPatientRuntimeState
from .contracts import LegacyRuntimeKeys
from .contracts import PatientCase


def build_patient_case_from_legacy(patient_uid: str,
                                   master_structure_reference_dict: Mapping[str, Any],
                                   *,
                                   patient_label: str = "",
                                   source_run_id: str = "",
                                   input_manifest_id: str = "",
                                   metadata: Mapping[str, Any] | None = None) -> PatientCase:
    """Build a patient identity contract from the legacy patient registry."""
    resolved_patient_uid = str(patient_uid).strip()
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
    patient_info_view = _build_patient_info_view(master_structure_info_dict, patient_uid)
    return LegacyPatientRuntimeState(
        patient_case=patient_case,
        master_structure_reference_dict=patient_reference_view,
        master_structure_info_dict=patient_info_view,
        legacy_keys=legacy_keys,
        metadata=dict(metadata or {}),
    )


def _build_patient_info_view(master_structure_info_dict: Mapping[str, Any],
                             patient_uid: str) -> dict[str, Any]:
    info_view = dict(master_structure_info_dict)

    by_patient = master_structure_info_dict.get("By patient")
    if isinstance(by_patient, Mapping):
        if patient_uid not in by_patient:
            raise KeyError(f"patient_uid not found in master_structure_info_dict['By patient']: {patient_uid}")
        info_view["By patient"] = {patient_uid: by_patient[patient_uid]}

    global_info = master_structure_info_dict.get("Global")
    if isinstance(global_info, Mapping):
        global_view = dict(global_info)
        if "Num cases" in global_view:
            global_view["Num cases"] = 1
        info_view["Global"] = global_view

    return info_view
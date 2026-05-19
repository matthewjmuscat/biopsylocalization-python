from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping, Optional, Sequence

import pydicom

from .dicom_routing_profile import DicomRoutingProfile
from .dicom_routing_profile import build_legacy_variseed_mim_routing_profile


INPUT_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_MANIFEST_DIR_NAME = "manifests"


@dataclass(frozen=True)
class InputManifestWriteResult:
    manifest_dir: Path
    summary_path: Path
    case_manifest_path: Path
    dicom_manifest_path: Path
    routing_profile_path: Path
    warnings_path: Path
    num_cases: int
    num_dicom_files: int
    warning_count: int

    def to_log_details(self) -> dict[str, Any]:
        return {
            "manifest_dir": str(self.manifest_dir),
            "summary_path": str(self.summary_path),
            "case_manifest_path": str(self.case_manifest_path),
            "dicom_manifest_path": str(self.dicom_manifest_path),
            "routing_profile_path": str(self.routing_profile_path),
            "warnings_path": str(self.warnings_path),
            "num_cases": self.num_cases,
            "num_dicom_files": self.num_dicom_files,
            "warning_count": self.warning_count,
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _path_key(path: Any) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def _path_for_csv(path: Any) -> str:
    if path is None:
        return ""
    return str(Path(path))


def _safe_getattr(dicom_dataset: Any, attribute_name: str, default: Any = None) -> Any:
    try:
        return getattr(dicom_dataset, attribute_name, default)
    except Exception:
        return default


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _build_generated_patient_uid(patient_name: Any, patient_id: Any) -> Optional[str]:
    if patient_name in (None, "") or patient_id in (None, ""):
        return None
    return f"{str(patient_name)} ({str(patient_id)})"


def _extract_number_from_string(value: str, allowed_prefixes: Sequence[str]) -> Optional[int]:
    if not value or not allowed_prefixes:
        return None
    prefix_pattern = "|".join(re.escape(prefix) for prefix in allowed_prefixes)
    match = re.search(rf"(?:{prefix_pattern})\s*(\d+)", value, re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))


def _iter_role_paths(role_dict: Mapping[str, Any]):
    for patient_uid, path_or_paths in role_dict.items():
        if isinstance(path_or_paths, (list, tuple, set)):
            for role_path in path_or_paths:
                yield patient_uid, role_path
        elif path_or_paths is not None:
            yield patient_uid, path_or_paths


def _register_role_paths(role_map: dict[str, dict[str, Any]], warnings: list[dict[str, Any]], role_name: str, role_dict: Mapping[str, Any]) -> None:
    for patient_uid, role_path in _iter_role_paths(role_dict):
        path_key = _path_key(role_path)
        if path_key in role_map:
            warnings.append({
                "warning_type": "duplicate_role_assignment",
                "patient_uid": str(patient_uid),
                "file_path": _path_for_csv(role_path),
                "message": "DICOM file was already assigned to a role in the manifest role map.",
                "existing_role": role_map[path_key]["selected_role"],
                "new_role": role_name,
            })
        role_map[path_key] = {
            "patient_uid": str(patient_uid),
            "selected_role": role_name,
        }


def _build_role_map(
    *,
    rtstruct_dcms_dict: Mapping[str, Any],
    rtdose_dcms_dict: Mapping[str, Any],
    rtplan_dcms_dict: Mapping[str, Any],
    us_dcms_dict: Mapping[str, Any],
    mr_t2_dcms_dict: Mapping[str, Any],
    mr_adc_dcms_dict: Mapping[str, Any],
    warnings: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    role_map: dict[str, dict[str, Any]] = {}
    _register_role_paths(role_map, warnings, "RTSTRUCT", rtstruct_dcms_dict)
    _register_role_paths(role_map, warnings, "RTDOSE", rtdose_dcms_dict)
    _register_role_paths(role_map, warnings, "RTPLAN", rtplan_dcms_dict)
    _register_role_paths(role_map, warnings, "US", us_dcms_dict)
    _register_role_paths(role_map, warnings, "MR_T2", mr_t2_dcms_dict)
    _register_role_paths(role_map, warnings, "MR_ADC", mr_adc_dcms_dict)
    return role_map


def _resolve_routing_reason(selected_role: str, modality: str, series_description: str, mr_acquisition_type: str) -> str:
    if selected_role == "RTSTRUCT":
        return "Modality == RTSTRUCT"
    if selected_role == "RTDOSE":
        return "Modality == RTDOSE"
    if selected_role == "RTPLAN":
        return "Modality == RTPLAN"
    if selected_role == "US" and modality == "US":
        return "Modality == US"
    if selected_role == "US" and modality == "MR" and mr_acquisition_type == "":
        return "Modality == MR and empty MRAcquisitionType fallback to US"
    if selected_role == "MR_T2":
        return "Modality == MR and SeriesDescription == T2"
    if selected_role == "MR_ADC":
        return "Modality == MR and SeriesDescription == ADC"
    if selected_role:
        return "Assigned by existing role dictionary"
    if modality == "MR" and series_description not in {"T2", "ADC"} and mr_acquisition_type != "":
        return "Unassigned MR file: SeriesDescription not T2/ADC and MRAcquisitionType not empty"
    return "Unassigned by current routing rules"


def _resolve_routing_rule_id(selected_role: str, modality: str, series_description: str, mr_acquisition_type: str) -> str:
    if selected_role == "RTSTRUCT":
        return "legacy_rtstruct_modality"
    if selected_role == "RTDOSE":
        return "legacy_rtdose_modality"
    if selected_role == "RTPLAN":
        return "legacy_rtplan_modality"
    if selected_role == "US" and modality == "US":
        return "legacy_us_modality"
    if selected_role == "US" and modality == "MR" and mr_acquisition_type == "":
        return "legacy_us_mr_empty_acquisition_type"
    if selected_role == "MR_T2":
        return "legacy_mr_t2_series_description"
    if selected_role == "MR_ADC":
        return "legacy_mr_adc_series_description"
    if selected_role:
        return "assigned_by_existing_role_dictionary"
    return "unassigned_by_current_routing_rules"


def _read_dicom_metadata(file_path: Path) -> dict[str, Any]:
    empty_metadata = {
        "patient_uid_generated": None,
        "patient_name": "",
        "patient_id": "",
        "modality": "",
        "series_description": "",
        "series_instance_uid": "",
        "study_instance_uid": "",
        "sop_instance_uid": "",
        "study_date": "",
        "mr_acquisition_type": "",
        "read_error": "",
    }
    try:
        dicom_dataset = pydicom.dcmread(file_path, defer_size="2 MB", stop_before_pixels=True)
    except Exception as exc:
        empty_metadata["read_error"] = repr(exc)
        return empty_metadata
    patient_name = _safe_getattr(dicom_dataset, "PatientName")
    patient_id = _safe_getattr(dicom_dataset, "PatientID")
    return {
        "patient_uid_generated": _build_generated_patient_uid(patient_name, patient_id),
        "patient_name": _safe_str(patient_name),
        "patient_id": _safe_str(patient_id),
        "modality": _safe_str(_safe_getattr(dicom_dataset, "Modality")),
        "series_description": _safe_str(_safe_getattr(dicom_dataset, "SeriesDescription")),
        "series_instance_uid": _safe_str(_safe_getattr(dicom_dataset, "SeriesInstanceUID")),
        "study_instance_uid": _safe_str(_safe_getattr(dicom_dataset, "StudyInstanceUID")),
        "sop_instance_uid": _safe_str(_safe_getattr(dicom_dataset, "SOPInstanceUID")),
        "study_date": _safe_str(_safe_getattr(dicom_dataset, "StudyDate")),
        "mr_acquisition_type": _safe_str(_safe_getattr(dicom_dataset, "MRAcquisitionType")),
        "read_error": "",
    }


def _build_dicom_manifest_rows(
    dicom_paths: Sequence[Any],
    role_map: Mapping[str, Mapping[str, Any]],
    routing_profile: DicomRoutingProfile,
    warnings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for discovery_index, dicom_path in enumerate(dicom_paths):
        file_path = Path(dicom_path)
        path_key = _path_key(file_path)
        metadata = _read_dicom_metadata(file_path)
        role_info = role_map.get(path_key, {})
        selected_role = _safe_str(role_info.get("selected_role"))
        role_patient_uid = role_info.get("patient_uid")
        generated_patient_uid = metadata["patient_uid_generated"]
        warning_messages = []
        if metadata["read_error"]:
            warning_messages.append("Could not read DICOM metadata while writing input manifest.")
            warnings.append({
                "warning_type": "dicom_metadata_read_error",
                "patient_uid": generated_patient_uid,
                "file_path": _path_for_csv(file_path),
                "message": "Could not read DICOM metadata while writing input manifest.",
                "error": metadata["read_error"],
            })
        if not selected_role:
            warning_messages.append("File was not assigned to a current input role.")
            warnings.append({
                "warning_type": "unassigned_dicom_file",
                "patient_uid": generated_patient_uid,
                "file_path": _path_for_csv(file_path),
                "message": "File was not assigned to a current input role.",
                "modality": metadata["modality"],
                "series_description": metadata["series_description"],
            })
        if role_patient_uid is not None and generated_patient_uid is not None and str(role_patient_uid) != str(generated_patient_uid):
            warning_messages.append("Role dictionary patient UID differs from DICOM metadata patient UID.")
            warnings.append({
                "warning_type": "patient_uid_mismatch",
                "patient_uid": generated_patient_uid,
                "role_patient_uid": str(role_patient_uid),
                "file_path": _path_for_csv(file_path),
                "message": "Role dictionary patient UID differs from DICOM metadata patient UID.",
            })
        if selected_role == "US" and metadata["modality"] == "MR":
            warning_messages.append("MR file routed to US by current fallback rule.")
            warnings.append({
                "warning_type": "mr_routed_to_us_fallback",
                "patient_uid": generated_patient_uid,
                "file_path": _path_for_csv(file_path),
                "message": "MR file routed to US by current fallback rule.",
                "series_description": metadata["series_description"],
                "mr_acquisition_type": metadata["mr_acquisition_type"],
            })

        rows.append({
            "Discovery index": discovery_index,
            "File path": _path_for_csv(file_path),
            "Patient UID (generated)": generated_patient_uid or "",
            "Patient UID (role dict)": _safe_str(role_patient_uid),
            "Patient Name": metadata["patient_name"],
            "Patient ID (from dicom)": metadata["patient_id"],
            "Modality": metadata["modality"],
            "Series Description": metadata["series_description"],
            "MRAcquisitionType": metadata["mr_acquisition_type"],
            "Series Instance UID": metadata["series_instance_uid"],
            "Study Instance UID": metadata["study_instance_uid"],
            "SOP Instance UID": metadata["sop_instance_uid"],
            "Study Date": metadata["study_date"],
            "Read error": metadata["read_error"],
            "Routing profile ID": routing_profile.profile_id,
            "Routing rule ID": _resolve_routing_rule_id(
                selected_role,
                metadata["modality"],
                metadata["series_description"],
                metadata["mr_acquisition_type"],
            ),
            "Selected role": selected_role,
            "Routing reason": _resolve_routing_reason(
                selected_role,
                metadata["modality"],
                metadata["series_description"],
                metadata["mr_acquisition_type"],
            ),
            "Warning bool": bool(warning_messages),
            "Warning messages": " | ".join(warning_messages),
        })
    return rows


def _first_metadata_for_patient(dicom_rows: Sequence[Mapping[str, Any]], patient_uid: str) -> Mapping[str, Any]:
    for row in dicom_rows:
        if row.get("Patient UID (generated)") == patient_uid or row.get("Patient UID (role dict)") == patient_uid:
            return row
    return {}


def _scalar_path(role_dict: Mapping[str, Any], patient_uid: str) -> str:
    role_path = role_dict.get(patient_uid)
    if role_path is None:
        return ""
    return _path_for_csv(role_path)


def _list_paths(role_dict: Mapping[str, Any], patient_uid: str) -> list[Any]:
    role_paths = role_dict.get(patient_uid, [])
    if role_paths is None:
        return []
    if isinstance(role_paths, (list, tuple, set)):
        return list(role_paths)
    return [role_paths]


def _build_case_manifest_rows(
    *,
    dicom_rows: Sequence[Mapping[str, Any]],
    rtstruct_dcms_dict: Mapping[str, Any],
    rtdose_dcms_dict: Mapping[str, Any],
    rtplan_dcms_dict: Mapping[str, Any],
    us_dcms_dict: Mapping[str, Any],
    mr_t2_dcms_dict: Mapping[str, Any],
    mr_adc_dcms_dict: Mapping[str, Any],
    fraction_prefixes: Sequence[str],
    warnings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    patient_uids = set(rtstruct_dcms_dict.keys())
    patient_uids.update(rtdose_dcms_dict.keys())
    patient_uids.update(rtplan_dcms_dict.keys())
    patient_uids.update(us_dcms_dict.keys())
    patient_uids.update(mr_t2_dcms_dict.keys())
    patient_uids.update(mr_adc_dcms_dict.keys())
    rows = []
    for patient_uid in sorted(str(uid) for uid in patient_uids):
        first_metadata = _first_metadata_for_patient(dicom_rows, patient_uid)
        patient_id = _safe_str(first_metadata.get("Patient ID (from dicom)"))
        has_rtstruct = patient_uid in rtstruct_dcms_dict
        has_rtdose = patient_uid in rtdose_dcms_dict
        has_rtplan = patient_uid in rtplan_dcms_dict
        core_complete = bool(has_rtstruct and has_rtdose and has_rtplan)
        if not core_complete:
            missing_roles = []
            if not has_rtstruct:
                missing_roles.append("RTSTRUCT")
            if not has_rtdose:
                missing_roles.append("RTDOSE")
            if not has_rtplan:
                missing_roles.append("RTPLAN")
            warnings.append({
                "warning_type": "case_missing_core_role",
                "patient_uid": patient_uid,
                "file_path": "",
                "message": "Case is missing one or more core roles.",
                "missing_roles": missing_roles,
            })

        us_paths = _list_paths(us_dcms_dict, patient_uid)
        mr_t2_paths = _list_paths(mr_t2_dcms_dict, patient_uid)
        mr_adc_paths = _list_paths(mr_adc_dcms_dict, patient_uid)
        rows.append({
            "Patient UID (generated)": patient_uid,
            "Patient Name": _safe_str(first_metadata.get("Patient Name")),
            "Patient ID (from dicom)": patient_id,
            "Fraction number (legacy parsed)": _extract_number_from_string(patient_id, fraction_prefixes),
            "Has RTSTRUCT": has_rtstruct,
            "Has RTDOSE": has_rtdose,
            "Has RTPLAN": has_rtplan,
            "Core RTSTRUCT/RTDOSE/RTPLAN complete": core_complete,
            "RTSTRUCT path": _scalar_path(rtstruct_dcms_dict, patient_uid),
            "RTDOSE path": _scalar_path(rtdose_dcms_dict, patient_uid),
            "RTPLAN path": _scalar_path(rtplan_dcms_dict, patient_uid),
            "Num US files": len(us_paths),
            "Num MR T2 files": len(mr_t2_paths),
            "Num MR ADC files": len(mr_adc_paths),
            "US paths": " | ".join(_path_for_csv(path) for path in us_paths),
            "MR T2 paths": " | ".join(_path_for_csv(path) for path in mr_t2_paths),
            "MR ADC paths": " | ".join(_path_for_csv(path) for path in mr_adc_paths),
        })
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _count_rows_by_value(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[_safe_str(row.get(key)) or "Unassigned"] += 1
    return dict(sorted(counts.items()))


def write_input_manifest_files(
    *,
    output_dir: Path,
    dicom_paths: Sequence[Any],
    rtstruct_dcms_dict: Mapping[str, Any],
    rtdose_dcms_dict: Mapping[str, Any],
    rtplan_dcms_dict: Mapping[str, Any],
    us_dcms_dict: Mapping[str, Any],
    mr_t2_dcms_dict: Mapping[str, Any],
    mr_adc_dcms_dict: Mapping[str, Any],
    fraction_prefixes: Sequence[str],
    routing_profile: Optional[DicomRoutingProfile] = None,
    runtime_logger: Any = None,
    manifest_dir_name: str = DEFAULT_MANIFEST_DIR_NAME,
) -> InputManifestWriteResult:
    start_time = time.perf_counter()
    if routing_profile is None:
        routing_profile = build_legacy_variseed_mim_routing_profile(fraction_prefixes)
    output_dir = Path(output_dir)
    manifest_dir = output_dir.joinpath(manifest_dir_name)
    summary_path = manifest_dir.joinpath("input_manifest_summary.json")
    case_manifest_path = manifest_dir.joinpath("input_case_manifest.csv")
    dicom_manifest_path = manifest_dir.joinpath("input_dicom_manifest.csv")
    routing_profile_path = manifest_dir.joinpath("input_routing_profile.json")
    warnings_path = manifest_dir.joinpath("input_manifest_warnings.jsonl")

    if runtime_logger is not None:
        runtime_logger.phase_start(
            "input.manifest",
            "Writing DICOM input manifest files.",
            details={
                "manifest_dir": str(manifest_dir),
                "num_dicom_paths": len(dicom_paths),
                "routing_profile_id": routing_profile.profile_id,
            },
        )

    warnings: list[dict[str, Any]] = []
    role_map = _build_role_map(
        rtstruct_dcms_dict=rtstruct_dcms_dict,
        rtdose_dcms_dict=rtdose_dcms_dict,
        rtplan_dcms_dict=rtplan_dcms_dict,
        us_dcms_dict=us_dcms_dict,
        mr_t2_dcms_dict=mr_t2_dcms_dict,
        mr_adc_dcms_dict=mr_adc_dcms_dict,
        warnings=warnings,
    )
    dicom_rows = _build_dicom_manifest_rows(dicom_paths, role_map, routing_profile, warnings)
    case_rows = _build_case_manifest_rows(
        dicom_rows=dicom_rows,
        rtstruct_dcms_dict=rtstruct_dcms_dict,
        rtdose_dcms_dict=rtdose_dcms_dict,
        rtplan_dcms_dict=rtplan_dcms_dict,
        us_dcms_dict=us_dcms_dict,
        mr_t2_dcms_dict=mr_t2_dcms_dict,
        mr_adc_dcms_dict=mr_adc_dcms_dict,
        fraction_prefixes=fraction_prefixes,
        warnings=warnings,
    )
    elapsed_seconds = time.perf_counter() - start_time
    summary = {
        "schema_version": INPUT_MANIFEST_SCHEMA_VERSION,
        "generated_utc": _utc_now_iso(),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "num_dicom_files": len(dicom_rows),
        "num_cases": len(case_rows),
        "warning_count": len(warnings),
        "routing_profile": {
            **routing_profile.to_summary_dict(),
            "path": str(routing_profile_path),
        },
        "file_counts_by_selected_role": _count_rows_by_value(dicom_rows, "Selected role"),
        "case_counts": {
            "rtstruct_patients": len(rtstruct_dcms_dict),
            "rtdose_patients": len(rtdose_dcms_dict),
            "rtplan_patients": len(rtplan_dcms_dict),
            "us_patients": len(us_dcms_dict),
            "mr_t2_patients": len(mr_t2_dcms_dict),
            "mr_adc_patients": len(mr_adc_dcms_dict),
        },
        "manifest_paths": {
            "summary": str(summary_path),
            "case_manifest": str(case_manifest_path),
            "dicom_manifest": str(dicom_manifest_path),
            "routing_profile": str(routing_profile_path),
            "warnings": str(warnings_path),
        },
    }

    manifest_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(dicom_manifest_path, dicom_rows)
    _write_csv(case_manifest_path, case_rows)
    _write_jsonl(warnings_path, warnings)
    with routing_profile_path.open("w", encoding="utf-8") as file_obj:
        json.dump(routing_profile.to_dict(), file_obj, indent=2, sort_keys=True, default=str)
        file_obj.write("\n")
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, sort_keys=True, default=str)
        file_obj.write("\n")

    result = InputManifestWriteResult(
        manifest_dir=manifest_dir,
        summary_path=summary_path,
        case_manifest_path=case_manifest_path,
        dicom_manifest_path=dicom_manifest_path,
        routing_profile_path=routing_profile_path,
        warnings_path=warnings_path,
        num_cases=len(case_rows),
        num_dicom_files=len(dicom_rows),
        warning_count=len(warnings),
    )
    if runtime_logger is not None:
        details = result.to_log_details()
        details["elapsed_seconds"] = round(elapsed_seconds, 6)
        runtime_logger.phase_end(
            "input.manifest",
            "Wrote DICOM input manifest files.",
            details=details,
        )
    return result

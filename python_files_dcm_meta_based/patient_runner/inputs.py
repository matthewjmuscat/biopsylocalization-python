"""Typed manifest-derived input paths for one standalone patient worker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from config.snapshots import canonical_sha256


PATIENT_INPUT_CASE_MANIFEST_COLUMNS = frozenset(
    {
        "Patient UID (generated)",
        "RTSTRUCT path",
        "RTDOSE path",
        "RTPLAN path",
        "US paths",
        "MR T2 paths",
        "MR ADC paths",
    }
)


@dataclass(frozen=True, slots=True)
class PatientInputPaths:
    """Resolved DICOM role paths needed to bootstrap one patient runtime.

    Paths are provenance and worker inputs, not loaded DICOM objects. Optional
    modality sequences may be empty. Current standalone preflight requires the
    RTSTRUCT, RTDOSE, and RTPLAN paths to exist before scientific execution.
    """

    patient_uid: str
    rtstruct: Path | None = None
    rtdose: Path | None = None
    rtplan: Path | None = None
    us: Sequence[Path] = ()
    mr_t2: Sequence[Path] = ()
    mr_adc: Sequence[Path] = ()

    def __post_init__(self) -> None:
        patient_uid = str(self.patient_uid)
        if patient_uid.strip() == "":
            raise ValueError("patient_uid cannot be empty")
        object.__setattr__(self, "patient_uid", patient_uid)
        for field_name in ("rtstruct", "rtdose", "rtplan"):
            object.__setattr__(self, field_name, _optional_path(getattr(self, field_name)))
        for field_name in ("us", "mr_t2", "mr_adc"):
            object.__setattr__(self, field_name, tuple(Path(path) for path in getattr(self, field_name)))

    @property
    def core_paths(self) -> Mapping[str, Path | None]:
        return {"rtstruct": self.rtstruct, "rtdose": self.rtdose, "rtplan": self.rtplan}

    @property
    def missing_core_roles(self) -> tuple[str, ...]:
        return tuple(
            role
            for role, path in self.core_paths.items()
            if path is None or not path.expanduser().is_file()
        )

    @property
    def core_paths_all_present(self) -> bool:
        return len(self.missing_core_roles) == 0

    @property
    def manifest_identity_sha256(self) -> str:
        """Fingerprint the resolved role/path assignment, not DICOM file bytes."""
        return canonical_sha256(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "patient_uid": self.patient_uid,
            "rtstruct": _path_text(self.rtstruct),
            "rtdose": _path_text(self.rtdose),
            "rtplan": _path_text(self.rtplan),
            "us": [path.as_posix() for path in self.us],
            "mr_t2": [path.as_posix() for path in self.mr_t2],
            "mr_adc": [path.as_posix() for path in self.mr_adc],
        }
        if include_identity:
            payload["manifest_identity_sha256"] = self.manifest_identity_sha256
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PatientInputPaths":
        inputs = cls(
            patient_uid=str(payload.get("patient_uid", "")),
            rtstruct=_optional_path(payload.get("rtstruct")),
            rtdose=_optional_path(payload.get("rtdose")),
            rtplan=_optional_path(payload.get("rtplan")),
            us=_path_sequence(payload.get("us", ()), "us"),
            mr_t2=_path_sequence(payload.get("mr_t2", ()), "mr_t2"),
            mr_adc=_path_sequence(payload.get("mr_adc", ()), "mr_adc"),
        )
        expected_identity = str(payload.get("manifest_identity_sha256", "")).strip()
        if expected_identity and expected_identity != inputs.manifest_identity_sha256:
            raise ValueError("patient input manifest identity does not match its role paths")
        return inputs

    @classmethod
    def from_case_manifest_row(cls, row: Mapping[str, Any]) -> "PatientInputPaths":
        """Build explicit role paths from one validated case-manifest row."""
        missing_columns = sorted(PATIENT_INPUT_CASE_MANIFEST_COLUMNS.difference(row.keys()))
        if missing_columns:
            raise ValueError("input case manifest row is missing required columns: {}".format(missing_columns))
        return cls(
            patient_uid=str(row.get("Patient UID (generated)", "")),
            rtstruct=_optional_path(row.get("RTSTRUCT path")),
            rtdose=_optional_path(row.get("RTDOSE path")),
            rtplan=_optional_path(row.get("RTPLAN path")),
            us=_split_manifest_paths(row.get("US paths")),
            mr_t2=_split_manifest_paths(row.get("MR T2 paths")),
            mr_adc=_split_manifest_paths(row.get("MR ADC paths")),
        )


def _optional_path(value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value).strip()).expanduser()


def _path_text(path: Path | None) -> str:
    return "" if path is None else path.as_posix()


def _path_sequence(value: Any, field_name: str) -> tuple[Path, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("{} must be a sequence of paths".format(field_name))
    return tuple(Path(str(path)).expanduser() for path in value)


def _split_manifest_paths(value: Any) -> tuple[Path, ...]:
    return tuple(
        Path(path_text.strip()).expanduser()
        for path_text in str(value or "").split("|")
        if path_text.strip()
    )


__all__ = ["PATIENT_INPUT_CASE_MANIFEST_COLUMNS", "PatientInputPaths"]

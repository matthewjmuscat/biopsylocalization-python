"""Typed contracts for patient-local execution.

These contracts intentionally describe orchestration state only. Scientific
arrays and legacy dictionaries remain in the existing modules until each stage is
migrated behind a patient-local boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


DEFAULT_ALL_REF_KEY = "All ref"
DEFAULT_BX_REF = "Bx ref"


class PatientStageName(str, Enum):
    """Closed names for initial patient-runner stages."""

    PREPROCESSING = "preprocessing"
    OPTIMIZATION = "optimization"
    SIMULATED_BIOPSY_FINALIZATION = "simulated_biopsy_finalization"
    SAMPLING_CLASSIFICATION = "sampling_classification"
    MC_SIMULATION = "mc_simulation"
    PATIENT_ARTIFACT_WRITING = "patient_artifact_writing"


class PatientStageStatus(str, Enum):
    """Stable status values used in run summaries and logs."""

    NOT_STARTED = "not_started"
    SKIPPED = "skipped"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


def _safe_path_name(value: str) -> str:
    safe = str(value).strip().replace("/", "_").replace("\\", "_")
    for old, new in ((" ", "_"), (":", "_"), (",", "_")):
        safe = safe.replace(old, new)
    return safe or "unknown"


def _stage_name_value(stage_name: PatientStageName | str) -> str:
    if isinstance(stage_name, PatientStageName):
        return stage_name.value
    return str(stage_name)


@dataclass(frozen=True, slots=True)
class PatientCase:
    """Identity and provenance for one patient run.

    This object should stay small. Heavy DICOM, structure, or dataframe state
    belongs in ``PatientRuntimeState``/legacy bridges, not in the patient identity
    contract.
    """

    patient_uid: str
    patient_label: str = ""
    source_run_id: str = ""
    input_manifest_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        patient_uid = str(self.patient_uid).strip()
        if patient_uid == "":
            raise ValueError("patient_uid cannot be empty")
        object.__setattr__(self, "patient_uid", patient_uid)
        object.__setattr__(self, "patient_label", str(self.patient_label).strip())
        object.__setattr__(self, "source_run_id", str(self.source_run_id).strip())
        object.__setattr__(self, "input_manifest_id", str(self.input_manifest_id).strip())
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def safe_patient_uid(self) -> str:
        """Return a filesystem-safe patient identifier."""
        return _safe_path_name(self.patient_uid)


@dataclass(frozen=True, slots=True)
class PatientRunConfig:
    """Configuration slice needed by the initial patient runner scaffold."""

    output_root: Path
    all_ref_key: str = DEFAULT_ALL_REF_KEY
    bx_ref: str = DEFAULT_BX_REF
    run_id: str = ""
    write_preprocessing_artifacts: bool = True
    write_patient_mc_artifacts: bool = True
    write_biopsy_mc_artifacts: bool = True
    csv_index: bool = False
    parquet_index: bool = False
    parquet_compression: str = "snappy"
    stop_on_stage_error: bool = True
    raise_on_stage_error: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "all_ref_key", str(self.all_ref_key).strip() or DEFAULT_ALL_REF_KEY)
        object.__setattr__(self, "bx_ref", str(self.bx_ref).strip() or DEFAULT_BX_REF)
        object.__setattr__(self, "run_id", str(self.run_id).strip())
        object.__setattr__(self, "parquet_compression", str(self.parquet_compression).strip() or "snappy")

    def patient_output_dir(self, patient_case: PatientCase) -> Path:
        """Return the run-local output directory for one patient."""
        return self.output_root.joinpath("patients", patient_case.safe_patient_uid)


@dataclass(slots=True)
class LegacyPatientRuntimeState:
    """Patient-local view of the existing legacy runtime dictionaries."""

    patient_case: PatientCase
    master_structure_reference_dict: MutableMapping[str, Any]
    master_structure_info_dict: MutableMapping[str, Any]
    all_ref_key: str = DEFAULT_ALL_REF_KEY
    bx_ref: str = DEFAULT_BX_REF
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.patient_case.patient_uid not in self.master_structure_reference_dict:
            raise KeyError(f"patient_uid not present in runtime state: {self.patient_case.patient_uid}")
        self.all_ref_key = str(self.all_ref_key).strip() or DEFAULT_ALL_REF_KEY
        self.bx_ref = str(self.bx_ref).strip() or DEFAULT_BX_REF
        self.metadata = dict(self.metadata)

    @property
    def patient_uid(self) -> str:
        return self.patient_case.patient_uid

    @property
    def pydicom_item(self) -> MutableMapping[str, Any]:
        """Return this patient's legacy structure/runtime dictionary."""
        return self.master_structure_reference_dict[self.patient_uid]

    @property
    def biopsy_structures(self) -> Sequence[Any]:
        """Return this patient's legacy biopsy structure list."""
        return self.pydicom_item[self.bx_ref]


@dataclass(frozen=True, slots=True)
class PatientStageResult:
    """Result summary for one patient-runner stage."""

    stage_name: str
    status: PatientStageStatus
    elapsed_seconds: float = 0.0
    artifact_count: int = 0
    output_paths: tuple[Path, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_name", _stage_name_value(self.stage_name))
        object.__setattr__(self, "status", PatientStageStatus(self.status))
        object.__setattr__(self, "elapsed_seconds", float(self.elapsed_seconds))
        object.__setattr__(self, "artifact_count", int(self.artifact_count))
        object.__setattr__(self, "output_paths", tuple(Path(path) for path in self.output_paths))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def succeeded(self) -> bool:
        return self.status == PatientStageStatus.SUCCEEDED

    def with_elapsed_seconds(self, elapsed_seconds: float) -> "PatientStageResult":
        return replace(self, elapsed_seconds=float(elapsed_seconds))

    @classmethod
    def success(cls,
                stage_name: PatientStageName | str,
                *,
                elapsed_seconds: float = 0.0,
                artifact_count: int = 0,
                output_paths: Sequence[Path] = (),
                warnings: Sequence[str] = (),
                metadata: Mapping[str, Any] | None = None) -> "PatientStageResult":
        return cls(
            stage_name=_stage_name_value(stage_name),
            status=PatientStageStatus.SUCCEEDED,
            elapsed_seconds=elapsed_seconds,
            artifact_count=artifact_count,
            output_paths=tuple(output_paths),
            warnings=tuple(warnings),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def skipped(cls,
                stage_name: PatientStageName | str,
                *,
                elapsed_seconds: float = 0.0,
                reason: str = "",
                metadata: Mapping[str, Any] | None = None) -> "PatientStageResult":
        resolved_metadata = dict(metadata or {})
        if reason:
            resolved_metadata["skip_reason"] = reason
        return cls(
            stage_name=_stage_name_value(stage_name),
            status=PatientStageStatus.SKIPPED,
            elapsed_seconds=elapsed_seconds,
            metadata=resolved_metadata,
        )

    @classmethod
    def failure(cls,
                stage_name: PatientStageName | str,
                *,
                elapsed_seconds: float = 0.0,
                exception: BaseException | None = None,
                warnings: Sequence[str] = (),
                metadata: Mapping[str, Any] | None = None) -> "PatientStageResult":
        resolved_metadata = dict(metadata or {})
        resolved_warnings = list(warnings)
        if exception is not None:
            resolved_metadata["exception_type"] = type(exception).__name__
            resolved_metadata["exception_message"] = str(exception)
            resolved_warnings.append(f"{type(exception).__name__}: {exception}")
        return cls(
            stage_name=_stage_name_value(stage_name),
            status=PatientStageStatus.FAILED,
            elapsed_seconds=elapsed_seconds,
            warnings=tuple(resolved_warnings),
            metadata=resolved_metadata,
        )


@dataclass(frozen=True, slots=True)
class PatientRunResult:
    """Run-level summary for one patient case."""

    patient_case: PatientCase
    status: PatientStageStatus
    output_root: Path
    elapsed_seconds: float = 0.0
    stage_results: tuple[PatientStageResult, ...] = ()
    artifact_paths: tuple[Path, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", PatientStageStatus(self.status))
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "elapsed_seconds", float(self.elapsed_seconds))
        object.__setattr__(self, "stage_results", tuple(self.stage_results))
        object.__setattr__(self, "artifact_paths", tuple(Path(path) for path in self.artifact_paths))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def succeeded(self) -> bool:
        return self.status == PatientStageStatus.SUCCEEDED

    @property
    def failed_stage_results(self) -> tuple[PatientStageResult, ...]:
        return tuple(result for result in self.stage_results if result.status == PatientStageStatus.FAILED)

    @classmethod
    def from_stage_results(cls,
                           patient_case: PatientCase,
                           output_root: Path,
                           stage_results: Sequence[PatientStageResult],
                           *,
                           elapsed_seconds: float = 0.0,
                           metadata: Mapping[str, Any] | None = None) -> "PatientRunResult":
        resolved_stage_results = tuple(stage_results)
        failed = any(result.status == PatientStageStatus.FAILED for result in resolved_stage_results)
        status = PatientStageStatus.FAILED if failed else PatientStageStatus.SUCCEEDED
        artifact_paths: list[Path] = []
        for result in resolved_stage_results:
            artifact_paths.extend(result.output_paths)
        return cls(
            patient_case=patient_case,
            status=status,
            output_root=output_root,
            elapsed_seconds=elapsed_seconds,
            stage_results=resolved_stage_results,
            artifact_paths=tuple(artifact_paths),
            metadata=dict(metadata or {}),
        )
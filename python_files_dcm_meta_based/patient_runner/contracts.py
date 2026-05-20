"""Typed contracts for patient-local execution.

These contracts intentionally describe orchestration state only. Scientific
arrays and legacy dictionaries remain in the existing modules until each stage is
migrated behind a patient-local boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


class PatientStageName(str, Enum):
    """Closed names for initial patient-runner stages."""

    LEGACY_BRIDGE = "legacy_bridge"
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


class PatientBatchExecutionBackend(str, Enum):
    """Execution backends currently supported by the patient batch runner."""

    SEQUENTIAL = "sequential"
    THREAD = "thread"


@dataclass(frozen=True, slots=True)
class LegacyRuntimeKeys:
    """Legacy dictionary key names supplied by the current pipeline config.

    The patient runner must not define these names independently from the main
    pipeline. The caller should build this from the same bootstrap/config state
    that created the legacy dictionaries.
    """

    all_ref_key: str
    bx_ref: str
    by_patient_key: str
    global_key: str
    global_num_cases_key: str

    def __post_init__(self) -> None:
        for legacy_key_field in fields(self):
            field_name = legacy_key_field.name
            field_value = str(getattr(self, field_name)).strip()
            if field_value == "":
                raise ValueError(f"{field_name} cannot be empty")
            object.__setattr__(self, field_name, field_value)


def _safe_path_name(value: str) -> str:
    safe = str(value).strip().replace("/", "_").replace("\\", "_")
    for old, new in ((" ", "_"), (":", "_"), (",", "_")):
        safe = safe.replace(old, new)
    return safe or "unknown"


def _stage_name_value(stage_name: PatientStageName | str) -> str:
    if isinstance(stage_name, PatientStageName):
        return stage_name.value
    return str(stage_name)


def validate_patient_uids(patient_uids: Sequence[Any], source_name: str) -> tuple[str, ...]:
    """Validate patient IDs without changing their dictionary identity values."""
    validated_patient_uids = tuple(patient_uids)
    if any(not isinstance(patient_uid, str) for patient_uid in validated_patient_uids):
        raise TypeError(f"{source_name} entries must be strings")
    if any(patient_uid.strip() == "" for patient_uid in validated_patient_uids):
        raise ValueError(f"{source_name} cannot contain empty patient IDs")
    if len(set(validated_patient_uids)) != len(validated_patient_uids):
        raise ValueError(f"{source_name} cannot contain duplicates")
    return validated_patient_uids


def resolve_legacy_patient_uids(master_structure_reference_dict: Mapping[str, Any],
                                patient_uids: Sequence[str] = ()) -> tuple[str, ...]:
    """Resolve exact patient IDs from a legacy patient registry.

    An empty requested list means every patient key in registry order. Requested
    IDs are validated and checked for exact membership; they are not stripped,
    case-folded, slugified, or otherwise rewritten.
    """
    requested_patient_uids = validate_patient_uids(patient_uids, "patient_uids")

    if requested_patient_uids:
        missing_patient_uids = tuple(
            patient_uid
            for patient_uid in requested_patient_uids
            if patient_uid not in master_structure_reference_dict
        )
        if missing_patient_uids:
            raise KeyError(
                "patient_uids not found in master_structure_reference_dict: "
                f"{missing_patient_uids}"
            )
        return requested_patient_uids

    return validate_patient_uids(
        tuple(master_structure_reference_dict.keys()),
        "master_structure_reference_dict",
    )


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
        if not isinstance(self.patient_uid, str):
            raise TypeError("patient_uid must be a string")
        patient_uid = self.patient_uid
        if patient_uid.strip() == "":
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
    legacy_keys: LegacyRuntimeKeys
    run_id: str = ""
    write_preprocessing_artifacts: bool = True
    write_patient_mc_artifacts: bool = True
    write_biopsy_mc_artifacts: bool = True
    csv_index: bool = False
    parquet_index: bool = False
    parquet_compression: str = "snappy"
    write_patient_run_manifest: bool = True
    stop_on_stage_error: bool = True
    raise_on_stage_error: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        if not isinstance(self.legacy_keys, LegacyRuntimeKeys):
            raise TypeError("legacy_keys must be a LegacyRuntimeKeys instance")
        object.__setattr__(self, "run_id", str(self.run_id).strip())
        object.__setattr__(self, "parquet_compression", str(self.parquet_compression).strip() or "snappy")
        object.__setattr__(self, "write_patient_run_manifest", bool(self.write_patient_run_manifest))

    @property
    def all_ref_key(self) -> str:
        return self.legacy_keys.all_ref_key

    @property
    def bx_ref(self) -> str:
        return self.legacy_keys.bx_ref

    def patient_output_dir(self, patient_case: PatientCase) -> Path:
        """Return the run-local output directory for one patient."""
        return self.output_root.joinpath("patients", patient_case.safe_patient_uid)


@dataclass(frozen=True, slots=True)
class PatientBatchRunConfig:
    """Configuration for a cohort-level batch of patient-local runs.

    This wraps `PatientRunConfig` rather than repeating its fields. The batch
    layer owns only batch selection and scheduling policy. Sequential execution
    is the default reference path; thread execution must be requested explicitly.
    """

    patient_config: PatientRunConfig
    patient_uids: Sequence[str] = ()
    max_workers: int = 1
    execution_backend: PatientBatchExecutionBackend = PatientBatchExecutionBackend.SEQUENTIAL
    write_batch_run_manifest: bool = True
    patient_labels: Mapping[str, str] = field(default_factory=dict)
    source_run_id: str = ""
    input_manifest_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.patient_config, PatientRunConfig):
            raise TypeError("patient_config must be a PatientRunConfig instance")
        patient_uids = validate_patient_uids(self.patient_uids, "patient_uids")
        max_workers = int(self.max_workers)
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        execution_backend = PatientBatchExecutionBackend(self.execution_backend)
        patient_label_uids = validate_patient_uids(tuple(self.patient_labels.keys()), "patient_labels")
        patient_labels = {
            patient_uid: str(self.patient_labels[patient_uid]).strip()
            for patient_uid in patient_label_uids
        }

        object.__setattr__(self, "patient_uids", patient_uids)
        object.__setattr__(self, "max_workers", max_workers)
        object.__setattr__(self, "execution_backend", execution_backend)
        object.__setattr__(self, "write_batch_run_manifest", bool(self.write_batch_run_manifest))
        object.__setattr__(self, "patient_labels", patient_labels)
        object.__setattr__(self, "source_run_id", str(self.source_run_id).strip())
        object.__setattr__(self, "input_manifest_id", str(self.input_manifest_id).strip())
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def output_root(self) -> Path:
        return self.patient_config.output_root

    @property
    def legacy_keys(self) -> LegacyRuntimeKeys:
        return self.patient_config.legacy_keys

    @property
    def run_id(self) -> str:
        return self.patient_config.run_id


@dataclass(slots=True)
class LegacyCohortRuntimeState:
    """Typed boundary around the legacy all-patient runtime dictionaries.

    This is a transitional wrapper. It names the compatibility boundary while
    stages are migrated away from raw all-patient dictionaries.
    """

    master_structure_reference_dict: MutableMapping[str, Any]
    master_structure_info_dict: MutableMapping[str, Any]
    legacy_keys: LegacyRuntimeKeys
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.legacy_keys, LegacyRuntimeKeys):
            raise TypeError("legacy_keys must be a LegacyRuntimeKeys instance")
        resolve_legacy_patient_uids(self.master_structure_reference_dict)
        self.metadata = dict(self.metadata)

    @property
    def patient_uids(self) -> tuple[str, ...]:
        return resolve_legacy_patient_uids(self.master_structure_reference_dict)

    @property
    def patient_count(self) -> int:
        return len(self.patient_uids)

    def resolve_patient_uids(self, patient_uids: Sequence[str] = ()) -> tuple[str, ...]:
        return resolve_legacy_patient_uids(self.master_structure_reference_dict, patient_uids)


@dataclass(slots=True)
class LegacyPatientRuntimeState:
    """Patient-local view of the existing legacy runtime dictionaries."""

    patient_case: PatientCase
    master_structure_reference_dict: MutableMapping[str, Any]
    master_structure_info_dict: MutableMapping[str, Any]
    legacy_keys: LegacyRuntimeKeys
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.patient_case.patient_uid not in self.master_structure_reference_dict:
            raise KeyError(f"patient_uid not present in runtime state: {self.patient_case.patient_uid}")
        if not isinstance(self.legacy_keys, LegacyRuntimeKeys):
            raise TypeError("legacy_keys must be a LegacyRuntimeKeys instance")
        self.metadata = dict(self.metadata)

    @property
    def all_ref_key(self) -> str:
        return self.legacy_keys.all_ref_key

    @property
    def bx_ref(self) -> str:
        return self.legacy_keys.bx_ref

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


@dataclass(frozen=True, slots=True)
class PatientBatchRunResult:
    """Run-level summary for a batch of patient-local cases."""

    status: PatientStageStatus
    output_root: Path
    patient_results: tuple[PatientRunResult, ...] = ()
    elapsed_seconds: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", PatientStageStatus(self.status))
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "patient_results", tuple(self.patient_results))
        object.__setattr__(self, "elapsed_seconds", float(self.elapsed_seconds))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def succeeded(self) -> bool:
        return self.status == PatientStageStatus.SUCCEEDED

    @property
    def patient_count(self) -> int:
        return len(self.patient_results)

    @property
    def artifact_paths(self) -> tuple[Path, ...]:
        artifact_paths: list[Path] = []
        for patient_result in self.patient_results:
            artifact_paths.extend(patient_result.artifact_paths)
        return tuple(artifact_paths)

    @property
    def failed_patient_results(self) -> tuple[PatientRunResult, ...]:
        return tuple(
            patient_result
            for patient_result in self.patient_results
            if patient_result.status == PatientStageStatus.FAILED
        )

    @classmethod
    def from_patient_results(cls,
                             output_root: Path,
                             patient_results: Sequence[PatientRunResult],
                             *,
                             elapsed_seconds: float = 0.0,
                             metadata: Mapping[str, Any] | None = None) -> "PatientBatchRunResult":
        resolved_patient_results = tuple(patient_results)
        if not resolved_patient_results:
            status = PatientStageStatus.SKIPPED
        elif any(patient_result.status == PatientStageStatus.FAILED for patient_result in resolved_patient_results):
            status = PatientStageStatus.FAILED
        else:
            status = PatientStageStatus.SUCCEEDED
        return cls(
            status=status,
            output_root=output_root,
            patient_results=resolved_patient_results,
            elapsed_seconds=elapsed_seconds,
            metadata=dict(metadata or {}),
        )

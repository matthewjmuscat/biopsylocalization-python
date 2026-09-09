"""Human-authored orchestration profiles for standalone patient runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback when tomli is installed.
    try:
        import tomli as tomllib
    except ModuleNotFoundError:  # pragma: no cover - handled at load time.
        tomllib = None

from .process_runner import PATIENT_ARTIFACT_RETENTION_LEVELS
from .process_runner import PATIENT_PROCESS_EXECUTION_MODES
from .process_runner import PATIENT_PROCESS_REQUESTED_JOB_NAMES
from .process_runner import PatientProcessFailurePolicy
from .process_runner import PatientProcessRunPlan
from .process_runner import build_patient_process_run_plan


PATIENT_ORCHESTRATION_PROFILE_SCHEMA_VERSION = "patient_orchestration_profile_v1"
PATIENT_ORCHESTRATION_EXECUTION_MODES = PATIENT_PROCESS_EXECUTION_MODES
_PROFILE_FIELDS = {
    "": frozenset(
        {
            "schema_version",
            "description",
            "enabled",
            "run",
            "inputs",
            "selection",
            "execution",
            "artifacts",
            "scientific_config",
            "metadata",
        }
    ),
    "run": frozenset({"run_id", "output_root", "pathway", "checkpoint"}),
    "inputs": frozenset({"case_manifest"}),
    "selection": frozenset({"patient_uids"}),
    "execution": frozenset({"mode", "requested_jobs", "failure_policy", "max_workers", "timeout_seconds"}),
    "artifacts": frozenset({"retention_level"}),
    "scientific_config": frozenset({"snapshot"}),
}


@dataclass(frozen=True, slots=True)
class PatientOrchestrationProfile:
    """Resolved orchestration choices from a human-authored TOML profile.

    Scientific parameters deliberately do not live here. The optional scientific
    config snapshot identifies the typed PipelineConfig provenance that a future
    live worker will load after the standalone runtime builder is available.
    """

    source_path: Path
    input_case_manifest_path: Path
    output_root: Path
    pathway_name: str
    checkpoint_name: str
    run_id: str
    patient_uids: Sequence[str] = ()
    requested_jobs: Sequence[str] = ("standalone_patient_runner",)
    execution_mode: str = "plan_only"
    failure_policy: PatientProcessFailurePolicy | str = PatientProcessFailurePolicy.STOP_ON_FAILURE
    max_workers: int = 1
    timeout_seconds: float | None = None
    retention_level: str = "minimal"
    scientific_config_snapshot_path: Path | None = None
    description: str = ""
    enabled: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = PATIENT_ORCHESTRATION_PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PATIENT_ORCHESTRATION_PROFILE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported patient orchestration profile schema_version {!r}; expected {!r}".format(
                    self.schema_version,
                    PATIENT_ORCHESTRATION_PROFILE_SCHEMA_VERSION,
                )
            )
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "input_case_manifest_path", Path(self.input_case_manifest_path))
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "pathway_name", _non_empty_string(self.pathway_name, "pathway_name"))
        object.__setattr__(self, "checkpoint_name", _non_empty_string(self.checkpoint_name, "checkpoint_name"))
        object.__setattr__(self, "run_id", _non_empty_string(self.run_id, "run_id"))
        object.__setattr__(self, "patient_uids", _non_empty_unique_strings(self.patient_uids, "patient_uids"))
        requested_jobs = _non_empty_unique_strings(self.requested_jobs, "requested_jobs")
        if not requested_jobs:
            raise ValueError("requested_jobs cannot be empty")
        unsupported_jobs = sorted(set(requested_jobs).difference(PATIENT_PROCESS_REQUESTED_JOB_NAMES))
        if unsupported_jobs:
            raise ValueError("unsupported requested_jobs: {}".format(unsupported_jobs))
        object.__setattr__(self, "requested_jobs", requested_jobs)
        execution_mode = _non_empty_string(self.execution_mode, "execution_mode").lower()
        if execution_mode not in PATIENT_ORCHESTRATION_EXECUTION_MODES:
            raise ValueError(
                "execution_mode must be one of: {}".format(", ".join(sorted(PATIENT_ORCHESTRATION_EXECUTION_MODES)))
            )
        object.__setattr__(self, "execution_mode", execution_mode)
        object.__setattr__(self, "failure_policy", PatientProcessFailurePolicy(self.failure_policy))
        max_workers = int(self.max_workers)
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        object.__setattr__(self, "max_workers", max_workers)
        timeout_seconds = self.timeout_seconds
        if timeout_seconds is not None:
            timeout_seconds = float(timeout_seconds)
            if timeout_seconds <= 0:
                raise ValueError("timeout_seconds must be positive when provided")
        object.__setattr__(self, "timeout_seconds", timeout_seconds)
        retention_level = _non_empty_string(self.retention_level, "retention_level").lower()
        if retention_level not in PATIENT_ARTIFACT_RETENTION_LEVELS:
            raise ValueError(
                "retention_level must be one of: {}".format(", ".join(sorted(PATIENT_ARTIFACT_RETENTION_LEVELS)))
            )
        object.__setattr__(self, "retention_level", retention_level)
        if self.scientific_config_snapshot_path is not None:
            object.__setattr__(self, "scientific_config_snapshot_path", Path(self.scientific_config_snapshot_path))
            if not self.scientific_config_snapshot_path.is_file():
                raise FileNotFoundError(
                    "scientific config snapshot does not exist: {}".format(self.scientific_config_snapshot_path)
                )
        object.__setattr__(self, "description", str(self.description).strip())
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "metadata", dict(self.metadata))
        _validate_checkpoint_pathway(self.checkpoint_name, self.pathway_name)

    @property
    def source_fingerprint_sha256(self) -> str:
        return _sha256_file(self.source_path)

    @property
    def scientific_config_snapshot_fingerprint_sha256(self) -> str:
        if self.scientific_config_snapshot_path is None or not self.scientific_config_snapshot_path.is_file():
            return ""
        return _sha256_file(self.scientific_config_snapshot_path)

    @property
    def input_case_manifest_fingerprint_sha256(self) -> str:
        return _sha256_file(self.input_case_manifest_path)

    def build_process_run_plan(self) -> PatientProcessRunPlan:
        """Compile this profile into the existing standalone process plan."""
        return build_patient_process_run_plan(
            input_case_manifest_path=self.input_case_manifest_path,
            output_root=self.output_root,
            pathway_name=self.pathway_name,
            checkpoint_name=self.checkpoint_name,
            patient_uids=self.patient_uids,
            run_id=self.run_id,
            failure_policy=self.failure_policy,
            max_workers=self.max_workers,
            timeout_seconds=self.timeout_seconds,
            execution_mode=self.execution_mode,
            requested_jobs=self.requested_jobs,
            scientific_config_snapshot_path=self.scientific_config_snapshot_path,
            retention_level=self.retention_level,
            metadata={
                "profile_schema_version": self.schema_version,
                "profile_source_path": self.source_path.as_posix(),
                "profile_source_format": "toml",
                "profile_source_fingerprint_sha256": self.source_fingerprint_sha256,
                "profile_description": self.description,
                "profile_execution_mode": self.execution_mode,
                "input_case_manifest_fingerprint_sha256": self.input_case_manifest_fingerprint_sha256,
                "scientific_config_snapshot_fingerprint_sha256": (
                    self.scientific_config_snapshot_fingerprint_sha256
                ),
                **self.metadata,
            },
        )


def load_patient_orchestration_profile(profile_path: Path | str) -> PatientOrchestrationProfile:
    """Load and strictly validate one standalone orchestration TOML profile."""
    resolved_path = Path(profile_path).expanduser().resolve()
    if resolved_path.suffix.lower() != ".toml":
        raise ValueError("patient orchestration profiles must use .toml")
    if tomllib is None:
        raise RuntimeError("TOML profiles require Python 3.11+ tomllib support or the tomli package")
    with resolved_path.open("rb") as profile_file:
        payload = tomllib.load(profile_file)
    if not isinstance(payload, dict):
        raise TypeError("patient orchestration profile root must be a TOML table")
    _reject_unknown_fields(payload, _PROFILE_FIELDS[""], "profile root")

    run = _mapping(payload.get("run", {}), "run")
    inputs = _mapping(payload.get("inputs", {}), "inputs")
    selection = _mapping(payload.get("selection", {}), "selection")
    execution = _mapping(payload.get("execution", {}), "execution")
    artifacts = _mapping(payload.get("artifacts", {}), "artifacts")
    scientific_config = _mapping(payload.get("scientific_config", {}), "scientific_config")
    for table_name, table in (
        ("run", run),
        ("inputs", inputs),
        ("selection", selection),
        ("execution", execution),
        ("artifacts", artifacts),
        ("scientific_config", scientific_config),
    ):
        _reject_unknown_fields(table, _PROFILE_FIELDS[table_name], table_name)
    source_dir = resolved_path.parent
    input_case_manifest_path = _resolve_profile_path(
        source_dir,
        inputs.get("case_manifest"),
        "inputs.case_manifest",
    )
    profile_enabled = bool(payload.get("enabled", True))
    if profile_enabled and not input_case_manifest_path.is_file():
        raise FileNotFoundError("input case manifest does not exist: {}".format(input_case_manifest_path))

    return PatientOrchestrationProfile(
        source_path=resolved_path,
        schema_version=str(payload.get("schema_version", "")).strip(),
        description=str(payload.get("description", "")),
        enabled=profile_enabled,
        input_case_manifest_path=input_case_manifest_path,
        output_root=_resolve_profile_path(source_dir, run.get("output_root"), "run.output_root"),
        pathway_name=str(run.get("pathway", "")),
        checkpoint_name=str(run.get("checkpoint", "")),
        run_id=str(run.get("run_id", "")),
        patient_uids=_string_sequence(selection.get("patient_uids", ()), "selection.patient_uids"),
        requested_jobs=_string_sequence(
            execution.get("requested_jobs", ("standalone_patient_runner",)),
            "execution.requested_jobs",
        ),
        execution_mode=str(execution.get("mode", "plan_only")),
        failure_policy=str(execution.get("failure_policy", PatientProcessFailurePolicy.STOP_ON_FAILURE.value)),
        max_workers=int(execution.get("max_workers", 1)),
        timeout_seconds=execution.get("timeout_seconds"),
        retention_level=str(artifacts.get("retention_level", "minimal")),
        scientific_config_snapshot_path=_resolve_optional_profile_path(
            source_dir,
            scientific_config.get("snapshot"),
        ),
        metadata=_mapping(payload.get("metadata", {}), "metadata"),
    )


def _validate_checkpoint_pathway(checkpoint_name: str, pathway_name: str) -> None:
    from .scientific_runner import get_patient_scientific_runner_checkpoint

    checkpoint = get_patient_scientific_runner_checkpoint(checkpoint_name)
    if checkpoint.pathway_name.value != pathway_name:
        raise ValueError(
            "checkpoint and pathway disagree: {} maps to {}, not {}".format(
                checkpoint_name,
                checkpoint.pathway_name.value,
                pathway_name,
            )
        )


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("{} must be a TOML table".format(field_name))
    return dict(value)


def _reject_unknown_fields(value: Mapping[str, Any], allowed_fields: frozenset[str], field_name: str) -> None:
    unknown_fields = sorted(set(str(key) for key in value).difference(allowed_fields))
    if unknown_fields:
        raise ValueError("{} contains unsupported fields: {}".format(field_name, unknown_fields))


def _string_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("{} must be a TOML array of strings".format(field_name))
    return tuple(str(item) for item in value)


def _non_empty_string(value: Any, field_name: str) -> str:
    resolved_value = str(value).strip()
    if resolved_value == "":
        raise ValueError("{} cannot be empty".format(field_name))
    return resolved_value


def _non_empty_unique_strings(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    resolved_values = tuple(_non_empty_string(value, "{} item".format(field_name)) for value in values)
    if len(set(resolved_values)) != len(resolved_values):
        raise ValueError("{} cannot contain duplicates".format(field_name))
    return resolved_values


def _resolve_profile_path(source_dir: Path, value: Any, field_name: str) -> Path:
    path_text = _non_empty_string(value, field_name)
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else (source_dir / path).resolve()


def _resolve_optional_profile_path(source_dir: Path, value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return _resolve_profile_path(source_dir, value, "scientific_config.snapshot")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "PATIENT_ORCHESTRATION_EXECUTION_MODES",
    "PATIENT_ORCHESTRATION_PROFILE_SCHEMA_VERSION",
    "PatientOrchestrationProfile",
    "load_patient_orchestration_profile",
]

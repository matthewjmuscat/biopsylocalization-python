from __future__ import annotations

"""Code-owned catalog of manifest-like artifact contracts.

This module inventories the manifest surfaces the codebase can produce. It does
not write pipeline manifests itself; each producer keeps ownership of its own IO
and schema. The catalog gives post-run tooling and documentation generators a
single queryable surface for what each manifest does and what it tracks.
"""

from collections import Counter
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


MANIFEST_CATALOG_SCHEMA_VERSION = "manifest_catalog_v1"


@dataclass(frozen=True, slots=True)
class ManifestContract:
    """Contract metadata for one produced or planned manifest surface."""

    manifest_key: str
    title: str
    scope: str
    artifact_data_class: str
    lifecycle_status: str
    default_relative_paths: tuple[str, ...]
    payload_format: str
    schema_version_source: str
    producer: str
    purpose: str
    tracks: tuple[str, ...]
    reader: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        _validate_non_empty(self.manifest_key, "manifest_key")
        _validate_non_empty(self.title, "title")
        _validate_non_empty(self.scope, "scope")
        _validate_non_empty(self.artifact_data_class, "artifact_data_class")
        _validate_non_empty(self.lifecycle_status, "lifecycle_status")
        _validate_non_empty(self.payload_format, "payload_format")
        _validate_non_empty(self.schema_version_source, "schema_version_source")
        _validate_non_empty(self.producer, "producer")
        _validate_non_empty(self.purpose, "purpose")
        _validate_non_empty_sequence(self.default_relative_paths, "default_relative_paths")
        _validate_non_empty_sequence(self.tracks, "tracks")
        object.__setattr__(self, "default_relative_paths", tuple(self.default_relative_paths))
        object.__setattr__(self, "tracks", tuple(self.tracks))

    def to_row(self) -> dict[str, Any]:
        """Return a CSV-friendly row for generated catalog reports."""
        row = asdict(self)
        for key, value in list(row.items()):
            if isinstance(value, tuple):
                row[key] = " | ".join(str(item) for item in value)
        row["schema_version"] = MANIFEST_CATALOG_SCHEMA_VERSION
        return row


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _validate_non_empty(value: str, field_name: str) -> None:
    if str(value).strip() == "":
        raise ValueError(f"{field_name} cannot be empty")


def _validate_non_empty_sequence(values: Sequence[str], field_name: str) -> None:
    if not values:
        raise ValueError(f"{field_name} cannot be empty")
    for value in values:
        if str(value).strip() == "":
            raise ValueError(f"{field_name} cannot contain empty values")


def _contract(
    manifest_key: str,
    title: str,
    scope: str,
    artifact_data_class: str,
    lifecycle_status: str,
    default_relative_paths: tuple[str, ...],
    payload_format: str,
    schema_version_source: str,
    producer: str,
    purpose: str,
    tracks: tuple[str, ...],
    *,
    reader: str = "",
    notes: str = "",
) -> ManifestContract:
    return ManifestContract(
        manifest_key=manifest_key,
        title=title,
        scope=scope,
        artifact_data_class=artifact_data_class,
        lifecycle_status=lifecycle_status,
        default_relative_paths=default_relative_paths,
        payload_format=payload_format,
        schema_version_source=schema_version_source,
        producer=producer,
        reader=reader,
        purpose=purpose,
        tracks=tracks,
        notes=notes,
    )


_MANIFEST_CONTRACTS: tuple[ManifestContract, ...] = (
    _contract(
        "input_manifest_summary",
        "DICOM input manifest summary",
        "run_input",
        "manifest",
        "current_durable",
        ("manifests/input_manifest_summary.json",),
        "json",
        "input_data.dicom_manifest.INPUT_MANIFEST_SCHEMA_VERSION",
        "input_data.dicom_manifest.write_input_manifest_files",
        "Summarize the input DICOM discovery and role-routing manifest family for one run.",
        (
            "input manifest generation time",
            "case and DICOM file counts",
            "selected-role counts",
            "routing profile reference",
            "paths to companion input manifest files",
            "warning count",
        ),
    ),
    _contract(
        "input_case_manifest",
        "Input case manifest",
        "run_input",
        "manifest_table",
        "current_durable",
        ("manifests/input_case_manifest.csv",),
        "csv",
        "input_data.dicom_manifest.INPUT_MANIFEST_SCHEMA_VERSION",
        "input_data.dicom_manifest.write_input_manifest_files",
        "Record the patient/case-level role assignments discovered before scientific execution.",
        (
            "patient UID",
            "patient name and ID metadata",
            "parsed fraction number",
            "core RT role completeness",
            "RTSTRUCT/RTDOSE/RTPLAN paths",
            "US/MR T2/MR ADC file counts and paths",
        ),
    ),
    _contract(
        "input_dicom_manifest",
        "Input DICOM manifest",
        "run_input",
        "manifest_table",
        "current_durable",
        ("manifests/input_dicom_manifest.csv",),
        "csv",
        "input_data.dicom_manifest.INPUT_MANIFEST_SCHEMA_VERSION",
        "input_data.dicom_manifest.write_input_manifest_files",
        "Record one row per discovered DICOM file and the selected logical role used by the loader.",
        (
            "source DICOM path",
            "DICOM modality and identifiers",
            "generated and role-dictionary patient UID",
            "selected role",
            "routing reason",
            "read warnings",
        ),
    ),
    _contract(
        "input_routing_profile",
        "Input DICOM routing profile",
        "run_input",
        "configuration_manifest",
        "current_durable",
        ("manifests/input_routing_profile.json",),
        "json",
        "input_data.dicom_routing_profile.ROUTING_PROFILE_SCHEMA_VERSION",
        "input_data.dicom_manifest.write_input_manifest_files",
        "Record the DICOM routing rules used to classify input files for a run.",
        (
            "routing profile ID",
            "routing profile schema version",
            "fraction prefixes",
            "role matching rules",
        ),
    ),
    _contract(
        "input_manifest_warnings",
        "Input manifest warnings",
        "run_input",
        "manifest_log",
        "current_durable",
        ("manifests/input_manifest_warnings.jsonl",),
        "jsonl",
        "input_data.dicom_manifest.INPUT_MANIFEST_SCHEMA_VERSION",
        "input_data.dicom_manifest.write_input_manifest_files",
        "Record non-fatal input discovery and routing warnings without aborting manifest construction.",
        (
            "warning type",
            "patient UID",
            "file path",
            "warning message",
            "role-routing details",
        ),
    ),
    _contract(
        "run_completion_manifest",
        "Legacy run completion manifest",
        "run",
        "manifest",
        "legacy_current",
        ("RUN_COMPLETE.json",),
        "json",
        "not_versioned_currently",
        "preprocessing.output_runtime_dirs.write_run_completion_manifest",
        "Mark a legacy run output directory as completed and record lightweight runtime identity.",
        (
            "completion status",
            "completion timestamp",
            "specific output directory",
            "raw MC output directory",
            "run output label and metadata",
            "random seed policy metadata",
            "case and structure counts",
        ),
    ),
    _contract(
        "run_manifest_index",
        "Run manifest index",
        "run",
        "manifest_index",
        "current_durable",
        ("manifests/run_manifest_index.json",),
        "json",
        "output_artifacts.manifest_index.RUN_MANIFEST_INDEX_SCHEMA_VERSION",
        "output_artifacts.manifest_index.ManifestIndexRecorder.write",
        "Index every manifest object recorded during one run, including written, constructed-only, skipped, and failed manifests.",
        (
            "manifest key",
            "produced status",
            "manifest path when written",
            "path existence at index write time",
            "scope and artifact class",
            "producer",
            "patient and stage context when available",
        ),
        notes="Currently emitted by patient_runner.run_patient_batch; broader legacy/runtime wiring can be added at other run boundaries.",
    ),
    _contract(
        "patient_run_manifest",
        "Patient run manifest",
        "patient",
        "manifest",
        "current_durable",
        ("patients/<patient_uid>/patient_run_manifest.json", "patient_run_manifest.json"),
        "json",
        "patient_runner.manifests.PATIENT_RUN_MANIFEST_SCHEMA_VERSION",
        "patient_runner.manifests.write_patient_run_manifest",
        "Record status, stage results, artifacts, and lightweight identity for one patient run.",
        (
            "patient UID and label",
            "source run and input manifest IDs",
            "patient metadata",
            "overall patient status",
            "output root",
            "elapsed time",
            "stage statuses and warnings",
            "artifact paths",
        ),
        reader="post_run.cohort_assembly.manifest_loader.load_patient_batch_result_from_manifest",
    ),
    _contract(
        "patient_batch_run_manifest",
        "Patient batch run manifest",
        "batch_run",
        "manifest",
        "current_durable",
        ("patient_batch_run_manifest.json", "patient_scientific_runner/patient_batch_run_manifest.json"),
        "json",
        "patient_runner.manifests.PATIENT_BATCH_RUN_MANIFEST_SCHEMA_VERSION",
        "patient_runner.manifests.write_patient_batch_run_manifest",
        "Aggregate patient-run results and provide the primary entry point for post-run assembly.",
        (
            "batch status",
            "output root",
            "elapsed time",
            "patient count and failed-patient count",
            "batch artifact paths",
            "per-patient status summaries",
            "run metadata",
        ),
        reader="post_run.cohort_assembly.manifest_loader.load_patient_batch_result_from_manifest",
    ),
    _contract(
        "patient_process_run_plan",
        "Standalone patient process run plan",
        "standalone_parent_worker",
        "orchestration_manifest",
        "current_scaffold",
        ("patient_process_runner/parent_run_plan.json",),
        "json",
        "patient_runner.process_runner.PATIENT_PROCESS_RUN_PLAN_SCHEMA_VERSION",
        "patient_runner.process_runner.write_patient_process_run_plan",
        "Record the parent process plan for standalone one-patient worker execution.",
        (
            "selected patient cases",
            "failure policy",
            "worker job paths",
            "dry-run or execution mode",
            "parent output directory",
        ),
        notes="The non-dry-run worker boundary is still scaffolded until patient-local runtime construction is implemented.",
    ),
    _contract(
        "patient_worker_job",
        "Standalone patient worker job packet",
        "standalone_parent_worker",
        "orchestration_manifest",
        "current_scaffold",
        ("patient_process_runner/jobs/<patient_uid>.json",),
        "json",
        "patient_runner.process_runner.PATIENT_WORKER_JOB_SCHEMA_VERSION",
        "patient_runner.process_runner.write_patient_worker_job_packets",
        "Describe one worker process invocation without embedding cohort-scale scientific state.",
        (
            "patient case identity",
            "worker output directory",
            "job index",
            "run metadata",
            "dry-run mode",
        ),
    ),
    _contract(
        "patient_worker_result",
        "Standalone patient worker result",
        "standalone_parent_worker",
        "orchestration_manifest",
        "current_scaffold",
        ("patient_process_runner/results/<patient_uid>.json",),
        "json",
        "patient_runner.process_runner.PATIENT_WORKER_RESULT_SCHEMA_VERSION",
        "patient_runner.process_runner.run_patient_worker_job",
        "Record the lightweight outcome of one patient worker process.",
        (
            "patient UID",
            "worker status",
            "elapsed time",
            "error message",
            "artifact and manifest paths",
        ),
    ),
    _contract(
        "output_artifact_inventory",
        "Output artifact inventory",
        "completed_run_inventory",
        "inventory_table",
        "migration_utility",
        ("output_artifact_inventory.csv", "output_artifact_inventory_summary.json"),
        "csv_and_json",
        "output_artifacts.inventory.OUTPUT_ARTIFACT_INVENTORY_SCHEMA_VERSION",
        "output_artifacts.inventory.write_output_artifact_inventory",
        "Inventory concrete files found under a completed run output tree.",
        (
            "relative path",
            "file extension and size",
            "artifact kind",
            "output section",
            "patient UID when path-derived",
            "legacy dataframe name",
            "current output class",
            "lifetime recommendation",
        ),
    ),
    _contract(
        "patient_artifact_manifest",
        "Patient artifact manifest",
        "patient_artifact_inventory",
        "manifest_table",
        "migration_utility",
        ("patient_artifact_manifest.csv", "per_patient/<patient_uid>_artifact_manifest.csv"),
        "csv",
        "output_artifacts.patient_artifacts.PATIENT_ARTIFACT_MANIFEST_SCHEMA_VERSION",
        "output_artifacts.patient_artifacts.write_patient_artifact_outputs",
        "List current patient-scoped output artifacts discovered from a completed run inventory.",
        (
            "patient UID",
            "artifact scope",
            "relative path",
            "artifact kind and extension",
            "normalized table name",
            "source stage",
            "builder or source",
            "canonical primary key policy",
            "shadow manifest status",
        ),
    ),
    _contract(
        "patient_stitch_plan",
        "Patient stitch plan",
        "patient_artifact_inventory",
        "planning_table",
        "migration_utility",
        ("patient_stitch_plan.csv",),
        "csv",
        "output_artifacts.patient_artifacts.PATIENT_STITCH_PLAN_SCHEMA_VERSION",
        "output_artifacts.patient_artifacts.write_patient_artifact_outputs",
        "Plan which patient fragments can reconstruct cohort-style outputs during migration.",
        (
            "normalized table name",
            "output section",
            "proposed lifetime class",
            "stitch readiness",
            "source stage",
            "builder or source",
            "canonical primary key",
            "stitch key",
            "pruning notes",
        ),
    ),
    _contract(
        "output_table_contracts",
        "Output table contracts",
        "output_contract_registry",
        "contract_table",
        "migration_utility",
        ("output_table_contracts.csv", "output_table_contracts_summary.json"),
        "csv_and_json",
        "output_artifacts.contracts.OUTPUT_TABLE_CONTRACT_SCHEMA_VERSION",
        "output_artifacts.contracts.write_output_table_contracts",
        "Record reviewed contract metadata for durable output tables discovered during inventory work.",
        (
            "normalized table name",
            "legacy output section",
            "source stage",
            "builder or source",
            "canonical primary key",
            "stitch key",
            "pruning assessment",
        ),
    ),
    _contract(
        "output_schema_registry",
        "Output schema registry and data dictionary",
        "output_contract_registry",
        "contract_registry",
        "current_durable",
        (
            "output_schema_coverage_report.csv",
            "output_schema_coverage_summary.json",
            "output_schema_data_dictionary.csv",
            "output_schema_data_dictionary.md",
        ),
        "csv_json_markdown",
        "output_artifacts.schema_registry.OUTPUT_SCHEMA_REGISTRY_VERSION",
        "output_artifacts.schema_registry.write_output_schema_data_dictionary",
        "Declare durable output table contracts and generate human-readable schema documentation.",
        (
            "table ID",
            "legacy table name and output section",
            "artifact scope and table family",
            "row grain",
            "canonical keys and join keys",
            "source stage and source fragment",
            "stitch method",
            "validation status",
            "retention policy",
        ),
    ),
    _contract(
        "output_assembly_plan",
        "Output assembly plan",
        "output_contract_registry",
        "planning_table",
        "current_durable",
        ("output_assembly_plan.csv",),
        "csv",
        "output_artifacts.assembly_planner.OUTPUT_ASSEMBLY_PLANNER_SCHEMA_VERSION",
        "output_artifacts.assembly_planner.output_assembly_plan_rows",
        "Translate output schema contracts into cohort assembly policy rows.",
        (
            "final table ID",
            "source fragment table ID",
            "assembly method",
            "validation order policy",
            "production order policy",
            "columns policy",
            "CSV index policy",
        ),
    ),
    _contract(
        "phase3c_output_surface_manifest",
        "Phase 3C patient-fragment output surface manifest",
        "validation_surface",
        "manifest_table",
        "migration_validation",
        ("phase3c_patient_fragment_output_surface/phase3c_artifact_manifest.csv",),
        "csv",
        "output_artifacts.phase3c_surface.PHASE3C_OUTPUT_SURFACE_SCHEMA_VERSION",
        "output_artifacts.phase3c_surface.write_phase3c_output_surface",
        "Describe patient-fragment and stitched validation artifacts written for Phase 3C migration checks.",
        (
            "source scope",
            "patient UID and biopsy index",
            "dataframe name",
            "relative path",
            "row and column counts",
            "MultiIndex column status",
            "canonical key policy note",
        ),
    ),
    _contract(
        "post_run_cohort_assembly_config",
        "Post-run cohort assembly job config",
        "post_run",
        "configuration_manifest",
        "current_durable",
        ("post_run/configs/cohort_assembly_jobs.json",),
        "json",
        "post_run.cohort_assembly.config.POST_RUN_COHORT_ASSEMBLY_CONFIG_SCHEMA_VERSION",
        "post_run.cohort_assembly.config.load_post_run_cohort_assembly_job_configs",
        "Configure one or more post-run cohort assembly jobs for CLI or GUI use.",
        (
            "job name",
            "patient-runner output directory",
            "output directory",
            "patient UID filters",
            "final and source table filters",
            "write-output flags",
            "job metadata",
        ),
    ),
    _contract(
        "patient_batch_cohort_assembly_manifest",
        "Patient batch cohort assembly manifest",
        "post_run",
        "manifest_table",
        "current_durable",
        ("cohort_assembly/patient_batch_cohort_assembly.csv",),
        "csv",
        "patient_runner.cohort_assembly.PATIENT_BATCH_COHORT_ASSEMBLY_SCHEMA_VERSION",
        "patient_runner.cohort_assembly.write_patient_batch_cohort_assembly_outputs",
        "Record cohort assembly decisions and assembled-table status for a patient batch.",
        (
            "assembly plan identity",
            "expected-artifact decision",
            "missing artifact severity",
            "assembled table path",
            "row and column counts",
            "patient filters",
        ),
    ),
    _contract(
        "patient_batch_cohort_validation_manifest",
        "Patient batch cohort validation manifest",
        "post_run",
        "validation_table",
        "current_durable",
        ("cohort_assembly/patient_batch_cohort_validation.csv",),
        "csv",
        "patient_runner.cohort_assembly.PATIENT_BATCH_COHORT_VALIDATION_SCHEMA_VERSION",
        "patient_runner.cohort_assembly.write_patient_batch_cohort_assembly_outputs",
        "Record validation details for post-run cohort assembly outputs.",
        (
            "assembled table identity",
            "validation status",
            "expected and observed row counts",
            "missing source fragments",
            "diagnostic notes",
        ),
    ),
    _contract(
        "dose_nn_render_scene_artifact_manifest",
        "Dose NN render scene artifact manifest",
        "render_scene",
        "scene_manifest",
        "current_durable",
        ("render_scenes/<scene_id>/manifest.json",),
        "json_plus_npz_arrays",
        "mc.visualization.dose_nn_scene_artifacts.DOSE_NN_RENDER_SCENE_ARTIFACT_SCHEMA_VERSION",
        "mc.visualization.dose_nn_scene_artifacts.write_dose_nn_render_scene_artifact",
        "Describe one compact selected dose nearest-neighbour render scene and validate its array payload.",
        (
            "scene ID",
            "arrays filename",
            "patient/run/biopsy metadata",
            "lattice and biopsy coordinate frames",
            "array names",
            "array shapes and dtypes",
            "array SHA-256 checksums",
        ),
        reader="mc.visualization.dose_nn_scene_artifacts.read_dose_nn_render_scene_artifact",
    ),
    _contract(
        "patient_scientific_context_manifest",
        "Patient scientific context manifest",
        "patient_context",
        "manifest",
        "planned",
        ("patients/<patient_uid>/patient_scientific_context_manifest.json",),
        "json_with_array_artifact_refs",
        "planned output_artifacts.scientific_context manifest schema version",
        "planned output_artifacts.scientific_context.manifest writer",
        "Index retained patient scientific context artifacts for post-run GUI, validation, and downstream analysis.",
        (
            "patient and run identity",
            "retention level",
            "coordinate frames",
            "transform event table",
            "array artifact specs",
            "table artifact specs",
            "render-scene artifact specs",
            "code/config compatibility identity",
        ),
        notes="Planned target from docs/architecture/PATIENT_SCIENTIFIC_CONTEXT_ARTIFACTS.md; not produced yet.",
    ),
)


def iter_manifest_contracts() -> tuple[ManifestContract, ...]:
    """Return all cataloged manifest contracts in stable key order."""
    return tuple(sorted(_MANIFEST_CONTRACTS, key=lambda contract: contract.manifest_key))


def manifest_contracts_by_key(
    contracts: Iterable[ManifestContract] | None = None,
) -> dict[str, ManifestContract]:
    """Return manifest contracts keyed by ``manifest_key`` and fail on duplicates."""
    resolved_contracts = iter_manifest_contracts() if contracts is None else tuple(contracts)
    contracts_by_key: dict[str, ManifestContract] = {}
    duplicate_keys: list[str] = []
    for contract in resolved_contracts:
        if contract.manifest_key in contracts_by_key:
            duplicate_keys.append(contract.manifest_key)
        contracts_by_key[contract.manifest_key] = contract
    if duplicate_keys:
        raise ValueError(f"duplicate manifest contract keys: {sorted(duplicate_keys)}")
    return contracts_by_key


def manifest_catalog_rows(
    contracts: Iterable[ManifestContract] | None = None,
) -> list[dict[str, Any]]:
    """Return CSV-ready rows for manifest catalog reports."""
    resolved_contracts = iter_manifest_contracts() if contracts is None else tuple(contracts)
    return [contract.to_row() for contract in sorted(resolved_contracts, key=lambda item: item.manifest_key)]


def summarize_manifest_catalog(
    contracts: Iterable[ManifestContract] | None = None,
) -> dict[str, Any]:
    """Return a compact JSON-ready summary of the manifest catalog."""
    resolved_contracts = iter_manifest_contracts() if contracts is None else tuple(contracts)
    return {
        "schema_version": MANIFEST_CATALOG_SCHEMA_VERSION,
        "generated_utc": _utc_now_iso(),
        "manifest_contract_count": len(resolved_contracts),
        "scope_counts": dict(Counter(contract.scope for contract in resolved_contracts)),
        "artifact_data_class_counts": dict(Counter(contract.artifact_data_class for contract in resolved_contracts)),
        "lifecycle_status_counts": dict(Counter(contract.lifecycle_status for contract in resolved_contracts)),
    }


def inspect_manifest_presence(
    run_dir: Path | str,
    contracts: Iterable[ManifestContract] | None = None,
) -> list[dict[str, Any]]:
    """Inspect which cataloged manifest paths are present under a run directory.

    This is a presence report, not a run-profile validator. A missing manifest is
    evidence about that output tree, but it is not automatically a failure unless
    a separate compatibility or retention policy says the manifest was required.
    """
    resolved_run_dir = Path(run_dir).expanduser().resolve(strict=False)
    resolved_contracts = iter_manifest_contracts() if contracts is None else tuple(contracts)
    rows: list[dict[str, Any]] = []
    for contract in sorted(resolved_contracts, key=lambda item: item.manifest_key):
        found_paths = _find_manifest_paths(resolved_run_dir, contract.default_relative_paths)
        if found_paths:
            presence_status = "found"
        elif contract.lifecycle_status == "planned":
            presence_status = "planned_not_expected"
        else:
            presence_status = "not_found_in_run"
        rows.append(
            {
                "schema_version": MANIFEST_CATALOG_SCHEMA_VERSION,
                "run_dir": resolved_run_dir.as_posix(),
                "manifest_key": contract.manifest_key,
                "title": contract.title,
                "scope": contract.scope,
                "artifact_data_class": contract.artifact_data_class,
                "lifecycle_status": contract.lifecycle_status,
                "payload_format": contract.payload_format,
                "presence_status": presence_status,
                "found_count": len(found_paths),
                "found_relative_paths": " | ".join(path.as_posix() for path in found_paths),
                "default_relative_paths": " | ".join(contract.default_relative_paths),
                "producer": contract.producer,
                "reader": contract.reader,
                "purpose": contract.purpose,
                "tracks": " | ".join(contract.tracks),
            }
        )
    return rows


def summarize_manifest_presence(presence_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize rows returned by ``inspect_manifest_presence``."""
    presence_status_counts = Counter(str(row.get("presence_status", "")) for row in presence_rows)
    lifecycle_status_counts = Counter(str(row.get("lifecycle_status", "")) for row in presence_rows)
    return {
        "schema_version": MANIFEST_CATALOG_SCHEMA_VERSION,
        "generated_utc": _utc_now_iso(),
        "run_dir": str(presence_rows[0].get("run_dir", "")) if presence_rows else "",
        "manifest_contract_count": len(presence_rows),
        "found_manifest_contract_count": int(presence_status_counts.get("found", 0)),
        "presence_status_counts": dict(presence_status_counts),
        "lifecycle_status_counts": dict(lifecycle_status_counts),
    }


def render_manifest_catalog_markdown(
    contracts: Iterable[ManifestContract] | None = None,
) -> str:
    """Render the manifest catalog as a small Markdown table."""
    resolved_contracts = iter_manifest_contracts() if contracts is None else tuple(contracts)
    lines = [
        "# Manifest Catalog",
        "",
        f"Schema version: `{MANIFEST_CATALOG_SCHEMA_VERSION}`",
        "",
        "| Manifest key | Scope | Status | Format | Producer | Tracks |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for contract in sorted(resolved_contracts, key=lambda item: (item.scope, item.manifest_key)):
        lines.append(
            "| {manifest_key} | {scope} | {status} | {format} | {producer} | {tracks} |".format(
                manifest_key=_markdown_cell(contract.manifest_key),
                scope=_markdown_cell(contract.scope),
                status=_markdown_cell(contract.lifecycle_status),
                format=_markdown_cell(contract.payload_format),
                producer=_markdown_cell(contract.producer),
                tracks=_markdown_cell("; ".join(contract.tracks)),
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_manifest_catalog(
    output_dir: Path | str,
    contracts: Iterable[ManifestContract] | None = None,
) -> tuple[Path, Path, Path]:
    """Write CSV, summary JSON, and Markdown views of the manifest catalog."""
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    rows = manifest_catalog_rows(contracts)
    catalog_path = resolved_output_dir / "manifest_catalog.csv"
    summary_path = resolved_output_dir / "manifest_catalog_summary.json"
    markdown_path = resolved_output_dir / "manifest_catalog.md"

    fieldnames = list(rows[0].keys()) if rows else ["schema_version"]
    with catalog_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summarize_manifest_catalog(contracts), file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    markdown_path.write_text(render_manifest_catalog_markdown(contracts), encoding="utf-8")
    return catalog_path, summary_path, markdown_path


def write_manifest_presence_report(
    run_dir: Path | str,
    output_dir: Path | str,
    contracts: Iterable[ManifestContract] | None = None,
) -> tuple[Path, Path]:
    """Write a CSV and JSON summary showing cataloged manifests found in a run."""
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    presence_rows = inspect_manifest_presence(run_dir, contracts)
    presence_path = resolved_output_dir / "manifest_presence.csv"
    summary_path = resolved_output_dir / "manifest_presence_summary.json"
    fieldnames = list(presence_rows[0].keys()) if presence_rows else ["schema_version"]
    with presence_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(presence_rows)
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summarize_manifest_presence(presence_rows), file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    return presence_path, summary_path


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _find_manifest_paths(run_dir: Path, relative_paths: Sequence[str]) -> tuple[Path, ...]:
    found_paths: dict[str, Path] = {}
    for relative_path in relative_paths:
        pattern = _relative_path_to_glob_pattern(relative_path)
        for path in run_dir.glob(pattern):
            if not path.is_file():
                continue
            relative_found_path = path.relative_to(run_dir)
            found_paths[relative_found_path.as_posix()] = relative_found_path
    return tuple(found_paths[key] for key in sorted(found_paths))


def _relative_path_to_glob_pattern(relative_path: str) -> str:
    return re.sub(r"<[^/<>]+>", "*", str(relative_path).strip())
"""Cohort assembly and validation helpers for patient-batch artifacts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from output_artifacts import normalize_legacy_table_name
from output_artifacts import write_dataframe_artifact
from output_artifacts.stitch_validation import SHADOW_STITCH_PAIRS
from output_artifacts.stitch_validation import ShadowStitchPair

from .contracts import PatientBatchRunResult
from .contracts import validate_patient_uids


PATIENT_BATCH_COHORT_ASSEMBLY_SCHEMA_VERSION = "patient_batch_cohort_assembly_v1"
PATIENT_BATCH_COHORT_VALIDATION_SCHEMA_VERSION = "patient_batch_cohort_validation_v1"
COHORT_ARTIFACT_PATIENT_UID = "Global"


@dataclass(frozen=True, slots=True)
class PatientBatchCohortAssemblyConfig:
    """Selection and output policy for optional post-run cohort assembly."""

    patient_uids: Sequence[str] = ()
    final_table_names: Sequence[str] = ()
    source_table_names: Sequence[str] = ()
    output_dir: Path | None = None
    write_outputs: bool = False
    write_assembled_tables: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "patient_uids", validate_patient_uids(self.patient_uids, "patient_uids"))
        object.__setattr__(self, "final_table_names", _validate_name_filter(self.final_table_names, "final_table_names"))
        object.__setattr__(self, "source_table_names", _validate_name_filter(self.source_table_names, "source_table_names"))
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "write_outputs", bool(self.write_outputs))
        object.__setattr__(self, "write_assembled_tables", bool(self.write_assembled_tables))


@dataclass(frozen=True, slots=True)
class PatientBatchCohortAssemblyResult:
    """In-memory cohort assembly result built from patient-batch artifacts."""

    batch_result: PatientBatchRunResult
    inventory_df: pd.DataFrame
    assembly_df: pd.DataFrame
    assembled_tables: Mapping[str, pd.DataFrame] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "assembled_tables", dict(self.assembled_tables))

    @property
    def assembled_table_count(self) -> int:
        return len(self.assembled_tables)

    @property
    def assembled_table_names(self) -> tuple[str, ...]:
        return tuple(self.assembled_tables.keys())


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_path_name(value: str) -> str:
    safe = str(value).strip().replace("/", "_").replace("\\", "_")
    for old, new in ((" ", "_"), (":", "_"), (",", "_")):
        safe = safe.replace(old, new)
    return safe or "unknown"


def _validate_name_filter(values: Sequence[str], source_name: str) -> tuple[str, ...]:
    resolved_values = tuple(values)
    if any(not isinstance(value, str) for value in resolved_values):
        raise TypeError(f"{source_name} entries must be strings")
    if any(value.strip() == "" for value in resolved_values):
        raise ValueError(f"{source_name} cannot contain empty values")
    if len(set(resolved_values)) != len(resolved_values):
        raise ValueError(f"{source_name} cannot contain duplicates")
    return resolved_values


def _assembly_config_or_default(config: PatientBatchCohortAssemblyConfig | None) -> PatientBatchCohortAssemblyConfig:
    return PatientBatchCohortAssemblyConfig() if config is None else config


def _selected_stitch_pairs(stitch_pairs: Sequence[ShadowStitchPair],
                           config: PatientBatchCohortAssemblyConfig) -> tuple[ShadowStitchPair, ...]:
    selected_pairs: list[ShadowStitchPair] = []
    final_table_names = set(config.final_table_names)
    source_table_names = set(config.source_table_names)
    for pair in stitch_pairs:
        if final_table_names and pair.final_table_name not in final_table_names:
            continue
        if source_table_names and pair.source_table_name not in source_table_names:
            continue
        selected_pairs.append(pair)
    return tuple(selected_pairs)


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".parquet"}:
        return "table"
    if suffix in {".json", ".jsonl", ".log"}:
        return "manifest_or_runtime_metadata"
    return "other"


def _output_relative_path(path: Path) -> Path:
    parts = path.parts
    if "Output CSVs" not in parts:
        return Path(path.name)
    output_index = parts.index("Output CSVs")
    return Path(*parts[output_index:])


def _path_parts(relative_path: Path) -> tuple[str, ...]:
    return tuple(part for part in relative_path.parts if part not in ("", "."))


def _strip_patient_prefix(filename_stem: str, patient_uid: str) -> str:
    if patient_uid and filename_stem.startswith(f"{patient_uid}-"):
        return filename_stem[len(patient_uid) + 1:]
    return filename_stem


def _classify_patient_batch_artifact(path: Path) -> dict[str, Any]:
    relative_path = _output_relative_path(path)
    parts = _path_parts(relative_path)
    output_section = parts[0] if parts else "unknown"
    patient_uid = ""
    legacy_dataframe_name = relative_path.stem

    if len(parts) >= 3 and parts[0] == "Output CSVs" and parts[1] in {"Preprocessing", "MC simulation"}:
        output_section = f"Output CSVs/{parts[1]}"
        patient_uid = parts[2]
        legacy_dataframe_name = _strip_patient_prefix(relative_path.stem, patient_uid)
    elif len(parts) >= 2 and parts[0] == "Output CSVs" and parts[1] == "Cohort":
        output_section = "Output CSVs/Cohort"
        legacy_dataframe_name = relative_path.stem

    return {
        "relative_path": relative_path.as_posix(),
        "artifact_kind": _artifact_kind(path),
        "output_section": output_section,
        "patient_uid": patient_uid,
        "legacy_dataframe_name": legacy_dataframe_name,
    }


def build_patient_batch_artifact_inventory(batch_result: PatientBatchRunResult) -> pd.DataFrame:
    """Build a lightweight inventory from the artifacts written by a batch run."""
    rows: list[dict[str, Any]] = []
    for artifact_path in batch_result.artifact_paths:
        path = Path(artifact_path)
        classification = _classify_patient_batch_artifact(path)
        rows.append(
            {
                "schema_version": PATIENT_BATCH_COHORT_ASSEMBLY_SCHEMA_VERSION,
                "generated_utc": _utc_now_iso(),
                "batch_output_root": batch_result.output_root.as_posix(),
                "artifact_path": path.as_posix(),
                "file_name": path.name,
                "file_extension": path.suffix.lower(),
                "file_size_bytes": int(path.stat().st_size) if path.exists() else 0,
                **classification,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "schema_version",
                "generated_utc",
                "batch_output_root",
                "artifact_path",
                "relative_path",
                "file_name",
                "file_extension",
                "file_size_bytes",
                "artifact_kind",
                "output_section",
                "patient_uid",
                "legacy_dataframe_name",
                "normalized_table_name",
            ]
        )

    inventory_df = pd.DataFrame(rows).sort_values("artifact_path").reset_index(drop=True)
    table_mask = inventory_df["artifact_kind"].eq("table")
    inventory_df.loc[table_mask, "normalized_table_name"] = inventory_df.loc[table_mask].apply(
        normalize_legacy_table_name,
        axis=1,
    )
    inventory_df.loc[~table_mask, "normalized_table_name"] = inventory_df.loc[
        ~table_mask,
        "legacy_dataframe_name",
    ]
    return inventory_df


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    raise ValueError(f"Unsupported table extension for patient batch assembly: {path}")


def assemble_patient_batch_cohort_tables(batch_result: PatientBatchRunResult,
                                         stitch_pairs: Sequence[ShadowStitchPair] = SHADOW_STITCH_PAIRS,
                                         *,
                                         assembly_config: PatientBatchCohortAssemblyConfig | None = None) -> PatientBatchCohortAssemblyResult:
    """Assemble selected cohort-style tables from patient artifacts."""
    config = _assembly_config_or_default(assembly_config)
    inventory_df = build_patient_batch_artifact_inventory(batch_result)
    rows: list[dict[str, Any]] = []
    assembled_tables: dict[str, pd.DataFrame] = {}

    for pair in _selected_stitch_pairs(stitch_pairs, config):
        source_rows = inventory_df[
            inventory_df["artifact_kind"].eq("table")
            & inventory_df["output_section"].eq(pair.source_output_section)
            & inventory_df["normalized_table_name"].eq(pair.source_table_name)
            & inventory_df["file_extension"].eq(pair.file_extension)
            & inventory_df["patient_uid"].ne("")
            & inventory_df["patient_uid"].ne(COHORT_ARTIFACT_PATIENT_UID)
        ].sort_values("artifact_path")
        if config.patient_uids:
            source_rows = source_rows[source_rows["patient_uid"].isin(config.patient_uids)]

        row: dict[str, Any] = {
            "schema_version": PATIENT_BATCH_COHORT_ASSEMBLY_SCHEMA_VERSION,
            "generated_utc": _utc_now_iso(),
            "final_table_name": pair.final_table_name,
            "source_table_name": pair.source_table_name,
            "source_output_section": pair.source_output_section,
            "file_extension": pair.file_extension,
            "stitch_method": pair.stitch_method,
            "source_file_count": int(len(source_rows)),
            "assembled_rows": 0,
            "assembled_columns": 0,
            "assembly_status": "not_run",
            "assembly_notes": "",
        }

        if source_rows.empty:
            row["assembly_status"] = "missing_source_fragments"
            row["assembly_notes"] = "No patient artifacts matched this stitch pair."
            rows.append(row)
            continue

        source_dataframes = [_read_table(Path(path)) for path in source_rows["artifact_path"]]
        assembled_df = pd.concat(source_dataframes, ignore_index=True)
        assembled_tables[pair.final_table_name] = assembled_df
        row["assembled_rows"] = int(len(assembled_df))
        row["assembled_columns"] = int(len(assembled_df.columns))
        row["assembly_status"] = "assembled"
        rows.append(row)

    assembly_df = pd.DataFrame(rows)
    return PatientBatchCohortAssemblyResult(
        batch_result=batch_result,
        inventory_df=inventory_df,
        assembly_df=assembly_df,
        assembled_tables=assembled_tables,
    )


def _canonical_column_order(dataframe: pd.DataFrame) -> list[Any]:
    def sort_key(column: object) -> tuple[str, ...]:
        if isinstance(column, tuple):
            return tuple(str(part) for part in column)
        return (str(column),)

    return sorted(list(dataframe.columns), key=sort_key)


def _canonical_value_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.loc[:, _canonical_column_order(dataframe)].astype("string")


def _row_hashes(dataframe: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(
        _canonical_value_dataframe(dataframe),
        index=False,
    ).sort_values(ignore_index=True)


def _ordered_hashes(dataframe: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(
        _canonical_value_dataframe(dataframe),
        index=False,
    )


def _compare_dataframes(assembled_df: pd.DataFrame, final_df: pd.DataFrame) -> dict[str, Any]:
    same_column_set = set(assembled_df.columns) == set(final_df.columns)
    same_column_order = assembled_df.columns.equals(final_df.columns)
    same_shape = tuple(assembled_df.shape) == tuple(final_df.shape)
    if same_column_set:
        row_hash_match = bool(_row_hashes(assembled_df).equals(_row_hashes(final_df)))
        ordered_hash_match = bool(_ordered_hashes(assembled_df).equals(_ordered_hashes(final_df)))
    else:
        row_hash_match = False
        ordered_hash_match = False

    return {
        "shape_match": same_shape,
        "column_set_match": same_column_set,
        "column_order_match": same_column_order,
        "row_hash_match_ignore_order": row_hash_match,
        "row_hash_match_in_order": ordered_hash_match,
        "assembled_rows": int(len(assembled_df)),
        "final_rows": int(len(final_df)),
        "assembled_columns": int(len(assembled_df.columns)),
        "final_columns": int(len(final_df.columns)),
    }


def validate_patient_batch_cohort_assembly(assembly_result: PatientBatchCohortAssemblyResult,
                                           final_cohort_dataframes: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare assembled patient-batch tables with legacy final cohort tables."""
    rows: list[dict[str, Any]] = []
    for assembly_row in assembly_result.assembly_df.to_dict("records"):
        final_table_name = str(assembly_row["final_table_name"])
        assembled_df = assembly_result.assembled_tables.get(final_table_name)
        final_df = final_cohort_dataframes.get(final_table_name)
        row: dict[str, Any] = {
            "schema_version": PATIENT_BATCH_COHORT_VALIDATION_SCHEMA_VERSION,
            "generated_utc": _utc_now_iso(),
            "final_table_name": final_table_name,
            "source_table_name": assembly_row["source_table_name"],
            "source_file_count": int(assembly_row["source_file_count"]),
            "validation_status": "not_run",
            "validation_notes": "",
        }
        if assembled_df is None:
            row["validation_status"] = "missing_assembled_table"
            row["validation_notes"] = "No assembled table was produced for this stitch pair."
            rows.append(row)
            continue
        if not isinstance(final_df, pd.DataFrame):
            row["validation_status"] = "missing_final_dataframe"
            row["validation_notes"] = "No legacy final cohort dataframe was provided for this table."
            rows.append(row)
            continue

        row.update(_compare_dataframes(assembled_df, final_df))
        if row["shape_match"] and row["column_set_match"] and row["row_hash_match_ignore_order"]:
            row["validation_status"] = "match"
        else:
            row["validation_status"] = "mismatch"
        rows.append(row)

    return pd.DataFrame(rows)


def run_patient_batch_cohort_assembly(batch_result: PatientBatchRunResult,
                                      assembly_config: PatientBatchCohortAssemblyConfig | None = None,
                                      *,
                                      final_cohort_dataframes: Mapping[str, pd.DataFrame] | None = None,
                                      stitch_pairs: Sequence[ShadowStitchPair] = SHADOW_STITCH_PAIRS) -> tuple[PatientBatchCohortAssemblyResult, pd.DataFrame | None, tuple[Path, ...]]:
    """Run optional post-run assembly, validation, and writing as one utility call."""
    config = _assembly_config_or_default(assembly_config)
    assembly_result = assemble_patient_batch_cohort_tables(
        batch_result,
        stitch_pairs,
        assembly_config=config,
    )
    validation_df = None
    if final_cohort_dataframes is not None:
        validation_df = validate_patient_batch_cohort_assembly(assembly_result, final_cohort_dataframes)

    written_paths: tuple[Path, ...] = ()
    if config.write_outputs:
        written_paths = write_patient_batch_cohort_assembly_outputs(
            assembly_result,
            output_dir=config.output_dir,
            validation_df=validation_df,
            write_assembled_tables=config.write_assembled_tables,
        )
    return assembly_result, validation_df, written_paths


def summarize_patient_batch_cohort_assembly(assembly_result: PatientBatchCohortAssemblyResult) -> dict[str, Any]:
    assembly_df = assembly_result.assembly_df
    if assembly_df.empty:
        return {
            "schema_version": PATIENT_BATCH_COHORT_ASSEMBLY_SCHEMA_VERSION,
            "generated_utc": _utc_now_iso(),
            "assembly_pair_count": 0,
            "assembly_status_counts": {},
        }
    return {
        "schema_version": PATIENT_BATCH_COHORT_ASSEMBLY_SCHEMA_VERSION,
        "generated_utc": _utc_now_iso(),
        "assembly_pair_count": int(len(assembly_df)),
        "assembly_status_counts": dict(Counter(assembly_df["assembly_status"])),
        "assembled_table_count": assembly_result.assembled_table_count,
        "artifact_count": int(len(assembly_result.inventory_df)),
    }


def summarize_patient_batch_cohort_validation(validation_df: pd.DataFrame) -> dict[str, Any]:
    if validation_df.empty:
        return {
            "schema_version": PATIENT_BATCH_COHORT_VALIDATION_SCHEMA_VERSION,
            "generated_utc": _utc_now_iso(),
            "validation_pair_count": 0,
            "validation_status_counts": {},
        }
    return {
        "schema_version": PATIENT_BATCH_COHORT_VALIDATION_SCHEMA_VERSION,
        "generated_utc": _utc_now_iso(),
        "validation_pair_count": int(len(validation_df)),
        "validation_status_counts": dict(Counter(validation_df["validation_status"])),
        "matched_count": int(validation_df["validation_status"].eq("match").sum()),
        "mismatch_count": int(validation_df["validation_status"].eq("mismatch").sum()),
        "missing_assembled_table_count": int(validation_df["validation_status"].eq("missing_assembled_table").sum()),
        "missing_final_dataframe_count": int(validation_df["validation_status"].eq("missing_final_dataframe").sum()),
    }


def write_patient_batch_cohort_assembly_outputs(assembly_result: PatientBatchCohortAssemblyResult,
                                                output_dir: Path | None = None,
                                                *,
                                                validation_df: pd.DataFrame | None = None,
                                                write_assembled_tables: bool = True) -> tuple[Path, ...]:
    """Write assembly inventory, summaries, and optional assembled tables."""
    if output_dir is None:
        resolved_output_dir = assembly_result.batch_result.output_root.joinpath(
            "validation",
            "patient_batch_cohort_assembly",
        )
    else:
        resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []

    inventory_path = resolved_output_dir.joinpath("patient_batch_artifact_inventory.csv")
    assembly_path = resolved_output_dir.joinpath("patient_batch_cohort_assembly.csv")
    summary_path = resolved_output_dir.joinpath("patient_batch_cohort_assembly_summary.json")
    assembly_result.inventory_df.to_csv(inventory_path, index=False)
    assembly_result.assembly_df.to_csv(assembly_path, index=False)
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summarize_patient_batch_cohort_assembly(assembly_result), file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    written_paths.extend((inventory_path, assembly_path, summary_path))

    if validation_df is not None:
        validation_path = resolved_output_dir.joinpath("patient_batch_cohort_validation.csv")
        validation_summary_path = resolved_output_dir.joinpath("patient_batch_cohort_validation_summary.json")
        validation_df.to_csv(validation_path, index=False)
        with validation_summary_path.open("w", encoding="utf-8") as file_obj:
            json.dump(summarize_patient_batch_cohort_validation(validation_df), file_obj, indent=2, sort_keys=True)
            file_obj.write("\n")
        written_paths.extend((validation_path, validation_summary_path))

    if write_assembled_tables:
        table_dir = resolved_output_dir.joinpath("assembled_tables")
        for table_name, dataframe in assembly_result.assembled_tables.items():
            written_paths.append(
                write_dataframe_artifact(
                    dataframe,
                    table_dir.joinpath(f"{_safe_path_name(table_name)}.csv"),
                )
            )

    return tuple(written_paths)

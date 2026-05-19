from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import normalize_legacy_table_name


SHADOW_STITCH_VALIDATION_SCHEMA_VERSION = "phase3a_shadow_stitch_validation_v1"


@dataclass(frozen=True)
class ShadowStitchPair:
    final_table_name: str
    source_table_name: str
    source_output_section: str
    file_extension: str = ".csv"
    stitch_method: str = "concat_patient_fragments"


SHADOW_STITCH_PAIRS = (
    ShadowStitchPair(
        "Cohort: 3D radiomic features all OAR and DIL structures",
        "3D radiomic features all OAR and DIL structures",
        "Output CSVs/Preprocessing",
    ),
    ShadowStitchPair(
        "Cohort: All MC structure transformation values",
        "All MC structure transformation values",
        "Output CSVs/MC simulation",
    ),
    ShadowStitchPair(
        "Cohort: Biopsy basic spatial features dataframe",
        "Biopsy basic spatial features dataframe",
        "Output CSVs/Preprocessing",
    ),
    ShadowStitchPair(
        "Cohort: Bx DVH metrics",
        "DVH metrics",
        "Output CSVs/MC simulation",
    ),
    ShadowStitchPair(
        "Cohort: Bx DVH metrics (generalized)",
        "DVH metrics (Dx, Vx) statistics",
        "Output CSVs/MC simulation",
    ),
    ShadowStitchPair(
        "Cohort: Global MR ADC statistics",
        "MR - Global MR ADC statistics",
        "Output CSVs/MC simulation",
    ),
    ShadowStitchPair(
        "Cohort: Global by voxel MR ADC statistics",
        "MR - Global by voxel MR ADC statistics",
        "Output CSVs/MC simulation",
    ),
    ShadowStitchPair(
        "Cohort: Global dosimetry (NEW)",
        "Dosimetry - Global dosimetry (NEW)",
        "Output CSVs/MC simulation",
    ),
    ShadowStitchPair(
        "Cohort: Global dosimetry by voxel",
        "Dosimetry - Global dosimetry by voxel statistics",
        "Output CSVs/MC simulation",
    ),
    ShadowStitchPair(
        "Cohort: Guidance-map firing depth recommendations dataframe",
        "Biopsy optimization - Guidance-map firing depth recommendations dataframe",
        "Output CSVs/Preprocessing",
    ),
    ShadowStitchPair(
        "Cohort: Nearest DILs to each biopsy",
        "Nearest DILs info dataframe",
        "Output CSVs/Preprocessing",
    ),
    ShadowStitchPair(
        "Cohort: Per sample point prostate double sextant classification",
        "Per sample point prostate double sextant classification",
        "Output CSVs/Preprocessing",
    ),
    ShadowStitchPair(
        "Cohort: Per voxel prostate double sextant classification",
        "Per voxel prostate double sextant classification",
        "Output CSVs/Preprocessing",
    ),
    ShadowStitchPair(
        "Cohort: Simulated biopsy planned vs realized centroid variation validation",
        "Simulated biopsy planned vs realized centroid variation validation",
        "Output CSVs/Preprocessing",
    ),
    ShadowStitchPair(
        "Cohort: Simulated biopsy preparation dataframe",
        "Simulated biopsy preparation dataframe",
        "Output CSVs/Preprocessing",
    ),
    ShadowStitchPair(
        "Cohort: Tissue class - distances global results",
        "Tissue class - distances global results",
        "Output CSVs/MC simulation",
    ),
    ShadowStitchPair(
        "Cohort: Tissue class - distances pt-wise results",
        "Tissue class - distances pt-wise results",
        "Output CSVs/MC simulation",
    ),
    ShadowStitchPair(
        "Cohort: Tissue class - distances voxel-wise results",
        "Tissue class - distances voxel-wise results",
        "Output CSVs/MC simulation",
    ),
    ShadowStitchPair(
        "Cohort: structure specific mc results",
        "Tissue class - Pt wise structure specific results",
        "Output CSVs/MC simulation",
    ),
    ShadowStitchPair(
        "Cohort: sum-to-one mc results",
        "Tissue class - sum-to-one mc results",
        "Output CSVs/MC simulation",
    ),
    ShadowStitchPair(
        "Cohort: tissue class global scores (structure)",
        "Tissue class - Global tissue by structure statistics",
        "Output CSVs/MC simulation",
    ),
    ShadowStitchPair(
        "Cohort: tissue volume above threshold",
        "Tissue volume above threshold",
        "Output CSVs/MC simulation",
    ),
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_path_name(value: str) -> str:
    safe = str(value).strip().replace("/", "_").replace("\\", "_")
    for old, new in ((" ", "_"), (":", "_"), (",", "_")):
        safe = safe.replace(old, new)
    return safe or "unknown"


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    raise ValueError(f"Unsupported table extension for shadow stitch validation: {path}")


def _write_table(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        dataframe.to_parquet(path, index=False)
        return
    dataframe.to_csv(path, index=False)


def _row_hashes(dataframe: pd.DataFrame) -> pd.Series:
    comparable_df = dataframe.reindex(sorted(dataframe.columns), axis=1).astype("string")
    return pd.util.hash_pandas_object(comparable_df, index=False).sort_values(ignore_index=True)


def _unique_row_hashes(dataframe: pd.DataFrame) -> pd.Series:
    return _row_hashes(dataframe.drop_duplicates(ignore_index=True))


def _ordered_hashes(dataframe: pd.DataFrame) -> pd.Series:
    comparable_df = dataframe.reindex(sorted(dataframe.columns), axis=1).astype("string")
    return pd.util.hash_pandas_object(comparable_df, index=False)


def _drop_auto_index_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    keep_columns = [column for column in dataframe.columns if not str(column).startswith("Unnamed:")]
    return dataframe.loc[:, keep_columns]


def _compare_dataframes_once(recreated_df: pd.DataFrame, final_df: pd.DataFrame) -> dict[str, Any]:
    same_column_set = set(recreated_df.columns) == set(final_df.columns)
    same_column_order = list(recreated_df.columns) == list(final_df.columns)
    same_shape = tuple(recreated_df.shape) == tuple(final_df.shape)
    if same_column_set:
        recreated_row_hashes = _row_hashes(recreated_df)
        final_row_hashes = _row_hashes(final_df)
        row_hash_match = bool(recreated_row_hashes.equals(final_row_hashes))
        ordered_hash_match = bool(_ordered_hashes(recreated_df).equals(_ordered_hashes(final_df)))
        row_set_match_ignore_duplicates = bool(_unique_row_hashes(recreated_df).equals(_unique_row_hashes(final_df)))
    else:
        row_hash_match = False
        ordered_hash_match = False
        row_set_match_ignore_duplicates = False

    return {
        "shape_match": same_shape,
        "column_set_match": same_column_set,
        "column_order_match": same_column_order,
        "row_hash_match_ignore_order": row_hash_match,
        "row_hash_match_in_order": ordered_hash_match,
        "row_set_match_ignore_duplicates": row_set_match_ignore_duplicates,
        "recreated_rows": int(len(recreated_df)),
        "final_rows": int(len(final_df)),
        "recreated_duplicate_rows": int(len(recreated_df) - len(recreated_df.drop_duplicates(ignore_index=True))),
        "final_duplicate_rows": int(len(final_df) - len(final_df.drop_duplicates(ignore_index=True))),
        "recreated_columns": int(len(recreated_df.columns)),
        "final_columns": int(len(final_df.columns)),
    }


def _compare_dataframes(recreated_df: pd.DataFrame, final_df: pd.DataFrame) -> dict[str, Any]:
    exact_comparison = _compare_dataframes_once(recreated_df, final_df)
    semantic_comparison = _compare_dataframes_once(
        _drop_auto_index_columns(recreated_df),
        _drop_auto_index_columns(final_df),
    )
    auto_index_columns_ignored = sorted(
        set(recreated_df.columns).union(set(final_df.columns))
        - set(_drop_auto_index_columns(pd.concat([recreated_df.head(0), final_df.head(0)], ignore_index=True)).columns)
    )

    return {
        **exact_comparison,
        "semantic_shape_match": semantic_comparison["shape_match"],
        "semantic_column_set_match": semantic_comparison["column_set_match"],
        "semantic_column_order_match": semantic_comparison["column_order_match"],
        "semantic_row_hash_match_ignore_order": semantic_comparison["row_hash_match_ignore_order"],
        "semantic_row_hash_match_in_order": semantic_comparison["row_hash_match_in_order"],
        "semantic_row_set_match_ignore_duplicates": semantic_comparison["row_set_match_ignore_duplicates"],
        "semantic_recreated_duplicate_rows": semantic_comparison["recreated_duplicate_rows"],
        "semantic_final_duplicate_rows": semantic_comparison["final_duplicate_rows"],
        "auto_index_columns_ignored": "; ".join(auto_index_columns_ignored),
    }


def _inventory_with_normalized_names(inventory_df: pd.DataFrame) -> pd.DataFrame:
    inventory_df = inventory_df.copy()
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


def run_shadow_stitch_validation(inventory_df: pd.DataFrame,
                                 output_dir: Path,
                                 stitch_pairs: tuple[ShadowStitchPair, ...] = SHADOW_STITCH_PAIRS) -> pd.DataFrame:
    """Recreate selected final cohort tables from current patient fragments and compare them.

    This is a validation helper only. It never writes into the completed run directory.
    """
    output_dir = Path(output_dir)
    shadow_table_dir = output_dir.joinpath("shadow_stitched_tables")
    inventory_df = _inventory_with_normalized_names(inventory_df)
    if inventory_df.empty:
        return pd.DataFrame()

    run_dir = Path(str(inventory_df["run_dir"].iloc[0]))
    rows: list[dict[str, Any]] = []
    for pair in stitch_pairs:
        source_rows = inventory_df[
            inventory_df["artifact_kind"].eq("table")
            & inventory_df["output_section"].eq(pair.source_output_section)
            & inventory_df["normalized_table_name"].eq(pair.source_table_name)
            & inventory_df["file_extension"].eq(pair.file_extension)
            & inventory_df["patient_uid"].ne("Global")
            & inventory_df["patient_uid"].ne("")
        ].sort_values("relative_path")

        final_rows = inventory_df[
            inventory_df["artifact_kind"].eq("table")
            & inventory_df["output_section"].eq("Output CSVs/Cohort")
            & inventory_df["normalized_table_name"].eq(pair.final_table_name)
            & inventory_df["file_extension"].eq(pair.file_extension)
        ].sort_values("relative_path")

        row: dict[str, Any] = {
            "schema_version": SHADOW_STITCH_VALIDATION_SCHEMA_VERSION,
            "generated_utc": _utc_now_iso(),
            "final_table_name": pair.final_table_name,
            "source_table_name": pair.source_table_name,
            "source_output_section": pair.source_output_section,
            "file_extension": pair.file_extension,
            "stitch_method": pair.stitch_method,
            "source_file_count": int(len(source_rows)),
            "final_file_count": int(len(final_rows)),
            "shadow_output_path": "",
            "validation_status": "not_run",
            "validation_notes": "",
        }

        if source_rows.empty:
            row["validation_status"] = "missing_source_fragments"
            row["validation_notes"] = "No patient source fragments matched this stitch pair."
            rows.append(row)
            continue
        if len(final_rows) != 1:
            row["validation_status"] = "missing_or_ambiguous_final_artifact"
            row["validation_notes"] = "Expected exactly one current final cohort artifact."
            rows.append(row)
            continue

        source_dataframes = [_read_table(run_dir.joinpath(relative_path)) for relative_path in source_rows["relative_path"]]
        recreated_df = pd.concat(source_dataframes, ignore_index=True)
        final_path = run_dir.joinpath(str(final_rows["relative_path"].iloc[0]))
        final_df = _read_table(final_path)
        shadow_path = shadow_table_dir.joinpath(f"{_safe_path_name(pair.final_table_name)}{pair.file_extension}")
        _write_table(recreated_df, shadow_path)
        row["shadow_output_path"] = str(shadow_path.relative_to(output_dir))
        row.update(_compare_dataframes(recreated_df, final_df))
        if row["shape_match"] and row["column_set_match"] and row["row_hash_match_ignore_order"]:
            row["validation_status"] = "match"
        elif (
            row["semantic_shape_match"]
            and row["semantic_column_set_match"]
            and row["semantic_row_hash_match_ignore_order"]
        ):
            row["validation_status"] = "semantic_match_ignored_csv_index"
        elif row["semantic_column_set_match"] and row["semantic_row_set_match_ignore_duplicates"]:
            row["validation_status"] = "semantic_row_set_match_duplicate_count_mismatch"
        else:
            row["validation_status"] = "mismatch"
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_shadow_stitch_validation(validation_df: pd.DataFrame) -> dict[str, Any]:
    if validation_df.empty:
        return {
            "schema_version": SHADOW_STITCH_VALIDATION_SCHEMA_VERSION,
            "generated_utc": _utc_now_iso(),
            "validation_pair_count": 0,
            "validation_status_counts": {},
        }
    return {
        "schema_version": SHADOW_STITCH_VALIDATION_SCHEMA_VERSION,
        "generated_utc": _utc_now_iso(),
        "validation_pair_count": int(len(validation_df)),
        "validation_status_counts": validation_df["validation_status"].value_counts().to_dict(),
        "exact_match_count": int(validation_df["validation_status"].eq("match").sum()),
        "semantic_match_ignored_csv_index_count": int(
            validation_df["validation_status"].eq("semantic_match_ignored_csv_index").sum()
        ),
        "semantic_row_set_duplicate_count_mismatch_count": int(
            validation_df["validation_status"].eq("semantic_row_set_match_duplicate_count_mismatch").sum()
        ),
        "true_mismatch_count": int(validation_df["validation_status"].eq("mismatch").sum()),
    }


def write_shadow_stitch_validation(validation_df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = output_dir.joinpath("shadow_stitch_validation.csv")
    summary_path = output_dir.joinpath("shadow_stitch_validation_summary.json")
    validation_df.to_csv(validation_path, index=False)
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summarize_shadow_stitch_validation(validation_df), file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    return validation_path, summary_path
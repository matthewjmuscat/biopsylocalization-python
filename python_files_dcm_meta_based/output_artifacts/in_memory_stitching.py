from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from legacy_data_keys import legacy_data_keys

from .exporters import write_dataframe_artifact
from .schema_registry import OutputSchemaRegistry
from .stitch_validation import SHADOW_STITCH_PAIRS
from .stitch_validation import ShadowStitchPair


IN_MEMORY_STITCH_VALIDATION_SCHEMA_VERSION = "phase3b_in_memory_stitch_validation_v1"
LEGACY_PATIENT_ALL_REFERENCE_KEYS = legacy_data_keys.patient_all_reference
LEGACY_BIOPSY_RUNTIME_KEYS = legacy_data_keys.biopsy_runtime


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_path_name(value: str) -> str:
    safe = str(value).strip().replace("/", "_").replace("\\", "_")
    for old, new in ((" ", "_"), (":", "_"), (",", "_")):
        safe = safe.replace(old, new)
    return safe or "unknown"


def _is_dataframe(value: object) -> bool:
    return isinstance(value, pd.DataFrame)


def _column_sort_key(column: object) -> tuple[str, ...]:
    if isinstance(column, tuple):
        return tuple(str(part) for part in column)
    return (str(column),)


def _canonical_column_order(dataframe: pd.DataFrame) -> list[Any]:
    return sorted(list(dataframe.columns), key=_column_sort_key)


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


def _cohort_sort_columns(stitch_pair: ShadowStitchPair) -> tuple[str, ...]:
    if stitch_pair.row_order_columns:
        return stitch_pair.row_order_columns
    spec = OutputSchemaRegistry().match_spec(
        stitch_pair.final_table_name,
        "Output CSVs/Cohort",
        stitch_pair.file_extension,
    )
    return spec.canonical_primary_key if spec is not None else ()


def _sort_dataframe_by_columns(dataframe: pd.DataFrame, sort_columns: tuple[str, ...]) -> pd.DataFrame:
    present_columns = [column for column in sort_columns if column in dataframe.columns]
    if not present_columns:
        return dataframe.reset_index(drop=True)

    sortable_df = dataframe.copy()
    helper_columns: list[str] = []
    for column_index, column in enumerate(present_columns):
        text_values = sortable_df[column].astype("string").fillna("")
        numeric_values = pd.to_numeric(text_values, errors="coerce")
        is_text_helper = f"__canonical_sort_is_text_{column_index}"
        numeric_helper = f"__canonical_sort_numeric_{column_index}"
        text_helper = f"__canonical_sort_text_{column_index}"
        sortable_df[is_text_helper] = numeric_values.isna()
        sortable_df[numeric_helper] = numeric_values.fillna(0)
        sortable_df[text_helper] = text_values
        helper_columns.extend([is_text_helper, numeric_helper, text_helper])

    return (
        sortable_df.sort_values(helper_columns, kind="mergesort")
        .drop(columns=helper_columns)
        .reset_index(drop=True)
    )


def _compare_dataframes(recreated_df: pd.DataFrame, final_df: pd.DataFrame) -> dict[str, Any]:
    same_column_set = set(recreated_df.columns) == set(final_df.columns)
    same_column_order = recreated_df.columns.equals(final_df.columns)
    same_shape = tuple(recreated_df.shape) == tuple(final_df.shape)
    if same_column_set:
        row_hash_match = bool(_row_hashes(recreated_df).equals(_row_hashes(final_df)))
        ordered_hash_match = bool(_ordered_hashes(recreated_df).equals(_ordered_hashes(final_df)))
    else:
        row_hash_match = False
        ordered_hash_match = False

    return {
        "shape_match": same_shape,
        "column_set_match": same_column_set,
        "column_order_match": same_column_order,
        "row_hash_match_ignore_order": row_hash_match,
        "row_hash_match_in_order": ordered_hash_match,
        "recreated_rows": int(len(recreated_df)),
        "final_rows": int(len(final_df)),
        "recreated_columns": int(len(recreated_df.columns)),
        "final_columns": int(len(final_df.columns)),
        "recreated_multiindex_columns": isinstance(recreated_df.columns, pd.MultiIndex),
        "final_multiindex_columns": isinstance(final_df.columns, pd.MultiIndex),
    }


def _preprocessing_fragment(pydicom_item: dict,
                            all_ref_key: str,
                            source_table_name: str) -> pd.DataFrame | None:
    dataframe_dict = pydicom_item[all_ref_key][
        LEGACY_PATIENT_ALL_REFERENCE_KEYS.preprocessing_output_dataframes_key
    ]
    dataframe = dataframe_dict.get(source_table_name)
    return dataframe if _is_dataframe(dataframe) else None


def _mc_multi_structure_fragment(pydicom_item: dict,
                                 all_ref_key: str,
                                 source_table_name: str) -> pd.DataFrame | None:
    dataframe_dict = pydicom_item[all_ref_key][LEGACY_PATIENT_ALL_REFERENCE_KEYS.mc_output_dataframes_key]
    dataframe = dataframe_dict.get(source_table_name)
    return dataframe if _is_dataframe(dataframe) else None


def _biopsy_output_fragments(pydicom_item: dict,
                             bx_ref: str,
                             source_table_name: str) -> list[pd.DataFrame]:
    dataframes: list[pd.DataFrame] = []
    for specific_bx_structure in pydicom_item.get(bx_ref, []):
        dataframe = specific_bx_structure.get(LEGACY_BIOPSY_RUNTIME_KEYS.output_dataframes_key, {}).get(source_table_name)
        if _is_dataframe(dataframe):
            dataframes.append(dataframe)
    return dataframes


def collect_patient_fragment_dataframes(master_structure_reference_dict: dict,
                                        all_ref_key: str,
                                        bx_ref: str,
                                        stitch_pair: ShadowStitchPair) -> list[pd.DataFrame]:
    dataframes: list[pd.DataFrame] = []
    for _patient_uid, pydicom_item in master_structure_reference_dict.items():
        if stitch_pair.source_output_section == "Output CSVs/Preprocessing":
            dataframe = _preprocessing_fragment(pydicom_item, all_ref_key, stitch_pair.source_table_name)
            if dataframe is not None:
                dataframes.append(dataframe)
            continue

        if stitch_pair.source_output_section == "Output CSVs/MC simulation":
            dataframe = _mc_multi_structure_fragment(pydicom_item, all_ref_key, stitch_pair.source_table_name)
            if dataframe is not None:
                dataframes.append(dataframe)
                continue
            dataframes.extend(_biopsy_output_fragments(pydicom_item, bx_ref, stitch_pair.source_table_name))

    return dataframes


def build_in_memory_stitch_validation(master_structure_reference_dict: dict,
                                      master_cohort_patient_data_and_dataframes: dict,
                                      all_ref_key: str,
                                      bx_ref: str,
                                      stitch_pairs: tuple[ShadowStitchPair, ...] = SHADOW_STITCH_PAIRS,
                                      *,
                                      return_stitched_tables: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    stitched_tables: dict[str, pd.DataFrame] = {}
    cohort_dataframe_dict = master_cohort_patient_data_and_dataframes.get("Dataframes", {})

    for stitch_pair in stitch_pairs:
        source_dataframes = collect_patient_fragment_dataframes(
            master_structure_reference_dict,
            all_ref_key,
            bx_ref,
            stitch_pair,
        )
        final_df = cohort_dataframe_dict.get(stitch_pair.final_table_name)
        row: dict[str, Any] = {
            "schema_version": IN_MEMORY_STITCH_VALIDATION_SCHEMA_VERSION,
            "generated_utc": _utc_now_iso(),
            "final_table_name": stitch_pair.final_table_name,
            "source_table_name": stitch_pair.source_table_name,
            "source_output_section": stitch_pair.source_output_section,
            "source_fragment_count": int(len(source_dataframes)),
            "validation_status": "not_run",
            "validation_notes": "",
        }

        if not source_dataframes:
            row["validation_status"] = "missing_source_fragments"
            row["validation_notes"] = "No in-memory patient fragments matched this stitch pair."
            rows.append(row)
            continue
        if not _is_dataframe(final_df):
            row["validation_status"] = "missing_final_dataframe"
            row["validation_notes"] = "No final cohort dataframe matched this stitch pair."
            rows.append(row)
            continue

        recreated_df = _sort_dataframe_by_columns(
            pd.concat(source_dataframes, ignore_index=True),
            _cohort_sort_columns(stitch_pair),
        )
        stitched_tables[stitch_pair.final_table_name] = recreated_df
        row.update(_compare_dataframes(recreated_df, final_df))
        if row["shape_match"] and row["column_set_match"] and row["row_hash_match_ignore_order"]:
            row["validation_status"] = "match"
        else:
            row["validation_status"] = "mismatch"
        rows.append(row)

    validation_df = pd.DataFrame(rows)
    if return_stitched_tables:
        return validation_df, stitched_tables
    return validation_df


def summarize_in_memory_stitch_validation(validation_df: pd.DataFrame) -> dict[str, Any]:
    if validation_df.empty:
        return {
            "schema_version": IN_MEMORY_STITCH_VALIDATION_SCHEMA_VERSION,
            "generated_utc": _utc_now_iso(),
            "validation_pair_count": 0,
            "validation_status_counts": {},
        }
    return {
        "schema_version": IN_MEMORY_STITCH_VALIDATION_SCHEMA_VERSION,
        "generated_utc": _utc_now_iso(),
        "validation_pair_count": int(len(validation_df)),
        "validation_status_counts": dict(Counter(validation_df["validation_status"])),
        "matched_count": int(validation_df["validation_status"].eq("match").sum()),
        "mismatch_count": int(validation_df["validation_status"].eq("mismatch").sum()),
        "missing_source_fragment_count": int(validation_df["validation_status"].eq("missing_source_fragments").sum()),
        "missing_final_dataframe_count": int(validation_df["validation_status"].eq("missing_final_dataframe").sum()),
    }


def write_in_memory_stitch_validation_outputs(validation_df: pd.DataFrame,
                                             stitched_tables: dict[str, pd.DataFrame],
                                             output_dir: Path,
                                             *,
                                             write_stitched_tables: bool = True) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = output_dir.joinpath("in_memory_stitch_validation.csv")
    summary_path = output_dir.joinpath("in_memory_stitch_validation_summary.json")
    validation_df.to_csv(validation_path, index=False)
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summarize_in_memory_stitch_validation(validation_df), file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")

    if write_stitched_tables:
        stitched_table_dir = output_dir.joinpath("in_memory_stitched_tables")
        for table_name, dataframe in stitched_tables.items():
            write_dataframe_artifact(
                dataframe,
                stitched_table_dir.joinpath(f"{_safe_path_name(table_name)}.csv"),
                csv_index=False,
            )

    return validation_path, summary_path
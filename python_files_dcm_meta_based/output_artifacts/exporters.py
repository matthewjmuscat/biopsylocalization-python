from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

from legacy_data_keys import legacy_data_keys


PHASE3B_DATAFRAME_EXPORT_SCHEMA_VERSION = "phase3b_dataframe_export_v1"
LEGACY_STRUCTURE_RECORD_KEYS = legacy_data_keys.structure_record
LEGACY_PATIENT_ALL_REFERENCE_KEYS = legacy_data_keys.patient_all_reference
LEGACY_BIOPSY_RUNTIME_KEYS = legacy_data_keys.biopsy_runtime

MC_MULTI_STRUCTURE_PARQUET_TABLE_NAMES = {
    "Tissue class - containment and distances (light) results",
}
BIOPSY_PARQUET_TABLE_NAMES = {
    "Point-wise dose output by MC trial number",
    "Point-wise MR ADC output by MC trial number",
    "Voxel-wise dose output by MC trial number",
    "Cumulative DVH by MC trial",
    "Differential DVH by MC trial",
}


@dataclass(frozen=True)
class DataframeArtifact:
    source_scope: str
    dataframe_name: str
    dataframe: pd.DataFrame
    relative_path: Path
    patient_uid: str | None = None
    biopsy_index: str | int | None = None
    file_extension: str = ".csv"

    @property
    def has_multiindex_columns(self) -> bool:
        return isinstance(self.dataframe.columns, pd.MultiIndex)


def write_dataframe_artifact(dataframe: pd.DataFrame,
                             output_path: Path,
                             *,
                             csv_index: bool = False,
                             parquet_index: bool = False,
                             parquet_compression: str = "snappy") -> Path:
    """Write a dataframe while preserving its columns, including MultiIndex columns.

    New Phase 3B outputs intentionally default to `index=False` for CSVs. For
    MultiIndex-column DataFrames, pandas writes multi-row headers and preserves
    the column index representation without adding an `Unnamed: 0` index column.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        dataframe.to_parquet(output_path, index=parquet_index, compression=parquet_compression)
        return output_path
    if suffix == ".csv":
        dataframe.to_csv(output_path, index=csv_index)
        return output_path
    raise ValueError(f"Unsupported dataframe artifact extension: {output_path}")


def write_dataframe_artifacts(artifacts: Iterator[DataframeArtifact] | list[DataframeArtifact],
                              output_root: Path,
                              *,
                              csv_index: bool = False,
                              parquet_index: bool = False,
                              parquet_compression: str = "snappy") -> list[Path]:
    written_paths: list[Path] = []
    for artifact in artifacts:
        written_paths.append(
            write_dataframe_artifact(
                artifact.dataframe,
                Path(output_root).joinpath(artifact.relative_path),
                csv_index=csv_index,
                parquet_index=parquet_index,
                parquet_compression=parquet_compression,
            )
        )
    return written_paths


def _is_dataframe(value: object) -> bool:
    return isinstance(value, pd.DataFrame)


def _patient_preprocessing_relative_path(patient_uid: str, dataframe_name: str) -> Path:
    return Path("Output CSVs").joinpath(
        "Preprocessing",
        str(patient_uid),
        f"{patient_uid}-{dataframe_name}.csv",
    )


def _patient_mc_relative_path(patient_uid: str, dataframe_name: str, file_extension: str) -> Path:
    return Path("Output CSVs").joinpath(
        "MC simulation",
        str(patient_uid),
        f"{patient_uid}-{dataframe_name}{file_extension}",
    )


def _biopsy_mc_relative_path(patient_uid: str,
                             biopsy_type: str,
                             biopsy_name: str,
                             biopsy_index: str | int,
                             dataframe_name: str,
                             file_extension: str) -> Path:
    return Path("Output CSVs").joinpath(
        "MC simulation",
        str(patient_uid),
        f"{biopsy_index}-{biopsy_name}",
        f"{patient_uid}-{biopsy_type}-{biopsy_name}-{biopsy_index}-{dataframe_name}{file_extension}",
    )


def _cohort_relative_path(dataframe_name: str) -> Path:
    return Path("Output CSVs").joinpath("Cohort", f"{dataframe_name}.csv")


def iter_patient_preprocessing_artifacts(master_structure_reference_dict: dict,
                                         all_ref_key: str) -> Iterator[DataframeArtifact]:
    for patient_uid, pydicom_item in master_structure_reference_dict.items():
        dataframe_dict = pydicom_item[all_ref_key][
            LEGACY_PATIENT_ALL_REFERENCE_KEYS.preprocessing_output_dataframes_key
        ]
        for dataframe_name, dataframe in dataframe_dict.items():
            if _is_dataframe(dataframe):
                yield DataframeArtifact(
                    source_scope="patient_preprocessing",
                    patient_uid=str(patient_uid),
                    dataframe_name=str(dataframe_name),
                    dataframe=dataframe,
                    relative_path=_patient_preprocessing_relative_path(str(patient_uid), str(dataframe_name)),
                )


def iter_patient_mc_artifacts(master_structure_reference_dict: dict,
                              all_ref_key: str) -> Iterator[DataframeArtifact]:
    for patient_uid, pydicom_item in master_structure_reference_dict.items():
        dataframe_dict = pydicom_item[all_ref_key][
            LEGACY_PATIENT_ALL_REFERENCE_KEYS.mc_output_dataframes_key
        ]
        for dataframe_name, dataframe in dataframe_dict.items():
            if _is_dataframe(dataframe):
                file_extension = ".parquet" if dataframe_name in MC_MULTI_STRUCTURE_PARQUET_TABLE_NAMES else ".csv"
                yield DataframeArtifact(
                    source_scope="patient_mc",
                    patient_uid=str(patient_uid),
                    dataframe_name=str(dataframe_name),
                    dataframe=dataframe,
                    relative_path=_patient_mc_relative_path(str(patient_uid), str(dataframe_name), file_extension),
                    file_extension=file_extension,
                )


def iter_biopsy_mc_artifacts(master_structure_reference_dict: dict,
                             bx_ref: str) -> Iterator[DataframeArtifact]:
    for patient_uid, pydicom_item in master_structure_reference_dict.items():
        for specific_bx_structure in pydicom_item[bx_ref]:
            biopsy_name = specific_bx_structure[LEGACY_STRUCTURE_RECORD_KEYS.roi_key]
            biopsy_type = specific_bx_structure[LEGACY_STRUCTURE_RECORD_KEYS.simulated_type_key]
            biopsy_index = specific_bx_structure[LEGACY_STRUCTURE_RECORD_KEYS.index_number_key]
            for dataframe_name, dataframe in specific_bx_structure[LEGACY_BIOPSY_RUNTIME_KEYS.output_dataframes_key].items():
                if _is_dataframe(dataframe):
                    file_extension = ".parquet" if dataframe_name in BIOPSY_PARQUET_TABLE_NAMES else ".csv"
                    yield DataframeArtifact(
                        source_scope="biopsy_mc",
                        patient_uid=str(patient_uid),
                        biopsy_index=biopsy_index,
                        dataframe_name=str(dataframe_name),
                        dataframe=dataframe,
                        relative_path=_biopsy_mc_relative_path(
                            str(patient_uid),
                            str(biopsy_type),
                            str(biopsy_name),
                            biopsy_index,
                            str(dataframe_name),
                            file_extension,
                        ),
                        file_extension=file_extension,
                    )


def iter_cohort_artifacts(master_cohort_patient_data_and_dataframes: dict) -> Iterator[DataframeArtifact]:
    for dataframe_name, dataframe in master_cohort_patient_data_and_dataframes["Dataframes"].items():
        if _is_dataframe(dataframe):
            yield DataframeArtifact(
                source_scope="cohort",
                dataframe_name=str(dataframe_name),
                dataframe=dataframe,
                relative_path=_cohort_relative_path(str(dataframe_name)),
            )
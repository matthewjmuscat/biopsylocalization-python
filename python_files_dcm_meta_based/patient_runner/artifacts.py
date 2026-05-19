"""Patient artifact collection and writing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from output_artifacts import DataframeArtifact
from output_artifacts import iter_biopsy_mc_artifacts
from output_artifacts import iter_patient_mc_artifacts
from output_artifacts import iter_patient_preprocessing_artifacts
from output_artifacts import write_dataframe_artifacts

from .contracts import LegacyPatientRuntimeState
from .contracts import PatientRunConfig


@dataclass(frozen=True, slots=True)
class PatientArtifactStore:
    """Controlled write surface for one patient's dataframe artifacts."""

    output_root: Path
    all_ref_key: str
    bx_ref: str
    include_preprocessing: bool = True
    include_patient_mc: bool = True
    include_biopsy_mc: bool = True
    csv_index: bool = False
    parquet_index: bool = False
    parquet_compression: str = "snappy"

    @classmethod
    def from_config(cls,
                    config: PatientRunConfig,
                    runtime_state: LegacyPatientRuntimeState) -> "PatientArtifactStore":
        return cls(
            output_root=config.patient_output_dir(runtime_state.patient_case),
            all_ref_key=config.all_ref_key,
            bx_ref=config.bx_ref,
            include_preprocessing=config.write_preprocessing_artifacts,
            include_patient_mc=config.write_patient_mc_artifacts,
            include_biopsy_mc=config.write_biopsy_mc_artifacts,
            csv_index=config.csv_index,
            parquet_index=config.parquet_index,
            parquet_compression=config.parquet_compression,
        )

    def collect(self, runtime_state: LegacyPatientRuntimeState) -> tuple[DataframeArtifact, ...]:
        return collect_patient_dataframe_artifacts(
            runtime_state,
            all_ref_key=self.all_ref_key,
            bx_ref=self.bx_ref,
            include_preprocessing=self.include_preprocessing,
            include_patient_mc=self.include_patient_mc,
            include_biopsy_mc=self.include_biopsy_mc,
        )

    def write(self, artifacts: Sequence[DataframeArtifact]) -> tuple[Path, ...]:
        return tuple(
            write_dataframe_artifacts(
                list(artifacts),
                self.output_root,
                csv_index=self.csv_index,
                parquet_index=self.parquet_index,
                parquet_compression=self.parquet_compression,
            )
        )


def collect_patient_dataframe_artifacts(runtime_state: LegacyPatientRuntimeState,
                                        *,
                                        all_ref_key: str,
                                        bx_ref: str,
                                        include_preprocessing: bool = True,
                                        include_patient_mc: bool = True,
                                        include_biopsy_mc: bool = True) -> tuple[DataframeArtifact, ...]:
    """Collect registered dataframe artifacts from a one-patient legacy view."""
    artifacts: list[DataframeArtifact] = []
    if include_preprocessing:
        artifacts.extend(iter_patient_preprocessing_artifacts(runtime_state.master_structure_reference_dict, all_ref_key))
    if include_patient_mc:
        artifacts.extend(iter_patient_mc_artifacts(runtime_state.master_structure_reference_dict, all_ref_key))
    if include_biopsy_mc:
        artifacts.extend(iter_biopsy_mc_artifacts(runtime_state.master_structure_reference_dict, bx_ref))
    return tuple(artifacts)


def write_patient_dataframe_artifacts(runtime_state: LegacyPatientRuntimeState,
                                      config: PatientRunConfig) -> tuple[Path, ...]:
    """Write one patient's currently available dataframe artifacts."""
    artifact_store = PatientArtifactStore.from_config(config, runtime_state)
    return artifact_store.write(artifact_store.collect(runtime_state))
"""Synthetic checks for retained scientific context artifact contracts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from output_artifacts.context_contracts import ArtifactRef
from output_artifacts.context_contracts import ArrayArtifactSpec
from output_artifacts.context_contracts import PatientArtifactIndex
from output_artifacts.context_contracts import TableArtifactSpec
from output_artifacts.context_contracts import read_patient_artifact_index
from output_artifacts.context_contracts import write_patient_artifact_index


class ScientificContextContractTests(unittest.TestCase):
    def test_artifact_ref_normalizes_storage_and_path(self) -> None:
        artifact_ref = _artifact_ref(storage_format=".ZARR", relative_path="context/dose_lattice.zarr")

        self.assertEqual(artifact_ref.storage_format, "zarr")
        self.assertEqual(artifact_ref.relative_path, "context/dose_lattice.zarr")
        self.assertEqual(artifact_ref.reader, "mc.simulation.per_patient.dose_context_artifacts.read")

    def test_artifact_ref_rejects_absolute_paths_and_unknown_formats(self) -> None:
        with self.assertRaisesRegex(ValueError, "relative_path"):
            _artifact_ref(relative_path="/tmp/context.zarr")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            _artifact_ref(storage_format="pickle")

    def test_array_artifact_spec_validates_rank_and_chunks(self) -> None:
        artifact_ref = _artifact_ref()
        spec = ArrayArtifactSpec(
            artifact_ref=artifact_ref,
            dataset_name="dose_values",
            symbolic_shape=("n_lattice",),
            shape=(128,),
            dtype="float64",
            units="Gy",
            coordinate_frame="patient_physical_mm",
            dimension_names=("lattice_point",),
            chunk_shape=(64,),
            compressor="zstd",
        )

        self.assertEqual(spec.shape, (128,))
        self.assertEqual(spec.chunk_shape, (64,))
        self.assertEqual(spec.to_dict()["artifact_ref"]["artifact_id"], artifact_ref.artifact_id)
        with self.assertRaisesRegex(ValueError, "symbolic_shape"):
            ArrayArtifactSpec(
                artifact_ref=artifact_ref,
                dataset_name="dose_values",
                symbolic_shape=("n", "m"),
                shape=(128,),
                dtype="float64",
            )

    def test_table_artifact_spec_validates_columns_and_row_count(self) -> None:
        artifact_ref = _artifact_ref(
            artifact_id="pointwise_dose_table",
            artifact_family="dose_table",
            relative_path="tables/dosimetry/pointwise_dose.parquet",
            storage_format="parquet",
        )
        spec = TableArtifactSpec(
            artifact_ref=artifact_ref,
            table_name="pointwise_dose",
            columns=("patient_uid", "biopsy_index", "trial_number", "dose"),
            row_count=12,
            primary_key_columns=("patient_uid", "biopsy_index", "trial_number"),
        )

        self.assertEqual(spec.row_count, 12)
        with self.assertRaisesRegex(ValueError, "columns"):
            TableArtifactSpec(artifact_ref=artifact_ref, table_name="bad", columns=())

    def test_patient_artifact_index_accumulates_refs_without_mutating(self) -> None:
        first_ref = _artifact_ref()
        second_ref = _artifact_ref(
            artifact_id="dose_nn_context",
            artifact_family="dose_nn_context",
            relative_path="context/dosimetry/dose_nn_context.zarr",
        )
        index = PatientArtifactIndex(patient_uid="synthetic")

        updated_index = index.add_artifact(first_ref).add_artifact(second_ref)

        self.assertEqual(len(index.artifacts), 0)
        self.assertEqual(len(updated_index.artifacts), 2)
        self.assertIn("dose_nn_context", updated_index.artifacts_by_id)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            updated_index.add_artifact(first_ref)

    def test_patient_artifact_index_round_trip(self) -> None:
        index = PatientArtifactIndex(
            patient_uid="synthetic",
            run_id="run_1",
            metadata={"retention_level": "context"},
        ).add_artifact(_artifact_ref())

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory).joinpath("patients", "synthetic", "manifest.json")
            written_path = write_patient_artifact_index(index, output_path)
            loaded_index = read_patient_artifact_index(written_path)

        self.assertEqual(loaded_index.patient_uid, "synthetic")
        self.assertEqual(loaded_index.run_id, "run_1")
        self.assertEqual(loaded_index.metadata["retention_level"], "context")
        self.assertEqual(loaded_index.artifacts[0].artifact_id, "dose_lattice")


def _artifact_ref(
    *,
    artifact_id: str = "dose_lattice",
    artifact_family: str = "dose_lattice_context",
    relative_path: str = "context/dosimetry/dose_lattice.zarr",
    storage_format: str = "zarr",
) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        title="Dose lattice",
        artifact_family=artifact_family,
        relative_path=relative_path,
        storage_format=storage_format,
        schema_version="dose_lattice_context_v1",
        patient_uid="synthetic",
        stage_name="mc_dosimetry",
        retention_level="context",
        producer="mc.simulation.per_patient.dose_context_artifacts.write",
        reader="mc.simulation.per_patient.dose_context_artifacts.read",
    )


if __name__ == "__main__":
    unittest.main()
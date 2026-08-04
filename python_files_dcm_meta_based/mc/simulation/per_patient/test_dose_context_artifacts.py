"""Synthetic checks for dose scientific context artifact specs."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mc.simulation.per_patient.dose import MC_DOSE_VALUE_COLUMN
from mc.simulation.per_patient.dose import PatientDoseBiopsyContext
from mc.simulation.per_patient.dose import PatientDoseLatticeContext
from mc.simulation.per_patient.dose import PatientDoseLocalizationOutputs
from mc.simulation.per_patient.dose_context_artifacts import build_patient_dose_biopsy_query_context_artifact_plan
from mc.simulation.per_patient.dose_context_artifacts import build_patient_dose_lattice_context_artifact_plan
from mc.simulation.per_patient.dose_context_artifacts import build_patient_dose_localization_context_artifact_plan
from mc.simulation.per_patient.dose_context_artifacts import open_patient_dose_zarr_array_artifact
from mc.simulation.per_patient.dose_context_artifacts import patient_dose_lattice_context_array_payload
from mc.simulation.per_patient.dose_context_artifacts import patient_dose_localization_context_array_payload
from mc.simulation.per_patient.dose_context_artifacts import write_patient_dose_context_zarr_arrays


class DoseContextArtifactPlanTests(unittest.TestCase):
    def test_lattice_context_plan_describes_zarr_arrays_without_storage_io(self) -> None:
        lattice_context = PatientDoseLatticeContext(
            patient_uid="synthetic_patient",
            localization_kind="dose",
            dose_reference_dict={},
            source_dose_and_gradient_array=np.zeros((2, 3, 14), dtype=np.float32),
            localization_map_array=np.zeros((2, 3, 7), dtype=np.float32),
            localization_map_flattened=np.zeros((6, 7), dtype=np.float32),
            physical_coordinates=np.zeros((6, 3), dtype=np.float64),
            sampled_values=np.arange(6, dtype=np.float64),
            kdtree=None,
            result_column=MC_DOSE_VALUE_COLUMN,
            output_key="dose_values",
            kdtree_key="dose_kdtree",
        )

        plan = build_patient_dose_lattice_context_artifact_plan(lattice_context)

        self.assertEqual(plan.patient_uid, "synthetic_patient")
        self.assertEqual(plan.artifact_refs[0].storage_format, "zarr")
        self.assertEqual(plan.artifact_refs[0].relative_path, "context/dosimetry/dose/lattice.zarr")
        self.assertEqual(plan.artifact_refs[0].metadata["kdtree_persisted"], False)
        self.assertEqual(plan.array_specs_by_dataset["physical_coordinates"].shape, (6, 3))
        self.assertEqual(plan.array_specs_by_dataset["sampled_values"].units, "Gy")

    def test_biopsy_query_context_plan_records_query_point_shapes(self) -> None:
        biopsy_context = PatientDoseBiopsyContext(
            patient_uid="synthetic_patient",
            biopsy_index=2,
            num_sample_points=4,
            roi="ROI_2",
            ref_number="BX2",
            simulated_bool=False,
            simulated_type="nominal",
            unshifted_sampled_points=np.zeros((4, 3), dtype=np.float64),
            sampled_points_bx_coord_sys=np.ones((4, 3), dtype=np.float64),
            bx_only_shifted_points=np.zeros((3, 4, 3), dtype=np.float64),
            bx_only_shifted_points_cutoff=np.zeros((2, 4, 3), dtype=np.float64),
            nominal_and_shifted_points=np.zeros((3, 4, 3), dtype=np.float64),
            stacked_nominal_and_shifted_points=np.zeros((12, 3), dtype=np.float64),
            biopsy_structure_info={"roi": "ROI_2"},
        )

        plan = build_patient_dose_biopsy_query_context_artifact_plan(biopsy_context)

        self.assertEqual(plan.artifact_refs[0].artifact_id, "biopsy_002_query_context")
        self.assertEqual(plan.artifact_refs[0].relative_path, "context/dosimetry/biopsy_002/query_points.zarr")
        self.assertEqual(plan.array_specs_by_dataset["nominal_and_shifted_points"].shape, (3, 4, 3))
        self.assertEqual(
            plan.array_specs_by_dataset["sampled_points_bx_coord_sys"].coordinate_frame,
            "biopsy_coordinate_system_mm",
        )

    def test_localization_context_plan_records_values_and_nn_rows(self) -> None:
        outputs = PatientDoseLocalizationOutputs(
            localization_kind="dose",
            result_column=MC_DOSE_VALUE_COLUMN,
            output_key="dose_values",
            nearest_neighbour_dataframe=_FakeNearestNeighbourRows(),
            values_by_point_nominal_and_trials=np.zeros((4, 3), dtype=np.float64),
        )

        plan = build_patient_dose_localization_context_artifact_plan(
            outputs,
            patient_uid="synthetic_patient",
            biopsy_index=2,
        )

        self.assertEqual(len(plan.artifact_refs), 2)
        self.assertEqual(plan.artifact_refs[0].relative_path, "context/dosimetry/dose/biopsy_002/localization_values.zarr")
        self.assertEqual(plan.artifact_refs[1].storage_format, "parquet")
        self.assertEqual(plan.array_specs_by_dataset["values_by_point_nominal_and_trials"].shape, (4, 3))
        self.assertEqual(plan.table_specs_by_name["nearest_neighbour_rows"].row_count, 12)
        self.assertEqual(
            plan.to_patient_artifact_index(run_id="run_1").artifacts_by_id["dose_biopsy_002_nearest_neighbour_rows"].storage_format,
            "parquet",
        )

    def test_write_and_lazy_read_lattice_zarr_arrays(self) -> None:
        lattice_context = PatientDoseLatticeContext(
            patient_uid="synthetic_patient",
            localization_kind="dose",
            dose_reference_dict={},
            source_dose_and_gradient_array=np.arange(84, dtype=np.float32).reshape(2, 3, 14),
            localization_map_array=np.arange(42, dtype=np.float32).reshape(2, 3, 7),
            localization_map_flattened=np.arange(42, dtype=np.float32).reshape(6, 7),
            physical_coordinates=np.arange(18, dtype=np.float64).reshape(6, 3),
            sampled_values=np.arange(6, dtype=np.float64),
            kdtree=None,
            result_column=MC_DOSE_VALUE_COLUMN,
            output_key="dose_values",
            kdtree_key="dose_kdtree",
        )
        plan = build_patient_dose_lattice_context_artifact_plan(lattice_context)

        with tempfile.TemporaryDirectory() as temporary_directory:
            written_paths = write_patient_dose_context_zarr_arrays(
                plan,
                patient_dose_lattice_context_array_payload(lattice_context),
                Path(temporary_directory),
            )
            reader = open_patient_dose_zarr_array_artifact(plan.artifact_refs[0], Path(temporary_directory))

            self.assertTrue(written_paths["dose_lattice_context"].exists())
            self.assertEqual(reader.dataset_shape("physical_coordinates"), (6, 3))
            np.testing.assert_array_equal(reader.read_array("sampled_values", np.s_[2:5]), np.array([2.0, 3.0, 4.0]))
            self.assertEqual(reader.open_group().attrs["artifact_ref"]["artifact_id"], "dose_lattice_context")

    def test_write_localization_zarr_array_ignores_parquet_table_spec(self) -> None:
        outputs = PatientDoseLocalizationOutputs(
            localization_kind="dose",
            result_column=MC_DOSE_VALUE_COLUMN,
            output_key="dose_values",
            nearest_neighbour_dataframe=_FakeNearestNeighbourRows(),
            values_by_point_nominal_and_trials=np.arange(12, dtype=np.float64).reshape(4, 3),
        )
        plan = build_patient_dose_localization_context_artifact_plan(
            outputs,
            patient_uid="synthetic_patient",
            biopsy_index=2,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            written_paths = write_patient_dose_context_zarr_arrays(
                plan,
                patient_dose_localization_context_array_payload(outputs),
                Path(temporary_directory),
            )
            reader = open_patient_dose_zarr_array_artifact(plan.artifact_refs[0], Path(temporary_directory))

            self.assertEqual(tuple(written_paths), ("dose_biopsy_002_localization_values",))
            np.testing.assert_array_equal(
                reader.read_array("values_by_point_nominal_and_trials", np.s_[:, 1]),
                np.array([1.0, 4.0, 7.0, 10.0]),
            )


class _FakeNearestNeighbourRows:
    columns = ("Trial num", "Original pt index", MC_DOSE_VALUE_COLUMN)

    def __len__(self) -> int:
        return 12


if __name__ == "__main__":
    unittest.main()
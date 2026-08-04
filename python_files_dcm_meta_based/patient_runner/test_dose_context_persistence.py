"""Synthetic tests for patient-runner dose context persistence."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from output_artifacts.context_contracts import read_patient_artifact_index

from mc.simulation.per_patient.dose import MC_DOSE_ORIGINAL_POINT_INDEX_COLUMN
from mc.simulation.per_patient.dose import MC_DOSE_TRIAL_COLUMN
from mc.simulation.per_patient.dose import MC_DOSE_VALUE_COLUMN
from mc.simulation.per_patient.dose import PatientDoseBiopsyContext
from mc.simulation.per_patient.dose import PatientDoseLatticeContext
from mc.simulation.per_patient.dose import PatientDoseLocalizationOutputs
from mc.visualization.dose_nn_context_render_service import materialize_dose_nn_saved_scene_artifact_from_patient_index
from mc.visualization.dose_nn_scene import DOSE_NN_NEAREST_DISTANCES_COLUMN
from mc.visualization.dose_nn_scene import DOSE_NN_NEAREST_DOSES_COLUMN
from mc.visualization.dose_nn_scene import DOSE_NN_NEAREST_POINTS_COLUMN
from mc.visualization.dose_nn_scene import DOSE_NN_QUERY_POINT_COLUMN
from mc.visualization.dose_nn_scene_artifacts import read_dose_nn_render_scene_artifact

from patient_runner.dose_context_persistence import PatientDoseContextArtifactPersister


class PatientDoseContextArtifactPersisterTests(unittest.TestCase):
    def test_persists_context_index_and_render_context_from_finalized_dose_output(self) -> None:
        lattice_context = _synthetic_lattice_context()
        biopsy_context = _synthetic_biopsy_context()
        localization_outputs = _synthetic_localization_outputs()

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            persister = PatientDoseContextArtifactPersister(
                patient_uid="synthetic_patient",
                output_root=output_root,
                run_id="run_1",
            )
            summary = persister.persist_dose_localization_context(
                lattice_context=lattice_context,
                biopsy_context=biopsy_context,
                localization_outputs=localization_outputs,
            )
            index = read_patient_artifact_index(summary.artifact_index_path)
            scene_artifact_dir = output_root.joinpath("render_scenes", "scene_from_persisted_context")
            materialize_dose_nn_saved_scene_artifact_from_patient_index(
                patient_artifact_index_path=summary.artifact_index_path,
                output_root=output_root,
                biopsy_index=2,
                scene_artifact_dir=scene_artifact_dir,
                scene_id="scene_from_persisted_context",
            )
            loaded_scene = read_dose_nn_render_scene_artifact(scene_artifact_dir)

        artifact_ids = set(index.artifacts_by_id)
        self.assertIn("dose_lattice_context", artifact_ids)
        self.assertIn("biopsy_002_query_context", artifact_ids)
        self.assertIn("dose_biopsy_002_localization_values", artifact_ids)
        self.assertIn("dose_biopsy_002_render_context", artifact_ids)
        self.assertNotIn("dose_biopsy_002_nearest_neighbour_rows", artifact_ids)
        self.assertEqual(index.run_id, "run_1")
        self.assertEqual(loaded_scene.metadata.biopsy_index, 2)
        np.testing.assert_array_equal(loaded_scene.nearest_distances, np.array([[1.0, 1.4], [1.0, 1.4], [1.1, 1.5], [1.1, 1.5]]))


def _synthetic_lattice_context() -> PatientDoseLatticeContext:
    lattice_points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    lattice_doses = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    localization_map_flattened = np.zeros((4, 7), dtype=np.float64)
    localization_map_flattened[:, 3:6] = lattice_points
    localization_map_flattened[:, 6] = lattice_doses
    return PatientDoseLatticeContext(
        patient_uid="synthetic_patient",
        localization_kind="dose",
        dose_reference_dict={},
        source_dose_and_gradient_array=np.zeros((2, 2, 14), dtype=np.float64),
        localization_map_array=np.zeros((2, 2, 7), dtype=np.float64),
        localization_map_flattened=localization_map_flattened,
        physical_coordinates=lattice_points,
        sampled_values=lattice_doses,
        kdtree=None,
        result_column=MC_DOSE_VALUE_COLUMN,
        output_key="dose_values",
        kdtree_key="dose_kdtree",
    )


def _synthetic_biopsy_context() -> PatientDoseBiopsyContext:
    return PatientDoseBiopsyContext(
        patient_uid="synthetic_patient",
        biopsy_index=2,
        num_sample_points=2,
        roi="ROI_2",
        ref_number="BX2",
        simulated_bool=False,
        simulated_type="nominal",
        unshifted_sampled_points=np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float64),
        sampled_points_bx_coord_sys=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        bx_only_shifted_points=np.zeros((2, 2, 3), dtype=np.float64),
        bx_only_shifted_points_cutoff=np.zeros((2, 2, 3), dtype=np.float64),
        nominal_and_shifted_points=np.zeros((3, 2, 3), dtype=np.float64),
        stacked_nominal_and_shifted_points=np.zeros((6, 3), dtype=np.float64),
        biopsy_structure_info={"roi": "ROI_2"},
    )


def _synthetic_localization_outputs() -> PatientDoseLocalizationOutputs:
    dataframe = pd.DataFrame(
        {
            MC_DOSE_ORIGINAL_POINT_INDEX_COLUMN: [0, 1, 0, 1],
            MC_DOSE_TRIAL_COLUMN: [0, 0, 1, 1],
            DOSE_NN_QUERY_POINT_COLUMN: [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [1.1, 0.0, 1.0],
            ],
            MC_DOSE_VALUE_COLUMN: [11.0, 22.0, 12.0, 23.0],
            DOSE_NN_NEAREST_POINTS_COLUMN: [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            ],
            DOSE_NN_NEAREST_DOSES_COLUMN: [[10.0, 20.0], [20.0, 30.0], [10.0, 20.0], [20.0, 30.0]],
            DOSE_NN_NEAREST_DISTANCES_COLUMN: [[1.0, 1.4], [1.0, 1.4], [1.1, 1.5], [1.1, 1.5]],
        }
    )
    return PatientDoseLocalizationOutputs(
        localization_kind="dose",
        result_column=MC_DOSE_VALUE_COLUMN,
        output_key="dose_values",
        nearest_neighbour_dataframe=dataframe,
        values_by_point_nominal_and_trials=np.array([[11.0, 12.0], [22.0, 23.0]], dtype=np.float64),
    )


if __name__ == "__main__":
    unittest.main()
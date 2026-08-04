"""Synthetic checks for rebuilding dose NN scenes from retained context."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mc.simulation.per_patient.dose import MC_DOSE_VALUE_COLUMN
from mc.simulation.per_patient.dose import PatientDoseLatticeContext
from mc.simulation.per_patient.dose_context_artifacts import build_patient_dose_lattice_context_artifact_plan
from mc.simulation.per_patient.dose_context_artifacts import patient_dose_lattice_context_array_payload
from mc.simulation.per_patient.dose_context_artifacts import write_patient_dose_context_zarr_arrays
from mc.visualization.dose_nn_context_bridge import build_dose_nn_render_context_artifact_plan
from mc.visualization.dose_nn_context_bridge import build_dose_nn_render_scene_from_context_artifacts
from mc.visualization.dose_nn_context_bridge import write_dose_nn_render_context_zarr_artifact
from mc.visualization.dose_nn_scene import DoseNNSceneMetadata
from mc.visualization.dose_nn_scene import build_dose_nn_render_scene


class DoseNNContextBridgeTests(unittest.TestCase):
    def test_render_scene_rebuilt_from_zarr_context_matches_runtime_scene(self) -> None:
        runtime_scene = _synthetic_runtime_scene()
        lattice_context = _synthetic_lattice_context(runtime_scene)
        lattice_plan = build_patient_dose_lattice_context_artifact_plan(lattice_context)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            write_patient_dose_context_zarr_arrays(
                lattice_plan,
                patient_dose_lattice_context_array_payload(lattice_context),
                output_root,
            )
            render_plan, written_paths = write_dose_nn_render_context_zarr_artifact(runtime_scene, output_root)

            rebuilt_scene = build_dose_nn_render_scene_from_context_artifacts(
                lattice_artifact_ref=lattice_plan.artifact_refs[0],
                render_context_artifact_ref=render_plan.artifact_refs[0],
                output_root=output_root,
            )

        self.assertIn("dose_biopsy_002_render_context", written_paths)
        self.assertEqual(rebuilt_scene.metadata.patient_uid, runtime_scene.metadata.patient_uid)
        self.assertEqual(rebuilt_scene.metadata.biopsy_index, runtime_scene.metadata.biopsy_index)
        np.testing.assert_array_equal(rebuilt_scene.lattice_points, runtime_scene.lattice_points)
        np.testing.assert_array_equal(rebuilt_scene.lattice_doses, runtime_scene.lattice_doses)
        np.testing.assert_array_equal(rebuilt_scene.original_point_indices, runtime_scene.original_point_indices)
        np.testing.assert_array_equal(rebuilt_scene.trial_numbers, runtime_scene.trial_numbers)
        np.testing.assert_array_equal(rebuilt_scene.biopsy_points, runtime_scene.biopsy_points)
        np.testing.assert_array_equal(rebuilt_scene.interpolated_biopsy_doses, runtime_scene.interpolated_biopsy_doses)
        np.testing.assert_array_equal(rebuilt_scene.nearest_lattice_points, runtime_scene.nearest_lattice_points)
        np.testing.assert_array_equal(rebuilt_scene.nearest_lattice_doses, runtime_scene.nearest_lattice_doses)
        np.testing.assert_array_equal(rebuilt_scene.nearest_distances, runtime_scene.nearest_distances)

    def test_render_context_plan_does_not_duplicate_lattice_arrays(self) -> None:
        plan = build_dose_nn_render_context_artifact_plan(_synthetic_runtime_scene())

        self.assertNotIn("lattice_points", plan.array_specs_by_dataset)
        self.assertNotIn("lattice_doses", plan.array_specs_by_dataset)
        self.assertIn("nearest_lattice_points", plan.array_specs_by_dataset)
        self.assertEqual(plan.artifact_refs[0].relative_path, "context/dosimetry/dose/biopsy_002/render_context.zarr")


def _synthetic_runtime_scene():
    return build_dose_nn_render_scene(
        metadata=DoseNNSceneMetadata(
            patient_uid="synthetic_patient",
            biopsy_roi="ROI_2",
            biopsy_index=2,
            localization_kind="dose",
            result_column=MC_DOSE_VALUE_COLUMN,
            source_label="runtime_synthetic",
        ),
        lattice_points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        lattice_doses=np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64),
        original_point_indices=np.array([0, 1, 0, 1], dtype=np.int64),
        trial_numbers=np.array([0, 0, 1, 1], dtype=np.int64),
        biopsy_points=np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [1.1, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        interpolated_biopsy_doses=np.array([11.0, 22.0, 12.0, 23.0], dtype=np.float64),
        nearest_lattice_points=np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            ],
            dtype=np.float64,
        ),
        nearest_lattice_doses=np.array(
            [
                [10.0, 20.0],
                [20.0, 30.0],
                [10.0, 20.0],
                [20.0, 30.0],
            ],
            dtype=np.float64,
        ),
        nearest_distances=np.array(
            [
                [1.0, 1.4],
                [1.0, 1.4],
                [1.1, 1.5],
                [1.1, 1.5],
            ],
            dtype=np.float64,
        ),
    )


def _synthetic_lattice_context(scene):
    localization_map_flattened = np.zeros((scene.lattice_points.shape[0], 7), dtype=np.float64)
    localization_map_flattened[:, 3:6] = scene.lattice_points
    localization_map_flattened[:, 6] = scene.lattice_doses
    return PatientDoseLatticeContext(
        patient_uid=scene.metadata.patient_uid,
        localization_kind=scene.metadata.localization_kind,
        dose_reference_dict={},
        source_dose_and_gradient_array=np.zeros((2, 2, 14), dtype=np.float64),
        localization_map_array=np.zeros((2, 2, 7), dtype=np.float64),
        localization_map_flattened=localization_map_flattened,
        physical_coordinates=scene.lattice_points,
        sampled_values=scene.lattice_doses,
        kdtree=None,
        result_column=scene.metadata.result_column,
        output_key="dose_values",
        kdtree_key="dose_kdtree",
    )


if __name__ == "__main__":
    unittest.main()
"""Synthetic checks for dose NN render scene artifact IO."""

from __future__ import annotations

import json
import tempfile
import unittest

import numpy as np

from mc.visualization.dose_nn_scene import DoseNNSceneMetadata, build_dose_nn_render_scene
from mc.visualization.dose_nn_scene_artifacts import (
    DOSE_NN_RENDER_SCENE_ARRAYS_FILENAME,
    DOSE_NN_RENDER_SCENE_MANIFEST_FILENAME,
    read_dose_nn_render_scene_artifact,
    read_dose_nn_render_scene_artifact_manifest,
    write_dose_nn_render_scene_artifact,
)


class DoseNNRenderSceneArtifactTests(unittest.TestCase):
    def test_scene_artifact_round_trip_preserves_arrays_and_metadata(self):
        scene = _synthetic_scene()

        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = write_dose_nn_render_scene_artifact(
                scene,
                temporary_directory,
                scene_id="synthetic_scene",
            )
            loaded_scene = read_dose_nn_render_scene_artifact(temporary_directory)

        self.assertEqual(manifest.scene_id, "synthetic_scene")
        self.assertEqual(loaded_scene.metadata.patient_uid, "synthetic")
        self.assertEqual(loaded_scene.metadata.extra["note"], "round trip")
        np.testing.assert_array_equal(loaded_scene.lattice_points, scene.lattice_points)
        np.testing.assert_array_equal(loaded_scene.nearest_lattice_points, scene.nearest_lattice_points)
        np.testing.assert_array_equal(loaded_scene.nearest_distances, scene.nearest_distances)

    def test_write_existing_artifact_fails_without_overwrite(self):
        scene = _synthetic_scene()

        with tempfile.TemporaryDirectory() as temporary_directory:
            write_dose_nn_render_scene_artifact(scene, temporary_directory, scene_id="synthetic_scene")
            with self.assertRaises(FileExistsError):
                write_dose_nn_render_scene_artifact(scene, temporary_directory, scene_id="synthetic_scene")

    def test_manifest_checksum_mismatch_fails_closed(self):
        scene = _synthetic_scene()

        with tempfile.TemporaryDirectory() as temporary_directory:
            write_dose_nn_render_scene_artifact(scene, temporary_directory, scene_id="synthetic_scene")
            manifest_path = _manifest_path(temporary_directory)
            with open(manifest_path, "r", encoding="utf-8") as manifest_file:
                manifest_payload = json.load(manifest_file)
            manifest_payload["arrays"][0]["sha256"] = "0" * 64
            with open(manifest_path, "w", encoding="utf-8") as manifest_file:
                json.dump(manifest_payload, manifest_file)

            with self.assertRaisesRegex(ValueError, "checksum"):
                read_dose_nn_render_scene_artifact(temporary_directory)

    def test_manifest_reader_exposes_array_specs_without_loading_arrays(self):
        scene = _synthetic_scene()

        with tempfile.TemporaryDirectory() as temporary_directory:
            write_dose_nn_render_scene_artifact(scene, temporary_directory, scene_id="synthetic_scene")
            manifest = read_dose_nn_render_scene_artifact_manifest(temporary_directory)

        self.assertEqual(manifest.arrays_filename, DOSE_NN_RENDER_SCENE_ARRAYS_FILENAME)
        self.assertIn("lattice_points", manifest.array_specs_by_name)


def _synthetic_scene():
    return build_dose_nn_render_scene(
        metadata=DoseNNSceneMetadata(
            patient_uid="synthetic",
            biopsy_roi="Bx 1",
            biopsy_index=2,
            source_label="unit-test",
            extra={"note": "round trip"},
        ),
        lattice_points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        lattice_doses=np.array([10.0, 20.0, 30.0]),
        original_point_indices=np.array([0, 1]),
        trial_numbers=np.array([0, 0]),
        biopsy_points=np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
        interpolated_biopsy_doses=np.array([11.0, 22.0]),
        nearest_lattice_points=np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            ]
        ),
        nearest_lattice_doses=np.array([[10.0, 20.0], [20.0, 30.0]]),
        nearest_distances=np.array([[1.0, 1.4], [1.0, 1.4]]),
    )


def _manifest_path(temporary_directory: str):
    from pathlib import Path

    return Path(temporary_directory).joinpath(DOSE_NN_RENDER_SCENE_MANIFEST_FILENAME)


if __name__ == "__main__":
    unittest.main()

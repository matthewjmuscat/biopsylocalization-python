"""Synthetic checks for saved-scene dose NN rendering service."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mc.visualization.dose_nn_pyvista import DoseNNPyVistaRenderSettings
from mc.visualization.dose_nn_pyvista import is_pyvista_available
from mc.visualization.dose_nn_render_service import export_saved_dose_nn_scene_trial_frames_pyvista
from mc.visualization.dose_nn_render_service import main
from mc.visualization.dose_nn_render_service import render_saved_dose_nn_scene_artifact_pyvista
from mc.visualization.dose_nn_scene import DoseNNRenderConfig, DoseNNSceneMetadata, build_dose_nn_render_scene
from mc.visualization.dose_nn_scene_artifacts import write_dose_nn_render_scene_artifact


class DoseNNRenderServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        if not is_pyvista_available():
            self.skipTest("PyVista is not available in this environment")

    def test_render_saved_scene_artifact_writes_export_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            scene_dir = Path(temporary_directory).joinpath("scene")
            output_path = Path(temporary_directory).joinpath("exports", "dose_nn.png")
            write_dose_nn_render_scene_artifact(_synthetic_scene(), scene_dir, scene_id="synthetic_scene")

            result = render_saved_dose_nn_scene_artifact_pyvista(
                scene_dir,
                output_path,
                config=DoseNNRenderConfig(selected_trials=(0,), vector_stride=2),
                settings=_test_settings(),
            )

            self.assertTrue(result.screenshot_path.is_file())
            self.assertTrue(result.provenance_path.is_file())
            self.assertGreater(result.screenshot_path.stat().st_size, 0)

    def test_cli_renders_saved_scene_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            scene_dir = Path(temporary_directory).joinpath("scene")
            output_path = Path(temporary_directory).joinpath("cli", "dose_nn.png")
            write_dose_nn_render_scene_artifact(_synthetic_scene(), scene_dir, scene_id="synthetic_scene")

            exit_code = main(
                (
                    "--scene-dir",
                    str(scene_dir),
                    "--output",
                    str(output_path),
                    "--trial",
                    "0",
                    "--vector-stride",
                    "2",
                    "--show-dose-colorwash",
                    "--dose-colorwash-opacity",
                    "0.3",
                    "--window-size",
                    "320",
                    "240",
                    "--no-axes",
                    "--no-scalar-bar",
                )
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.is_file())
            self.assertTrue(output_path.with_suffix(".png.provenance.json").is_file())

    def test_export_saved_scene_trial_frames_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            scene_dir = Path(temporary_directory).joinpath("scene")
            output_dir = Path(temporary_directory).joinpath("frames")
            write_dose_nn_render_scene_artifact(_synthetic_scene(), scene_dir, scene_id="synthetic_scene")

            result = export_saved_dose_nn_scene_trial_frames_pyvista(
                scene_dir,
                output_dir,
                selected_trials=(0,),
                config=DoseNNRenderConfig(show_dose_colorwash=True),
                settings=_test_settings(),
            )

            self.assertEqual(len(result.frame_paths), 1)
            self.assertTrue(result.frame_paths[0].is_file())
            self.assertTrue(result.manifest_path.is_file())

    def test_cli_exports_trial_frames_without_screenshot_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            scene_dir = Path(temporary_directory).joinpath("scene")
            output_dir = Path(temporary_directory).joinpath("frames")
            write_dose_nn_render_scene_artifact(_synthetic_scene(), scene_dir, scene_id="synthetic_scene")

            exit_code = main(
                (
                    "--scene-dir",
                    str(scene_dir),
                    "--export-trial-frames-dir",
                    str(output_dir),
                    "--trial",
                    "0",
                    "--frames-per-second",
                    "6",
                    "--window-size",
                    "320",
                    "240",
                    "--no-axes",
                    "--no-scalar-bar",
                )
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_dir.joinpath("frame_sequence_manifest.json").is_file())


def _test_settings() -> DoseNNPyVistaRenderSettings:
    return DoseNNPyVistaRenderSettings(
        off_screen=True,
        window_size=(320, 240),
        show_axes=False,
        show_scalar_bar=False,
    )


def _synthetic_scene():
    return build_dose_nn_render_scene(
        metadata=DoseNNSceneMetadata(
            patient_uid="synthetic",
            biopsy_roi="Bx 1",
            biopsy_index=1,
            source_label="render-service-test",
        ),
        lattice_points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        lattice_doses=np.array([10.0, 20.0, 30.0, 40.0], dtype=float),
        original_point_indices=np.array([0, 1], dtype=int),
        trial_numbers=np.array([0, 0], dtype=int),
        biopsy_points=np.array([[0.5, 0.0, 1.0], [1.5, 0.0, 1.0]], dtype=float),
        interpolated_biopsy_doses=np.array([15.0, 25.0], dtype=float),
        nearest_lattice_points=np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            ],
            dtype=float,
        ),
        nearest_lattice_doses=np.array([[10.0, 20.0], [20.0, 30.0]], dtype=float),
        nearest_distances=np.array([[1.0, 1.1], [1.0, 1.1]], dtype=float),
    )


if __name__ == "__main__":
    unittest.main()
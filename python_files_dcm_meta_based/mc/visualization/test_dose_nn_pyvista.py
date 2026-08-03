"""Synthetic checks for the PyVista dose NN renderer backend."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mc.visualization.dose_nn_pyvista import DoseNNPyVistaRenderSettings
from mc.visualization.dose_nn_pyvista import build_pyvista_dose_nn_export_provenance
from mc.visualization.dose_nn_pyvista import build_pyvista_dose_nn_plotter
from mc.visualization.dose_nn_pyvista import export_dose_nn_scene_pyvista
from mc.visualization.dose_nn_pyvista import is_pyvista_available
from mc.visualization.dose_nn_scene import DoseNNRenderConfig, DoseNNSceneMetadata, build_dose_nn_render_scene
from mc.visualization.dose_nn_scene import prepare_dose_nn_render_scene


class DoseNNPyVistaRendererTests(unittest.TestCase):
    def setUp(self) -> None:
        if not is_pyvista_available():
            self.skipTest("PyVista is not available in this environment")

    def test_build_plotter_adds_expected_scene_layers(self) -> None:
        prepared_scene = prepare_dose_nn_render_scene(_synthetic_scene())

        plotter = build_pyvista_dose_nn_plotter(prepared_scene, settings=_test_settings())
        try:
            actor_names = set(plotter.renderer.actors.keys())
        finally:
            plotter.close()

        self.assertIn("dose_lattice_points", actor_names)
        self.assertIn("biopsy_query_points", actor_names)
        self.assertIn("dose_nn_nearest_points", actor_names)
        self.assertIn("dose_nn_vectors", actor_names)

    def test_build_plotter_respects_layer_toggles(self) -> None:
        prepared_scene = prepare_dose_nn_render_scene(
            _synthetic_scene(),
            DoseNNRenderConfig(
                show_lattice_points=False,
                show_nearest_neighbour_vectors=False,
            ),
        )

        plotter = build_pyvista_dose_nn_plotter(prepared_scene, settings=_test_settings())
        try:
            actor_names = set(plotter.renderer.actors.keys())
        finally:
            plotter.close()

        self.assertNotIn("dose_lattice_points", actor_names)
        self.assertIn("biopsy_query_points", actor_names)
        self.assertIn("dose_nn_nearest_points", actor_names)
        self.assertNotIn("dose_nn_vectors", actor_names)

    def test_export_provenance_records_scene_and_render_summary(self) -> None:
        prepared_scene = prepare_dose_nn_render_scene(
            _synthetic_scene(),
            DoseNNRenderConfig(selected_trials=(0,), vector_stride=2),
        )

        provenance = build_pyvista_dose_nn_export_provenance(
            prepared_scene,
            settings=_test_settings(),
            screenshot_path=Path("synthetic.png"),
        )

        self.assertEqual(provenance["backend"], "pyvista")
        self.assertEqual(provenance["scene_metadata"]["patient_uid"], "synthetic")
        self.assertEqual(provenance["render_config"]["selected_trials"], [0])
        self.assertEqual(provenance["prepared_scene_summary"]["biopsy_point_count"], 2)
        self.assertEqual(provenance["prepared_scene_summary"]["vector_count"], 2)

    def test_export_writes_screenshot_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory).joinpath("synthetic.png")
            try:
                result = export_dose_nn_scene_pyvista(
                    _synthetic_scene(),
                    output_path,
                    settings=_test_settings(),
                )
            except Exception as exc:
                self.skipTest("PyVista screenshot export is unavailable in this environment: {}".format(exc))

            self.assertTrue(result.screenshot_path.is_file())
            self.assertTrue(result.provenance_path.is_file())
            self.assertGreater(result.screenshot_path.stat().st_size, 0)


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
            source_label="unit-test",
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
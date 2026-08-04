"""Synthetic checks for dose NN saved-scene selector helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mc.visualization.dose_nn_pyvista import DoseNNPyVistaRenderSettings
from mc.visualization.dose_nn_pyvista import is_pyvista_available
from mc.visualization.dose_nn_scene import DoseNNRenderConfig
from mc.visualization.dose_nn_scene import DoseNNSceneMetadata
from mc.visualization.dose_nn_scene import build_dose_nn_render_scene
from mc.visualization.dose_nn_scene_artifacts import DOSE_NN_RENDER_SCENE_MANIFEST_FILENAME
from mc.visualization.dose_nn_scene_artifacts import write_dose_nn_render_scene_artifact
from mc.visualization.dose_nn_selector import DOSE_NN_SAVED_SCENE_GROUP_KEY
from mc.visualization.dose_nn_selector import build_saved_dose_nn_scene_choice_group
from mc.visualization.dose_nn_selector import discover_saved_dose_nn_scene_options
from mc.visualization.dose_nn_selector import render_saved_dose_nn_scene_selection_pyvista


class DoseNNSelectorTests(unittest.TestCase):
    def test_discover_saved_scene_options_reads_manifest_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scene_dir = root.joinpath("patient_a", "render_scenes", "scene_1")
            write_dose_nn_render_scene_artifact(_synthetic_scene(), scene_dir, scene_id="synthetic scene")
            root.joinpath("not_a_scene").mkdir()
            root.joinpath("not_a_scene", DOSE_NN_RENDER_SCENE_MANIFEST_FILENAME).write_text(
                '{"schema_version": "other"}',
                encoding="utf-8",
            )

            options = discover_saved_dose_nn_scene_options(root, suggested_export_root=root.joinpath("exports"))

        self.assertEqual(len(options), 1)
        option = options[0]
        self.assertEqual(option.option_key, "synthetic_scene")
        self.assertEqual(option.scene_id, "synthetic scene")
        self.assertEqual(option.patient_uid, "synthetic")
        self.assertEqual(option.biopsy_roi, "Bx 1")
        self.assertEqual(option.biopsy_index, 1)
        self.assertEqual(option.num_lattice_points, 4)
        self.assertEqual(option.num_query_points, 2)
        self.assertEqual(option.num_nearest_neighbours, 2)
        self.assertIn("query rows: 2", option.display_label)
        self.assertEqual(option.suggested_export_output_dir.name, "synthetic_scene")

    def test_choice_group_uses_saved_scene_options(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            scene_dir = Path(temporary_directory).joinpath("scene")
            write_dose_nn_render_scene_artifact(_synthetic_scene(), scene_dir, scene_id="synthetic_scene")
            options = discover_saved_dose_nn_scene_options(temporary_directory)

        choice_group = build_saved_dose_nn_scene_choice_group(options)

        self.assertEqual(choice_group.group_key, DOSE_NN_SAVED_SCENE_GROUP_KEY)
        self.assertEqual(choice_group.selection_mode, "single")
        self.assertEqual(choice_group.options[0].option_key, "synthetic_scene")
        self.assertTrue(choice_group.options[0].selected_by_default)

    def test_render_unknown_saved_scene_option_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown dose NN saved scene option"):
            render_saved_dose_nn_scene_selection_pyvista((), ("missing",), Path("unused"))

    def test_render_saved_scene_selection_writes_outputs(self) -> None:
        if not is_pyvista_available():
            self.skipTest("PyVista is not available in this environment")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scene_dir = root.joinpath("scene")
            output_dir = root.joinpath("exports")
            write_dose_nn_render_scene_artifact(_synthetic_scene(), scene_dir, scene_id="synthetic_scene")
            options = discover_saved_dose_nn_scene_options(root)

            results = render_saved_dose_nn_scene_selection_pyvista(
                options,
                ("synthetic_scene",),
                output_dir,
                config=DoseNNRenderConfig(selected_trials=(0,), vector_stride=2),
                settings=DoseNNPyVistaRenderSettings(
                    off_screen=True,
                    window_size=(320, 240),
                    show_axes=False,
                    show_scalar_bar=False,
                ),
            )

            self.assertEqual(len(results), 1)
            self.assertTrue(output_dir.joinpath("synthetic_scene.png").is_file())
            self.assertTrue(output_dir.joinpath("synthetic_scene.png.provenance.json").is_file())

            with self.assertRaises(FileExistsError):
                render_saved_dose_nn_scene_selection_pyvista(
                    options,
                    ("synthetic_scene",),
                    output_dir,
                    settings=DoseNNPyVistaRenderSettings(
                        off_screen=True,
                        window_size=(320, 240),
                        show_axes=False,
                        show_scalar_bar=False,
                    ),
                )


def _synthetic_scene():
    return build_dose_nn_render_scene(
        metadata=DoseNNSceneMetadata(
            patient_uid="synthetic",
            biopsy_roi="Bx 1",
            biopsy_index=1,
            source_label="selector-test",
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
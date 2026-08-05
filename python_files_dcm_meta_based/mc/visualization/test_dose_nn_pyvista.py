"""Synthetic checks for the PyVista dose NN renderer backend."""

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np

from mc.visualization.dose_nn_pyvista import DoseNNPyVistaRenderSettings
from mc.visualization.dose_nn_pyvista import build_pyvista_dose_nn_export_provenance
from mc.visualization.dose_nn_pyvista import build_pyvista_dose_nn_plotter
from mc.visualization.dose_nn_pyvista import export_dose_nn_trial_frame_sequence_pyvista
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

    def test_build_plotter_supports_dose_colorwash_layer(self) -> None:
        prepared_scene = prepare_dose_nn_render_scene(
            _synthetic_scene(),
            DoseNNRenderConfig(
                show_lattice_points=False,
                show_dose_colorwash=True,
            ),
        )

        plotter = build_pyvista_dose_nn_plotter(prepared_scene, settings=_test_settings())
        try:
            actor_names = set(plotter.renderer.actors.keys())
        finally:
            plotter.close()

        self.assertNotIn("dose_lattice_points", actor_names)
        self.assertIn("dose_colorwash_points", actor_names)
        self.assertIn("dose_nn_vectors", actor_names)

    def test_scalar_bar_background_renders_behind_foreground_props(self) -> None:
        prepared_scene = prepare_dose_nn_render_scene(_synthetic_scene())

        plotter = build_pyvista_dose_nn_plotter(
            prepared_scene,
            settings=DoseNNPyVistaRenderSettings(
                off_screen=True,
                window_size=(320, 240),
                show_axes=False,
                show_scalar_bar=True,
                dose_scalar_bar_show_background=True,
            ),
        )
        try:
            view_props = plotter.renderer.GetViewProps()
            view_props.InitTraversal()
            display_locations = []
            scalar_bar_titles = []
            scalar_bar_frame_flags = []
            scalar_bar_background_flags = []
            scalar_bar_title_separations = []
            for _ in range(view_props.GetNumberOfItems()):
                prop = view_props.GetNextProp()
                if prop.GetClassName() == "vtkScalarBarActor":
                    scalar_bar_titles.append(prop.GetTitle())
                    scalar_bar_frame_flags.append(int(prop.GetDrawFrame()))
                    scalar_bar_background_flags.append(int(prop.GetDrawBackground()))
                    scalar_bar_title_separations.append(int(prop.GetVerticalTitleSeparation()))
                prop_property = prop.GetProperty() if hasattr(prop, "GetProperty") else None
                if prop_property is not None and hasattr(prop_property, "GetDisplayLocation"):
                    display_locations.append(int(prop_property.GetDisplayLocation()))
        finally:
            plotter.close()

        self.assertIn("Dose (Gy)", scalar_bar_titles)
        self.assertEqual(scalar_bar_frame_flags, [0])
        self.assertEqual(scalar_bar_background_flags, [0])
        self.assertEqual(scalar_bar_title_separations, [12])
        self.assertIn(0, display_locations)
        self.assertIn(1, display_locations)

    def test_build_plotter_supports_logarithmic_dose_color_scaling(self) -> None:
        prepared_scene = prepare_dose_nn_render_scene(_synthetic_scene())

        plotter = build_pyvista_dose_nn_plotter(
            prepared_scene,
            settings=DoseNNPyVistaRenderSettings(
                off_screen=True,
                window_size=(320, 240),
                dose_color_scale_mode="log",
                dose_color_scale_min=10.0,
                dose_color_scale_max=40.0,
                show_axes=False,
                show_scalar_bar=False,
            ),
        )
        try:
            actor = plotter.renderer.actors["dose_lattice_points"]
            log_scale_enabled = bool(actor.mapper.lookup_table.log_scale)
        finally:
            plotter.close()

        self.assertTrue(log_scale_enabled)

    def test_build_plotter_adds_custom_world_axis_labels(self) -> None:
        prepared_scene = prepare_dose_nn_render_scene(_synthetic_scene())

        plotter = build_pyvista_dose_nn_plotter(
            prepared_scene,
            settings=DoseNNPyVistaRenderSettings(
                off_screen=True,
                window_size=(320, 240),
                show_axes=True,
                show_scalar_bar=False,
                dose_scalar_bar_title="Dose (Gy)",
                x_axis_label="Custom X (mm)",
                y_axis_label="Custom Y (mm)",
                z_axis_label="Custom Z (mm)",
                axes_title_font_size=21,
                axes_tick_label_font_size=13,
            ),
        )
        try:
            cube_axes_actor = plotter.renderer.cube_axes_actor
            axis_titles = (
                cube_axes_actor.GetXTitle(),
                cube_axes_actor.GetYTitle(),
                cube_axes_actor.GetZTitle(),
            )
            x_title_font_size = cube_axes_actor.GetTitleTextProperty(0).GetFontSize()
            x_tick_font_size = cube_axes_actor.GetLabelTextProperty(0).GetFontSize()
        finally:
            plotter.close()

        self.assertEqual(axis_titles, ("Custom X (mm)", "Custom Y (mm)", "Custom Z (mm)"))
        self.assertEqual(x_title_font_size, 21)
        self.assertEqual(x_tick_font_size, 13)

    def test_point_colorwash_supports_center_fade_opacity(self) -> None:
        prepared_scene = prepare_dose_nn_render_scene(
            _synthetic_scene(),
            DoseNNRenderConfig(
                show_lattice_points=False,
                show_dose_colorwash=True,
                spatial_radius_mm=3.0,
            ),
        )

        plotter = build_pyvista_dose_nn_plotter(
            prepared_scene,
            settings=DoseNNPyVistaRenderSettings(
                off_screen=True,
                window_size=(320, 240),
                dose_colorwash_style="points",
                dose_colorwash_opacity=0.2,
                dose_colorwash_point_opacity_mode="center_fade",
                dose_colorwash_point_opacity_min=0.02,
                show_axes=False,
                show_scalar_bar=False,
            ),
        )
        try:
            actor = plotter.renderer.actors["dose_colorwash_points"]
            opacity_values = actor.mapper.dataset.point_data["dose_colorwash_opacity"]
        finally:
            plotter.close()

        self.assertEqual(opacity_values.shape[0], prepared_scene.lattice_points.shape[0])
        self.assertGreater(float(np.max(opacity_values)), float(np.min(opacity_values)))
        self.assertLessEqual(float(np.max(opacity_values)), 0.2)
        self.assertGreaterEqual(float(np.min(opacity_values)), 0.02)

    def test_build_plotter_supports_reference_biopsy_points(self) -> None:
        prepared_scene = prepare_dose_nn_render_scene(
            _synthetic_scene_with_trials(),
            DoseNNRenderConfig(
                selected_trials=(1,),
                reference_trial_numbers=(0,),
                show_reference_biopsy_points=True,
                show_lattice_points=False,
            ),
        )

        plotter = build_pyvista_dose_nn_plotter(prepared_scene, settings=_test_settings())
        try:
            actor_names = set(plotter.renderer.actors.keys())
        finally:
            plotter.close()

        self.assertIn("biopsy_query_points", actor_names)
        self.assertIn("reference_biopsy_points", actor_names)
        self.assertNotIn("dose_lattice_points", actor_names)

    def test_build_plotter_supports_dose_volume_colorwash_layer(self) -> None:
        prepared_scene = prepare_dose_nn_render_scene(
            _synthetic_rectilinear_volume_scene(),
            DoseNNRenderConfig(
                show_lattice_points=False,
                show_dose_colorwash=True,
            ),
        )

        plotter = build_pyvista_dose_nn_plotter(
            prepared_scene,
            settings=DoseNNPyVistaRenderSettings(
                off_screen=True,
                window_size=(320, 240),
                dose_colorwash_style="volume",
                dose_colorwash_opacity=0.3,
                show_axes=False,
                show_scalar_bar=False,
            ),
        )
        try:
            actor_names = set(plotter.renderer.actors.keys())
        finally:
            plotter.close()

        self.assertNotIn("dose_colorwash_points", actor_names)
        self.assertIn("dose_colorwash_volume", actor_names)

    def test_volume_colorwash_masks_uncropped_lattice_for_spatial_radius(self) -> None:
        prepared_scene = prepare_dose_nn_render_scene(
            _synthetic_rectilinear_volume_scene(),
            DoseNNRenderConfig(
                show_lattice_points=False,
                show_dose_colorwash=True,
                spatial_radius_mm=0.5,
            ),
        )

        plotter = build_pyvista_dose_nn_plotter(
            prepared_scene,
            settings=DoseNNPyVistaRenderSettings(
                off_screen=True,
                window_size=(320, 240),
                dose_colorwash_style="volume",
                dose_color_scale_min=2.0,
                dose_color_scale_max=6.0,
                show_axes=False,
                show_scalar_bar=False,
            ),
        )
        try:
            actor_names = set(plotter.renderer.actors.keys())
        finally:
            plotter.close()

        self.assertEqual(prepared_scene.lattice_points.shape[0], 2)
        self.assertEqual(prepared_scene.colorwash_lattice_points.shape[0], 8)
        self.assertEqual(int(np.count_nonzero(prepared_scene.colorwash_lattice_visibility_mask)), 2)
        self.assertIn("dose_colorwash_volume", actor_names)

    def test_volume_colorwash_builds_cropped_rectilinear_subvolume(self) -> None:
        prepared_scene = prepare_dose_nn_render_scene(
            _synthetic_rectilinear_four_by_four_volume_scene(),
            DoseNNRenderConfig(
                show_lattice_points=False,
                show_dose_colorwash=True,
                spatial_radius_mm=0.1,
            ),
        )

        plotter = build_pyvista_dose_nn_plotter(
            prepared_scene,
            settings=DoseNNPyVistaRenderSettings(
                off_screen=True,
                window_size=(320, 240),
                dose_colorwash_style="volume",
                show_axes=False,
                show_scalar_bar=False,
            ),
        )
        try:
            actor = plotter.renderer.actors["dose_colorwash_volume"]
            volume_dimensions = actor.mapper.dataset.dimensions
            volume_bounds = actor.GetBounds()
        finally:
            plotter.close()

        self.assertEqual(volume_dimensions, (2, 2, 2))
        self.assertEqual(volume_bounds, (0.0, 1.0, 0.0, 1.0, 0.0, 1.0))

    def test_volume_colorwash_skips_when_spatial_radius_selects_no_lattice_points(self) -> None:
        prepared_scene = prepare_dose_nn_render_scene(
            _synthetic_rectilinear_volume_scene(),
            DoseNNRenderConfig(
                show_lattice_points=False,
                show_dose_colorwash=True,
                spatial_radius_mm=0.1,
            ),
        )

        plotter = build_pyvista_dose_nn_plotter(
            prepared_scene,
            settings=DoseNNPyVistaRenderSettings(
                off_screen=True,
                window_size=(320, 240),
                dose_colorwash_style="volume",
                show_axes=False,
                show_scalar_bar=False,
            ),
        )
        try:
            actor_names = set(plotter.renderer.actors.keys())
        finally:
            plotter.close()

        self.assertEqual(int(np.count_nonzero(prepared_scene.colorwash_lattice_visibility_mask)), 0)
        self.assertNotIn("dose_colorwash_volume", actor_names)

    def test_volume_colorwash_requires_complete_three_dimensional_lattice(self) -> None:
        prepared_scene = prepare_dose_nn_render_scene(
            _synthetic_scene(),
            DoseNNRenderConfig(show_lattice_points=False, show_dose_colorwash=True),
        )

        with self.assertRaisesRegex(ValueError, "dose volume colorwash"):
            build_pyvista_dose_nn_plotter(
                prepared_scene,
                settings=DoseNNPyVistaRenderSettings(
                    off_screen=True,
                    window_size=(320, 240),
                    dose_colorwash_style="volume",
                    show_axes=False,
                    show_scalar_bar=False,
                ),
            )

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

    def test_export_trial_frame_sequence_writes_frames_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory).joinpath("frames")
            try:
                result = export_dose_nn_trial_frame_sequence_pyvista(
                    _synthetic_scene_with_trials(),
                    output_dir,
                    selected_trials=(0, 1),
                    frames_per_second=6.0,
                    base_config=DoseNNRenderConfig(
                        show_lattice_points=False,
                        show_dose_colorwash=True,
                        reference_trial_numbers=(0,),
                        show_reference_biopsy_points=True,
                        vector_stride=2,
                    ),
                    settings=_test_settings(),
                )
            except Exception as exc:
                self.skipTest("PyVista frame export is unavailable in this environment: {}".format(exc))

            self.assertEqual(len(result.frame_paths), 2)
            self.assertTrue(result.frame_paths[0].is_file())
            self.assertTrue(result.manifest_path.is_file())
            with open(result.manifest_path, "r", encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
            self.assertEqual(manifest["base_render_config"]["reference_trial_numbers"], [0])
            self.assertTrue(manifest["base_render_config"]["show_reference_biopsy_points"])

    def test_export_trial_frame_sequence_fails_for_unavailable_trial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "unavailable trials"):
                export_dose_nn_trial_frame_sequence_pyvista(
                    _synthetic_scene(),
                    temporary_directory,
                    selected_trials=(99,),
                    settings=_test_settings(),
                )


def _test_settings() -> DoseNNPyVistaRenderSettings:
    return DoseNNPyVistaRenderSettings(
        off_screen=True,
        window_size=(320, 240),
        dose_colorwash_opacity=0.35,
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


def _synthetic_scene_with_trials():
    return build_dose_nn_render_scene(
        metadata=DoseNNSceneMetadata(
            patient_uid="synthetic",
            biopsy_roi="Bx 1",
            biopsy_index=1,
            source_label="unit-test-movie",
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
        original_point_indices=np.array([0, 1, 0, 1], dtype=int),
        trial_numbers=np.array([0, 0, 1, 1], dtype=int),
        biopsy_points=np.array(
            [[0.5, 0.0, 1.0], [1.5, 0.0, 1.0], [0.7, 0.1, 1.0], [1.7, 0.1, 1.0]],
            dtype=float,
        ),
        interpolated_biopsy_doses=np.array([15.0, 25.0, 16.0, 26.0], dtype=float),
        nearest_lattice_points=np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            ],
            dtype=float,
        ),
        nearest_lattice_doses=np.array([[10.0, 20.0], [20.0, 30.0], [10.0, 20.0], [20.0, 30.0]]),
        nearest_distances=np.array([[1.0, 1.1], [1.0, 1.1], [1.0, 1.1], [1.0, 1.1]], dtype=float),
    )


def _synthetic_rectilinear_volume_scene():
    lattice_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    return build_dose_nn_render_scene(
        metadata=DoseNNSceneMetadata(
            patient_uid="synthetic",
            biopsy_roi="Bx 1",
            biopsy_index=1,
            source_label="unit-test-volume",
        ),
        lattice_points=lattice_points,
        lattice_doses=np.arange(8, dtype=float),
        original_point_indices=np.array([0, 1], dtype=int),
        trial_numbers=np.array([0, 0], dtype=int),
        biopsy_points=np.array([[0.25, 0.25, 1.2], [0.75, 0.75, 1.2]], dtype=float),
        interpolated_biopsy_doses=np.array([2.0, 4.0], dtype=float),
        nearest_lattice_points=np.array(
            [
                [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
                [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            ],
            dtype=float,
        ),
        nearest_lattice_doses=np.array([[4.0, 5.0], [6.0, 7.0]], dtype=float),
        nearest_distances=np.array([[0.5, 0.6], [0.5, 0.6]], dtype=float),
    )


def _synthetic_rectilinear_four_by_four_volume_scene():
    lattice_points = np.array([[x, y, z] for x in range(4) for y in range(4) for z in range(4)], dtype=float)
    return build_dose_nn_render_scene(
        metadata=DoseNNSceneMetadata(
            patient_uid="synthetic",
            biopsy_roi="Bx 1",
            biopsy_index=1,
            source_label="unit-test-volume-four-by-four",
        ),
        lattice_points=lattice_points,
        lattice_doses=np.arange(lattice_points.shape[0], dtype=float),
        original_point_indices=np.array([0], dtype=int),
        trial_numbers=np.array([0], dtype=int),
        biopsy_points=np.array([[0.0, 0.0, 0.0]], dtype=float),
        interpolated_biopsy_doses=np.array([1.0], dtype=float),
        nearest_lattice_points=np.array([[[0.0, 0.0, 0.0]]], dtype=float),
        nearest_lattice_doses=np.array([[1.0]], dtype=float),
        nearest_distances=np.array([[0.0]], dtype=float),
    )


if __name__ == "__main__":
    unittest.main()
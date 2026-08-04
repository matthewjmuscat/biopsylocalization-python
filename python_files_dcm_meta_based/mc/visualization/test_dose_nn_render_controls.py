"""Synthetic checks for dose-owned render control translation."""

from __future__ import annotations

import unittest

from mc.visualization.dose_nn_pyvista import DoseNNPyVistaRenderSettings
from mc.visualization.dose_nn_render_controls import DoseNNRenderControlSelection
from mc.visualization.dose_nn_render_controls import dose_nn_pyvista_settings_from_control_selection
from mc.visualization.dose_nn_render_controls import dose_nn_render_config_from_control_selection
from mc.visualization.dose_nn_render_controls import normalize_dose_nn_render_control_selection


class DoseNNRenderControlTests(unittest.TestCase):
    def test_control_selection_builds_render_config_and_pyvista_settings(self) -> None:
        selection = DoseNNRenderControlSelection(
            selected_trials=(1,),
            dose_threshold_min=10.0,
            dose_threshold_max=40.0,
            max_lattice_points=500,
            spatial_radius_mm=12.5,
            biopsy_point_stride=2,
            vector_stride=3,
            show_reference_biopsy_points=True,
            show_lattice_points=False,
            show_dose_colorwash=True,
            dose_colorwash_style="volume",
            dose_colorwash_opacity=0.35,
            dose_colorwash_point_size=9.0,
            show_nearest_neighbour_points=False,
            show_scalar_bar=False,
        )

        config = dose_nn_render_config_from_control_selection(selection, available_trials=(0, 1, 2))
        settings = dose_nn_pyvista_settings_from_control_selection(
            selection,
            available_trials=(0, 1, 2),
            base_settings=DoseNNPyVistaRenderSettings(window_size=(320, 240)),
        )

        self.assertEqual(config.selected_trials, (1,))
        self.assertEqual(config.reference_trial_numbers, (0,))
        self.assertFalse(config.show_lattice_points)
        self.assertTrue(config.show_dose_colorwash)
        self.assertFalse(config.show_nearest_neighbour_points)
        self.assertEqual(config.vector_stride, 3)
        self.assertEqual(settings.window_size, (320, 240))
        self.assertEqual(settings.dose_colorwash_style, "volume")
        self.assertEqual(settings.dose_colorwash_opacity, 0.35)
        self.assertFalse(settings.show_scalar_bar)

    def test_control_selection_accepts_point_alias(self) -> None:
        selection = normalize_dose_nn_render_control_selection(
            DoseNNRenderControlSelection(dose_colorwash_style="point"),
        )

        self.assertEqual(selection.dose_colorwash_style, "points")

    def test_control_selection_rejects_unavailable_trials(self) -> None:
        with self.assertRaisesRegex(ValueError, "selected_trials"):
            dose_nn_render_config_from_control_selection(
                DoseNNRenderControlSelection(selected_trials=(99,)),
                available_trials=(0, 1),
            )

    def test_control_selection_rejects_invalid_colorwash_style(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported dose colorwash style"):
            normalize_dose_nn_render_control_selection(
                DoseNNRenderControlSelection(dose_colorwash_style="surface"),
            )

    def test_control_selection_rejects_invalid_numeric_controls(self) -> None:
        with self.assertRaisesRegex(ValueError, "dose_threshold_min"):
            normalize_dose_nn_render_control_selection(
                DoseNNRenderControlSelection(dose_threshold_min=20.0, dose_threshold_max=10.0),
            )
        with self.assertRaisesRegex(ValueError, "dose_colorwash_opacity"):
            normalize_dose_nn_render_control_selection(
                DoseNNRenderControlSelection(dose_colorwash_opacity=1.2),
            )


if __name__ == "__main__":
    unittest.main()
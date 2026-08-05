"""Synthetic checks for dose-owned render control translation."""

from __future__ import annotations

import unittest

from mc.visualization.dose_nn_pyvista import DoseNNPyVistaRenderSettings
from mc.visualization.dose_nn_render_controls import DoseNNRenderControlSelection
from mc.visualization.dose_nn_render_controls import dose_nn_pyvista_settings_from_control_selection
from mc.visualization.dose_nn_render_controls import dose_nn_render_config_from_control_selection
from mc.visualization.dose_nn_render_controls import normalize_dose_nn_render_control_selection
from mc.visualization.dose_nn_tk_render_controls import _parse_trial_numbers


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
            dose_colorwash_volume_max_voxels=125_000,
            dose_color_scale_mode="log",
            dose_color_scale_min=5.0,
            dose_color_scale_max=30.0,
            dose_colorwash_opacity=0.35,
            dose_colorwash_point_opacity_mode="center_fade",
            dose_colorwash_point_opacity_min=0.05,
            dose_colorwash_point_size=9.0,
            lattice_point_size=7.0,
            biopsy_point_size=14.0,
            reference_biopsy_point_size=13.0,
            nearest_point_size=11.0,
            vector_line_width=4.0,
            show_nearest_neighbour_points=False,
            show_scalar_bar=False,
            dose_scalar_bar_title="Dose (Gy)",
            dose_scalar_bar_show_background=False,
            dose_scalar_bar_title_font_size=20,
            dose_scalar_bar_label_font_size=16,
            axes_title_font_size=22,
            axes_tick_label_font_size=15,
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
        self.assertEqual(settings.dose_colorwash_volume_max_voxels, 125_000)
        self.assertEqual(settings.dose_color_scale_mode, "log")
        self.assertEqual(settings.dose_color_scale_min, 5.0)
        self.assertEqual(settings.dose_color_scale_max, 30.0)
        self.assertEqual(settings.dose_colorwash_opacity, 0.35)
        self.assertEqual(settings.dose_colorwash_point_opacity_mode, "center_fade")
        self.assertEqual(settings.dose_colorwash_point_opacity_min, 0.05)
        self.assertEqual(settings.dose_colorwash_point_size, 9.0)
        self.assertEqual(settings.lattice_point_size, 7.0)
        self.assertEqual(settings.biopsy_point_size, 14.0)
        self.assertEqual(settings.reference_biopsy_point_size, 13.0)
        self.assertEqual(settings.nearest_point_size, 11.0)
        self.assertEqual(settings.vector_line_width, 4.0)
        self.assertFalse(settings.show_scalar_bar)
        self.assertEqual(settings.dose_scalar_bar_title, "Dose (Gy)")
        self.assertFalse(settings.dose_scalar_bar_show_background)
        self.assertEqual(settings.dose_scalar_bar_title_font_size, 20)
        self.assertEqual(settings.dose_scalar_bar_label_font_size, 16)
        self.assertEqual(settings.axes_title_font_size, 22)
        self.assertEqual(settings.axes_tick_label_font_size, 15)

    def test_control_selection_accepts_point_alias(self) -> None:
        selection = normalize_dose_nn_render_control_selection(
            DoseNNRenderControlSelection(dose_colorwash_style="point"),
        )

        self.assertEqual(selection.dose_colorwash_style, "points")

    def test_control_selection_accepts_logarithmic_color_scale_alias(self) -> None:
        selection = normalize_dose_nn_render_control_selection(
            DoseNNRenderControlSelection(dose_color_scale_mode="logarithmic"),
        )

        self.assertEqual(selection.dose_color_scale_mode, "log")

    def test_control_selection_accepts_center_fade_opacity_alias(self) -> None:
        selection = normalize_dose_nn_render_control_selection(
            DoseNNRenderControlSelection(dose_colorwash_point_opacity_mode="central fade"),
        )

        self.assertEqual(selection.dose_colorwash_point_opacity_mode, "center_fade")

    def test_tk_trial_parser_accepts_ranges_and_lists(self) -> None:
        self.assertEqual(_parse_trial_numbers("0-3, 10; 12"), (0, 1, 2, 3, 10, 12))
        self.assertEqual(_parse_trial_numbers("all"), None)

        with self.assertRaisesRegex(ValueError, "trial ranges must be ascending"):
            _parse_trial_numbers("5-3")

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
        with self.assertRaisesRegex(ValueError, "dose_color_scale_min"):
            normalize_dose_nn_render_control_selection(
                DoseNNRenderControlSelection(dose_color_scale_min=40.0, dose_color_scale_max=10.0),
            )
        with self.assertRaisesRegex(ValueError, "log dose color scaling"):
            normalize_dose_nn_render_control_selection(
                DoseNNRenderControlSelection(
                    dose_color_scale_mode="log",
                    dose_color_scale_min=0.0,
                    dose_color_scale_max=10.0,
                ),
            )
        with self.assertRaisesRegex(ValueError, "dose_colorwash_point_opacity_min"):
            normalize_dose_nn_render_control_selection(
                DoseNNRenderControlSelection(
                    dose_colorwash_opacity=0.1,
                    dose_colorwash_point_opacity_min=0.2,
                ),
            )


if __name__ == "__main__":
    unittest.main()
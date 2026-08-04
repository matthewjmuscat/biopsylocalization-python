"""Synthetic checks for dose NN saved-scene selector helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ui.render_broker import RenderBrokerDecision
from ui.render_broker import RenderBrokerDialogResult
from ui.render_broker import RenderBrokerSessionState

from mc.visualization.dose_nn_pyvista import DoseNNPyVistaRenderSettings
from mc.visualization.dose_nn_pyvista import is_pyvista_available
from mc.visualization.dose_nn_render_controls import DoseNNRenderControlSelection
from mc.visualization.dose_nn_scene import DoseNNRenderConfig
from mc.visualization.dose_nn_scene import DoseNNSceneMetadata
from mc.visualization.dose_nn_scene import build_dose_nn_render_scene
from mc.visualization.dose_nn_scene_artifacts import DOSE_NN_RENDER_SCENE_MANIFEST_FILENAME
from mc.visualization.dose_nn_scene_artifacts import write_dose_nn_render_scene_artifact
from mc.visualization.dose_nn_selector import DOSE_NN_SAVED_SCENE_GROUP_KEY
from mc.visualization.dose_nn_selector import build_saved_dose_nn_scene_broker_request
from mc.visualization.dose_nn_selector import build_saved_dose_nn_scene_choice_group
from mc.visualization.dose_nn_selector import discover_saved_dose_nn_scene_options
from mc.visualization.dose_nn_selector import handle_saved_dose_nn_scene_broker_decision_pyvista
from mc.visualization.dose_nn_selector import render_saved_dose_nn_scene_selection_pyvista
from mc.visualization.dose_nn_selector import run_saved_dose_nn_scene_controlled_selector_session
from mc.visualization.dose_nn_selector import run_saved_dose_nn_scene_selector_session


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
        self.assertEqual(option.available_trials, (0,))
        self.assertEqual(option.lattice_dose_range, (10.0, 40.0))
        self.assertIn("query rows: 2", option.display_label)
        self.assertIn("trials: 1", option.display_label)
        self.assertEqual(option.suggested_export_output_dir.name, "synthetic_scene")

    def test_choice_group_uses_saved_scene_options(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            scene_dir = Path(temporary_directory).joinpath("scene")
            write_dose_nn_render_scene_artifact(_synthetic_scene(), scene_dir, scene_id="synthetic_scene")
            options = discover_saved_dose_nn_scene_options(temporary_directory)

        choice_group = build_saved_dose_nn_scene_choice_group(options)

        self.assertEqual(choice_group.group_key, DOSE_NN_SAVED_SCENE_GROUP_KEY)
        self.assertEqual(choice_group.selection_mode, "single")
        self.assertTrue(choice_group.allow_pyvista)
        self.assertEqual(choice_group.default_backend, "pyvista")
        self.assertEqual(choice_group.options[0].option_key, "synthetic_scene")
        self.assertTrue(choice_group.options[0].selected_by_default)

    def test_broker_request_wraps_saved_scene_choice_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            scene_dir = Path(temporary_directory).joinpath("scene")
            write_dose_nn_render_scene_artifact(_synthetic_scene(), scene_dir, scene_id="synthetic_scene")
            options = discover_saved_dose_nn_scene_options(temporary_directory)

        request = build_saved_dose_nn_scene_broker_request(options, summary_lines=("one scene",))

        self.assertEqual(request.title, "Dose NN saved-scene renderer")
        self.assertEqual(request.summary_lines, ("one scene",))
        self.assertEqual(request.choice_groups[0].group_key, DOSE_NN_SAVED_SCENE_GROUP_KEY)
        self.assertEqual(request.continue_button_label, "Exit renderer")

    def test_render_unknown_saved_scene_option_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown dose NN saved scene option"):
            render_saved_dose_nn_scene_selection_pyvista((), ("missing",), Path("unused"))

    def test_broker_decision_requires_saved_scene_group_and_pyvista(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported dose NN render broker group"):
            handle_saved_dose_nn_scene_broker_decision_pyvista(
                RenderBrokerDecision(action="render", group_key="other", render_backend="pyvista"),
                (),
                Path("unused"),
            )

    def test_selector_session_can_continue_without_opening_tk(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scene_dir = root.joinpath("scene")
            write_dose_nn_render_scene_artifact(_synthetic_scene(), scene_dir, scene_id="synthetic_scene")
            dialog_adapter = _ContinueDialogAdapter()

            session_state = run_saved_dose_nn_scene_selector_session(
                root,
                root.joinpath("exports"),
                dialog_adapter=dialog_adapter,
            )

        self.assertFalse(session_state.timeout_disabled_for_run)
        self.assertEqual(dialog_adapter.request.title, "Dose NN saved-scene renderer")
        self.assertTrue(dialog_adapter.request.choice_groups[0].allow_pyvista)
        with self.assertRaisesRegex(ValueError, "PyVista"):
            handle_saved_dose_nn_scene_broker_decision_pyvista(
                RenderBrokerDecision(
                    action="render",
                    group_key=DOSE_NN_SAVED_SCENE_GROUP_KEY,
                    render_backend="plotly",
                ),
                (),
                Path("unused"),
            )

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

    def test_render_saved_scene_selection_accepts_control_selection(self) -> None:
        if not is_pyvista_available():
            self.skipTest("PyVista is not available in this environment")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scene_dir = root.joinpath("scene")
            output_dir = root.joinpath("exports")
            write_dose_nn_render_scene_artifact(_synthetic_scene(), scene_dir, scene_id="synthetic_scene")
            options = discover_saved_dose_nn_scene_options(root)

            render_saved_dose_nn_scene_selection_pyvista(
                options,
                ("synthetic_scene",),
                output_dir,
                control_selection=DoseNNRenderControlSelection(
                    selected_trials=(0,),
                    show_dose_colorwash=True,
                    show_lattice_points=False,
                    dose_colorwash_opacity=0.2,
                    show_scalar_bar=False,
                ),
                settings=DoseNNPyVistaRenderSettings(
                    off_screen=True,
                    window_size=(320, 240),
                    show_axes=False,
                ),
            )

            provenance_path = output_dir.joinpath("synthetic_scene.png.provenance.json")
            with open(provenance_path, "r", encoding="utf-8") as provenance_file:
                provenance = json.load(provenance_file)
            self.assertEqual(provenance["render_config"]["selected_trials"], [0])
            self.assertTrue(provenance["render_config"]["show_dose_colorwash"])
            self.assertFalse(provenance["render_config"]["show_lattice_points"])
            self.assertEqual(provenance["render_settings"]["dose_colorwash_opacity"], 0.2)
            self.assertFalse(provenance["render_settings"]["show_scalar_bar"])

    def test_render_saved_scene_selection_rejects_ambiguous_controls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scene_dir = root.joinpath("scene")
            write_dose_nn_render_scene_artifact(_synthetic_scene(), scene_dir, scene_id="synthetic_scene")
            options = discover_saved_dose_nn_scene_options(root)

            with self.assertRaisesRegex(ValueError, "config or control_selection"):
                render_saved_dose_nn_scene_selection_pyvista(
                    options,
                    ("synthetic_scene",),
                    root.joinpath("exports"),
                    config=DoseNNRenderConfig(),
                    control_selection=DoseNNRenderControlSelection(),
                )

    def test_render_saved_scene_selection_rejects_unavailable_control_trial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scene_dir = root.joinpath("scene")
            write_dose_nn_render_scene_artifact(_synthetic_scene(), scene_dir, scene_id="synthetic_scene")
            options = discover_saved_dose_nn_scene_options(root)

            with self.assertRaisesRegex(ValueError, "unavailable trials"):
                render_saved_dose_nn_scene_selection_pyvista(
                    options,
                    ("synthetic_scene",),
                    root.joinpath("exports"),
                    control_selection=DoseNNRenderControlSelection(selected_trials=(99,)),
                )

    def test_controlled_selector_session_collects_controls_before_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scene_dir = root.joinpath("scene")
            write_dose_nn_render_scene_artifact(_synthetic_scene(), scene_dir, scene_id="synthetic_scene")
            dialog_adapter = _RenderThenContinueDialogAdapter()
            control_adapter = _ControlSelectionAdapter(
                DoseNNRenderControlSelection(
                    selected_trials=(0,),
                    show_dose_colorwash=True,
                    show_lattice_points=False,
                )
            )

            with patch(
                "mc.visualization.dose_nn_selector.render_saved_dose_nn_scene_selection_pyvista"
            ) as render_mock:
                session_state = run_saved_dose_nn_scene_controlled_selector_session(
                    root,
                    root.joinpath("exports"),
                    dialog_adapter=dialog_adapter,
                    control_dialog_adapter=control_adapter,
                )

        self.assertFalse(session_state.timeout_disabled_for_run)
        self.assertEqual(dialog_adapter.call_count, 2)
        self.assertEqual(control_adapter.option.scene_id, "synthetic_scene")
        render_mock.assert_called_once()
        self.assertFalse(render_mock.call_args.kwargs["settings"].off_screen)
        self.assertFalse(render_mock.call_args.kwargs["control_selection"].show_lattice_points)
        self.assertTrue(render_mock.call_args.kwargs["control_selection"].show_dose_colorwash)

    def test_controlled_selector_session_skips_render_when_controls_cancelled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scene_dir = root.joinpath("scene")
            write_dose_nn_render_scene_artifact(_synthetic_scene(), scene_dir, scene_id="synthetic_scene")
            dialog_adapter = _RenderThenContinueDialogAdapter()
            control_adapter = _ControlSelectionAdapter(None)

            with patch(
                "mc.visualization.dose_nn_selector.render_saved_dose_nn_scene_selection_pyvista"
            ) as render_mock:
                run_saved_dose_nn_scene_controlled_selector_session(
                    root,
                    root.joinpath("exports"),
                    dialog_adapter=dialog_adapter,
                    control_dialog_adapter=control_adapter,
                )

        render_mock.assert_not_called()


class _ContinueDialogAdapter:
    def __init__(self) -> None:
        self.request = None

    def collect_selection(self, request, session_state):
        self.request = request
        return RenderBrokerDialogResult(
            decision=RenderBrokerDecision(action="continue"),
            session_state=RenderBrokerSessionState(
                timeout_disabled_for_run=session_state.timeout_disabled_for_run,
            ),
        )


class _RenderThenContinueDialogAdapter:
    def __init__(self) -> None:
        self.call_count = 0

    def collect_selection(self, request, session_state):
        self.call_count += 1
        if self.call_count == 1:
            return RenderBrokerDialogResult(
                decision=RenderBrokerDecision(
                    action="render",
                    group_key=DOSE_NN_SAVED_SCENE_GROUP_KEY,
                    selected_option_keys=("synthetic_scene",),
                    render_backend="pyvista",
                ),
                session_state=session_state,
            )
        return RenderBrokerDialogResult(
            decision=RenderBrokerDecision(action="continue"),
            session_state=session_state,
        )


class _ControlSelectionAdapter:
    def __init__(self, selection):
        self.selection = selection
        self.option = None
        self.initial_selection = None

    def collect_control_selection(self, option, initial_selection=None):
        self.option = option
        self.initial_selection = initial_selection
        return self.selection


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
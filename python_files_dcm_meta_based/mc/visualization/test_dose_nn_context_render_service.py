"""Synthetic checks for post-run dose NN context render service."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from output_artifacts.context_contracts import PatientArtifactIndex
from output_artifacts.context_contracts import write_patient_artifact_index
from ui.render_broker import RenderBrokerDecision
from ui.render_broker import RenderBrokerDialogResult
from ui.render_broker import RenderBrokerSessionState

from mc.simulation.per_patient.dose import MC_DOSE_VALUE_COLUMN
from mc.simulation.per_patient.dose import PatientDoseLatticeContext
from mc.simulation.per_patient.dose_context_artifacts import build_patient_dose_lattice_context_artifact_plan
from mc.simulation.per_patient.dose_context_artifacts import patient_dose_lattice_context_array_payload
from mc.simulation.per_patient.dose_context_artifacts import write_patient_dose_context_zarr_arrays
from mc.visualization.dose_nn_context_bridge import write_dose_nn_render_context_zarr_artifact
from mc.visualization.dose_nn_context_render_service import main
from mc.visualization.dose_nn_context_render_service import materialize_and_run_dose_nn_context_selector_session
from mc.visualization.dose_nn_context_render_service import materialize_dose_nn_saved_scene_artifact_from_patient_index
from mc.visualization.dose_nn_scene import DoseNNSceneMetadata
from mc.visualization.dose_nn_scene import build_dose_nn_render_scene
from mc.visualization.dose_nn_scene_artifacts import read_dose_nn_render_scene_artifact


class DoseNNContextRenderServiceTests(unittest.TestCase):
    def test_materializes_saved_scene_from_patient_artifact_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = _write_context_fixture(Path(temporary_directory))
            scene_artifact_dir = fixture.output_root.joinpath("render_scenes", "scene_from_index")

            result = materialize_dose_nn_saved_scene_artifact_from_patient_index(
                patient_artifact_index_path=fixture.index_path,
                output_root=fixture.output_root,
                lattice_artifact_id="dose_lattice_context",
                render_context_artifact_id="dose_biopsy_002_render_context",
                scene_artifact_dir=scene_artifact_dir,
                scene_id="scene_from_index",
            )
            loaded_scene = read_dose_nn_render_scene_artifact(scene_artifact_dir)

        self.assertEqual(result.manifest.scene_id, "scene_from_index")
        self.assertEqual(result.lattice_artifact_ref.artifact_id, "dose_lattice_context")
        np.testing.assert_array_equal(loaded_scene.nearest_distances, fixture.runtime_scene.nearest_distances)

    def test_selector_session_materializes_scene_before_collecting_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = _write_context_fixture(Path(temporary_directory))
            scene_artifact_dir = fixture.output_root.joinpath("render_scenes", "scene_for_selector")
            dialog_adapter = _ContinueDialogAdapter()

            result = materialize_and_run_dose_nn_context_selector_session(
                patient_artifact_index_path=fixture.index_path,
                output_root=fixture.output_root,
                lattice_artifact_id="dose_lattice_context",
                render_context_artifact_id="dose_biopsy_002_render_context",
                scene_artifact_dir=scene_artifact_dir,
                scene_id="scene_for_selector",
                export_dir=fixture.output_root.joinpath("exports"),
                dialog_adapter=dialog_adapter,
            )

        self.assertEqual(result.materialization.manifest.scene_id, "scene_for_selector")
        self.assertEqual(dialog_adapter.request.title, "Dose NN saved-scene renderer")
        self.assertEqual(dialog_adapter.request.choice_groups[0].options[0].option_key, "scene_for_selector")

    def test_cli_materialize_only_writes_saved_scene(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = _write_context_fixture(Path(temporary_directory))
            scene_artifact_dir = fixture.output_root.joinpath("render_scenes", "scene_from_cli")
            exit_code = main(
                (
                    "--patient-artifact-index",
                    str(fixture.index_path),
                    "--output-root",
                    str(fixture.output_root),
                    "--lattice-artifact-id",
                    "dose_lattice_context",
                    "--render-context-artifact-id",
                    "dose_biopsy_002_render_context",
                    "--scene-artifact-dir",
                    str(scene_artifact_dir),
                    "--scene-id",
                    "scene_from_cli",
                )
            )
            loaded_scene = read_dose_nn_render_scene_artifact(scene_artifact_dir)

        self.assertEqual(exit_code, 0)
        self.assertEqual(loaded_scene.available_trials, (0, 1))

    def test_missing_artifact_id_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = _write_context_fixture(Path(temporary_directory))
            with self.assertRaisesRegex(ValueError, "missing artifact ID"):
                materialize_dose_nn_saved_scene_artifact_from_patient_index(
                    patient_artifact_index_path=fixture.index_path,
                    output_root=fixture.output_root,
                    lattice_artifact_id="missing_lattice",
                    render_context_artifact_id="dose_biopsy_002_render_context",
                    scene_artifact_dir=fixture.output_root.joinpath("render_scenes", "bad"),
                    scene_id="bad",
                )


class _Fixture:
    def __init__(self, output_root: Path, index_path: Path, runtime_scene) -> None:
        self.output_root = output_root
        self.index_path = index_path
        self.runtime_scene = runtime_scene


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


def _write_context_fixture(output_root: Path) -> _Fixture:
    runtime_scene = _synthetic_runtime_scene()
    lattice_context = _synthetic_lattice_context(runtime_scene)
    lattice_plan = build_patient_dose_lattice_context_artifact_plan(lattice_context)
    write_patient_dose_context_zarr_arrays(
        lattice_plan,
        patient_dose_lattice_context_array_payload(lattice_context),
        output_root,
    )
    render_plan, _written_paths = write_dose_nn_render_context_zarr_artifact(runtime_scene, output_root)
    index = PatientArtifactIndex(patient_uid=runtime_scene.metadata.patient_uid)
    index = index.add_artifact(lattice_plan.artifact_refs[0]).add_artifact(render_plan.artifact_refs[0])
    index_path = output_root.joinpath("context", "manifest.json")
    write_patient_artifact_index(index, index_path)
    return _Fixture(output_root=output_root, index_path=index_path, runtime_scene=runtime_scene)


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
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        lattice_doses=np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64),
        original_point_indices=np.array([0, 1, 0, 1], dtype=np.int64),
        trial_numbers=np.array([0, 0, 1, 1], dtype=np.int64),
        biopsy_points=np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.1, 0.0, 1.0], [1.1, 0.0, 1.0]],
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
        nearest_lattice_doses=np.array([[10.0, 20.0], [20.0, 30.0], [10.0, 20.0], [20.0, 30.0]], dtype=np.float64),
        nearest_distances=np.array([[1.0, 1.4], [1.0, 1.4], [1.1, 1.5], [1.1, 1.5]], dtype=np.float64),
    )


def _synthetic_lattice_context(scene) -> PatientDoseLatticeContext:
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
"""Post-run service for launching dose NN renders from retained context artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from output_artifacts.context_contracts import ArtifactRef
from output_artifacts.context_contracts import read_patient_artifact_index
from ui.render_broker import RenderBrokerDialogAdapter
from ui.render_broker import RenderBrokerSessionState
from ui.render_broker import RenderBrokerTimeoutPolicy

from .dose_nn_context_bridge import materialize_dose_nn_saved_scene_artifact_from_context
from .dose_nn_render_controls import DoseNNRenderControlSelection
from .dose_nn_selector import DoseNNRenderControlSelectionAdapter
from .dose_nn_selector import run_saved_dose_nn_scene_controlled_selector_session
from .dose_nn_scene_artifacts import DoseNNRenderSceneArtifactManifest


@dataclass(frozen=True, slots=True)
class DoseNNContextMaterializationResult:
    """Result of materializing a saved-scene artifact from retained context."""

    manifest: DoseNNRenderSceneArtifactManifest
    scene_artifact_dir: Path
    lattice_artifact_ref: ArtifactRef
    render_context_artifact_ref: ArtifactRef


@dataclass(frozen=True, slots=True)
class DoseNNContextSelectorSessionResult:
    """Result of materializing context and running the saved-scene selector."""

    materialization: DoseNNContextMaterializationResult
    session_state: RenderBrokerSessionState


def materialize_dose_nn_saved_scene_artifact_from_patient_index(
    *,
    patient_artifact_index_path: Path | str,
    output_root: Path | str,
    lattice_artifact_id: str,
    render_context_artifact_id: str,
    scene_artifact_dir: Path | str,
    scene_id: str,
    overwrite: bool = False,
) -> DoseNNContextMaterializationResult:
    """Materialize one saved-scene artifact from refs in a patient artifact index."""
    index = read_patient_artifact_index(patient_artifact_index_path)
    artifacts_by_id = index.artifacts_by_id
    lattice_artifact_ref = _required_artifact_ref(artifacts_by_id, lattice_artifact_id)
    render_context_artifact_ref = _required_artifact_ref(artifacts_by_id, render_context_artifact_id)
    resolved_scene_artifact_dir = Path(scene_artifact_dir)
    manifest = materialize_dose_nn_saved_scene_artifact_from_context(
        lattice_artifact_ref=lattice_artifact_ref,
        render_context_artifact_ref=render_context_artifact_ref,
        output_root=output_root,
        scene_artifact_dir=resolved_scene_artifact_dir,
        scene_id=scene_id,
        overwrite=overwrite,
    )
    return DoseNNContextMaterializationResult(
        manifest=manifest,
        scene_artifact_dir=resolved_scene_artifact_dir,
        lattice_artifact_ref=lattice_artifact_ref,
        render_context_artifact_ref=render_context_artifact_ref,
    )


def materialize_and_run_dose_nn_context_selector_session(
    *,
    patient_artifact_index_path: Path | str,
    output_root: Path | str,
    lattice_artifact_id: str,
    render_context_artifact_id: str,
    scene_artifact_dir: Path | str,
    scene_id: str,
    export_dir: Path | str,
    scene_search_root: Path | str | None = None,
    suggested_export_root: Path | str | None = None,
    initial_control_selection: DoseNNRenderControlSelection | None = None,
    dialog_adapter: RenderBrokerDialogAdapter | None = None,
    control_dialog_adapter: DoseNNRenderControlSelectionAdapter | None = None,
    timeout_policy: RenderBrokerTimeoutPolicy | None = None,
    initial_session_state: RenderBrokerSessionState | None = None,
    overwrite: bool = False,
) -> DoseNNContextSelectorSessionResult:
    """Materialize retained context into a saved scene and launch the selector loop."""
    materialization = materialize_dose_nn_saved_scene_artifact_from_patient_index(
        patient_artifact_index_path=patient_artifact_index_path,
        output_root=output_root,
        lattice_artifact_id=lattice_artifact_id,
        render_context_artifact_id=render_context_artifact_id,
        scene_artifact_dir=scene_artifact_dir,
        scene_id=scene_id,
        overwrite=overwrite,
    )
    search_root = Path(scene_search_root) if scene_search_root is not None else materialization.scene_artifact_dir.parent
    session_state = run_saved_dose_nn_scene_controlled_selector_session(
        search_root,
        export_dir,
        suggested_export_root=suggested_export_root,
        initial_control_selection=initial_control_selection,
        dialog_adapter=dialog_adapter,
        control_dialog_adapter=control_dialog_adapter,
        timeout_policy=timeout_policy,
        initial_session_state=initial_session_state,
        overwrite=overwrite,
    )
    return DoseNNContextSelectorSessionResult(
        materialization=materialization,
        session_state=session_state,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for materializing and optionally launching a dose NN context render."""
    args = _build_argument_parser().parse_args(argv)
    if bool(args.launch_selector):
        if args.export_dir is None:
            raise ValueError("--export-dir is required when --launch-selector is used")
        result = materialize_and_run_dose_nn_context_selector_session(
            patient_artifact_index_path=args.patient_artifact_index,
            output_root=args.output_root,
            lattice_artifact_id=args.lattice_artifact_id,
            render_context_artifact_id=args.render_context_artifact_id,
            scene_artifact_dir=args.scene_artifact_dir,
            scene_id=args.scene_id,
            export_dir=args.export_dir,
            scene_search_root=args.scene_search_root,
            suggested_export_root=args.suggested_export_root,
            overwrite=bool(args.overwrite),
        )
        print("[dose-nn-context-render] materialized {}".format(result.materialization.scene_artifact_dir))
        return 0

    result = materialize_dose_nn_saved_scene_artifact_from_patient_index(
        patient_artifact_index_path=args.patient_artifact_index,
        output_root=args.output_root,
        lattice_artifact_id=args.lattice_artifact_id,
        render_context_artifact_id=args.render_context_artifact_id,
        scene_artifact_dir=args.scene_artifact_dir,
        scene_id=args.scene_id,
        overwrite=bool(args.overwrite),
    )
    print("[dose-nn-context-render] materialized {}".format(result.scene_artifact_dir))
    return 0


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize a dose NN saved scene from retained context artifacts.",
    )
    parser.add_argument("--patient-artifact-index", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--lattice-artifact-id", required=True)
    parser.add_argument("--render-context-artifact-id", required=True)
    parser.add_argument("--scene-artifact-dir", required=True, type=Path)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--launch-selector", action="store_true")
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--scene-search-root", type=Path, default=None)
    parser.add_argument("--suggested-export-root", type=Path, default=None)
    return parser


def _required_artifact_ref(artifacts_by_id: dict[str, ArtifactRef], artifact_id: str) -> ArtifactRef:
    resolved_artifact_id = str(artifact_id).strip()
    if resolved_artifact_id in artifacts_by_id:
        return artifacts_by_id[resolved_artifact_id]
    raise ValueError("patient artifact index is missing artifact ID: {}".format(resolved_artifact_id))


if __name__ == "__main__":
    raise SystemExit(main())
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


DOSE_NN_RENDER_CONTEXT_ARTIFACT_FAMILY = "dose_nn_render_context"
DOSE_LATTICE_CONTEXT_ARTIFACT_FAMILY = "dose_lattice_context"
DEFAULT_DOSE_NN_LOCALIZATION_KIND = "dose"


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
    lattice_artifact_id: str | None = None,
    render_context_artifact_id: str | None = None,
    localization_kind: str = DEFAULT_DOSE_NN_LOCALIZATION_KIND,
    biopsy_index: int | None = None,
    scene_artifact_dir: Path | str,
    scene_id: str,
    overwrite: bool = False,
) -> DoseNNContextMaterializationResult:
    """Materialize one saved-scene artifact from refs in a patient artifact index."""
    index = read_patient_artifact_index(patient_artifact_index_path)
    lattice_artifact_ref = _resolve_lattice_artifact_ref(
        index.artifacts,
        artifact_id=lattice_artifact_id,
        localization_kind=localization_kind,
    )
    render_context_artifact_ref = _resolve_render_context_artifact_ref(
        index.artifacts,
        artifact_id=render_context_artifact_id,
        localization_kind=localization_kind,
        biopsy_index=biopsy_index,
    )
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
    lattice_artifact_id: str | None = None,
    render_context_artifact_id: str | None = None,
    localization_kind: str = DEFAULT_DOSE_NN_LOCALIZATION_KIND,
    biopsy_index: int | None = None,
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
        localization_kind=localization_kind,
        biopsy_index=biopsy_index,
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
    if bool(args.list_contexts):
        index = read_patient_artifact_index(args.patient_artifact_index)
        _print_available_contexts(index.artifacts, localization_kind=args.localization_kind)
        return 0

    output_root = _required_cli_arg(args.output_root, "--output-root")
    scene_artifact_dir = _required_cli_arg(args.scene_artifact_dir, "--scene-artifact-dir")
    scene_id = _required_cli_arg(args.scene_id, "--scene-id")
    if bool(args.launch_selector):
        if args.export_dir is None:
            raise ValueError("--export-dir is required when --launch-selector is used")
        result = materialize_and_run_dose_nn_context_selector_session(
            patient_artifact_index_path=args.patient_artifact_index,
            output_root=output_root,
            lattice_artifact_id=args.lattice_artifact_id,
            render_context_artifact_id=args.render_context_artifact_id,
            localization_kind=args.localization_kind,
            biopsy_index=args.biopsy_index,
            scene_artifact_dir=scene_artifact_dir,
            scene_id=scene_id,
            export_dir=args.export_dir,
            scene_search_root=args.scene_search_root,
            suggested_export_root=args.suggested_export_root,
            overwrite=bool(args.overwrite),
        )
        print("[dose-nn-context-render] materialized {}".format(result.materialization.scene_artifact_dir))
        return 0

    result = materialize_dose_nn_saved_scene_artifact_from_patient_index(
        patient_artifact_index_path=args.patient_artifact_index,
        output_root=output_root,
        lattice_artifact_id=args.lattice_artifact_id,
        render_context_artifact_id=args.render_context_artifact_id,
        localization_kind=args.localization_kind,
        biopsy_index=args.biopsy_index,
        scene_artifact_dir=scene_artifact_dir,
        scene_id=scene_id,
        overwrite=bool(args.overwrite),
    )
    print("[dose-nn-context-render] materialized {}".format(result.scene_artifact_dir))
    return 0


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize a dose NN saved scene from retained context artifacts.",
    )
    parser.add_argument("--patient-artifact-index", required=True, type=Path)
    parser.add_argument("--output-root", required=False, type=Path)
    parser.add_argument("--lattice-artifact-id", default=None)
    parser.add_argument("--render-context-artifact-id", default=None)
    parser.add_argument("--localization-kind", default=DEFAULT_DOSE_NN_LOCALIZATION_KIND)
    parser.add_argument("--biopsy-index", type=int, default=None)
    parser.add_argument("--scene-artifact-dir", required=False, type=Path)
    parser.add_argument("--scene-id", required=False)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-contexts", action="store_true")
    parser.add_argument("--launch-selector", action="store_true")
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--scene-search-root", type=Path, default=None)
    parser.add_argument("--suggested-export-root", type=Path, default=None)
    return parser


def _resolve_lattice_artifact_ref(
    artifact_refs: Sequence[ArtifactRef],
    *,
    artifact_id: str | None,
    localization_kind: str,
) -> ArtifactRef:
    if artifact_id is not None:
        return _required_artifact_ref(artifact_refs, artifact_id)
    candidates = tuple(
        artifact_ref
        for artifact_ref in artifact_refs
        if artifact_ref.artifact_family == DOSE_LATTICE_CONTEXT_ARTIFACT_FAMILY
        and _normal_token(artifact_ref.metadata.get("localization_kind", "")) == _normal_token(localization_kind)
    )
    return _single_artifact_ref(candidates, "dose lattice context", "--lattice-artifact-id")


def _resolve_render_context_artifact_ref(
    artifact_refs: Sequence[ArtifactRef],
    *,
    artifact_id: str | None,
    localization_kind: str,
    biopsy_index: int | None,
) -> ArtifactRef:
    if artifact_id is not None:
        return _required_artifact_ref(artifact_refs, artifact_id)
    normalized_localization_kind = _normal_token(localization_kind)
    candidates = tuple(
        artifact_ref
        for artifact_ref in artifact_refs
        if artifact_ref.artifact_family == DOSE_NN_RENDER_CONTEXT_ARTIFACT_FAMILY
        and _normal_token(artifact_ref.metadata.get("localization_kind", "")) == normalized_localization_kind
        and (biopsy_index is None or _artifact_biopsy_index(artifact_ref) == int(biopsy_index))
    )
    return _single_artifact_ref(candidates, "dose NN render context", "--render-context-artifact-id or --biopsy-index")


def _required_artifact_ref(artifact_refs: Sequence[ArtifactRef], artifact_id: str) -> ArtifactRef:
    artifacts_by_id = {artifact_ref.artifact_id: artifact_ref for artifact_ref in artifact_refs}
    resolved_artifact_id = str(artifact_id).strip()
    if resolved_artifact_id in artifacts_by_id:
        return artifacts_by_id[resolved_artifact_id]
    raise ValueError("patient artifact index is missing artifact ID: {}".format(resolved_artifact_id))


def _required_cli_arg(value: object | None, arg_name: str) -> object:
    if value is None:
        raise ValueError("{} is required unless --list-contexts is used".format(arg_name))
    return value


def _single_artifact_ref(
    candidates: Sequence[ArtifactRef],
    context_label: str,
    disambiguation_arg: str,
) -> ArtifactRef:
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise ValueError("patient artifact index has no matching {} artifact".format(context_label))
    candidate_ids = ", ".join(artifact_ref.artifact_id for artifact_ref in candidates)
    raise ValueError(
        "patient artifact index has multiple matching {} artifacts; pass {}. Candidates: {}".format(
            context_label,
            disambiguation_arg,
            candidate_ids,
        )
    )


def _print_available_contexts(artifact_refs: Sequence[ArtifactRef], *, localization_kind: str) -> None:
    lattice_refs = tuple(
        artifact_ref
        for artifact_ref in artifact_refs
        if artifact_ref.artifact_family == DOSE_LATTICE_CONTEXT_ARTIFACT_FAMILY
        and _normal_token(artifact_ref.metadata.get("localization_kind", "")) == _normal_token(localization_kind)
    )
    render_refs = tuple(
        artifact_ref
        for artifact_ref in artifact_refs
        if artifact_ref.artifact_family == DOSE_NN_RENDER_CONTEXT_ARTIFACT_FAMILY
        and _normal_token(artifact_ref.metadata.get("localization_kind", "")) == _normal_token(localization_kind)
    )
    print("[dose-nn-context-render] lattice contexts:")
    for artifact_ref in lattice_refs:
        print("  {} localization_kind={}".format(artifact_ref.artifact_id, artifact_ref.metadata.get("localization_kind", "")))
    print("[dose-nn-context-render] render contexts:")
    for artifact_ref in render_refs:
        print(
            "  {} biopsy_index={} roi={} localization_kind={}".format(
                artifact_ref.artifact_id,
                artifact_ref.metadata.get("biopsy_index", ""),
                artifact_ref.metadata.get("biopsy_roi", ""),
                artifact_ref.metadata.get("localization_kind", ""),
            )
        )


def _artifact_biopsy_index(artifact_ref: ArtifactRef) -> int | None:
    biopsy_index = artifact_ref.metadata.get("biopsy_index")
    if biopsy_index is None or str(biopsy_index).strip() == "":
        return None
    return int(biopsy_index)


def _normal_token(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


if __name__ == "__main__":
    raise SystemExit(main())
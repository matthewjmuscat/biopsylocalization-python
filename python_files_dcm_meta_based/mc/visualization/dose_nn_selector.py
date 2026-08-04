"""Saved-scene selection helpers for dose NN render artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import re

from ui.render_broker import RenderBrokerChoiceGroup
from ui.render_broker import RenderBrokerChoiceOption
from ui.render_broker import RenderBrokerDecision
from ui.render_broker import RenderBrokerDialogAdapter
from ui.render_broker import RenderBrokerRequest
from ui.render_broker import RenderBrokerSessionState
from ui.render_broker import RenderBrokerTimeoutPolicy
from ui.render_broker import render_backend_includes
from ui.render_broker import run_render_broker_session

from .dose_nn_pyvista import DoseNNPyVistaExportResult
from .dose_nn_pyvista import DoseNNPyVistaRenderSettings
from .dose_nn_render_service import render_saved_dose_nn_scene_artifact_pyvista
from .dose_nn_scene import DoseNNRenderConfig
from .dose_nn_scene_artifacts import DOSE_NN_RENDER_SCENE_ARTIFACT_SCHEMA_VERSION
from .dose_nn_scene_artifacts import DOSE_NN_RENDER_SCENE_MANIFEST_FILENAME
from .dose_nn_scene_artifacts import DoseNNRenderSceneArtifactManifest
from .dose_nn_scene_artifacts import read_dose_nn_render_scene_artifact_manifest


DOSE_NN_SAVED_SCENE_GROUP_KEY = "dose_nn_saved_scene"


@dataclass(frozen=True, slots=True)
class DoseNNSavedSceneOption:
    """One saved scene artifact available for post-run rendering."""

    option_key: str
    scene_id: str
    scene_artifact_dir: Path
    display_label: str
    patient_uid: str
    biopsy_roi: str
    biopsy_index: int | None
    source_label: str
    num_lattice_points: int
    num_query_points: int
    num_nearest_neighbours: int
    available_trials: tuple[int, ...] = ()
    lattice_dose_range: tuple[float, float] | None = None
    suggested_export_output_dir: Path | None = None


def discover_saved_dose_nn_scene_options(
    search_root: Path | str,
    *,
    suggested_export_root: Path | str | None = None,
    strict: bool = False,
) -> tuple[DoseNNSavedSceneOption, ...]:
    """Discover saved dose NN render scene artifacts below a directory."""
    resolved_search_root = Path(search_root)
    resolved_suggested_export_root = None if suggested_export_root is None else Path(suggested_export_root)
    manifest_paths = sorted(resolved_search_root.rglob(DOSE_NN_RENDER_SCENE_MANIFEST_FILENAME))
    used_option_keys: set[str] = set()
    options: list[DoseNNSavedSceneOption] = []
    for manifest_path in manifest_paths:
        scene_artifact_dir = manifest_path.parent
        try:
            manifest = read_dose_nn_render_scene_artifact_manifest(scene_artifact_dir)
            if manifest.schema_version != DOSE_NN_RENDER_SCENE_ARTIFACT_SCHEMA_VERSION:
                continue
            option = build_saved_dose_nn_scene_option(
                manifest,
                scene_artifact_dir,
                search_root=resolved_search_root,
                suggested_export_root=resolved_suggested_export_root,
                used_option_keys=used_option_keys,
            )
        except Exception:
            if strict:
                raise
            continue
        used_option_keys.add(option.option_key)
        options.append(option)
    return tuple(options)


def build_saved_dose_nn_scene_option(
    manifest: DoseNNRenderSceneArtifactManifest,
    scene_artifact_dir: Path | str,
    *,
    search_root: Path | str | None = None,
    suggested_export_root: Path | str | None = None,
    used_option_keys: set[str] | None = None,
) -> DoseNNSavedSceneOption:
    """Build a selector option from a saved scene manifest without loading arrays."""
    resolved_scene_artifact_dir = Path(scene_artifact_dir)
    relative_scene_dir = _relative_scene_dir(resolved_scene_artifact_dir, search_root)
    option_key = _unique_option_key(
        manifest.scene_id,
        relative_scene_dir,
        used_option_keys or set(),
    )
    num_lattice_points = int(manifest.summary.get("num_lattice_points", _first_dimension(manifest, "lattice_points")))
    num_query_points = int(manifest.summary.get("num_query_points", _first_dimension(manifest, "biopsy_points")))
    num_nearest_neighbours = int(
        manifest.summary.get("num_nearest_neighbours", _nearest_neighbour_count(manifest))
    )
    available_trials = _available_trials_from_summary(manifest.summary)
    lattice_dose_range = _scalar_range_from_summary(manifest.summary, "lattice_dose_range")
    metadata = manifest.metadata
    patient_uid = str(metadata.get("patient_uid", ""))
    biopsy_roi = str(metadata.get("biopsy_roi", ""))
    biopsy_index = metadata.get("biopsy_index")
    if biopsy_index is not None:
        biopsy_index = int(biopsy_index)
    source_label = str(metadata.get("source_label", ""))
    suggested_export_output_dir = None
    if suggested_export_root is not None:
        suggested_export_output_dir = Path(suggested_export_root).joinpath(_sanitize_path_fragment(option_key))

    return DoseNNSavedSceneOption(
        option_key=option_key,
        scene_id=str(manifest.scene_id),
        scene_artifact_dir=resolved_scene_artifact_dir,
        display_label=_format_saved_scene_label(
            scene_id=str(manifest.scene_id),
            patient_uid=patient_uid,
            biopsy_roi=biopsy_roi,
            biopsy_index=biopsy_index,
            source_label=source_label,
            num_lattice_points=num_lattice_points,
            num_query_points=num_query_points,
            num_nearest_neighbours=num_nearest_neighbours,
            available_trials=available_trials,
            lattice_dose_range=lattice_dose_range,
        ),
        patient_uid=patient_uid,
        biopsy_roi=biopsy_roi,
        biopsy_index=biopsy_index,
        source_label=source_label,
        num_lattice_points=num_lattice_points,
        num_query_points=num_query_points,
        num_nearest_neighbours=num_nearest_neighbours,
        available_trials=available_trials,
        lattice_dose_range=lattice_dose_range,
        suggested_export_output_dir=suggested_export_output_dir,
    )


def build_saved_dose_nn_scene_choice_group(
    options: Sequence[DoseNNSavedSceneOption],
) -> RenderBrokerChoiceGroup:
    """Build the generic broker choice group for saved dose NN scenes."""
    choice_options = tuple(
        RenderBrokerChoiceOption(
            option_key=option.option_key,
            display_label=option.display_label,
            selected_by_default=(option_index == 0),
            suggested_export_output_dir=option.suggested_export_output_dir,
        )
        for option_index, option in enumerate(tuple(options))
    )
    return RenderBrokerChoiceGroup(
        group_key=DOSE_NN_SAVED_SCENE_GROUP_KEY,
        display_label="Saved dose NN scenes",
        description="Select a retained scene artifact for post-run PyVista rendering.",
        selection_mode="single",
        options=choice_options,
        default_backend="pyvista",
        render_action_label="Render saved scene",
        empty_state_message="No saved dose NN scene artifacts were found.",
        allow_pyvista=True,
    )


def build_saved_dose_nn_scene_broker_request(
    options: Sequence[DoseNNSavedSceneOption],
    *,
    summary_lines: Sequence[str] = (),
    timeout_policy: RenderBrokerTimeoutPolicy | None = None,
) -> RenderBrokerRequest:
    """Build a broker request for saved dose NN scene review."""
    return RenderBrokerRequest(
        title="Dose NN saved-scene renderer",
        summary_lines=tuple(str(summary_line) for summary_line in tuple(summary_lines)),
        choice_groups=(build_saved_dose_nn_scene_choice_group(options),),
        continue_button_label="Continue without rendering",
        timeout_policy=timeout_policy,
    )


def handle_saved_dose_nn_scene_broker_decision_pyvista(
    decision: RenderBrokerDecision,
    options: Mapping[str, DoseNNSavedSceneOption] | Sequence[DoseNNSavedSceneOption],
    output_dir: Path | str,
    *,
    config: DoseNNRenderConfig | None = None,
    settings: DoseNNPyVistaRenderSettings | None = None,
    overwrite: bool = False,
) -> tuple[DoseNNPyVistaExportResult, ...]:
    """Render selected saved scenes from a broker decision through PyVista."""
    if decision.group_key != DOSE_NN_SAVED_SCENE_GROUP_KEY:
        raise ValueError("unsupported dose NN render broker group: {}".format(decision.group_key))
    if not render_backend_includes(decision.render_backend, "pyvista"):
        raise ValueError("dose NN saved-scene selector currently supports PyVista rendering only")
    return render_saved_dose_nn_scene_selection_pyvista(
        options,
        decision.selected_option_keys,
        output_dir,
        config=config,
        settings=settings,
        overwrite=overwrite,
    )


def run_saved_dose_nn_scene_selector_session(
    search_root: Path | str,
    output_dir: Path | str,
    *,
    suggested_export_root: Path | str | None = None,
    config: DoseNNRenderConfig | None = None,
    settings: DoseNNPyVistaRenderSettings | None = None,
    dialog_adapter: RenderBrokerDialogAdapter | None = None,
    timeout_policy: RenderBrokerTimeoutPolicy | None = None,
    initial_session_state: RenderBrokerSessionState | None = None,
    overwrite: bool = False,
) -> RenderBrokerSessionState:
    """Run the saved-scene selector loop through the generic render broker."""
    options = discover_saved_dose_nn_scene_options(
        search_root,
        suggested_export_root=suggested_export_root,
    )
    request = build_saved_dose_nn_scene_broker_request(
        options,
        summary_lines=("{} saved scene artifact(s) discovered.".format(len(options)),),
        timeout_policy=timeout_policy,
    )

    resolved_dialog_adapter = dialog_adapter
    if resolved_dialog_adapter is None:
        from ui.tk_render_broker import TkRenderBrokerDialogAdapter

        resolved_dialog_adapter = TkRenderBrokerDialogAdapter()

    def _handle_decision(decision: RenderBrokerDecision) -> None:
        handle_saved_dose_nn_scene_broker_decision_pyvista(
            decision,
            options,
            output_dir,
            config=config,
            settings=settings,
            overwrite=overwrite,
        )

    return run_render_broker_session(
        request,
        resolved_dialog_adapter,
        _handle_decision,
        initial_session_state=initial_session_state,
    )


def render_saved_dose_nn_scene_selection_pyvista(
    options: Mapping[str, DoseNNSavedSceneOption] | Sequence[DoseNNSavedSceneOption],
    selected_option_keys: Sequence[str],
    output_dir: Path | str,
    *,
    config: DoseNNRenderConfig | None = None,
    settings: DoseNNPyVistaRenderSettings | None = None,
    overwrite: bool = False,
) -> tuple[DoseNNPyVistaExportResult, ...]:
    """Render selected saved scene options through PyVista."""
    options_by_key = _options_by_key(options)
    resolved_output_dir = Path(output_dir)
    results: list[DoseNNPyVistaExportResult] = []
    for selected_option_key in tuple(selected_option_keys):
        if selected_option_key not in options_by_key:
            raise ValueError("unknown dose NN saved scene option: {}".format(selected_option_key))
        option = options_by_key[selected_option_key]
        output_path = resolved_output_dir.joinpath("{}.png".format(_sanitize_path_fragment(option.option_key)))
        provenance_path = output_path.with_suffix(".png.provenance.json")
        if not overwrite and (output_path.exists() or provenance_path.exists()):
            raise FileExistsError("dose NN render output already exists: {}".format(output_path))
        results.append(
            render_saved_dose_nn_scene_artifact_pyvista(
                option.scene_artifact_dir,
                output_path,
                config=config,
                settings=settings,
                provenance_path=provenance_path,
            )
        )
    return tuple(results)


def _options_by_key(
    options: Mapping[str, DoseNNSavedSceneOption] | Sequence[DoseNNSavedSceneOption],
) -> dict[str, DoseNNSavedSceneOption]:
    if isinstance(options, Mapping):
        return {str(option_key): option for option_key, option in options.items()}
    return {option.option_key: option for option in tuple(options)}


def _relative_scene_dir(scene_artifact_dir: Path, search_root: Path | str | None) -> str:
    if search_root is None:
        return scene_artifact_dir.name
    try:
        return scene_artifact_dir.relative_to(Path(search_root)).as_posix()
    except ValueError:
        return scene_artifact_dir.as_posix()


def _unique_option_key(scene_id: str, relative_scene_dir: str, used_option_keys: set[str]) -> str:
    base_key = _sanitize_option_key(scene_id) or _sanitize_option_key(relative_scene_dir) or "scene"
    option_key = base_key
    duplicate_index = 2
    while option_key in used_option_keys:
        option_key = "{}__{:02d}".format(base_key, duplicate_index)
        duplicate_index += 1
    return option_key


def _first_dimension(manifest: DoseNNRenderSceneArtifactManifest, array_name: str) -> int:
    array_spec = manifest.array_specs_by_name.get(array_name)
    if array_spec is None or len(array_spec.shape) == 0:
        return 0
    return int(array_spec.shape[0])


def _nearest_neighbour_count(manifest: DoseNNRenderSceneArtifactManifest) -> int:
    array_spec = manifest.array_specs_by_name.get("nearest_lattice_points")
    if array_spec is None or len(array_spec.shape) < 2:
        return 0
    return int(array_spec.shape[1])


def _format_saved_scene_label(
    *,
    scene_id: str,
    patient_uid: str,
    biopsy_roi: str,
    biopsy_index: int | None,
    source_label: str,
    num_lattice_points: int,
    num_query_points: int,
    num_nearest_neighbours: int,
    available_trials: tuple[int, ...] = (),
    lattice_dose_range: tuple[float, float] | None = None,
) -> str:
    identity_parts = [part for part in (patient_uid, biopsy_roi, _biopsy_index_label(biopsy_index)) if part != ""]
    identity_label = " / ".join(identity_parts) if identity_parts else "Unlabeled scene"
    source_suffix = "" if source_label == "" else " | {}".format(source_label)
    trial_suffix = "" if len(available_trials) == 0 else " | trials: {}".format(len(available_trials))
    dose_suffix = "" if lattice_dose_range is None else " | dose: {:.3g}-{:.3g}".format(*lattice_dose_range)
    return (
        "{} | {} | lattice points: {} | query rows: {} | k: {}{}{}{}".format(
            identity_label,
            scene_id,
            int(num_lattice_points),
            int(num_query_points),
            int(num_nearest_neighbours),
            trial_suffix,
            dose_suffix,
            source_suffix,
        )
    )


def _available_trials_from_summary(summary: Mapping[str, Any]) -> tuple[int, ...]:
    return tuple(int(trial_number) for trial_number in tuple(summary.get("available_trials", ())))


def _scalar_range_from_summary(summary: Mapping[str, Any], key: str) -> tuple[float, float] | None:
    scalar_range = summary.get(key)
    if not isinstance(scalar_range, Mapping):
        return None
    if "min" not in scalar_range or "max" not in scalar_range:
        return None
    return (float(scalar_range["min"]), float(scalar_range["max"]))


def _biopsy_index_label(biopsy_index: int | None) -> str:
    if biopsy_index is None:
        return ""
    return "biopsy {}".format(int(biopsy_index))


def _sanitize_option_key(value: str) -> str:
    sanitized_value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return sanitized_value.strip("_")


def _sanitize_path_fragment(value: str) -> str:
    sanitized_value = _sanitize_option_key(value)
    return sanitized_value or "dose_nn_scene"
"""Post-run service and CLI for rendering saved dose NN scene artifacts."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .dose_nn_pyvista import DoseNNPyVistaMovieExportResult
from .dose_nn_pyvista import DoseNNPyVistaExportResult
from .dose_nn_pyvista import DoseNNPyVistaFrameSequenceExportResult
from .dose_nn_pyvista import DoseNNPyVistaRenderSettings
from .dose_nn_pyvista import capture_dose_nn_scene_camera_pyvista
from .dose_nn_pyvista import export_dose_nn_trial_movie_pyvista
from .dose_nn_pyvista import export_dose_nn_trial_frame_sequence_pyvista
from .dose_nn_pyvista import export_dose_nn_scene_pyvista
from .dose_nn_scene import DoseNNRenderConfig
from .dose_nn_scene_artifacts import read_dose_nn_render_scene_artifact


def render_saved_dose_nn_scene_artifact_pyvista(
    scene_artifact_dir: Path | str,
    output_path: Path | str,
    *,
    config: DoseNNRenderConfig | None = None,
    settings: DoseNNPyVistaRenderSettings | None = None,
    provenance_path: Path | str | None = None,
) -> DoseNNPyVistaExportResult:
    """Read a saved dose NN scene artifact and export it through PyVista."""
    scene = read_dose_nn_render_scene_artifact(scene_artifact_dir)
    return export_dose_nn_scene_pyvista(
        scene,
        output_path,
        config=config,
        settings=settings,
        provenance_path=provenance_path,
    )


def capture_saved_dose_nn_scene_camera_pyvista(
    scene_artifact_dir: Path | str,
    *,
    config: DoseNNRenderConfig | None = None,
    settings: DoseNNPyVistaRenderSettings | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Open a saved scene interactively and return the user-selected PyVista camera."""
    scene = read_dose_nn_render_scene_artifact(scene_artifact_dir)
    return capture_dose_nn_scene_camera_pyvista(scene, config=config, settings=settings)


def export_saved_dose_nn_scene_trial_frames_pyvista(
    scene_artifact_dir: Path | str,
    output_dir: Path | str,
    *,
    selected_trials: tuple[int, ...] | None = None,
    max_frames: int | None = None,
    frames_per_second: float = 12.0,
    config: DoseNNRenderConfig | None = None,
    settings: DoseNNPyVistaRenderSettings | None = None,
    overwrite: bool = False,
) -> DoseNNPyVistaFrameSequenceExportResult:
    """Read a saved scene artifact and export one PyVista frame per trial."""
    scene = read_dose_nn_render_scene_artifact(scene_artifact_dir)
    return export_dose_nn_trial_frame_sequence_pyvista(
        scene,
        output_dir,
        selected_trials=selected_trials,
        max_frames=max_frames,
        frames_per_second=frames_per_second,
        base_config=config,
        settings=settings,
        overwrite=overwrite,
    )


def export_saved_dose_nn_scene_trial_movie_pyvista(
    scene_artifact_dir: Path | str,
    output_dir: Path | str,
    *,
    video_path: Path | str | None = None,
    video_format: str | None = None,
    selected_trials: tuple[int, ...] | None = None,
    max_frames: int | None = None,
    frames_per_second: float = 12.0,
    camera_z_orbit_degrees: float = 0.0,
    config: DoseNNRenderConfig | None = None,
    settings: DoseNNPyVistaRenderSettings | None = None,
    overwrite: bool = False,
) -> DoseNNPyVistaMovieExportResult:
    """Read a saved scene artifact, render per-trial frames, and encode a movie."""
    scene = read_dose_nn_render_scene_artifact(scene_artifact_dir)
    return export_dose_nn_trial_movie_pyvista(
        scene,
        output_dir,
        video_path=video_path,
        video_format=video_format,
        selected_trials=selected_trials,
        max_frames=max_frames,
        frames_per_second=frames_per_second,
        camera_z_orbit_degrees=camera_z_orbit_degrees,
        base_config=config,
        settings=settings,
        overwrite=overwrite,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for rendering a saved dose NN scene artifact."""
    args = _build_argument_parser().parse_args(argv)
    if args.backend != "pyvista":
        raise ValueError("unsupported dose NN render backend: {}".format(args.backend))

    if args.export_trial_movie_path is not None:
        settings = _pyvista_settings_from_args(args)
        if bool(args.capture_camera):
            settings = replace(
                settings,
                camera_position=capture_saved_dose_nn_scene_camera_pyvista(
                    args.scene_dir,
                    config=_config_from_args(args),
                    settings=settings,
                ),
            )
        result = export_saved_dose_nn_scene_trial_movie_pyvista(
            args.scene_dir,
            _movie_output_dir_from_args(args),
            video_path=args.export_trial_movie_path,
            video_format=args.movie_format,
            selected_trials=_selected_trials_from_args(args),
            max_frames=args.max_trial_frames,
            frames_per_second=args.frames_per_second,
            camera_z_orbit_degrees=args.camera_z_orbit_degrees,
            config=_config_from_args(args, include_selected_trials=False),
            settings=settings,
            overwrite=bool(args.overwrite),
        )
        print("[dose-nn-render] wrote {} frame(s)".format(len(result.frame_paths)))
        print("[dose-nn-render] wrote {}".format(result.video_path))
        print("[dose-nn-render] wrote {}".format(result.manifest_path))
        return 0

    if args.export_trial_frames_dir is not None:
        result = export_saved_dose_nn_scene_trial_frames_pyvista(
            args.scene_dir,
            args.export_trial_frames_dir,
            selected_trials=_selected_trials_from_args(args),
            max_frames=args.max_trial_frames,
            frames_per_second=args.frames_per_second,
            config=_config_from_args(args, include_selected_trials=False),
            settings=_pyvista_settings_from_args(args),
            overwrite=bool(args.overwrite),
        )
        print("[dose-nn-render] wrote {} frame(s)".format(len(result.frame_paths)))
        print("[dose-nn-render] wrote {}".format(result.manifest_path))
        return 0

    result = render_saved_dose_nn_scene_artifact_pyvista(
        args.scene_dir,
        _required_screenshot_output_path(args),
        config=_config_from_args(args),
        settings=_pyvista_settings_from_args(args),
        provenance_path=args.provenance_path,
    )
    print("[dose-nn-render] wrote {}".format(result.screenshot_path))
    print("[dose-nn-render] wrote {}".format(result.provenance_path))
    return 0


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a saved dose NN scene artifact without rerunning scientific code.",
    )
    parser.add_argument("--scene-dir", required=True, type=Path, help="Directory containing manifest.json and arrays.")
    parser.add_argument("--output", type=Path, default=None, help="Screenshot output path, usually .png.")
    parser.add_argument("--provenance-path", type=Path, default=None, help="Optional provenance JSON output path.")
    parser.add_argument("--backend", choices=("pyvista",), default="pyvista")
    parser.add_argument("--export-trial-frames-dir", type=Path, default=None)
    parser.add_argument("--export-trial-movie-path", type=Path, default=None)
    parser.add_argument("--movie-format", choices=("mp4", "webm"), default=None)
    parser.add_argument("--camera-z-orbit-degrees", type=float, default=0.0)
    parser.add_argument("--capture-camera", action="store_true")
    parser.add_argument("--frames-per-second", type=float, default=12.0)
    parser.add_argument("--max-trial-frames", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trial", action="append", type=int, default=None, help="Trial number to include; repeatable.")
    parser.add_argument("--dose-threshold-min", type=float, default=None)
    parser.add_argument("--dose-threshold-max", type=float, default=None)
    parser.add_argument("--max-lattice-points", type=int, default=None)
    parser.add_argument("--spatial-radius-mm", type=float, default=None)
    parser.add_argument("--biopsy-point-stride", type=int, default=1)
    parser.add_argument("--vector-stride", type=int, default=1)
    parser.add_argument("--hide-biopsy-points", action="store_true")
    parser.add_argument("--show-reference-biopsy-points", action="store_true")
    parser.add_argument(
        "--reference-biopsy-trial",
        action="append",
        type=int,
        default=None,
        help="Trial whose biopsy query points should remain visible as a reference; repeatable.",
    )
    lattice_point_group = parser.add_mutually_exclusive_group()
    lattice_point_group.add_argument("--show-lattice-points", action="store_true")
    lattice_point_group.add_argument("--hide-lattice-points", action="store_true")
    parser.add_argument("--show-dose-colorwash", action="store_true")
    parser.add_argument("--hide-nearest-neighbour-points", action="store_true")
    parser.add_argument("--hide-nearest-neighbour-vectors", action="store_true")
    parser.add_argument("--window-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"), default=(1200, 900))
    parser.add_argument("--background-color", default="white")
    parser.add_argument("--dose-colormap", default="coolwarm")
    parser.add_argument("--dose-color-scale-mode", choices=("linear", "log"), default="linear")
    parser.add_argument("--dose-color-scale-min", type=float, default=None)
    parser.add_argument("--dose-color-scale-max", type=float, default=None)
    parser.add_argument("--lattice-point-size", type=float, default=5.0)
    parser.add_argument("--dose-colorwash-style", choices=("points", "volume", "auto"), default="points")
    parser.add_argument("--dose-colorwash-volume-max-voxels", type=int, default=250_000)
    parser.add_argument("--dose-colorwash-point-size", type=float, default=12.0)
    parser.add_argument("--dose-colorwash-opacity", type=float, default=0.08)
    parser.add_argument("--dose-colorwash-point-opacity-mode", choices=("constant", "center_fade"), default="constant")
    parser.add_argument("--dose-colorwash-point-opacity-min", type=float, default=0.02)
    parser.add_argument("--biopsy-point-size", type=float, default=12.0)
    parser.add_argument("--reference-biopsy-point-size", type=float, default=10.0)
    parser.add_argument("--nearest-point-size", type=float, default=8.0)
    parser.add_argument("--vector-line-width", type=float, default=2.0)
    parser.add_argument("--no-axes", action="store_true")
    parser.add_argument("--no-scalar-bar", action="store_true")
    parser.add_argument("--dose-scalar-bar-title", default="Dose (Gy)")
    parser.add_argument("--dose-scalar-bar-title-font-size", type=int, default=18)
    parser.add_argument("--dose-scalar-bar-label-font-size", type=int, default=14)
    parser.add_argument("--no-scalar-bar-background", action="store_true")
    parser.add_argument("--x-axis-label", default="Left-Right x (mm)")
    parser.add_argument("--y-axis-label", default="Posterior-Anterior y (mm)")
    parser.add_argument("--z-axis-label", default="Inferior-Superior z (mm)")
    parser.add_argument("--axes-title-font-size", type=int, default=18)
    parser.add_argument("--axes-tick-label-font-size", type=int, default=14)
    return parser


def _config_from_args(args: argparse.Namespace, *, include_selected_trials: bool = True) -> DoseNNRenderConfig:
    selected_trials = _selected_trials_from_args(args) if include_selected_trials else None
    return DoseNNRenderConfig(
        selected_trials=selected_trials,
        reference_trial_numbers=_reference_trial_numbers_from_args(args),
        dose_threshold_min=args.dose_threshold_min,
        dose_threshold_max=args.dose_threshold_max,
        max_lattice_points=args.max_lattice_points,
        spatial_radius_mm=args.spatial_radius_mm,
        biopsy_point_stride=args.biopsy_point_stride,
        vector_stride=args.vector_stride,
        show_biopsy_points=not bool(args.hide_biopsy_points),
        show_reference_biopsy_points=_show_reference_biopsy_points_from_args(args),
        show_lattice_points=_show_lattice_points_from_args(args),
        show_dose_colorwash=bool(args.show_dose_colorwash),
        show_nearest_neighbour_points=not bool(args.hide_nearest_neighbour_points),
        show_nearest_neighbour_vectors=not bool(args.hide_nearest_neighbour_vectors),
    )


def _selected_trials_from_args(args: argparse.Namespace) -> tuple[int, ...] | None:
    return None if args.trial is None else tuple(int(trial_number) for trial_number in args.trial)


def _reference_trial_numbers_from_args(args: argparse.Namespace) -> tuple[int, ...] | None:
    if args.reference_biopsy_trial is not None:
        return tuple(int(trial_number) for trial_number in args.reference_biopsy_trial)
    if bool(args.show_reference_biopsy_points):
        return (0,)
    return None


def _show_reference_biopsy_points_from_args(args: argparse.Namespace) -> bool:
    return bool(args.show_reference_biopsy_points or args.reference_biopsy_trial is not None)


def _show_lattice_points_from_args(args: argparse.Namespace) -> bool:
    if bool(args.show_lattice_points):
        return True
    if bool(args.hide_lattice_points):
        return False
    if bool(args.show_dose_colorwash):
        return False
    return True


def _required_screenshot_output_path(args: argparse.Namespace) -> Path:
    if args.output is None:
        raise ValueError("--output is required unless a frame/movie export option is used")
    return args.output


def _movie_output_dir_from_args(args: argparse.Namespace) -> Path:
    if args.export_trial_frames_dir is not None:
        return args.export_trial_frames_dir
    return args.export_trial_movie_path.parent


def _pyvista_settings_from_args(args: argparse.Namespace) -> DoseNNPyVistaRenderSettings:
    return DoseNNPyVistaRenderSettings(
        off_screen=True,
        window_size=tuple(int(value) for value in args.window_size),
        background_color=args.background_color,
        dose_colormap=args.dose_colormap,
        dose_color_scale_mode=args.dose_color_scale_mode,
        dose_color_scale_min=args.dose_color_scale_min,
        dose_color_scale_max=args.dose_color_scale_max,
        lattice_point_size=args.lattice_point_size,
        dose_colorwash_style=args.dose_colorwash_style,
        dose_colorwash_volume_max_voxels=args.dose_colorwash_volume_max_voxels,
        dose_colorwash_point_size=args.dose_colorwash_point_size,
        dose_colorwash_opacity=args.dose_colorwash_opacity,
        dose_colorwash_point_opacity_mode=args.dose_colorwash_point_opacity_mode,
        dose_colorwash_point_opacity_min=args.dose_colorwash_point_opacity_min,
        biopsy_point_size=args.biopsy_point_size,
        reference_biopsy_point_size=args.reference_biopsy_point_size,
        nearest_point_size=args.nearest_point_size,
        vector_line_width=args.vector_line_width,
        show_axes=not bool(args.no_axes),
        show_scalar_bar=not bool(args.no_scalar_bar),
        dose_scalar_bar_title=args.dose_scalar_bar_title,
        dose_scalar_bar_title_font_size=args.dose_scalar_bar_title_font_size,
        dose_scalar_bar_label_font_size=args.dose_scalar_bar_label_font_size,
        dose_scalar_bar_show_background=not bool(args.no_scalar_bar_background),
        x_axis_label=args.x_axis_label,
        y_axis_label=args.y_axis_label,
        z_axis_label=args.z_axis_label,
        axes_title_font_size=args.axes_title_font_size,
        axes_tick_label_font_size=args.axes_tick_label_font_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())
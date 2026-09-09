from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


DEFAULT_OUTPUT_FIGURES_DIR_NAME = "Output figures"
DEFAULT_GUIDANCE_MAP_OUTPUT_DIR_NAME = "Guidance maps"


@dataclass(frozen=True)
class GuidanceMapPlanningConfig:
    """Configuration for non-plotting guidance-map planning tables."""

    candidate_holes_k: int = 1
    candidate_axis_line_length_mm: float = 1000.0
    downcast_threshold: float = 0.25


@dataclass(frozen=True)
class GuidanceMapRenderConfig:
    """Presentation/export policy for post-computation guidance-map rendering."""

    enabled: bool = False
    plot_name: str = "guidance maps"
    output_figures_dir_name: str = DEFAULT_OUTPUT_FIGURES_DIR_NAME
    output_dir_name: str = DEFAULT_GUIDANCE_MAP_OUTPUT_DIR_NAME
    save_formats: tuple[str, ...] = ("svg", "pdf", "html")
    image_width: int = 1300
    image_height: int = 1300
    image_scale: float = 1.0
    axis_title_font_size: int = 24
    axis_tick_font_size: int = 20
    legend_font_size: int = 20
    annotation_font_size: int = 20
    distance_annotation_font_size: int = 20
    fire_annotation_font_size: int = 20
    colorbar_tick_font_size: int = 20
    template_label_font_size: int = 20
    colorbar_title_font_size: int = 20
    fire_annotation_style: str = "compact_table"
    fire_table_position: str = "outside top center"
    draw_orientation_diagram: bool = False
    show_titles: bool = False
    show_euler_annotation_box: bool = True
    candidate_plot_rank: int | Sequence[int] | str = 1
    validate_firing_df_builder: bool = False
    strict_precomputed_guidance: bool = False
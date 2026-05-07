from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import production_plots
import rich_preambles


GUIDANCE_MAP_PATIENT_OUTPUT_DIR_DICT_KEY = "Patient specific guidance map figures directory dict"
GUIDANCE_MAP_OUTPUT_DIR_KEY = "Guidance map figures dir"
DEFAULT_OUTPUT_FIGURES_DIR_NAME = "Output figures"
DEFAULT_GUIDANCE_MAP_OUTPUT_DIR_NAME = "Guidance maps"


@dataclass(frozen=True)
class GuidanceMapRenderConfig:
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


def build_guidance_map_output_directories(master_structure_reference_dict,
                                         master_structure_info_dict,
                                         specific_output_dir,
                                         guidance_map_render_config):
    specific_output_dir = Path(specific_output_dir)
    output_figures_dir = specific_output_dir.joinpath(
        guidance_map_render_config.output_figures_dir_name,
    )
    output_figures_dir.mkdir(parents=True, exist_ok=True)

    guidance_map_output_dir = output_figures_dir.joinpath(
        guidance_map_render_config.output_dir_name,
    )
    guidance_map_output_dir.mkdir(parents=True, exist_ok=True)

    patient_output_dir_dict = {}
    for patientUID in master_structure_reference_dict.keys():
        patient_output_dir = guidance_map_output_dir.joinpath(patientUID)
        patient_output_dir.mkdir(parents=True, exist_ok=True)
        patient_output_dir_dict[patientUID] = patient_output_dir

    global_output_dir = guidance_map_output_dir.joinpath("Global")
    global_output_dir.mkdir(parents=True, exist_ok=True)
    patient_output_dir_dict["Global"] = global_output_dir

    global_info = master_structure_info_dict.setdefault("Global", {})
    global_info[GUIDANCE_MAP_PATIENT_OUTPUT_DIR_DICT_KEY] = patient_output_dir_dict
    global_info[GUIDANCE_MAP_OUTPUT_DIR_KEY] = guidance_map_output_dir
    return patient_output_dir_dict


def render_guidance_maps_for_run(master_structure_reference_dict,
                                 master_structure_info_dict,
                                 dil_ref,
                                 oar_ref,
                                 rectum_ref,
                                 all_ref_key,
                                 structs_referenced_dict,
                                 plot_open3d_structure_set_complete_demonstration_bool,
                                 biopsy_fire_travel_distances,
                                 biopsy_needle_compartment_length,
                                 interp_inter_slice_dist,
                                 interp_intra_slice_dist,
                                 radius_for_normals_estimation,
                                 max_nn_for_normals_estimation,
                                 biopsy_needle_tip_length,
                                 guidance_map_render_config,
                                 important_info=None,
                                 live_display=None,
                                 patients_progress=None,
                                 completed_progress=None,
                                 runtime_logger=None):
    if guidance_map_render_config.enabled is False:
        return None

    global_info = master_structure_info_dict.setdefault("Global", {})
    specific_output_dir = global_info.get("Specific output dir")
    if specific_output_dir is None:
        raise ValueError("Cannot render guidance maps before 'Specific output dir' is initialized.")

    if live_display is None:
        live_display = rich_preambles.NullLiveDisplay()

    patient_output_dir_dict = build_guidance_map_output_directories(
        master_structure_reference_dict,
        master_structure_info_dict,
        specific_output_dir,
        guidance_map_render_config,
    )

    num_cases = len(master_structure_reference_dict)
    if runtime_logger is not None:
        runtime_logger.phase_start(
            "guidance_maps.render",
            "Rendering guidance maps from precomputed guidance data.",
            details={
                "num_cases": num_cases,
                "guidance_map_output_dir": global_info.get(GUIDANCE_MAP_OUTPUT_DIR_KEY),
            },
        )

    if important_info is not None:
        important_info.add_text_line(
            "Rendering guidance maps from precomputed guidance data.",
            live_display,
        )

    processing_patients_task = None
    processing_patients_completed_task = None
    if patients_progress is not None and completed_progress is not None:
        patientUID_default = "Initializing"
        description = "Rendering guidance maps [{}]...".format(patientUID_default)
        processing_patients_task = patients_progress.add_task(
            "[red]" + description,
            total=num_cases,
        )
        processing_patients_completed_task = completed_progress.add_task(
            "[green]Rendering guidance maps",
            total=num_cases,
            visible=False,
        )

    live_display_was_managed = False
    try:
        live_display.stop()
        live_display_was_managed = True
    except Exception:
        live_display_was_managed = False

    try:
        for patientUID, pydicom_item in master_structure_reference_dict.items():
            if runtime_logger is not None:
                runtime_logger.info(
                    "guidance_maps.render",
                    "Rendering guidance maps for patient.",
                    patient_uid=patientUID,
                )

            if processing_patients_task is not None:
                description = "Rendering guidance maps [{}]...".format(patientUID)
                patients_progress.update(
                    processing_patients_task,
                    description="[red]" + description,
                )

            patient_output_dir = patient_output_dir_dict[patientUID]
            production_plots.guidance_map_transducer_angle_sagittal_and_max_plane_transverse(
                patientUID,
                patient_output_dir,
                pydicom_item,
                dil_ref,
                oar_ref,
                rectum_ref,
                all_ref_key,
                structs_referenced_dict,
                plot_open3d_structure_set_complete_demonstration_bool,
                biopsy_fire_travel_distances,
                biopsy_needle_compartment_length,
                interp_inter_slice_dist,
                interp_intra_slice_dist,
                radius_for_normals_estimation,
                max_nn_for_normals_estimation,
                important_info,
                live_display,
                guidance_map_render_config.image_scale,
                guidance_map_render_config.image_width,
                guidance_map_render_config.image_height,
                guidance_map_render_config.plot_name,
                biopsy_needle_tip_length,
                save_formats=list(guidance_map_render_config.save_formats),
                axis_title_font_size=guidance_map_render_config.axis_title_font_size,
                axis_tick_font_size=guidance_map_render_config.axis_tick_font_size,
                legend_font_size=guidance_map_render_config.legend_font_size,
                annotation_font_size=guidance_map_render_config.annotation_font_size,
                distance_annotation_font_size=guidance_map_render_config.distance_annotation_font_size,
                fire_annotation_font_size=guidance_map_render_config.fire_annotation_font_size,
                colorbar_tick_font_size=guidance_map_render_config.colorbar_tick_font_size,
                template_label_font_size=guidance_map_render_config.template_label_font_size,
                colorbar_title_font_size=guidance_map_render_config.colorbar_title_font_size,
                fire_annotation_style=guidance_map_render_config.fire_annotation_style,
                fire_table_position=guidance_map_render_config.fire_table_position,
                draw_orientation_diagram=guidance_map_render_config.draw_orientation_diagram,
                show_titles=guidance_map_render_config.show_titles,
                show_euler_annotation_box=guidance_map_render_config.show_euler_annotation_box,
                candidate_plot_rank=guidance_map_render_config.candidate_plot_rank,
                validate_firing_df_builder=guidance_map_render_config.validate_firing_df_builder,
                strict_precomputed_guidance=guidance_map_render_config.strict_precomputed_guidance,
            )

            if processing_patients_task is not None:
                patients_progress.update(processing_patients_task, advance=1)
                completed_progress.update(processing_patients_completed_task, advance=1)

        if processing_patients_task is not None:
            patients_progress.update(processing_patients_task, visible=False)
            completed_progress.update(processing_patients_completed_task, visible=True)
            live_display.refresh()
    finally:
        if live_display_was_managed:
            try:
                live_display.start(refresh=True)
            except TypeError:
                live_display.start()
            live_display.refresh()

    if runtime_logger is not None:
        runtime_logger.phase_end(
            "guidance_maps.render",
            "Completed guidance-map rendering.",
            details={"num_cases": num_cases},
            clear_phase=True,
        )

    return patient_output_dir_dict
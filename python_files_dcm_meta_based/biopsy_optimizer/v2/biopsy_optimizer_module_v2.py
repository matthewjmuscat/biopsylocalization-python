from typing import Any, Optional, Sequence, Tuple

import math
import numpy as np

import MC_simulator_convex
import custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p
from biopsy_optimizer.v2.config import (
	DEFAULT_OPTIMIZER_V2_STAGE_CONFIGS,
	OptimizerV2SearchConfig,
	OptimizerV2StageConfig,
	OptimizerV2VisualizationConfig,
	build_default_optimizer_v2_search_config,
	build_default_optimizer_v2_visualization_config,
	build_optimizer_v2_search_config_with_trial_counts,
)
from biopsy_optimizer.v2.contracts import (
	OptimizerV2CandidatePool,
	OptimizerV2ChunkLayout,
)


def build_target_candidate_lattice(
	target_points_array: np.ndarray,
	lattice_spacing_mm: float,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
	normalized_target_points = _validate_xyz_points_array(target_points_array, "target_points_array")
	if lattice_spacing_mm <= 0.0:
		raise ValueError("lattice_spacing_mm must be positive")

	min_bounds = np.amin(normalized_target_points, axis=0)
	max_bounds = np.amax(normalized_target_points, axis=0)

	lattice_sizex = int(math.ceil(abs(max_bounds[0] - min_bounds[0]) / lattice_spacing_mm) + 1)
	lattice_sizey = int(math.ceil(abs(max_bounds[1] - min_bounds[1]) / lattice_spacing_mm) + 1)
	lattice_sizez = int(math.ceil(abs(max_bounds[2] - min_bounds[2]) / lattice_spacing_mm) + 1)

	full_lattice_points = MC_simulator_convex.generate_cubic_lattice(
		lattice_spacing_mm,
		lattice_sizex,
		lattice_sizey,
		lattice_sizez,
		min_bounds,
	)

	return full_lattice_points, min_bounds, (lattice_sizex, lattice_sizey, lattice_sizez)


def prune_candidate_lattice_to_target_interior(
	target_zslices_list: Sequence[np.ndarray],
	full_lattice_points: np.ndarray,
	constant_z_slice_polygons_handler_option: str,
	remove_consecutive_duplicate_points_in_polygons: bool,
	kernel_type: str,
	include_edges_in_log: bool = False,
	log_sub_dirs_list: Optional[Sequence[str]] = None,
	log_file_name: Optional[str] = None,
	structure_info: Optional[dict] = None,
	create_containment_results_dataframe: bool = False,
) -> OptimizerV2CandidatePool:
	normalized_lattice_points = _validate_xyz_points_array(full_lattice_points, "full_lattice_points")
	normalized_target_zslices_list = _validate_zslices_list(target_zslices_list)

	if create_containment_results_dataframe and structure_info is None:
		raise ValueError("structure_info is required when create_containment_results_dataframe is True")

	test_struct_to_relative_struct_1d_mapping_array = np.array([0], dtype=np.int32)
	containment_result_cp_arr, prepper_output_tuple = (
		custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.custom_point_containment_mother_function(
			[normalized_target_zslices_list],
			normalized_lattice_points[np.newaxis, :, :],
			test_struct_to_relative_struct_1d_mapping_array,
			constant_z_slice_polygons_handler_option=constant_z_slice_polygons_handler_option,
			remove_consecutive_duplicate_points_in_polygons=remove_consecutive_duplicate_points_in_polygons,
			log_sub_dirs_list=list(log_sub_dirs_list or []),
			log_file_name=log_file_name,
			include_edges_in_log=include_edges_in_log,
			kernel_type=kernel_type,
		)
	)

	contained_mask = containment_result_cp_arr[0].get().astype(bool)
	contained_point_indices = np.flatnonzero(contained_mask)
	candidate_points = normalized_lattice_points[contained_mask]

	containment_results_dataframe = None
	if create_containment_results_dataframe:
		containment_results_dataframe = (
			custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.create_containment_results_dataframe_type_2I(
				structure_info,
				prepper_output_tuple[0],
				normalized_lattice_points[np.newaxis, :, :],
				containment_result_cp_arr,
				do_not_convert_column_names_to_categorical=["Pt contained bool"],
				float_dtype=np.float32,
				int_dtype=np.int32,
			)
		)

	return OptimizerV2CandidatePool(
		lattice_spacing_mm=np.nan,
		lattice_origin=np.full(3, np.nan, dtype=float),
		lattice_shape_xyz=(0, 0, 0),
		full_lattice_points=normalized_lattice_points,
		contained_mask=contained_mask,
		contained_point_indices=contained_point_indices,
		candidate_points=candidate_points,
		nearest_zslice_index_and_values_3d_arr=prepper_output_tuple[0],
		containment_results_dataframe=containment_results_dataframe,
	)


def build_target_candidate_pool(
	target_points_array: np.ndarray,
	target_zslices_list: Sequence[np.ndarray],
	search_config: OptimizerV2SearchConfig,
	constant_z_slice_polygons_handler_option: str,
	remove_consecutive_duplicate_points_in_polygons: bool,
	kernel_type: str,
	include_edges_in_log: bool = False,
	log_sub_dirs_list: Optional[Sequence[str]] = None,
	log_file_name: Optional[str] = None,
	structure_info: Optional[dict] = None,
	create_containment_results_dataframe: bool = False,
) -> OptimizerV2CandidatePool:
	full_lattice_points, lattice_origin, lattice_shape_xyz = build_target_candidate_lattice(
		target_points_array=target_points_array,
		lattice_spacing_mm=search_config.lattice_spacing_mm,
	)

	candidate_pool = prune_candidate_lattice_to_target_interior(
		target_zslices_list=target_zslices_list,
		full_lattice_points=full_lattice_points,
		constant_z_slice_polygons_handler_option=constant_z_slice_polygons_handler_option,
		remove_consecutive_duplicate_points_in_polygons=remove_consecutive_duplicate_points_in_polygons,
		kernel_type=kernel_type,
		include_edges_in_log=include_edges_in_log,
		log_sub_dirs_list=log_sub_dirs_list,
		log_file_name=log_file_name,
		structure_info=structure_info,
		create_containment_results_dataframe=create_containment_results_dataframe,
	)
	candidate_pool.lattice_spacing_mm = search_config.lattice_spacing_mm
	candidate_pool.lattice_origin = lattice_origin
	candidate_pool.lattice_shape_xyz = lattice_shape_xyz
	return candidate_pool


def visualize_target_candidate_pool(
	candidate_pool: OptimizerV2CandidatePool,
	target_points_array: np.ndarray,
	visualization_config: Optional[OptimizerV2VisualizationConfig] = None,
	additional_point_clouds: Optional[Sequence[Any]] = None,
) -> None:
	resolved_visualization_config = visualization_config or build_default_optimizer_v2_visualization_config()
	if not (
		resolved_visualization_config.plot_candidate_lattice_bool
		or resolved_visualization_config.plot_candidate_containment_bool
		or resolved_visualization_config.plot_selected_candidate_points_bool
	):
		return

	import point_containment_tools
	import plotting_funcs

	normalized_target_points = _validate_xyz_points_array(target_points_array, "target_points_array")
	additional_clouds_list = list(additional_point_clouds or [])
	target_point_cloud = point_containment_tools.create_point_cloud(normalized_target_points, np.array([0.0, 0.0, 1.0]))

	if resolved_visualization_config.plot_candidate_containment_bool:
		if candidate_pool.containment_results_dataframe is None:
			raise ValueError("candidate_pool.containment_results_dataframe is required for containment visualization")
		plotting_funcs.plot_containment_info_dataframe_to_point_cloud_plus_other_clouds(
			candidate_pool.containment_results_dataframe,
			"Test pt X",
			"Test pt Y",
			"Test pt Z",
			"Pt clr R",
			"Pt clr G",
			"Pt clr B",
			additional_point_clouds=[target_point_cloud] + additional_clouds_list,
		)

	if resolved_visualization_config.plot_candidate_lattice_bool:
		full_lattice_point_cloud = point_containment_tools.create_point_cloud(candidate_pool.full_lattice_points, np.array([0.0, 0.0, 0.0]))
		candidate_point_cloud = point_containment_tools.create_point_cloud(candidate_pool.candidate_points, np.array([0.0, 1.0, 0.0]))
		plotting_funcs.plot_geometries(full_lattice_point_cloud, candidate_point_cloud, target_point_cloud, *additional_clouds_list, label='Unknown')

	if resolved_visualization_config.plot_selected_candidate_points_bool:
		selected_candidate_indices = resolved_visualization_config.resolve_candidate_indices(candidate_pool.candidate_points.shape[0])
		if selected_candidate_indices.size == 0:
			raise ValueError("no candidate indices were selected for visualization")

		selected_candidate_points = candidate_pool.candidate_points[selected_candidate_indices]
		candidate_point_cloud = point_containment_tools.create_point_cloud(candidate_pool.candidate_points, np.array([0.0, 1.0, 0.0]))
		selected_candidate_point_cloud = point_containment_tools.create_point_cloud(selected_candidate_points, np.array([1.0, 0.0, 1.0]))
		plotting_funcs.plot_geometries(candidate_point_cloud, selected_candidate_point_cloud, target_point_cloud, *additional_clouds_list, label='Unknown')


def _validate_xyz_points_array(points_array: np.ndarray, array_name: str) -> np.ndarray:
	normalized_points_array = np.asarray(points_array, dtype=float)
	if normalized_points_array.ndim != 2 or normalized_points_array.shape[1] != 3:
		raise ValueError("{} must have shape (num_points, 3)".format(array_name))
	if normalized_points_array.shape[0] == 0:
		raise ValueError("{} cannot be empty".format(array_name))
	return normalized_points_array


def _validate_zslices_list(target_zslices_list: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
	if len(target_zslices_list) == 0:
		raise ValueError("target_zslices_list cannot be empty")

	normalized_zslices_list = []
	for zslice_index, zslice_points in enumerate(target_zslices_list):
		normalized_zslice_points = _validate_xyz_points_array(zslice_points, "target_zslices_list[{}]".format(zslice_index))
		normalized_zslices_list.append(normalized_zslice_points)

	return normalized_zslices_list


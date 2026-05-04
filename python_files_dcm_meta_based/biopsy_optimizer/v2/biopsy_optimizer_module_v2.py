from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import math
import numpy as np

import MC_simulator_convex
import custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p


@dataclass(frozen=True)
class OptimizerV2StageConfig:
	stage_name: str
	num_trials: int
	survivor_fraction: Optional[float] = None
	survivor_limit: Optional[int] = None

	def __post_init__(self) -> None:
		if self.num_trials <= 0:
			raise ValueError("num_trials must be positive")
		if self.survivor_fraction is None and self.survivor_limit is None:
			raise ValueError("at least one survivor control must be provided")
		if self.survivor_fraction is not None and not (0.0 < self.survivor_fraction <= 1.0):
			raise ValueError("survivor_fraction must be in (0, 1]")
		if self.survivor_limit is not None and self.survivor_limit <= 0:
			raise ValueError("survivor_limit must be positive")

	def resolve_survivor_count(self, num_candidates: int) -> int:
		if num_candidates <= 0:
			return 0

		resolved_counts = []
		if self.survivor_fraction is not None:
			resolved_counts.append(max(1, int(math.ceil(num_candidates * self.survivor_fraction))))
		if self.survivor_limit is not None:
			resolved_counts.append(min(num_candidates, self.survivor_limit))

		return min(resolved_counts)


DEFAULT_OPTIMIZER_V2_STAGE_CONFIGS = (
	OptimizerV2StageConfig("stage_a", 16, survivor_fraction=0.10, survivor_limit=256),
	OptimizerV2StageConfig("stage_b", 64, survivor_fraction=0.20, survivor_limit=64),
	OptimizerV2StageConfig("stage_c", 256, survivor_limit=16),
)


@dataclass(frozen=True)
class OptimizerV2SearchConfig:
	lattice_spacing_mm: float = 1.0
	stage_configs: Tuple[OptimizerV2StageConfig, ...] = DEFAULT_OPTIMIZER_V2_STAGE_CONFIGS

	def __post_init__(self) -> None:
		if self.lattice_spacing_mm <= 0.0:
			raise ValueError("lattice_spacing_mm must be positive")
		if not self.stage_configs:
			raise ValueError("stage_configs cannot be empty")

		trial_counts = [stage_config.num_trials for stage_config in self.stage_configs]
		if any(next_count <= current_count for current_count, next_count in zip(trial_counts, trial_counts[1:])):
			raise ValueError("stage trial counts must increase strictly")


@dataclass(frozen=True)
class OptimizerV2VisualizationConfig:
	plot_candidate_lattice_bool: bool = False
	plot_candidate_containment_bool: bool = False
	plot_selected_candidate_points_bool: bool = False
	candidate_indices_to_plot: Tuple[int, ...] = ()
	num_random_candidates_to_plot: int = 0
	trial_indices_to_plot: Tuple[int, ...] = ()
	num_random_trials_to_plot: int = 0
	random_seed: int = 0

	def __post_init__(self) -> None:
		if self.num_random_candidates_to_plot < 0:
			raise ValueError("num_random_candidates_to_plot cannot be negative")
		if self.num_random_trials_to_plot < 0:
			raise ValueError("num_random_trials_to_plot cannot be negative")

	def resolve_candidate_indices(self, num_candidates: int) -> np.ndarray:
		return _resolve_visualization_indices(
			total_count=num_candidates,
			explicit_indices=self.candidate_indices_to_plot,
			num_random_indices=self.num_random_candidates_to_plot,
			random_seed=self.random_seed,
		)

	def resolve_trial_indices(self, num_trials: int) -> np.ndarray:
		return _resolve_visualization_indices(
			total_count=num_trials,
			explicit_indices=self.trial_indices_to_plot,
			num_random_indices=self.num_random_trials_to_plot,
			random_seed=self.random_seed,
		)


@dataclass
class OptimizerV2CandidatePool:
	lattice_spacing_mm: float
	lattice_origin: np.ndarray
	lattice_shape_xyz: Tuple[int, int, int]
	full_lattice_points: np.ndarray
	contained_mask: np.ndarray
	contained_point_indices: np.ndarray
	candidate_points: np.ndarray
	nearest_zslice_index_and_values_3d_arr: np.ndarray
	containment_results_dataframe: Optional[Any] = None


@dataclass(frozen=True)
class OptimizerV2ChunkLayout:
	candidate_indices_global: Tuple[int, ...]
	num_trials: int
	include_nominal: bool = True
	nominal_relative_structure_index: int = 0
	trial_relative_structure_start_index: int = 1

	def __post_init__(self) -> None:
		if self.num_trials < 0:
			raise ValueError("num_trials cannot be negative")
		if self.nominal_relative_structure_index < 0:
			raise ValueError("nominal_relative_structure_index cannot be negative")
		if self.trial_relative_structure_start_index < 0:
			raise ValueError("trial_relative_structure_start_index cannot be negative")
		for candidate_index in self.candidate_indices_global:
			if candidate_index < 0:
				raise ValueError("candidate indices must be non-negative")

	@property
	def num_candidates(self) -> int:
		return len(self.candidate_indices_global)

	@property
	def num_test_structures_per_candidate(self) -> int:
		return self.num_trials + int(self.include_nominal)

	@property
	def num_test_structures(self) -> int:
		return self.num_candidates * self.num_test_structures_per_candidate

	def build_candidate_metadata_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		candidate_indices_global = np.empty(self.num_test_structures, dtype=np.int32)
		trial_indices = np.empty(self.num_test_structures, dtype=np.int32)
		is_nominal = np.zeros(self.num_test_structures, dtype=bool)

		write_index = 0
		for candidate_index_global in self.candidate_indices_global:
			if self.include_nominal:
				candidate_indices_global[write_index] = candidate_index_global
				trial_indices[write_index] = -1
				is_nominal[write_index] = True
				write_index += 1

			for trial_index in range(self.num_trials):
				candidate_indices_global[write_index] = candidate_index_global
				trial_indices[write_index] = trial_index
				write_index += 1

		return candidate_indices_global, trial_indices, is_nominal

	def build_test_struct_to_relative_struct_mapping(self) -> np.ndarray:
		test_struct_to_relative_struct = np.empty(self.num_test_structures, dtype=np.int32)
		write_index = 0
		for _ in self.candidate_indices_global:
			if self.include_nominal:
				test_struct_to_relative_struct[write_index] = self.nominal_relative_structure_index
				write_index += 1

			for trial_index in range(self.num_trials):
				test_struct_to_relative_struct[write_index] = self.trial_relative_structure_start_index + trial_index
				write_index += 1

		return test_struct_to_relative_struct

	def build_metadata_dataframe(self):
		import pandas

		candidate_indices_global, trial_indices, is_nominal = self.build_candidate_metadata_arrays()
		return pandas.DataFrame(
			{
				"Test struct input index": np.arange(self.num_test_structures, dtype=np.int32),
				"Candidate global index": candidate_indices_global,
				"Trial index": trial_indices,
				"Is nominal": is_nominal,
			}
		)


def build_default_optimizer_v2_search_config() -> OptimizerV2SearchConfig:
	return OptimizerV2SearchConfig()


def build_optimizer_v2_search_config_with_trial_counts(
	stage_trial_counts: Sequence[int],
	lattice_spacing_mm: float = 1.0,
	template_stage_configs: Sequence[OptimizerV2StageConfig] = DEFAULT_OPTIMIZER_V2_STAGE_CONFIGS,
) -> OptimizerV2SearchConfig:
	if len(stage_trial_counts) != len(template_stage_configs):
		raise ValueError("stage_trial_counts must match the number of template stage configs")

	stage_configs = tuple(
		OptimizerV2StageConfig(
			template_stage_config.stage_name,
			int(stage_trial_count),
			survivor_fraction=template_stage_config.survivor_fraction,
			survivor_limit=template_stage_config.survivor_limit,
		)
		for template_stage_config, stage_trial_count in zip(template_stage_configs, stage_trial_counts)
	)

	return OptimizerV2SearchConfig(
		lattice_spacing_mm=lattice_spacing_mm,
		stage_configs=stage_configs,
	)


def build_default_optimizer_v2_visualization_config() -> OptimizerV2VisualizationConfig:
	return OptimizerV2VisualizationConfig()


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


def _resolve_visualization_indices(
	total_count: int,
	explicit_indices: Sequence[int],
	num_random_indices: int,
	random_seed: int,
) -> np.ndarray:
	if total_count < 0:
		raise ValueError("total_count cannot be negative")

	resolved_indices = []
	for explicit_index in explicit_indices:
		if explicit_index < 0 or explicit_index >= total_count:
			raise ValueError("visualization index {} is out of range for total_count {}".format(explicit_index, total_count))
		resolved_indices.append(int(explicit_index))

	if num_random_indices > 0 and total_count > 0:
		remaining_indices = np.setdiff1d(np.arange(total_count, dtype=np.int32), np.array(resolved_indices, dtype=np.int32), assume_unique=False)
		if remaining_indices.size > 0:
			random_generator = np.random.default_rng(random_seed)
			num_random_indices_resolved = min(num_random_indices, remaining_indices.size)
			resolved_indices.extend(random_generator.choice(remaining_indices, size=num_random_indices_resolved, replace=False).tolist())

	if not resolved_indices:
		return np.empty(0, dtype=np.int32)

	return np.array(sorted(set(resolved_indices)), dtype=np.int32)

"""Patient-level MC dose output contracts and localization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from legacy_data_keys import legacy_data_keys

from .contracts import MCDoseSimulationConfig
from .legacy_keys import legacy_mc_keys

MC_DOSE_BIOPSY_OUTPUT_KEYS = legacy_mc_keys.biopsy_outputs.dose_output_keys
MC_DOSE_LOCALIZATION_KIND_DOSE = "dose"
MC_DOSE_LOCALIZATION_KIND_GRADIENT = "gradient"
MC_DOSE_VALUE_COLUMN = "Dose val (interpolated)"
MC_DOSE_GRADIENT_VALUE_COLUMN = "Dose grad val (interpolated)"
MC_DOSE_ORIGINAL_POINT_INDEX_COLUMN = "Original pt index"
MC_DOSE_TRIAL_COLUMN = "Trial num"
MC_DOSE_DVH_HISTOGRAM_QUANTILE_PERCENTILES = tuple(range(5, 100, 5))


@dataclass(slots=True)
class PatientDoseLatticeContext:
    """Dose-grid values and nearest-neighbour index for one localization kind."""

    patient_uid: str
    localization_kind: str
    dose_reference_dict: Mapping[str, Any]
    source_dose_and_gradient_array: Any
    localization_map_array: Any
    localization_map_flattened: Any
    physical_coordinates: Any
    sampled_values: Any
    kdtree: Any
    result_column: str
    output_key: str
    kdtree_key: str
    lattice_point_cloud: Any = None
    thresholded_lattice_point_cloud: Any = None


@dataclass(slots=True)
class PatientDoseBiopsyContext:
    """Patient-local biopsy inputs for dose or dose-gradient localization."""

    patient_uid: str
    biopsy_index: int
    num_sample_points: int
    roi: Any
    ref_number: Any
    simulated_bool: Any
    simulated_type: Any
    unshifted_sampled_points: Any
    sampled_points_bx_coord_sys: Any
    bx_only_shifted_points: Any
    bx_only_shifted_points_cutoff: Any
    nominal_and_shifted_points: Any
    stacked_nominal_and_shifted_points: Any
    biopsy_structure_info: dict[str, Any]


@dataclass(slots=True)
class PatientDoseLocalizationOutputs:
    """Nearest-neighbour localization outputs for one biopsy and localization kind."""

    localization_kind: str
    result_column: str
    output_key: str
    nearest_neighbour_dataframe: Any
    values_by_point_nominal_and_trials: Any

    def legacy_biopsy_updates(self) -> dict[str, Any]:
        return {self.output_key: self.values_by_point_nominal_and_trials}


@dataclass(slots=True)
class PatientDoseDVHOutputs:
    """DVH outputs compiled from one biopsy's dose localization array."""

    differential_dvh_dict: dict[str, Any]
    cumulative_dvh_dict: dict[str, Any]
    dose_volume_metrics_dict: dict[str, Any]

    def legacy_biopsy_updates(self) -> dict[str, Any]:
        output_keys = legacy_mc_keys.biopsy_outputs
        return {
            output_keys.differential_dvh_dict_key: self.differential_dvh_dict,
            output_keys.cumulative_dvh_dict_key: self.cumulative_dvh_dict,
            output_keys.dose_volume_metrics_dict_key: self.dose_volume_metrics_dict,
        }


def _as_numpy_array(array_like: Any) -> Any:
    import numpy as np

    try:
        import cupy as cp
    except ImportError:
        return np.asarray(array_like)
    return cp.asnumpy(array_like)


def _normalize_dose_localization_kind(localization_kind: str) -> str:
    normalized_kind = str(localization_kind).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized_kind in {"dose", "dosimetry"}:
        return MC_DOSE_LOCALIZATION_KIND_DOSE
    if normalized_kind in {"gradient", "dose_gradient", "dosegrad"}:
        return MC_DOSE_LOCALIZATION_KIND_GRADIENT
    raise ValueError(f"Unsupported dose localization kind: {localization_kind!r}")


def _build_legacy_biopsy_structure_info(biopsy_structure: Mapping[str, Any]) -> dict[str, Any]:
    structure_record_keys = legacy_data_keys.structure_record
    structure_metadata_keys = legacy_data_keys.structure_metadata
    return {
        structure_metadata_keys.structure_id_key: biopsy_structure[structure_record_keys.roi_key],
        structure_metadata_keys.struct_ref_type_key: biopsy_structure[structure_record_keys.struct_type_key],
        structure_metadata_keys.dicom_ref_number_key: biopsy_structure[structure_record_keys.ref_number_key],
        structure_record_keys.index_number_key: biopsy_structure[structure_record_keys.index_number_key],
        structure_record_keys.simulated_bool_key: biopsy_structure.get(structure_record_keys.simulated_bool_key),
        structure_record_keys.simulated_type_key: biopsy_structure.get(structure_record_keys.simulated_type_key),
    }


def patient_has_dose_reference(patient_reference_dict: Mapping[str, Any], *, dose_ref: str) -> bool:
    """Return whether one legacy patient dictionary contains dose-grid data."""
    return str(dose_ref) in patient_reference_dict


def build_patient_dose_lattice_context(patient_uid: str,
                                       patient_reference_dict: Mapping[str, Any],
                                       *,
                                       dose_ref: str,
                                       localization_kind: str = MC_DOSE_LOCALIZATION_KIND_DOSE,
                                       mutate_reference: bool = True) -> PatientDoseLatticeContext:
    """Build the dose or dose-gradient KD-tree context used by the oracle."""
    import numpy as np
    import scipy

    normalized_kind = _normalize_dose_localization_kind(localization_kind)
    dose_reference_keys = legacy_mc_keys.dose_reference
    output_keys = legacy_mc_keys.biopsy_outputs
    dose_reference_dict = patient_reference_dict[str(dose_ref)]
    source_array = dose_reference_dict[dose_reference_keys.dose_and_gradient_map_array_key]

    if normalized_kind == MC_DOSE_LOCALIZATION_KIND_DOSE:
        localization_map_array = source_array[:, :, :7]
        result_column = MC_DOSE_VALUE_COLUMN
        output_key = output_keys.dose_values_nominal_and_trials_array_key
        kdtree_key = dose_reference_keys.dose_kdtree_key
        lattice_point_cloud_key = dose_reference_keys.dose_grid_point_cloud_key
        thresholded_lattice_point_cloud_key = dose_reference_keys.dose_grid_point_cloud_thresholded_key
    else:
        localization_map_array = np.delete(source_array, np.r_[6:10, 11:14], axis=2)
        result_column = MC_DOSE_GRADIENT_VALUE_COLUMN
        output_key = output_keys.dose_gradient_values_nominal_and_trials_array_key
        kdtree_key = dose_reference_keys.dose_gradient_kdtree_key
        lattice_point_cloud_key = dose_reference_keys.dose_grid_gradient_point_cloud_key
        thresholded_lattice_point_cloud_key = dose_reference_keys.dose_grid_gradient_point_cloud_thresholded_key

    localization_map_flattened = np.reshape(localization_map_array, (-1, 7), order="C")
    physical_coordinates = localization_map_flattened[:, 3:6]
    sampled_values = localization_map_flattened[:, 6]
    kdtree = scipy.spatial.KDTree(physical_coordinates)
    if mutate_reference:
        dose_reference_dict[kdtree_key] = kdtree

    return PatientDoseLatticeContext(
        patient_uid=str(patient_uid),
        localization_kind=normalized_kind,
        dose_reference_dict=dose_reference_dict,
        source_dose_and_gradient_array=source_array,
        localization_map_array=localization_map_array,
        localization_map_flattened=localization_map_flattened,
        physical_coordinates=physical_coordinates,
        sampled_values=sampled_values,
        kdtree=kdtree,
        result_column=result_column,
        output_key=output_key,
        kdtree_key=kdtree_key,
        lattice_point_cloud=dose_reference_dict.get(lattice_point_cloud_key),
        thresholded_lattice_point_cloud=dose_reference_dict.get(thresholded_lattice_point_cloud_key),
    )


def resolve_patient_target_prescription_dose(patient_reference_dict: Mapping[str, Any], *, plan_ref: str) -> Any:
    """Read the target prescription dose used by the legacy DVH block."""
    dose_reference_keys = legacy_mc_keys.dose_reference
    prescription_doses = patient_reference_dict[str(plan_ref)][dose_reference_keys.prescription_doses_dict_key]
    return prescription_doses[dose_reference_keys.target_prescription_key]


def build_patient_dose_biopsy_context(patient_uid: str,
                                      biopsy_index: int,
                                      biopsy_structure: Mapping[str, Any],
                                      *,
                                      num_mc_dose_simulations: int) -> PatientDoseBiopsyContext:
    """Build the nominal-plus-shifted biopsy point arrays consumed by dose localization."""
    import numpy as np

    identity_keys = legacy_mc_keys.biopsy_identity
    biopsy_runtime_keys = legacy_data_keys.biopsy_runtime
    intermediate_keys = legacy_mc_keys.containment_intermediates
    bx_only_shifted_points = _as_numpy_array(biopsy_structure[intermediate_keys.bx_only_shifted_points_array_key])
    bx_only_shifted_points_cutoff = bx_only_shifted_points[0:int(num_mc_dose_simulations)]
    unshifted_sampled_points = biopsy_structure[biopsy_runtime_keys.random_uniformly_sampled_volume_points_array_key]
    unshifted_sampled_points_3d = np.expand_dims(unshifted_sampled_points, axis=0)
    nominal_and_shifted_points = np.concatenate((unshifted_sampled_points_3d, bx_only_shifted_points_cutoff))
    stacked_nominal_and_shifted_points = np.reshape(nominal_and_shifted_points, (-1, 3), order="C")
    return PatientDoseBiopsyContext(
        patient_uid=str(patient_uid),
        biopsy_index=int(biopsy_index),
        num_sample_points=int(biopsy_structure[biopsy_runtime_keys.num_sampled_bx_points_key]),
        roi=biopsy_structure[identity_keys.roi_key],
        ref_number=biopsy_structure[identity_keys.ref_number_key],
        simulated_bool=biopsy_structure[identity_keys.simulated_bool_key],
        simulated_type=biopsy_structure[identity_keys.simulated_type_key],
        unshifted_sampled_points=unshifted_sampled_points,
        sampled_points_bx_coord_sys=biopsy_structure[
            biopsy_runtime_keys.random_uniformly_sampled_volume_points_bx_coord_sys_array_key
        ],
        bx_only_shifted_points=bx_only_shifted_points,
        bx_only_shifted_points_cutoff=bx_only_shifted_points_cutoff,
        nominal_and_shifted_points=nominal_and_shifted_points,
        stacked_nominal_and_shifted_points=stacked_nominal_and_shifted_points,
        biopsy_structure_info=_build_legacy_biopsy_structure_info(biopsy_structure),
    )


def run_patient_dose_nearest_neighbour_localization(
    biopsy_context: PatientDoseBiopsyContext,
    lattice_context: PatientDoseLatticeContext,
    *,
    dose_config: MCDoseSimulationConfig,
    num_mc_dose_simulations: int,
) -> Any:
    """Run the existing dosimetric localizer for one biopsy and dose-grid context."""
    import dosimetric_localizer

    return dosimetric_localizer.dosimetric_localization_dataframe_version(
        biopsy_context.stacked_nominal_and_shifted_points,
        biopsy_context.patient_uid,
        biopsy_context.biopsy_structure_info,
        lattice_context.kdtree,
        lattice_context.sampled_values,
        dose_config.num_dose_calc_NN,
        int(num_mc_dose_simulations),
        biopsy_context.num_sample_points,
        dose_config.idw_power,
        result_col_name=lattice_context.result_column,
    )


def compile_patient_dose_localization_array(localization_dataframe: Any, *, result_column: str) -> Any:
    """Pivot localizer rows to the legacy point-by-trial array shape."""
    pivoted_dataframe = localization_dataframe.pivot(
        index=MC_DOSE_ORIGINAL_POINT_INDEX_COLUMN,
        columns=MC_DOSE_TRIAL_COLUMN,
        values=result_column,
    )
    pivoted_dataframe = pivoted_dataframe.sort_index(axis=0).sort_index(axis=1)
    return pivoted_dataframe.to_numpy()


def run_patient_dose_localization_for_biopsy(
    biopsy_context: PatientDoseBiopsyContext,
    lattice_context: PatientDoseLatticeContext,
    *,
    dose_config: MCDoseSimulationConfig,
    num_mc_dose_simulations: int,
) -> PatientDoseLocalizationOutputs:
    """Run and compile one biopsy's dose or dose-gradient localization array."""
    nearest_neighbour_dataframe = run_patient_dose_nearest_neighbour_localization(
        biopsy_context,
        lattice_context,
        dose_config=dose_config,
        num_mc_dose_simulations=int(num_mc_dose_simulations),
    )
    localization_array = compile_patient_dose_localization_array(
        nearest_neighbour_dataframe,
        result_column=lattice_context.result_column,
    )
    return PatientDoseLocalizationOutputs(
        localization_kind=lattice_context.localization_kind,
        result_column=lattice_context.result_column,
        output_key=lattice_context.output_key,
        nearest_neighbour_dataframe=nearest_neighbour_dataframe,
        values_by_point_nominal_and_trials=localization_array,
    )


def write_patient_dose_localization_outputs_to_legacy_record(
    biopsy_structure: dict[str, Any],
    localization_outputs: PatientDoseLocalizationOutputs,
) -> dict[str, Any]:
    """Write one localization result array to a legacy biopsy dictionary."""
    biopsy_structure.update(localization_outputs.legacy_biopsy_updates())
    return biopsy_structure


def calculate_patient_dose_dvh_outputs(
    dose_values_by_point_nominal_and_trials: Any,
    *,
    dose_config: MCDoseSimulationConfig,
    ctv_dose: float,
    bx_sample_pts_volume_element: float,
) -> PatientDoseDVHOutputs:
    """Compile differential DVH, cumulative DVH, and V-percent metrics for one biopsy."""
    import numpy as np

    dose_values_array = np.asarray(dose_values_by_point_nominal_and_trials)
    if dose_values_array.ndim != 2:
        raise ValueError("dose_values_by_point_nominal_and_trials must be a 2D point-by-trial array")

    num_sampled_bx_points = dose_values_array.shape[0]
    num_nominal_and_all_dose_trials = dose_values_array.shape[1]
    differential_dvh_resolution = int(dose_config.differential_dvh_resolution)
    differential_dvh_range = (0, np.amax(dose_values_array))
    differential_dvh_histogram_counts_by_trial = np.empty(
        [num_nominal_and_all_dose_trials, differential_dvh_resolution]
    )
    differential_dvh_histogram_edges_by_trial = np.empty(
        [num_nominal_and_all_dose_trials, differential_dvh_resolution + 1]
    )

    for trial_index in range(num_nominal_and_all_dose_trials):
        dose_values_for_trial = dose_values_array[:, trial_index]
        histogram_counts, histogram_edges = np.histogram(
            dose_values_for_trial,
            bins=differential_dvh_resolution,
            range=differential_dvh_range,
        )
        differential_dvh_histogram_counts_by_trial[trial_index, :] = histogram_counts
        differential_dvh_histogram_edges_by_trial[trial_index, :] = histogram_edges

    dose_volume_metrics_dict: dict[str, Any] = {}
    for vol_dose_percent in dose_config.v_percent_DVH_to_calc_list:
        metric_key = str(vol_dose_percent)
        dose_threshold = (vol_dose_percent / 100) * ctv_dose
        truth_matrix_for_vol_dose_percent = dose_values_array > dose_threshold
        counts_for_vol_dose_percent = np.sum(truth_matrix_for_vol_dose_percent, axis=0)
        percent_for_vol_dose_percent = (counts_for_vol_dose_percent / num_sampled_bx_points) * 100
        dvh_metric_all_trials_arr = np.array(percent_for_vol_dose_percent[1:].tolist())
        dose_volume_metrics_dict[metric_key] = {
            "Nominal": percent_for_vol_dose_percent[0],
            "All MC trials list": percent_for_vol_dose_percent[1:].tolist(),
            "Mean": np.mean(dvh_metric_all_trials_arr),
            "STD": np.std(dvh_metric_all_trials_arr),
            "Quantiles": {
                "Q" + str(q): np.quantile(dvh_metric_all_trials_arr, q / 100)
                for q in dose_config.volume_DVH_quantiles_to_calculate
            },
        }

    differential_dvh_histogram_volume_by_trial = (
        differential_dvh_histogram_counts_by_trial * bx_sample_pts_volume_element
    )
    differential_dvh_histogram_percent_by_trial = (
        differential_dvh_histogram_counts_by_trial / num_sampled_bx_points
    ) * 100
    differential_dvh_dict = {
        "Counts arr": differential_dvh_histogram_counts_by_trial,
        "Percent arr": differential_dvh_histogram_percent_by_trial,
        "Volume arr (cubic mm)": differential_dvh_histogram_volume_by_trial,
        "Dose bins (edges) arr (Gy)": differential_dvh_histogram_edges_by_trial,
        "Quantiles counts dict": {
            "Q" + str(q): np.quantile(differential_dvh_histogram_counts_by_trial[1:], q / 100, axis=0)
            for q in MC_DOSE_DVH_HISTOGRAM_QUANTILE_PERCENTILES
        },
        "Quantiles percent dict": {
            "Q" + str(q): np.quantile(differential_dvh_histogram_percent_by_trial[1:], q / 100, axis=0)
            for q in MC_DOSE_DVH_HISTOGRAM_QUANTILE_PERCENTILES
        },
        "Quantiles volume dict": {
            "Q" + str(q): np.quantile(differential_dvh_histogram_volume_by_trial[1:], q / 100, axis=0)
            for q in MC_DOSE_DVH_HISTOGRAM_QUANTILE_PERCENTILES
        },
    }

    cumulative_dvh_counts_d0 = np.sum(differential_dvh_histogram_counts_by_trial, axis=1, keepdims=True)
    cumulative_dvh_counts_above_dose = num_sampled_bx_points - np.cumsum(
        differential_dvh_histogram_counts_by_trial,
        axis=1,
    )
    cumulative_dvh_counts_by_trial = np.concatenate((cumulative_dvh_counts_d0, cumulative_dvh_counts_above_dose), axis=1)
    cumulative_dvh_volume_by_trial = cumulative_dvh_counts_by_trial * bx_sample_pts_volume_element
    cumulative_dvh_percent_by_trial = (cumulative_dvh_counts_by_trial / num_sampled_bx_points) * 100
    cumulative_dvh_dose_values = differential_dvh_histogram_edges_by_trial[0].copy()
    cumulative_dvh_dict = {
        "Counts arr": cumulative_dvh_counts_by_trial,
        "Percent arr": cumulative_dvh_percent_by_trial,
        "Volume arr (cubic mm)": cumulative_dvh_volume_by_trial,
        "Dose vals arr (Gy)": cumulative_dvh_dose_values,
        "Quantiles counts dict": {
            "Q" + str(q): np.quantile(cumulative_dvh_counts_by_trial[1:], q / 100, axis=0)
            for q in MC_DOSE_DVH_HISTOGRAM_QUANTILE_PERCENTILES
        },
        "Quantiles percent dict": {
            "Q" + str(q): np.quantile(cumulative_dvh_percent_by_trial[1:], q / 100, axis=0)
            for q in MC_DOSE_DVH_HISTOGRAM_QUANTILE_PERCENTILES
        },
        "Quantiles volume dict": {
            "Q" + str(q): np.quantile(cumulative_dvh_volume_by_trial[1:], q / 100, axis=0)
            for q in MC_DOSE_DVH_HISTOGRAM_QUANTILE_PERCENTILES
        },
    }

    return PatientDoseDVHOutputs(
        differential_dvh_dict=differential_dvh_dict,
        cumulative_dvh_dict=cumulative_dvh_dict,
        dose_volume_metrics_dict=dose_volume_metrics_dict,
    )


def compile_patient_dose_dvh_outputs_for_biopsy(
    biopsy_structure: Mapping[str, Any],
    *,
    dose_config: MCDoseSimulationConfig,
    ctv_dose: float,
    bx_sample_pts_volume_element: float,
    dose_values_by_point_nominal_and_trials: Any | None = None,
) -> PatientDoseDVHOutputs:
    """Compile DVH outputs using an explicit array or the legacy dose array on the biopsy."""
    output_keys = legacy_mc_keys.biopsy_outputs
    resolved_dose_values = (
        dose_values_by_point_nominal_and_trials
        if dose_values_by_point_nominal_and_trials is not None
        else biopsy_structure[output_keys.dose_values_nominal_and_trials_array_key]
    )
    return calculate_patient_dose_dvh_outputs(
        resolved_dose_values,
        dose_config=dose_config,
        ctv_dose=float(ctv_dose),
        bx_sample_pts_volume_element=float(bx_sample_pts_volume_element),
    )


def write_patient_dose_dvh_outputs_to_legacy_record(
    biopsy_structure: dict[str, Any],
    dvh_outputs: PatientDoseDVHOutputs,
) -> dict[str, Any]:
    """Write one biopsy's DVH outputs to a legacy biopsy dictionary."""
    biopsy_structure.update(dvh_outputs.legacy_biopsy_updates())
    return biopsy_structure


@dataclass(slots=True)
class PatientDoseOutputs:
    """Dose outputs collected from one patient's biopsy records."""

    patient_uid: str
    biopsy_outputs: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "patient_uid": self.patient_uid,
            "biopsy_outputs": self.biopsy_outputs,
        }


def collect_patient_dose_outputs(patient_uid: str,
                                 patient_reference_dict: Mapping[str, Any],
                                 *,
                                 bx_ref: str) -> PatientDoseOutputs:
    """Collect dose artifacts written into one patient dictionary."""
    identity_keys = legacy_mc_keys.biopsy_identity
    biopsy_outputs: list[dict[str, Any]] = []
    for biopsy_index, biopsy_structure in enumerate(patient_reference_dict.get(bx_ref, ())):
        outputs = {
            output_key: biopsy_structure.get(output_key)
            for output_key in MC_DOSE_BIOPSY_OUTPUT_KEYS
            if output_key in biopsy_structure
        }
        if outputs:
            biopsy_outputs.append(
                {
                    identity_keys.roi_key: biopsy_structure.get(identity_keys.roi_key),
                    identity_keys.ref_number_key: biopsy_structure.get(identity_keys.ref_number_key),
                    identity_keys.index_number_key: biopsy_structure.get(identity_keys.index_number_key, biopsy_index),
                    identity_keys.simulated_bool_key: biopsy_structure.get(identity_keys.simulated_bool_key),
                    identity_keys.simulated_type_key: biopsy_structure.get(identity_keys.simulated_type_key),
                    "outputs": outputs,
                }
            )
    return PatientDoseOutputs(
        patient_uid=str(patient_uid),
        biopsy_outputs=biopsy_outputs,
    )

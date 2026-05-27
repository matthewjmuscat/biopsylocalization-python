"""Patient-level MC MR ADC localization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
from typing import Any, Mapping

from legacy_data_keys import legacy_data_keys

from .contracts import MCMRSimulationConfig
from .legacy_keys import legacy_mc_keys

MC_MR_BIOPSY_OUTPUT_KEYS = legacy_mc_keys.biopsy_outputs.mr_output_keys
MC_MR_VALUE_COLUMN = "MR val (interpolated)"
MC_MR_ORIGINAL_POINT_INDEX_COLUMN = "Original pt index"
MC_MR_TRIAL_COLUMN = "Trial num"


@dataclass(slots=True)
class PatientMRLatticeContext:
    """Filtered MR ADC lattice values and nearest-neighbour index for one patient."""

    patient_uid: str
    mr_adc_reference_dict: Mapping[str, Any]
    filtered_non_negative_adc_mr_phys_space_array: Any
    physical_coordinates: Any
    sampled_values: Any
    kdtree: Any
    result_column: str
    output_key: str
    kdtree_key: str
    lattice_point_cloud: Any = None
    thresholded_lattice_point_cloud: Any = None


@dataclass(slots=True)
class PatientMRBiopsyContext:
    """Patient-local biopsy inputs for MR ADC localization."""

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
class PatientMRLocalizationOutputs:
    """Nearest-neighbour MR ADC localization outputs for one biopsy."""

    result_column: str
    output_key: str
    nearest_neighbour_dataframe: Any
    values_by_point_nominal_and_trials: Any

    def legacy_biopsy_updates(self) -> dict[str, Any]:
        return {self.output_key: self.values_by_point_nominal_and_trials}


@dataclass(slots=True)
class PatientMROutputs:
    """MR ADC outputs collected from one patient's biopsy records."""

    patient_uid: str
    biopsy_outputs: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "patient_uid": self.patient_uid,
            "biopsy_outputs": self.biopsy_outputs,
        }


@dataclass(slots=True)
class PatientMRStageResult:
    """Output bundle from running patient-local MR ADC localization."""

    patient_uid: str
    patient_reference_dict: dict[str, Any]
    master_structure_reference_dict: dict[str, dict[str, Any]]
    master_structure_info_dict: dict[str, Any]
    mr_outputs: PatientMROutputs
    mr_reference_available: bool
    biopsy_count: int
    performed_flags: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.patient_uid = str(self.patient_uid)
        self.biopsy_count = int(self.biopsy_count)
        self.mr_reference_available = bool(self.mr_reference_available)
        self.performed_flags = dict(self.performed_flags or {})
        self.metadata = dict(self.metadata or {})


def _as_numpy_array(array_like: Any) -> Any:
    import numpy as np

    try:
        import cupy as cp
    except ImportError:
        return np.asarray(array_like)
    return cp.asnumpy(array_like)


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


def patient_has_mr_adc_reference(patient_reference_dict: Mapping[str, Any], *, mr_adc_ref: str) -> bool:
    """Return whether one legacy patient dictionary contains MR ADC data."""
    return str(mr_adc_ref) in patient_reference_dict


def build_patient_mr_lattice_context(patient_uid: str,
                                     patient_reference_dict: Mapping[str, Any],
                                     *,
                                     mr_adc_ref: str,
                                     filter_out_negatives: bool = True,
                                     mutate_reference: bool = True) -> PatientMRLatticeContext:
    """Build the filtered MR ADC KD-tree context used by the oracle."""
    import scipy
    import lattice_reconstruction_tools

    mr_reference_keys = legacy_mc_keys.mr_reference
    output_keys = legacy_mc_keys.biopsy_outputs
    mr_adc_reference_dict = patient_reference_dict[str(mr_adc_ref)]
    filtered_non_negative_adc_mr_phys_space_array = (
        lattice_reconstruction_tools.reconstruct_mr_lattice_with_coordinates_from_dict_v2(
            mr_adc_reference_dict,
            filter_out_negatives=filter_out_negatives,
        )
    )
    mr_adc_phys_coords_array = filtered_non_negative_adc_mr_phys_space_array[:, 0:3]
    mr_adc_values_array = filtered_non_negative_adc_mr_phys_space_array[:, 3]
    mr_adc_kdtree = scipy.spatial.KDTree(mr_adc_phys_coords_array)
    if mutate_reference:
        mr_adc_reference_dict[mr_reference_keys.mr_adc_kdtree_key] = mr_adc_kdtree

    return PatientMRLatticeContext(
        patient_uid=str(patient_uid),
        mr_adc_reference_dict=mr_adc_reference_dict,
        filtered_non_negative_adc_mr_phys_space_array=filtered_non_negative_adc_mr_phys_space_array,
        physical_coordinates=mr_adc_phys_coords_array,
        sampled_values=mr_adc_values_array,
        kdtree=mr_adc_kdtree,
        result_column=MC_MR_VALUE_COLUMN,
        output_key=output_keys.mr_adc_values_nominal_and_trials_array_key,
        kdtree_key=mr_reference_keys.mr_adc_kdtree_key,
        lattice_point_cloud=mr_adc_reference_dict.get(mr_reference_keys.mr_adc_grid_point_cloud_key),
        thresholded_lattice_point_cloud=mr_adc_reference_dict.get(
            mr_reference_keys.mr_adc_grid_point_cloud_thresholded_key
        ),
    )


def build_patient_mr_biopsy_context(patient_uid: str,
                                    biopsy_index: int,
                                    biopsy_structure: Mapping[str, Any],
                                    *,
                                    num_mc_mr_simulations: int) -> PatientMRBiopsyContext:
    """Build the nominal-plus-shifted biopsy point arrays consumed by MR localization."""
    import numpy as np

    identity_keys = legacy_mc_keys.biopsy_identity
    biopsy_runtime_keys = legacy_data_keys.biopsy_runtime
    intermediate_keys = legacy_mc_keys.containment_intermediates
    bx_only_shifted_points = _as_numpy_array(biopsy_structure[intermediate_keys.bx_only_shifted_points_array_key])
    bx_only_shifted_points_cutoff = bx_only_shifted_points[0:int(num_mc_mr_simulations)]
    unshifted_sampled_points = biopsy_structure[biopsy_runtime_keys.random_uniformly_sampled_volume_points_array_key]
    unshifted_sampled_points_3d = np.expand_dims(unshifted_sampled_points, axis=0)
    nominal_and_shifted_points = np.concatenate((unshifted_sampled_points_3d, bx_only_shifted_points_cutoff))
    stacked_nominal_and_shifted_points = np.reshape(nominal_and_shifted_points, (-1, 3), order="C")
    return PatientMRBiopsyContext(
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


def run_patient_mr_nearest_neighbour_localization(
    biopsy_context: PatientMRBiopsyContext,
    lattice_context: PatientMRLatticeContext,
    *,
    mr_config: MCMRSimulationConfig,
    num_mc_mr_simulations: int,
) -> Any:
    """Run the existing MR ADC localizer for one biopsy."""
    import mr_localizers

    return mr_localizers.mr_localization_dataframe_version(
        biopsy_context.stacked_nominal_and_shifted_points,
        biopsy_context.patient_uid,
        biopsy_context.biopsy_structure_info,
        lattice_context.kdtree,
        lattice_context.sampled_values,
        mr_config.num_mr_calc_NN,
        int(num_mc_mr_simulations),
        biopsy_context.num_sample_points,
        mr_config.idw_power,
    )


def compile_patient_mr_localization_array(localization_dataframe: Any) -> Any:
    """Pivot MR localizer rows to the legacy point-by-trial array shape."""
    pivoted_dataframe = localization_dataframe.pivot(
        index=MC_MR_ORIGINAL_POINT_INDEX_COLUMN,
        columns=MC_MR_TRIAL_COLUMN,
        values=MC_MR_VALUE_COLUMN,
    )
    pivoted_dataframe = pivoted_dataframe.sort_index(axis=0).sort_index(axis=1)
    return pivoted_dataframe.to_numpy()


def run_patient_mr_localization_for_biopsy(
    biopsy_context: PatientMRBiopsyContext,
    lattice_context: PatientMRLatticeContext,
    *,
    mr_config: MCMRSimulationConfig,
    num_mc_mr_simulations: int,
) -> PatientMRLocalizationOutputs:
    """Run and compile one biopsy's MR ADC localization array."""
    nearest_neighbour_dataframe = run_patient_mr_nearest_neighbour_localization(
        biopsy_context,
        lattice_context,
        mr_config=mr_config,
        num_mc_mr_simulations=int(num_mc_mr_simulations),
    )
    localization_array = compile_patient_mr_localization_array(nearest_neighbour_dataframe)
    return PatientMRLocalizationOutputs(
        result_column=lattice_context.result_column,
        output_key=lattice_context.output_key,
        nearest_neighbour_dataframe=nearest_neighbour_dataframe,
        values_by_point_nominal_and_trials=localization_array,
    )


def write_patient_mr_localization_outputs_to_legacy_record(
    biopsy_structure: dict[str, Any],
    localization_outputs: PatientMRLocalizationOutputs,
) -> dict[str, Any]:
    """Write one MR ADC localization result array to a legacy biopsy dictionary."""
    biopsy_structure.update(localization_outputs.legacy_biopsy_updates())
    return biopsy_structure


def _reject_patient_mr_stage_side_effect_options(mr_config: MCMRSimulationConfig) -> None:
    unsupported_options = []
    if mr_config.raw_data_mc_mr_dump_bool:
        unsupported_options.append("raw_data_mc_mr_dump_bool")
    if mr_config.show_NN_mr_adc_demonstration_plots:
        unsupported_options.append("show_NN_mr_adc_demonstration_plots")
    if mr_config.show_NN_mr_adc_demonstration_plots_all_trials_at_once:
        unsupported_options.append("show_NN_mr_adc_demonstration_plots_all_trials_at_once")
    if unsupported_options:
        raise ValueError(
            "Patient MR ADC localization stage does not perform raw CSV dumps or plotting; "
            "use the MR legacy adapter for those validation/debug surfaces. Unsupported options: "
            + ", ".join(unsupported_options)
        )


def _resolve_num_mc_mr_simulations(master_structure_info_dict: Mapping[str, Any]) -> int:
    master_keys = legacy_mc_keys.master_info
    try:
        num_mc_mr_simulations = master_structure_info_dict[master_keys.global_key][
            master_keys.mc_info_key
        ][master_keys.num_mr_simulations_key]
    except KeyError as exc:
        raise KeyError(
            "master_structure_info_dict['Global']['MC info']['Num MC MR simulations'] "
            "is required for patient MR ADC localization"
        ) from exc
    return int(num_mc_mr_simulations)


def _set_patient_mr_performed_flag(master_structure_info_dict: dict[str, Any], performed_flag: Any) -> dict[str, Any]:
    master_keys = legacy_mc_keys.master_info
    global_info = master_structure_info_dict.setdefault(master_keys.global_key, {})
    mc_info = global_info.setdefault(master_keys.mc_info_key, {})
    mc_info[master_keys.mr_performed_key] = bool(performed_flag)
    return {master_keys.mr_performed_key: mc_info[master_keys.mr_performed_key]}


def run_patient_mr_adc_localization_stage(
    *,
    patient_uid: str,
    patient_reference_dict: dict[str, Any],
    patient_info_dict: Mapping[str, Any] | None,
    bx_ref: str,
    mr_adc_ref: str,
    config: MCMRSimulationConfig,
    global_info: Mapping[str, Any] | None = None,
    mutate_input: bool = True,
) -> PatientMRStageResult:
    """Run MR ADC localization for one patient without oracle UI or file-writing side effects."""
    from .convex_legacy_adapter import build_single_patient_mc_master_info

    _reject_patient_mr_stage_side_effect_options(config)
    patient_uid = str(patient_uid)
    working_patient_reference_dict = patient_reference_dict if mutate_input else copy.deepcopy(patient_reference_dict)
    master_structure_reference_dict = {patient_uid: working_patient_reference_dict}
    master_structure_info_dict = build_single_patient_mc_master_info(
        patient_uid,
        patient_info_dict,
        global_info=global_info,
    )
    performed_flags = _set_patient_mr_performed_flag(
        master_structure_info_dict,
        config.perform_mc_mr_sim,
    )
    mr_reference_available = patient_has_mr_adc_reference(
        working_patient_reference_dict,
        mr_adc_ref=mr_adc_ref,
    )
    if not mr_reference_available:
        mr_outputs = collect_patient_mr_outputs(
            patient_uid,
            working_patient_reference_dict,
            bx_ref=bx_ref,
        )
        return PatientMRStageResult(
            patient_uid=patient_uid,
            patient_reference_dict=working_patient_reference_dict,
            master_structure_reference_dict=master_structure_reference_dict,
            master_structure_info_dict=master_structure_info_dict,
            mr_outputs=mr_outputs,
            mr_reference_available=False,
            biopsy_count=0,
            performed_flags=performed_flags,
            metadata={"mutated_input": bool(mutate_input), "skip_reason": "patient_missing_mr_adc_reference"},
        )

    num_mc_mr_simulations = _resolve_num_mc_mr_simulations(master_structure_info_dict)
    lattice_context = build_patient_mr_lattice_context(
        patient_uid,
        working_patient_reference_dict,
        mr_adc_ref=mr_adc_ref,
        filter_out_negatives=True,
        mutate_reference=True,
    )
    biopsy_count = 0
    for biopsy_index, biopsy_structure in enumerate(working_patient_reference_dict.get(bx_ref, ())):
        biopsy_context = build_patient_mr_biopsy_context(
            patient_uid,
            biopsy_index,
            biopsy_structure,
            num_mc_mr_simulations=num_mc_mr_simulations,
        )
        localization_outputs = run_patient_mr_localization_for_biopsy(
            biopsy_context,
            lattice_context,
            mr_config=config,
            num_mc_mr_simulations=num_mc_mr_simulations,
        )
        write_patient_mr_localization_outputs_to_legacy_record(
            biopsy_structure,
            localization_outputs,
        )
        biopsy_count += 1

    mr_outputs = collect_patient_mr_outputs(
        patient_uid,
        working_patient_reference_dict,
        bx_ref=bx_ref,
    )
    return PatientMRStageResult(
        patient_uid=patient_uid,
        patient_reference_dict=working_patient_reference_dict,
        master_structure_reference_dict=master_structure_reference_dict,
        master_structure_info_dict=master_structure_info_dict,
        mr_outputs=mr_outputs,
        mr_reference_available=True,
        biopsy_count=biopsy_count,
        performed_flags=performed_flags,
        metadata={
            "mutated_input": bool(mutate_input),
            "num_mc_mr_simulations": num_mc_mr_simulations,
        },
    )


def collect_patient_mr_outputs(patient_uid: str,
                               patient_reference_dict: Mapping[str, Any],
                               *,
                               bx_ref: str) -> PatientMROutputs:
    """Collect MR ADC artifacts written into one patient dictionary."""
    identity_keys = legacy_mc_keys.biopsy_identity
    biopsy_outputs: list[dict[str, Any]] = []
    for biopsy_index, biopsy_structure in enumerate(patient_reference_dict.get(bx_ref, ())):
        outputs = {
            output_key: biopsy_structure.get(output_key)
            for output_key in MC_MR_BIOPSY_OUTPUT_KEYS
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
    return PatientMROutputs(
        patient_uid=str(patient_uid),
        biopsy_outputs=biopsy_outputs,
    )
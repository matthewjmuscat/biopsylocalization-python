"""Patient-level wrappers for downstream MC output dataframe fragments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

import dataframe_builders
from legacy_data_keys import legacy_data_keys

from .convex_legacy_adapter import build_single_patient_mc_master_info
from .legacy_keys import legacy_mc_keys
from .mr import patient_has_mr_adc_reference


DEFAULT_MC_DOSE_OUTPUT_VALUE_COLUMNS = ("Dose (Gy)", "Dose grad (Gy/mm)")


@dataclass(frozen=True, slots=True)
class PatientMCOutputTableConfig:
    """Config for building one patient's downstream MC dataframe fragments."""

    bx_ref: str
    all_ref_key: str
    structs_referenced_list: Sequence[str] = ()
    dose_ref: str = ""
    plan_ref: str = ""
    mr_adc_ref: str = ""
    biopsy_z_voxel_length: float = 1.0
    num_mc_dose_simulations: int | None = None
    dose_value_columns: Sequence[str] = DEFAULT_MC_DOSE_OUTPUT_VALUE_COLUMNS
    d_x_DVH_to_calc_list: Sequence[float] = ()
    v_percent_DVH_to_calc_list: Sequence[float] = ()
    default_ctv_dose: float = 13.5
    include_transform_tables: bool = True
    include_tissue_tables: bool = True
    include_dose_tables: bool = True
    include_dvh_trial_tables: bool = True
    include_dvh_metric_statistics: bool = False
    include_mr_tables: bool = True
    annotate_optimizer_v2_downstream_mc: bool = False
    downstream_trial_count: int | None = None
    use_alternate_mr_by_voxel_builder: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "structs_referenced_list", tuple(self.structs_referenced_list))
        object.__setattr__(self, "dose_value_columns", tuple(self.dose_value_columns))
        object.__setattr__(self, "d_x_DVH_to_calc_list", tuple(self.d_x_DVH_to_calc_list))
        object.__setattr__(self, "v_percent_DVH_to_calc_list", tuple(self.v_percent_DVH_to_calc_list))
        object.__setattr__(self, "biopsy_z_voxel_length", float(self.biopsy_z_voxel_length))
        object.__setattr__(self, "default_ctv_dose", float(self.default_ctv_dose))
        if self.biopsy_z_voxel_length <= 0:
            raise ValueError("biopsy_z_voxel_length must be > 0")


@dataclass(frozen=True, slots=True)
class PatientMCBiopsyOutputTable:
    """One biopsy-scoped MC dataframe fragment stored on a legacy biopsy record."""

    biopsy_index: int
    table_name: str
    dataframe: pd.DataFrame


@dataclass(frozen=True, slots=True)
class PatientMCOutputTableBundle:
    """Downstream MC dataframe fragments available for one patient."""

    patient_uid: str
    patient_mc_dataframes: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    biopsy_mc_dataframes: Sequence[PatientMCBiopsyOutputTable] = ()
    returned_dataframes: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    annotations_applied: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "patient_uid", str(self.patient_uid))
        object.__setattr__(self, "patient_mc_dataframes", dict(self.patient_mc_dataframes))
        object.__setattr__(self, "biopsy_mc_dataframes", tuple(self.biopsy_mc_dataframes))
        object.__setattr__(self, "returned_dataframes", dict(self.returned_dataframes))

    @property
    def patient_table_count(self) -> int:
        return len(self.patient_mc_dataframes)

    @property
    def biopsy_table_count(self) -> int:
        return len(self.biopsy_mc_dataframes)

    @property
    def patient_table_names(self) -> tuple[str, ...]:
        return tuple(self.patient_mc_dataframes.keys())

    @property
    def biopsy_table_names(self) -> tuple[str, ...]:
        return tuple(table.table_name for table in self.biopsy_mc_dataframes)


def _single_patient_master_reference(patient_uid: str,
                                     patient_reference_dict: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(patient_uid): patient_reference_dict}


def _patient_mc_dataframe_store(patient_reference_dict: Mapping[str, Any],
                                all_ref_key: str,
                                *,
                                create: bool = False) -> Mapping[str, Any]:
    all_reference = patient_reference_dict[all_ref_key]
    store_key = legacy_data_keys.patient_all_reference.mc_output_dataframes_key
    if create:
        return all_reference.setdefault(store_key, {})
    return all_reference.get(store_key) or {}


def _patient_biopsy_dataframe_store(specific_bx_structure: Mapping[str, Any]) -> Mapping[str, Any]:
    return specific_bx_structure.get(legacy_data_keys.biopsy_runtime.output_dataframes_key) or {}


def _build_output_master_info(patient_uid: str,
                              patient_info_dict: Mapping[str, Any] | None,
                              config: PatientMCOutputTableConfig,
                              *,
                              global_info: Mapping[str, Any] | None = None) -> dict[str, Any]:
    master_info = build_single_patient_mc_master_info(
        patient_uid,
        patient_info_dict,
        global_info=global_info,
    )
    if config.num_mc_dose_simulations is not None:
        master_keys = legacy_mc_keys.master_info
        mc_info = master_info.setdefault(master_keys.global_key, {}).setdefault(master_keys.mc_info_key, {})
        mc_info[master_keys.num_dose_simulations_key] = int(config.num_mc_dose_simulations)
    return master_info


def collect_patient_mc_output_tables(patient_uid: str,
                                     patient_reference_dict: Mapping[str, Any],
                                     *,
                                     bx_ref: str,
                                     all_ref_key: str,
                                     returned_dataframes: Mapping[str, pd.DataFrame] | None = None,
                                     annotations_applied: bool = False) -> PatientMCOutputTableBundle:
    """Collect patient- and biopsy-scoped MC dataframe fragments from legacy storage."""
    patient_mc_dataframes = {
        str(table_name): dataframe
        for table_name, dataframe in _patient_mc_dataframe_store(patient_reference_dict, all_ref_key).items()
        if isinstance(dataframe, pd.DataFrame)
    }
    biopsy_mc_dataframes: list[PatientMCBiopsyOutputTable] = []
    for biopsy_index, specific_bx_structure in enumerate(patient_reference_dict.get(bx_ref, ())) :
        for table_name, dataframe in _patient_biopsy_dataframe_store(specific_bx_structure).items():
            if isinstance(dataframe, pd.DataFrame):
                biopsy_mc_dataframes.append(
                    PatientMCBiopsyOutputTable(
                        biopsy_index=biopsy_index,
                        table_name=str(table_name),
                        dataframe=dataframe,
                    )
                )
    return PatientMCOutputTableBundle(
        patient_uid=patient_uid,
        patient_mc_dataframes=patient_mc_dataframes,
        biopsy_mc_dataframes=tuple(biopsy_mc_dataframes),
        returned_dataframes=returned_dataframes or {},
        annotations_applied=annotations_applied,
    )


def build_patient_mc_transform_output_tables(patient_uid: str,
                                             patient_reference_dict: dict[str, Any],
                                             config: PatientMCOutputTableConfig) -> PatientMCOutputTableBundle:
    """Build one patient's MC transform dataframe fragment."""
    if not config.structs_referenced_list:
        raise ValueError("structs_referenced_list is required to build MC transform output tables")
    master_reference = _single_patient_master_reference(patient_uid, patient_reference_dict)
    table_keys = legacy_mc_keys.output_tables
    returned_dataframe = dataframe_builders.all_structure_shifts_by_trial_dataframe_builder(
        master_reference,
        config.structs_referenced_list,
        config.bx_ref,
        config.all_ref_key,
    )
    return collect_patient_mc_output_tables(
        patient_uid,
        patient_reference_dict,
        bx_ref=config.bx_ref,
        all_ref_key=config.all_ref_key,
        returned_dataframes={table_keys.all_structure_transformations_key: returned_dataframe},
    )


def build_patient_mc_tissue_output_tables(patient_uid: str,
                                          patient_reference_dict: dict[str, Any],
                                          config: PatientMCOutputTableConfig) -> PatientMCOutputTableBundle:
    """Build one patient's downstream containment/tissue MC dataframe fragments."""
    master_reference = _single_patient_master_reference(patient_uid, patient_reference_dict)
    table_keys = legacy_mc_keys.output_tables
    returned_dataframes: dict[str, pd.DataFrame] = {}

    returned_dataframes[table_keys.tissue_structure_specific_ptwise_key] = (
        dataframe_builders.cohort_and_multi_biopsy_mc_structure_specific_pt_wise_results_dataframe_builder(
            master_reference,
            config.bx_ref,
            config.all_ref_key,
        )
    )
    sum_to_one_dataframe = dataframe_builders.cohort_and_multi_biopsy_mc_sum_to_one_pt_wise_results_dataframe_builder(
        master_reference,
        config.bx_ref,
        config.all_ref_key,
    )
    returned_dataframes[table_keys.tissue_sum_to_one_ptwise_key] = sum_to_one_dataframe
    if not sum_to_one_dataframe.empty:
        returned_dataframes[table_keys.tissue_sum_to_one_global_scores_key] = (
            dataframe_builders.cohort_mc_sum_to_one_global_scores_dataframe_builder(sum_to_one_dataframe)
        )
    returned_dataframes[table_keys.tissue_global_by_structure_key] = (
        dataframe_builders.global_scores_by_specific_structure_dataframe_builder(
            master_reference,
            config.bx_ref,
            config.all_ref_key,
        )
    )
    (
        returned_dataframes[table_keys.tissue_distances_global_key],
        returned_dataframes[table_keys.tissue_distances_ptwise_key],
        returned_dataframes[table_keys.tissue_distances_voxelwise_key],
    ) = dataframe_builders.cohort_relative_structure_distances_dataframe_builder(
        master_reference,
        config.bx_ref,
        config.all_ref_key,
    )
    dataframe_builders.cohort_containment_results_and_distances_dataframe_builder_light(
        master_reference,
        config.bx_ref,
        config.all_ref_key,
    )
    annotations_applied = False
    if config.annotate_optimizer_v2_downstream_mc:
        if config.downstream_trial_count is None:
            raise ValueError("downstream_trial_count is required when optimizer-v2 MC annotation is enabled")
        annotate_patient_optimizer_v2_outputs_with_downstream_mc_scores(
            patient_uid,
            patient_reference_dict,
            all_ref_key=config.all_ref_key,
            downstream_trial_count=config.downstream_trial_count,
        )
        annotations_applied = True
    return collect_patient_mc_output_tables(
        patient_uid,
        patient_reference_dict,
        bx_ref=config.bx_ref,
        all_ref_key=config.all_ref_key,
        returned_dataframes=returned_dataframes,
        annotations_applied=annotations_applied,
    )


def build_patient_mc_dose_output_tables(patient_uid: str,
                                        patient_reference_dict: dict[str, Any],
                                        config: PatientMCOutputTableConfig,
                                        *,
                                        patient_info_dict: Mapping[str, Any] | None = None,
                                        global_info: Mapping[str, Any] | None = None) -> PatientMCOutputTableBundle:
    """Build one patient's downstream dose/DVH MC dataframe fragments."""
    if not config.dose_ref or config.dose_ref not in patient_reference_dict:
        return collect_patient_mc_output_tables(
            patient_uid,
            patient_reference_dict,
            bx_ref=config.bx_ref,
            all_ref_key=config.all_ref_key,
        )

    master_reference = _single_patient_master_reference(patient_uid, patient_reference_dict)
    table_keys = legacy_mc_keys.output_tables
    returned_dataframes: dict[str, pd.DataFrame] = {}
    dataframe_builders.all_dose_data_by_trial_and_pt_from_dataframe_builder_and_voxelizer_v4(
        master_reference,
        config.bx_ref,
        config.biopsy_z_voxel_length,
        config.dose_ref,
    )
    returned_dataframes[table_keys.dosimetry_global_by_voxel_key] = (
        dataframe_builders.global_dosimetry_by_voxel_values_dataframe_builder_v3_generalized(
            master_reference,
            config.bx_ref,
            config.all_ref_key,
            config.dose_ref,
            list(config.dose_value_columns),
        )
    )
    returned_dataframes[table_keys.dosimetry_global_key] = (
        dataframe_builders.global_dosimetry_by_biopsy_dataframe_builder_NEW_multiindex_df(
            master_reference,
            config.bx_ref,
            config.all_ref_key,
            config.dose_ref,
            list(config.dose_value_columns),
        )
    )
    if config.include_dvh_trial_tables:
        master_info = _build_output_master_info(
            patient_uid,
            patient_info_dict,
            config,
            global_info=global_info,
        )
        dataframe_builders.differential_dvh_dataframe_all_mc_trials_dataframe_builder_v2(
            master_reference,
            master_info,
            config.bx_ref,
            config.dose_ref,
        )
        dataframe_builders.cumulative_dvh_dataframe_all_mc_trials_dataframe_builder_v2(
            master_reference,
            master_info,
            config.bx_ref,
            config.dose_ref,
        )
    if config.include_dvh_metric_statistics:
        if not config.plan_ref:
            raise ValueError("plan_ref is required when DVH metric statistics are enabled")
        returned_dataframes[table_keys.dvh_metrics_generalized_key] = (
            dataframe_builders.dvh_metrics_calculator_and_dataframe_builder_cohort(
                master_reference,
                config.bx_ref,
                config.all_ref_key,
                config.dose_ref,
                config.plan_ref,
                list(config.d_x_DVH_to_calc_list),
                list(config.v_percent_DVH_to_calc_list),
                default_ctv_dose=config.default_ctv_dose,
            )
        )
    return collect_patient_mc_output_tables(
        patient_uid,
        patient_reference_dict,
        bx_ref=config.bx_ref,
        all_ref_key=config.all_ref_key,
        returned_dataframes=returned_dataframes,
    )


def build_patient_mc_mr_output_tables(patient_uid: str,
                                      patient_reference_dict: dict[str, Any],
                                      config: PatientMCOutputTableConfig) -> PatientMCOutputTableBundle:
    """Build one patient's downstream MR ADC MC dataframe fragments."""
    if not config.mr_adc_ref or not patient_has_mr_adc_reference(patient_reference_dict, config.mr_adc_ref):
        return collect_patient_mc_output_tables(
            patient_uid,
            patient_reference_dict,
            bx_ref=config.bx_ref,
            all_ref_key=config.all_ref_key,
        )

    master_reference = _single_patient_master_reference(patient_uid, patient_reference_dict)
    biopsy_keys = legacy_mc_keys.biopsy_outputs
    table_keys = legacy_mc_keys.output_tables
    returned_dataframes: dict[str, pd.DataFrame] = {}
    dataframe_builders.all_mr_data_by_trial_and_pt_from_dataframe_builder_and_voxelizer_v4(
        master_reference,
        config.bx_ref,
        config.biopsy_z_voxel_length,
        config.mr_adc_ref,
        biopsy_keys.mr_adc_values_nominal_and_trials_array_key,
        table_keys.mr_adc_column_prefix,
        table_keys.pointwise_mr_adc_by_trial_key,
    )
    returned_dataframes[table_keys.mr_global_statistics_key] = dataframe_builders.global_mr_values_dataframe_builder(
        master_reference,
        config.bx_ref,
        config.all_ref_key,
        config.mr_adc_ref,
        table_keys.mr_adc_column_prefix,
        table_keys.pointwise_mr_adc_by_trial_key,
        table_keys.mr_global_statistics_label,
    )
    mr_by_voxel_builder = (
        dataframe_builders.global_mr_by_voxel_values_dataframe_builder_ALTERNATE
        if config.use_alternate_mr_by_voxel_builder
        else dataframe_builders.global_mr_by_voxel_values_dataframe_builder
    )
    returned_dataframes[table_keys.mr_global_by_voxel_statistics_key] = mr_by_voxel_builder(
        master_reference,
        config.bx_ref,
        config.all_ref_key,
        config.mr_adc_ref,
        table_keys.mr_adc_column_prefix,
        table_keys.pointwise_mr_adc_by_trial_key,
        table_keys.mr_global_by_voxel_statistics_label,
    )
    return collect_patient_mc_output_tables(
        patient_uid,
        patient_reference_dict,
        bx_ref=config.bx_ref,
        all_ref_key=config.all_ref_key,
        returned_dataframes=returned_dataframes,
    )


def annotate_patient_optimizer_v2_outputs_with_downstream_mc_scores(patient_uid: str,
                                                                    patient_reference_dict: dict[str, Any],
                                                                    *,
                                                                    all_ref_key: str,
                                                                    downstream_trial_count: int) -> None:
    """Annotate one patient's optimizer-v2 dataframe fragments with downstream MC scores."""
    from biopsy_optimizer.v2.live_integration import (
        annotate_target_dil_optimizer_v2_outputs_with_downstream_mc_scores,
    )

    annotate_target_dil_optimizer_v2_outputs_with_downstream_mc_scores(
        _single_patient_master_reference(patient_uid, patient_reference_dict),
        all_ref_key,
        int(downstream_trial_count),
    )


def build_patient_mc_downstream_output_tables(patient_uid: str,
                                              patient_reference_dict: dict[str, Any],
                                              config: PatientMCOutputTableConfig,
                                              *,
                                              patient_info_dict: Mapping[str, Any] | None = None,
                                              global_info: Mapping[str, Any] | None = None) -> PatientMCOutputTableBundle:
    """Build configured downstream MC dataframe fragments for one patient."""
    returned_dataframes: dict[str, pd.DataFrame] = {}
    annotations_applied = False
    if config.include_transform_tables:
        returned_dataframes.update(
            build_patient_mc_transform_output_tables(patient_uid, patient_reference_dict, config).returned_dataframes
        )
    if config.include_tissue_tables:
        tissue_bundle = build_patient_mc_tissue_output_tables(patient_uid, patient_reference_dict, config)
        returned_dataframes.update(tissue_bundle.returned_dataframes)
        annotations_applied = annotations_applied or tissue_bundle.annotations_applied
    if config.include_dose_tables:
        returned_dataframes.update(
            build_patient_mc_dose_output_tables(
                patient_uid,
                patient_reference_dict,
                config,
                patient_info_dict=patient_info_dict,
                global_info=global_info,
            ).returned_dataframes
        )
    if config.include_mr_tables:
        returned_dataframes.update(
            build_patient_mc_mr_output_tables(patient_uid, patient_reference_dict, config).returned_dataframes
        )
    return collect_patient_mc_output_tables(
        patient_uid,
        patient_reference_dict,
        bx_ref=config.bx_ref,
        all_ref_key=config.all_ref_key,
        returned_dataframes=returned_dataframes,
        annotations_applied=annotations_applied,
    )
from __future__ import annotations

"""Machine-readable output schema registry and human-readable dictionary tools.

The registry is the durable contract layer for output tables. Phase-specific
surfaces can come and go, but durable outputs should be represented here so the
pipeline, GUI, validation reports, and downstream analysis code can agree on
row grain, join keys, lineage, validation status, and retention policy.
"""

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd


OUTPUT_SCHEMA_REGISTRY_VERSION = "phase3d_output_schema_registry_v1"
OUTPUT_SCHEMA_COVERAGE_SCHEMA_VERSION = "phase3d_output_schema_coverage_v1"
EXPECTED_CURRENT_REGISTRY_COUNT = 62


@dataclass(frozen=True)
class CanonicalKeySpec:
    """Named canonical key group used to join tables across output families."""

    key_id: str
    columns: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class OutputTableSpec:
    """Contract for one durable output table.

    This is intentionally more formal than a filename list. It records what a
    row means, how the table joins to other outputs, whether it is a source or
    derived product, and how validated the current implementation is.
    """

    table_id: str
    legacy_table_name: str
    legacy_output_section: str
    file_extension: str
    artifact_scope: str
    table_family: str
    row_grain: str
    canonical_primary_key: tuple[str, ...]
    join_keys: tuple[str, ...]
    legacy_key_columns: tuple[str, ...]
    columns_policy: str
    storage_format: str
    has_multiindex_columns: bool | None
    source_stage: str
    source_fragment_table_id: str
    stitch_method: str
    aggregation_builder: str
    validation_status: str
    retention_policy: str
    downstream_usage: str
    notes: str = ""
    match_mode: str = "exact"

    def to_row(self) -> dict[str, Any]:
        """Return a CSV-friendly registry row with tuple fields flattened."""

        row = asdict(self)
        for key, value in list(row.items()):
            if isinstance(value, tuple):
                row[key] = " | ".join(value)
            if value is None:
                row[key] = "unknown_until_runtime"
        row["schema_version"] = OUTPUT_SCHEMA_REGISTRY_VERSION
        return row


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _key(*columns: str) -> tuple[str, ...]:
    return columns


def _spec(table_id: str,
          legacy_table_name: str,
          legacy_output_section: str,
          artifact_scope: str,
          table_family: str,
          row_grain: str,
          canonical_primary_key: tuple[str, ...],
          *,
          file_extension: str = ".csv",
          join_keys: tuple[str, ...] = (),
          legacy_key_columns: tuple[str, ...] = ("Bx refnum", "Relative DIL ref num", "Structure ref num"),
          columns_policy: str = "legacy_preserve",
          storage_format: str = "csv",
          has_multiindex_columns: bool | None = None,
          source_stage: str = "",
          source_fragment_table_id: str = "",
          stitch_method: str = "none",
          aggregation_builder: str = "",
          validation_status: str = "legacy_review",
          retention_policy: str = "retain_core",
          downstream_usage: str = "unknown",
          notes: str = "",
          match_mode: str = "exact") -> OutputTableSpec:
    """Build an output-table spec with conservative defaults.

    The helper keeps the 63 table entries reviewable without repeating every
    default field. A table should only use these defaults when the default is a
    deliberate policy, not when information is unknown.
    """

    if not source_stage:
        source_stage = {
            "Output CSVs/Preprocessing": "preprocessing",
            "Output CSVs/MC simulation": "mc_simulation",
            "Output CSVs/Cohort": "cohort_finalization",
            "manifests": "manifest",
            "uncertainties_file_auto_generated": "run_metadata",
        }.get(legacy_output_section, "unknown")
    return OutputTableSpec(
        table_id=table_id,
        legacy_table_name=legacy_table_name,
        legacy_output_section=legacy_output_section,
        file_extension=file_extension,
        artifact_scope=artifact_scope,
        table_family=table_family,
        row_grain=row_grain,
        canonical_primary_key=canonical_primary_key,
        join_keys=join_keys,
        legacy_key_columns=legacy_key_columns,
        columns_policy=columns_policy,
        storage_format=storage_format,
        has_multiindex_columns=has_multiindex_columns,
        source_stage=source_stage,
        source_fragment_table_id=source_fragment_table_id,
        stitch_method=stitch_method,
        aggregation_builder=aggregation_builder,
        validation_status=validation_status,
        retention_policy=retention_policy,
        downstream_usage=downstream_usage,
        notes=notes,
        match_mode=match_mode,
    )


CANONICAL_KEY_SPECS = (
    CanonicalKeySpec("patient_fraction_key", _key("Patient ID"), "Split into base patient and fraction later."),
    CanonicalKeySpec("biopsy_key", _key("Patient ID", "Bx index")),
    CanonicalKeySpec("structure_key", _key("Patient ID", "Structure type", "Structure index")),
    CanonicalKeySpec("relative_structure_key", _key("Patient ID", "Bx index", "Relative structure type", "Relative structure index")),
    CanonicalKeySpec("voxel_key", _key("Patient ID", "Bx index", "Voxel index")),
    CanonicalKeySpec("mc_trial_key", _key("Patient ID", "Bx index", "MC trial num")),
    CanonicalKeySpec("mc_trial_point_key", _key("Patient ID", "Bx index", "MC trial num", "point index")),
    CanonicalKeySpec("mc_trial_voxel_key", _key("Patient ID", "Bx index", "MC trial num", "Voxel index")),
    CanonicalKeySpec("dose_bin_key", _key("Patient ID", "Bx index", "MC trial num", "dose-bin identity")),
)


OUTPUT_TABLE_SPECS = (
    _spec("cohort_structure_radiomic_features", "Cohort: 3D radiomic features all OAR and DIL structures", "Output CSVs/Cohort", "cohort", "radiomics", "structure", _key("Patient ID", "Structure type", "Structure index"), stitch_method="concat_rows", source_fragment_table_id="patient_structure_radiomic_features", validation_status="validated_phase3c", downstream_usage="used_by_sibling_repo"),
    _spec("cohort_all_mc_structure_transforms", "Cohort: All MC structure transformation values", "Output CSVs/Cohort", "cohort", "mc_transform", "structure_trial", _key("Patient ID", "Structure type", "Structure index", "Trial"), stitch_method="concat_rows", source_fragment_table_id="patient_all_mc_structure_transforms", validation_status="validated_phase3c", downstream_usage="used_by_sibling_repo"),
    _spec("cohort_biopsy_basic_spatial_features", "Cohort: Biopsy basic spatial features dataframe", "Output CSVs/Cohort", "cohort", "biopsy_geometry", "biopsy", _key("Patient ID", "Bx index"), stitch_method="concat_rows", source_fragment_table_id="patient_biopsy_basic_spatial_features", validation_status="validated_phase3c", downstream_usage="used_by_sibling_repo"),
    _spec("cohort_bx_dvh_metrics_generalized_legacy", "Cohort: Bx DVH metrics (generalized)", "Output CSVs/Cohort", "cohort", "dvh", "biopsy_metric", _key("Patient ID", "Bx index", "metric identity"), stitch_method="recompute_downstream", source_fragment_table_id="patient_dvh_metrics_generalized_legacy", validation_status="validated_phase3c_legacy", retention_policy="reimplement_later", downstream_usage="used_by_sibling_repo", notes="Candidate for clean post-run/GUI DVH implementation after sister-repo validation."),
    _spec("cohort_global_mr_adc_statistics", "Cohort: Global MR ADC statistics", "Output CSVs/Cohort", "cohort", "mr_adc", "biopsy", _key("Patient ID", "Bx index", "statistic identity"), stitch_method="concat_current_summary_fragments", source_fragment_table_id="patient_global_mr_adc_statistics", validation_status="validated_phase3c", downstream_usage="not_found_in_sibling_scan", notes="Scientifically a summary product; current patient fragments are already summarized."),
    _spec("cohort_global_by_voxel_mr_adc_statistics", "Cohort: Global by voxel MR ADC statistics", "Output CSVs/Cohort", "cohort", "mr_adc", "biopsy_voxel", _key("Patient ID", "Bx index", "Voxel index", "dose/MR statistic identity"), stitch_method="concat_rows", source_fragment_table_id="patient_global_by_voxel_mr_adc_statistics", validation_status="validated_phase3c", downstream_usage="not_found_in_sibling_scan"),
    _spec("cohort_global_dosimetry", "Cohort: Global dosimetry (NEW)", "Output CSVs/Cohort", "cohort", "dosimetry", "biopsy", _key("Patient ID", "Bx index", "statistic identity"), columns_policy="multiindex_preserve", has_multiindex_columns=True, stitch_method="concat_current_summary_fragments", source_fragment_table_id="patient_global_dosimetry", validation_status="validated_phase3c", downstream_usage="used_by_sibling_repo", notes="Scientifically a dose summary product; current patient fragments are already summarized."),
    _spec("cohort_global_dosimetry_by_voxel", "Cohort: Global dosimetry by voxel", "Output CSVs/Cohort", "cohort", "dosimetry", "biopsy_voxel", _key("Patient ID", "Bx index", "Voxel index", "dose/MR statistic identity"), columns_policy="multiindex_preserve", has_multiindex_columns=True, stitch_method="concat_rows", source_fragment_table_id="patient_global_dosimetry_by_voxel", validation_status="validated_phase3c", downstream_usage="used_by_sibling_repo"),
    _spec("cohort_guidance_map_firing_depth", "Cohort: Guidance-map firing depth recommendations dataframe", "Output CSVs/Cohort", "cohort", "guidance_map", "guidance_candidate", _key("Firing depth row UID"), stitch_method="concat_rows", source_fragment_table_id="patient_guidance_map_firing_depth", validation_status="validated_phase3c", downstream_usage="unknown"),
    _spec("cohort_nearest_dils", "Cohort: Nearest DILs to each biopsy", "Output CSVs/Cohort", "cohort", "biopsy_geometry", "biopsy_relative_structure", _key("Patient ID", "Bx index", "Relative struct type", "Relative DIL index"), stitch_method="concat_rows", source_fragment_table_id="patient_nearest_dils", validation_status="validated_phase3c", downstream_usage="used_by_sibling_repo"),
    _spec("cohort_double_sextant_per_sample", "Cohort: Per sample point prostate double sextant classification", "Output CSVs/Cohort", "cohort", "spatial_classification", "biopsy_sample_point", _key("Patient ID", "Bx index", "Voxel index", "sample point identity"), stitch_method="concat_rows", source_fragment_table_id="patient_double_sextant_per_sample", validation_status="validated_phase3c", downstream_usage="not_found_in_sibling_scan"),
    _spec("cohort_double_sextant_per_voxel", "Cohort: Per voxel prostate double sextant classification", "Output CSVs/Cohort", "cohort", "spatial_classification", "biopsy_voxel", _key("Patient ID", "Bx index", "Voxel index"), stitch_method="concat_rows", source_fragment_table_id="patient_double_sextant_per_voxel", validation_status="validated_phase3c", downstream_usage="used_by_sibling_repo"),
    _spec("cohort_planned_vs_realized_centroid_validation", "Cohort: Simulated biopsy planned vs realized centroid variation validation", "Output CSVs/Cohort", "cohort", "validation", "biopsy_validation_field", _key("Patient ID", "Bx index", "validation comparison field"), stitch_method="concat_rows", source_fragment_table_id="patient_planned_vs_realized_centroid_validation", validation_status="validated_phase3c", retention_policy="validation_only", downstream_usage="not_for_analysis", notes="Migration/QA validation table; keep in validation outputs, not normal cohort CSV export."),
    _spec("cohort_simulated_biopsy_preparation", "Cohort: Simulated biopsy preparation dataframe", "Output CSVs/Cohort", "cohort", "biopsy_geometry", "biopsy", _key("Patient ID", "Bx index"), stitch_method="concat_rows", source_fragment_table_id="patient_simulated_biopsy_preparation", validation_status="validated_phase3c", downstream_usage="used_by_sibling_repo"),
    _spec("cohort_tissue_distances_global", "Cohort: Tissue class - distances global results", "Output CSVs/Cohort", "cohort", "tissue_class", "biopsy_relative_structure", _key("Patient ID", "Bx index", "Relative struct type", "Relative structure index"), columns_policy="multiindex_preserve", has_multiindex_columns=True, stitch_method="concat_current_summary_fragments", source_fragment_table_id="patient_tissue_distances_global", validation_status="validated_phase3c", downstream_usage="used_by_sibling_repo", notes="Scientifically summary statistics from lower-level distance rows."),
    _spec("cohort_tissue_distances_ptwise", "Cohort: Tissue class - distances pt-wise results", "Output CSVs/Cohort", "cohort", "tissue_class", "biopsy_trial_point_structure", _key("Patient ID", "Bx index", "MC trial num", "point index", "Relative struct type", "Relative structure index"), columns_policy="multiindex_preserve", has_multiindex_columns=True, stitch_method="concat_rows", source_fragment_table_id="patient_tissue_distances_ptwise", validation_status="validated_phase3c", downstream_usage="used_by_sibling_repo"),
    _spec("cohort_tissue_distances_voxelwise", "Cohort: Tissue class - distances voxel-wise results", "Output CSVs/Cohort", "cohort", "tissue_class", "biopsy_trial_voxel_structure", _key("Patient ID", "Bx index", "MC trial num", "Voxel index", "Relative struct type", "Relative structure index"), columns_policy="multiindex_preserve", has_multiindex_columns=True, stitch_method="concat_rows", source_fragment_table_id="patient_tissue_distances_voxelwise", validation_status="validated_phase3c", downstream_usage="used_by_sibling_repo"),
    _spec("cohort_global_sum_to_one_mc", "Cohort: global sum-to-one mc results", "Output CSVs/Cohort", "cohort", "tissue_class", "biopsy_tissue_class", _key("Patient ID", "Bx index", "Tissue class"), stitch_method="aggregate_from_long_form", source_fragment_table_id="patient_sum_to_one_mc_results", aggregation_builder="dataframe_builders.cohort_mc_sum_to_one_global_scores_dataframe_builder", validation_status="needs_aggregation_builder", downstream_usage="used_by_sibling_repo", notes="Derived summary of the long-form sum-to-one MC table; base patient artifact is already registered, downstream repos can regenerate this summary."),
    _spec("cohort_structure_specific_mc_results", "Cohort: structure specific mc results", "Output CSVs/Cohort", "cohort", "tissue_class", "biopsy_trial_point_structure", _key("Patient ID", "Bx index", "MC trial num", "point/voxel identity", "structure identity"), stitch_method="concat_rows", source_fragment_table_id="patient_tissue_structure_specific_ptwise", validation_status="validated_phase3c", retention_policy="retain_core", downstream_usage="not_found_in_sibling_scan", notes="Granular structure-specific MC table; distinct from aggregated tissue-class global scores and retained as a likely base artifact pending downstream review."),
    _spec("cohort_sum_to_one_mc_results", "Cohort: sum-to-one mc results", "Output CSVs/Cohort", "cohort", "tissue_class", "biopsy_voxel_tissue_class", _key("Patient ID", "Bx index", "Voxel index", "Tissue class"), stitch_method="concat_rows", source_fragment_table_id="patient_sum_to_one_mc_results", validation_status="validated_phase3c", downstream_usage="used_by_sibling_repo"),
    _spec("cohort_tissue_class_global_scores_structure", "Cohort: tissue class global scores (structure)", "Output CSVs/Cohort", "cohort", "tissue_class", "biopsy_relative_structure", _key("Patient ID", "Bx index", "Relative struct type", "Relative structure index"), stitch_method="concat_current_summary_fragments", source_fragment_table_id="patient_tissue_class_global_by_structure", validation_status="validated_phase3c", retention_policy="downstream_calculable", downstream_usage="not_found_in_sibling_scan", notes="Aggregated biopsy-by-relative-structure summary computed from granular structure-specific MC results; can be regenerated downstream from lower-level artifacts."),
    _spec("uncertainties_final", "Uncertainties dataframe (final)", "Output CSVs/Cohort", "run_metadata", "uncertainty", "run", _key("uncertainty parameter name"), legacy_key_columns=(), stitch_method="manifest_metadata", validation_status="metadata_only", retention_policy="migrate_to_manifest", downstream_usage="unknown"),
    _spec("uncertainties_unedited", "Uncertainties dataframe (unedited)", "Output CSVs/Cohort", "run_metadata", "uncertainty", "run", _key("uncertainty parameter name"), legacy_key_columns=(), stitch_method="manifest_metadata", validation_status="metadata_only", retention_policy="migrate_to_manifest", downstream_usage="unknown"),

    _spec("patient_all_mc_structure_transforms", "All MC structure transformation values", "Output CSVs/MC simulation", "patient", "mc_transform", "structure_trial", _key("Patient ID", "Structure type", "Structure index", "Trial"), stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("biopsy_cumulative_dvh_by_trial", "Cumulative DVH by MC trial", "Output CSVs/MC simulation", "biopsy", "dvh", "biopsy_trial_dose_bin", _key("Patient ID", "Bx index", "MC trial num", "dose-bin identity"), file_extension=".parquet", storage_format="parquet", stitch_method="recompute_downstream", validation_status="phase3c_source_surface", retention_policy="retain_validation_only", downstream_usage="used_by_sibling_repo"),
    _spec("patient_dvh_metrics_generalized_legacy", "DVH metrics (Dx, Vx) statistics", "Output CSVs/MC simulation", "patient", "dvh", "biopsy_metric", _key("Patient ID", "Bx index", "metric identity"), stitch_method="recompute_downstream", validation_status="phase3c_source_surface_legacy", retention_policy="reimplement_later"),
    _spec("biopsy_differential_dvh_by_trial", "Differential DVH by MC trial", "Output CSVs/MC simulation", "biopsy", "dvh", "biopsy_trial_dose_bin", _key("Patient ID", "Bx index", "MC trial num", "dose-bin identity"), file_extension=".parquet", storage_format="parquet", stitch_method="recompute_downstream", validation_status="phase3c_source_surface", retention_policy="retain_validation_only", downstream_usage="used_by_sibling_repo"),
    _spec("patient_global_dosimetry", "Dosimetry - Global dosimetry (NEW)", "Output CSVs/MC simulation", "patient", "dosimetry", "biopsy", _key("Patient ID", "Bx index", "statistic identity"), columns_policy="multiindex_preserve", has_multiindex_columns=True, stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("patient_global_dosimetry_by_voxel", "Dosimetry - Global dosimetry by voxel statistics", "Output CSVs/MC simulation", "patient", "dosimetry", "biopsy_voxel", _key("Patient ID", "Bx index", "Voxel index", "dose/MR statistic identity"), columns_policy="multiindex_preserve", has_multiindex_columns=True, stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("patient_global_mr_adc_statistics", "MR - Global MR ADC statistics", "Output CSVs/MC simulation", "patient", "mr_adc", "biopsy", _key("Patient ID", "Bx index", "statistic identity"), stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("patient_global_by_voxel_mr_adc_statistics", "MR - Global by voxel MR ADC statistics", "Output CSVs/MC simulation", "patient", "mr_adc", "biopsy_voxel", _key("Patient ID", "Bx index", "Voxel index", "dose/MR statistic identity"), stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("biopsy_pointwise_mr_adc_by_trial", "Point-wise MR ADC output by MC trial number", "Output CSVs/MC simulation", "biopsy", "mr_adc", "biopsy_trial_point", _key("Patient ID", "Bx index", "MC trial num", "point index"), file_extension=".parquet", storage_format="parquet", stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("biopsy_pointwise_dose_by_trial", "Point-wise dose output by MC trial number", "Output CSVs/MC simulation", "biopsy", "dosimetry", "biopsy_trial_point", _key("Patient ID", "Bx index", "MC trial num", "point index"), file_extension=".parquet", storage_format="parquet", stitch_method="source_fragment", validation_status="phase3c_source_surface", downstream_usage="used_by_sibling_repo"),
    _spec("patient_tissue_class_global_by_structure", "Tissue class - Global tissue by structure statistics", "Output CSVs/MC simulation", "patient", "tissue_class", "biopsy_relative_structure", _key("Patient ID", "Bx index", "Relative structure type", "Relative structure index"), stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("patient_tissue_structure_specific_ptwise", "Tissue class - Pt wise structure specific results", "Output CSVs/MC simulation", "patient", "tissue_class", "biopsy_trial_point_structure", _key("Patient ID", "Bx index", "MC trial num", "point index", "Relative structure type", "Relative structure index"), stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("biopsy_tissue_containment_distances_light", "Tissue class - containment and distances (light) results", "Output CSVs/MC simulation", "biopsy", "tissue_class", "biopsy_trial_point_structure", _key("Patient ID", "Bx index", "MC trial num", "point/voxel identity", "Relative structure type", "Relative structure index"), file_extension=".parquet", storage_format="parquet", stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("patient_tissue_distances_global", "Tissue class - distances global results", "Output CSVs/MC simulation", "patient", "tissue_class", "biopsy_relative_structure", _key("Patient ID", "Bx index", "Relative struct type", "Relative structure index"), columns_policy="multiindex_preserve", has_multiindex_columns=True, stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("patient_tissue_distances_ptwise", "Tissue class - distances pt-wise results", "Output CSVs/MC simulation", "patient", "tissue_class", "biopsy_trial_point_structure", _key("Patient ID", "Bx index", "MC trial num", "point index", "Relative struct type", "Relative structure index"), columns_policy="multiindex_preserve", has_multiindex_columns=True, stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("patient_tissue_distances_voxelwise", "Tissue class - distances voxel-wise results", "Output CSVs/MC simulation", "patient", "tissue_class", "biopsy_trial_voxel_structure", _key("Patient ID", "Bx index", "MC trial num", "Voxel index", "Relative struct type", "Relative structure index"), columns_policy="multiindex_preserve", has_multiindex_columns=True, stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("patient_sum_to_one_mc_results", "Tissue class - sum-to-one mc results", "Output CSVs/MC simulation", "patient", "tissue_class", "biopsy_voxel_tissue_class", _key("Patient ID", "Bx index", "Voxel index", "Tissue class"), stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("biopsy_voxelwise_dose_by_trial", "Voxel-wise dose output by MC trial number", "Output CSVs/MC simulation", "biopsy", "dosimetry", "biopsy_trial_voxel", _key("Patient ID", "Bx index", "MC trial num", "Voxel index"), file_extension=".parquet", storage_format="parquet", stitch_method="source_fragment", validation_status="phase3c_source_surface", downstream_usage="used_by_sibling_repo"),

    _spec("patient_biopsy_basic_spatial_features", "Biopsy basic spatial features dataframe", "Output CSVs/Preprocessing", "patient", "biopsy_geometry", "biopsy", _key("Patient ID", "Bx index"), stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("patient_structure_radiomic_features", "3D radiomic features all OAR and DIL structures", "Output CSVs/Preprocessing", "patient", "radiomics", "structure", _key("Patient ID", "Structure type", "Structure index"), stitch_method="source_fragment", validation_status="phase3c_source_surface", downstream_usage="used_by_sibling_repo"),
    _spec("patient_optimizer_cumulative_projection", "Biopsy optimization - Cumulative projection (all points within prostate) dataframe", "Output CSVs/Preprocessing", "patient", "optimizer", "optimizer_projection", _key("Patient ID", "Relative DIL index", "projection/test-location row identity"), stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("patient_optimizer_dil_centroids", "Biopsy optimization - DIL centroids optimal targeting dataframe", "Output CSVs/Preprocessing", "patient", "optimizer", "optimizer_target", _key("Patient ID", "Relative DIL index"), stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("patient_guidance_map_firing_depth", "Biopsy optimization - Guidance-map firing depth recommendations dataframe", "Output CSVs/Preprocessing", "patient", "guidance_map", "guidance_candidate", _key("Firing depth row UID"), stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("patient_optimizer_optimal_dil", "Biopsy optimization - Optimal DIL targeting dataframe", "Output CSVs/Preprocessing", "patient", "optimizer", "optimizer_target", _key("Patient ID", "Relative DIL index"), stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("patient_optimizer_optimal_dil_lattice", "Biopsy optimization - Optimal DIL targeting entire lattice dataframe", "Output CSVs/Preprocessing", "patient", "optimizer", "optimizer_lattice_point", _key("Patient ID", "Relative DIL index", "test-location row identity"), stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("patient_optimizer_v2_ranked_candidates", "Biopsy optimization - Target DIL optimizer v2 ranked candidates dataframe", "Output CSVs/Preprocessing", "patient", "optimizer", "optimizer_candidate", _key("Patient ID", "Bx index", "Candidate index global"), stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("patient_optimizer_v2_summary", "Biopsy optimization - Target DIL optimizer v2 summary dataframe", "Output CSVs/Preprocessing", "patient", "optimizer", "biopsy", _key("Patient ID", "Bx index"), stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("patient_optimizer_v2_tested_candidates", "Biopsy optimization - Target DIL optimizer v2 tested candidates dataframe", "Output CSVs/Preprocessing", "patient", "optimizer", "optimizer_stage_candidate", _key("Patient ID", "Bx index", "Candidate index global", "Stage name"), stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("patient_mr_adc_summary_by_structure", "MR - ADC - summary statistics by structure dataframe", "Output CSVs/Preprocessing", "patient", "mr_adc", "structure", _key("Patient ID", "Structure type", "Structure index", "MR statistic identity"), stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("patient_nearest_dils", "Nearest DILs info dataframe", "Output CSVs/Preprocessing", "patient", "biopsy_geometry", "biopsy_relative_structure", _key("Patient ID", "Bx index", "Relative struct type", "Relative DIL index"), stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("patient_double_sextant_per_sample", "Per sample point prostate double sextant classification", "Output CSVs/Preprocessing", "patient", "spatial_classification", "biopsy_sample_point", _key("Patient ID", "Bx index", "Voxel index", "sample point identity"), stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("patient_double_sextant_per_voxel", "Per voxel prostate double sextant classification", "Output CSVs/Preprocessing", "patient", "spatial_classification", "biopsy_voxel", _key("Patient ID", "Bx index", "Voxel index"), stitch_method="source_fragment", validation_status="phase3c_source_surface", downstream_usage="used_by_sibling_repo"),
    _spec("patient_selected_structures", "Selected structures", "Output CSVs/Preprocessing", "patient", "preprocessing", "structure", _key("Patient ID", "Struct ref type", "Index number"), stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("patient_simulated_biopsy_preparation", "Simulated biopsy preparation dataframe", "Output CSVs/Preprocessing", "patient", "biopsy_geometry", "biopsy", _key("Patient ID", "Bx index"), stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("patient_planned_vs_realized_centroid_validation", "Simulated biopsy planned vs realized centroid variation validation", "Output CSVs/Preprocessing", "patient", "validation", "biopsy_validation_field", _key("Patient ID", "Bx index", "validation comparison field"), stitch_method="source_fragment", validation_status="phase3c_source_surface"),
    _spec("patient_structure_preprocessing_timings", "Structure preprocessing timings", "Output CSVs/Preprocessing", "patient", "preprocessing", "structure_timing", _key("Patient ID", "Structure ref type", "Structure index", "preprocessing phase/subphase"), stitch_method="source_fragment", validation_status="phase3c_source_surface"),

    _spec("input_case_manifest", "input_case_manifest", "manifests", "run_metadata", "input_manifest", "run", _key("manifest row identity"), legacy_key_columns=(), stitch_method="manifest_metadata", validation_status="metadata_only", retention_policy="retain_core"),
    _spec("input_dicom_manifest", "input_dicom_manifest", "manifests", "run_metadata", "input_manifest", "run", _key("input path"), legacy_key_columns=(), stitch_method="manifest_metadata", validation_status="metadata_only", retention_policy="retain_core"),
    _spec("uncertainty_generated_sidecar", "uncertainties_file_auto_generated", "uncertainties_file_auto_generated", "run_metadata", "uncertainty", "run", _key("uncertainty parameter name"), legacy_key_columns=(), stitch_method="manifest_metadata", validation_status="metadata_only", retention_policy="migrate_to_manifest", match_mode="prefix"),
)


class OutputSchemaRegistry:
    """Reviewable collection of output table specs with legacy-name matching."""

    def __init__(self, specs: tuple[OutputTableSpec, ...] = OUTPUT_TABLE_SPECS):
        self.specs = specs
        table_ids = [spec.table_id for spec in specs]
        duplicate_ids = sorted([table_id for table_id, count in Counter(table_ids).items() if count > 1])
        if duplicate_ids:
            raise ValueError(f"Duplicate output schema registry table IDs: {duplicate_ids}")

    def to_dataframe(self) -> pd.DataFrame:
        """Return all registry specs as a machine-readable dataframe."""

        return pd.DataFrame([spec.to_row() for spec in self.specs])

    def match_spec(self, table_name: str, output_section: str, file_extension: str) -> OutputTableSpec | None:
        """Find the registry spec for a legacy table name and output location."""

        matches: list[OutputTableSpec] = []
        for spec in self.specs:
            if spec.legacy_output_section != output_section or spec.file_extension != file_extension:
                continue
            if spec.match_mode == "exact" and spec.legacy_table_name == table_name:
                matches.append(spec)
            elif spec.match_mode == "prefix" and table_name.startswith(spec.legacy_table_name):
                matches.append(spec)
        if len(matches) > 1:
            raise ValueError(f"Multiple schema specs matched {table_name!r} in {output_section!r}: {matches}")
        return matches[0] if matches else None


def _infer_output_section(relative_path: str) -> str:
    """Infer the legacy output section from a Phase 3C manifest path."""

    parts = Path(str(relative_path)).parts
    if len(parts) >= 2 and parts[0] == "Output CSVs":
        return "/".join(parts[:2])
    if parts[:1] == ("manifests",):
        return "manifests"
    if parts and parts[0].startswith("uncertainties_file_auto_generated"):
        return "uncertainties_file_auto_generated"
    return parts[0] if parts else "unknown"


def _manifest_matches(manifest_df: pd.DataFrame, registry: OutputSchemaRegistry) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach registry table IDs to Phase 3C manifest rows."""

    if manifest_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []
    for row in manifest_df.to_dict(orient="records"):
        output_section = _infer_output_section(str(row.get("relative_path", "")))
        table_name = str(row.get("dataframe_name", ""))
        file_extension = str(row.get("file_extension", ""))
        spec = registry.match_spec(table_name, output_section, file_extension)
        out_row = {**row, "matched_output_section": output_section}
        if spec is None:
            unmatched_rows.append(out_row)
            continue
        out_row["table_id"] = spec.table_id
        rows.append(out_row)
    return pd.DataFrame(rows), pd.DataFrame(unmatched_rows)


def build_output_schema_coverage_report(phase3c_manifest_df: pd.DataFrame,
                                        stitch_validation_df: pd.DataFrame,
                                        registry: OutputSchemaRegistry | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the expected registry surface with a concrete Phase 3C run.

    The first dataframe has one row per registry table. The second dataframe
    contains manifest rows that were written by Phase 3C but do not yet have a
    matching registry entry.
    """

    registry = OutputSchemaRegistry() if registry is None else registry
    matched_manifest_df, unmatched_manifest_df = _manifest_matches(phase3c_manifest_df, registry)
    stitch_rows_by_final = {
        str(row.get("final_table_name", "")): row
        for row in stitch_validation_df.to_dict(orient="records")
    } if not stitch_validation_df.empty else {}

    coverage_rows: list[dict[str, Any]] = []
    for spec in registry.specs:
        spec_manifest_rows = matched_manifest_df[matched_manifest_df["table_id"].eq(spec.table_id)] if not matched_manifest_df.empty else pd.DataFrame()
        phase3c_artifact_count = int(len(spec_manifest_rows))
        patient_fragment_count = 0
        stitched_final_count = 0
        total_rows = 0
        total_bytes = 0
        source_scopes = ""
        phase3c_roles = ""
        has_multiindex_runtime = False
        if not spec_manifest_rows.empty:
            patient_fragment_count = int(spec_manifest_rows["phase3c_role"].eq("patient_fragment_artifact").sum())
            stitched_final_count = int(spec_manifest_rows["phase3c_role"].eq("stitched_final_artifact").sum())
            total_rows = int(pd.to_numeric(spec_manifest_rows["row_count"], errors="coerce").fillna(0).sum())
            total_bytes = int(pd.to_numeric(spec_manifest_rows["file_size_bytes"], errors="coerce").fillna(0).sum())
            source_scopes = "; ".join(sorted(set(spec_manifest_rows["source_scope"].astype(str))))
            phase3c_roles = "; ".join(sorted(set(spec_manifest_rows["phase3c_role"].astype(str))))
            has_multiindex_runtime = bool(spec_manifest_rows["has_multiindex_columns"].astype(bool).any())

        stitch_row = stitch_rows_by_final.get(spec.legacy_table_name, {})
        stitch_validation_status = str(stitch_row.get("validation_status", ""))
        if stitch_validation_status == "match":
            coverage_status = "validated_stitched_final"
        elif phase3c_artifact_count > 0:
            coverage_status = "phase3c_artifact_present"
        elif spec.validation_status in {"metadata_only", "needs_aggregation_builder", "needs_recompute_validation", "needs_live_phase3c_validation"}:
            coverage_status = spec.validation_status
        elif spec.validation_status == "needs_phase3d_route":
            coverage_status = "missing_phase3d_route"
        else:
            coverage_status = "not_present_in_phase3c_surface"

        coverage_rows.append({
            **spec.to_row(),
            "coverage_schema_version": OUTPUT_SCHEMA_COVERAGE_SCHEMA_VERSION,
            "phase3c_artifact_count": phase3c_artifact_count,
            "phase3c_patient_fragment_count": patient_fragment_count,
            "phase3c_stitched_final_count": stitched_final_count,
            "phase3c_total_rows": total_rows,
            "phase3c_total_file_size_bytes": total_bytes,
            "phase3c_source_scopes": source_scopes,
            "phase3c_roles": phase3c_roles,
            "phase3c_has_multiindex_columns": has_multiindex_runtime,
            "stitch_validation_status": stitch_validation_status,
            "stitch_source_table_name": stitch_row.get("source_table_name", ""),
            "stitch_source_fragment_count": stitch_row.get("source_fragment_count", ""),
            "stitch_recreated_rows": stitch_row.get("recreated_rows", ""),
            "stitch_final_rows": stitch_row.get("final_rows", ""),
            "coverage_status": coverage_status,
        })
    return pd.DataFrame(coverage_rows), unmatched_manifest_df


def summarize_output_schema_coverage(coverage_df: pd.DataFrame,
                                     unmatched_manifest_df: pd.DataFrame) -> dict[str, Any]:
    """Summarize registry coverage for runtime logs and manifest JSON files."""

    if coverage_df.empty:
        return {
            "schema_version": OUTPUT_SCHEMA_COVERAGE_SCHEMA_VERSION,
            "generated_utc": _utc_now_iso(),
            "registry_table_count": 0,
            "unmatched_phase3c_manifest_count": int(len(unmatched_manifest_df)),
        }
    return {
        "schema_version": OUTPUT_SCHEMA_COVERAGE_SCHEMA_VERSION,
        "generated_utc": _utc_now_iso(),
        "registry_version": OUTPUT_SCHEMA_REGISTRY_VERSION,
        "expected_current_registry_count": EXPECTED_CURRENT_REGISTRY_COUNT,
        "registry_table_count": int(len(coverage_df)),
        "phase3c_present_table_spec_count": int((coverage_df["phase3c_artifact_count"] > 0).sum()),
        "phase3c_missing_table_spec_count": int((coverage_df["phase3c_artifact_count"] == 0).sum()),
        "validated_stitched_final_count": int(coverage_df["coverage_status"].eq("validated_stitched_final").sum()),
        "unmatched_phase3c_manifest_count": int(len(unmatched_manifest_df)),
        "coverage_status_counts": dict(Counter(coverage_df["coverage_status"])),
        "artifact_scope_counts": dict(Counter(coverage_df["artifact_scope"])),
        "table_family_counts": dict(Counter(coverage_df["table_family"])),
        "stitch_method_counts": dict(Counter(coverage_df["stitch_method"])),
        "validation_status_counts": dict(Counter(coverage_df["validation_status"])),
        "retention_policy_counts": dict(Counter(coverage_df["retention_policy"])),
    }


def write_output_schema_coverage_report(phase3c_manifest_df: pd.DataFrame,
                                        stitch_validation_df: pd.DataFrame,
                                        output_dir: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    """Write Phase 3C registry coverage CSV/JSON artifacts."""

    output_dir = Path(output_dir)
    coverage_df, unmatched_manifest_df = build_output_schema_coverage_report(
        phase3c_manifest_df=phase3c_manifest_df,
        stitch_validation_df=stitch_validation_df,
    )
    summary = summarize_output_schema_coverage(coverage_df, unmatched_manifest_df)
    coverage_path = output_dir.joinpath("output_schema_coverage.csv")
    unmatched_path = output_dir.joinpath("output_schema_unmatched_phase3c_manifest.csv")
    summary_path = output_dir.joinpath("output_schema_coverage_summary.json")
    coverage_df.to_csv(coverage_path, index=False)
    unmatched_manifest_df.to_csv(unmatched_path, index=False)
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    return coverage_path, unmatched_path, summary_path, summary


def _human_use_text(spec: OutputTableSpec) -> str:
    if spec.validation_status == "needs_live_phase3c_validation":
        return "Route is wired, but keep using the legacy final table until a Phase 3C run validates the stitched patient fragments."
    if spec.validation_status == "needs_phase3d_route":
        return "Use the legacy final table for now; do not promote this to the patient-scoped surface until a patient-fragment route validates."
    if spec.retention_policy == "reimplement_later":
        return "Keep for validation, but prefer a clean recomputation service before GUI or downstream reliance."
    if spec.retention_policy == "retain_validation_only":
        return "Useful as a validation source; avoid treating it as the final public table shape."
    if spec.artifact_scope in {"patient", "biopsy"}:
        return "Use as a patient-level source fragment; join through the canonical primary key before cohort aggregation."
    if spec.artifact_scope == "cohort" and spec.stitch_method == "concat_rows":
        return "Use as a cohort table after patient fragments have been stitched and validated."
    if spec.artifact_scope == "cohort" and spec.stitch_method in {"aggregate_from_long_form", "join_derived"}:
        return "Use after the explicit aggregate or join builder is validated against legacy output."
    if spec.artifact_scope == "run_metadata":
        return "Use as run-level metadata; do not join to biopsy or voxel observations without an explicit key."
    return "Use according to the validation and retention fields."


def _implementation_next_step(spec: OutputTableSpec) -> str:
    if spec.validation_status == "needs_live_phase3c_validation":
        return "Run Phase 3C with the new patient fragments, confirm the stitch pair matches, then mark the cohort spec validated."
    if spec.validation_status == "needs_phase3d_route":
        return "Add a patient-fragment builder/store route, export it through Phase 3C, stitch rows, and validate against the legacy final table."
    if spec.validation_status == "needs_aggregation_builder":
        return "Write or identify the aggregate builder, validate from source fragments, then update the registry status."
    if spec.retention_policy == "reimplement_later":
        return "Freeze as legacy during validation; compare against a clean recomputation path before promotion."
    if spec.validation_status == "validated_phase3c":
        return "Keep the registry entry current and preserve this validation gate when the phase-specific surface is renamed."
    if spec.validation_status == "metadata_only":
        return "Move toward manifest/config metadata if this remains a run-level table."
    return "Review lineage, row grain, validation evidence, and downstream consumers before changing retention."


def build_output_schema_data_dictionary(registry: OutputSchemaRegistry | None = None) -> pd.DataFrame:
    """Build a human-readable data dictionary from registry specs.

    This dataframe is intentionally wider than a short summary table. It is the
    audit view a human should read when deciding how a table is produced, joined,
    validated, retained, and eventually exposed to the GUI or downstream analyses.
    """

    registry = OutputSchemaRegistry() if registry is None else registry
    rows: list[dict[str, Any]] = []
    for spec in registry.specs:
        rows.append({
            "table_id": spec.table_id,
            "legacy_table_name": spec.legacy_table_name,
            "legacy_output_section": spec.legacy_output_section,
            "artifact_scope": spec.artifact_scope,
            "table_family": spec.table_family,
            "row_grain": spec.row_grain,
            "canonical_primary_key": " | ".join(spec.canonical_primary_key),
            "join_keys": " | ".join(spec.join_keys),
            "legacy_key_columns": " | ".join(spec.legacy_key_columns),
            "source_stage": spec.source_stage,
            "source_fragment_table_id": spec.source_fragment_table_id,
            "stitch_or_build_method": spec.stitch_method,
            "aggregation_builder": spec.aggregation_builder,
            "validation_status": spec.validation_status,
            "retention_policy": spec.retention_policy,
            "downstream_usage": spec.downstream_usage,
            "storage_format": spec.storage_format,
            "file_extension": spec.file_extension,
            "columns_policy": spec.columns_policy,
            "has_multiindex_columns": "unknown_until_runtime" if spec.has_multiindex_columns is None else str(spec.has_multiindex_columns),
            "match_mode": spec.match_mode,
            "how_to_use": _human_use_text(spec),
            "implementation_next_step": _implementation_next_step(spec),
            "notes": spec.notes,
        })
    return pd.DataFrame(rows)


def _markdown_value(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").strip()
    return text or "not specified"


def render_output_schema_data_dictionary_markdown(data_dictionary_df: pd.DataFrame,
                                                  generated_utc: str | None = None) -> str:
    """Render the schema data dictionary as a reviewable Markdown document."""

    if generated_utc is None:
        generated_utc = _utc_now_iso()

    lines = [
        "# Output Schema Data Dictionary",
        "",
        f"Generated UTC: {generated_utc}",
        "",
        "This file is generated from the output schema registry. Update the registry, not this generated view, when table contracts change.",
        "",
        "## How To Read This",
        "",
        "Each section describes one durable output table. The important audit questions are: what one row represents, which canonical columns identify that row, where the table comes from, how it is stitched or rebuilt, whether it is validated, and whether it should remain in the core output surface.",
        "",
    ]

    for row in data_dictionary_df.to_dict(orient="records"):
        lines.extend([
            f"## {_markdown_value(row.get('table_id'))}",
            "",
            f"- Legacy table: `{_markdown_value(row.get('legacy_table_name'))}`",
            f"- Legacy location: `{_markdown_value(row.get('legacy_output_section'))}` as `{_markdown_value(row.get('file_extension'))}`",
            f"- Scope/family: `{_markdown_value(row.get('artifact_scope'))}` / `{_markdown_value(row.get('table_family'))}`",
            f"- Row grain: `{_markdown_value(row.get('row_grain'))}`",
            f"- Canonical primary key: `{_markdown_value(row.get('canonical_primary_key'))}`",
            f"- Legacy key columns: `{_markdown_value(row.get('legacy_key_columns'))}`",
            f"- Source stage: `{_markdown_value(row.get('source_stage'))}`",
            f"- Source fragment table ID: `{_markdown_value(row.get('source_fragment_table_id'))}`",
            f"- Stitch/build method: `{_markdown_value(row.get('stitch_or_build_method'))}`",
            f"- Aggregation builder: `{_markdown_value(row.get('aggregation_builder'))}`",
            f"- Validation status: `{_markdown_value(row.get('validation_status'))}`",
            f"- Retention policy: `{_markdown_value(row.get('retention_policy'))}`",
            f"- Downstream usage: `{_markdown_value(row.get('downstream_usage'))}`",
            f"- Storage/columns: `{_markdown_value(row.get('storage_format'))}`, `{_markdown_value(row.get('columns_policy'))}`, MultiIndex columns `{_markdown_value(row.get('has_multiindex_columns'))}`",
            f"- How to use: {_markdown_value(row.get('how_to_use'))}",
            f"- Implementation next step: {_markdown_value(row.get('implementation_next_step'))}",
            f"- Notes: {_markdown_value(row.get('notes'))}",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def write_output_schema_data_dictionary(output_dir: Path,
                                        registry: OutputSchemaRegistry | None = None) -> tuple[Path, Path]:
    """Write CSV and Markdown data-dictionary views of the registry."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dictionary_df = build_output_schema_data_dictionary(registry)
    csv_path = output_dir.joinpath("output_schema_data_dictionary.csv")
    markdown_path = output_dir.joinpath("output_schema_data_dictionary.md")
    data_dictionary_df.to_csv(csv_path, index=False)
    markdown_path.write_text(
        render_output_schema_data_dictionary_markdown(data_dictionary_df),
        encoding="utf-8",
    )
    return csv_path, markdown_path
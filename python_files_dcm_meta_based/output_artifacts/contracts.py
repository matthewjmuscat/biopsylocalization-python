from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd


OUTPUT_TABLE_CONTRACT_SCHEMA_VERSION = "phase2_output_table_contracts_v1"
UNSTABLE_LEGACY_KEY_COLUMNS = "Bx refnum; Relative DIL ref num; Structure ref num"
CANONICAL_KEY_POLICY_NOTE = (
    "Do not rely on refnum columns for uniqueness. Canonical biopsy identity is "
    "Patient ID + Bx index. Canonical non-biopsy structure identity is "
    "Patient ID + structure/ref type + structure index. Refnum columns are legacy/source attributes."
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _path_parts(relative_path: str) -> tuple[str, ...]:
    return tuple(part for part in Path(str(relative_path)).parts if part not in ("", "."))


def _strip_patient_prefix(filename_stem: str, patient_uid: str | None) -> str:
    if patient_uid and filename_stem.startswith(f"{patient_uid}-"):
        return filename_stem[len(patient_uid) + 1:]
    return filename_stem


def _parse_biopsy_index_from_dir(relative_path: str) -> str:
    parts = _path_parts(relative_path)
    if len(parts) < 4 or parts[0] != "Output CSVs" or parts[1] != "MC simulation":
        return ""
    biopsy_dir = parts[3]
    return biopsy_dir.split("-", 1)[0]


def _normalize_legacy_table_name(row: pd.Series) -> str:
    relative_path = str(row.get("relative_path", ""))
    output_section = str(row.get("output_section", ""))
    legacy_name = str(row.get("legacy_dataframe_name", ""))
    patient_uid = str(row.get("patient_uid", ""))
    parts = _path_parts(relative_path)
    if output_section != "Output CSVs/MC simulation" or len(parts) < 4 or patient_uid == "Global":
        return legacy_name

    biopsy_index = _parse_biopsy_index_from_dir(relative_path)
    if biopsy_index == "":
        return legacy_name

    stem = Path(relative_path).stem
    stem = _strip_patient_prefix(stem, patient_uid)
    marker = f"-{biopsy_index}-"
    if marker in stem:
        return stem.rsplit(marker, 1)[1]
    return legacy_name


def _source_stage(output_section: str, table_name: str) -> str:
    if output_section == "Output CSVs/Preprocessing":
        return "preprocessing"
    if output_section == "Output CSVs/MC simulation":
        if "MR ADC" in table_name or table_name.startswith("MR -"):
            return "mc_mr"
        if "DVH" in table_name or "dose" in table_name.lower() or "Dosimetry" in table_name:
            return "mc_dosimetry"
        if "Tissue" in table_name or "tissue" in table_name:
            return "mc_tissue_containment"
        return "mc_simulation"
    if output_section == "Output CSVs/Cohort":
        if "MR ADC" in table_name:
            return "cohort_mr_postprocessing"
        if "DVH" in table_name or "dosimetry" in table_name.lower():
            return "cohort_dosimetry_postprocessing"
        if "tissue" in table_name.lower() or "Tissue class" in table_name:
            return "cohort_tissue_postprocessing"
        if "Uncertainties" in table_name:
            return "run_uncertainty_configuration"
        return "cohort_postprocessing"
    if output_section == "manifests":
        return "input_manifest"
    return "run_metadata_or_other"


def _builder_or_source(table_name: str, output_section: str) -> str:
    exact = {
        "Selected structures": "biopsy_localization_convex_main.py selected-structure discovery dataframe assembly",
        "Structure preprocessing timings": "preprocessing.structure_processing.non_biopsy_structure_processing.preprocess_non_biopsy_structure(...) timing output",
        "Nearest DILs info dataframe": "dataframe_builders.bx_nearest_dils_dataframe_builder(...)",
        "Biopsy basic spatial features dataframe": "dataframe_builders.biopsy_basic_spatial_features_information_dataframe_builder(...)",
        "Simulated biopsy preparation dataframe": "preprocessing.biopsy_processing.simulated_biopsy_preparation.simulated_biopsy_preparation_dataframe_builder(...)",
        "Biopsy optimization - DIL centroids optimal targeting dataframe": "dataframe_builders.dil_optimization_results_dataframe_builder(...)",
        "Biopsy optimization - Optimal DIL targeting dataframe": "dataframe_builders.dil_optimization_results_dataframe_builder(...)",
        "Biopsy optimization - Optimal DIL targeting entire lattice dataframe": "dataframe_builders.dil_optimization_results_dataframe_builder(...)",
        "Biopsy optimization - Guidance-map firing depth recommendations dataframe": "guidance_maps.planning.precompute_guidance_map_firing_depth_recommendations_for_run(...)",
        "Biopsy optimization - Target DIL optimizer v2 summary dataframe": "biopsy_optimizer.v2.live_integration.run_target_dil_optimizer_v2_for_live_simulated_family(...)",
        "Biopsy optimization - Target DIL optimizer v2 ranked candidates dataframe": "biopsy_optimizer.v2.live_integration.run_target_dil_optimizer_v2_for_live_simulated_family(...)",
        "Biopsy optimization - Target DIL optimizer v2 tested candidates dataframe": "biopsy_optimizer.v2.live_integration.run_target_dil_optimizer_v2_for_live_simulated_family(...)",
        "Cohort: Nearest DILs to each biopsy": "dataframe_builders.bx_nearest_dils_dataframe_builder(...)",
        "Cohort: Biopsy basic spatial features dataframe": "dataframe_builders.biopsy_basic_spatial_features_information_dataframe_builder(...)",
        "Cohort: Simulated biopsy preparation dataframe": "dataframe_builders.cohort_simulated_biopsy_preparation_dataframe_builder(...)",
        "Cohort: Guidance-map firing depth recommendations dataframe": "guidance_maps.planning.precompute_guidance_map_firing_depth_recommendations_for_run(...)",
        "Cohort: 3D radiomic features all OAR and DIL structures": "dataframe_builders.cohort_structure_features_dataframe_builder(...)",
        "Cohort: All MC structure transformation values": "dataframe_builders.all_structure_shifts_by_trial_dataframe_builder(...)",
        "Cohort: structure specific mc results": "dataframe_builders.cohort_and_multi_biopsy_mc_structure_specific_pt_wise_results_dataframe_builder(...)",
        "Cohort: sum-to-one mc results": "dataframe_builders.cohort_and_multi_biopsy_mc_sum_to_one_pt_wise_results_dataframe_builder(...)",
        "Cohort: global sum-to-one mc results": "dataframe_builders.cohort_mc_sum_to_one_global_scores_dataframe_builder(...)",
        "Cohort: tissue class global scores (structure)": "dataframe_builders.global_scores_by_specific_structure_dataframe_builder(...)",
        "Cohort: tissue volume above threshold": "dataframe_builders.tissue_volume_threshold_dataframe_builder_NEW(...)",
        "Cohort: DIL global tissue scores and DIL features": "dataframe_builders.bx_global_score_to_target_dil_3d_radiomic_features_dataframe_builder(...)",
        "Cohort: Tissue class - distances global results": "dataframe_builders.cohort_relative_structure_distances_dataframe_builder(...)",
        "Cohort: Tissue class - distances pt-wise results": "dataframe_builders.cohort_relative_structure_distances_dataframe_builder(...)",
        "Cohort: Tissue class - distances voxel-wise results": "dataframe_builders.cohort_relative_structure_distances_dataframe_builder(...)",
        "Cohort: Global dosimetry by voxel": "dataframe_builders.global_dosimetry_by_voxel_values_dataframe_builder_v3_generalized(...)",
        "Cohort: Global dosimetry (NEW)": "dataframe_builders.global_dosimetry_by_biopsy_dataframe_builder_NEW_multiindex_df(...)",
        "Cohort: Bx DVH metrics": "dataframe_builders.dvh_metrics_dataframe_builder_sp_biopsy(...)",
        "Cohort: Bx DVH metrics (generalized)": "dataframe_builders.dvh_metrics_calculator_and_dataframe_builder_cohort(...)",
        "Cohort: Global MR ADC statistics": "dataframe_builders.global_mr_values_dataframe_builder(...)",
        "Cohort: Global by voxel MR ADC statistics": "dataframe_builders.global_mr_by_voxel_values_dataframe_builder_ALTERNATE(...)",
        "Cohort: Per sample point prostate double sextant classification": "preprocessing.biopsy_processing.biopsy_double_sextant.biopsy_double_sextant_processer(...)",
        "Cohort: Per voxel prostate double sextant classification": "preprocessing.biopsy_processing.biopsy_double_sextant.biopsy_double_sextant_processer(...)",
        "Cohort: Simulated biopsy planned vs realized centroid variation validation": "preprocessing.biopsy_processing.biopsy_centroid_variation_validation.validate_simulated_biopsy_planned_vs_realized_centroid_variation(...)",
        "All MC structure transformation values": "dataframe_builders.all_structure_shifts_by_trial_dataframe_builder(...)",
        "Tissue class - Global tissue by structure statistics": "dataframe_builders.global_scores_by_specific_structure_dataframe_builder(...)",
        "Tissue class - Pt wise structure specific results": "dataframe_builders.cohort_and_multi_biopsy_mc_structure_specific_pt_wise_results_dataframe_builder(...)",
        "Tissue class - containment and distances (light) results": "dataframe_builders.cohort_containment_results_and_distances_dataframe_builder_light(...)",
        "Tissue class - distances global results": "dataframe_builders.cohort_relative_structure_distances_dataframe_builder(...)",
        "Tissue class - distances pt-wise results": "dataframe_builders.cohort_relative_structure_distances_dataframe_builder(...)",
        "Tissue class - distances voxel-wise results": "dataframe_builders.cohort_relative_structure_distances_dataframe_builder(...)",
        "Tissue class - sum-to-one mc results": "dataframe_builders.cohort_and_multi_biopsy_mc_sum_to_one_pt_wise_results_dataframe_builder(...)",
        "MR - ADC - summary statistics by structure dataframe": "dataframe_builders.dataframe_mr_summary_statistics(...) stored during non-biopsy structure preprocessing",
        "Uncertainties dataframe (final)": "uncertainty_file_writer / uncertainty configuration output",
        "Uncertainties dataframe (unedited)": "uncertainty_file_writer / uncertainty configuration output",
        "input_case_manifest": "input_data.write_input_manifest_files(...)",
        "input_dicom_manifest": "input_data.write_input_manifest_files(...)",
    }
    if table_name in exact:
        return exact[table_name]
    if table_name == "Cumulative DVH by MC trial":
        return "dataframe_builders.cumulative_dvh_dataframe_all_mc_trials_dataframe_builder_v2(...)"
    if table_name == "Differential DVH by MC trial":
        return "dataframe_builders.differential_dvh_dataframe_all_mc_trials_dataframe_builder_v2(...)"
    if table_name == "Point-wise dose output by MC trial number":
        return "dataframe_builders.all_dose_data_by_trial_and_pt_from_dataframe_builder_and_voxelizer_v4(...)"
    if table_name == "Voxel-wise dose output by MC trial number":
        return "dataframe_builders.all_dose_data_by_trial_and_pt_from_dataframe_builder_and_voxelizer_v4(...)"
    if table_name == "Point-wise MR ADC output by MC trial number":
        return "dataframe_builders.all_mr_data_by_trial_and_pt_from_dataframe_builder_and_voxelizer_v4(...)"
    if table_name == "Tissue volume above threshold":
        return "dataframe_builders.tissue_volume_threshold_dataframe_builder_NEW(...)"
    if table_name in {"DVH metrics", "DVH metrics (Dx, Vx) statistics"}:
        return "dataframe_builders.dvh_metrics_dataframe_builder_sp_biopsy(...) or per-patient MC dataframe dict"
    if table_name in {"Dosimetry - Global dosimetry (NEW)", "Dosimetry - Global dosimetry by voxel statistics"}:
        return "dataframe_builders global dosimetry builders stored in per-patient MC dataframe dict"
    if table_name in {"MR - Global MR ADC statistics", "MR - Global by voxel MR ADC statistics"}:
        return "dataframe_builders global MR builders stored in per-patient MC dataframe dict"
    if output_section == "Output CSVs/Preprocessing" and table_name.startswith("Biopsy optimization - Cumulative projection"):
        return "biopsy_optimizer.v1.biopsy_optimizer_module_v1(...) cumulative projection output"
    if table_name.startswith("uncertainties_file_auto_generated"):
        return "uncertainty_file_writer generated uncertainty configuration sidecar"
    return "needs_builder_trace"


def _canonical_primary_key(table_name: str, output_section: str) -> str:
    if table_name in {"input_case_manifest", "input_dicom_manifest"}:
        return "manifest row identity; input path or patient UID plus selected role"
    if table_name.startswith("uncertainties_file_auto_generated") or table_name.startswith("Uncertainties dataframe"):
        return "uncertainty parameter name or row index; run-level configuration identity"
    if table_name == "Selected structures":
        return "Patient ID + Struct ref type + Index number"
    if table_name == "Structure preprocessing timings":
        return "Patient ID + Structure ref type + Structure index + preprocessing phase/subphase"
    if table_name == "MR - ADC - summary statistics by structure dataframe":
        return "Patient ID + Structure type + Structure index + MR statistic identity"
    if "Nearest DIL" in table_name:
        return "Patient ID + Bx index + Relative struct type + Relative DIL index"
    if "Biopsy basic spatial" in table_name:
        return "Patient ID + Bx index"
    if "Simulated biopsy preparation" in table_name:
        return "Patient ID + Bx index; fallback pre-realization key Patient ID + Target structure type + Target structure index + Multiplicity index"
    if "Guidance-map firing depth" in table_name:
        return "Firing depth row UID; fallback Patient ID + Relative struct type + Relative struct index + Candidate hole rank + Firing depth row index"
    if "Target DIL optimizer v2 summary" in table_name:
        return "Patient ID + Bx index"
    if "Target DIL optimizer v2 ranked" in table_name:
        return "Patient ID + Bx index + Candidate index global or Candidate rank"
    if "Target DIL optimizer v2 tested" in table_name:
        return "Patient ID + Bx index + Candidate index global + Stage name or Stage index"
    if "DIL centroids optimal" in table_name or "Optimal DIL targeting" in table_name:
        return "Patient ID + Relative DIL index; entire-lattice outputs additionally require test-location row identity"
    if table_name.startswith("Biopsy optimization - Cumulative projection"):
        return "Patient ID + Relative DIL index + projection/test-location row identity"
    if "planned vs realized centroid variation validation" in table_name:
        return "Patient ID + Bx index + validation comparison field"
    if table_name in {"Cumulative DVH by MC trial", "Differential DVH by MC trial"}:
        return "Patient ID + Bx index + MC trial num + dose-bin identity"
    if table_name == "Point-wise dose output by MC trial number":
        return "Patient ID + Bx index + MC trial num + point index"
    if table_name == "Point-wise MR ADC output by MC trial number":
        return "Patient ID + Bx index + MC trial num + point index"
    if table_name == "Voxel-wise dose output by MC trial number":
        return "Patient ID + Bx index + MC trial num + Voxel index"
    if "tissue volume above threshold" in table_name.lower():
        return "Patient ID + Bx index + tissue/structure identity + threshold identity"
    if "Global tissue by structure statistics" in table_name:
        return "Patient ID + Bx index + Relative structure type + Relative structure index"
    if "Pt wise structure specific" in table_name:
        return "Patient ID + Bx index + MC trial num + point index + Relative structure type + Relative structure index"
    if "containment and distances (light)" in table_name:
        return "Patient ID + Bx index + MC trial num + point/voxel identity + Relative structure type + Relative structure index"
    if "All MC structure transformation" in table_name:
        return "Patient ID + Structure type + Structure index + Trial"
    if "radiomic features" in table_name:
        return "Patient ID + Structure type + Structure index"
    if "structure specific mc results" in table_name:
        return "Patient ID + Bx index + MC trial num + point/voxel identity + structure identity"
    if "sum-to-one" in table_name:
        return "Patient ID + Bx index + MC trial num + point/voxel identity + tissue class"
    if "tissue class global scores" in table_name or "DIL global tissue scores" in table_name:
        return "Patient ID + Bx index + Relative struct type + Relative structure index"
    if "distances global" in table_name:
        return "Patient ID + Bx index + Relative struct type + Relative structure index"
    if "distances pt-wise" in table_name:
        return "Patient ID + Bx index + MC trial num + point index + Relative struct type + Relative structure index"
    if "distances voxel-wise" in table_name:
        return "Patient ID + Bx index + MC trial num + Voxel index + Relative struct type + Relative structure index"
    if "dosimetry by voxel" in table_name or "by voxel" in table_name:
        return "Patient ID + Bx index + Voxel index + dose/MR statistic identity"
    if "Global dosimetry" in table_name or "MR ADC statistics" in table_name:
        return "Patient ID + Bx index + statistic identity"
    if "DVH metrics" in table_name:
        return "Patient ID + Bx index + metric identity"
    if "double sextant" in table_name:
        return "Patient ID + Bx index + Voxel index; per-sample table additionally includes sample point identity"
    return "needs_phase2_key_review"


def _stitch_key(table_name: str, output_section: str) -> str:
    if output_section == "Output CSVs/Preprocessing":
        return "Patient ID"
    if output_section == "Output CSVs/MC simulation":
        if table_name in {
            "Cumulative DVH by MC trial",
            "Differential DVH by MC trial",
            "Point-wise dose output by MC trial number",
            "Point-wise MR ADC output by MC trial number",
            "Voxel-wise dose output by MC trial number",
            "Tissue volume above threshold",
        }:
            return "Patient ID + Bx index"
        return "Patient ID"
    if output_section == "Output CSVs/Cohort":
        if table_name.startswith("Uncertainties"):
            return "run-level; not patient-stitchable"
        return "Patient ID, with table-specific biopsy/structure/trial keys"
    if output_section == "manifests":
        return "run-level manifest; not patient-stitchable"
    return "needs_phase2_stitch_review"


def _proposed_lifetime_class(table_name: str, output_section: str, current_output_class: str) -> str:
    if table_name.startswith("Uncertainties") or table_name.startswith("uncertainties_file_auto_generated"):
        return "run_metadata"
    if output_section == "manifests":
        return "run_metadata"
    if "DVH metrics" in table_name:
        return "downstream_calculable_or_optional_derived"
    if table_name == "Cumulative DVH by MC trial" or table_name == "Differential DVH by MC trial":
        return "patient_derived_stitchable"
    if output_section == "Output CSVs/Preprocessing":
        return "patient_appendable"
    if output_section == "Output CSVs/MC simulation":
        return "patient_or_biopsy_fragment_stitchable" if current_output_class == "patient_derived_table_requiring_stitch_only" else "patient_appendable"
    if output_section == "Output CSVs/Cohort":
        if table_name in {
            "Cohort: global sum-to-one mc results",
            "Cohort: tissue class global scores (structure)",
            "Cohort: tissue volume above threshold",
            "Cohort: DIL global tissue scores and DIL features",
            "Cohort: Tissue class - distances global results",
            "Cohort: Global dosimetry (NEW)",
            "Cohort: Global MR ADC statistics",
            "Cohort: Bx DVH metrics",
            "Cohort: Bx DVH metrics (generalized)",
        }:
            return "final_stage_after_patient_fragments"
        return "cohort_named_but_likely_stitchable"
    return current_output_class


def _pruning_assessment(table_name: str) -> tuple[str, str]:
    if table_name in {"Cohort: Bx DVH metrics", "DVH metrics"}:
        return (
            "deprecated_candidate",
            "Legacy/non-generalized DVH metrics path appears superseded by generalized DVH metrics and can likely be recalculated downstream.",
        )
    if table_name in {"Cohort: Bx DVH metrics (generalized)", "DVH metrics (Dx, Vx) statistics"}:
        return (
            "downstream_calculable_candidate",
            "DVH metrics are derived from dose/DVH source tables and may not need to be emitted by the core localization pipeline long term.",
        )
    if table_name in {"Cumulative DVH by MC trial", "Differential DVH by MC trial"}:
        return (
            "derived_heavy_output_review",
            "DVH curves are derived from dose distributions; keep until downstream recalculation contract and validation are in place.",
        )
    if table_name.startswith("Uncertainties dataframe"):
        return (
            "metadata_keep_or_manifest_migrate",
            "Run uncertainty configuration should remain auditable, but may move from cohort CSV output to manifest/config metadata.",
        )
    return "retain_for_validation", "Keep until full-cohort validation and downstream replacement plan are complete."


def _contract_confidence(builder_or_source: str, key: str) -> str:
    if builder_or_source == "needs_builder_trace" or key == "needs_phase2_key_review":
        return "low"
    if "fallback" in key or "or" in key:
        return "medium"
    return "high"


def build_output_table_contracts(inventory_df: pd.DataFrame) -> pd.DataFrame:
    table_df = inventory_df[inventory_df["artifact_kind"].eq("table")].copy()
    if table_df.empty:
        return pd.DataFrame(columns=[
            "schema_version",
            "normalized_table_name",
            "output_section",
            "file_extension",
            "current_output_class",
            "proposed_lifetime_class",
            "source_stage",
            "builder_or_source",
            "current_file_count",
            "canonical_primary_key",
            "stitch_key",
            "unstable_legacy_key_columns",
            "key_policy_notes",
            "pruning_assessment",
            "pruning_notes",
            "contract_confidence",
        ])

    table_df["normalized_table_name"] = table_df.apply(_normalize_legacy_table_name, axis=1)
    group_columns = ["normalized_table_name", "output_section", "file_extension", "output_class"]
    rows: list[dict[str, Any]] = []
    for group_key, group in table_df.groupby(group_columns, dropna=False):
        table_name, output_section, file_extension, current_output_class = group_key
        builder_or_source = _builder_or_source(str(table_name), str(output_section))
        canonical_primary_key = _canonical_primary_key(str(table_name), str(output_section))
        pruning_assessment, pruning_notes = _pruning_assessment(str(table_name))
        rows.append({
            "schema_version": OUTPUT_TABLE_CONTRACT_SCHEMA_VERSION,
            "normalized_table_name": table_name,
            "output_section": output_section,
            "file_extension": file_extension,
            "current_output_class": current_output_class,
            "proposed_lifetime_class": _proposed_lifetime_class(
                str(table_name),
                str(output_section),
                str(current_output_class),
            ),
            "source_stage": _source_stage(str(output_section), str(table_name)),
            "builder_or_source": builder_or_source,
            "current_file_count": int(len(group)),
            "canonical_primary_key": canonical_primary_key,
            "stitch_key": _stitch_key(str(table_name), str(output_section)),
            "unstable_legacy_key_columns": UNSTABLE_LEGACY_KEY_COLUMNS,
            "key_policy_notes": CANONICAL_KEY_POLICY_NOTE,
            "pruning_assessment": pruning_assessment,
            "pruning_notes": pruning_notes,
            "contract_confidence": _contract_confidence(builder_or_source, canonical_primary_key),
        })
    return pd.DataFrame(rows).sort_values(["output_section", "normalized_table_name"]).reset_index(drop=True)


def summarize_output_table_contracts(contracts_df: pd.DataFrame) -> dict[str, Any]:
    if contracts_df.empty:
        return {
            "schema_version": OUTPUT_TABLE_CONTRACT_SCHEMA_VERSION,
            "generated_utc": _utc_now_iso(),
            "contract_count": 0,
            "proposed_lifetime_class_counts": {},
            "pruning_assessment_counts": {},
            "contract_confidence_counts": {},
        }
    return {
        "schema_version": OUTPUT_TABLE_CONTRACT_SCHEMA_VERSION,
        "generated_utc": _utc_now_iso(),
        "contract_count": int(len(contracts_df)),
        "proposed_lifetime_class_counts": dict(Counter(contracts_df["proposed_lifetime_class"])),
        "pruning_assessment_counts": dict(Counter(contracts_df["pruning_assessment"])),
        "contract_confidence_counts": dict(Counter(contracts_df["contract_confidence"])),
        "needs_builder_trace_count": int((contracts_df["builder_or_source"] == "needs_builder_trace").sum()),
        "needs_key_review_count": int((contracts_df["canonical_primary_key"] == "needs_phase2_key_review").sum()),
    }


def write_output_table_contracts(contracts_df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contracts_path = output_dir.joinpath("output_table_contracts.csv")
    summary_path = output_dir.joinpath("output_table_contracts_summary.json")
    contracts_df.to_csv(contracts_path, index=False)
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summarize_output_table_contracts(contracts_df), file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    return contracts_path, summary_path
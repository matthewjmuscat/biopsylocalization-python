# Phase 3D Dataframe Status Audit

Source inventory: validation_outputs/output_inventory/MC_sim_out-_Date-May-18-2026_Time-17_30_14/

- total_contracts: 63
- expected_not_simple_concat: 18
- stitching_category_counts: {'patient_fragment_source_table': 30, 'simple_concat_validated_or_in_phase3c_registry': 11, 'simple_concat_missing_fragment_route': 4, 'aggregation': 3, 'metadata_not_concat': 3, 'metadata_manifest': 2, 'aggregation_from_per_biopsy_raw': 2, 'derived_heavy_maybe_recompute': 2, 'derived_downstream_calculable': 2, 'derived_deprecated': 2, 'aggregation_or_join': 1, 'aggregation_from_distance_raw': 1}
- promotion_status_counts: {'phase3c_source_surface': 30, 'needs_explicit_builder_or_pruning_decision': 15, 'validate_current_phase3c_run_then_promote': 11, 'needs_patient_fragment_route': 4, 'manifest_config_surface': 3}
- removal_assessment_counts: {'retain_for_validation': 46, 'no_downstream_reference_found_keep_until_validation': 9, 'deprecated_candidate': 2, 'downstream_calculable_candidate': 2, 'derived_heavy_output_review': 2, 'metadata_keep_or_manifest_migrate': 2}
- downstream_referenced_contracts: 27

## Expected Not Simple Concat

| normalized_table_name                                            | stitching_category              | promotion_status                           | removal_assessment                                  |   downstream_reference_count |
|:-----------------------------------------------------------------|:--------------------------------|:-------------------------------------------|:----------------------------------------------------|-----------------------------:|
| Cohort: global sum-to-one mc results                             | aggregation                     | needs_explicit_builder_or_pruning_decision | retain_for_validation                               |                            3 |
| Cohort: tissue class global scores (structure)                   | aggregation                     | needs_explicit_builder_or_pruning_decision | no_downstream_reference_found_keep_until_validation |                            0 |
| Cohort: tissue volume above threshold                            | aggregation                     | needs_explicit_builder_or_pruning_decision | no_downstream_reference_found_keep_until_validation |                            0 |
| Cohort: Tissue class - distances global results                  | aggregation_from_distance_raw   | needs_explicit_builder_or_pruning_decision | retain_for_validation                               |                            7 |
| Cohort: Global MR ADC statistics                                 | aggregation_from_per_biopsy_raw | needs_explicit_builder_or_pruning_decision | no_downstream_reference_found_keep_until_validation |                            0 |
| Cohort: Global dosimetry (NEW)                                   | aggregation_from_per_biopsy_raw | needs_explicit_builder_or_pruning_decision | retain_for_validation                               |                            2 |
| Cohort: DIL global tissue scores and DIL features                | aggregation_or_join             | needs_explicit_builder_or_pruning_decision | no_downstream_reference_found_keep_until_validation |                            0 |
| Cohort: Bx DVH metrics                                           | derived_deprecated              | needs_explicit_builder_or_pruning_decision | deprecated_candidate                                |                            1 |
| DVH metrics                                                      | derived_deprecated              | needs_explicit_builder_or_pruning_decision | deprecated_candidate                                |                            8 |
| Cohort: Bx DVH metrics (generalized)                             | derived_downstream_calculable   | needs_explicit_builder_or_pruning_decision | downstream_calculable_candidate                     |                            1 |
| DVH metrics (Dx, Vx) statistics                                  | derived_downstream_calculable   | needs_explicit_builder_or_pruning_decision | downstream_calculable_candidate                     |                            0 |
| Cumulative DVH by MC trial                                       | derived_heavy_maybe_recompute   | needs_explicit_builder_or_pruning_decision | derived_heavy_output_review                         |                            2 |
| Differential DVH by MC trial                                     | derived_heavy_maybe_recompute   | needs_explicit_builder_or_pruning_decision | derived_heavy_output_review                         |                            2 |
| Uncertainties dataframe (final)                                  | metadata_manifest               | needs_explicit_builder_or_pruning_decision | metadata_keep_or_manifest_migrate                   |                            0 |
| Uncertainties dataframe (unedited)                               | metadata_manifest               | needs_explicit_builder_or_pruning_decision | metadata_keep_or_manifest_migrate                   |                            0 |
| input_case_manifest                                              | metadata_not_concat             | manifest_config_surface                    | retain_for_validation                               |                            0 |
| input_dicom_manifest                                             | metadata_not_concat             | manifest_config_surface                    | retain_for_validation                               |                            0 |
| uncertainties_file_auto_generated Date-May-18-2026 Time-17,55,36 | metadata_not_concat             | manifest_config_surface                    | retain_for_validation                               |                            0 |

## Simple Concat But Missing Pair Or Fragment Route

| normalized_table_name                                                      | stitching_category                   | promotion_status             | removal_assessment                                  |   downstream_reference_count |
|:---------------------------------------------------------------------------|:-------------------------------------|:-----------------------------|:----------------------------------------------------|-----------------------------:|
| Cohort: 3D radiomic features all OAR and DIL structures                    | simple_concat_missing_fragment_route | needs_patient_fragment_route | retain_for_validation                               |                            7 |
| Cohort: Per sample point prostate double sextant classification            | simple_concat_missing_fragment_route | needs_patient_fragment_route | no_downstream_reference_found_keep_until_validation |                            0 |
| Cohort: Per voxel prostate double sextant classification                   | simple_concat_missing_fragment_route | needs_patient_fragment_route | retain_for_validation                               |                            3 |
| Cohort: Simulated biopsy planned vs realized centroid variation validation | simple_concat_missing_fragment_route | needs_patient_fragment_route | no_downstream_reference_found_keep_until_validation |                            0 |

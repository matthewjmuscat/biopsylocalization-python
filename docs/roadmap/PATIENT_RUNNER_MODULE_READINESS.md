# Patient Runner Module Readiness

Last updated: 2026-05-23

This is the working checklist for moving the full `biopsy_localization_convex_main.py`
pipeline toward patient-local execution while preserving the validated cohort path
as the oracle. The goal is to make each scientific stage callable for one patient,
then let cohort wrappers remain simple ordered loops during validation.

Guardrails for this checklist:

- no scientific behavior changes in readiness passes,
- no edits to `python_files_dcm_meta_based/MC_simulator_convex.py` in the current pass,
- no raw CUDA/kernel math edits in the current pass,
- cohort aggregation, manifests, validation, and output stitching are allowed to stay
  cohort/run scoped,
- patient-ready means the stage has a one-patient entrypoint that can be called by
  a future patient runner without reading unrelated patients for scientific state.

## Status Legend

| Status | Meaning |
| --- | --- |
| Complete | Patient-level entrypoint exists and the current cohort wrapper can use it without changing behavior. |
| Partial | Some patient-level helpers exist, but the main stage still has cohort-only state, aggregation, or ordering assumptions. |
| Missing | Main-facing stage is still cohort-only or has no stable patient entrypoint. |
| Assembly | This is intentionally run/cohort scoped; patient fragments may feed it, but the stage itself is not a patient scientific module. |
| Out of scope | Intentionally excluded from the current pass. |

## Current Pass Summary

Completed in the 2026-05-23 pass:

- `pull_raw_structure_contours_for_patient(...)` added in `python_files_dcm_meta_based/preprocessing/structure_processing/raw_contour_pulling.py` and used by the cohort wrapper.
- `process_patient_real_biopsies(...)` added in `python_files_dcm_meta_based/preprocessing/biopsy_processing/biopsy_processor.py` and used by the cohort wrapper.
- `assign_patient_simulated_biopsy_targets(...)` added in `python_files_dcm_meta_based/preprocessing/biopsy_processing/simulated_biopsy_preparation.py` and used by the target-assignment cohort wrapper.
- `plan_patient_simulated_biopsies(...)` added in `python_files_dcm_meta_based/preprocessing/biopsy_processing/simulated_biopsy_planner.py` and used by the cohort wrapper.
- `determine_patient_realized_biopsy_targeting(...)` added in `python_files_dcm_meta_based/preprocessing/biopsy_processing/realized_biopsy_targeting.py` and used by the cohort wrapper.

## Main Pipeline Readiness Checklist

| Order | Pipeline stage | Main-facing function/module | Current status | Remaining work |
| --- | --- | --- | --- | --- |
| 1 | Input discovery and modality routing | Inline in `biopsy_localization_convex_main.py`; `UID_generator(...)` | Assembly | Keep run-scoped. Later extract discovery into a run manifest builder that emits patient case inputs. |
| 2 | Structure reference/bootstrap dictionaries | `structure_referencer(...)` | Partial | Split patient dictionary construction from global count/summary updates. Keep legacy dictionary shape during validation. |
| 3 | Transform precompute/random seed config | `configure_transform_precompute_settings(...)`, `configure_runtime_random_seed_settings(...)` | Assembly | Keep run-scoped; patient stages should receive resolved config values. |
| 4 | Run output directory creation | `create_run_output_directories(...)` | Assembly | Keep run-scoped; patient outputs should write under per-patient subdirectories. |
| 5 | Input manifest writing | `write_input_manifest_files(...)` | Assembly | Keep run-scoped; add patient case manifests when discovery is split. |
| 6 | MR ADC input normalization | `normalize_patient_mr_adc_input(...)`, `validate_and_normalize_mr_adc_inputs_for_cohort(...)` | Complete | Wrapper is patient-level already. The cohort wrapper tracks cross-patient unit consistency as a run validation concern. |
| 7 | Dose grid preprocessing | `build_dose_grid_runtime_objects_for_patient(...)`, `build_dose_grids_for_cohort(...)` | Complete | Future runner should pass patient-local progress/log adapters instead of cohort Rich tasks. |
| 8 | MR ADC grid preprocessing | `build_mr_adc_grid_runtime_objects_for_patient(...)`, `build_mr_adc_grids_for_cohort(...)` | Complete | Preserve `filter_out_negatives=True`. Future runner should pass patient-local progress/log adapters. |
| 9 | Raw contour pulling | `pull_raw_structure_contours_for_patient(...)`, `pull_raw_structure_contours_for_cohort(...)` | Complete | Completed this pass. Future typed wrapper should replace direct legacy dictionary arguments. |
| 10 | Unique structure selection | `select_patient_unique_structures(...)`, `select_unique_structures_for_cohort(...)` | Complete | Patient function exists. Global structure count update should eventually move to cohort assembly/summary. |
| 11 | Selected-structure legacy validation sidecar | `begin_selected_structures_legacy_validation(...)`, `finalize_selected_structures_legacy_validation(...)` | Assembly | Keep sidecar/oracle scoped; do not include in normal patient runner. |
| 12 | Standard non-biopsy structure preprocessing | `preprocess_non_biopsy_structure(...)`, `process_standard_non_biopsy_structure_preprocessing_stage(...)` | Partial | Add a patient-level family wrapper that processes prostate/rectum/urethra/DIL for one patient in the same order. Keep existing sidecar as oracle. |
| 13 | Non-biopsy legacy validation sidecar | `begin/prepare/finalize_standard_non_biopsy_structure_stage_legacy_validation(...)` | Assembly | Keep validation-only. Remove or disable after patient module validation is stable. |
| 14 | Prostate-only MR ADC summary | `process_patient_prostate_only_mr_adc(...)`, `prostate_only_mr_adc_processer(...)` | Complete | Patient function and legacy comparison surface exist. |
| 15 | Real biopsy preprocessing | `process_patient_real_biopsies(...)`, `real_biopsy_processer(...)` | Complete | Completed this pass. Future typed wrapper should isolate geometry config and progress adapters. |
| 16 | Simulated biopsy preparation: target assignment | `assign_patient_simulated_biopsy_targets(...)`, `assign_simulated_biopsy_targets(...)` | Complete | Completed this pass for target assignment only. |
| 17 | Simulated biopsy preparation: multiplicity expansion | `expand_simulated_biopsy_multiplicity(...)` | Partial | Add `expand_patient_simulated_biopsy_multiplicity(...)`; keep global biopsy count refresh in cohort wrapper. |
| 18 | Simulated biopsy preparation: length policy | `determine_simulated_biopsy_lengths(...)` | Partial | Add patient-level length determination after confirming no hidden cohort fallback is reintroduced. Current policy is patient-compatible. |
| 19 | Simulated biopsy preparation dataframe | `simulated_biopsy_preparation_dataframe_builder(...)` | Partial | Add `build_patient_simulated_biopsy_preparation_dataframe(...)` and make cohort output assembly concatenate patient fragments. |
| 20 | Simulated biopsy planning | `plan_patient_simulated_biopsies(...)`, `simulated_biopsy_planner_processer(...)` | Complete | Completed this pass. Future runner may replace `parallel_pool` with a sequential patient execution context. |
| 21 | Uncertainty generation and attachment | `prepare_and_attach_uncertainty_data(...)`, `attach_uncertainty_data_from_dataframe(...)` | Partial | Keep template generation run-scoped. Add patient-level attachment from a resolved uncertainty dataframe fragment. |
| 22 | Transform generation/prep | `MC_prepper_funcs.generate_transformations(...)` | Missing | Build a patient-level transform generation/prep wrapper outside `MC_simulator_convex.py`. Do not change sampled transform math. |
| 23 | Optimizer v1 | `biopsy_optimizer_module_v1(...)` | Missing | Add patient-level optimizer v1 wrapper or mark as legacy-only if v2 becomes the validated target path. |
| 24 | Optimizer v2 live integration | `run_target_dil_optimizer_v2_for_live_simulated_family(...)` | Partial | Internal v2 modules are modular, but the main live integration surface still needs a patient-stage wrapper and validation against current outputs. |
| 25 | Simulated biopsy finalization | `simulated_biopsy_processer(...)` | Partial | Add `process_patient_simulated_biopsies(...)` and route cohort wrapper through it. Structure-level helper boundaries already exist. |
| 26 | Planned-vs-realized centroid validation | `validate_simulated_biopsy_planned_vs_realized_centroid_variation(...)` | Partial | Add patient validation helper that returns one patient fragment plus a run-level summarizer. |
| 27 | Realized biopsy targeting | `determine_patient_realized_biopsy_targeting(...)`, `realized_biopsy_targeting_processer(...)` | Complete | Completed this pass. |
| 28 | Pickled preprocessed bundle export/load | `export_preprocessed_pickle_bundle(...)`, `load_selected_pickle_bundle_run(...)` | Assembly | Keep run-scoped. Future patient runner can consume patient case fragments from bundles. |
| 29 | Sampled biopsy processing | `sampled_biopsy_processing_processer(...)` | Partial | Split sampling arg construction, result storage, and biopsy-frame coordinate generation into patient-level surfaces. Preserve sampling order and any random-color side effects during validation. |
| 30 | Prostate double-sextant classification | `biopsy_double_sextant_processer(...)` | Partial | Patient sample-point helper exists. Be careful moving per-voxel aggregation because random tie-breaking order could affect outputs. |
| 31 | MC simulation | `MC_simulator_convex.simulator_parallel(...)` | Out of scope | Explicitly excluded from this pass. Build separate patient-facing MC modules later without editing the oracle file first. |
| 32 | Cohort simulated-biopsy preparation table | `dataframe_builders.cohort_simulated_biopsy_preparation_dataframe_builder(...)` | Assembly | Replace with concatenation of patient preparation fragments after row-order validation. |
| 33 | Guidance map precompute/render | `precompute_guidance_map_firing_depth_recommendations_for_run(...)`, `render_guidance_maps_for_run(...)` | Partial | Treat render as run/UI scoped. Add patient precompute fragments if guidance maps become patient-runner outputs. |
| 34 | Optimizer-v2 downstream annotations | `annotate_target_dil_optimizer_v2_outputs_with_downstream_mc_scores(...)`, `annotate_target_dil_optimizer_v2_outputs_with_biopsy_sampling_audit(...)` | Partial | Add patient-level annotation wrappers after sampled biopsy and MC outputs are patient-local. |
| 35 | Final dataframe builders | `dataframe_builders.*` calls in main | Assembly | Move final cohort tables to assembly from patient base artifacts. Avoid schema churn during migration. |
| 36 | Phase3B/Phase3C output surface | `build_in_memory_stitch_validation(...)`, `write_phase3c_output_surface(...)` | Assembly | Existing patient-fragment output/assembly surface is validated as a shadow gate, not scientific recomputation. |
| 37 | Patient-runner validation gate | `run_patient_runner_main_validation(...)` | Assembly | Keep as validation gate until scientific patient stages are independently runnable. |
| 38 | Run completion manifest | `write_run_completion_manifest(...)` | Assembly | Keep run-scoped; patient manifests are handled by `patient_runner`. |

## Next Implementation Order

Recommended next non-MC items:

1. Add `process_patient_standard_non_biopsy_structure_families(...)` around the existing non-biopsy family loop.
2. Finish simulated biopsy preparation by extracting multiplicity, length, and preparation dataframe patient helpers.
3. Add patient-level simulated biopsy finalization wrapper.
4. Split sampled biopsy processing into patient sampling args, patient result storage, and patient biopsy-frame coordinate generation.
5. Add patient-level uncertainty attachment from an already resolved uncertainty dataframe.
6. Build transform-generation wrappers in `MC_prepper_funcs.py` without entering `MC_simulator_convex.py` or raw kernel code.
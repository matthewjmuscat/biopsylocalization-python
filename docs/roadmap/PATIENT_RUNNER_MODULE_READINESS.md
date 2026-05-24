# Patient Runner Module Readiness

Last updated: 2026-05-23

This is the working checklist for moving the full `biopsy_localization_convex_main.py`
pipeline toward patient-local execution while preserving the validated cohort path
as the oracle. The goal is to make each scientific stage callable for one patient
without wiring those new entrypoints into the frozen cohort/oracle path until a
dedicated patient runner is ready to compare against it.

Guardrails for this checklist:

- no scientific behavior changes in readiness passes,
- no edits to `python_files_dcm_meta_based/MC_simulator_convex.py` in the current pass,
- no raw CUDA/kernel math edits in the current pass,
- cohort aggregation, manifests, validation, and output stitching are allowed to stay
  cohort/run scoped,
- patient-ready means the stage has a one-patient entrypoint that can be called by
  a future patient runner without reading unrelated patients for scientific state,
  not that the existing cohort wrapper has been rewritten to call it.

Canonical module placement is governed by
`../architecture/PATIENT_MODULE_TREE_GUIDE.md`.

## Quick Human Snapshot

Working definition used in this file:

- a patient-level module means one callable that performs one iteration of the
  legacy cohort loop for that stage,
- helper functions do not count by themselves if the main stage still has no
  clear one-patient entrypoint,
- run-scoped discovery, manifests, assembly, and validation remain allowed to
  stay outside the patient-stage surface.

If you want the shortest answer to "what is still missing for a per-patient
runner?", it is this:

- several preprocessing stages are already patient-ready,
- several biopsy stages already have partial patient helpers but still do not
  have the one callable that packages the legacy cohort-loop body for one
  patient,
- the runner itself is still mostly orchestration/artifact writing and does not
  yet execute the full scientific stage sequence independently,
- the later MC and MR simulation tranche still needs dedicated patient wrappers
  around the active legacy transform and simulation surfaces.

### Already Has A Real Patient Stage

- MR ADC input normalization
- Dose-grid preprocessing
- MR ADC grid preprocessing
- Raw contour pulling
- Unique structure selection
- Prostate-only MR ADC summary
- Real biopsy preprocessing
- Simulated biopsy target assignment
- Simulated biopsy planning
- Realized biopsy targeting

### Has Patient Helpers But Still Missing The Full Patient Stage

- Standard non-biopsy structure preprocessing
  reason: the file already has a per-structure primitive, but still lacks one
  patient-family wrapper that runs prostate, rectum, urethra, and DIL for one
  patient in the legacy order.
- Simulated biopsy preparation
  reason: target assignment exists, but multiplicity expansion, length policy,
  preparation-dataframe building, and the overall one-patient preparation
  wrapper are still cohort-oriented.
- Simulated biopsy finalization
  reason: the stage has reusable transport and geometry helpers, but still lacks
  one patient-level wrapper for finalizing all simulated biopsies for one
  patient.
- Uncertainty attachment
  reason: attachment logic exists, but the current main-facing stage is still a
  run-scoped file/template/UI workflow rather than a clean patient attach step.
- Planned-vs-realized centroid validation
  reason: the current function already writes patient fragments, but it still
  returns a cohort-wide dataframe/summary rather than exposing an explicit
  patient-stage surface plus run summarizer.
- Sampled biopsy processing
  reason: the file is internally split, but the full stage is still cohort/MC
  adjacent and not yet safe to treat as a finished patient-stage surface.
- Prostate double-sextant classification
  reason: a patient sample-point builder exists, but the stage still includes
  cohort aggregation and random tie behavior in voxel-level aggregation.

### Runner-Relevant But Intentionally Run Scoped

- Input discovery and modality routing
- Structure reference/bootstrap dictionaries
- Transform/random-seed config
- Run output directory creation
- Input manifest writing
- Pickled preprocessed bundle export/load
- Final dataframe builders and cohort table assembly
- Phase3B/Phase3C output surface
- Patient-runner validation gate
- Run completion manifest

### Later MC And MR Simulation Modules To Build

These are not the next tranche, but they are part of the eventual patient-runner
surface and should be included in planning.

For the MC/MR tranche, the target is not a thin wrapper around the current
top-level legacy entrypoints. The target is patient-level modules for the loop
bodies that currently live inside those oracle functions.

- Patient transform-bank generation modules extracted from the patient-level work
  inside `MC_prepper_funcs.generate_transformations(...)`
  purpose: create one patient's transform/precompute state without changing
  transform math.
- Patient BX-only transform application modules extracted from the patient-level
  work inside `MC_prepper_funcs.biopsy_only_transformer(...)`
  purpose: apply the already generated biopsy-only transform bank for one
  patient.
- Patient relative-structure transform modules extracted from the patient-level
  work inside `MC_prepper_funcs.biopsy_transformer_to_relative_structures(...)`
  purpose: apply the active relative-structure transform path for one patient.
- Do not target `MC_prepper_funcs.biopsy_and_structure_shifter(...)` unless the
  older global anatomy-shift path is deliberately revived. The active main path
  currently uses BX-only plus relative-structure transforms instead.
- Patient containment/dose simulation modules extracted from the patient-facing
  loop bodies inside `MC_simulator_convex.simulator_parallel(...)`
  purpose: reproduce one patient's active containment and dose simulation path
  while keeping the current oracle file frozen.
- Patient MR localization/simulation modules extracted from the patient-facing
  loop bodies inside `MC_simulator_MR.simulator_parallel(...)`
  purpose: reproduce one patient's MR ADC localization/simulation path after
  biopsy MC transforms already exist.
- Downstream patient MC annotation/output wrappers after patient-local sampling
  and MC outputs exist
  purpose: replace current downstream cohort annotations only after upstream
  patient MC state is stable.

Likely home when that later tranche starts:

- a future dedicated MC scientific family such as `python_files_dcm_meta_based/mc/per_patient/`,
- or, if extraction begins beside the current oracle files first, thin explicit
  patient wrappers next to the existing MC files until the MC family is stable.

## Status Legend

| Status | Meaning |
| --- | --- |
| Complete | Additive patient-level entrypoint exists for future runner use; the frozen cohort/oracle path does not need to call it. |
| Partial | Some patient-level helpers exist, but the main stage still has cohort-only state, aggregation, or ordering assumptions. |
| Missing | Main-facing stage is still cohort-only or has no stable patient entrypoint. |
| Assembly | This is intentionally run/cohort scoped; patient fragments may feed it, but the stage itself is not a patient scientific module. |
| Out of scope | Intentionally excluded from the current pass. |

## Current Pass Summary

Completed in the 2026-05-23 pass:

- The initial additive patient wrappers were moved out of the temporary top-level
  `patient_stages` area and into family-local `per_patient/` homes inside the
  existing scientific tree.

- `pull_raw_structure_contours_for_patient(...)` lives in `python_files_dcm_meta_based/preprocessing/structure_processing/per_patient/raw_contour_pulling.py` for future runner use; the cohort wrapper remains on its frozen body.
- `process_patient_real_biopsies(...)` lives in `python_files_dcm_meta_based/preprocessing/biopsy_processing/per_patient/real_biopsy_processing.py` for future runner use; the cohort wrapper remains on its frozen body.
- `assign_patient_simulated_biopsy_targets(...)` lives in `python_files_dcm_meta_based/preprocessing/biopsy_processing/per_patient/simulated_biopsy_preparation.py` for future runner use; the target-assignment cohort wrapper remains on its frozen body.
- `plan_patient_simulated_biopsies(...)` lives in `python_files_dcm_meta_based/preprocessing/biopsy_processing/per_patient/simulated_biopsy_planning.py` for future runner use; the cohort wrapper remains on its frozen body.
- `determine_patient_realized_biopsy_targeting(...)` lives in `python_files_dcm_meta_based/preprocessing/biopsy_processing/per_patient/realized_biopsy_targeting.py` for future runner use; the cohort wrapper remains on its frozen body.

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
| 9 | Raw contour pulling | `preprocessing.structure_processing.per_patient.raw_contour_pulling.pull_raw_structure_contours_for_patient(...)`, `pull_raw_structure_contours_for_cohort(...)` | Complete | Additive patient module exists in the owning structure-processing family. Current cohort wrapper remains frozen; future typed wrapper should replace direct legacy dictionary arguments in the patient runner. |
| 10 | Unique structure selection | `select_patient_unique_structures(...)`, `select_unique_structures_for_cohort(...)` | Complete | Patient function exists. Global structure count update should eventually move to cohort assembly/summary. |
| 11 | Selected-structure legacy validation sidecar | `begin_selected_structures_legacy_validation(...)`, `finalize_selected_structures_legacy_validation(...)` | Assembly | Keep sidecar/oracle scoped; do not include in normal patient runner. |
| 12 | Standard non-biopsy structure preprocessing | `preprocess_non_biopsy_structure(...)`, `process_standard_non_biopsy_structure_preprocessing_stage(...)` | Partial | Add a patient-level family wrapper that processes prostate/rectum/urethra/DIL for one patient in the same order. Do not route the frozen cohort wrapper through it yet. Keep existing sidecar as oracle. |
| 13 | Non-biopsy legacy validation sidecar | `begin/prepare/finalize_standard_non_biopsy_structure_stage_legacy_validation(...)` | Assembly | Keep validation-only. Remove or disable after patient module validation is stable. |
| 14 | Prostate-only MR ADC summary | `process_patient_prostate_only_mr_adc(...)`, `prostate_only_mr_adc_processer(...)` | Complete | Patient function and legacy comparison surface exist. |
| 15 | Real biopsy preprocessing | `preprocessing.biopsy_processing.per_patient.real_biopsy_processing.process_patient_real_biopsies(...)`, `real_biopsy_processer(...)` | Complete | Additive patient module exists in the owning biopsy-processing family. Current cohort wrapper remains frozen; future typed wrapper should isolate geometry config and progress adapters. |
| 16 | Simulated biopsy preparation: target assignment | `preprocessing.biopsy_processing.per_patient.simulated_biopsy_preparation.assign_patient_simulated_biopsy_targets(...)`, `assign_simulated_biopsy_targets(...)` | Complete | Additive patient module exists for target assignment only in the owning biopsy-processing family. Current cohort wrapper remains frozen. |
| 17 | Simulated biopsy preparation: multiplicity expansion | `expand_simulated_biopsy_multiplicity(...)` | Partial | Add `expand_patient_simulated_biopsy_multiplicity(...)`; keep global biopsy count refresh in cohort wrapper. |
| 18 | Simulated biopsy preparation: length policy | `determine_simulated_biopsy_lengths(...)` | Partial | Add patient-level length determination after confirming no hidden cohort fallback is reintroduced. Current policy is patient-compatible. |
| 19 | Simulated biopsy preparation dataframe | `simulated_biopsy_preparation_dataframe_builder(...)` | Partial | Add `build_patient_simulated_biopsy_preparation_dataframe(...)` and make cohort output assembly concatenate patient fragments. |
| 20 | Simulated biopsy planning | `preprocessing.biopsy_processing.per_patient.simulated_biopsy_planning.plan_patient_simulated_biopsies(...)`, `simulated_biopsy_planner_processer(...)` | Complete | Additive patient module exists in the owning biopsy-processing family. Current cohort wrapper remains frozen; future runner may replace `parallel_pool` with a sequential patient execution context. |
| 21 | Uncertainty generation and attachment | `prepare_and_attach_uncertainty_data(...)`, `attach_uncertainty_data_from_dataframe(...)` | Partial | Keep template generation run-scoped. Add patient-level attachment from a resolved uncertainty dataframe fragment. |
| 22 | Transform generation/prep | `MC_prepper_funcs.generate_transformations(...)` | Missing | Build a patient-level transform generation/prep wrapper outside `MC_simulator_convex.py`. Do not change sampled transform math. |
| 23 | Optimizer v1 | `biopsy_optimizer_module_v1(...)` | Missing | Add patient-level optimizer v1 wrapper or mark as legacy-only if v2 becomes the validated target path. |
| 24 | Optimizer v2 live integration | `run_target_dil_optimizer_v2_for_live_simulated_family(...)` | Partial | Internal v2 modules are modular, but the main live integration surface still needs a patient-stage wrapper and validation against current outputs. |
| 25 | Simulated biopsy finalization | `simulated_biopsy_processer(...)` | Partial | Add `process_patient_simulated_biopsies(...)` for future runner use. Do not route the frozen cohort wrapper through it yet. Structure-level helper boundaries already exist. |
| 26 | Planned-vs-realized centroid validation | `validate_simulated_biopsy_planned_vs_realized_centroid_variation(...)` | Partial | Add patient validation helper that returns one patient fragment plus a run-level summarizer. |
| 27 | Realized biopsy targeting | `preprocessing.biopsy_processing.per_patient.realized_biopsy_targeting.determine_patient_realized_biopsy_targeting(...)`, `realized_biopsy_targeting_processer(...)` | Complete | Additive patient module exists in the owning biopsy-processing family. Current cohort wrapper remains frozen. |
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

1. Add `process_patient_standard_non_biopsy_structure_families(...)` beside the existing non-biopsy family loop without routing the current cohort wrapper through it.
2. Finish simulated biopsy preparation by extracting multiplicity, length, and preparation dataframe patient helpers.
3. Add patient-level simulated biopsy finalization wrapper without routing the current cohort wrapper through it.
4. Split sampled biopsy processing into patient sampling args, patient result storage, and patient biopsy-frame coordinate generation without changing the current cohort wrapper.
5. Add patient-level uncertainty attachment from an already resolved uncertainty dataframe.
6. Build transform-generation wrappers in `MC_prepper_funcs.py` without entering `MC_simulator_convex.py` or raw kernel code.

Recommended later MC/MR tranche after the non-MC patient-stage surface is more
complete:

1. Extract patient transform-generation/precompute modules from the per-patient work inside `MC_prepper_funcs.generate_transformations(...)`.
2. Extract patient BX-only transform application modules from the per-patient work inside `MC_prepper_funcs.biopsy_only_transformer(...)`.
3. Extract patient relative-structure transform modules from the per-patient work inside `MC_prepper_funcs.biopsy_transformer_to_relative_structures(...)`.
4. Extract patient containment/dose simulation modules from the per-patient loop bodies inside `MC_simulator_convex.simulator_parallel(...)` without editing the oracle file first.
5. Extract patient MR localization/simulation modules from the per-patient loop bodies inside `MC_simulator_MR.simulator_parallel(...)`.
6. Add downstream patient MC annotation/output wrappers only after the upstream patient MC state is validated.
# Patient Runner Module Readiness

Last updated: 2026-05-24

This is the working checklist for moving the full `biopsy_localization_convex_main.py`
pipeline toward patient-local execution while preserving the validated cohort path
as the oracle. The goal is to make each scientific stage callable for one patient
without wiring those new entrypoints into the frozen cohort/oracle path until a
dedicated patient runner is ready to compare against it.

Guardrails for this checklist:

- no scientific behavior changes in readiness passes,
- no edits to `python_files_dcm_meta_based/MC_simulator_convex.py` in the current pass,
- no raw CUDA/kernel math edits in the current pass,
- Rich is a presentation adapter, not a scientific dependency; avoid adding new
  required Rich/progress/live-display arguments to patient scientific modules,
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
- the remaining missing or partial stages are now mostly structure bootstrap,
  optimizers, guidance-map precompute, downstream annotations, and actual MC/MR
  simulation,
- a thin runner/validation scaffold can exist while modules are still being
  extracted, but the complete patient runner should come after the remaining
  patient-stage modules have stable contracts,
- the later MC and MR simulation tranche still needs dedicated patient wrappers
  around the active legacy transform and simulation surfaces.

## Presentation/Rich Decoupling Policy

Rich should remain available for the legacy CLI/batch surface while the patient
runner is being validated, but patient scientific modules should be written as if
Rich can be swapped for a GUI adapter or for no presentation layer at all.

Working order for this policy:

1. Now: keep building patient modules, but avoid adding new required Rich args
   whenever possible.
2. Near-term: add a thin progress/log/event adapter layer so patient modules can
   run with Rich, a GUI, or a null/headless implementation.
3. After patient-runner validation: remove remaining Rich references from patient
   scientific functions and leave Rich only in legacy/main/frontend wrappers.

Existing Rich/progress/live-display arguments in patient functions are migration
debt, not placement failures. Do not stop patient-stage extraction just to remove
all of that debt upfront. Also do not route frozen legacy cohort wrappers through
new patient functions only to clean up Rich; preserve the oracle path until a
deliberate validation pass is ready.

### Already Has A Real Patient Stage

- MR ADC input normalization
- Dose-grid preprocessing
- MR ADC grid preprocessing
- Raw contour pulling
- Unique structure selection
- Standard non-biopsy structure preprocessing
- Prostate-only MR ADC summary
- Real biopsy preprocessing
- Simulated biopsy preparation
- Simulated biopsy finalization
- Simulated biopsy target assignment
- Simulated biopsy planning
- Sampled biopsy processing
- Uncertainty attachment
- Realized biopsy targeting
- Planned-vs-realized centroid validation
- Prostate double-sextant classification
- Transform generation/prep
- BX-only transform application
- Relative-structure transform application

### Has Patient Helpers But Still Missing The Full Patient Stage

None currently identified in this pass.

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

Current MC-prep patient module home:

- `python_files_dcm_meta_based/mc/prep/per_patient/`

Future MC simulation module home:

- `python_files_dcm_meta_based/mc/simulation/per_patient/`

## Status Legend

| Status | Meaning |
| --- | --- |
| Complete | Additive patient-level entrypoint exists for future runner use; the frozen cohort/oracle path does not need to call it. |
| Partial | Some patient-level helpers exist, but the main stage still has cohort-only state, aggregation, or ordering assumptions. |
| Missing | Main-facing stage is still cohort-only or has no stable patient entrypoint. |
| Assembly | This is intentionally run/cohort scoped; patient fragments may feed it, but the stage itself is not a patient scientific module. |
| Out of scope | Intentionally excluded from the current pass. |

## Current Pass Summary

Completed through the 2026-05-24 pass:

- The initial additive patient wrappers were moved out of the temporary top-level
  `patient_stages` area and into family-local `per_patient/` homes inside the
  existing scientific tree.

- `pull_raw_structure_contours_for_patient(...)` lives in `python_files_dcm_meta_based/preprocessing/structure_processing/per_patient/raw_contour_pulling.py` for future runner use; the cohort wrapper remains on its frozen body.
- `process_patient_standard_non_biopsy_structure_families(...)` lives in `python_files_dcm_meta_based/preprocessing/structure_processing/non_biopsy_structure_loop.py` for future runner use; the cohort wrapper remains on its frozen body.
- `process_patient_real_biopsies(...)` lives in `python_files_dcm_meta_based/preprocessing/biopsy_processing/per_patient/real_biopsy_processing.py` for future runner use; the cohort wrapper remains on its frozen body.
- `prepare_patient_simulated_biopsies(...)` plus the patient-local target assignment, multiplicity expansion, length policy, and preparation dataframe helpers live in `python_files_dcm_meta_based/preprocessing/biopsy_processing/per_patient/simulated_biopsy_preparation.py` for future runner use; the cohort wrapper remains on its frozen body.
- `plan_patient_simulated_biopsies(...)` lives in `python_files_dcm_meta_based/preprocessing/biopsy_processing/per_patient/simulated_biopsy_planning.py` for future runner use; the cohort wrapper remains on its frozen body.
- `process_patient_simulated_biopsies(...)` lives in `python_files_dcm_meta_based/preprocessing/biopsy_processing/per_patient/simulated_biopsy_processing.py` for future runner use; the cohort wrapper remains on its frozen body.
- `determine_patient_realized_biopsy_targeting(...)` lives in `python_files_dcm_meta_based/preprocessing/biopsy_processing/per_patient/realized_biopsy_targeting.py` for future runner use; the cohort wrapper remains on its frozen body.
- `attach_patient_uncertainty_data_from_dataframe(...)` lives in `python_files_dcm_meta_based/preprocessing/uncertainty_attachment.py` for future runner use with a resolved patient uncertainty dataframe fragment; template generation, file workflow, and cohort bookkeeping remain on the frozen run-scoped path.
- `process_patient_sampled_biopsies(...)` plus the patient-local sampling-arg, sampled-result storage, and biopsy-coordinate helpers live in `python_files_dcm_meta_based/preprocessing/biopsy_processing/sampled_biopsy_processing.py`; the cohort wrapper remains on its frozen body.
- `build_patient_simulated_biopsy_centroid_variation_validation_fragment(...)` and `assemble_simulated_biopsy_centroid_variation_validation_fragments(...)` live in `python_files_dcm_meta_based/preprocessing/biopsy_processing/per_patient/centroid_variation_validation.py` for future runner use; the cohort validation wrapper remains on its frozen body.
- `build_patient_biopsy_double_sextant_sample_point_fragment(...)`, `assemble_biopsy_double_sextant_classification_fragments(...)`, and `store_patient_biopsy_double_sextant_voxel_fragment(...)` live in `python_files_dcm_meta_based/preprocessing/biopsy_processing/per_patient/double_sextant_classification.py` for future runner use; the cohort wrapper remains on its frozen body. The per-voxel table remains assembled from patient sample-point fragments to preserve legacy aggregation and random-tie behavior.
- `generate_transformations_for_patient(...)`, `apply_patient_biopsy_self_transforms(...)`, and `apply_patient_relative_structure_transforms(...)` live in `python_files_dcm_meta_based/mc/prep/per_patient/` as additive patient-local MC prep surfaces; the frozen cohort wrappers in `MC_prepper_funcs.py` remain on their original bodies.

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
| 12 | Standard non-biopsy structure preprocessing | `process_patient_standard_non_biopsy_structure_families(...)`, `preprocess_non_biopsy_structure(...)`, `process_standard_non_biopsy_structure_preprocessing_stage(...)` | Complete | Additive patient family-stage entrypoint now exists in the owning structure-processing module. Keep the frozen cohort wrapper and existing sidecar as the oracle path. |
| 13 | Non-biopsy legacy validation sidecar | `begin/prepare/finalize_standard_non_biopsy_structure_stage_legacy_validation(...)` | Assembly | Keep validation-only. Remove or disable after patient module validation is stable. |
| 14 | Prostate-only MR ADC summary | `process_patient_prostate_only_mr_adc(...)`, `prostate_only_mr_adc_processer(...)` | Complete | Patient function and legacy comparison surface exist. |
| 15 | Real biopsy preprocessing | `preprocessing.biopsy_processing.per_patient.real_biopsy_processing.process_patient_real_biopsies(...)`, `real_biopsy_processer(...)` | Complete | Additive patient module exists in the owning biopsy-processing family. Current cohort wrapper remains frozen; future typed wrapper should isolate geometry config and progress adapters. |
| 16 | Simulated biopsy preparation: target assignment | `preprocessing.biopsy_processing.per_patient.simulated_biopsy_preparation.assign_patient_simulated_biopsy_targets(...)`, `assign_simulated_biopsy_targets(...)` | Complete | Additive patient target-assignment module exists in the owning biopsy-processing family. Current cohort wrapper remains frozen. |
| 17 | Simulated biopsy preparation: multiplicity expansion | `preprocessing.biopsy_processing.per_patient.simulated_biopsy_preparation.expand_patient_simulated_biopsy_multiplicity(...)`, `expand_simulated_biopsy_multiplicity(...)` | Complete | Additive patient multiplicity expansion exists. Keep the legacy cohort-wide biopsy-count refresh in the frozen cohort wrapper or later assembly layer. |
| 18 | Simulated biopsy preparation: length policy | `preprocessing.biopsy_processing.per_patient.simulated_biopsy_preparation.determine_patient_simulated_biopsy_lengths(...)`, `determine_simulated_biopsy_lengths(...)` | Complete | Additive patient-level length determination exists and stays on patient-local data only; cohort-derived fallback methods remain excluded. |
| 19 | Simulated biopsy preparation dataframe | `preprocessing.biopsy_processing.per_patient.simulated_biopsy_preparation.build_patient_simulated_biopsy_preparation_dataframe(...)`, `simulated_biopsy_preparation_dataframe_builder(...)` | Complete | Additive patient preparation-fragment builder exists. Future cohort output assembly can concatenate patient fragments after row-order validation. |
| 20 | Simulated biopsy planning | `preprocessing.biopsy_processing.per_patient.simulated_biopsy_planning.plan_patient_simulated_biopsies(...)`, `simulated_biopsy_planner_processer(...)` | Complete | Additive patient module exists in the owning biopsy-processing family. Current cohort wrapper remains frozen; future runner may replace `parallel_pool` with a sequential patient execution context. |
| 21 | Uncertainty generation and attachment | `preprocessing.uncertainty_attachment.attach_patient_uncertainty_data_from_dataframe(...)`, `prepare_and_attach_uncertainty_data(...)`, `attach_uncertainty_data_from_dataframe(...)` | Complete | Additive patient-level attachment from a resolved uncertainty dataframe fragment exists. Keep template generation, file prompts, and run-level dataframe bookkeeping on the frozen run-scoped path. |
| 22 | Transform generation/prep | `mc.prep.per_patient.transform_generation.generate_transformations_for_patient(...)`, `mc.prep.per_patient.biopsy_self_transforms.apply_patient_biopsy_self_transforms(...)`, `mc.prep.per_patient.relative_structure_transforms.apply_patient_relative_structure_transforms(...)`, `MC_prepper_funcs.generate_transformations(...)`, `MC_prepper_funcs.biopsy_only_transformer(...)`, `MC_prepper_funcs.biopsy_transformer_to_relative_structures(...)` | Complete | Additive patient-level MC prep surfaces now exist in the dedicated MC prep scientific package. Frozen cohort wrappers in `MC_prepper_funcs.py` remain on their original bodies and are not routed through the patient entrypoints. |
| 23 | Optimizer v1 | `biopsy_optimizer_module_v1(...)` | Missing | Add patient-level optimizer v1 wrapper or mark as legacy-only if v2 becomes the validated target path. |
| 24 | Optimizer v2 live integration | `run_target_dil_optimizer_v2_for_live_simulated_family(...)` | Partial | Internal v2 modules are modular, but the main live integration surface still needs a patient-stage wrapper and validation against current outputs. |
| 25 | Simulated biopsy finalization | `preprocessing.biopsy_processing.per_patient.simulated_biopsy_processing.process_patient_simulated_biopsies(...)`, `simulated_biopsy_processer(...)` | Complete | Additive patient finalization module exists in the owning biopsy-processing family. Current cohort wrapper remains frozen and is not routed through the patient module. |
| 26 | Planned-vs-realized centroid validation | `preprocessing.biopsy_processing.per_patient.centroid_variation_validation.build_patient_simulated_biopsy_centroid_variation_validation_fragment(...)`, `preprocessing.biopsy_processing.per_patient.centroid_variation_validation.assemble_simulated_biopsy_centroid_variation_validation_fragments(...)`, `validate_simulated_biopsy_planned_vs_realized_centroid_variation(...)` | Complete | Additive patient fragment builder plus run-level summarizer exist. Current cohort validation wrapper remains frozen and is not routed through the patient module. |
| 27 | Realized biopsy targeting | `preprocessing.biopsy_processing.per_patient.realized_biopsy_targeting.determine_patient_realized_biopsy_targeting(...)`, `realized_biopsy_targeting_processer(...)` | Complete | Additive patient module exists in the owning biopsy-processing family. Current cohort wrapper remains frozen. |
| 28 | Pickled preprocessed bundle export/load | `export_preprocessed_pickle_bundle(...)`, `load_selected_pickle_bundle_run(...)` | Assembly | Keep run-scoped. Future patient runner can consume patient case fragments from bundles. |
| 29 | Sampled biopsy processing | `preprocessing.biopsy_processing.sampled_biopsy_processing.process_patient_sampled_biopsies(...)`, `sampled_biopsy_processing_processer(...)` | Complete | Additive patient stage now exists in the owning biopsy-processing module and preserves the current per-patient sampling order by composing patient-local sampling-arg, sampled-result storage, and biopsy-coordinate helpers. Current cohort wrapper remains frozen. |
| 30 | Prostate double-sextant classification | `preprocessing.biopsy_processing.per_patient.double_sextant_classification.build_patient_biopsy_double_sextant_sample_point_fragment(...)`, `preprocessing.biopsy_processing.per_patient.double_sextant_classification.assemble_biopsy_double_sextant_classification_fragments(...)`, `biopsy_double_sextant_processer(...)` | Complete | Additive patient sample-point fragment builder plus run-level per-voxel summarizer exist. Per-voxel aggregation stays run-level to preserve legacy random-tie behavior. Current cohort wrapper remains frozen. |
| 31 | MC simulation | `MC_simulator_convex.simulator_parallel(...)` | Out of scope | Explicitly excluded from this pass. Build separate patient-facing MC modules later without editing the oracle file first. |
| 32 | Cohort simulated-biopsy preparation table | `dataframe_builders.cohort_simulated_biopsy_preparation_dataframe_builder(...)` | Assembly | Replace with concatenation of patient preparation fragments after row-order validation. Patient fragment building now exists in the owning biopsy-processing family. |
| 33 | Guidance map precompute/render | `precompute_guidance_map_firing_depth_recommendations_for_run(...)`, `render_guidance_maps_for_run(...)` | Partial | Treat render as run/UI scoped. Add patient precompute fragments if guidance maps become patient-runner outputs. |
| 34 | Optimizer-v2 downstream annotations | `annotate_target_dil_optimizer_v2_outputs_with_downstream_mc_scores(...)`, `annotate_target_dil_optimizer_v2_outputs_with_biopsy_sampling_audit(...)` | Partial | Add patient-level annotation wrappers after sampled biopsy and MC outputs are patient-local. |
| 35 | Final dataframe builders | `dataframe_builders.*` calls in main | Assembly | Move final cohort tables to assembly from patient base artifacts. Avoid schema churn during migration. |
| 36 | Phase3B/Phase3C output surface | `build_in_memory_stitch_validation(...)`, `write_phase3c_output_surface(...)` | Assembly | Existing patient-fragment output/assembly surface is validated as a shadow gate, not scientific recomputation. |
| 37 | Patient-runner validation gate | `run_patient_runner_main_validation(...)` | Assembly | Keep as validation gate until scientific patient stages are independently runnable. |
| 38 | Run completion manifest | `write_run_completion_manifest(...)` | Assembly | Keep run-scoped; patient manifests are handled by `patient_runner`. |

## Next Implementation Order

Recommended order of operations from this point:

1. Keep extracting missing patient scientific modules where the next loop-body
   boundary is clear, while avoiding new required Rich dependencies.
2. Add a small presentation-neutral progress/log/event adapter surface with Rich,
   null/headless, and future GUI implementations. This should be thin enough not
   to become a new orchestration framework.
3. Split structure reference/bootstrap dictionary construction into patient-local
   dictionary construction plus run-level count/summary assembly.
4. Add or expose guidance-map patient precompute fragments, while keeping
   guidance-map rendering run/UI scoped.
5. Decide whether optimizer v1 should receive a patient wrapper or be marked
   legacy-only if optimizer v2 is the validated target path.
6. Add a patient-stage wrapper around the optimizer-v2 live integration surface
   after confirming its required inputs, memory behavior, rendering policy, and
   output annotations.
7. Extract patient containment/dose simulation modules from the per-patient loop
   bodies inside `MC_simulator_convex.simulator_parallel(...)` without editing the
   oracle file first.
8. Extract patient MR localization/simulation modules from the per-patient loop
   bodies inside `MC_simulator_MR.simulator_parallel(...)`.
9. Add downstream patient MC annotation/output wrappers only after the upstream
   patient MC state is validated.
10. Build out the complete patient runner after the scientific patient-stage
    modules above have stable contracts. Earlier runner work should stay limited
    to scaffolding, manifests, adapters, and shadow-validation harnesses.
11. After patient-runner validation against the legacy cohort oracle, remove
    remaining Rich dependencies from patient scientific functions and keep Rich
    only in legacy/main/frontend wrappers.
# Patient Runner Module Readiness

Last updated: 2026-05-27

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
- optimizer-v1, convex MC containment/dose simulation, and MR ADC MC localization
  now have true additive patient-local scientific stages,
- a thin runner/validation scaffold can exist while modules are still being
  extracted, but the complete patient runner should come after the remaining
  patient-stage modules have stable contracts,
- convex/MR MC simulation now has typed patient contracts, containment/dose/MR
  output collectors, singleton legacy adapters, containment helper slices, additive
  dose/dose-gradient localization plus DVH helper slices, and additive MR ADC
  localization helpers plus patient-local convex and MR ADC stages, plus downstream
  MC annotation/output table wrappers; remaining MC work is runner wiring, parity
  validation, and inactive legacy dose statistics only if parity proves them
  required.

## Presentation/Rich Decoupling Policy

Rich should remain available for the legacy CLI/batch surface while the patient
runner is being validated, but patient scientific modules should be written as if
Rich can be swapped for a GUI adapter or for no presentation layer at all.

Working order for this policy:

1. Now: keep building patient modules, but avoid adding new required Rich args
   whenever possible.
2. Near-term: use the thin progress/log/event adapter layer in
  `python_files_dcm_meta_based/presentation/` so patient modules can run with
  Rich, a GUI, or a null/headless implementation.
3. After patient-runner validation: remove remaining Rich references from patient
   scientific functions and leave Rich only in legacy/main/frontend wrappers.

Existing Rich/progress/live-display arguments in patient functions are migration
debt, not placement failures. Do not stop patient-stage extraction just to remove
all of that debt upfront. Also do not route frozen legacy cohort wrappers through
new patient functions only to clean up Rich; preserve the oracle path until a
deliberate validation pass is ready.

Current boundary cleanup status:

- biopsy-processing patient modules now resolve missing progress/layout/live
  objects through `preprocessing/biopsy_processing/per_patient/_presentation.py`,
  so future runner calls can omit Rich/UI objects while legacy wrappers may still
  pass their existing presentation state,
- MC prep patient modules lazily import plotting helpers only behind existing
  diagnostic plot flags,
- optimizer-v2 and MC MR ADC scientific patient stages reject render/raw-dump
  side effects and keep those surfaces on their legacy adapters,
- older shared geometry helpers still accept legacy presentation-shaped objects;
  those should be refactored after patient-runner parity is stable, not during
  the current extraction pass.

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
- MC convex containment/dose simulation
- MC MR ADC localization
- Optimizer v1
- Optimizer-v2 target-DIL localization

### Has Patient Helpers But Still Missing The Full Patient Stage

- Inactive legacy MC dose statistics and voxelization helpers should only be
  extracted if downstream parity proves those artifacts are still required.

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

## Patient-Runner Tranche Plan

Patient scientific modules should remain standalone in their owning scientific
packages. Tranches are runner/orchestration recipes only: they group existing
patient modules into the legacy-compatible order without becoming new scientific
implementations.

Patient discovery is not a scientific tranche. Input directory scanning,
modality routing, UID discovery, case selection, prompts, and run input manifests
remain run-scoped discovery/bootstrap work. The first patient-runner boundary may
consume discovered patient case inputs and legacy-compatible reference fragments,
but it must not start owning DICOM discovery policy.

Dependency edges and pathway selection should be the scientific runner source of
truth. Tranches remain useful as debug/documentation blocks and manifest labels,
but they should be easy to remove and must not bypass graph validation. The
current dependency vocabulary and conservative hard-edge map live in
`docs/architecture/PATIENT_RUNNER_DEPENDENCY_GRAPH.md`; the full graph-node view,
current executable adapter view, pathway presets, and validation helpers live in
`python_files_dcm_meta_based/patient_runner/scientific_dependencies.py`.

Recommended tranche structure for patient-runner shadow work:

1. Compatibility bootstrap/reference boundary
   - Consumes already-discovered patient inputs, legacy key names, and selected
     patient IDs.
   - Builds or carves one-patient runtime/reference/info state for downstream
     patient stages.
   - Does not scan input directories, choose modalities, prompt users, or own
     cohort discovery policy.
2. Grid preprocessing tranche
   - Dose-grid runtime object construction.
   - MR ADC input normalization and MR ADC grid runtime object construction.
   - Any patient-local lattice/KD-tree/grid artifacts that are inputs to later
     anatomical, biopsy, optimizer, MC, or guidance stages.
   - This tranche intentionally precedes anatomical preprocessing because some
     structure processing consumes grid/lattice information that must already be
     available.
   - The current opt-in runner stage covers dose-grid runtime objects, MR ADC
     input normalization, and MR ADC grid runtime objects with render side
     effects rejected at the runner boundary.
3. Anatomical preprocessing tranche
   - Raw contour pulling and selected/unique structure setup.
   - OAR/prostate, rectum, urethra, and DIL non-biopsy structure preprocessing.
   - Prostate-only MR ADC structure summary after the relevant anatomical
     structures exist.
   - The current opt-in runner stage covers raw contour pulling, selected/unique
     structure selection, standard non-biopsy structure processing, and
     prostate-only MR ADC summary finalization with legacy presentation objects
     adapted to null/headless shims.
   - Stage-local validation sidecars remain validation/oracle scoped, not normal
     runner steps.
4. Biopsy preprocessing tranche
   - Real-biopsy geometry processing.
   - Simulated-biopsy target assignment, multiplicity expansion, length policy,
     preparation dataframe, and planning.
   - Uncertainty attachment once the resolved run-level uncertainty dataframe
     fragment is available.
5. Pre-optimizer transform and optimizer tranche
   - Transform-bank generation when required by optimizer/search behavior.
   - Optimizer-v1 and optimizer-v2 patient-local stages.
6. Post-optimizer biopsy realization tranche
   - Simulated-biopsy finalization, which the legacy path runs after optimizer-v2.
   - Planned-vs-realized centroid validation and post-optimizer biopsy
     annotations that depend on finalized simulated biopsy geometry.
   - Realized targeting if parity shows it belongs after finalization for the
     patient-runner recipe; otherwise keep it in the biopsy preprocessing tranche
     to match the legacy point of mutation.
7. Sampling and classification tranche
   - Sampled-biopsy processing and biopsy-frame coordinate storage.
   - Optimizer-v2 sampling audit annotation.
   - Double-sextant sample-point fragments plus run-level per-voxel assembly.
8. MC prep and simulation tranche
   - BX-only/self transforms and relative-structure transforms immediately
     before MC simulation.
   - Convex containment/dose simulation and MR ADC localization.
   - Downstream MC score/output annotations that depend on completed MC outputs.
9. Output, guidance, assembly, and parity tranche
   - Patient artifact writing, guidance-map precompute/render handoff if enabled,
     cohort table assembly, and post-run parity.
   - Cohort aggregation stays outside scientific patient modules.

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
- Patient MR localization/simulation module extracted from the patient-facing
  loop body inside `MC_simulator_MR.simulator_parallel(...)`
  purpose: reproduce one patient's MR ADC localization/simulation path after
  biopsy MC transforms already exist. `run_patient_mr_adc_localization_stage(...)`
  now provides this additive stage without raw CSV dumping, plotting, or Rich
  progress dependencies.
- Downstream patient MC annotation/output wrappers after patient-local sampling
  and MC outputs exist
  purpose: expose current downstream MC dataframe-fragment builders and optimizer
  annotations through singleton-patient wrappers while leaving final cohort table
  assembly run-scoped.

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

Completed through the 2026-05-27 pass:

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
- `preprocessing/biopsy_processing/per_patient/_presentation.py` owns the null/legacy presentation boundary for biopsy patient modules that still call older geometry helpers. Runner-facing calls should omit Rich/UI objects; legacy wrappers can still pass them during oracle validation.
- `attach_patient_uncertainty_data_from_dataframe(...)` lives in `python_files_dcm_meta_based/preprocessing/uncertainty_attachment.py` for future runner use with a resolved patient uncertainty dataframe fragment; template generation, file workflow, and cohort bookkeeping remain on the frozen run-scoped path.
- `process_patient_sampled_biopsies(...)` plus the patient-local sampling-arg, sampled-result storage, and biopsy-coordinate helpers live in `python_files_dcm_meta_based/preprocessing/biopsy_processing/sampled_biopsy_processing.py`; the cohort wrapper remains on its frozen body.
- `build_patient_simulated_biopsy_centroid_variation_validation_fragment(...)` and `assemble_simulated_biopsy_centroid_variation_validation_fragments(...)` live in `python_files_dcm_meta_based/preprocessing/biopsy_processing/per_patient/centroid_variation_validation.py` for future runner use; the cohort validation wrapper remains on its frozen body.
- `build_patient_biopsy_double_sextant_sample_point_fragment(...)`, `assemble_biopsy_double_sextant_classification_fragments(...)`, and `store_patient_biopsy_double_sextant_voxel_fragment(...)` live in `python_files_dcm_meta_based/preprocessing/biopsy_processing/per_patient/double_sextant_classification.py` for future runner use; the cohort wrapper remains on its frozen body. The per-voxel table remains assembled from patient sample-point fragments to preserve legacy aggregation and random-tie behavior.
- `generate_transformations_for_patient(...)`, `apply_patient_biopsy_self_transforms(...)`, and `apply_patient_relative_structure_transforms(...)` live in `python_files_dcm_meta_based/mc/prep/per_patient/` as additive patient-local MC prep surfaces; the frozen cohort wrappers in `MC_prepper_funcs.py` remain on their original bodies.
- `MCReferenceKeys`, `MCContainmentSimulationConfig`, `MCDoseSimulationConfig`, `MCMRSimulationConfig`, `LegacyMCKeyBundle`, containment/dose/MR output collectors, a neutral patient relative-structure inventory module, containment biopsy/input/dilation-bank helpers, containment core wrappers around the existing helper/kernel APIs, per-biopsy containment statistics/writeback helpers, dose/dose-gradient localization helpers, DVH compile/writeback helpers, `run_patient_mc_convex_stage(...)`, `run_patient_mr_adc_localization_stage(...)`, downstream MC output table/annotation wrappers, `run_patient_mc_convex_legacy_adapter(...)`, and `run_patient_mc_mr_legacy_adapter(...)` live in `python_files_dcm_meta_based/mc/simulation/per_patient/` as the additive patient-local MC simulation landing zone; `MC_simulator_convex.simulator_parallel(...)` and `MC_simulator_MR.simulator_parallel(...)` remain untouched as the oracles.
- `ProgressEvent`, `ProgressSink`, null legacy shims, and `RichProgressSink` live in `python_files_dcm_meta_based/presentation/` as a presentation-neutral adapter surface for future patient modules.
- `build_patient_structure_reference_bootstrap_fragment(...)`, `PatientStructureReferenceState`, `PatientStructureInfoState`, typed registries, run-level assembly, and dose/plan/MR attachment helpers live in `python_files_dcm_meta_based/preprocessing/structure_reference_bootstrap.py`; the frozen `structure_referencer(...)` oracle remains untouched.
- `python_files_dcm_meta_based/legacy_data_keys.py` owns generic legacy master-info, patient-reference, structure-record, structure-metadata, structure-geometry, biopsy runtime/sample, nested dataframe-store, and artifact sentinel spellings for additive adapters; older live mutation wrappers remain deferred until they cross a patient/runner/artifact boundary.
- `run_patient_optimizer_v1_stage(...)` lives in `python_files_dcm_meta_based/biopsy_optimizer/v1/per_patient/patient_stage.py` as the additive patient-local scientific stage copied from the optimizer-v1 patient loop. `run_patient_optimizer_v1_legacy_adapter(...)` remains a singleton-patient validation bridge around the legacy optimizer-v1 oracle.
- `run_patient_target_dil_optimizer_v2_live_adapter(...)` lives in `python_files_dcm_meta_based/biopsy_optimizer/v2/per_patient/live_adapter.py` as a singleton-patient oracle bridge around the current optimizer-v2 live integration surface. `run_patient_target_dil_optimizer_v2_stage(...)` lives in `python_files_dcm_meta_based/biopsy_optimizer/v2/per_patient/target_dil_stage.py` as the additive patient-local scientific stage built from the existing optimizer-v2 candidate-pool, scoring, staged-runner, and output modules; render/UI review remains outside that scientific surface.
- `precompute_guidance_map_firing_depth_recommendations_for_patient(...)` lives in `python_files_dcm_meta_based/guidance_maps/planning.py`; the run wrapper now loops over this patient entrypoint and keeps rendering run/UI scoped.
- `patient_runner/scientific_config.py` and `patient_runner/scientific_stages.py` now expose an opt-in grid-preprocessing runner boundary for dose-grid runtime object construction, MR ADC input normalization, and MR ADC grid runtime object construction before anatomical preprocessing.
- `patient_runner/scientific_config.py` and `patient_runner/scientific_stages.py` now expose an opt-in anatomical-preprocessing runner boundary for raw contour pulling, selected/unique structure selection, standard non-biopsy structure processing, and prostate-only MR ADC summary finalization before biopsy-facing preprocessing.

## Main Pipeline Readiness Checklist

| Order | Pipeline stage | Main-facing function/module | Current status | Remaining work |
| --- | --- | --- | --- | --- |
| 1 | Input discovery and modality routing | Inline in `biopsy_localization_convex_main.py`; `UID_generator(...)` | Assembly | Keep run-scoped. Later extract discovery into a run manifest builder that emits patient case inputs. |
| 2 | Structure reference/bootstrap dictionaries | `preprocessing.structure_reference_bootstrap.build_patient_structure_reference_bootstrap_fragment(...)`, `structure_referencer(...)` | Complete | Additive patient bootstrap surface exists with typed patient reference/info states and legacy dict adapters. Validate row/key parity against the frozen `structure_referencer(...)` oracle before routing a full patient runner through it. |
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
| 15 | Real biopsy preprocessing | `preprocessing.biopsy_processing.per_patient.real_biopsy_processing.process_patient_real_biopsies(...)`, `real_biopsy_processer(...)` | Complete | Additive patient module exists in the owning biopsy-processing family. Runner-facing calls no longer require layout/progress/live-display objects; a null presentation boundary adapts older geometry helpers internally. Current cohort wrapper remains frozen. |
| 16 | Simulated biopsy preparation: target assignment | `preprocessing.biopsy_processing.per_patient.simulated_biopsy_preparation.assign_patient_simulated_biopsy_targets(...)`, `assign_simulated_biopsy_targets(...)` | Complete | Additive patient target-assignment module exists in the owning biopsy-processing family. Current cohort wrapper remains frozen. |
| 17 | Simulated biopsy preparation: multiplicity expansion | `preprocessing.biopsy_processing.per_patient.simulated_biopsy_preparation.expand_patient_simulated_biopsy_multiplicity(...)`, `expand_simulated_biopsy_multiplicity(...)` | Complete | Additive patient multiplicity expansion exists. Keep the legacy cohort-wide biopsy-count refresh in the frozen cohort wrapper or later assembly layer. |
| 18 | Simulated biopsy preparation: length policy | `preprocessing.biopsy_processing.per_patient.simulated_biopsy_preparation.determine_patient_simulated_biopsy_lengths(...)`, `determine_simulated_biopsy_lengths(...)` | Complete | Additive patient-level length determination exists and stays on patient-local data only; cohort-derived fallback methods remain excluded. |
| 19 | Simulated biopsy preparation dataframe | `preprocessing.biopsy_processing.per_patient.simulated_biopsy_preparation.build_patient_simulated_biopsy_preparation_dataframe(...)`, `simulated_biopsy_preparation_dataframe_builder(...)` | Complete | Additive patient preparation-fragment builder exists. Future cohort output assembly can concatenate patient fragments after row-order validation. |
| 20 | Simulated biopsy planning | `preprocessing.biopsy_processing.per_patient.simulated_biopsy_planning.plan_patient_simulated_biopsies(...)`, `simulated_biopsy_planner_processer(...)` | Complete | Additive patient module exists in the owning biopsy-processing family. Runner-facing calls no longer require progress objects. Current cohort wrapper remains frozen; future runner may replace `parallel_pool` with a sequential patient execution context. |
| 21 | Uncertainty generation and attachment | `preprocessing.uncertainty_attachment.attach_patient_uncertainty_data_from_dataframe(...)`, `prepare_and_attach_uncertainty_data(...)`, `attach_uncertainty_data_from_dataframe(...)` | Complete | Additive patient-level attachment from a resolved uncertainty dataframe fragment exists. Keep template generation, file prompts, and run-level dataframe bookkeeping on the frozen run-scoped path. |
| 22 | Transform generation/prep | `mc.prep.per_patient.transform_generation.generate_transformations_for_patient(...)`, `mc.prep.per_patient.biopsy_self_transforms.apply_patient_biopsy_self_transforms(...)`, `mc.prep.per_patient.relative_structure_transforms.apply_patient_relative_structure_transforms(...)`, `MC_prepper_funcs.generate_transformations(...)`, `MC_prepper_funcs.biopsy_only_transformer(...)`, `MC_prepper_funcs.biopsy_transformer_to_relative_structures(...)` | Complete | Additive patient-level MC prep surfaces now exist in the dedicated MC prep scientific package. Frozen cohort wrappers in `MC_prepper_funcs.py` remain on their original bodies and are not routed through the patient entrypoints. |
| 23 | Optimizer v1 | `biopsy_optimizer.v1.per_patient.patient_stage.run_patient_optimizer_v1_stage(...)`, `biopsy_optimizer.v1.per_patient.legacy_adapter.run_patient_optimizer_v1_legacy_adapter(...)`, `biopsy_optimizer_module_v1(...)` | Complete | Additive patient-local scientific stage now owns the copied patient loop, per-DIL lattice containment/scoring, guidance-map max-plane output, and patient writeback. Plotting/debug/log side effects are rejected from the scientific stage and remain on the legacy adapter/oracle path. |
| 24 | Optimizer v2 live integration | `biopsy_optimizer.v2.per_patient.target_dil_stage.run_patient_target_dil_optimizer_v2_stage(...)`, `biopsy_optimizer.v2.per_patient.live_adapter.run_patient_target_dil_optimizer_v2_live_adapter(...)`, `run_target_dil_optimizer_v2_for_live_simulated_family(...)` | Complete | Additive patient-local optimizer-v2 scientific stage now calls the existing candidate-pool, scoring, staged-runner, and output modules for one patient. The live adapter remains available as the singleton oracle bridge; render/UI review stays out of the scientific patient stage. |
| 25 | Simulated biopsy finalization | `preprocessing.biopsy_processing.per_patient.simulated_biopsy_processing.process_patient_simulated_biopsies(...)`, `simulated_biopsy_processer(...)` | Complete | Additive patient finalization module exists in the owning biopsy-processing family. Runner-facing calls no longer require layout/progress/live-display objects; a null presentation boundary adapts older geometry helpers internally. Current cohort wrapper remains frozen and is not routed through the patient module. |
| 26 | Planned-vs-realized centroid validation | `preprocessing.biopsy_processing.per_patient.centroid_variation_validation.build_patient_simulated_biopsy_centroid_variation_validation_fragment(...)`, `preprocessing.biopsy_processing.per_patient.centroid_variation_validation.assemble_simulated_biopsy_centroid_variation_validation_fragments(...)`, `validate_simulated_biopsy_planned_vs_realized_centroid_variation(...)` | Complete | Additive patient fragment builder plus run-level summarizer exist. Current cohort validation wrapper remains frozen and is not routed through the patient module. |
| 27 | Realized biopsy targeting | `preprocessing.biopsy_processing.per_patient.realized_biopsy_targeting.determine_patient_realized_biopsy_targeting(...)`, `realized_biopsy_targeting_processer(...)` | Complete | Additive patient module exists in the owning biopsy-processing family. Runner-facing calls no longer require progress objects. Current cohort wrapper remains frozen. |
| 28 | Pickled preprocessed bundle export/load | `export_preprocessed_pickle_bundle(...)`, `load_selected_pickle_bundle_run(...)` | Assembly | Keep run-scoped. Future patient runner can consume patient case fragments from bundles. |
| 29 | Sampled biopsy processing | `preprocessing.biopsy_processing.sampled_biopsy_processing.process_patient_sampled_biopsies(...)`, `sampled_biopsy_processing_processer(...)` | Complete | Additive patient stage now exists in the owning biopsy-processing module and preserves the current per-patient sampling order by composing patient-local sampling-arg, sampled-result storage, and biopsy-coordinate helpers. Current cohort wrapper remains frozen. |
| 30 | Prostate double-sextant classification | `preprocessing.biopsy_processing.per_patient.double_sextant_classification.build_patient_biopsy_double_sextant_sample_point_fragment(...)`, `preprocessing.biopsy_processing.per_patient.double_sextant_classification.assemble_biopsy_double_sextant_classification_fragments(...)`, `biopsy_double_sextant_processer(...)` | Complete | Additive patient sample-point fragment builder plus run-level per-voxel summarizer exist. Per-voxel aggregation stays run-level to preserve legacy random-tie behavior. Current cohort wrapper remains frozen. |
| 31 | MC simulation | `mc.simulation.per_patient.run_patient_mc_convex_stage(...)`, `mc.simulation.per_patient.run_patient_mr_adc_localization_stage(...)`, `mc.simulation.per_patient.run_patient_mc_convex_legacy_adapter(...)`, `mc.simulation.per_patient.run_patient_mc_mr_legacy_adapter(...)`, `MC_simulator_convex.simulator_parallel(...)`, `MC_simulator_MR.simulator_parallel(...)` | Complete | True additive patient-local stages now exist for active convex containment/dose plus MR ADC localization. The convex stage composes the package-local relative-structure inventory, containment setup/input builders, containment core wrappers, per-biopsy containment statistics/writeback helpers, dose/dose-gradient localization helpers, and DVH compile/writeback helpers while leaving singleton convex/MR oracle adapters available for comparison. Remaining work is runner sequencing and parity validation; extract inactive/legacy dose statistics or voxelization only if downstream parity requires them. |
| 32 | Cohort simulated-biopsy preparation table | `dataframe_builders.cohort_simulated_biopsy_preparation_dataframe_builder(...)` | Assembly | Replace with concatenation of patient preparation fragments after row-order validation. Patient fragment building now exists in the owning biopsy-processing family. |
| 33 | Guidance map precompute/render | `precompute_guidance_map_firing_depth_recommendations_for_patient(...)`, `precompute_guidance_map_firing_depth_recommendations_for_run(...)`, `render_guidance_maps_for_run(...)` | Complete | Additive patient precompute entrypoint exists and the run wrapper calls it. Keep rendering run/UI scoped unless a future product surface needs a dedicated adapter. |
| 34 | Optimizer-v2 downstream annotations | `mc.simulation.per_patient.annotate_patient_optimizer_v2_outputs_with_downstream_mc_scores(...)`, `annotate_target_dil_optimizer_v2_outputs_with_downstream_mc_scores(...)`, `annotate_target_dil_optimizer_v2_outputs_with_biopsy_sampling_audit(...)` | Complete | MC-score annotation has a singleton-patient wrapper. Biopsy-sampling audit annotation is already downstream of patient sampled-biopsy fragments and can remain with the optimizer-v2 adapter until runner wiring decides where to call it. |
| 35 | Final dataframe builders | `dataframe_builders.*` calls in main | Assembly | Move final cohort tables to assembly from patient base artifacts. Avoid schema churn during migration. |
| 36 | Phase3B/Phase3C output surface | `build_in_memory_stitch_validation(...)`, `write_phase3c_output_surface(...)` | Assembly | Existing patient-fragment output/assembly surface is validated as a shadow gate, not scientific recomputation. |
| 37 | Patient-runner validation gate and post-run parity | `run_patient_runner_main_validation(...)`, `run_patient_runner_post_run_parity(...)`, `compare_patient_runner_parity.py` | Assembly | Shadow-output validation already checks artifact/export/assembly from completed legacy state. Post-run parity now reuses existing CSV comparators after independent completed runs, starting with legacy cohort CSVs vs patient-runner assembled cohort tables. Keep both outside normal scientific execution until patient stages are independently runnable. |
| 38 | Run completion manifest | `write_run_completion_manifest(...)` | Assembly | Keep run-scoped; patient manifests are handled by `patient_runner`. |

## Next Implementation Order

Recommended order of operations from this point:

1. Keep patient discovery explicitly outside the scientific dependency graph.
  Pathways may consume discovered patient cases, manifests, legacy key names,
  and carved runtime state; they must not own DICOM discovery, modality routing,
  prompts, or cohort case selection.
2. Maintain the patient-runner scientific dependency graph and named pathway
  presets as the gate for more stage adapters. Pathways should express
  intentional scientific workflows such as anatomical QA, optimization shadow,
  current dosimetry shadow, or full current pipeline shadow.
3. Keep tranches as removable debug/documentation groupings over graph nodes.
  Tranches may consume discovered patient cases, manifests, legacy key names,
  and carved runtime state; they must not own DICOM discovery, modality routing,
  prompts, or cohort case selection.
4. Extend the runner scientific config boundary in
  `patient_runner/scientific_config.py` so it can map legacy main config values
  into separate grid preprocessing, anatomical preprocessing, biopsy
  preprocessing, optimizer, MC prep/simulation, guidance, and output/parity
  tranche config groups.
5. Add the missing adapters that let the executable pathway view converge toward
  the full graph view, especially transform generation, simulated-biopsy
  finalization, and sampling/classification. Current adapters cover grid
  preprocessing, anatomical preprocessing, real-biopsy processing,
  simulated-biopsy preparation, simulated-biopsy planning, uncertainty
  attachment, realized targeting, sampled-biopsy processing, MC prep, MC
  simulation, optimizer, and guidance.
6. Wire simulated-biopsy finalization as a separate post-optimizer stage because
  the legacy path runs it after optimizer-v2, not inside early preprocessing.
7. Add a separate scientific shadow-validation mode after the graph/pathway
  sequence is configured from main. Keep `SHADOW_OUTPUT` as
  artifact/export/assembly validation from completed legacy state.
8. Add durable stage-state parity manifests beside patient artifacts: performed
  flags, skip reasons, output keys present, counts, dataframe shapes, and hashes
  where useful.
9. Compare independently run patient-stage outputs against the frozen cohort
  oracle through post-run parity surfaces before live routing.
9. After patient-runner validation against the legacy cohort oracle, remove
  remaining Rich dependencies from patient scientific functions and keep Rich
  only in legacy/main/frontend wrappers.
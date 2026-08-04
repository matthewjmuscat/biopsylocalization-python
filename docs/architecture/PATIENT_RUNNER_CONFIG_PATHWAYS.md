# Patient Runner Config Pathways

Last updated: 2026-08-03

## Purpose

This document maps the current configuration pathways before the next config
rewrite pass. It is intentionally an investigation artifact and rewrite guide,
not an implementation spec that claims the current code is already clean.

The immediate goal is to prevent the patient runner, scientific-shadow lane, and
future GUI from inheriting the current main-local configuration sprawl as their
permanent language.

Use this map when changing config surfaces that touch:

- `biopsy_localization_convex_main.py`
- `python_files_dcm_meta_based/config/pipeline.py`
- `python_files_dcm_meta_based/patient_runner/scientific_config.py`
- per-patient optimizer, MC, preprocessing, and guidance adapters

## Investigated Surfaces

Current durable config surfaces:

- `config/pipeline.py`: root `PipelineConfig` plus UI, artifact, preprocessing,
  replay, guidance, optimizer runtime, random seed, and validation-sidecar
  groups.
- `patient_runner/scientific_config.py`: patient-runner opt-in stage config that
  already matches the current scientific stage graph.
- `patient_runner/main_validation.py`: main-facing patient-runner validation
  config and mode selection.
- `patient_runner/scientific_shadow.py`: pathway-level scientific-shadow config
  and evidence/manifest controls.
- `preprocessing/dose_grid_processing.py` and
  `preprocessing/mr_adc_grid_processing.py`: grid-stage config leaves with
  render flags still embedded.
- `preprocessing/structure_processing/non_biopsy_structure_processing.py`:
  anatomical preprocessing config with scientific geometry, kernel policy, and
  debug/render flags mixed together.
- `biopsy_optimizer/v1/per_patient/legacy_adapter.py`: optimizer-v1 adapter
  config with core search settings and presentation/debug side effects mixed
  together.
- `biopsy_optimizer/v2/config.py`: optimizer-v2 search and visualization policy.
- `biopsy_optimizer/v2/per_patient/live_adapter.py`: optimizer-v2 live adapter
  config with search, calibration, validation, render, and anatomy-reference
  fields mixed together.
- `mc/simulation/per_patient/contracts.py`: MC reference/runtime/containment/dose
  and MR simulation config contracts.
- `guidance_maps/config.py` and `startup/guidance_map_workflow.py`: guidance
  planning and render config.

Current transitional legacy config carriers:

- `master_structure_info_dict["Global"]["Preprocessing info"]`
- `master_structure_info_dict["Global"]["MC info"]`
- `master_structure_info_dict["Global"]["Random info"]`
- `master_structure_info_dict["Global"]["Specific output dir"]`
- `master_structure_info_dict["Global"]["Raw MC output dir"]`

These legacy dictionaries still need to be written during oracle validation, but
they should be treated as run metadata and compatibility snapshots, not as the
future config authority.

## Current Main Flow

The main file currently performs four separate config roles in one place:

1. Declares legacy dictionary key names and scientific labels.
2. Declares many user/runtime/science/debug local variables.
3. Builds a partial `PipelineConfig` from only some of those locals.
4. Passes many remaining locals directly into scientific modules.

Current high-level flow:

```text
main-local constants and knobs
  -> PipelineConfig subset
      -> preprocessing/adapters/guidance/replay/random/validation sidecars
  -> loose direct call arguments
      -> optimizer v1/v2
      -> MC transform generation and MC simulation
      -> simulated-biopsy prep/planning/finalization
      -> sampled-biopsy processing
      -> output table generation and validation surfaces
  -> legacy Global dictionaries
      -> Preprocessing info
      -> MC info
      -> Random info
      -> output directory metadata
      -> downstream modules that still read legacy config/status
```

`PipelineConfig` is therefore useful, but it is not yet the full config root.
The highest-risk rewrite failure would be adding a patient-runner bridge that
copies every loose local into `PatientRunnerScientificConfig` directly. That
would make the new runner depend on the same scattered source of truth that the
config rewrite is meant to replace.

## Target Run Profile Boundary

The standalone patient runner needs a run profile, but that profile should be
the orchestration layer above scientific config, not a second source of
scientific truth.

Target flow:

```text
human TOML run profile
  -> typed resolved run plan
  -> input discovery or retained input manifests
  -> standalone patient-runner jobs
  -> optional legacy-oracle job
  -> post-run assembly
  -> validation
```

The run profile should choose what to execute and what evidence to produce:

- input folder or input manifest source,
- selected patients/fractions,
- pathway and checkpoint names,
- whether to run standalone patient workers,
- whether to run the legacy oracle,
- patient-first execution order when both standalone and legacy are requested,
- post-run cohort assembly jobs,
- validation jobs,
- output root, run labels, failure/retry policy, and resource limits.

Scientific defaults and model parameters should continue moving into
`PipelineConfig` and domain configs. The run profile can reference or embed a
resolved scientific config, but it should not copy the hundreds of loose
`main` locals into a parallel TOML schema.

The future GUI should edit this run profile and call public runner/assembly/
validation entrypoints. It should not depend on `biopsy_localization_convex_main.py`
locals or private in-memory dictionaries.

Current TOML status: TOML profiles have started in the validation layer, where
they select completed run folders and comparator jobs. They do not yet define
production scientific run parameters. The production run-profile layer should
follow the same pattern later: human TOML at the edge, typed Python config as
the runtime authority, and JSON as generated evidence/manifest output.

## Current Config Tree By Domain

### Startup, Inputs, And Legacy Keys

Current source:

- legacy key locals: `all_ref_key`, `bx_ref`, `by_patient_key`, `global_key`,
  `global_num_cases_key`, `oar_ref`, `dil_ref`, `rectum_ref_key`,
  `urethra_ref_key`
- input folders and modality discovery names
- `fraction_prefixes`
- data-removal dictionaries for biopsy, DIL, prostate, urethra, and rectum
- contour-name lists for each structure family

Current consumers:

- input discovery and DICOM parsing
- raw contour pulling and structure selection
- legacy bridge and `LegacyRuntimeKeys`
- patient-runner carved runtime state

Target owner:

```text
PipelineConfig
  input
  legacy_keys
  structure_taxonomy
  data_removals
```

Rewrite note: key names should continue to use existing typed key bundles at
runner boundaries. Do not duplicate `Global`, `By patient`, `All ref`, `Bx ref`,
or `Num cases` string literals in new runner/config code.

### Structure Registry And Tissue Labels

Current source:

- contour family lists
- uncertainty defaults per structure family
- tissue-class labels and hierarchy fields inside `structs_referenced_dict`
- `tissue_volume_operator_dictionary`
- plotting colors for real and simulated biopsy types

Current consumers:

- raw contour pulling
- selected-structure selection
- non-biopsy preprocessing
- real/simulated biopsy processing
- MC containment and downstream dataframe builders
- output artifacts and QA-facing tables

Target owner:

```text
PipelineConfig
  structure_registry
    references
    contour_matching
    uncertainty_defaults
    tissue_labels
    tissue_volume_policy
    colors
```

Rewrite note: `structs_referenced_dict` can remain the legacy emission format,
but the editable config should be a typed structure registry. The bridge should
produce the legacy dict as an adapter output.

### Artifacts, Replay, And Output Roots

Current source:

- `ArtifactConfig`
- `RuntimeReplayConfig`
- `FrozenPreprocessedBundleConfig`
- specific output dir and raw MC output dir written into legacy global info

Current consumers:

- output directory creation
- preprocessed pickle export/load
- replay rebuild of non-picklable runtime objects
- patient-runner validation output roots
- guidance-map render output folders

Target owner:

```text
PipelineConfig
  artifacts
    output_roots
    preprocessed_bundle
    export_policy
  replay
    display_thresholds
    loaded_bundle_mismatch_policy
```

Rewrite note: frozen-with-artifact values and runtime-overridable replay values
must stay separate. A future GUI can edit replay thresholds without changing the
scientific meaning of a saved preprocessed bundle.

### Grid Preprocessing

Current source:

- `DoseGridProcessingConfig`
- `MRADCGridProcessingConfig`
- MR ADC input normalization config in `PatientMRADCInputNormalizationStageConfig`
- replay display thresholds in `RuntimeReplayConfig`

Current consumers:

- legacy cohort grid builders
- patient-runner grid preprocessing stage
- downstream dose and MR ADC simulation

Target owner:

```text
PipelineConfig
  preprocessing
    grid
      dose
      mr_adc
      mr_adc_input_normalization
      debug
        dose_render
        mr_adc_render
```

Debug subgroup candidates:

- `show_3d_dose_renderings`
- `show_3d_dose_renderings_thresholded`
- `show_3d_mr_adc_renderings`
- `show_3d_mr_adc_renderings_thresholded`

Rewrite note: patient-runner grid config currently rejects render side effects.
The target config should keep render/debug choices separate from the core grid
settings so patient-runner scientific stages can accept only the core slice.

### Anatomical And Non-Biopsy Preprocessing

Current source:

- `PreprocessingConfig`
- `NonBiopsyStructurePreprocessingConfig`
- structure references and `structs_referenced_dict`
- MR ADC containment demonstration flags
- CUDA log and geometry debug flags

Current consumers:

- legacy main non-biopsy processing
- patient-runner anatomical preprocessing
- frozen preprocessed bundle snapshot
- guidance-map precompute
- output table generation

Target owner:

```text
PipelineConfig
  preprocessing
    interpolation
    geometry
    volume_and_dimension
    kernel_policy
    anatomical
    debug
      cuda_logs
      volume_demonstrations
      dimension_demonstrations
      mr_adc_demonstrations
      mesh_display
      binary_masks
      ellipsoid_display
```

Debug subgroup candidates:

- `generate_cuda_log_files_volume_calculation`
- `generate_cuda_log_files_structure_dimension_calculation`
- `demonstrate_volume_calculation_correctness_bool_1`
- `plot_volume_calculation_containment_result_bool_1_old`
- `plot_binary_mask_bool`
- `demonstrate_structure_dimension_calculation_correctness_bool_1`
- `demonstrate_structure_dimension_calculation_correctness_bool_1_old`
- `demonstrate_mr_adc_pcd_containment_correctness_bool`
- `display_structure_surface_mesh_bool`
- `show_equivalent_ellipsoid_from_pca_bool`

Rewrite note: this domain is already partly in `PipelineConfig`, but the fields
are flat. The next pass should split core geometry/kernel settings from debug
and presentation toggles without changing values.

### Biopsy Geometry, Sampling, And Simulated Biopsies

Current source:

- biopsy radius and needle dimensions
- `bx_sample_pts_lattice_spacing`
- simulated-biopsy type registry in `bx_sim_locations_dict`
- simulated-biopsy fraction policy
- simulated-biopsy length method
- centroid/optimal/target-v2 simulated biopsy keys
- simulated-biopsy planning radius and centroid-line defaults
- sampled-biopsy display flag

Current consumers:

- real-biopsy processing
- simulated-biopsy preparation
- simulated-biopsy planning
- optimizer-v1/v2 simulated biopsy producer behavior
- simulated-biopsy finalization
- sampled-biopsy processing
- MC transform application and simulation
- QA/output tables

Target owner:

```text
PipelineConfig
  biopsy
    physical_geometry
    sampling
    simulated_biopsy
      type_registry
      fraction_policy
      length_policy
      planning
      producer_policy
    debug
      planned_core_plotting
      reconstructed_biopsy_display
      centroid_variation_validation
```

Debug subgroup candidates:

- `plot_simulated_cores_immediately`
- `show_reconstructed_biopsy_in_biopsy_coord_sys_tr_and_rot`
- planned-vs-realized centroid validation output toggle, if made optional later

Rewrite note: simulated-biopsy finalization should receive a producer contract,
not an optimizer-specific assumption. The current post-optimizer and dosimetry
shadow pathways use optimization as the producer because that is the legacy
order. Realized targeting belongs after finalization, once both real and
simulated biopsy geometry have centroid/shape state.

### Transform Generation And Random Seeds

Current source:

- `RandomSeedConfig`
- `num_stochastic_targeting_transform_samples_input`
- optimizer-v2 search config, which determines required transform-bank size
- MC simulation counts, which determine downstream transform needs
- legacy `Global` `MC info` and `Random info`

Current consumers:

- `configure_transform_precompute_settings`
- `configure_runtime_random_seed_settings`
- `configure_transform_generation_counts`
- transform-bank generation
- optimizer-v1 seed application
- optimizer-v2 search and downstream-comparable scoring
- MC prep

Target owner:

```text
PipelineConfig
  random_seeds
  transforms
    required_samples
    consumers
      optimizer_v2
      stochastic_targeting
      mc_containment
      mc_dose
      mc_mr
    debug
      generated_sample_manifest
```

Rewrite note: legacy `MC info` can keep a copied snapshot for downstream
compatibility, but required transform counts should be computed from typed config
and written outward, not discovered by reading the legacy global dict first.

### Optimizer V1

Current source:

- loose main locals for optimizer-v1 lattice/search policy
- `OptimizerV1LegacyConfig`
- global random seed for optimizer-v1
- debug/presentation booleans in the same adapter config

Current consumers:

- legacy optimizer-v1 cohort function
- patient optimizer-v1 adapter
- optimizer output dataframe builders
- guidance-map max-plane outputs

Target owner:

```text
PipelineConfig
  optimization
    optimizer_v1
      search
      geometry
      runtime_limits
      debug
        containment_demonstrations
        lattice_plots
        point_plots
        cuda_logs
        contour_display
```

Debug subgroup candidates:

- `plot_each_normal_dist_containment_result_bool`
- `plot_optimization_point_lattice_bool`
- `show_optimization_point_bool`
- `demonstrate_dil_optimization_points_inside_correctness_bool_1`
- `demonstrate_dil_optimization_points_inside_correctness_bool_2`
- `demonstrate_dil_optimization_points_inside_correctness_num_3`
- `generate_cuda_log_files_biopsy_optimizer`
- `display_optimization_contour_plots_bool`

Rewrite note: the patient-runner optimizer-v1 stage currently rejects side-effect
options. Splitting debug fields allows GUI/debug launches to opt in explicitly
while scientific shadow uses the side-effect-free core config.

### Optimizer V2

Current source:

- `OptimizerV2SearchConfig`
- adaptive block parameters in main
- calibration/chunk settings in main
- validation and benchmark toggles in main
- render and Plotly export settings in main
- `OptimizerV2LiveConfig`

Current consumers:

- live optimizer-v2 integration
- patient optimizer-v2 adapter
- transform-bank sample sizing
- optimized simulated-biopsy producer contract
- optional render/export flows
- downstream MC-score annotations

Target owner:

```text
PipelineConfig
  optimization
    optimizer_v2
      search
      capacity
      calibration
      validation
      benchmarking
      render
        stage_clouds
        winner_debug
        plotly_export
        filters
        layer_styles
      anatomy_refs
```

Debug subgroup candidates:

- `optimizer_v2_validate_nearest_z_helper_against_ver5_bool`
- `optimizer_v2_benchmark_isolated_winner_validation_bool`
- `optimizer_v2_render_stage_boundary_candidate_clouds_bool`
- `optimizer_v2_render_stage_names`
- `optimizer_v2_render_backend`
- `optimizer_v2_render_plotly_export_bool`
- `optimizer_v2_render_dialog_timeout_seconds`
- `optimizer_v2_render_winner_containment_debug_bool`
- `optimizer_v2_render_patient_whitelist`
- `optimizer_v2_render_roi_whitelist`
- `optimizer_v2_render_layer_style_by_name`

Rewrite note: optimizer-v2 is the clearest example where subgroups matter. Search
policy, capacity/calibration, validation/benchmarking, and render/export should
not be one flat config. The future GUI should be able to hide render/debug
subgroups during normal scientific runs.

### MC Prep And MC Simulation

Current source:

- MC simulation counts
- biopsy transform flags
- `MCSimulationRuntimeConfig`
- `MCContainmentSimulationConfig`
- `MCDoseSimulationConfig`
- `MCMRSimulationConfig`
- raw data dump flags
- display/demo flags
- CUDA log flags
- legacy `MC info`

Current consumers:

- transform generation and MC prep
- legacy `MC_simulator_convex.simulator_parallel`
- legacy `MC_simulator_MR.simulator_parallel`
- per-patient MC adapters
- dataframe builders
- optimizer-v2 downstream MC-score annotation

Target owner:

```text
PipelineConfig
  simulation
    mc
      counts
      prep
      containment
      dose
      mr_adc
      tissue_classification
      nearest_neighbor
      output_policy
      debug
        raw_dumps
        containment_demonstrations
        dose_demonstrations
        mr_demonstrations
        transform_inspection
        plotly_shift_checks
        cuda_logs
```

Debug subgroup candidates:

- `inspect_self_biopsy_dilate_bool`
- `inspect_self_biopsy_dilate_and_rotate_bool`
- `inspect_self_biopsy_dilate_and_rotate_and_translate_bool`
- `inspect_relative_structure_rotate_and_shift_number`
- `plot_uniform_shifts_to_check_plotly`
- `plot_translation_vectors_pointclouds`
- `plot_shifted_biopsies`
- `show_NN_dose_demonstration_plots`
- `show_NN_dose_demonstration_plots_all_trials_at_once`
- `show_num_containment_demonstration_plots`
- `plot_cupy_containment_distribution_results`
- `show_num_nearest_neighbour_surface_boundary_demonstration`
- `show_num_relative_structure_centroid_demonstration`
- `show_NN_mr_adc_demonstration_plots`
- `show_NN_mr_adc_demonstration_plots_all_trials_at_once`
- `raw_data_mc_dosimetry_dump_bool`
- `raw_data_mc_containment_dump_bool`
- `raw_data_mc_MR_dump_bool`
- `generate_cuda_log_files_MC_containment_sim`

Rewrite note: raw dump controls should be grouped under output/debug policy
because they can create very large disk outputs and should not be accidentally
visible as ordinary scientific settings.

### Guidance Maps

Current source:

- `GuidanceMapPlanningConfig`
- `GuidanceMapRenderConfig`
- biopsy firing distances and needle geometry
- interpolation/normal-estimation settings
- render validation and strictness flags

Current consumers:

- patient guidance precompute
- cohort guidance dataframe builder
- guidance rendering workflow
- output directory metadata

Target owner:

```text
PipelineConfig
  guidance_maps
    planning
    render
    validation
    debug
      transducer_plane_demo
```

Debug subgroup candidates:

- `plot_guidance_map_transducer_plane_open3d_structure_set_complete_demonstration_bool`
- `validate_firing_df_builder_behavior`
- render rank behavior and strict precomputed-guidance behavior can be GUI-visible
  runtime controls, but should remain separate from planning science.

Rewrite note: guidance planning is patient/science scoped. Guidance rendering is
run/UI scoped and should not be required by patient-runner scientific shadow.

### Validation, Shadow Runs, And Output Surfaces

Current source:

- `ValidationSidecarConfig`
- Phase 3B and 3C booleans
- `PatientRunnerMainValidationConfig`
- `PatientScientificShadowConfig`
- output artifact writing flags

Current consumers:

- selected-structure legacy validation
- non-biopsy legacy validation
- prostate-only MR ADC legacy validation
- in-memory stitch validation
- patient-fragment output surface
- patient-runner shadow output validation
- patient-runner scientific shadow validation

Target owner:

```text
PipelineConfig
  validation
    preprocessing_sidecars
    output_surface
    patient_runner
      shadow_output
      scientific_shadow
    evidence
      manifests
      dataframe_snapshots
      hashes
```

Debug subgroup candidates:

- legacy sidecar toggles
- Phase 3B/3C write toggles
- patient-runner output filters
- scientific-shadow state isolation and manifest controls
- dataframe snapshot controls

Rewrite note: validation config is not scientific config. It should wrap a
scientific config/pathway and evidence policy, rather than being embedded inside
the scientific stage config.

## Bridge To PatientRunnerScientificConfig

The bridge from main/root config into `PatientRunnerScientificConfig` should be a
narrow adapter, not a second source of truth.

Target bridge shape:

```text
PipelineConfig + discovered input/runtime resources
  -> build_patient_runner_scientific_config(...)
      -> PatientRunnerScientificConfig
  -> PatientScientificShadowConfig or live runner config
```

Bridge mapping:

| Patient-runner slice | Config source after rewrite | Legacy compatibility inputs |
| --- | --- | --- |
| `grid_preprocessing` | `preprocessing.grid` | dose/MR refs, pydicom patient state |
| `anatomical_preprocessing` | `preprocessing.anatomical`, `structure_registry` | RTSTRUCT path mapping, legacy refs/dicts |
| `preprocessing` | `biopsy`, `preprocessing.geometry`, `uncertainty` | `structs_referenced_dict`, uncertainty dataframe |
| `transform_generation` | `transforms`, `random_seeds`, MC/optimizer sample requirements | copied `MC info` and `Random info` snapshots |
| `optimization` | `optimization.optimizer_v1`, `optimization.optimizer_v2` | legacy adapter configs and parallel pool |
| `simulated_biopsy_finalization` | `biopsy.simulated_biopsy.producer_policy`, geometry core | `structs_referenced_dict`, selected producer outputs |
| `sampling_classification` | `biopsy.sampling`, `mc.simulation.biopsy_z_voxel_length` | biopsy structures and legacy output stores |
| `mc_prep` | `simulation.mc.prep`, `simulation.mc.counts` | copied `MC info` sample counts |
| `mc_simulation` | `simulation.mc.containment`, `dose`, `mr_adc` | legacy MC adapter contracts |
| `guidance` | `guidance_maps.planning` | selected anatomy and biopsy state |

Bridge rule: the adapter may emit legacy dictionaries or adapter dataclasses for
validated modules, but it should not make `master_structure_info_dict["Global"]`
the place where future GUI config lives.

## Dependency Tree For Config Rewrite

A config rewrite should follow the same scientific dependency spine as the
runner, with validation/config wrappers outside it:

```text
startup/input/legacy keys
  -> structure registry and data removals
  -> artifact/replay policy
  -> grid preprocessing config
  -> anatomical preprocessing config
  -> biopsy geometry and simulated-biopsy config
  -> transform generation and random seed config
  -> optimizer config
      -> optimizer-v1 core/debug
      -> optimizer-v2 search/capacity/calibration/validation/render
  -> simulated-biopsy producer policy
  -> sampling/classification config
  -> MC prep and MC simulation config
      -> containment/dose/MR/output/debug
  -> guidance planning/render config
  -> validation and patient-runner shadow config
```

Cross-cutting runtime resources should stay out of frozen config:

- parallel pools
- live display/progress objects
- runtime logger instances
- file handles
- Open3D objects
- large arrays and dataframes

Those resources belong in runner/runtime context objects such as
`PatientScientificStageResources`, not in serializable user config.

## Debug Subgroup Policy

Every grouped config that has side effects or expensive optional evidence should
use the same internal pattern:

```text
DomainConfig
  core         # values that define scientific behavior or required runtime policy
  output       # normal artifacts and small manifests
  validation   # parity checks and sidecar comparisons
  debug        # visual inspection, demos, raw dumps, benchmarks, CUDA logs
  gui          # optional labels/help/visibility metadata later, not used by science
```

Pruning rule:

- Production/scientific-shadow runs should be able to pass only `core` plus
  required `output`/`validation` evidence.
- Debug runs can opt into `debug` explicitly.
- GUI panels can expose or hide subgroups without changing the domain adapter
  API.
- Patient-runner stage adapters should reject or ignore unsupported debug/render
  side effects at the boundary.

This split is more important than the exact class names. The rewrite should make
it mechanically easy to remove or hide debug subgroups later.

## Recommended Rewrite Sequence

1. Add typed config subgroups without changing any default values or call order.
2. Create builder functions that reproduce the current legacy adapter configs
   from those subgroups.
3. Move only the already-partial domains in `PipelineConfig` first:
   preprocessing debug split, grid debug split, optimizer-v2 subgroup split,
   MC subgroup shell.
4. Add `build_patient_runner_scientific_config(...)` as a bridge from typed root
   config plus discovered runtime resources.
5. Wire `PatientScientificShadowConfig` from the bridge for a small patient UID
   set, keeping legacy main output routing unchanged.
6. Compare scientific-shadow evidence against the legacy oracle before widening
   to full cohort or removing legacy globals.
7. Only after parity is credible, demote `Preprocessing info`, `MC info`, and
   `Random info` to snapshots/metadata instead of config carriers.

## Non-Goals For The Next Pass

Do not rewrite raw scientific algorithms during this config pass.

Do not edit the raw MC simulator or CUDA/kernel math as part of config cleanup.

Do not remove `master_structure_reference_dict` or `master_structure_info_dict`
yet. The next pass should quarantine them behind bridge/adapters, not pretend
they are gone.

Do not make the GUI schema first. Build clean typed Python config first, then add
GUI/file serialization views over it.

## Current Cleanup Status And Next Gate

The config-boundary pass now has the pieces needed for the first
scientific-shadow validation gate:

1. `PipelineConfig` owns the current patient-runner-facing root groups for
  legacy references, structure registry, grid preprocessing, biopsy settings,
  preprocessing, optimizer-v1/v2, MC, guidance, and validation sidecars.
2. `PreprocessingConfig` stores interpolation, geometry, kernel-execution, and
  debug subgroups while preserving the legacy flat attribute names as
  compatibility properties.
3. Optimizer-v2 and MC have nested runtime/debug/output group shells that map to
  the existing patient-runner adapter contracts.
4. `build_patient_runner_scientific_config(...)` constructs the current
  `PatientRunnerScientificConfig` tree from `PipelineConfig` plus runtime and
  discovered resources.
5. `biopsy_localization_convex_main.py` still re-derives flat locals from
  `PipelineConfig` for the legacy oracle path, but new patient-runner work
  should consume the typed config tree instead of those translated locals.

Do not migrate main-facing declarations into TOML or JSON as the runtime
authority before this validation gate. The next stable source is the typed
Python `PipelineConfig`. TOML should remain a human-authored profile layer and
JSON should remain generated evidence or resolved-plan output until
scientific-shadow parity is credible. Neither should become a parallel config
language that can drift from the Python contracts.

New debug/render controls should follow the same rule. Add typed config fields
where the patient-runner stage can consume them, then expose them through a run
profile later. Do not add fresh render toggles to legacy main as the owning
surface for new patient-runner behavior.

The next validation sequence is:

1. Run the legacy/default cohort path and compare against the clean May 22 or
  May 21 baseline to verify the wider config bridge preserved outputs.
2. Run small `SCIENTIFIC_SHADOW` patient sets through the typed builder in
  staged pathways: biopsy preprocessing, optimization, post-optimizer biopsy
  realization, sampling/classification, then current dosimetry.
3. Widen scientific-shadow validation only after the staged runs resolve runtime
  context gaps such as RTSTRUCT paths, MR ADC unit state, RNG, parallel pool,
  and view-list availability.

# Patient Runner

This package is the Phase C.0 scaffold for patient-local execution. It is not a
scientific rewrite. The legacy all-patient monolith remains the validation oracle
while this package introduces typed contracts and one-patient orchestration.

Initial scope:

- define patient-runner contracts such as `PatientCase`, `PatientRunConfig`,
  `PatientStageResult`, and `PatientRunResult`,
- carve a one-patient view out of the legacy runtime dictionaries without deep
  copying heavy scientific state,
- receive legacy reference/info dictionary key names through `LegacyRuntimeKeys`
  from the caller's current pipeline config/bootstrap state,
- write currently available patient dataframe artifacts through the existing
  `output_artifacts` exporter surface,
- provide a minimal stage runner with timing and error capture.

Current Phase C.1 scope:

- wrap the raw legacy all-patient dictionaries in `LegacyCohortRuntimeState`
  before batch execution,
- resolve an ordered batch of patient IDs from the legacy patient registry,
- run each patient through the existing `PatientStage` sequence,
- preserve deterministic result ordering when the thread backend is explicitly
  enabled,
- report a typed `PatientBatchRunResult` containing per-patient timings,
  statuses, and artifact paths.

The batch layer uses `PatientBatchRunConfig`, which wraps `PatientRunConfig`
rather than duplicating its legacy keys, output-root policy, or artifact-writing
settings. Sequential execution is the default. The thread backend is currently
appropriate only for the artifact-writing stage because Phase C.1 reads shared
in-memory legacy objects. Process-level isolation is deferred until migrated
stages have serializable, patient-local inputs.

Compatibility boundaries:

- raw `master_structure_reference_dict` and `master_structure_info_dict` objects
  may enter only through legacy bridge/batch-from-legacy functions, where they
  are wrapped in `LegacyCohortRuntimeState`,
- patient stages should receive typed runner objects, not the all-patient
  dictionaries directly,
- patient IDs are preserved exactly for dictionary lookup; validation rejects
  non-string, empty, or duplicate IDs, but does not strip or rewrite them,
- filesystem-safe names are derived only for output paths and must not become
  patient identity keys.

Parallelism policy:

- `PatientBatchExecutionBackend.SEQUENTIAL` is the reference backend and default,
- thread parallelism is acceptable only for the current artifact-writing stage,
  which reads patient-local dictionary views and writes independent files,
- scientific stages with Open3D, optimizer, MC, large arrays, or native memory
  pressure should not be threaded through shared legacy dictionaries,
- process-level patient workers can be added after patient inputs are
  serializable and memory budgets are explicit,
- `max_workers` should eventually be bounded by measured per-patient memory,
  not just CPU count.

The detailed recommendations and non-recommendations for patient-level
parallelism live in `../../docs/roadmap/PATIENT_RUNNER_UPGRADE_ROADMAP.md` under
"Parallelism Opportunities and Recommendations".

Current Phase D scope:

- build an inventory from `PatientBatchRunResult.artifact_paths`,
- optionally assemble selected cohort-style tables from written patient
  artifacts using the existing stitch-pair definitions,
- allow callers to select patients, source tables, final cohort table names, and
  the assembly output directory,
- optionally compare assembled tables to legacy final cohort dataframes supplied
  by the validation caller,
- write assembly inventories, summaries, validation tables, and assembled shadow
  tables only when the assembly config requests output writing.

Assembly is intentionally a separate post-run utility. Patient execution should
produce durable per-patient artifacts first. Cohort assembly can then be run
later by a CLI, validation workflow, or future GUI for a selected patient set and
selected tables without rerunning the patient pipeline.

Assembly preserves the `PatientBatchRunResult.artifact_paths` construction order,
which follows legacy patient/artifact construction order. It does not sort rows
by filesystem path. CSV readback is schema-aware for known MultiIndex-column
tables so validation does not treat secondary header rows as data rows. Validation
keeps raw dataframe comparison fields and also checks the CSV artifact-equivalent
final dataframe, which is the comparison used for persisted-output status.

Typed patient-local interface direction:

- the current bridge still builds a one-patient legacy-shaped view for old code
  that expects legacy dictionaries,
- new stage wrappers should expose typed patient inputs and outputs to the
  runner, then translate to legacy dict shape only at the old function boundary,
- each migrated stage should move repeated dictionary access behind typed
  accessors or small dataclasses,
- behavior-preserving preprocessing wrappers should accept explicit legacy key
  names and patient-local state so future typed adapters can replace the backing
  dictionary without changing the runner-facing stage contract,
- wrappers should remain domain-selectable where the future GUI may expose
  separate actions, such as dose mapping, MR mapping, targeting, QA, or full-run
  execution,
- once a stage no longer reads raw legacy paths internally, the legacy adapter
  can be replaced without changing the runner contract.

Logging and instrumentation direction:

- every patient run should record start/end timestamps, elapsed seconds, status,
  warnings, artifact counts, and output paths,
- future process workers should also record attempt number, worker PID, exit
  code, timeout status, retry reason, and measured memory where available,
- stage and patient logs should be written beside patient artifacts so a failed
  patient can be retried or inspected without rerunning the cohort.

Current manifest surface:

- each `run_patient_case` call writes `patient_run_manifest.json` beside that
  patient's artifacts when `PatientRunConfig.write_patient_run_manifest` is true,
- each `run_patient_batch` call writes `patient_batch_run_manifest.json` under
  the batch output root when `PatientBatchRunConfig.write_batch_run_manifest` is
  true,
- manifests are generated from `PatientRunResult` and `PatientBatchRunResult`,
  so stage timing/status/artifact metadata is recorded through the same typed
  result contracts consumed by validation and assembly.

Scientific stage config boundary:

- `scientific_config.py` owns the opt-in `PatientRunnerScientificConfig` bundle
  and stage-group configs for grid preprocessing, anatomical preprocessing,
  preprocessing, MC prep, MC simulation, optimization, and guidance-map
  precompute,
- grid preprocessing is represented as its own opt-in stage before anatomical
  preprocessing; it currently wraps patient-local dose-grid runtime object
  construction, MR ADC input normalization, and MR ADC grid runtime object
  construction with render side effects disabled at the runner boundary,
- anatomical preprocessing is represented as its own opt-in stage before
  biopsy-facing preprocessing; it currently wraps raw contour pulling, selected
  structure selection, standard non-biopsy structure processing, and
  prostate-only MR ADC summary finalization with legacy presentation objects
  adapted to null/headless shims,
- preprocessing is represented explicitly through currently patient-local slices
  such as real-biopsy geometry processing, simulated-biopsy preparation,
  simulated-biopsy planning, uncertainty attachment, realized targeting, and
  sampled-biopsy processing; heavier non-biopsy structure preprocessing can be
  added to the same boundary as those signatures are cleaned,
- `scientific_stages.py` contains thin runner adapters that translate
  `LegacyPatientRuntimeState` plus the scientific config bundle into calls to
  existing patient scientific modules,
- `scientific_dependencies.py` owns both the full scientific graph-node view and
  the current executable adapter view, plus named pathway presets such as
  `current_dosimetry_shadow`,
- `build_patient_scientific_stages(...)` is opt-in and does not change
  `default_patient_stages()`, which remains artifact-only for the current
  shadow-output validation path.
- direct stage-order builds validate the configured stage subset by default;
  named pathway builds require the pathway's enabled stage configs unless the
  caller explicitly marks upstream stages as already satisfied for a controlled
  preprocessed/debug state.
- the full graph now names transform generation, simulated-biopsy finalization,
  and sampling/classification separately; the executable adapter slice remains
  coarser until those missing stage adapters are split out,
- simulated-biopsy finalization is intentionally not folded into the early
  preprocessing adapter because the legacy path runs it after optimizer-v2; it
  should become a separate opt-in stage before scientific shadow routing.

Scientific tranche direction:

- patient scientific modules stay standalone in their owning scientific package;
  `scientific_tranches.py` defines ordered tranche recipes in `patient_runner`,
  but those recipes are orchestration only,
- dependency edges and pathway selection should be treated as the execution
  source of truth; tranches are removable debug/documentation groupings over
  graph nodes, not the dependency model itself,
- patient discovery is not a tranche: DICOM discovery, modality routing, patient
  selection, prompts, and input manifests remain run-scoped discovery/bootstrap
  work outside scientific stage recipes,
- tranche recipes may start from discovered patient cases plus carved or built
  one-patient runtime/reference/info state,
- grid preprocessing and anatomical preprocessing are separate tranches, with
  grid preprocessing ordered first because some structure processing consumes
  grid/lattice information that must already be available; grid preprocessing
  owns the current dose-grid, MR ADC normalization, and MR ADC grid adapters
  plus later patient-local lattice/grid/KD-tree artifacts, while anatomical
  preprocessing owns the current raw-contour, selected/unique structure,
  standard non-biopsy structure, and prostate-only MR ADC adapters,
- biopsy preprocessing, pre-optimizer transforms/optimizers, post-optimizer
  biopsy realization, sampling/classification, MC prep/simulation, and
  output/guidance/assembly/parity should remain separate tranche recipes so the
  legacy ordering is visible and testable.
- the dependency/pathway terminology and current conservative graph are tracked
  in `docs/architecture/PATIENT_RUNNER_DEPENDENCY_GRAPH.md`.

Main-facing validation gate:

- `biopsy_localization_convex_main.py` keeps the legacy path as the oracle and
  calls the patient-runner validation hook only through an explicit mode,
- `PatientRunnerMainValidationMode.SHADOW_OUTPUT` runs after the legacy in-memory
  outputs exist; it writes patient-runner artifacts from that completed state,
  assembles cohort tables, and compares them with the legacy final dataframes,
- this first gate validates the artifact/export/assembly layer, not independent
  scientific recomputation.

Post-run parity surface:

- `patient_runner.parity.run_patient_runner_post_run_parity(...)` compares two
  completed output surfaces after both runs finish,
- the default surface compares legacy final cohort CSVs with patient-runner
  assembled cohort tables using the existing stitch-pair registry,
- optional recursive CSV comparison reuses the existing validation comparator
  and is intended for roots that deliberately use path-compatible layouts,
- future stage-state comparisons should be added as durable manifests beside
  patient artifacts, not by inspecting live master dictionaries during a run.

Scientific modularization rule:

- when moving remaining main-facing scientific blocks into modules, first move
  the existing code into thin wrappers that preserve inputs, ordering, side
  effects, and output keys,
- do not introduce new scientific data types, algorithms, formulas, or helper
  abstractions in the same pass,
- scientific behavior changes require a separate deliberate change with focused
  validation and review.

Current implementation guardrail:

- keep the validated legacy/semi-modular cohort path as the oracle,
- build patient scientific modules in their owning scientific packages; see
  `../../docs/architecture/PATIENT_MODULE_TREE_GUIDE.md`,
- keep Rich/UI objects out of runner-facing patient calls; if an older helper
  still needs a legacy presentation-shaped object, adapt it inside the owning
  scientific package boundary rather than in `patient_runner/`,
- do not create a module-local `_presentation.py` by default; use one only when
  a package must quarantine old presentation-shaped helper arguments. Clean
  scientific stages should either reject presentation options or use shared
  `presentation/` adapters at the outer UI boundary,
- prefer explicit patient entrypoints in the stage file or a family-local
  `per_patient/` subpackage over a parallel top-level `../patient_stages/`
  tree,
- do not recreate a top-level `../patient_stages/` tree,
- for high-risk MC simulation work, prefer copy-assisted patient-module
  extraction over in-place cleanup of `MC_simulator_convex.py`,
- do not rewire frozen cohort/oracle wrappers to call new patient modules during
  additive extraction; call those modules from the patient runner once that path
  is ready for comparison,
- keep scientific patient modules sequential by default; batch/process
  parallelism can be added after patient-local inputs are stable,
- do not create module-local `ALL_CAPS` constants for legacy dictionary keys when
  the value should come from `LegacyRuntimeKeys`, a typed accessor, or another
  adapter contract.

Near-term non-goals:

- no changes to scientific algorithms,
- no output schema churn,
- no broad cleanup inside `MC_simulator_convex.py` or other legacy oracle files,
- no replacement of the legacy dictionaries as the validation backing store.

The next integration step is to validate the current semi-modular main path,
then wire the remaining pre-MC patient stage tranche and begin separate
patient-facing MC module extraction.

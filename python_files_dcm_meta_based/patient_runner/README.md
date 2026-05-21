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

Main-facing validation gate:

- `biopsy_localization_convex_main.py` keeps the legacy path as the oracle and
  calls the patient-runner validation hook only through an explicit mode,
- `PatientRunnerMainValidationMode.SHADOW_OUTPUT` runs after the legacy in-memory
  outputs exist; it writes patient-runner artifacts from that completed state,
  assembles cohort tables, and compares them with the legacy final dataframes,
- this first gate validates the artifact/export/assembly layer, not independent
  scientific recomputation.

Scientific modularization rule:

- when moving remaining main-facing scientific blocks into modules, first move
  the existing code into thin wrappers that preserve inputs, ordering, side
  effects, and output keys,
- do not introduce new scientific data types, algorithms, formulas, or helper
  abstractions in the same pass,
- scientific behavior changes require a separate deliberate change with focused
  validation and review.

Near-term non-goals:

- no changes to scientific algorithms,
- no output schema churn,
- no broad cleanup inside `MC_simulator_convex.py`,
- no replacement of the legacy dictionaries as the validation backing store.

The next integration step is to finish main-facing preprocessing modularization
as exact wrapper extraction, then wire the pre-MC patient stage tranche behind
the validation gate.

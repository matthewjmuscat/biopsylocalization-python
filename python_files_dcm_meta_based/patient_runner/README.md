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
parallelism live in `PATIENT_RUNNER_UPGRADE_ROADMAP.md` under "Parallelism
Opportunities and Recommendations".

Current Phase D scope:

- build an inventory from `PatientBatchRunResult.artifact_paths`,
- assemble cohort-style tables from written patient artifacts using the existing
  stitch-pair definitions,
- optionally compare assembled tables to legacy final cohort dataframes supplied
  by the validation caller,
- write assembly inventories, summaries, validation tables, and assembled shadow
  tables under a validation output directory.

Near-term non-goals:

- no changes to scientific algorithms,
- no output schema churn,
- no broad cleanup inside `MC_simulator_convex.py`,
- no replacement of the legacy dictionaries as the validation backing store.

The next integration step is to migrate one scientific stage behind a typed
patient-local interface, then add process-isolated patient workers once that
stage's inputs are serializable and memory requirements are measured.

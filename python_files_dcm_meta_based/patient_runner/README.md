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

- resolve an ordered batch of patient IDs from the legacy patient registry,
- run each patient through the existing `PatientStage` sequence,
- preserve deterministic result ordering even when optional thread parallelism is
  enabled,
- report a typed `PatientBatchRunResult` containing per-patient timings,
  statuses, and artifact paths.

The batch layer uses `PatientBatchRunConfig`, which wraps `PatientRunConfig`
rather than duplicating its legacy keys, output-root policy, or artifact-writing
settings. Optional parallelism currently uses threads because Phase C.1 only
writes dataframe artifacts from shared in-memory legacy objects. Process-level
isolation is deferred until migrated stages have serializable, patient-local
inputs.

Compatibility boundaries:

- raw `master_structure_reference_dict` and `master_structure_info_dict` objects
  may enter only through legacy bridge/batch-from-legacy functions,
- patient stages should receive typed runner objects, not the all-patient
  dictionaries directly,
- patient IDs are preserved exactly for dictionary lookup; validation rejects
  non-string, empty, or duplicate IDs, but does not strip or rewrite them,
- filesystem-safe names are derived only for output paths and must not become
  patient identity keys.

Parallelism policy:

- thread parallelism is acceptable only for the current artifact-writing stage,
  which reads patient-local dictionary views and writes independent files,
- scientific stages with Open3D, optimizer, MC, large arrays, or native memory
  pressure should not be threaded through shared legacy dictionaries,
- process-level patient workers can be added after patient inputs are
  serializable and memory budgets are explicit,
- `max_workers` should eventually be bounded by measured per-patient memory,
  not just CPU count.

Near-term non-goals:

- no changes to scientific algorithms,
- no output schema churn,
- no broad cleanup inside `MC_simulator_convex.py`,
- no replacement of the legacy dictionaries as the validation backing store.

The next integration step is to assemble the generated patient artifacts and
compare them against the legacy cohort oracle through the existing
stitch-validation machinery.

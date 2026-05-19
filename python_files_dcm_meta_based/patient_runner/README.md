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

Near-term non-goals:

- no changes to scientific algorithms,
- no output schema churn,
- no broad cleanup inside `MC_simulator_convex.py`,
- no replacement of the legacy dictionaries as the validation backing store.

The next integration step is to wrap one existing patient-local stage behind this
runner, then compare the generated patient artifacts against the legacy cohort
oracle through the existing stitch-validation machinery.
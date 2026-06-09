# Post-Run Utilities

This package contains utilities that operate on completed run outputs. They do
not call scientific stages and do not mutate the validated legacy/main pathway.

The first utility is cohort assembly from patient-runner artifacts:

```bash
PYTHONPATH=python_files_dcm_meta_based \
  python -m post_run.cohort_assembly.cli \
  python_files_dcm_meta_based/post_run/configs/cohort_assembly_jobs.json
```

The command is intentionally config-first. The same API can be called by a GUI:

```python
from post_run import run_post_run_cohort_assembly_jobs

results = run_post_run_cohort_assembly_jobs("python_files_dcm_meta_based/post_run/configs/cohort_assembly_jobs.json")
```

The sample config is disabled by default. Enable a job after a patient-scientific
runner has been executed with artifact writing enabled; the Jun 08 full-current
validation run intentionally has `artifact_count=0`, so it validates scientific
execution but is not yet an assembly source.

## Boundary

- Inputs: completed `patient_batch_run_manifest.json` and per-patient manifests.
- Outputs: `cohort_assembly/` reports and optional `assembled_tables/` CSVs.
- Engine: the existing `patient_runner.cohort_assembly` implementation.
- Future GUI contract: pass a config path or a `PostRunCohortAssemblyJobConfig`
  object, then display the returned summary and written paths.
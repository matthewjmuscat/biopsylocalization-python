# Validation Run Codebook

This codebook describes the TOML-driven validation workflow for completed run
outputs. It is intentionally an orchestration layer: validation profiles choose
which completed runs to compare and which comparator jobs to execute. They do
not define scientific run parameters.

## Architecture Contract

- Human-authored validation profiles are TOML.
- Resolved execution summaries and comparator outputs are JSON/CSV provenance.
- Scientific runtime configuration remains owned by typed Python config objects,
  such as `PipelineConfig` and patient-runner configs.
- Validation profiles operate on completed run folders, patient-runner manifests,
  assembled artifacts, and cohort CSV surfaces.
- Future GUI/product code should call the same public validation runner or typed
  validation config boundary instead of duplicating comparator call shapes.

This follows the broader config migration direction: TOML is the editable user
profile, JSON is generated evidence, and the public scientific repository keeps
stable typed contracts that a private GUI/product repository can consume.

## One-Command Entry Point

From the repository root:

```bash
/home/matthew-muscat/.local/share/virtualenvs/biopsylocalization-python-a85Yh81c/bin/python \
  python_files_dcm_meta_based/run_validation.py \
  python_files_dcm_meta_based/validation/configs/validation_jobs.toml
```

The default config path is `python_files_dcm_meta_based/validation/configs/validation_jobs.toml`,
so this shorter form is equivalent when that profile is the one you want:

```bash
/home/matthew-muscat/.local/share/virtualenvs/biopsylocalization-python-a85Yh81c/bin/python \
  python_files_dcm_meta_based/run_validation.py
```

Use `--dry-run` to inspect resolved commands without running the comparators:

```bash
/home/matthew-muscat/.local/share/virtualenvs/biopsylocalization-python-a85Yh81c/bin/python \
  python_files_dcm_meta_based/run_validation.py --dry-run
```

The runner injects `PYTHONPATH` for child comparator scripts, so you do not need
to export environment variables for normal use.

## Available TOML Profiles

Current checked-in profiles:

- `validation/configs/validation_jobs.toml`: default Jun25/Jun26 local profile,
  including intrarun patient-runner parity and same-subset full-vs-split
  reconstruction.
- `validation/configs/intrarun_patient_runner_parity.toml`: focused intrarun
  profile for one completed run.
- `validation/configs/interrun_full_vs_split_reconstructed.toml`: focused
  full-vs-split patient-runner profile.
- `validation/configs/legacy_cohort_regression.toml`: focused legacy/cohort
  baseline-vs-candidate profile.

The older `validation/configs/validation_jobs.json` remains supported for
compatibility and for machine-generated job manifests, but new hand-edited
profiles should be TOML.

## TOML Layout

Every profile has the same top-level shape:

```toml
schema_version = "validation_run_config_v1"
description = "Short human description."

[defaults]
abs_tol = 1e-8
rel_tol = 1e-6

[runs]
candidate = "../Data/Output data/MC_sim_out- ..."

[paths]
candidate_patient_runner = "../Data/Output data/MC_sim_out- .../patient_scientific_runner"

[run_groups.group_name]
description = "What this group validates."
enabled = true

[[run_groups.group_name.jobs]]
name = "job_name"
script = "compare_cohort_runs"
baseline_run = "baseline"
candidate_run = "candidate"
output_dir = "validation_outputs/<run_key>/cohort_vs_baseline"
```

`runs` are completed top-level output folders. `paths` are named non-run paths,
usually nested patient-runner output directories. Jobs should refer to these
names instead of repeating long paths.

## Intrarun Validation

Use intrarun validation when one completed run contains both the legacy/oracle
cohort output and patient-runner artifacts. The normal job sequence is:

1. `post_run_cohort_assembly`: assemble patient-runner artifacts into cohort
   tables.
2. `compare_patient_runner_parity`: compare those assembled cohort tables with
   the same run's legacy/oracle cohort output.

Relevant TOML fields:

```toml
[[run_groups.intrarun_patient_runner_parity.jobs]]
name = "assemble_patient_runner_cohort"
script = "post_run_cohort_assembly"
patient_runner_output_path = "single_shot_patient_runner"

[[run_groups.intrarun_patient_runner_parity.jobs]]
name = "patient_runner_parity_vs_legacy"
script = "compare_patient_runner_parity"
legacy_output_run = "single_shot"
patient_runner_output_path = "single_shot_patient_runner"
output_dir = "validation_outputs/<run_key>/patient_runner_parity"
```

Main summary to inspect:

```text
validation_outputs/<run_key>/patient_runner_parity/patient_runner_post_run_parity_summary.json
```

Expected pass signal:

```text
overall_status: passed
missing_artifact_failure_count: 0
```

## Interrun: Full Vs Split Patient Runner

Use this when you ran one full/single-shot patient-runner run and two or more
split patient-runner runs over the same patient set. The comparator reconstructs
both cohort surfaces through the same assembly policy and then compares final
cohort CSVs.

Relevant TOML fields:

```toml
[[run_groups.full_vs_split_reconstructed.jobs]]
name = "full_vs_split_reconstructed"
script = "compare_reconstructed_cohort_runs"
reference_patient_runner_output_path = "full_patient_runner"
split_patient_runner_outputs_paths = [
  "split_a_patient_runner",
  "split_b_patient_runner",
]
output_dir = "validation_outputs/<run_key>/full_vs_split_reconstructed"
```

Do not enable `allow_patient_set_mismatch` for the real equivalence gate. The
full run's patient UID set should match the union of the split runs.

Main summary to inspect:

```text
validation_outputs/<run_key>/full_vs_split_reconstructed/reconstructed_cohort_comparison_summary.json
```

Expected pass signal:

```text
overall_status: passed
patient_uid_sets_match: true
non_ok_file_count: 0
missing_file_count: 0
```

## Interrun: Legacy Cohort Regression

Use this when comparing two completed legacy/main output runs, usually a new
candidate against a reference baseline.

Relevant TOML fields:

```toml
[[run_groups.legacy_cohort_regression.jobs]]
name = "cohort_final_tables_vs_baseline"
script = "compare_cohort_runs"
baseline_run = "baseline"
candidate_run = "candidate"
output_dir = "validation_outputs/<candidate_key>/cohort_vs_<baseline_key>"
```

For broader run health, add `validate_run_against_baseline`. For noisy all-CSV
investigation, use `compare_run_csv_outputs` only when final cohort parity is
not enough to locate a difference.

## Supported Job Scripts

The TOML `script` field supports:

- `post_run_cohort_assembly`
- `compare_patient_runner_parity`
- `compare_reconstructed_cohort_runs`
- `compare_cohort_runs`
- `validate_run_against_baseline`
- `compare_run_csv_outputs`

## Generated Provenance

Every configured run writes:

```text
validation_outputs/configured_validation_last_run.json
```

That JSON records the source TOML path, config format, config schema version,
enabled jobs, resolved commands, return codes, and dry-run status. Treat it as
generated validation provenance, not as the hand-edited source of truth.
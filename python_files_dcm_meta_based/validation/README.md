# Validation Runbook

This folder contains the reusable validation package plus config-driven launchers for checking a new run against known reference output. The existing top-level scripts are still the canonical comparators; the launcher here removes the need to rebuild long command lines by hand.

## Quick Start

From the repository root:

```bash
PYTHONPATH=python_files_dcm_meta_based /home/matthew-muscat/.local/share/virtualenvs/biopsylocalization-python-a85Yh81c/bin/python python_files_dcm_meta_based/validation/scripts/run_validation_jobs.py
```

The launcher reads `validation/configs/validation_jobs.json`, runs every enabled job, and writes a machine-readable summary to:

```text
validation_outputs/configured_validation_last_run.json
```

To use a different config file, pass its path as the single positional argument. The intended day-to-day workflow is to edit the JSON, not construct command options repeatedly.

## Available Scripts

| Script | Use it for | Main inputs | Output |
| --- | --- | --- | --- |
| `validate_run_against_baseline.py` | Broad run health: run-completion manifests, logs, recursive CSV comparisons, warnings/exceptions. | `baseline`, `candidate` run folders. | JSON/CSV diagnostics under the configured `output_dir`. |
| `compare_cohort_runs.py` | Main oracle gate for final cohort CSV tables. This is the cleanest scientific-regression check for legacy-output parity. | `baseline`, `candidate` run folders. | Per-table diff summaries under `output_dir`. |
| `compare_run_csv_outputs.py` | Recursive all-CSV comparison. Useful for investigating intermediate artifacts and diagnostic files. | `baseline`, `candidate` run folders. | Recursive CSV diff summaries under `output_dir`. |
| `compare_patient_runner_parity.py` | Patient-runner parity against a legacy/oracle run once patient-runner artifacts are available. | `legacy_output`, `patient_runner_output`. | Assembled parity tables and optional recursive CSV diffs. |
| `compare_reconstructed_cohort_runs.py` | Full-vs-split patient-runner validation after decomposing patient artifacts and reconstructing both cohort surfaces through one assembly policy. | One reference patient-runner output plus one-or-more split patient-runner outputs. | Reconstructed cohort surfaces, assembly reports, and cohort CSV comparison outputs under `output_dir`. |

## Config Layout

The default config is:

```text
python_files_dcm_meta_based/validation/configs/validation_jobs.json
```

It has four main sections:

| Section | Purpose |
| --- | --- |
| `defaults` | Shared tolerances and small script defaults. |
| `runs` | Named output-run folders, usually under `../Data/Output data/...`. |
| `paths` | Named non-run folders, such as a nested patient-runner output directory. |
| `run_groups` | Named groups of validation jobs. Toggle a whole group with `enabled`. |

Each job has a `script`, a short `name`, path references, an `output_dir`, and optional script-specific settings. Keep `output_dir` short, because full run-folder names are long enough to hit OS filename limits when nested repeatedly.

## Current Default Groups

| Group | Enabled | Purpose |
| --- | --- | --- |
| `latest_standard_candidate_core` | Yes | Fast current checks for the Jun 08 12:59 latest standard run: one recursive health check and one cohort oracle check against Jun 06 13:51. |
| `latest_standard_candidate_reference_set` | Yes | Cohort-table oracle checks against the current reference set. |
| `latest_patient_runner_parity` | No | Template for patient-runner parity once a runner output directory is ready. |
| `recursive_all_csv_reference_set` | No | Heavier recursive CSV comparisons for non-final artifacts. Enable only when investigating. |

## Adding A New Candidate Run

1. Add the candidate run folder under `runs` with a short key, for example `jun05_1012_standard_candidate`.
2. Copy an existing job group and update `candidate_run` to that key.
3. Give every job a short `output_dir`, usually under `validation_outputs/<candidate-key>/...`.
4. Run `validation/scripts/run_validation_jobs.py` with no options.
5. Check `validation_outputs/configured_validation_last_run.json`; any job with nonzero `returncode` needs review.

## Interpretation Notes

`compare_cohort_runs.py` is the preferred final scientific-output gate. Recursive CSV comparisons are intentionally noisier because they include diagnostics, timings, and implementation artifacts.

Known interpretation from the latest Jun 08 12:59 validation set:

- Cohort outputs matched Jun 06 13:51, Jun 05 21:17, Jun 05 15:05, Jun 04 22:23, Jun 04 00:32, Jun 03 14:36, and Jun 02 01:47 with 22/22 tables and zero drift.
- The Jun 04 14:18 comparison showed two known structural row-alignment differences but no numeric scientific drift.
- Recursive `validate_run_against_baseline.py` found the candidate run completed cleanly with no tracebacks or exception events; recursive differences were diagnostic/performance artifacts rather than final cohort drift.
- The `full_current_pipeline_shadow` runner manifests from the Jun 08 12:59 run showed both scientific shadow and live patient-scientific runner execution succeeded for all 4 patients through guidance. Artifact writing remained disabled for this scientific-stage checkpoint, so patient-runner cohort assembly/parity still requires a separate artifact-writing and assembly pass.

## Patient-Runner Validation

The patient-runner parity group is disabled until a runner output directory is ready. When enabling it, set `patient_runner_output_path` to a named path under `paths`, then run the same launcher. This keeps the legacy/oracle run and the patient-runner artifact location explicit in JSON.

## Full-Vs-Split Reconstruction Validation

Use `compare_reconstructed_cohort_runs.py` after a full/reference patient-runner run and two-or-more split patient-runner runs have completed with artifact writing enabled. The validator loads patient batch manifests, combines split patient results by patient UID, reconstructs both cohort surfaces through the same assembly planner, writes both reconstructed `Output CSVs/Cohort` surfaces, and then runs the standard cohort CSV comparator.

Example direct call:

```bash
PYTHONPATH=python_files_dcm_meta_based \
	/home/matthew-muscat/.local/share/virtualenvs/biopsylocalization-python-a85Yh81c/bin/python \
	python_files_dcm_meta_based/compare_reconstructed_cohort_runs.py \
	/path/to/full_run/patient_scientific_runner \
	/path/to/split_a/patient_scientific_runner \
	/path/to/split_b/patient_scientific_runner \
	--output-dir validation_outputs/<candidate-key>/full_vs_split_reconstructed
```

For config-driven use, add a job with script `compare_reconstructed_cohort_runs`, one `reference_patient_runner_output_*` field, and a list-valued `split_patient_runner_outputs_*` field. The patient UID sets must match by default; pass `allow_patient_set_mismatch: true` only for exploratory partial checks.

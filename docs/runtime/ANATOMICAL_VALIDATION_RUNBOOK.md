# Standalone Anatomical Validation

Last updated: 2026-09-09

## Scope And Gate

Phase 2C provides validation preparation, standalone batch finalization, and a
paired anatomical comparison. Phase 2D is the user-operated patient run. No
real-patient parity is claimed by the synthetic tests.

User-operated evidence at commit `0f2a6361bfb6dee8c27a0dc375c2297657b4e368`
now passes exact 0/0 for 181 F1/F2, 184 F1/F2 and 194 F2 (including ADC).
The first 194 F2 attempt failed in both input builders because ADC files were
missing; its fresh retry passed. This establishes the captured singleton input
migration gate only. Continue with [anatomical independence](ANATOMICAL_INDEPENDENCE_RUNBOOK.md)
before exposing later standalone pathways.

The paired lanes run in fresh, sequential child processes:

1. **Standalone:** typed patient input bootstrap, then existing patient grid and
   anatomical stage adapters.
2. **Legacy input:** unchanged `structure_referencer` on a singleton patient,
   then the same grid and anatomical stage adapters.

This tests independent input construction and its downstream consequences. It
does not independently validate shared algorithms, reproduce the whole legacy
main workflow, or establish cohort-order/split-run equivalence. Existing legacy
scientific sidecars and the oracle remain required for those later gates.

## Ownership

| Boundary | Owner | Deliberately excluded |
| --- | --- | --- |
| Reuse scientific config; prepare current execution provenance | `startup/standalone_preparation.py` | Patient loading, new scientific defaults |
| Validate patient results; write batch manifest and run index | `patient_runner/process_finalization.py` | Scientific arrays and computation |
| Capture and compare bounded numerical evidence | `validation/anatomical_checkpoint.py` | Arbitrary runtime serialization, DICOM reading, recomputation |
| Legacy singleton input adapter and capture hook | `validation/anatomical_execution.py` | Changes to oracle or kernels |
| Two-process execution and comparison report | `validation/anatomical_pair.py` | Automatic approval of scientific differences |

## Prerequisites

- Run commands yourself on the scientific workstation. They are not an invitation
  for automated inspection of patient inputs or output artifacts.
- Use a fresh, restricted output directory **outside the source tree**. Commit or
  otherwise freeze code first; do not edit source or change environment between
  provenance preparation and execution. Tracked changes and untracked source-like
  files contribute to the strict source identity.
- Obtain a generated `resolved_scientific_config.json`, a retained discovery
  `input_case_manifest.csv`, and `input_routing_profile.json`. Confirm the manifest
  still assigns the intended files to the exact patient UID. Input fingerprints
  describe role/path assignments, not DICOM file bytes; do not change input files
  between lanes.
- Scientific execution still requires the existing CUDA/RAPIDS/geometry stack.
  Only preparation, planning, comparison, and CLI help are CPU-safe.
- Begin with one representative patient with RTSTRUCT, RTDOSE, and RTPLAN.
  Current worker preflight requires all three paths even for F1, whose bootstrap
  intentionally omits dose attachment. Optional ADC is processed when supplied.

### Obtaining The First Scientific Snapshot

An existing generated snapshot can be reused without changing its scientific
values. If none exists, export the configured values from legacy main:

```bash
pipenv run python python_files_dcm_meta_based/biopsy_localization_convex_main.py \
  --export-scientific-config /absolute/fresh-config/resolved_scientific_config.json
```

This transitional command returns immediately after existing `PipelineConfig`
construction, before discovery or patient processing, and refuses overwrite.
Legacy import-time dependencies are still required. It is not a CPU-safe default
factory; extraction of defaults remains a later config migration. Ordinary main
execution with no arguments is unchanged. Do not use test fixtures as production
scientific defaults.

## Prepare A New Execution

From the repository root, replace the example absolute paths:

```bash
pipenv run python python_files_dcm_meta_based/prepare_patient_scientific_run.py \
  --scientific-config-snapshot /absolute/source/resolved_scientific_config.json \
  --routing-profile /absolute/discovery/manifests/input_routing_profile.json \
  --output-dir /absolute/new-validation/provenance
```

The destination must be absent or empty and must not overlap the source snapshot
directory or source routing file. The service writes the four standard provenance
artifacts plus `preparation_record.json`. The record identifies the reused config
and routing sources by path and hashes while code/environment identity belongs
to the **new** execution. It does not modify or relabel historical identities.
A late failure may leave partial new files; use another fresh destination.

## Plan One Patient

```bash
pipenv run python python_files_dcm_meta_based/run_patient_scientific_standalone.py \
  --input-case-manifest /absolute/discovery/manifests/input_case_manifest.csv \
  --scientific-config-snapshot /absolute/new-validation/provenance/resolved_scientific_config.json \
  --run-compatibility-identity /absolute/new-validation/provenance/run_compatibility_identity.json \
  --output-root /absolute/new-validation/plan \
  --patient-uid 'EXACT MANIFEST UID' \
  --pathway-name anatomical_qa --checkpoint-name anatomical_qa
```

This writes jobs without patient execution. Use the exact generated job path
listed in `patient_process_run_plan.json`; do not infer it by editing the UID.

## Execute The Paired Gate

Choose and record tolerances before inspecting results. The example below uses
exact equality as an initial deterministic smoke gate, not a universal clinical
tolerance. The comparator uses `abs(reference-candidate) <= abs_tol +
rel_tol*abs(reference)` for finite numeric quantities. One absolute tolerance
spans quantities with different units; any relaxation needs field-specific review
of the unit-labelled report, not a blanket increase until the comparison passes.

```bash
pipenv run python python_files_dcm_meta_based/validate_patient_anatomical.py \
  --job /absolute/new-validation/plan/worker_jobs/GENERATED_JOB.json \
  --output-dir /absolute/new-validation/pair \
  --abs-tol 0 --rel-tol 0 --timeout-seconds 3600
```

Both lanes verify planned config/compatibility files and current source/environment
before loading patient state. Each has a `worker.log`, worker-result JSON, patient
manifest, and (after successful anatomy) `validation/anatomical/` evidence under
its own output root. The top-level `anatomical_pair_summary.json` reports failure
if either lane fails. `anatomical_comparison.json` is produced only when both lanes
have valid matching patient/job/config evidence and the exact required stage order.

## Numerical Evidence

`anatomical_checkpoint.json` and `anatomical_arrays.npz` retain a bounded field
inventory: ROI identity, raw and reconstructed contours, available interpolated
geometry/mesh/centroids/volume, dose pixels/scaling/physical grid, retained ADC
metadata/values/cloud points, and selected preprocessing tables. NPZ loading uses
`allow_pickle=False`; unsupported runtime values fail rather than stringify.

Comparison checks identities, scientific config, field contracts, shapes, dtypes,
discrete data, and NaN/infinity masks exactly, then actual finite values against
the stated tolerances. It reports missing fields and exclusions. Empty/raw-only
geometry and incomplete present modalities cannot pass. ADC lattices discarded by
the scientific producer are not recomputed just for evidence; coverage reports
that limitation. Numeric table columns are compared, not only dataframe shapes.

Acceptance requires all expected stages to succeed, sufficient coverage for the
selected patient/modalities, and no unexplained differences. A failure may expose
a migration defect, unsupported capture representation, numerical variability,
or a pre-existing oracle behavior; diagnose the named field before changing code
or tolerances. Keep reports local; share sanitized summaries, not patient arrays.

## Ordinary Standalone Runs

The normal parent now writes `patient_batch_run_manifest.json` and
`manifests/run_manifest_index.json` after execution. Successful workers must have
matching patient/run/provenance, ordered stage evidence, and existing in-root
artifacts. Failed, dry-run, or unlaunched patients make the batch non-assembly-ready;
the post-run loader rejects it. Timeout/launch failures get durable results and
unlaunched patients stay explicit. Rerunning into existing patient outputs, batch
manifest, or run index is rejected; patient-level resume is not implemented.

The paired gate is a validation utility, not a cohort run. Its lane outputs are
inspected through their patient/checkpoint manifests rather than batch assembly.

## Following Gates

After the singleton input-migration gate, run a small diverse set with optional
ADC and different fractions, then investigate patient-order/split independence.
In particular the legacy dose cohort wrapper carries an inferred threshold from
one patient to the next; standalone processing resolves it independently. Do not
alter the oracle to hide that difference. Only then enable biopsy/optimization,
followed by MC/output/guidance checkpoints. Retire sidecars individually once
replacement numerical and cohort/split gates cover their scientific behavior.

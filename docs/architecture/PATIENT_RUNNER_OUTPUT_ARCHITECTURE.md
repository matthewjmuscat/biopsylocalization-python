# Patient-Runner Output Architecture

This note defines the output boundary for the per-patient runner migration. The
scientific runner should produce durable patient artifacts first. Cohort tables
should be derived from those artifacts by a separate assembly service that can be
run automatically after the main algorithm or manually after one or more
compatible runs complete.

## Goals

- Make per-patient artifacts the primary durable output surface for patient
  scientific stages.
- Preserve successful patient outputs when another patient fails.
- Keep cohort-style outputs as derived products, not as the only durable output
  boundary.
- Allow cohort assembly to run separately after the main algorithm completes.
- Allow config to request automatic cohort assembly at the end of a run.
- Let the assembly service discover what it can produce from artifact manifests
  and output contracts, rather than from hard-coded ad hoc file lists.
- Enforce strict compatibility before stitching artifacts from multiple runs.
- Keep the validated legacy cohort path as the oracle until each output family
  has explicit artifact and assembly validation evidence.

## Non-Goals

- Do not make scientific stages responsible for cohort assembly.
- Do not require raw scientific modules to know final cohort filenames.
- Do not silently stitch incompatible runs or artifact schema versions.
- Do not promise every legacy final cohort CSV is a simple patient concat. Some
  outputs are run-level, cohort-derived, validation-only, or better rebuilt by a
  later post-run service.

## Output Layers

1. Patient artifact production
   Each patient stage writes or exposes patient-scoped artifacts for every table
   it produced during the selected pathway. A failed patient still leaves its
   completed artifacts and manifest entries on disk.

2. Patient artifact manifest
   Each patient output directory should contain a manifest with patient UID,
   run ID, pathway name, stage statuses, artifact entries, schema versions,
   paths, row counts, column fingerprints where available, and skipped or failed
   artifact reasons.

3. Run artifact manifest
   The run-level manifest aggregates patient manifests and records the run DAG,
   configuration hash, code/version identity, artifact registry version, and
   compatibility policy. This is the entry point for post-run assembly.

4. Cohort assembly service
   A separate module consumes one or more compatible run manifests, discovers
   eligible artifact families, applies the registered assembly policy, writes
   cohort-style tables, and records assembly evidence.

5. Post-run parity
   Validation compares assembled patient-runner cohort tables against the legacy
   final cohort outputs. This remains outside normal scientific execution.

## Artifact Contracts

Every durable output family should have an artifact contract. The current
`OutputSchemaRegistry` is the seed of this contract layer, but the contract needs
to become the single source for output planning and assembly.

Each contract should include:

- `table_id`
- legacy table name and output section
- artifact scope: `patient`, `run`, `cohort`, or `validation_only`
- producing stage or stage family
- required pathway products or DAG nodes
- row grain
- primary keys and stable sort keys
- file format and storage policy
- schema version and compatibility version
- assembly policy: `concat_patient_fragments`, `aggregate_patient_fragments`,
  `copy_run_level`, `external_source`, `validation_only`, or `not_stitchable`
- expected missing reasons when a pathway does not produce the artifact
- validation status and retention policy

The assembly service should not need a new code path for each new table. Adding
a new patient-stitchable artifact should usually mean adding or updating a
contract, then relying on the generic planner and assembly engine.

## Planning And Discovery

Assembly should use both contract planning and manifest discovery.

- Contract planning answers: what should this DAG/pathway be able to produce?
- Manifest discovery answers: what was actually produced and where is it?
- The assembly report should separate `not_applicable`, `skipped_by_config`,
  `missing_source_fragments`, `failed_patient`, `assembled`, and `validated`.

This distinction is important for partial runs. A cohort output may be missing
because the selected pathway did not include the producing stage, because output
writing was disabled, or because a patient failed.

## Compatibility Rules

Single-run assembly is allowed when the run manifest and artifact contracts are
internally consistent.

Multi-run assembly should be allowed only when all selected manifests satisfy a
strict compatibility check:

- same artifact contract version
- compatible output schema version for every selected table
- compatible pathway products for the requested output family
- compatible configuration fields that affect scientific values
- compatible input role/routing profile where relevant
- no mixed legacy-key policy unless an explicit migration adapter declares it
  compatible

If compatibility cannot be proven, assembly must fail closed and write a clear
diagnostic report.

## Current State

- The live scientific runner can already append `patient_artifact_writing` when
  `include_artifact_writing=True`.
- The Jun 08 12:59 `full_current_pipeline_shadow` run validated scientific
  shadow and live patient-scientific execution through guidance, but artifact
  writing was intentionally disabled.
- The current stitch registry covers 20 cohort-style patient-fragment outputs.
- The Jun 08 legacy cohort directory contains 22 final CSVs. The uncertainty
  CSVs and global sum-to-one policy still need explicit output contracts before
  they should be treated as fully reconstructed patient-runner cohort outputs.
- `compare_patient_runner_parity.py` already compares legacy final cohort CSVs
  against assembled patient-runner cohort tables, but it expects assembled tables
  to exist before it runs.
- Validation and assembly still assume legacy physical paths such as
  `Output CSVs/Cohort`, `Output CSVs/Preprocessing`, and
  `Output CSVs/MC simulation` in several places. These names should become
  compatibility aliases, not the long-term logical artifact identifiers.

## Reorganization Strategy

Output cleanup should happen in layers so validation remains usable throughout
the migration.

1. Keep writing legacy-compatible paths while adding stable logical artifact IDs
   to manifests and contracts. Existing validation scripts should continue to
   compare `Output CSVs/Cohort` until the new manifest-aware validators are
   proven equivalent.

2. Add manifest-aware discovery APIs. Validation, parity, inventory, and assembly
   should ask the manifest/registry for artifact locations instead of building
   paths like `Output CSVs/Cohort` directly.

3. Introduce cleaner physical names as aliases after manifest-aware validation is
   in place. For example, a future layout may use `patients/<uid>/tables/...`
   and `cohort/tables/...`, while still emitting legacy aliases during the
   transition.

4. Rename table files only after each table has a stable `table_id`, legacy-name
   alias, schema version, and downstream compatibility note. The stable API
   should be the contract ID, not the filename.

5. Retire legacy paths only after the validators and downstream consumers can use
   manifest/contract IDs, and after a validation run proves old-path and new-path
   outputs are equivalent.

## Implementation Sequence

1. Validate `full_current_pipeline_shadow` with artifact writing still disabled.
   This isolates guidance behavior after the validated dosimetry checkpoint.
   Status: complete for the Jun 08 12:59 run.

2. Enable patient artifact writing for the full current pathway and validate that
   each patient writes manifests and artifacts without changing scientific
   results.

3. Add a post-run assembly entry point that can be run from CLI/config and that
   writes `cohort_assembly/assembled_tables`, an assembly manifest, and a summary
   report.

4. Replace hard-coded stitch selection with contract-driven planning backed by
   manifest discovery. The existing `SHADOW_STITCH_PAIRS` can remain as the
   compatibility bridge during migration.

5. Run patient-runner parity against the legacy cohort oracle for all currently
   contract-covered tables.

6. Close non-covered output families by assigning each legacy final output one
   of these policies: patient-stitchable, run-level aggregate, external/run-only,
   validation-only, or reimplement-later.

7. Once parity is clean for the contracted surface, make automatic cohort
   assembly a config-controlled post-run option while keeping the manual CLI path
   available.

## Directory Contract

The exact paths can evolve, but the durable shape should be stable:

```text
<run_output>/
  patient_scientific_runner/
    run_artifact_manifest.json
    patient_batch_run_manifest.json
    patients/
      <patient_uid>/
        patient_run_manifest.json
        patient_artifact_manifest.json
        Output CSVs/
          Preprocessing/
          MC simulation/
  cohort_assembly/
    cohort_assembly_manifest.json
    patient_batch_cohort_assembly.csv
    patient_batch_cohort_validation.csv
    assembled_tables/
      <cohort_table>.csv
```

Patient artifacts should remain useful even when cohort assembly is never run.
Cohort assembly should be repeatable from manifests without rerunning scientific
stages.
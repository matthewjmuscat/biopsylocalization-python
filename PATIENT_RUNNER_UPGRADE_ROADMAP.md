# Patient Runner Upgrade Roadmap

Last updated: 2026-05-19

This document is the durable, public planning surface for the migration from the
legacy all-patient monolith toward a validated per-patient runner. It should hold
accepted direction and upgrade sequencing. Private notes can still exist for
scratch reasoning, but stable decisions should graduate here.

## Documentation Policy

Use `DOCUMENTATION_INDEX.md` as the public map for durable documentation.

Use public Markdown docs for decisions that affect the codebase contract:

- pipeline architecture and migration phases,
- output artifact/schema policy,
- patient-runner compatibility rules,
- accepted deprecations/removals,
- validation gates,
- input/output contracts used by downstream analysis repos.

Use `.private_notes/` only for temporary research thinking, rough audits, and
planning that is not yet ready to become a codebase contract. Private notes are
ignored by git and should not be the only place where an implemented direction is
documented.

Prefer a small number of discoverable roadmap/design docs over many narrow notes.
When a narrow note becomes durable, either link it from this roadmap or fold its
important decisions into this file.

Recommended long-term shape:

- `README.md`: license, citation, and a short pointer to the documentation map,
- `DOCUMENTATION_INDEX.md`: current doc inventory and ownership map,
- `PATIENT_RUNNER_UPGRADE_ROADMAP.md`: active migration plan and sequencing,
- module-local docs beside the code they govern,
- dated generated audits under `validation_outputs/`,
- ignored `.private_notes/` only for scratch work.

## Current Migration Principle

The legacy monolith remains useful as a validation oracle, not as the future
architecture. New work should move toward:

- patient-local runtime execution,
- explicit patient artifacts,
- stable canonical join keys,
- durable table registry/data dictionary entries,
- downstream analysis joins outside the main algorithm,
- no hidden all-patient runtime dependencies inside the patient runner.

## Base Table Policy

The main algorithm should produce clean base artifacts, not every possible
analysis convenience table. Base artifacts should have:

- clear row grain,
- canonical primary/join keys,
- source-stage ownership,
- schema registry coverage,
- validation against legacy outputs where applicable.

Derived analysis tables, such as joined tissue-score/radiomics tables or
summary aggregates, can be regenerated downstream from the base tables. They do
not need to be core patient-runner outputs unless a downstream contract truly
requires them.

## Runtime State Migration

The current `master_structure_reference_dict` and `master_structure_info_dict`
are practical research-era containers, but they now carry too many meanings.
They should be reduced gradually rather than replaced in one large rewrite.

### master_structure_reference_dict

Current roles:

- patient registry,
- DICOM-derived structure store,
- preprocessing state,
- simulation state,
- dataframe/artifact cache,
- scratch communication bus between stages.

Migration direction:

1. Introduce thin typed wrappers around repeated access patterns.
2. Start with patient artifact/dataframe stores, because these touch the output
   registry and per-patient runner directly.
3. Keep `from_legacy_dict` / `to_legacy_dict` bridges during validation.
4. Move one stage at a time to typed accessors.
5. Eventually make the raw dictionary a compatibility/serialization layer, not
   the primary module API.

Good first object candidates:

- `PatientRuntimeState`,
- `PatientArtifactStore`,
- `StructureRecord`,
- `BiopsyRecord`,
- `PatientCase`.

Dataclasses are a good fit for these internal runtime objects because they are
standard-library, typed, readable, and light. They should not replace table
schema registries or manifests at external boundaries.

### master_structure_info_dict

Current roles:

- global run metadata,
- patient counts,
- cohort summary cache,
- selected config values,
- output path/state holder.

Migration direction:

Split this into explicit contracts:

- `RunManifest`: run ID, code version, output root, timing, patient list,
- `InputManifest`: discovered DICOM objects and inclusion/exclusion decisions,
- `PipelineConfig`: selected runtime parameters,
- `PatientSummary`: patient-local counts and metadata,
- `CohortSummary`: downstream assembler/analysis product only.

The patient runner should not require a true cohort summary to process one
patient.

## Removed Cohort-Derived Runtime Quantities

The patient-runner path should not compute all-patient statistics during the
main algorithm. Removed/disabled pathways are documented in
`python_files_dcm_meta_based/PATIENT_RUNNER_COHORT_DERIVED_QUANTITIES.md`.

Current policy:

- keep per-biopsy centroid-variation measurements,
- keep per-biopsy/per-patient uncertainty modes,
- remove `Global mean` biopsy variation uncertainty,
- remove `real mean` and `real normal` simulated biopsy length modes,
- remove all-patient mean fallback from `match real`,
- reintroduce cohort-derived priors later only as explicit per-patient sidecar
  inputs if scientifically needed.

## Input Contract Direction

Raw DICOM and local input data should remain ignored by git. The desired upgrade
is not to commit data, but to define a clearer local input root and contract.

DICOM matching should remain metadata-driven. Folder structure may help local
organization, but matching authority should come from DICOM identity and spatial
metadata, including patient/study/series identifiers, frame-of-reference identity,
referenced structure set information, and registration/spatial tags where
available.

A later sidecar/preflight utility can produce optional per-patient files such as
biopsy-variation or length priors. That utility should reuse the same contour and
biopsy measurement modules as the main algorithm and should stamp outputs with
code version, config identity, DICOM identity, and measurement method.

## Phase Plan

### Phase A: Validation Closure

Run the legacy all-patient pipeline with patient artifacts enabled and confirm
Phase 3C/3D stitch validation. Target: all registered stitch pairs match.

### Phase B: Crash Stability and Runner-Critical Hardening

Fix silent hard-exit failure points before expanding the runner. Runtime logs
should identify native-heavy or memory-heavy stages well enough to resume or
reduce workload safely.

Current known hard-exit signature from 2026-05-19:

- latest run died in optimizer-v2,
- no Python traceback was recorded,
- no runtime `ERROR` event was recorded,
- runtime status remained `running`, which means normal Python exception and
  `atexit` cleanup did not execute,
- last checkpoint was `optimizer_v2.structure.target_pack.end`, immediately
  before/inside the 10,000-trial prepared target containment pack for
  `181 (F1) / Bx_Tr_sim_target_dil_v2 DIL_RP`.

This points toward a hard termination path such as SIGKILL, native crash, session
termination, or allocation failure below Python exception handling. The same
input subset completed in a prior run, so this should be treated as an
optimizer-v2/native-heavy stability issue rather than a deterministic schema or
input-manifest bug.

### Phase C: Patient Runner Skeleton

Build a runner entrypoint that can execute one patient case independently using
existing modules with minimal scientific drift. It should write patient artifacts
only.

### Phase D: Cohort Assembly

Build or formalize a lightweight assembly step that concatenates validated base
patient artifacts and compares them to the legacy cohort outputs.

### Phase E: Contract/Object Cleanup

Gradually replace direct dictionary access with typed runtime wrappers and
explicit manifests. This should happen after the runner is useful enough to
support full-cohort execution and paper work.

### Phase F: Optional Preflight/Sidecar Utilities

Later, implement lightweight utilities for optional per-patient sidecar inputs.
These should share the exact same scientific modules as the main runner and do
only the narrow measurement job needed for the sidecar.

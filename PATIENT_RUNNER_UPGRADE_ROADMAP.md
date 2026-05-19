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

## Pipeline Boundary Map

Use these labels when documenting, extracting, or removing blocks from
`biopsy_localization_convex_main.py`. The labels are intentionally short so the
whole pipeline can be reviewed at a high level before individual functions are
rewired.

| Block | Scope | Patient-runner status | Notes |
| --- | --- | --- | --- |
| Discovery | run/cohort | keep outside patient execution | Finds inputs, builds manifests, resolves shared config, creates run/output roots. Discovery ends when patient case manifests exist. |
| Configuration/bootstrap | run/cohort | keep outside patient execution | Builds `PipelineConfig`, reference names, logging, UI, pools, and shared constants. |
| Preprocessing | patient | modularize for patient execution | Structure selection, interpolation, volumes, shape/radiomic features, MR summaries, biopsy preprocessing, and simulated biopsy preparation/planning. |
| Optimization | patient | modularize for patient execution | Optimizer v1/v2 should operate on one patient case plus shared config. |
| Simulated biopsy finalization | patient | modularize for patient execution | Finalizes simulated cores and validates planned-vs-realized per-biopsy geometry. |
| Sampling/classification | patient | modularize for patient execution | Sampled biopsy processing, target audit annotation, and double-sextant classification. |
| MC simulation | patient | modularize for patient execution | Transform generation, containment, dose, and MR simulation for one patient. |
| Patient artifact writing | patient | core patient-runner output | Writes stable base artifacts with canonical keys. |
| Cohort assembly | cohort/downstream | not per patient | Concatenates patient artifacts and builds required cohort outputs. |
| Migration validation | cohort/downstream | not per patient | Compares assembled patient outputs against the legacy cohort oracle. |
| Downstream analysis | sister repo or dedicated workflow | remove from main runtime | Random forest, derived joined summaries, production plots, and similar analysis products. |

A block becomes patient-runnable when it can operate on one `PatientCase` plus
shared configuration without reading other patients' in-memory state. If it
requires concatenation, aggregation, model fitting, or validation against the
legacy cohort output, it belongs in cohort assembly, migration validation, or a
sister analysis workflow.

## Run Plan Direction

The migration runner should use ordered run plans rather than one mutually
exclusive mode. A validation run may execute multiple steps in sequence, while a
normal future run may execute only patient execution and assembly.

Example validation plan:

```text
legacy_cohort -> patient_batch -> assemble_patient_outputs -> validate_against_legacy
```

Example future production plan:

```text
patient_batch -> assemble_patient_outputs
```

Given the native-heavy crash history, the long-term runner should allow these
steps to run in separate processes or separate run directories so memory and GPU
state can be released between major phases.

## Output Simplification Audit

Last reviewed: 2026-05-19.

Search roots used for the first sister-repository scan:

- `/home/matthew-muscat/Documents/UBC/Research/biopsy_dosimetry_analysis`
- `/home/matthew-muscat/Documents/UBC/Research/biopsy_tissue_class_stat_analysis`
- `/home/matthew-muscat/Documents/UBC/Research/biopsy_tissue_class_stat_analysis_corrected`

This scan is an input to removal decisions, not an automatic deletion rule.
"No direct sister reference found" means the exact cohort CSV name was not found
in code/docs during the scan; the user should still confirm whether the output is
scientifically useless, should be regenerated downstream, or should remain a
base artifact.

### Directly Referenced by Sister Repositories

Keep these available until the sibling contract is updated or a replacement
artifact is validated.

- `Cohort: 3D radiomic features all OAR and DIL structures.csv`
- `Cohort: Biopsy basic spatial features dataframe.csv`
- `Cohort: Nearest DILs to each biopsy.csv`
- `Cohort: Simulated biopsy preparation dataframe.csv`
- `Cohort: sum-to-one mc results.csv`
- `Cohort: global sum-to-one mc results.csv`
- `Cohort: Tissue class - distances global results.csv`
- `Cohort: Tissue class - distances pt-wise results.csv`
- `Cohort: Tissue class - distances voxel-wise results.csv`
- `Cohort: Per voxel prostate double sextant classification.csv`
- `Cohort: Global dosimetry (NEW).csv`
- `Cohort: Global dosimetry by voxel.csv`
- `Cohort: Bx DVH metrics (generalized).csv`

### No Direct Sister Reference Found in First Scan

These are simplification candidates. Prefer removing them from the core output
surface or regenerating them in cohort assembly/downstream unless the user
confirms they are base outputs.

- `Cohort: DIL global tissue scores and DIL features.csv` - derived join; keep out of patient runtime. - so this i think can proabbly just be deleted since it is a derived join of two base tables
- `Cohort: Global MR ADC statistics.csv` - summary product; may be optional unless MR analysis requires it.
- `Cohort: Global by voxel MR ADC statistics.csv` - MR output candidate; no direct sibling reference found.
- `Cohort: Guidance-map firing depth recommendations dataframe.csv` - likely useful for GUI/guidance workflow, not sister analysis.
- `Cohort: Per sample point prostate double sextant classification.csv` - per-sample detail; voxel-level table is the referenced one.
- `Cohort: Simulated biopsy planned vs realized centroid variation validation.csv` - migration/QA validation surface, not downstream analysis. - this is just a validation table, will never be used for analysis, should be gated and not output in a normal output folder but rather in a validation folder
- `Cohort: tissue class global scores (structure).csv` - summary product; may be regenerated from lower-level artifacts.
- `Cohort: tissue volume above threshold.csv` - summary product; no direct sibling reference found. - this should be calculated downstream i would think in an alsysis pipeline
- `Cohort: All MC structure transformation values.csv` - no exact sister reference found in the first scan.
- `Cohort: structure specific mc results.csv` - no exact sister reference found in the first scan, although it is a plausible base tissue-class artifact.
- `Cohort: Bx DVH metrics.csv` - no exact direct sister reference found; generalized DVH metrics are referenced. - this i think should be deleted as it is the old deprecated version

### Removal and Quarantine Ledger

Remove or quarantine non-core runtime surfaces before building the patient
orchestrator, so the new lane does not inherit boundaries around code that should
not survive.

| Item | Current evidence | Direction |
| --- | --- | --- |
| Random forest tumor morphology analysis | Runs inside main when `num_patients > 1`; no reason to be core localization runtime. | Remove from main or move to a sister/downstream analysis workflow. |
| FANOVA/Sobol pathway | Inputs default to zero, but imports, flags, runtime branch, and CSV export path remain in main. | Quarantine or remove from core runtime if confirmed deprecated. |
| Deprecated CSV writer calls | `csv_writers.csv_writer_containment` and `csv_writers.csv_writer_dosimetry` are commented legacy paths. | Remove dead/commented writer blocks after confirming no active dependency. |
| Production plots | Already skipped in main. | Keep outside core runtime as a dedicated plotting workflow. |
| Derived joined summaries | `Cohort: DIL global tissue scores and DIL features` is a convenience join. | Regenerate downstream from base artifacts if needed. |
| Global sum-to-one MC summary | Directly referenced by tissue-class repos but derived from long-form sum-to-one rows. | Keep during migration, then rebuild in cohort assembly or sister repo. |
| Legacy/simple DVH metrics | Generalized DVH metrics are referenced; old `Cohort: Bx DVH metrics.csv` was not found directly. | Confirm whether old DVH metric output can be removed or replaced by a clean DVH service. |

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

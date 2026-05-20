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

- `Cohort: Global MR ADC statistics.csv` - summary product; may be optional unless MR analysis requires it.
- `Cohort: Global by voxel MR ADC statistics.csv` - MR output candidate; no direct sibling reference found.
- `Cohort: Guidance-map firing depth recommendations dataframe.csv` - likely useful for GUI/guidance workflow, not sister analysis.
- `Cohort: Per sample point prostate double sextant classification.csv` - per-sample detail; voxel-level table is the referenced one.
- `Cohort: tissue class global scores (structure).csv` - summary product; may be regenerated from lower-level artifacts.
- `Cohort: All MC structure transformation values.csv` - no exact sister reference found in the first scan.
- `Cohort: structure specific mc results.csv` - no exact sister reference found in the first scan, although it is a plausible base tissue-class artifact.

### Removed or Gated After User Review

These first-pass decisions simplify the core output schema before the patient
orchestrator is built.

- `Cohort: DIL global tissue scores and DIL features.csv` - removed from the core output schema; it is a derived join of base tissue-score and radiomic tables and should be regenerated downstream if needed.
- `Cohort: tissue volume above threshold.csv` - removed from the core output schema; threshold summaries should be calculated by an analysis pipeline from base tissue-class outputs.
- `Cohort: Bx DVH metrics.csv` - removed from the core output schema; this is the old deprecated DVH surface and is superseded by `Cohort: Bx DVH metrics (generalized).csv` until a clean DVH service replaces both.
- `Cohort: Simulated biopsy planned vs realized centroid variation validation.csv` - validation-only; it should be gated into validation output, not written as a normal cohort CSV.

### Tissue-Class Grain Clarification

`Cohort: structure specific mc results.csv` and
`Cohort: tissue class global scores (structure).csv` are related but not the
same table.

- `Cohort: structure specific mc results.csv` is the granular long-form MC output at approximately patient + biopsy + MC trial + point/voxel + relative structure grain. This is closer to a base artifact.
- `Cohort: tissue class global scores (structure).csv` is an aggregated biopsy-level/by-relative-structure summary computed from the granular structure-specific MC results. This is a derived summary and can be regenerated from lower-level artifacts.

### Removal and Quarantine Ledger

Remove or quarantine non-core runtime surfaces before building the patient
orchestrator, so the new lane does not inherit boundaries around code that should
not survive.

| Item | Current evidence | Direction |
| --- | --- | --- |
| Random forest tumor morphology analysis | Runs inside main when `num_patients > 1`; no reason to be core localization runtime. | Removed from main-facing runtime; source preserved under `python_files_dcm_meta_based/deprecated/`. |
| FANOVA/Sobol pathway | Inputs default to zero, but imports, flags, runtime branch, and CSV export path remain in main. | Removed from main-facing runtime; source preserved under `python_files_dcm_meta_based/deprecated/`. |
| Deprecated CSV writer calls | `csv_writers.csv_writer_containment` and `csv_writers.csv_writer_dosimetry` are commented legacy paths. | Removed from main-facing runtime; source preserved under `python_files_dcm_meta_based/deprecated/`. |
| Production plots | Already skipped in main. | Keep outside core runtime as a dedicated plotting workflow. |
| Derived joined summaries | `Cohort: DIL global tissue scores and DIL features` is a convenience join. | Removed from core output schema; regenerate downstream from base artifacts if needed. |
| Global sum-to-one MC summary | Directly referenced by tissue-class repos but derived from long-form sum-to-one rows. | Keep during migration, then rebuild in cohort assembly or sister repo. |
| Tissue volume threshold summary | No direct sister-reference found; summary is downstream-calculable. | Removed from core output schema; calculate downstream if needed. |
| Legacy/simple DVH metrics | Generalized DVH metrics are referenced; old `Cohort: Bx DVH metrics.csv` was not found directly. | Removed from core output schema; generalized DVH remains during migration. |

### Deferred MC Simulator Cleanup Audit

`python_files_dcm_meta_based/MC_simulator_convex.py` still contains substantial
research-era and debug-era code that should not be inherited blindly by the
patient runner. This is important cleanup work, but it should be handled as a
separate audit lane rather than as a blocker for the first patient-runner
scaffold.

Candidate cleanup areas:

- old commented-out implementation blocks that are clearly deprecated,
- block-quoted timing/debug experiments that should either be deleted or gated
  with explicit disabled branches such as `if False`,
- tissue-volume-threshold calculations that are downstream-calculable from base
  tissue-class artifacts,
- old/simple DVH metric calculations that are superseded by generalized DVH or
  can be rebuilt by dosimetry analysis workflows,
- differential/cumulative DVH runtime dictionaries and dose-volume metric caches
  if no current registered artifact, validation route, or sister workflow depends
  on them being produced inside the main MC simulator.

Cleanup rule: do not remove scientific calculations from the MC simulator solely
because they are not currently exported. First trace whether they feed registered
base artifacts, validation outputs, planning/guidance state, sister repositories,
or manuscript-era results. Each removal should identify the replacement location:
patient artifact, cohort assembly, dosimetry/tissue-class sister workflow, or
deleted with no replacement.

Recommended audit artifact: a table with one row per candidate block/function
covering current producer, current consumer, output artifact if any, sister-repo
or paper usage, proposed action, and validation requirement.

### Deferred Render Asset Cleanup

The repository contains old Open3D camera/render JSON files and old screen
capture PNGs. These should not become part of the patient-runner contract.

Current evidence from the 2026-05-19 render audit:

- 118 loose root-level `ScreenCamera_*.json`, `DepthCamera_*.json`, and
  `RenderOption_*.json` files were ignored local artifacts with no direct code
  references; they were moved into the ignored `render_jsons/root/` holding
  folder,
- 2 loose camera/render JSONs under `python_files_dcm_meta_based/` were also
  ignored local artifacts; they were moved into
  `render_jsons/python_files_dcm_meta_based/`,
- 7 named Open3D screen-camera JSONs remain under `open3d_views_jsons/` because
  the legacy MC/MR demonstration plotting paths still reference them through
  `dose_views_jsons_paths_list` and `containment_views_jsons_paths_list`,
- the optimizer-v2 render/GUI path does not rely on the loose root-level camera
  JSON pile,
- the sister analysis repositories searched in the output audit do not reference
  these camera/render JSON files or the old `plot_two_views_side_by_side` helper,
- top-level PNG captures are old local artifacts and have been moved into the
  ignored `png/` holding folder.

Recommendation:

- keep the 7 `open3d_views_jsons/` files only as temporary legacy demo presets,
- do not make Open3D camera JSON files part of any new patient-runner, GUI, or
  scientific-media contract,
- replace the old manual still-frame workflow with a named render-job surface:
  scene type, patient/structure selection, layer selection, camera preset, frame
  schedule, output resolution, output directory, and export manifest,
- use deterministic frame export as the stable contract: write PNG/SVG/PDF
  frames plus a manifest, then optionally package frames into MP4 or GIF,
- keep Open3D as the first interactive/debug backend because the repo already
  has Open3D geometry objects,
- use Plotly plus Kaleido for lightweight static publication-style exports where
  it is sufficient,
- consider PyVista/VTK for higher-quality offscreen scientific movies if Open3D
  offscreen rendering or Plotly export becomes limiting,
- consider imageio or direct `ffmpeg` packaging for turning deterministic frame
  directories into video artifacts,
- keep generated screenshots, timing captures, exploratory images, and loose
  camera dumps out of git.

Deletion gate: once the legacy MC/MR demonstration plotting paths are removed or
rewired through the new render-job module, delete `open3d_views_jsons/` unless a
specific preset is promoted into a named render preset contract.

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

Current bridge boundary: raw `master_structure_reference_dict` and
`master_structure_info_dict` are still accepted by the additive runner only at
the legacy bridge and batch-from-legacy entrypoints. This is intentional for
validation against the monolith, but it is not the desired long-term stage API.
New stages should receive typed patient-runner objects. A near-term cleanup can
introduce a `LegacyCohortRuntimeState` wrapper around the two master dictionaries
to make the transitional boundary explicit before deeper stage migrations.

Patient identity policy: patient IDs are lookup keys in the legacy dictionaries,
so the runner must preserve them exactly. Validation may reject non-string,
empty, or duplicate IDs, but it should not strip, case-fold, slugify, or coerce
patient IDs before dictionary lookup. Filesystem-safe patient directory names are
derived output-path values only and must not become runtime identity keys.

## Patient Runner Data and Code Standards

The patient-runner module should be treated as a clean new boundary around the
existing scientific code, not as a place to copy the monolith's informal state
style. New code should be explicit, typed, auditable, and boring to read.

### Data Type Choices

Use standard-library Python 3.11 typing first. Do not add a new validation
dependency until a real external-boundary problem requires it.

Recommended types:

- `@dataclass(frozen=True, slots=True)` for immutable value objects, configs,
  manifests, identity records, and completed stage results.
- Plain `@dataclass(slots=True)` for intentionally mutable runtime wrappers,
  such as a patient-local compatibility bridge around legacy dictionaries.
- `Enum` or `Literal` for closed stage/status names that are used in logs,
  manifests, and run plans.
- `TypedDict` for narrow legacy-dictionary views when the dictionary shape must
  remain visible during migration.
- `Protocol` for small interfaces such as artifact writers, stage runners, and
  validators when multiple implementations are expected.
- `pathlib.Path` for filesystem contracts; avoid passing raw path strings across
  new module boundaries.
- `Mapping`, `MutableMapping`, `Sequence`, and `Iterable` in function signatures
  when callers should not depend on concrete container classes.
- `numpy.ndarray` and `pandas.DataFrame` remain acceptable for scientific arrays
  and tables, but their expected shape, columns, grain, and key fields must be
  documented at the boundary that accepts or returns them.

Potential later option: Pydantic can be evaluated for serialized manifests,
sidecar files, or user-editable configuration if plain dataclasses plus
`__post_init__` validation become too weak. It should not be introduced for hot
numeric loops, dataframe schemas, or internal scratch state unless there is a
clear benefit.

### Contract Layers

Keep these contract layers separate:

- Runtime contracts describe Python objects passed between patient-runner stages.
- Artifact contracts describe files, table grain, keys, schemas, and registry
  metadata.
- Validation contracts describe how patient outputs are compared to legacy
  cohort outputs.
- Legacy bridge contracts describe the minimum subset of
  `master_structure_reference_dict` and `master_structure_info_dict` required by
  an existing stage.

The patient runner should not smuggle all-patient state through a patient-local
API. If a stage needs cohort-level information, that dependency must be an
explicit sidecar input, cohort-assembly step, or validation step.

Initial object contracts:

- `PatientCase`: immutable patient identity plus input references needed to run
  one patient.
- `PatientRunConfig`: immutable patient-run settings derived from the shared
  pipeline config.
- `PatientRuntimeState`: patient-local wrapper around the existing legacy state,
  used only while stages are being migrated.
- `PatientArtifactStore`: controlled write surface for registered patient
  artifacts and manifests.
- `PatientStageResult`: immutable status, timing, warnings, and artifact summary
  for one stage.
- `PatientRunResult`: immutable run-level summary that can be consumed by cohort
  assembly and validation.

### Code Style Rules

New patient-runner code should follow these rules unless a local module has a
stronger convention:

- small modules with one responsibility each,
- small functions with explicit inputs and return values,
- no hidden mutation of global state,
- no bare string stage names spread across modules; centralize constants or
  enums,
- no block-quoted dead code or large commented-out implementation alternatives,
- timing through a shared timing/logging helper rather than ad hoc stopwatch
  snippets,
- debug or demonstration code behind explicit config flags, not implicit
  comments,
- exceptions should identify patient ID, stage, and artifact or structure when
  available,
- logs should include stage start/end, elapsed seconds, artifact counts, and
  validation summaries,
- artifact writers should be idempotent where practical and should write a
  manifest that can be audited without loading Python pickles.

Docstrings should explain contract, grain, and side effects. They should not
restate obvious assignments. Comments should be reserved for non-obvious
scientific assumptions, legacy compatibility constraints, or invariants that a
future maintainer could accidentally break.

### Migration Discipline

The first patient-runner code should wrap existing scientific implementation
rather than rewrite it. Refactors should be contract-preserving until assembled
patient outputs match the legacy cohort oracle. Cleanup passes, especially in
`MC_simulator_convex.py`, should be separate from patient-runner scaffolding
unless the cleanup is required to expose a patient-local boundary.

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

## Parallelism Opportunities and Recommendations

Per-patient execution creates a real opportunity for faster full-cohort runs,
but the safe unit of parallelism depends on the stage contract.

The long-term target is not "run the whole patient pipeline in many threads."
The safer target is bounded patient-level process workers: each worker receives
one patient case plus explicit serialized inputs, writes that patient's artifacts
to an independent output directory, records timing/memory status, and exits so
native/GPU memory can be released by the operating system.

### Good Opportunities

- Artifact writing can use light thread parallelism when each patient writes
  independent files and no shared scientific state is mutated.
- Full patient jobs may eventually run in parallel as separate processes once
  inputs are patient-local and serializable.
- A small bounded worker count such as 2 or 4 patients at a time is plausible if
  measured peak memory leaves enough headroom.
- Phase-level process isolation can improve crash recovery even before full
  patient-level parallelism: discovery, legacy validation, patient execution,
  cohort assembly, and validation can run as separate process steps.
- Per-patient process workers can make failures easier to resume because a
  failed patient can be retried without keeping the whole cohort in memory.

### Recommendations Before Full Patient Parallelism

- Keep a correct sequential patient runner as the reference path.
- Measure peak resident memory, runtime, artifact count, and output size for one
  representative patient before increasing `max_workers`.
- Benchmark worker counts in a ladder such as 1, 2, then 4, stopping when memory
  headroom, I/O pressure, or native stability becomes questionable.
- Prefer process workers over threads for optimizer, Open3D, MC simulation,
  large-array, or native-heavy stages.
- Make each worker write to a patient-specific directory and a patient-specific
  manifest; cohort assembly should concatenate artifacts after patient workers
  finish.
- Keep cohort-level aggregation, validation, plotting, and model fitting outside
  patient workers.
- Treat worker count as a config value derived from measured memory budget, not
  from CPU count alone.

### Non-Recommendations

- Do not run native-heavy scientific stages for multiple patients in threads
  while sharing `master_structure_reference_dict` or global mutable state.
- Do not parallelize the full runner before the sequential patient runner is
  validated against the legacy cohort oracle.
- Do not let multiple workers write the same cohort-level output file.
- Do not assume that a serial full-cohort run implies unlimited patient-level
  parallelism; each worker may duplicate imports, arrays, DICOM state, geometry,
  optimizer state, and output buffers.
- Do not choose `max_workers=os.cpu_count()` for memory-heavy stages without a
  measured memory budget and failure recovery plan.

### Practical Direction

If the existing all-patient run can complete with many patients and more than
100 total real/simulated biopsies, bounded patient-level process execution is a
reasonable optimization target. The expected safe path is to make each process
hold only a small patient-local working set, write durable patient artifacts,
then exit. That may allow 2 or 4 concurrent patients while keeping total memory
below the current all-patient peak, but it must be proven with instrumentation
rather than assumed.

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

Immediate Phase C.0 scope:

- create a small patient-runner module/package without rewriting scientific
  algorithms,
- define the minimal contracts for `PatientCase`, patient-local runtime state,
  run config, and patient-run result,
- add a bridge that can carve one patient out of the legacy dictionaries and run
  existing stages against that patient-local state,
- write registered patient artifacts through the existing artifact/registry
  surface,
- keep the legacy all-patient monolith as the validation oracle during this
  phase,
- defer broad MC simulator cleanup until the audit above identifies safe removal
  boundaries.

The first patient-runner implementation should be boring on purpose: no new
science, no output schema churn, and no deep cleanup mixed into the runner
scaffold. Its job is to prove that one patient can move through the existing
pipeline boundary with explicit inputs, outputs, timing, and logs.

Phase C.1 adds a batch layer around the one-patient runner. It should resolve an
ordered patient list from the legacy patient registry, carve each patient through
the legacy bridge, run the existing patient stage sequence, optionally use
thread parallelism for patient artifact writing, and return typed batch results.
The batch config must wrap `PatientRunConfig` rather than duplicating legacy key
names or output policy.

Phase C.1 parallelism is deliberately conservative. Threads are acceptable for
the current artifact-writing stage because each patient writes independent files
from existing in-memory objects. They are not the target model for native-heavy
or memory-heavy science stages. Once patient inputs are serializable and stage
contracts no longer rely on shared all-patient dictionaries, add a process-worker
entrypoint that can execute one patient per process and release native/GPU state
when the process exits. Worker count should be limited by measured per-patient
memory and output I/O pressure, not by CPU count alone.

Expected next steps:

1. tighten the transitional legacy-cohort boundary, optionally with a
  `LegacyCohortRuntimeState` wrapper,
2. add cohort assembly/stitch validation for artifacts produced by
  `PatientBatchRunResult`,
3. migrate one scientific stage behind a typed patient-local interface,
4. add process-isolated patient execution only after the stage input contract is
  serializable and memory requirements are measured.

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

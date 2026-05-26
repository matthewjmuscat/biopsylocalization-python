# Patient Runner Upgrade Roadmap

Last updated: 2026-05-23

This document is the durable, public planning surface for the migration from the
legacy all-patient monolith toward a validated per-patient runner. It should hold
accepted direction and upgrade sequencing. Private notes can still exist for
scratch reasoning, but stable decisions should graduate here.

## Documentation Policy

Use `../DOCUMENTATION_INDEX.md` as the public map for durable documentation.

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
- `docs/DOCUMENTATION_INDEX.md`: current doc inventory and ownership map,
- `docs/roadmap/PATIENT_RUNNER_UPGRADE_ROADMAP.md`: active migration plan and sequencing,
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

## Current Architecture Decision: Oracle Separate From Runner

Accepted direction as of 2026-05-22: do not keep mutating the legacy cohort path
into the per-patient path. The validated main/cohort path should remain the
oracle, while new patient-facing scientific modules are built as a separate
runner path and compared against that oracle.

Canonical placement for those patient-facing scientific modules now lives in
`../architecture/PATIENT_MODULE_TREE_GUIDE.md`.

Keep three layers distinct:

- Legacy oracle path: current main plus existing cohort modules. This path stays
  runnable and trusted after validation. Avoid in-place MC simulator cleanup here
  unless a focused bug fix is explicitly approved.
- Patient scientific modules: one-patient functions that preserve the same
  scientific operations, dictionary writebacks, array shapes, and ordering as the
  oracle for that stage. These may initially copy known-good legacy scientific
  blocks into new patient modules instead of refactoring the legacy file. Prefer
  stage-local patient modules inside the existing scientific package tree, with a
  family-local `per_patient/` subpackage when a family grows. Do not introduce a
  top-level `python_files_dcm_meta_based/patient_stages/` tree.
- Runner and assembly layer: selects patients, calls patient stages, writes
  patient artifacts, assembles cohort tables, and compares against the oracle.

For high-risk scientific blocks, especially MC simulation, copy-assisted
extraction is preferred over clever refactoring. A safe pattern is to create a
small patient-module skeleton with explicit inputs/config/output notes, then
paste the relevant legacy scientific loop body into the patient function with
minimal mechanical edits. Validate that patient module, then compose it into the
runner. This keeps the legacy oracle untouched and reduces accidental algorithm
drift.

Validation should be staged but not endless:

- validate the current semi-modular main path as the checkpoint oracle,
- validate each newly copied/extracted patient submodule with focused comparisons,
- validate the assembled per-patient outputs against the oracle at integration
  gates,
- avoid maintaining permanent duplicate sidecars after a stage has a stable
  patient module and final assembly comparison.

Parity validation should be wired as a post-run saved-state comparison, not as an
inline fork inside the scientific path. The legacy cohort path should run end to
end and persist its oracle state/artifacts. The patient runner should then run
end to end in its own output root and persist patient artifacts plus assembled
cohort tables. A reusable comparator layer should load both completed surfaces,
compare registered keys, arrays, dataframe schemas/row grains, sorted or stable
row values, numeric tolerances, and performed flags, then write durable validation
reports. This keeps both execution paths uninterrupted and prevents validation
code from becoming hidden scientific routing.

The full configuration overhaul remains intentionally after first patient-runner
validation. Until then, patient stages should use explicit transitional config
contracts and adapter mapping from the current main settings. After the runner is
validated, the config layer can be redesigned for GUI/UI compatibility, run
plans, patient selection, stage toggles, and richer validation modes, followed by
another validation pass against the same oracle outputs.

Batch parallelism is not a first-order requirement for the scientific runner.
The reference runner should be sequential and deterministic first. Any existing
`starmap`-style helper needed inside a patient stage can be represented as a
plain loop in the patient module or through a sequential execution context. Batch
parallelism, process workers, and memory-aware scheduling come later after the
patient-local data contract is stable.

## Legacy Datatype Boundary Direction

The runner should not become permanently coupled to `master_structure_reference_dict`,
`master_structure_info_dict`, or other legacy mutable dictionaries. These objects
remain the validation backing store for now, but new code should isolate them at
adapter boundaries.

Near-term rule:

- first extract main-facing scientific blocks into behavior-preserving wrappers,
- make wrapper entrypoints patient-scoped where practical,
- pass legacy dictionary key names and threshold/config values explicitly,
- write the same legacy keys during the validation period,
- keep typed runner contracts and artifact manifests independent of the legacy
  dictionary shape.

Datatype direction for new patient surfaces:

- use dataclasses for stable patient identity, configuration, stage results,
  and patient-state boundaries,
- keep legacy dictionaries as adapter/output forms while oracle parity is still
  being tested,
- avoid converting every nested structure record to a dataclass in the same pass
  unless the fields and mutability rules are stable,
- prefer explicit conversion methods such as `to_legacy_dict()` and
  `from_legacy_dict(...)` over scattering ad hoc dictionary construction through
  runner code,
- keep generic legacy dictionary spellings in
  `python_files_dcm_meta_based/legacy_data_keys.py` and package/family-specific
  spellings in local key contract modules when a stage still writes old
  dictionaries; do not duplicate `Global`, `By patient`, `MC info`, `Ref #`,
  generic structure metadata/geometry/sample keys, or `MC data: ...` strings
  across adapters and collectors,
- do not broad-refactor raw legacy key literals in the frozen oracle or older
  mutable preprocessing wrappers just to satisfy style; move those call sites to
  contracts only when they cross into additive patient, runner, artifact, or
  validation boundaries,
- allow shallow `dict(...)` copies at adapter boundaries for metadata and legacy
  compatibility, but do not treat those copies as the final scientific data
  model.

For MC containment specifically, extract the setup and computation in small
validated slices. The patient-local relative-structure inventory should remain a
neutral module, while containment owns the dilation bank, per-biopsy input prep,
core containment helper calls, and statistics/writeback logic. These additive
helpers exist, but the frozen oracle path should remain the live route until a
patient-level parity harness proves row/key equivalence. The raw CUDA containment
and nearest-neighbour kernels should remain untouched; patient modules should
call the same kernel helper APIs as the oracle until parity is proven.

For MC dose specifically, the additive patient module now owns dose and
dose-gradient lattice context construction, per-biopsy nearest-neighbour
localization through the existing `dosimetric_localizer` helper, point-by-trial
array compilation, and DVH compile/writeback helpers. It deliberately does not
own raw CSV dumps, plotting, Rich progress, or live routing through the frozen
oracle. Legacy inactive dose-statistics and voxelization blocks should only be
extracted if downstream parity checks prove those outputs are still required.

For MC MR specifically, the additive patient module now owns filtered MR ADC
lattice reconstruction, KD-tree context construction, per-biopsy
nearest-neighbour localization through the existing `mr_localizers` helper,
point-by-trial array compilation, output collection, and legacy biopsy-record
writeback. It deliberately does not own raw CSV dumps, plotting, Rich progress,
or live routing through `MC_simulator_MR.py`. A singleton MR oracle adapter exists
for validation runs, but the MR simulator itself remains frozen until a
patient-level parity harness proves row/key equivalence.

For downstream MC outputs specifically, the additive patient module now owns
singleton wrappers around the existing MC transform, tissue/containment,
dosimetry/DVH, MR ADC dataframe-fragment builders, and the optimizer-v2
downstream MC-score annotation call. These wrappers build the same legacy
patient and biopsy dataframe stores for one patient, but they do not write CSVs,
perform cohort stitching, or route the frozen main/oracle path.

This makes each stage replaceable in two steps: first the old code is moved
behind a named boundary with identical behavior, then a typed data model can be
introduced behind the same runner-facing contract after validation is green.

Current preprocessing examples of this direction include
`python_files_dcm_meta_based/preprocessing/dose_grid_processing.py`,
`python_files_dcm_meta_based/preprocessing/mr_adc_grid_processing.py`, and
`python_files_dcm_meta_based/preprocessing/structure_processing/raw_contour_pulling.py`.
Each keeps legacy side effects during validation while moving patient-scoped
runtime work out of `biopsy_localization_convex_main.py`.

Future GUI-facing orchestration should preserve this separation. A GUI or CLI
should be able to offer actions such as map dose, map MR, run targeting, run QA,
or execute a full predefined pipeline by selecting independent stage surfaces
based on available input data, user choice, or both.

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

The detailed stage-by-stage readiness checklist lives in
`PATIENT_RUNNER_MODULE_READINESS.md`. Use that document for implementation
tracking; keep this section as the higher-level boundary map.

Use these labels when documenting, extracting, or removing blocks from
`biopsy_localization_convex_main.py`. The labels are intentionally short so the
whole pipeline can be reviewed at a high level before individual functions are
extracted.

| Block | Scope | Patient-runner status | Notes |
| --- | --- | --- | --- |
| Discovery | run/cohort | keep outside patient execution | Finds inputs, builds manifests, resolves shared config, creates run/output roots. Discovery ends when patient case manifests exist. |
| Configuration/bootstrap | run/cohort | keep outside patient execution | Builds `PipelineConfig`, reference names, logging, UI, pools, and shared constants. |
| Preprocessing | patient | modularize for patient execution | Structure selection, interpolation, volumes, shape/radiomic features, MR summaries, biopsy preprocessing, and simulated biopsy preparation/planning. |
| Optimization | patient | modularize for patient execution | Optimizer v1/v2 should operate on one patient case plus shared config. |
| Simulated biopsy finalization | patient | modularize for patient execution | Finalizes simulated cores and validates planned-vs-realized per-biopsy geometry. |
| Sampling/classification | patient | modularize for patient execution | Sampled biopsy processing, target audit annotation, and double-sextant classification. |
| MC simulation | patient | build separate patient modules | Transform generation, containment, dose, and MR simulation for one patient. Leave the existing cohort MC simulator callable as the oracle. |
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
New stages should receive typed patient-runner objects. Phase C.1 introduces
`LegacyCohortRuntimeState` as the named transitional boundary around the two
master dictionaries before deeper stage migrations.

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

"Behind a typed patient-local interface" means the runner calls a small typed
stage boundary rather than passing raw all-patient dictionaries into scientific
code directly. During migration, that boundary may still adapt typed inputs into
the one-patient legacy-shaped dictionary expected by old functions. The adapter
is temporary: it localizes legacy shape assumptions so the stage internals can be
moved from raw dictionary access to typed accessors/dataclasses without changing
the runner API.

Important distinction: the first adapter does not require rewriting the
scientific module. The adapter can still call the existing function with the
one-patient legacy-shaped dictionary it expects. Refactoring the scientific
module comes later, once the wrapper and validation prove that the stage is
correctly isolated. This lets the runner become cleaner before the old scientific
functions are fully rewritten.

Best attack for phasing out the dictionaries:

1. identify one patient stage and list the exact legacy keys/arrays/tables it
  reads and writes,
2. define a small typed input/output contract for that stage,
3. build a legacy adapter that fills the contract from the current one-patient
  legacy-shaped view,
4. keep validation comparing adapter-backed outputs to the legacy oracle,
5. move repeated internal dictionary access into typed accessors,
6. eventually replace the adapter's backing store without changing the stage
  contract.

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
- do not define module-local `ALL_CAPS` constants for legacy dictionary or
  dataframe keys when the value should come from the active key policy, typed
  accessor, or adapter contract; `ALL_CAPS` is acceptable for true constants
  such as schema versions and filenames,
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

For MC simulation, prefer separate patient modules over in-place legacy-module
cleanup. The current cohort MC files are the oracle. New patient modules may
copy the relevant scientific body and adapt only the outer function signature,
patient-local inputs, progress/log hooks, and return/writeback boundary. If a
cohort wrapper is needed during transition, it should be a transparent loop over
the patient module and should preserve the existing stage order.

Avoid designing scientific stage contracts around `starmap` or other batch
parallel APIs. A patient-local module should be runnable with a plain loop. A
future execution context can supply process or thread mapping only after the
stage contract is serializable and validated.

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

This process-worker model also improves robustness. If patients A, B, and C are
running as separate worker processes and C exits unexpectedly, the parent runner
can detect the missing/failed C result, preserve successful A and B artifacts,
retry only C up to a configured attempt limit, and record the failure if retries
do not recover. This does not eliminate native crashes, but it prevents one
silent patient failure from invalidating the entire batch without evidence.

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
the legacy bridge, run the existing patient stage sequence, optionally use the
explicit thread backend for patient artifact writing, and return typed batch
results. The batch config must wrap `PatientRunConfig` rather than duplicating
legacy key names or output policy. Sequential execution is the reference backend
and default.

Phase C.1 parallelism is deliberately conservative. Threads are acceptable for
the current artifact-writing stage because each patient writes independent files
from existing in-memory objects. They are not the target model for native-heavy
or memory-heavy science stages. Once patient inputs are serializable and stage
contracts no longer rely on shared all-patient dictionaries, add a process-worker
entrypoint that can execute one patient per process and release native/GPU state
when the process exits. Worker count should be limited by measured per-patient
memory and output I/O pressure, not by CPU count alone.

Expected next steps:

1. add cohort assembly/stitch validation for artifacts produced by
  `PatientBatchRunResult`,
2. migrate one scientific stage behind a typed patient-local interface,
3. keep the migrated runner stages sequential until patient-local inputs and
  outputs are stable,
4. add process-isolated patient execution only after the stage input contract is
  serializable and memory requirements are measured.

### Phase D: Cohort Assembly

Build or formalize a lightweight assembly step that concatenates validated base
patient artifacts and compares them to the legacy cohort outputs.

Initial Phase D scope:

- inventory the artifact files written by `PatientBatchRunResult`,
- optionally assemble selected cohort-style tables from patient fragments using
  the existing stitch-pair definitions,
- allow assembly callers to select patients, source table names, final cohort
  table names, and output directories,
- optionally compare assembled tables to legacy final cohort dataframes supplied
  by the validation caller,
- write validation-side outputs outside the production patient artifact tree
  only when requested.

This phase validates the patient artifact surface. It does not yet migrate a
scientific stage, replace the legacy dictionaries, or add process workers.

The assembly layer should remain usable as a standalone utility. A future CLI or
GUI can inspect patient artifact inventories, show table/schema versions and
compatibility status, let the user select patients and tables, choose a cohort
output directory, then assemble only the requested outputs. This keeps patient
execution durable and incremental: adding a patient artifact later should not
require rerunning the full patient pipeline just to rebuild a cohort table.

### Phase E: Scientific Stage Migration and Instrumentation

Move scientific stages behind typed patient-local interfaces one stage at a
time. Phase E eventually covers every scientific module required for per-patient
execution, but it should not be one giant rewrite. Each stage migration needs a
typed contract, a legacy adapter, validation against the monolith, and patient
run logging.

Phase E should not keep blending two concepts: the legacy cohort oracle and the
new patient-runner pathway. Legacy cohort code remains the comparator. New
patient-facing modules are the implementation path. Additive extraction should
not rewire the frozen cohort/oracle path to call new patient modules. Separate
validation utilities can iterate over patient modules when the patient-runner
path is ready to compare against the oracle.

For MC simulation, prefer this extraction order:

1. transform sample generation and MC prep,
2. biopsy-only transforms,
3. biopsy-to-relative-structure transforms,
4. nominal containment,
5. MC containment and distance calculations,
6. dose localization and dose result compilation,
7. DVH/statistical summaries,
8. MR ADC MC localization.

Each item should first produce a patient-level function. If a main-facing cohort
wrapper is needed, it should call the same patient function in a simple loop and
preserve the current cohort-stage ordering for validation.

Required instrumentation for migrated patient stages:

- patient and stage start/end timestamps,
- elapsed seconds,
- status and exception summaries,
- warning counts/messages,
- artifact counts and output paths,
- config/run IDs and input manifest IDs,
- future process-worker attempt number, PID, exit code, timeout/retry reason,
  and memory measurements where available.

Current manifest/log surface:

- `patient_run_manifest.json` is written beside each patient's output artifacts
  from `PatientRunResult`,
- `patient_batch_run_manifest.json` is written under the batch output root from
  `PatientBatchRunResult`,
- manifest writing is enabled by default but controlled by
  `PatientRunConfig.write_patient_run_manifest` and
  `PatientBatchRunConfig.write_batch_run_manifest`,
- these manifests are the first durable logging surface; richer event logs and
  memory/process fields should build on them rather than create a separate
  hidden status format.

Main-facing validation gate:

- legacy main remains the oracle path and must stay callable,
- patient-runner execution is introduced through an explicit gate, not by
  replacing the legacy path in-place,
- `shadow_output` mode runs after legacy outputs are present, writes patient
  artifacts from the completed legacy state, assembles cohort outputs, and
  compares them with the legacy final dataframes,
- this mode validates the patient artifact/export/assembly layer before any
  independently recomputed scientific patient stages are promoted,
- later dual-science validation modes may run both the legacy path and migrated
  patient stages in one invocation, but that promotion is intentional and gated.

Scientific modularization rule:

- first-pass modularization means moving existing main-facing scientific code
  into main-facing wrappers with the same inputs, ordering, side effects, and
  legacy dictionary keys,
- do not change scientific algorithms, formulas, object semantics, or data shape
  in the same pass as orchestration extraction,
- any later scientific cleanup must be small, deliberate, immediately tested,
  and understood as a scientific behavior change rather than runner plumbing.

Current Phase E preprocessing status:

- non-biopsy structure preprocessing has an extracted modular surface and uses
  that modular path by default,
- selected-structure selection has an extracted modular surface and a legacy
  sidecar comparison,
- MR ADC input checking/series normalization has a patient-level helper and a
  cohort wrapper,
- prostate-only MR ADC post-processing is already behind a preprocessing helper,
- dose-grid runtime-object construction lives behind
  `preprocessing/dose_grid_processing.py`, with the same legacy dictionary
  writeback keys and dose helper calls,
- ADC-MR grid runtime-object construction lives behind
  `preprocessing/mr_adc_grid_processing.py`, with the same legacy dictionary
  writeback keys and MR helper calls,
- patient raw-contour pulling lives behind
  `preprocessing/structure_processing/raw_contour_pulling.py`, with the same
  RTSTRUCT read path and legacy contour/centroid writeback keys,
- the OAR/rectum/urethra/DIL preprocessing path uses
  `process_standard_non_biopsy_structure_preprocessing_stage(...)`, which
  dispatches to the shared family loop; DIL-specific behavior is controlled
  through family configuration/context rather than a separate DIL module,
- `Structure preprocessing validation` is registered as a validation-only
  patient preprocessing artifact for focused modular-vs-legacy validation runs,
- the opt-in validation shadow path still carries the large legacy family loops
  in `main`, but validation now wraps the whole OAR/rectum/urethra/DIL stage
  from outside through
  `preprocessing/structure_processing/non_biopsy_structure_stage_validation.py`,
- the legacy inline family loops no longer call the modular single-structure
  processor from inside each structure loop,
- bottom-of-main helper extraction is complete: DICOM identity lives in
  `preprocessing/dicom_identity.py`, cohort structure shell/reference building
  lives in `preprocessing/structure_referencer.py`, uncertainty data objects
  live with uncertainty attachment, and old geometry holder classes have been
  moved out of `main`,
- the next repository-maintenance pass should organize root planning markdowns
  into a tracked docs area and archive completed private planning notes under a
  private completed folder,
- the next nearby work should validate the current semi-modular main checkpoint,
  then build patient-runner stages for the remaining pre-MC path and begin the
  separate MC patient-module extraction described above.

Validation cadence:

- avoid a full validation run after every tiny helper extraction,
- validate in tranches: current semi-modular main checkpoint, then focused
  patient-submodule checks, then pre-MC patient-runner assembly, then MC as its
  own heavier tranche,
- final-output validation is intentionally indirect; it checks durable output
  contracts and uses manifests/stage artifacts to localize mismatches,
- once a stage has passed focused patient-module validation, avoid adding
  permanent duplicate sidecar logic unless it is needed to diagnose a known
  mismatch.

### Phase F: Contract/Object Cleanup

Gradually replace direct dictionary access with typed runtime wrappers and
explicit manifests. This should happen after the runner is useful enough to
support full-cohort execution and paper work.

### Phase G: Optional Preflight/Sidecar Utilities

Later, implement lightweight utilities for optional per-patient sidecar inputs.
These should share the exact same scientific modules as the main runner and do
only the narrow measurement job needed for the sidecar.

# Documentation Index

Last updated: 2026-08-03

This is the public map for repository documentation. It separates durable project
contracts from private scratch notes and generated audit artifacts.

## Documentation Layers

### README

`README.md` should stay small: license, citation, and a link to this index. It
should not become the planning surface for every refactor.

### Public durable docs

Use tracked Markdown for decisions that affect the codebase contract, validation
strategy, or future developer behavior. These docs should be discoverable from
this index.

Current durable docs:

- `architecture/PATIENT_MODULE_TREE_GUIDE.md` - canonical ownership and
  placement guide for patient-level scientific modules and the
  science-orchestration split.
- `roadmap/PATIENT_RUNNER_UPGRADE_ROADMAP.md` - active migration roadmap toward
  a validated per-patient runner.
- `runtime/RUNTIME_LOGGING_DESIGN.md` - runtime logging, crash localization, and
  failure evidence policy.
- `input/INPUT_DICOM_DATA_ASSESSMENT.md` - current and future input-data
  assumptions.
- `input/INPUT_DATA_MANIFEST_DESIGN.md` - provenance/manifest design for input
  discovery.
- `architecture/CONFIG_LAYER_REWRITE_PLAN.md` - future configuration-layer
  direction.
- `architecture/PATIENT_RUNNER_CONFIG_PATHWAYS.md` - current config-pathway
  inventory and debug-subgroup rewrite map for patient-runner/scientific-shadow
  config work.
- `architecture/PATIENT_RUNNER_DEPENDENCY_GRAPH.md` - dependency graph and
  pathway/checkpoint vocabulary for patient-runner scientific orchestration.
- `architecture/PATIENT_RUNNER_OUTPUT_ARCHITECTURE.md` - output-layer contract
  for patient artifacts, manifests, post-run assembly, and parity.
- `architecture/PATIENT_SCIENTIFIC_CONTEXT_ARTIFACTS.md` - durable patient
  context-artifact strategy for retaining inspectable scientific arrays,
  transform provenance, and post-run GUI/reanalysis context without relying on
  pickles.
- `architecture/PATIENT_RUNNER_PROCESS_ARCHITECTURE.md` - standalone
  parent/worker process architecture for moving the patient runner outside the
  legacy all-patient runtime.
- `architecture/DOSIMETRIC_NN_RENDER_SURFACE.md` - additive dose
  nearest-neighbour render surface for publication figures, debug inspection,
  and future GUI integration without changing MC dose math or patient-runner
  execution semantics.
- `architecture/GUI_AND_STARTUP_ARCHITECTURE_PLAN.md` - GUI/startup boundary
  plan.
- `boundaries/PICKLE_EXPORT_BOUNDARIES.md` - pickle export/load boundary
  contract.
- `roadmap/PATIENT_RUNNER_MODULE_READINESS.md` - stage-by-stage patient-runner
  readiness checklist and tranche map.
- `roadmap/VALIDATION_HARDENING_AND_ARCHITECTURE_AUDIT.md` - Jun23 validation
  hardening direction, split-cohort equivalence plan, typed runtime migration
  explanation, and Markdown documentation audit.

### Module-local design docs

Keep detailed docs beside the module they govern when they are most useful to
someone reading that code.

Current module-local docs:

- `python_files_dcm_meta_based/output_artifacts/OUTPUT_SCHEMA_REGISTRY_GUIDE.md`
- `python_files_dcm_meta_based/PATIENT_RUNNER_COHORT_DERIVED_QUANTITIES.md`
- `python_files_dcm_meta_based/biopsy_optimizer/v2/OPTIMIZER_V2_DESIGN.md`
- `python_files_dcm_meta_based/biopsy_optimizer/v2/OPTIMIZER_V2_PERFORMANCE.md`
- `python_files_dcm_meta_based/guidance_maps/GUIDANCE_MAP_WORKFLOW.md`
- `python_files_dcm_meta_based/input_data/DICOM_INPUT_SHAPE.md`
- `python_files_dcm_meta_based/patient_runner/README.md`
- `python_files_dcm_meta_based/post_run/README.md`
- `python_files_dcm_meta_based/deprecated/README.md`
- `python_files_dcm_meta_based/validation/README.md`
- `python_files_dcm_meta_based/validation/RUN_VALIDATION_CODEBOOK.md`
- `python_files_dcm_meta_based/ui/RENDER_BROKER_DESIGN.md`

### Package extraction docs

Docs for work that may move into a separate repository can stay in the relevant
package/prototype folder until extraction.

Current package extraction docs:

- `custom_PIP/README.md`
- `custom_PIP/STANDALONE_PACKAGE_DESIGN.md`

### Generated and audit docs

Generated outputs and dated audits should remain under their output/audit folder
and outside Git tracking. Durable conclusions from those audits should be
promoted into roadmap or design docs instead of linking tracked docs to generated
files.

### Private notes

`.private_notes/` is ignored by git and is appropriate for rough research
thinking, local audits, and incomplete planning. It should not be the only source
for an implemented decision. When a decision becomes stable, move the durable
part into a tracked doc and link it from this index.

## Current Target Structure

The root README and license remain at repository root. Durable planning and
architecture docs live under `docs/`:

```text
README.md
docs/
  DOCUMENTATION_INDEX.md
  architecture/
    CONFIG_LAYER_REWRITE_PLAN.md
    GUI_AND_STARTUP_ARCHITECTURE_PLAN.md
    PATIENT_MODULE_TREE_GUIDE.md
    PATIENT_RUNNER_CONFIG_PATHWAYS.md
    PATIENT_RUNNER_DEPENDENCY_GRAPH.md
    DOSIMETRIC_NN_RENDER_SURFACE.md
    PATIENT_SCIENTIFIC_CONTEXT_ARTIFACTS.md
    PATIENT_RUNNER_OUTPUT_ARCHITECTURE.md
    PATIENT_RUNNER_PROCESS_ARCHITECTURE.md
  boundaries/
    PICKLE_EXPORT_BOUNDARIES.md
  input/
    INPUT_DICOM_DATA_ASSESSMENT.md
    INPUT_DATA_MANIFEST_DESIGN.md
  roadmap/
    PATIENT_RUNNER_MODULE_READINESS.md
    PATIENT_RUNNER_UPGRADE_ROADMAP.md
    VALIDATION_HARDENING_AND_ARCHITECTURE_AUDIT.md
  runtime/
    RUNTIME_LOGGING_DESIGN.md
```

Module-specific docs should remain beside the modules they govern.

## Cleanup Rules

- Add new durable planning docs only when they define a lasting contract or
  migration stage.
- Prefer updating an existing roadmap/design doc over adding another root-level
  note.
- Keep private notes private when they are speculative, but graduate accepted
  decisions into tracked docs.
- Keep generated audit outputs dated and under `validation_outputs/`; this
  directory is git-ignored, so promote durable conclusions into tracked docs
  instead of committing generated audit files.
- Link new module-specific docs from this index.
- If a doc describes a removed or disabled pathway, state that explicitly near
  the top.

# Patient Module Tree Guide

Last updated: 2026-05-23

This document is the source of truth for where new patient-level modules belong
and which existing module locations are temporary migration debt. Use it before
adding, moving, or naming patient-facing code so the repository does not drift
into parallel scientific trees.

This guide is intentionally about module ownership and placement. It does not
change scientific behavior, validation requirements, or the rule that the
validated cohort/oracle path stays frozen until deliberate comparison passes are
ready.

## Core Rule

Prefer patient-local scientific code inside the existing scientific package
tree. Do not grow a parallel top-level
`python_files_dcm_meta_based/patient_stages/` tree.

If older branches or notes still reference
`python_files_dcm_meta_based/patient_stages/`, treat that as historical
staging only and relocate the code into canonical stage-local homes before
continuing extraction.

Keep three layers distinct:

- Legacy oracle path: current main plus the validated cohort wrappers. This path
  stays runnable and trusted after validation.
- Scientific stage modules: patient-local and cohort-facing scientific code that
  owns geometry, dose, MR, targeting, optimization, and other domain logic.
- Orchestration, assembly, and product surfaces: runner contracts, manifests,
  artifact writing, assembly, validation harnesses, startup, runtime logging,
  UI, and other workflow control code.

## Ownership Map

| Path | Ownership | Put new code here when... | Keep out |
| --- | --- | --- | --- |
| `python_files_dcm_meta_based/biopsy_localization_convex_main.py` | Legacy oracle orchestration | preserving the validated all-patient call graph or adding a tightly scoped approved bug fix | new patient scientific modules, broad cleanup, new runner logic |
| `python_files_dcm_meta_based/preprocessing/` | Pre-MC scientific stages | adding or extracting patient-level science for preprocessing, structure work, biopsy work, uncertainty attachment, dose, or MR stages | batch orchestration, manifests, output assembly, GUI/startup policy |
| `python_files_dcm_meta_based/preprocessing/structure_processing/` | Structure-science family | raw contour pulling, non-biopsy preprocessing, selected-structure logic, prostate-only MR ADC structure summaries | patient-runner stage lists, assembly outputs, cross-run logging policy |
| `python_files_dcm_meta_based/preprocessing/biopsy_processing/` | Biopsy-science family | real/simulated biopsy preparation, planning, finalization, targeting, sampled-biopsy preprocessing, biopsy QA helpers | batch execution backend policy, artifact manifests, UI workflow code |
| `python_files_dcm_meta_based/biopsy_optimizer/` | Optimization science | optimizer wrappers, patient-local optimization adapters, target-ranking science | patient batch orchestration, shadow-output assembly |
| `python_files_dcm_meta_based/guidance_maps/` | Guidance-map science | patient-local guidance-map precompute/planning logic and domain-specific helpers | generic runner contracts, runtime logging, GUI bootstrapping |
| `python_files_dcm_meta_based/output_artifacts/` | Artifact contracts and assembly | dataframe export surfaces, schema contracts, cohort assembly, shadow stitching, output inventory | new scientific geometry, targeting, or MC algorithms |
| `python_files_dcm_meta_based/patient_runner/` | Typed orchestration | patient case contracts, stage sequencing, manifests, batch execution, runner validation hooks | new scientific math, geometry, targeting, optimizer, or MC algorithms |
| `python_files_dcm_meta_based/validation/` and validation helpers | Comparison and regression surfaces | oracle comparisons, rerunnable validation scripts, mismatch localization | canonical scientific implementations |
| `python_files_dcm_meta_based/input_data/` | Input discovery contracts | manifests, routing profiles, DICOM input shape and provenance surfaces | downstream scientific stage logic |
| `python_files_dcm_meta_based/startup/` | Bootstrap and runtime wiring | startup flow, logging, process watchdogs, pickle-load workflow, runtime configuration glue | new scientific stage implementations |
| `python_files_dcm_meta_based/ui/` | UI/product surface | user-facing controls, rendering broker behavior, UI-only view models | scientific stage algorithms |
| Current MC files beside repo root (`MC_prepper_funcs.py`, `MC_simulator_convex.py`, `MC_simulator_MR.py`) | Current MC oracle surface | approved MC bug fixes or deliberate patient-MC extraction beside the existing MC family | patient-runner orchestration or unrelated preprocessing code |

## Preferred Placement Pattern

The repository already shows the preferred pattern in several places:

- `preprocessing/dose_grid_processing.py` keeps a patient entrypoint beside the
  cohort wrapper.
- `preprocessing/mr_adc_grid_processing.py` does the same for MR ADC mapping.
- `preprocessing/structure_selection.py` keeps patient and cohort entrypoints in
  the owning stage module.
- `preprocessing/structure_processing/prostate_only_mr_adc.py` keeps
  patient-local science and comparison helpers in the structure-science family.

Prefer these patterns, in order:

1. Same stage file

Use this when one patient entrypoint and one cohort wrapper clearly belong to
the same algorithm. Make the patient surface obvious with names such as
`build_*_for_patient(...)`, `process_patient_*`, or
`determine_patient_*`.

2. Family-local `per_patient/` subpackage

Use this inside an existing scientific family when several patient-only files
would otherwise clutter the main stage file or when the family needs a clearly
visible patient-only area.

Preferred future examples:

- `preprocessing/structure_processing/per_patient/`
- `preprocessing/biopsy_processing/per_patient/`
- `guidance_maps/per_patient/`
- a future dedicated MC family such as `mc/per_patient/` if MC extraction grows
  beyond a small number of files

3. New top-level scientific package

Use this only when introducing a genuinely new scientific family. Do not create
top-level mirror trees just to hold patient variants of modules that already
have a natural home elsewhere.

## Science vs Orchestration Boundary

This tree policy should support a future clean split between inspectable
scientific surfaces and private workflow/product surfaces.

Scientific packages should own:

- patient-local scientific stage functions,
- cohort wrappers that preserve validated behavior during migration,
- domain data transforms, geometry, targeting, optimization, and analysis math,
- stage-local validation helpers that compare modular output to oracle output.

Orchestration/product packages should own:

- patient runner contracts and stage sequencing,
- artifact writing, manifests, and cohort assembly,
- runtime logging, retries, worker policy, and startup/bootstrap,
- UI and product-specific interaction surfaces.

Boundary packages should stay narrow:

- `output_artifacts/` owns output contracts and assembly, not science.
- `validation/` owns comparisons and audits, not canonical implementations.
- `input_data/` owns discovery/manifests, not downstream scientific stages.

## Current Canonical Patient-Module Homes

Current repository placement after the 2026-05-23 cleanup pass:

| Stage | Canonical home |
| --- | --- |
| Raw contour pulling | `python_files_dcm_meta_based/preprocessing/structure_processing/per_patient/raw_contour_pulling.py` |
| Real biopsy preprocessing | `python_files_dcm_meta_based/preprocessing/biopsy_processing/per_patient/real_biopsy_processing.py` |
| Simulated biopsy target assignment | `python_files_dcm_meta_based/preprocessing/biopsy_processing/per_patient/simulated_biopsy_preparation.py` |
| Simulated biopsy planning | `python_files_dcm_meta_based/preprocessing/biopsy_processing/per_patient/simulated_biopsy_planning.py` |
| Realized biopsy targeting | `python_files_dcm_meta_based/preprocessing/biopsy_processing/per_patient/realized_biopsy_targeting.py` |

Do not recreate these same entrypoints under a top-level
`python_files_dcm_meta_based/patient_stages/` tree.

## Naming and Review Rules

- Make patient entrypoints obvious in the function name.
- Prefer stage-local ownership over convenience imports.
- Do not move scientific code into `patient_runner/` just because the runner
  calls it.
- Do not create a second implementation tree with the same scientific purpose.
- If a stage already has a clean patient-local home, keep it there and have the
  runner import it from that canonical location.
- If a stage does not yet have a clean patient-local home, create that home in
  the owning scientific family instead of defaulting to a new top-level package.

## Anti-Drift Checklist

Before adding or moving a patient-facing module, check:

1. Is this scientific logic, or is it orchestration/assembly/logging/UI?
2. Which existing scientific family already owns this stage?
3. Can the patient entrypoint live in the same stage file with an explicit
   patient name?
4. If not, should a family-local `per_patient/` subpackage hold it?
5. Are we accidentally adding more code under a parallel top-level patient tree
  or under `patient_runner/` that belongs elsewhere?
6. Does the readiness checklist need to point at the canonical home after the
   move?

If the answer would create a parallel tree, stop and update the plan before
editing code.
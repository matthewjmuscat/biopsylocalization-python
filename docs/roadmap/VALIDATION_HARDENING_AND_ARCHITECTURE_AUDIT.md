# Validation Hardening And Architecture Audit

Last updated: 2026-06-23

## Purpose

This note records the near-term direction after the Jun21/Jun22 patient-runner
parity work. It is intentionally a planning and documentation artifact, not a
scientific-code change request.

Current rule: do not edit scientific functions or raw kernel code while the next
validation layer is being hardened. If a later migration really needs a new
scientific implementation, add it beside the old implementation as a versioned
patient-runner path, validate it against the legacy oracle, and leave the legacy
path intact until validation evidence is strong.

## Accepted Direction

The migration should proceed in this order:

1. Harden validation and artifact expectations.
2. Use split-cohort and subset equivalence tests to prove scale-out behavior.
3. Treat the per-patient runner as the primary development path.
4. Keep the legacy pathway as the scientific oracle during cleanup.
5. Move config, output, documentation, and runtime-state cleanup through small
   validation-gated slices.
6. Defer scientific-function rewrites unless they are explicitly required.

This keeps cleanup subordinate to scientific reproducibility.

## Split-Cohort Equivalence

The intended scale-out strategy is to run cohort halves separately, assemble the
patient artifacts, and validate that each half is internally consistent against
the legacy oracle. If the pipeline is genuinely patient-local, the union of two
validated halves should be scientifically equivalent to a single full-cohort
patient-runner execution.

Known retired cohort-derived pathways support this assumption:

- Global-mean biopsy centroid variation was removed from runtime uncertainty
  generation.
- Simulated-biopsy length modes based on all-patient real-biopsy length mean or
  all-patient normal sampling were removed.
- Current `match real` simulated-biopsy length fallback is patient-compatible:
  matched real biopsy, then same-patient/same-DIL mean, then configured full
  needle compartment length.

Remaining risks to make explicit:

- Transform generation currently reports a cohort-stream seed. If per-patient
  transform banks depend on iteration order, split-vs-single byte identity is
  not guaranteed until transform seeds are patient-derived or recorded/replayed
  per patient.
- Output assembly must use stable table IDs, canonical keys, deterministic row
  ordering, and consistent CSV/index policy.
- Multi-run assembly must fail closed unless manifests prove compatibility.

Recommended proof before relying on full split-cohort operation should use a
decompose-and-reconstruct comparison rather than raw folder diffing. The point
is to reduce each run to registry-known patient artifacts, reconstruct the
cohort surface through one deterministic policy, and compare those reconstructed
surfaces. That prevents validation from being fooled by harmless differences in
filesystem layout, run grouping, output folder names, or multi-run artifact
ordering.

The reusable validation function should do this:

1. Select a small cohort subset that can run as one legacy/cohort run.
2. Run it once as a single cohort.
3. Run the same subset as two or more split runs.
4. Decompose each run into registry-known patient artifacts and manifest
  provenance.
5. Reconstruct each cohort surface through the same assembly policy.
6. Compare reconstructed surfaces by table ID, row grain, canonical keys,
  numeric values, text values, row-set hashes, ordered hashes, and manifest
  seed provenance.
7. Treat any difference as a validation finding unless it is explained by a
   deliberate run-profile setting.

For long-term reproducibility, transform-generation RNG should move toward the
same policy already used for optimizer-v1: stable patient-derived streams with
resolved patient seeds recorded in manifests.

## Validation Hardening Plan

The current schema registry and post-run assembly are strong, but validation
still treats deliberate absences too much like failures. The next layer should
be an expected-artifact policy generated from the registry plus the resolved run
profile.

The expected-artifact checker should classify every durable artifact as one of:

- `required`: must exist for the selected run profile.
- `optional`: may exist; absence is a warning or informational note.
- `validation_only`: expected only when its sidecar/config is enabled.
- `disabled_by_config`: deliberately absent for this run.
- `not_applicable`: upstream stage/pathway was not selected.
- `failed_patient`: absent because an input patient failed or was excluded.
- `deprecated`: retained only for old compatibility or audit history.
- `downstream_calculable`: not a core runtime product.

Comparator and assembly summaries should then report:

- missing required artifacts as failures,
- disabled validation sidecars as passing but explicitly disabled,
- not-applicable artifacts as passing when the pathway really skipped the
  producer,
- unexpected extra artifacts as warnings,
- numeric/text/schema drift in compared required artifacts as failures.

This would prevent standard runs from failing only because a validation-only
sidecar, such as planned-vs-realized centroid variation, was intentionally off.

Useful implementation homes:

- `python_files_dcm_meta_based/output_artifacts/schema_registry.py` remains the
  table contract source of truth.
- A new `output_artifacts/expected_artifacts.py` can map registry specs plus run
  profiles to expected artifacts.
- `post_run/cohort_assembly` can include expected-artifact summaries in assembly
  reports.
- `patient_runner/parity.py` can consume those summaries so overall parity does
  not fail on disabled optional sidecars.
- `validation/` comparators can keep their numeric/text comparison role and
  delegate missing-file interpretation to the expected-artifact layer.

## Typed Runtime Migration

Typed runtime migration does not mean converting the whole legacy master
dictionary into dataclasses in one pass. It means gradually moving repeated
runtime access patterns behind small typed boundaries while preserving the raw
legacy dictionaries as compatibility storage during validation.

Good first slices:

- `PatientCase`: patient identity and provenance.
- `PatientRuntimeState`: one-patient state boundary around legacy patient data.
- `StructureRecord`: stable access to patient + structure type + structure
  index identity and key fields.
- `BiopsyRecord`: stable access to patient + biopsy index identity and biopsy
  artifact stores.
- `PatientArtifactStore`: typed read/write surface for patient dataframes.
- `RunArtifactManifest`: run-level index of produced artifacts and their
  registry specs.

Rules for this migration:

- Start at boundaries that are already adapter-like: artifact writing,
  manifests, validation, post-run assembly, and patient-runner contracts.
- Do not rewrite scientific math just to introduce types.
- Keep `from_legacy_dict(...)` and `to_legacy_dict(...)` adapters while the
  legacy oracle is still used.
- Use typed wrappers to reduce duplicated string-key access, not to hide
  scientific behavior.
- After each slice, validate the same outputs against the legacy oracle.

## Scientific Function V2 Strategy

If a scientific-level function later needs real modernization, use a side-by-side
versioned path:

1. Keep the legacy function unchanged.
2. Implement `function_v2` or a clearly named new module in the owning
   scientific package.
3. Route only the per-patient runner through the new function.
4. Compare the per-patient runner output against the legacy oracle.
5. Promote the new function only after table-level and, when needed,
   patient-level parity evidence is recorded.

This is preferred over editing the monolith or replacing a scientific function in
place. It preserves the oracle and makes each migration reviewable.

## Current Documentation Audit

Tracked Markdown inventory as of this audit has 29 files. The documentation
index is useful and should remain the public map, but it is behind the current
repository state.

Docs that are current and should remain durable:

- `docs/roadmap/PATIENT_RUNNER_UPGRADE_ROADMAP.md`
- `docs/roadmap/PATIENT_RUNNER_MODULE_READINESS.md`
- `docs/architecture/PATIENT_MODULE_TREE_GUIDE.md`
- `docs/architecture/PATIENT_RUNNER_DEPENDENCY_GRAPH.md`
- `docs/architecture/PATIENT_RUNNER_OUTPUT_ARCHITECTURE.md`
- `docs/architecture/PATIENT_RUNNER_CONFIG_PATHWAYS.md`
- `docs/architecture/CONFIG_LAYER_REWRITE_PLAN.md`
- `python_files_dcm_meta_based/output_artifacts/OUTPUT_SCHEMA_REGISTRY_GUIDE.md`
- `python_files_dcm_meta_based/PATIENT_RUNNER_COHORT_DERIVED_QUANTITIES.md`
- `python_files_dcm_meta_based/validation/README.md`
- `python_files_dcm_meta_based/post_run/README.md`

Docs that should be linked or re-linked from the documentation index:

- `docs/architecture/PATIENT_RUNNER_DEPENDENCY_GRAPH.md`
- `docs/architecture/PATIENT_RUNNER_OUTPUT_ARCHITECTURE.md`
- `docs/roadmap/PATIENT_RUNNER_MODULE_READINESS.md`
- `python_files_dcm_meta_based/post_run/README.md`
- `python_files_dcm_meta_based/validation/README.md`
- `python_files_dcm_meta_based/preprocessing/MR_ADC_TODO.md` or a replacement
  if the TODO is obsolete.

Docs needing freshness review rather than immediate deletion:

- `python_files_dcm_meta_based/preprocessing/MR_ADC_TODO.md`: likely a local TODO
  that should either be resolved, converted to a roadmap item, or archived.
- `custom_PIP/STANDALONE_PACKAGE_DESIGN.md`: keep if package extraction remains
  relevant; otherwise archive as historical design.
- `python_files_dcm_meta_based/biopsy_optimizer/v2/OPTIMIZER_V2_PERFORMANCE.md`:
  keep if it reflects current optimizer-v2 performance policy; otherwise add a
  status note.
- `python_files_dcm_meta_based/ui/RENDER_BROKER_DESIGN.md`: keep as GUI/render
  direction, but review after patient-runner validation stabilizes.

Private ignored Markdown notes are already partly organized under
`.private_notes/completed/`. Large active private notes that appear superseded by
tracked docs should be mined for any missing durable facts, then moved to
`.private_notes/completed/` with no further tracked references. In particular,
the public docs now cover much of the durable content from older patient-scoped
pipeline, output dataframe, and simulated-biopsy refactor notes.

Recommended private-note rule:

- Active scratch planning lives in `.private_notes/`.
- Used private notes move to `.private_notes/completed/`.
- Durable decisions get promoted into `docs/` or module-local Markdown and
  linked from `docs/DOCUMENTATION_INDEX.md`.
- Generated audit outputs stay under ignored `validation_outputs/`.

## Immediate Next Work Packets

1. Add expected-artifact policy and status classification.
2. Update post-run assembly and parity summaries to consume expected-artifact
   status.
3. Design and validate patient-derived transform RNG or per-patient transform
   seed replay.
4. Add the generalized decompose-and-reconstruct validation function for one
  full run versus two-or-more split runs.
5. Run split-vs-single equivalence on a small cohort subset.
6. Continue pipeline config migration: move resolved runtime configuration
  toward TOML/JSON-backed provenance, make resolved config manifests explicit,
  and retire duplicated loose config locals only after validation gates pass.
7. Update documentation index and archive stale private notes after their useful
   content is promoted.

Recommended commit messages for these future slices:

- `feat(validation): classify expected artifacts by run profile`
- `feat(validation): report disabled sidecars separately from missing artifacts`
- `docs(roadmap): record split-cohort validation hardening plan`
- `refactor(rng): make transform generation patient-derived for split-run parity`
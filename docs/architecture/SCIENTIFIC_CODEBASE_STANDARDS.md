# Scientific Codebase Engineering Standards

Last updated: 2026-09-09

## Purpose

This repository backs published biopsy-localization research and is intended to
become a reusable open scientific engine. Code must therefore be scientifically
correct, inspectable, auditable, and understandable by contributors who did not
participate in its original development.

The goal is not maximal commentary or abstraction. The goal is that a reader can
identify ownership, inputs, outputs, assumptions, side effects, provenance, and
validation evidence without reverse-engineering a monolithic call chain.

## Architectural Direction

The open-source repository owns:

- scientific algorithms and typed scientific contracts;
- headless patient/run orchestration APIs;
- input manifests and resolved configuration;
- versioned artifacts, schemas, and provenance;
- validation, comparison, and post-run reconstruction services;
- renderer-neutral scientific scene contracts.

A future Studio Superposition application may own professional desktop workflow,
packaging, deployment, monitoring, review, collaboration, and reporting. It must
consume the open engine through stable APIs, CLI/service entrypoints, run
profiles, manifests, and artifacts. Scientific modules must not depend on the
commercial application or a particular desktop toolkit.

## Documentation Inside Code

### Module Documentation

Every new boundary module should begin with a concise module docstring stating:

1. what the module owns;
2. what it deliberately does not own;
3. its relationship to legacy compatibility code, when relevant.

A module docstring should orient a reader, not repeat its filename.

### Public Contracts And Services

Public dataclasses, protocols, readers, writers, stage entrypoints, and service
functions require docstrings that describe:

- semantic inputs and outputs;
- mutations and durable side effects;
- units, coordinate frames, array dimensions, or row grain where relevant;
- failure behavior and validation assumptions;
- whether the API is production, transitional, legacy-oracle, or post-run only.

For complex array or table contracts, dimension names and identity keys belong in
typed artifact specs as well as prose. Prose alone is not a schema.

### Comments

Use comments for scientific intent, invariants, non-obvious ordering, and
compatibility constraints. Do not narrate obvious assignments line by line.
Prefer extracting a named helper over adding a large banner around an opaque
block.

Temporary compatibility behavior should say what must become true before it can
be removed.

## Ownership And Dependency Rules

- Scientific modules own computation, not GUI workflow or output-folder policy.
- Startup/orchestration modules own sequencing, not scientific formulas.
- Config modules contain pure data only: no pools, GUI handles, file handles,
  mutable scientific state, or imported CUDA execution.
- Patient workers own one patient's runtime state and write durable artifacts.
- Parent processes own lightweight plans, statuses, paths, and retry policy.
- Cohort assembly and validation consume completed artifacts after execution.
- Renderer backends consume renderer-neutral scene/context contracts.
- Legacy dictionaries remain compatibility storage until typed replacements have
  parity evidence; they are not the target public API.

Package facades used by planning, manifests, or post-run inspection must not
initialize CUDA or import heavy scientific execution merely to expose contracts.

## Configuration And Run Profiles

- Frozen typed Python dataclasses are the runtime scientific authority.
- TOML is the preferred human-authored orchestration/profile format.
- JSON is generated resolved provenance and manifest output.
- Run profiles choose jobs, patients, paths, resources, and evidence policy.
- Scientific parameters compile into `PipelineConfig`; they must not be copied
  into an unrelated orchestration schema.
- Config snapshots are canonical, schema-versioned, and fingerprinted.
- Unsupported runtime objects in config snapshots fail closed.

## Artifact And Merge Policy

For newly produced artifacts, cross-run combination defaults to
`strict_exact_v1`. The following dimensions must match exactly:

- scientific `PipelineConfig` fingerprint;
- effective source-tree fingerprint, including uncommitted tracked and
  non-ignored untracked source changes;
- input/bootstrap policy fingerprint;
- Python/platform/dependency environment fingerprint;
- output schema registry version.

Patient selection and the full case-manifest file hash are not merge dimensions:
split runs intentionally contain different patients. Duplicate patient UIDs must
still fail, and future patient-input identities should verify that repeated
patient sources are identical.

Development runs derive source identity from Git. Packaged applications without
a `.git` directory must inject an immutable build/release identity through
`BIOPSYLOCALIZATION_SOURCE_IDENTITY`; missing source identity fails closed.
`PatientInputPaths.manifest_identity_sha256` currently identifies DICOM role/path
assignment only. It does not hash patient DICOM bytes. Any future content-level
input identity must be designed explicitly with privacy, cost, and DICOM
de-identification requirements in mind.

Historical runs without strict identities may be compared only through an
explicit `legacy_allow_missing` mode. Identified and unidentified runs must never
be mixed. Compatibility relaxation must later be based on named scientifically
relevant fields, not ad hoc exceptions.

## Validation Ladder

Testing depth should match risk:

1. **Contract/unit tests**: normalization, serialization, schemas, fingerprints,
   dependency checks, and failure policy using synthetic data.
2. **Synthetic parity tests**: typed adapters versus existing direct/legacy
   helper behavior on small deterministic structures.
3. **Boundary integration tests**: worker subprocess, manifests, artifact
   discovery, post-run assembly, and fail-closed compatibility.
4. **User-operated one-patient validation**: real data only when explicitly run
   by the user, comparing standalone and isolated from-legacy surfaces.
5. **Subset/cohort parity**: assembled outputs versus the frozen legacy oracle.
6. **Split-versus-single reconstruction**: identical scientific identity and
   patient union, reconstructed through the same assembly policy.

Raw patient data must not be inspected or executed by automated coding sessions
unless explicitly authorized. Synthetic tests should cover routine development.

## Legacy Sidecar Retirement

A legacy sidecar may be retired only when:

- its replacement starts from independently built patient state;
- the replacement writes equal or stronger typed evidence;
- focused patient-level parity passes;
- post-run artifact/cohort parity covers its scientific output;
- retained documentation identifies the replacement gate;
- the legacy oracle remains available wherever broader migration validation
  still depends on it.

Sidecars should be removed individually, not in a broad cleanup pass.

## Review Checklist

For each new or changed boundary, reviewers should be able to answer:

- Who owns this behavior?
- Is the API patient-, run-, cohort-, validation-, or presentation-scoped?
- What typed contract crosses the boundary?
- What scientific assumptions, units, frames, shapes, or row grains apply?
- What mutable state or files are written?
- What manifest/provenance records the result?
- What compatibility identity governs reuse or merging?
- What executable test would catch a regression?
- Is legacy behavior preserved or explicitly versioned?
- Can a headless caller use it without importing GUI or unnecessary GPU code?

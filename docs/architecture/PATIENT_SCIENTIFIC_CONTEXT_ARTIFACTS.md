# Patient Scientific Context Artifacts

Last updated: 2026-08-03

## Purpose

This note defines the durable context-artifact direction for the standalone
per-patient runner. The goal is to preserve enough patient-level scientific
context after a run to support post-run inspection, GUI rendering, downstream
analysis, validation, and future method development without relying on legacy
pickles or rerunning the full pipeline.

The immediate forcing example is the dosimetric nearest-neighbour render GUI,
which needs more context than the normal pointwise/voxelwise dose tables retain.
The same design applies more broadly to biopsy transforms, non-biopsy structure
transforms, dose/MR lattices, optimizer scenes, tissue containment, and future
debug or publication figure tools.

## Core Recommendation

Do not choose between transformation-specific provenance and
transformation-agnostic resolved outputs. Retain both, but assign them different
roles.

```text
transformation-specific provenance
  explains how a configuration was generated and audited

transformation-agnostic resolved artifacts
  record the concrete arrays and coordinates consumed by later tools
```

The resolved artifacts are the stable post-run API for GUI and downstream tools.
The provenance artifacts are the audit trail that lets a future reader or
validator understand which transformation model produced those resolved facts.

This avoids two bad extremes:

- retaining only high-level config, which can become unreplayable when transform
  models or code change;
- retaining only resolved arrays, which supports rendering but loses the
  scientific explanation and validation lineage.

## Relationship To Pickles

Pickles remain a transitional compatibility mechanism only. They are useful for
legacy replay because they can preserve rich Python object graphs, but they
should not be the durable boundary for the standalone patient runner.

The target boundary is manifest-backed patient artifacts:

- JSON for manifests, provenance summaries, schema versions, and path indexes;
- Parquet for tabular/indexed products and dataframe-like outputs;
- chunked array stores for large numeric arrays and tensors;
- compact selected render-scene artifacts for publication/debug surfaces.

The artifact reader should reconstruct useful analysis contexts from these
contracts, not from private in-memory dictionaries or pickled object graphs.

## Recommended Storage Model

Use the storage format that matches the data shape.

### JSON Manifests

Use JSON for small, inspectable metadata:

- patient UID, run ID, pathway, stage graph, and artifact entries;
- coordinate-frame registry and frame names;
- schema versions and compatibility versions;
- code identity and dirty-state metadata;
- input manifest references and source DICOM role paths;
- artifact checksums, shapes, dtypes, units, and row counts;
- transform model family, algorithm version, and config fingerprint.

JSON should not carry large numeric arrays.

### Parquet Tables

Use Parquet for row-oriented scientific products:

- pointwise dose by trial;
- voxelwise dose by trial;
- DVH curves;
- summary statistics;
- artifact indexes;
- event logs where rows are naturally independent records.

Parquet is much better than CSV for retained outputs, but it is not the right
primary representation for large repeated geometry tensors. Do not store giant
nested nearest-neighbour coordinate lists in Parquet when lattice indices and
array artifacts can represent the same information more compactly.

### Chunked Array Stores

Use chunked numeric array artifacts for large N-dimensional scientific context.
Zarr is the preferred long-term candidate because it is Python-native, chunked,
directory-friendly, and compatible with manifest-backed artifact layouts. HDF5
is acceptable where a single-file container is preferable. NPZ is acceptable for
small selected scene artifacts or early prototypes, but it is not the long-term
large-array store.

Examples:

- dose or MR lattice scalar arrays;
- physical coordinate arrays when the grid is irregular;
- trial-by-point biopsy query coordinates when they cannot be regenerated;
- per-trial transform matrices;
- nearest-neighbour index and distance tensors;
- sampled points or voxelized structure masks;
- selected render-scene subsets.

Preserve source precision by default. Downcast or quantize only behind an
explicit retention policy with validation evidence.

## Artifact Families

The target patient output should evolve toward this shape:

```text
patients/<patient_uid>/
  patient_artifact_manifest.json
  context/
    coordinate_frames.json
    transform_events.parquet
    dose_lattice.zarr/
    mr_lattice.zarr/
    biopsy_geometry.zarr/
    non_biopsy_geometry.zarr/
    dose_nn_context.zarr/        optional by retention policy
  tables/
    pointwise_dose.parquet
    voxelwise_dose.parquet
    dvh.parquet
  render_scenes/
    dose_nn_<biopsy>_<selection>.json
    dose_nn_<biopsy>_<selection>.npz or .zarr
```

The exact physical paths can change, but the manifest entries and artifact IDs
should be stable.

## Transformation Artifacts

Transformation models are likely to change for both biopsy and non-biopsy
objects, so artifact design should not assume one permanent transform family.

Each transformation-producing stage should emit two layers.

### Transformation-Specific Provenance

This layer is model-aware. It records:

- transform family and algorithm version;
- source and target coordinate frames;
- random seed or sampled random-variable provenance;
- parameter names, units, covariance/correlation assumptions, and constraints;
- order of operations;
- source structures and input artifact references;
- code/config/schema fingerprints;
- validation status for the transform producer.

This layer is for audit and specialized replay. It is allowed to evolve by
schema version as transform models become more sophisticated.

### Transformation-Agnostic Resolved Outputs

This layer records what downstream stages actually consumed:

- resolved 4x4 transform matrices or equivalent maps per trial/object;
- transformed centroids or sampled points in canonical physical space;
- resolved query points used for dose/MR/tissue localization;
- trial identifiers and object identifiers;
- links to the source coordinate frames and provenance event IDs.

This is the safer API for GUI and downstream analysis because it does not
require the caller to understand every historical transform model. A future GUI
can render or compare old and new runs from resolved outputs while still showing
the model-specific provenance on demand.

## Dose Nearest-Neighbour Context

For the dose render GUI, the lightweight complete context should not be a raw
long dataframe of repeated nearest-neighbour coordinates. Prefer compact arrays:

```text
dose_lattice:
  regular grid metadata or physical_coordinates[n_lattice, 3]
  dose_values[n_lattice]
  gradient_values[n_lattice] or gradient_vectors[n_lattice, 3]

biopsy_query_context:
  trial_numbers[n_trials]
  original_point_indices[n_points]
  query_points[n_trials, n_points, 3]
  voxel_index[n_points] when available

dose_nn_context:
  nearest_lattice_indices[n_trials, n_points, k]
  nearest_distances[n_trials, n_points, k]
  interpolated_dose[n_trials, n_points]
```

Nearest-lattice coordinates can then be reconstructed by indexing into the dose
lattice. This keeps the artifact complete for rendering and auditing without
storing repeated coordinate triples for every query point and neighbour.

The full `dose_nn_context` should be retention-policy controlled. For routine
scientific runs, it may be enough to retain dose lattice, query geometry, and
derived dose tables, then recompute selected NN rows on demand. For publication
figures or debugging, retain selected compact render scenes or selected NN
context arrays.

## Retention Policies

Retention should be explicit in the run profile and manifest. Suggested levels:

```text
minimal
  final/derived tables plus enough manifest metadata for validation

context
  canonical lattices, resolved transforms, query geometry, and derived tables

diagnostic
  selected NN tensors, selected debug surfaces, and render-scene artifacts

full_debug
  complete high-volume tensors for selected patients only
```

The default standalone-run target should be `context`, not `minimal`, once the
artifact surface is validated. `full_debug` should require explicit patient and
stage selection because it can grow quickly.

## Validation Requirements

Context artifacts are scientific outputs and need validation, not just file IO.

Required checks:

- writer/reader round trips preserve shapes, dtypes, units, and coordinate-frame
  references;
- manifest entries point to existing artifacts and record checksums or
  fingerprints;
- resolved query points reproduce legacy-localizer results for sampled cases;
- derived Parquet tables can be regenerated from context artifacts where that is
  part of the contract;
- transform provenance event IDs link to resolved arrays that used them;
- multi-run assembly fails closed when context-artifact schema or scientific
  compatibility differs;
- validators compare selected standalone context artifacts against the legacy
  oracle during migration.

## GUI And Post-Run Tools

Post-run GUI tools should read manifests and context artifacts. They should not
depend on `biopsy_localization_convex_main.py`, legacy mutable dictionaries, or
pickled runtime objects.

The GUI should be able to:

- list patients, biopsies, trials, structures, and available context artifacts;
- request an on-demand derived scene from retained context;
- choose a renderer backend;
- export figures and provenance summaries;
- fail clearly when a requested scene requires a context level that was not
  retained.

Inline runtime code should only write artifacts. It should not open interactive
render windows as part of the numerical loop.

## Migration Strategy

1. Keep legacy CSV/parquet outputs and validation surfaces stable.
2. Add manifest entries for new context artifacts beside existing outputs.
3. Start with selected dose-NN scene artifacts because they solve an immediate
   figure need and test the boundary.
4. Add dose lattice and biopsy query context artifacts for selected patients.
5. Add transform event provenance and resolved transform arrays.
6. Rebuild selected derived tables from context artifacts and validate against
   legacy outputs.
7. Expand retention-policy controls into the run profile once validation is
   credible.

This sequence lets the codebase move toward durable, inspectable scientific
context without destabilizing the current legacy oracle or forcing all runs to
retain full debug tensors.

# Runtime Artifact Upgrade Register

Last updated: 2026-08-03

## Purpose

This register tracks code and artifact surfaces that look inefficient, brittle,
or hard to audit while working on the patient-runner and dose-render migration.
It is not approval to edit scientific code. It is a place to record the issue,
the likely improvement path, and the rough expected benefit so future passes can
be chosen deliberately and validated appropriately.

Impact scale: `small`, `medium`, `large`, `massive`.

Validation burden scale: `small`, `medium`, `large`, `massive`.

## Open Upgrade Opportunities

### Raw MC Dosimetry Nearest-Neighbour Dataframe Dumps

- Status: open.
- Anchor: `biopsy_localization_convex_main.py` currently warns that
  `raw_data_mc_dosimetry_dump_bool` may write hundreds of gigabytes. The legacy
  default and `MCOutputDumpConfig` default are both `False`, and the current
  patient convex MC stage rejects raw dump side effects.
- Problem: this is not an active default-output problem. The remaining issue is
  that the old escape hatch is too heavy and too row-oriented to be the right
  answer when a future run genuinely needs deeper retained context.
- Recommendation: retire or quarantine the raw dataframe dump path once a
  replacement context artifact exists. The replacement should preserve
  equivalent or greater information in manifest-backed arrays: dose lattice
  once, query geometry, nearest-lattice indices, nearest distances,
  interpolated dose tensors, and stage provenance. Human-readable dataframes
  should be reconstructed on demand for selected views.
- Expected storage/performance benefit: massive.
- Expected cleanliness benefit: large.
- Validation burden: large, because reconstructed views must match legacy dose
  outputs and selected nearest-neighbour rows.
- Touch policy: do not edit MC dose math or localizer behavior as part of the
  renderer backend. Additive context writers should be validated first; removal
  or hard deprecation of the old raw dump flag should happen only after the
  replacement artifact contract is in place.

### Master Structure Reference Dictionary As Primary Runtime Carrier

- Status: open.
- Anchor: `master_structure_reference_dict` and related structure-reference
  dictionaries flow through legacy main, preprocessing, optimizer, MC, output,
  and patient-runner adapter surfaces.
- Problem: one mutable nested dictionary carries mixed concerns: source DICOM
  identity, structure geometry, runtime state, derived scientific outputs,
  display support, and export payloads. That makes memory use, serialization,
  schema documentation, and per-patient isolation hard to reason about.
- Recommendation: migrate toward typed patient-scoped runtime/context contracts
  plus manifest-backed artifact stores. Keep legacy adapters at boundaries until
  validation proves equivalence. Split identity/provenance, geometry arrays,
  derived tables, and presentation-only fields into separate documented artifact
  families.
- Expected storage/performance benefit: medium to large.
- Expected cleanliness benefit: massive.
- Validation burden: massive, because this is a central legacy runtime carrier.
- Touch policy: identify and document during render work; do not refactor this
  dictionary while validating the dose GUI path.

### Pickle Bundles As Durable Scientific Output

- Status: open.
- Anchor: preprocessing and results pickle export/load paths used for legacy
  replay and transitional reconstruction.
- Problem: pickles preserve useful object graphs but are opaque, Python-version
  sensitive, hard to audit, and poor as a long-term scientific artifact format.
- Recommendation: keep pickles as transitional replay tools only. Move durable
  output toward JSON manifests, Parquet tables where naturally tabular, Zarr
  arrays for large tensors, and generated data dictionaries.
- Expected storage/performance benefit: medium.
- Expected cleanliness benefit: large.
- Validation burden: medium to large.
- Touch policy: do not remove pickle paths until manifest-backed artifacts can
  reconstruct the needed post-run contexts.

### Dataframe Outputs For Heavy Repeated Geometry

- Status: open.
- Anchor: dose, containment, and cohort outputs that store or derive large
  repeated row-oriented views from naturally tensor-like state.
- Problem: dataframes are excellent analysis views, but they can be expensive as
  primary storage for repeated trial-by-point-by-neighbour geometry.
- Recommendation: store primary runtime state in compact machine-readable
  artifacts, then materialize dataframes post-run through documented
  constructors for selected patient/biopsy/trial/table views.
- Expected storage/performance benefit: large to massive, depending on table.
- Expected cleanliness benefit: large.
- Validation burden: medium to large.
- Touch policy: constructor outputs should be compared against current dataframe
  outputs before replacing any production artifact surface.

### Rendering Or GUI Logic Embedded In Scientific Runtime

- Status: in progress.
- Anchor: the old dose nearest-neighbour visualization prototype inside legacy
  MC/localization execution.
- Problem: interactive rendering inside scientific runtime complicates testing,
  rerendering, provenance, and dependency boundaries.
- Recommendation: keep scientific runtime responsible for artifacts only. Render
  saved scenes/context post-run through renderer-neutral contracts and backend
  adapters such as PyVista.
- Expected performance benefit: small to medium.
- Expected cleanliness benefit: large.
- Validation burden: small to medium for additive renderer paths; larger only if
  inline capture hooks are introduced.
- Touch policy: current renderer work stays additive and post-run.
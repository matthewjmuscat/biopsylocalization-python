# Biopsy Pipeline Dependency Graph

## Purpose

This document is a current-state dependency map of the biopsy pipeline in `python_files_dcm_meta_based/biopsy_localization_convex_main.py` and its immediate helper modules.

It is written to support safe, additive work on a new target-lesion-specific lane without accidentally breaking the legacy optimizer path.

This is a map of how the code works today, not a proposal for how it should ideally work after a full refactor.

## Scope

Inspected surfaces for this pass:

- `python_files_dcm_meta_based/biopsy_localization_convex_main.py`
- `python_files_dcm_meta_based/biopsy_optimizer.py`
- `python_files_dcm_meta_based/biopsy_transporter.py`
- `python_files_dcm_meta_based/MC_prepper_funcs.py`
- `python_files_dcm_meta_based/cupy_functions.py`
- `python_files_dcm_meta_based/uncertainty_file_writer.py`

Primary focus:

- biopsy shell creation
- legacy DIL optimizer outputs
- real biopsy preprocessing
- simulated-length determination
- simulated biopsy construction
- late biopsy metadata assignment
- biopsy uncertainty aggregation
- sampled-biopsy point generation
- biopsy coordinate-system generation
- transform generation and immediate consumers

Explicitly deferred for now:

- supporting multiple length-selection methods per simulated-biopsy type
- broad cleanup of downstream reporting modules
- large-scale re-architecture beyond what is needed to keep the next lane safe

## Short Version

The current biopsy pipeline is not one block. It is a staged contract that spans early shell creation, late geometry materialization, late metadata enrichment, still-later sampled-point generation, then uncertainty transfer, then transform generation.

The key truths that matter for the next lane are:

1. Simulated biopsy shells already carry `Relative structure type` and `Relative structure ref #` very early, during `structure_referencer(...)`.
2. The `match real` simulated-length logic does not depend on the later `Target DIL by centroid dict` block. It builds its own real-lengths-by-nearest-DIL map from real biopsy centroids and DIL centroids.
3. Optimal simulated-biopsy placement has a real hard dependency on the legacy optimizer because `biopsy_transporter.biopsy_transporter_optimal(...)` reads each DIL's `Biopsy optimization: Optimal biopsy location dataframe`.
4. Simulated biopsies intentionally recreate nearly the same geometry contract as real biopsies, including variation fields, because later code still reads those fields.
5. The late `Target DIL by centroid dict` and related metadata are important for reporting and grouping, but they are not the thing that currently drives simulated-length matching or transform generation.
6. Transform generation depends on stored `Uncertainty data`, and optional needle-compartment shifts also depend on `Reconstructed biopsy cylinder length (from contour data)`.
7. Any attempt to move transform sampling earlier must isolate the CuPy RNG stream from the legacy optimizer, because both use global `cp.random` draws today.

## Current Stage Order

| Stage | Current location | Main writes / effect | Must stay before next stage? | Notes |
| --- | --- | --- | --- | --- |
| 0 | `biopsy_localization_convex_main.py:10725+` | creates structure shells via `structure_referencer(...)` | yes | simulated biopsy shells already get relative-structure identity here |
| 1 | `biopsy_localization_convex_main.py:4371-5068` | legacy DIL optimizer writes optimal-location dataframes onto each DIL | yes for optimal simulated cores only | centroid-based simulated placement does not need this block |
| 2 | `biopsy_localization_convex_main.py:5096-5443` | real biopsy preprocessing writes geometry, line-fit, length, variation, volume fields | yes | this materializes the real-biopsy contract that later code reuses |
| 3 | `biopsy_localization_convex_main.py:5446-5535` | computes real-length aggregates and per-DIL real-length bins | yes | `match real` is a scalar prepass, but it still depends on stage 2 outputs |
| 4 | `biopsy_localization_convex_main.py:5536-5889` | simulated biopsy construction writes almost the same contract as stage 2 | yes | optimal simulated placement depends on stage 1 and stage 3, and on stage 1 optimizer output for optimal mode |
| 5 | `biopsy_localization_convex_main.py:5900-6055` | writes biopsy-in-prostate and target-DIL metadata | no for core simulation path | important late metadata, mostly for classification / analysis |
| 6 | `biopsy_localization_convex_main.py:6064-6092` | aggregates cohort mean biopsy variation from real biopsies only | only for `Global mean` uncertainty mode | simulated biopsies are excluded here |
| 7 | `biopsy_localization_convex_main.py:6840-6914` | samples voxel/volume points inside each reconstructed biopsy | yes | transform code works from sampled biopsy points, not only shell geometry |
| 8 | `biopsy_localization_convex_main.py:6917-7088` | builds biopsy-oriented coordinate system and bx-frame sampled points | yes for bx-frame downstream analyses | this is late but still before uncertainty / transforms |
| 9 | `biopsy_localization_convex_main.py:7188-7315` | creates uncertainty dataframe and transfers `Uncertainty data` object onto structures | yes | transform generation reads this object |
| 10 | `biopsy_localization_convex_main.py:7356-7450` and `MC_prepper_funcs.py` | generates random draws, then biopsy-only transforms, then relative-structure transforms | yes | this is the start of MC motion realization |

## Core Object Contract For Biopsies

The biopsy pipeline is centered on entries under:

- `master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]`

The code behaves as though each biopsy entry gradually matures through the pipeline rather than being replaced with a new typed object.

The most important contract groups are:

### Shell identity fields

Written at shell creation time:

- `ROI`
- `Ref #`
- `Index number`
- `Struct type`
- `Simulated bool`
- `Simulated type`
- `Relative structure type`
- `Relative structure name`
- `Relative structure ref #`

### Geometry and line-fit fields

Written by real and simulated biopsy preprocessing blocks:

- `Raw contour pts zslice list`
- `Raw contour pts`
- `Equal num zslice contour pts`
- `Inter-slice interpolation information`
- `Intra-slice interpolation information`
- `Structure centroid pts`
- `Structure global centroid`
- `Best fit line of centroid pts`
- `Centroid line sample pts`
- `Centroid line vec (bx needle base to bx needle tip)`
- `Centroid line unit vec (bx needle base to bx needle tip)`
- `Centroid line vec length (bx needle base to bx needle tip)`
- `Reconstructed biopsy cylinder length (from contour data)`
- `Reconstructed structure pts arr`
- `Reconstructed structure point cloud`
- `Reconstructed structure delaunay global`

### Variation and volume fields

- `Centroid variation arr`
- `Mean centroid variation`
- `Maximum projected distance between original centroids`
- `Structure volume`
- `Maximum pairwise distance`
- `Voxel size for structure volume calc`

### Sampled-point fields

Written later by sampling / coordinate-system stages:

- `Random uniformly sampled volume pts arr`
- `Random uniformly sampled volume pts pcd`
- `Bounding box for random uniformly sampled volume pts`
- `Num sampled bx pts`
- `Random uniformly sampled volume pts bx coord sys arr`
- `Random uniformly sampled volume pts bx coord sys pcd`

### Metadata fields

Written by the late all-biopsies block:

- `Bx location in prostate dict`
- `Target DIL by centroid dict`
- `Target DIL by surfaces dict`
- `Nearest DILs info dict`

### MC / uncertainty fields

- `Uncertainty data`
- `MC data: Generated normal dist random samples arr`
- `MC data: Generated normal dist random samples dilations arr`
- `MC data: Generated normal dist random samples rotations arr`
- `MC data: Generated uniform dist (biopsy needle compartment) random distance (z_needle) samples arr`
- `MC data: Generated uniform (biopsy needle compartment) random vectors (z_needle) samples arr`
- `MC data: bx only shifted 3darr`
- `MC data: bx and structure shifted dict`

## Stage 0: Shell Creation In `structure_referencer(...)`

Primary location:

- `python_files_dcm_meta_based/biopsy_localization_convex_main.py:10725+`
- simulated biopsy shell creation around `python_files_dcm_meta_based/biopsy_localization_convex_main.py:11043-11116`

What this stage does:

- builds the nested structure dictionaries for all structure types
- creates real biopsy shells from DICOM contour identities
- creates simulated biopsy shells by iterating `bx_sim_locations_dict`
- initializes a large number of later-populated biopsy fields to `None`

Important simulated-shell facts:

- each simulated biopsy shell already stores `Relative structure type`
- each simulated biopsy shell already stores `Relative structure ref #`
- each simulated biopsy shell already stores `Simulated type`
- all later MC and output slots are already present as placeholders here

Why this matters:

- the new lane does not need to wait for the late target-DIL metadata block just to know what structure a simulated biopsy is relative to
- a `v2` sidecar can attach very early by reading shell identity and relative-reference fields without yet materializing geometry

Surprising detail:

- current simulated-biopsy shell creation is gated by the existing `bx_sim_locations_dict` logic and an explicit patient check in this region; this should be treated as real current behavior, not assumed away
- that explicit patient check is currently `PatientID == 'F2'`
- the current simulated-biopsy constructor is DIL-driven, not real-biopsy-driven
- once a patient passes the gate, the code iterates eligible relative structures and enabled simulated types to create shells
- with the current configuration, that means one simulated set per eligible DIL per enabled simulated type
- multiple real biopsies targeting the same DIL do not currently produce multiple simulated sets
- a DIL with no matched real biopsy can still receive simulated biopsy shells under the current legacy regime

## Stage 1: Legacy DIL Optimizer

Primary location:

- `python_files_dcm_meta_based/biopsy_localization_convex_main.py:4371-5068`
- helper surface: `python_files_dcm_meta_based/biopsy_optimizer.py`

Main writes onto each DIL:

- `Biopsy optimization: DIL centroid optimal biopsy location dataframe`
- `Biopsy optimization: Optimal biopsy location dataframe`
- `Biopsy optimization: Optimal biopsy location (all tested lattice points) dataframe`
- `Biopsy optimization: Optimal biopsy location (zero lattice) dataframe`
- `Biopsy optimization: cubic lattice of optimization points only in dil`
- `Biopsy optimization: guidance map max-planes dataframe`

Patient-level writes:

- `Biopsy optimization: All points outside of DILs (zero points) dataframe`
- `Biopsy optimization: All points within DILs (tested points) dataframe`
- `Biopsy optimization: Optimal biopsy location (entire cubic lattice) dataframe`
- `Biopsy optimization - Cumulative projection (all points within prostate) dataframe`

Hard prerequisites:

- DIL interpolation / centroid / geometry data
- selected prostate info
- DIL-containing lattice points

Real downstream dependency:

- `biopsy_transporter.biopsy_transporter_optimal(...)` depends on `Biopsy optimization: Optimal biopsy location dataframe`

Not a hard dependency for everything:

- centroid-based simulated placement does not need optimizer outputs
- transform generation does not read optimizer outputs directly

Important caution:

- this optimizer uses global CuPy RNG internally
- transform generation also uses global CuPy RNG later
- moving CuPy-based prep earlier without RNG isolation will silently change legacy optimizer draws

## Stage 2: Real Biopsy Preprocessing

Primary location:

- `python_files_dcm_meta_based/biopsy_localization_convex_main.py:5096-5443`

Inputs it truly depends on:

- `Raw contour pts zslice list` from shell creation
- interpolation settings
- biopsy radius and geometry helpers

High-value writes in this block:

- `Raw contour pts`
- `Equal num zslice contour pts`
- `Inter-slice interpolation information`
- `Intra-slice interpolation information`
- `Point cloud raw`
- `Interpolated structure point cloud dict`
- `Centroid variation arr`
- `Mean centroid variation`
- `Maximum projected distance between original centroids`
- `Structure centroid pts`
- `Reconstructed biopsy cylinder length (from contour data)`
- `Best fit line of centroid pts`
- `Centroid line unit vec (bx needle base to bx needle tip)`
- `Centroid line vec (bx needle base to bx needle tip)`
- `Centroid line vec length (bx needle base to bx needle tip)`
- `Centroid line sample pts`
- `Reconstructed structure pts arr`
- `Reconstructed structure point cloud`
- `Reconstructed structure delaunay global`
- `Structure global centroid`
- `Structure volume`
- `Maximum pairwise distance`
- `Voxel size for structure volume calc`

Why this block is a hard anchor:

- this is where real biopsy length becomes concrete
- this is where later uncertainty-related variation values are created
- this is where the line-fit and reconstructed biopsy geometry contract is materialized for later point sampling and MC work

Important downstream consumers:

- simulated-length block reads `Reconstructed biopsy cylinder length (from contour data)`
- simulated-length block reads `Structure global centroid` to bin real cores by nearest DIL centroid
- uncertainty writer may read `Mean centroid variation` or `Maximum projected distance between original centroids`
- sampled-point generation depends on reconstructed biopsy geometry existing
- coordinate-system generation depends on `Best fit line of centroid pts`, `Centroid line vec ...`, and sampled points generated from reconstructed geometry

## Stage 3: Simulated-Length Determination

Primary location:

- `python_files_dcm_meta_based/biopsy_localization_convex_main.py:5446-5535`

What it actually does:

- scans all real biopsies
- collects a global real-length list
- builds `real_bx_lengths_by_dil[patientUID][dil_refnum]`
- computes cohort mean / std of real biopsy lengths

Important implementation detail:

- the real-to-DIL grouping in this block is built directly from real biopsy centroids and DIL centroids
- it does not consume the later `Target DIL by centroid dict` assignment from stage 5

Exact current dependency for `match real`:

- real biopsy must already have `Reconstructed biopsy cylinder length (from contour data)`
- real biopsy must already have `Structure global centroid`
- DILs must already have `Structure global centroid`
- simulated shell must already have `Relative structure type`
- simulated shell must already have `Relative structure ref #`

Current length modes:

- `full`
- `real normal`
- `real mean`
- `match real`

What `match real` does today:

- starts with global mean as fallback
- if simulated biopsy is relative to a DIL, uses the mean of real lengths associated with that same DIL ref number for that patient

Why this matters for the new lane:

- a safe early `v2` scalar prepass can likely be built around this same contract
- it does not need the late target-DIL metadata block
- it does need real biopsy geometry-derived length outputs first

## Stage 4: Simulated Biopsy Construction

Primary location:

- `python_files_dcm_meta_based/biopsy_localization_convex_main.py:5536-5889`

What this block does in order:

1. chooses a scalar length according to the selected mode
2. creates a nominal synthetic cylinder with `biopsy_creator.biopsy_points_creater_by_transport_for_sim_bxs(...)`
3. transports that nominal cylinder either by centroid or by optimizer-derived optimal location
4. rebuilds the same interpolation, centroid, line-fit, reconstruction, variation, and volume contract used by real biopsies

Hard prerequisites:

- simulated shell identity from stage 0
- scalar length decision from stage 3
- if centroid simulated type: relative-structure centroid placement support
- if optimal simulated type: legacy optimizer outputs from stage 1

Main writes:

- `Raw contour pts zslice list`
- `Raw contour pts`
- `Equal num zslice contour pts`
- `Inter-slice interpolation information`
- `Intra-slice interpolation information`
- `Point cloud raw`
- `Interpolated structure point cloud dict`
- `Centroid variation arr`
- `Mean centroid variation`
- `Maximum projected distance between original centroids`
- `Structure centroid pts`
- `Reconstructed biopsy cylinder length (from contour data)`
- `Best fit line of centroid pts`
- `Centroid line unit vec (bx needle base to bx needle tip)`
- `Centroid line vec (bx needle base to bx needle tip)`
- `Centroid line vec length (bx needle base to bx needle tip)`
- `Centroid line sample pts`
- `Reconstructed structure pts arr`
- `Reconstructed structure point cloud`
- `Reconstructed structure delaunay global`
- `Structure global centroid`
- `Structure volume`
- `Maximum pairwise distance`
- `Voxel size for structure volume calc`

Non-obvious but important:

- simulated biopsies still compute `Centroid variation arr`, `Mean centroid variation`, and `Maximum projected distance between original centroids`
- those fields are not safe to dismiss as decorative while the uncertainty writer still supports `Per biopsy mean` and `Per biopsy max`

This is the main reason the real biopsy block, simulated-length block, and simulated-biopsy block should currently be treated as one dependency unit rather than three independent slices.

## Stage 5: Late All-Biopsies Metadata Assignment

Primary location:

- `python_files_dcm_meta_based/biopsy_localization_convex_main.py:5900-6055`

What it writes for each biopsy:

- `Bx location in prostate dict`
- `Target DIL by centroid dict`
- `Target DIL by surfaces dict`
- `Nearest DILs info dict`

How it works:

- computes biopsy location relative to selected prostate centroid and dimensions
- computes nearest DIL by centroid distance
- computes nearest DIL by surface distance using DIL interpolated points and KD-tree queries

Important dependency classification:

- this block is real and meaningful
- but it is mostly metadata and grouping, not a hard prerequisite for simulated length selection or transform generation

Implication for refactor safety:

- this block can remain late or even be moved later as a reporting-oriented pass, provided downstream tables and plots still see the same fields

## Stage 6: Cohort Mean Biopsy Variation Aggregation

Primary location:

- `python_files_dcm_meta_based/biopsy_localization_convex_main.py:6064-6092`

What it does:

- iterates over biopsies
- skips simulated biopsies explicitly
- aggregates `Mean centroid variation` from real biopsies only
- writes `master_structure_info_dict["Global"]["Mean biopsy centroid variation"]`

Why this matters:

- `Global mean` biopsy uncertainty mode reads this global value later
- if the uncertainty mode stays `Per biopsy mean` or `Per biopsy max`, this stage becomes less central to the next lane

Dependency classification:

- required only for the `Global mean` branch of the uncertainty writer
- otherwise mostly a late scalar aggregation step

## Stage 7: Sampled Biopsy Volume Points

Primary location:

- `python_files_dcm_meta_based/biopsy_localization_convex_main.py:6840-6914`

What this stage writes:

- `Random uniformly sampled volume pts arr`
- `Random uniformly sampled volume pts pcd`
- `Bounding box for random uniformly sampled volume pts`
- `Num sampled bx pts`

What it depends on in practice:

- reconstructed biopsy geometry already existing for each biopsy
- the sampling-argument list assembled earlier from the reconstructed biopsy contract

Why it matters:

- later transform code works on sampled biopsy points
- these sampled points become the actual point set carried through biopsy-only transforms and downstream containment / dose logic

Dependency classification:

- hard prerequisite for transform application
- not merely a visualization or output step

## Stage 8: Biopsy-Oriented Coordinate System

Primary location:

- `python_files_dcm_meta_based/biopsy_localization_convex_main.py:6917-7088`

Inputs it consumes:

- `Best fit line of centroid pts`
- `Centroid line vec (bx needle base to bx needle tip)`
- `Reconstructed structure pts arr`
- `Random uniformly sampled volume pts arr`

What it writes:

- `Random uniformly sampled volume pts bx coord sys arr`
- `Random uniformly sampled volume pts bx coord sys pcd`

What the code is doing:

- translates the biopsy so the inferior/apex endpoint becomes the local origin
- rotates the fitted biopsy axis to the z-axis
- stores sampled biopsy points in biopsy frame coordinates

Dependency classification:

- hard prerequisite for any downstream biopsy-frame voxel indexing and per-voxel sextant work
- not required for the existence of uncertainty data itself

## Stage 9: Uncertainty Dataframe Creation And Transfer

Primary locations:

- `python_files_dcm_meta_based/biopsy_localization_convex_main.py:7188-7315`
- `python_files_dcm_meta_based/uncertainty_file_writer.py:213-305`

What the writer does for biopsies:

- starts from default per-structure sigma lists in `structs_referenced_dict`
- optionally adds biopsy variation terms in quadrature
- supports these biopsy modes:
  - `Per biopsy max`
  - `Per biopsy mean`
  - `Global mean`
  - `Default only`

Exact consumed biopsy fields in the writer:

- `Maximum projected distance between original centroids` for `Per biopsy max`
- `Mean centroid variation` for `Per biopsy mean`
- `master_structure_info_dict["Global"]["Mean biopsy centroid variation"]` for `Global mean`

What the main file transfers back onto structures:

- a populated `Uncertainty data` object containing means and sigmas for:
  - translations
  - dilations
  - rotations

Why this stage is a hard boundary:

- transform generation reads `Uncertainty data` directly
- before this transfer, transform sampling cannot proceed correctly

## Stage 10: Transform Generation And Immediate Application

Primary locations:

- `python_files_dcm_meta_based/biopsy_localization_convex_main.py:7356-7450`
- `python_files_dcm_meta_based/MC_prepper_funcs.py:1-220`
- `python_files_dcm_meta_based/cupy_functions.py:1-120`

This is really three sub-stages:

1. `MC_prepper_funcs.generate_transformations(...)`
2. `MC_prepper_funcs.biopsy_only_transformer(...)`
3. `MC_prepper_funcs.biopsy_transformer_to_relative_structures(...)`

### 10a. Random draw generation

Generated fields:

- `MC data: Generated normal dist random samples dilations arr`
- `MC data: Generated normal dist random samples rotations arr`
- `MC data: Generated normal dist random samples arr`
- `MC data: Generated uniform dist (biopsy needle compartment) random distance (z_needle) samples arr`

Consumed inputs:

- `Uncertainty data`
- biopsy length for optional uniform needle-compartment shift generation

Exact biopsy-length dependency:

- `cupy_functions.MC_simulator_shift_biopsy_structures_uniform_generator_cupy(...)` reads `Reconstructed biopsy cylinder length (from contour data)` and computes
  `biopsy_needle_compartment_length - bx_core_length`

### 10b. Biopsy-only transforms

Consumed inputs:

- `Random uniformly sampled volume pts arr`
- `Best fit line of centroid pts`
- `Structure global centroid`
- generated dilation / rotation / translation samples

Generated / updated fields:

- `MC data: Generated uniform (biopsy needle compartment) random vectors (z_needle) samples arr`
- total biopsy-only transformed sampled-point arrays used for later MC steps

### 10c. Relative-structure transforms

This stage uses the biopsy motion to generate the relative-motion view needed by downstream containment logic.

Important design truth:

- translation / rotation draws are shared conceptual inputs across later steps
- relative-structure dilations are not equivalent to biopsy motion and stay on the structure side of the workflow

## Field-Level Producer / Consumer Map

| Field | First writer | Main consumers | Classification |
| --- | --- | --- | --- |
| `Relative structure type` / `Relative structure ref #` | stage 0 shell creation | simulated-length block, simulated transporter, any future early v2 targeting logic | functionally required and available early |
| `Biopsy optimization: Optimal biopsy location dataframe` | stage 1 optimizer | `biopsy_transporter.biopsy_transporter_optimal(...)` | hard dependency only for optimal simulated placement |
| `Reconstructed biopsy cylinder length (from contour data)` | stage 2 real / stage 4 simulated | simulated-length stats, uniform needle-compartment shift generator | functionally required |
| `Structure global centroid` | stage 2 real / stage 4 simulated | simulated-length real-to-DIL binning, late target-DIL metadata, transform helpers | functionally required |
| `Mean centroid variation` | stage 2 real / stage 4 simulated | uncertainty writer `Per biopsy mean`, cohort global mean aggregation | functionally required under current uncertainty modes |
| `Maximum projected distance between original centroids` | stage 2 real / stage 4 simulated | uncertainty writer `Per biopsy max` | functionally required under current uncertainty modes |
| `Best fit line of centroid pts` | stage 2 real / stage 4 simulated | biopsy-frame generation, biopsy-only transform logic | functionally required |
| `Reconstructed structure pts arr` | stage 2 real / stage 4 simulated | late target-DIL surface-distance checks, plotting, downstream geometry consumers | functionally required |
| `Random uniformly sampled volume pts arr` | stage 7 | biopsy-only transform logic, downstream MC simulators | functionally required and late |
| `Random uniformly sampled volume pts bx coord sys arr` | stage 8 | per-voxel sextant and bx-frame analyses | late analytical contract |
| `Target DIL by centroid dict` / `Target DIL by surfaces dict` / `Nearest DILs info dict` | stage 5 | reporting, grouping, downstream tables / plots | meaningful but not a prerequisite for transform generation |

## What Can Move Earlier vs What Must Stay Late

### Can be split into an earlier scalar-only prepass

- any new `v2` lane object that only needs simulated-shell identity
- simulated-length helper logic that only consumes already-known real biopsy lengths and DIL centroids
- planning / routing objects that do not yet require sampled biopsy points or uncertainty transfer

### Must stay after real biopsy preprocessing

- anything that needs `Reconstructed biopsy cylinder length (from contour data)`
- anything that needs the real biopsy line-fit / centroid variation contract
- `match real` length selection itself, unless real length is exposed earlier by a dedicated scalar prepass

### Must stay after simulated-biopsy construction

- any code that needs simulated biopsies to satisfy the same geometry contract as real biopsies
- per-biopsy uncertainty modes, unless those uncertainty inputs are deliberately redesigned

### Can stay late without blocking a new lane

- `Target DIL by centroid dict`
- `Target DIL by surfaces dict`
- `Nearest DILs info dict`
- biopsy-in-prostate classification metadata

### Must stay after sampled-point generation and uncertainty transfer

- transform generation and application
- any exact reuse of transform draw arrays

## Refactor Implications For A `v2` Lane

The safest immediate strategy is not to modularize the whole pipeline at once.

The safer strategy is:

1. preserve the current dict contract
2. add sidecar `v2` logic beside the legacy path
3. modularize the new path first
4. only extract legacy-adjacent wrappers when the contract is already understood

### Safe insertion points for the next lane

#### Early shell-aware sidecar

A new helper can safely read simulated biopsy shells immediately after `structure_referencer(...)` and build `v2` routing state keyed by:

- patient UID
- simulated biopsy index / ref
- simulated type
- relative structure type
- relative structure ref #

This helper should not yet depend on optimizer outputs or sampled points.

#### Post-real-biopsy scalar prepass

A new helper can safely run after the real biopsy block and before the simulated biopsy block to compute:

- real biopsy length summaries
- per-patient / per-DIL real biopsy length bins
- any `v2` scalar selection metadata

This is the cleanest place to separate the length-selection problem from the simulated-geometry construction problem.

#### Simulated-construction sidecar

A `v2` simulated constructor can remain adjacent to the current simulated biopsy block and should write the same downstream contract fields, at least initially.

That preserves compatibility with:

- sampled-point generation
- biopsy-frame generation
- uncertainty writer
- transform generation

### Naming direction

For the next lane, use explicit `v2`-suffixed names for new lane-owned state and helpers, rather than overloading existing legacy names. The new lane should coexist with the current optimizer and current simulated-biopsy types rather than replacing them in place.

## Thoughts On Full Main Modularization

The end-state you want is reasonable: `main()` should mostly be orchestration and stage calls.

But the correct way to get there in this codebase is incremental extraction around real dependency boundaries, not a broad mechanical split.

The strongest extraction seams already visible are:

1. shell creation and structure registration
2. DIL optimization
3. real biopsy preprocessing
4. simulated-length scalar preparation
5. simulated biopsy construction
6. late biopsy metadata enrichment
7. sampled-point generation
8. biopsy-frame generation
9. uncertainty preparation and transfer
10. transform generation and application

The best tandem strategy for the current task is:

- create new `v2` modules for the new lane first
- keep them called from `main()` adjacent to the legacy blocks they depend on
- once those boundaries are stable, wrap the adjacent legacy blocks with matching helper functions
- only then consider collapsing larger spans of `main()` into a pure call-space

That gets the modularization moving without forcing a risky whole-pipeline rewrite.

## Open Validation Checks Before Python Edits

1. Confirm the exact simulated-biopsy types and `v2` names that should coexist with the current `centroid` and `optimal` simulated types.
2. Confirm whether the new lane should intentionally preserve simulated-biopsy variation-field generation for compatibility with the current uncertainty writer, or whether the uncertainty path will be changed at the same time.
3. Confirm whether any new exact-draw-sharing requirement means transform draw generation must be exposed as a reusable prepass instead of remaining where it currently is.
4. Confirm whether the patient filter behavior in simulated shell creation is intentional current production behavior or temporary logic that should not constrain the `v2` lane.

## Bottom Line

The real dependency wall is not the late target-DIL metadata block.

The real dependency wall is the combined unit formed by:

- real biopsy preprocessing
- real-length aggregation / nearest-DIL binning
- simulated biopsy construction
- later uncertainty and transform preparation that still consume the resulting biopsy contract

That is the unit the next lane has to respect.

If we modularize in tandem with the current task, the correct first modularization target is the new lane itself plus a small scalar prepass for length-selection inputs, not an immediate large-scale rewrite of all of `main()`.
# Simulated Biopsy Refactor Plan

## Goal

Refactor simulated biopsy handling into explicit, movable modules without changing behavior prematurely.

The immediate objective is not to reorder the pipeline yet. The immediate objective is to split the current inline simulated-biopsy block into clean module boundaries while keeping it in the same execution position as the legacy flow, validate that the modularized version behaves the same, and only then move the appropriate pieces before or after the optimizer.

This document is the working implementation plan for that refactor.

## High-Level Model

The refactor is based on three distinct states. These states must not be collapsed into one another.

### 1. Preparation State

Preparation is metadata-only. It decides:

- target assignment,
- multiplicity,
- matched real biopsy identity,
- nominal simulated biopsy length,
- preparation dataframe output.

Current home:

- `python_files_dcm_meta_based/preprocessing/biopsy_processing/simulated_biopsy_preparation.py`

Preparation must remain the source of truth for nominal simulated biopsy length and family metadata.

### 2. Planning State

Planning is a canonical simulated biopsy object in a local planning frame. It is not final world-space geometry.

Planning exists so that:

- optimizer v2 can operate on a stable pre-transport biopsy representation,
- a future biopsy point sampler can consume a canonical biopsy object,
- planning geometry is separated from the final realized geometry that downstream parsing expects.

Planning must not overwrite the normal final biopsy geometry fields in the master structure dictionary.

### 3. Realized State

Realized state is the final transported, reconstructed, world-space simulated biopsy geometry.

This is the only state that downstream biopsy parsing, sextant classification, voxel binning, and MC-related code should consume.

## Current Pipeline Shape

The current relevant execution order in `biopsy_localization_convex_main.py` is:

1. real biopsy processing
2. simulated biopsy preparation
3. legacy optimizer
4. inline simulated biopsy realization block
5. downstream all-biopsies parsing and later MC-related steps

That means the current inline simulated block is post-optimizer and writes the final realized geometry that later code consumes.

## Critical Invariants

These are non-negotiable invariants for the refactor.

### Planning geometry must not be written into final biopsy fields

The biggest architectural risk is accidentally storing canonical planning geometry into the standard biopsy fields before final transport.

If that happens, later code will silently treat placeholder geometry as final realized geometry.

### Preparation remains the only source of nominal simulated length

The length logic already lives in `simulated_biopsy_preparation.py`. Main and later modules should consume prepared length values rather than recomputing legacy cohort statistics locally.

### Downstream parsing must continue to read final realized geometry only

Later code still depends on the realized biopsy geometry contract. That downstream location should not be moved until the new modular boundaries are proven.

### First pass must preserve behavior before reordering

The first pass is a modularization pass, not a movement pass.

We should first extract the inline simulated-biopsy logic into modules while keeping it in the same place in the pipeline. Only after that validation succeeds should we move planning earlier and keep realization later.

## Current Realized Simulated Biopsy Contract

The current inline simulated-biopsy block writes the standard downstream biopsy fields. Any first-pass modularized replacement must continue writing the same fields.

Current writeback includes:

- `Raw contour pts zslice list`
- `Raw contour pts`
- `Equal num zslice contour pts`
- `Inter-slice interpolation information`
- `Intra-slice interpolation information`
- `Maximum pairwise distance`
- `Structure volume`
- `Voxel size for structure volume calc`
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

Downstream code later reads at least the following as part of the canonical realized-biopsy surface:

- `Reconstructed biopsy cylinder length (from contour data)`
- `Centroid line vec (bx needle base to bx needle tip)`
- `Raw contour pts zslice list`
- `Reconstructed structure pts arr`

That contract must remain stable through the first pass.

## Recommended Refactor Strategy

The refactor should happen in phases.

### Phase 1: Modularize In Place

This is the first implementation pass and the recommended immediate next step.

#### Objective

Split the current inline simulated-biopsy block into distinct modules while keeping the overall execution position unchanged.

In other words:

- simulated preparation stays where it is,
- optimizer stays where it is,
- the new modularized simulated realization path stays where the current inline block is,
- downstream parsing stays where it is.

This gives a clean validation checkpoint before any reorder.

#### Why this is the correct first pass

This pass isolates behavioral risk.

If we both modularize and reorder at the same time, then a failure could come from:

- module extraction,
- state-contract drift,
- changed ordering,
- or transport/planning separation mistakes.

If we modularize first in place, then the only moving part is extraction quality.

#### Proposed module boundaries for Phase 1

All new simulated-biopsy modules should live beside the current prep and real biopsy modules under:

- `python_files_dcm_meta_based/preprocessing/biopsy_processing/`

Recommended structure:

1. `simulated_biopsy_processor.py`
2. `simulated_biopsy_planner.py`
3. optional shared helper for biopsy geometry finalization

`biopsy_transporter.py` should remain the low-level transport helper file for now.

#### Proposed responsibilities

##### `simulated_biopsy_processor.py`

This should become the new top-level orchestrator that replaces the inline simulated-biopsy block in main.

Responsibilities:

- own the patient loop and progress updates for simulated biopsies,
- call the planner,
- call the final transport dispatcher using the existing transport helpers,
- call the geometry finalization path,
- write final realized fields to the master structure dictionary,
- return `live_display`.

This module should sit in the same pipeline position as the current legacy inline simulated block during Phase 1.

##### `simulated_biopsy_planner.py`

This module should build canonical planned simulated biopsy geometry from the preparation metadata.

Responsibilities:

- read nominal length from the simulated biopsy preparation dict,
- build the straight local-frame biopsy shell,
- define the canonical planning-frame representation,
- store planning state on the biopsy object in a dedicated planning dict,
- return the planned z-slice list or equivalent planning geometry.

Even though the planner will eventually move before the optimizer, in Phase 1 it should still be called by the new simulated processor in the current legacy location.

That gives us a validated planning interface before any movement.

##### Optional shared helper: biopsy geometry finalizer

This is the optional cleanup step.

There is substantial overlap between the real biopsy processor and the current simulated-biopsy geometry finalization logic. If that overlap is extracted, both real and simulated processors can call the same helper that converts a z-slice list into the standard biopsy geometry fields.

This helper is optional for the first pass. It is useful, but it should not block the core simulated modularization.

If implemented, the helper should:

- accept a biopsy z-slice list,
- run interpolation,
- reconstruct the biopsy cylinder,
- calculate centroid-line and geometry features,
- calculate structure volume,
- write the standard downstream biopsy fields.

If implemented in Phase 1, it must preserve current behavior exactly.

#### Planning dict for future movement

To avoid rework in Phase 2, Phase 1 should already introduce a dedicated simulated biopsy planning dict.

Suggested minimum contents:

- `Planning complete`
- `Planning frame`
- `Nominal length mm`
- `Planned biopsy radius mm`
- `Planned centroid count`
- `Planned centroid separation mm`
- `Planned raw contour pts zslice list`
- `Planning source`

This dict should be additive and should not replace any current canonical biopsy field.

#### Transport in Phase 1

In Phase 1, transport should still occur in the new simulated processor in the same location as the legacy inline block.

The processor should continue using the existing transport helpers:

- centroid transport via `biopsy_transporter.biopsy_transporter_centroid(...)`
- optimal transport via `biopsy_transporter.biopsy_transporter_optimal(...)`

We should not introduce a second transport system in the first pass.

#### Legacy block handling in Phase 1

After the modularized replacement is inserted and validated, the legacy inline block in main should be preserved under `if False:` as the short-term archive.

That is the agreed intermediate archival style for large block removals in this repo.

#### Validation gate for Phase 1

Phase 1 is not complete until all of the following are done:

1. editor diagnostics are clean on the touched files,
2. the inline legacy block in main is under `if False:`,
3. the new modularized processor writes the same realized-field surface,
4. one representative run confirms that downstream parsing still sees final realized geometry,
5. a focused comparison confirms no drift in the core realized-biopsy fields for the tested case.

### Phase 2: Split Planning From Realization In The Pipeline

Only after Phase 1 validation should we change the ordering.

#### Objective

Move the planner before the optimizer and keep the final realization after the optimizer.

Target shape:

1. real biopsy processing
2. simulated biopsy preparation
3. simulated biopsy planning
4. optimizer
5. simulated biopsy final transport and realization
6. downstream parsing

#### Why this order is correct

This matches the logical dependency graph:

- planning depends on prepared metadata, not on optimizer output,
- optimizer may need planning geometry in the future,
- final transport for optimal simulated biopsies depends on optimizer output,
- downstream parsing depends on final realized geometry.

#### Important caution

When Phase 2 happens, planning must already be stored in its own dedicated planning dict so that moving it earlier does not expose downstream code to pre-final geometry.

### Phase 3: Add A Transport Dispatcher Abstraction

Once planning and realization are split in the pipeline, add a transport dispatcher that answers:

- where should this simulated biopsy be placed finally,
- which transport family should be used,
- which optimizer output should be consulted.

The dispatcher should hide whether the source is:

- legacy centroid transport,
- legacy optimal transport,
- future optimizer v2 output,
- or any future transport family.

This keeps the post-optimizer simulated processor from hardcoding optimizer-specific field assumptions.

### Phase 4: Add The Biopsy Point Sampler For Optimizer V2

This phase depends on the planner existing as a stable pre-optimizer seam.

The future biopsy point sampler should consume planning geometry, not final realized geometry.

That is the correct place for stochastic optimizer-v2 support.

The sampler should be deterministic and reusable.

That means:

- the same sampler definition should be usable on pre-optimizer planning geometry and on post-optimizer realized geometry,
- sampling should be defined cleanly enough that a planning biopsy can be sampled once and then repeatedly transported if that is the efficient strategy for coarse-to-fine or stochastic optimization loops,
- the optimizer should not depend on a second special-purpose sampling scheme that differs from the downstream realized-geometry sampling interpretation.

Target dependency shape:

1. preparation
2. planning
3. biopsy point sampler
4. optimizer v2
5. final transport and realization

### Phase 5: Extract Shared Finalization Logic If Still Worthwhile

After the simulated path is modularized and reordered safely, revisit the duplication between:

- `biopsy_processor.py`
- the new simulated biopsy realization/finalization path

At that point, if the shared surface is still large and stable, extract the common z-slice-to-biopsy-fields logic into a dedicated helper.

This is useful, but it is not required before the planner/realizer boundary is proven.

## Proposed Phase 1 File Layout

Recommended initial file additions:

- `python_files_dcm_meta_based/preprocessing/biopsy_processing/simulated_biopsy_processor.py`
- `python_files_dcm_meta_based/preprocessing/biopsy_processing/simulated_biopsy_planner.py`

Optional:

- `python_files_dcm_meta_based/preprocessing/biopsy_processing/biopsy_geometry_helper.py`

The existing files should remain:

- `python_files_dcm_meta_based/preprocessing/biopsy_processing/biopsy_processor.py`
- `python_files_dcm_meta_based/preprocessing/biopsy_processing/simulated_biopsy_preparation.py`
- `python_files_dcm_meta_based/biopsy_transporter.py`

## Proposed Phase 1 API Shape

These names are proposals, not rigid requirements, but the boundaries are intentional.

### Planner API

Example shape:

```python
def build_simulated_biopsy_planning_state(
    specific_structure,
    simulated_bx_rad,
    num_centroids_for_sim_bxs,
    plot_simulated_cores_immediately,
):
    ...
```

Output expectation:

- populate a simulated biopsy planning dict,
- return planned z-slice geometry in the canonical planning frame if needed immediately.

### Processor API

Example shape:

```python
def simulated_biopsy_processer(
    master_structure_reference_dict,
    master_structure_info_dict,
    structs_referenced_dict,
    bx_ref,
    dil_ref,
    parallel_pool,
    interp_inter_slice_dist,
    interp_intra_slice_dist,
    interp_dist_caps,
    biopsy_radius,
    voxel_size_for_structure_volume_calc_non_bx,
    factor_for_voxel_size,
    cupy_array_upper_limit_NxN_size_input,
    layout_groups,
    nearest_zslice_vals_and_indices_cupy_generic_max_size,
    generate_cuda_log_files_volume_calculation,
    constant_z_slice_polygons_handler_option,
    remove_consecutive_duplicate_points_in_polygons,
    include_edges_in_log_files,
    custom_cuda_kernel_type,
    demonstrate_volume_calculation_correctness_bool_1,
    plot_volume_calculation_containment_result_bool_1_old,
    plot_binary_mask_bool,
    patients_progress,
    structures_progress,
    completed_progress,
    indeterminate_progress_sub,
    live_display,
):
    ...
```

Responsibilities:

- loop the simulated biopsies,
- obtain planning geometry,
- resolve transport,
- realize final geometry,
- write the standard realized-biopsy fields.

### Optional geometry helper API

Example shape:

```python
def finalize_biopsy_geometry_from_zslice_list(...):
    ...
```

This helper would be shared by real and simulated biopsy processors if extracted.

## What Should Not Happen In Phase 1

The following should be explicitly avoided in the first pass.

### Do not move the planner before the optimizer yet

That is Phase 2, not Phase 1.

### Do not change downstream parser location yet

The downstream parser should continue consuming final realized geometry exactly as it does now.

### Do not replace `biopsy_transporter.py`

The transport primitives already exist. The correct first step is to route through them cleanly, not rewrite them.

### Do not collapse planning and final geometry into the same fields

That would create silent correctness drift.

## Validation Checklist For The First Implementation Pass

When Phase 1 is implemented, the validation pass should answer these questions explicitly.

### Structural validation

- Is the simulated inline block in main replaced by a module call?
- Is the legacy inline block archived under `if False:`?
- Are diagnostics clean in main and the new modules?

### Behavioral validation

- Does the new processor still read nominal simulated length only from the prep layer?
- Does centroid transport still use the existing centroid helper?
- Does optimal transport still use the optimizer-produced optimal-location dataframe?
- Are the realized simulated-biopsy fields still written under the same keys as before?

### Downstream validation

- Do the later biopsy parsing stages still run without missing-field failures?
- Do the later sextant and sampled-biopsy data products still operate on realized final geometry?

### Comparison validation

For one representative case, compare before and after for at least:

- `Raw contour pts zslice list`
- `Structure global centroid`
- `Reconstructed biopsy cylinder length (from contour data)`
- `Centroid line vec (bx needle base to bx needle tip)`
- `Reconstructed structure pts arr`

If there is drift, stop and resolve it before Phase 2.

## Recommended Immediate Next Coding Pass

The next coding pass should implement Phase 1 only.

That means:

1. create the new simulated-biopsy module boundaries,
2. keep them in the current post-optimizer position,
3. preserve the current realized writeback contract,
4. archive the legacy inline block under `if False:`,
5. run a focused validation pass.

Only after that succeeds should the planner be moved before the optimizer.

## Summary Decision

The agreed safe path is:

1. modularize the current simulated-biopsy block in place,
2. validate it,
3. then split planning before the optimizer and realization after the optimizer,
4. then add the transport dispatcher and sampler seams for optimizer v2.

This plan intentionally prefers correctness and reviewability over collapsing multiple architectural moves into one pass.
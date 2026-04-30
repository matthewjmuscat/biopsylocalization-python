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

Planning now has two layers:

- planned raw contour z-slice geometry,
- a reconstructed biopsy model prepared for sampling.

The reconstructed model is the important new seam for optimizer v2.

That model should be built from the planned z-slice list using the same low-level helper that the realized finalizer uses later on final world-space geometry.

### 3. Realized State

Realized state is the final transported, reconstructed, world-space simulated biopsy geometry.

This is the only state that downstream biopsy parsing, sextant classification, voxel binning, and MC-related code should consume.

## Current Implemented Boundary

The code now has a shared lower-level builder in:

- `python_files_dcm_meta_based/preprocessing/biopsy_processing/biopsy_geometry_helper.py`

Current shared helper:

- `build_reconstructed_biopsy_model_for_sampling_from_zslice_list(...)`

This helper is the shared reconstruction seam between planning and realization.

It accepts a biopsy contour z-slice list and returns a reconstructed biopsy model prepared for sampling, including at least:

- raw contour point array,
- centroid array,
- structure global centroid,
- centroid-line fit and derived vectors,
- reconstructed biopsy array,
- reconstructed point cloud,
- reconstructed global Delaunay object,
- rotated constant-z representation of the reconstructed biopsy.

The realized finalizer still owns interpolation, volume calculation, and writeback into the canonical downstream biopsy keys.

The planner now owns the planning-side copy of that reconstructed biopsy model in the planning dict.

## Current Pipeline Shape

The current relevant execution order in `biopsy_localization_convex_main.py` is:

1. real biopsy processing
2. simulated biopsy preparation
3. legacy optimizer
4. inline simulated biopsy realization block
5. downstream all-biopsies parsing and later MC-related steps

That means the current inline simulated block is post-optimizer and writes the final realized geometry that later code consumes.

The current modularized execution order is now:

1. real biopsy processing
2. simulated biopsy preparation
3. simulated biopsy planning
4. legacy optimizer
5. simulated biopsy transport and realization
6. downstream parsing and MC-related steps

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

## Adjacent Revisit Items

These are stage-boundary items that are coupled to the simulated-biopsy refactor, but should not be confused with the simulated-geometry contract itself.

### Pickle Provenance And Config Contract

The pickle export path now carries lightweight provenance metadata, but the full load-time contract is still incomplete until the config layer exists.

Still to revisit:

- persist the authoritative configuration payload that generated the dataset,
- define which configuration fields are immutable on load versus safely overridable,
- add a proper load-time compatibility/warning layer that compares current runtime settings against exported provenance,
- include an explicit code-version compatibility policy rather than relying only on git metadata.

For now, exported provenance is useful for inspection, but it is not yet the full reproducibility/configuration contract.

### Render And Debug Surface

The processed-dataset render/debug surface belongs immediately after pickle export/load-rebuild and before biopsy sampling. That ordering should be preserved because it gives a scientific-validation checkpoint on the rebuilt runtime objects before later stochastic steps mutate or derive downstream artifacts.

Current state:

- the active processed-dataset render/debug block has been extracted out of main into a dedicated helper surface,
- the Open3D structure/dose/MR-ADC render path remains live,
- the Plotly structure-plus-dose render path remains live,
- the Plotly MR overlay path is still intentionally deferred.

Still to revisit:

- decide whether this render/debug surface should become an explicit stage module with its own config object,
- unify debug render flags so this surface is not controlled by scattered booleans in main,
- support MR rendering as a family rather than treating ADC as the only mature path,
- add a T2-specific render contract once the T2 pathway is ready enough to validate scientifically,
- decide whether validation outputs from this stage should be written to disk in a structured way instead of being purely interactive.

### Validation Rhythm

The code-level refactor can continue, but scientific validation should remain frequent. The intended workflow is:

- refactor a local seam,
- run a validation dataset,
- inspect outputs for scientific drift,
- only then continue moving upstream or downstream boundaries.

That validation cadence matters more than perfect code cleanliness for these boundary extractions.

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
- `Planned reconstructed biopsy model dict`
- `Planning source`

This dict should be additive and should not replace any current canonical biopsy field.

The `Planned reconstructed biopsy model dict` is the pre-optimizer object that future v2 sampling should consume.

It should remain planning-scoped. It must not be written into the normal realized biopsy keys.

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

Status update:

- the first transport-dispatch seam now exists,
- the current transporter still preserves the centroid and optimal translation math,
- the remaining work in this phase is selection-contract cleanup and diagnostics rather than basic code movement.

The next pass for this phase should make the dispatcher answer two separate questions:

- which transport family applies,
- which final candidate location should be kept.

That split matters because future optimizer-v2 and guidance-map flows may want to expose more than one retained candidate even when the actual transport family is still just one of the legacy families.

### Phase 3A: Reconcile Intended And Realized Targeting

The early targeter should remain in the preparation layer.

That early pass records intended targeting metadata only:

- which structure the simulated biopsy is supposed to target,
- where multiplicity and family membership came from,
- what the optimizer or transport stage is expected to preserve.

That data should continue to live in the simulated biopsy preparation dict.

After optimizer-driven transport and final simulated-biopsy realization, run a second targeting pass on the fully realized biopsy geometry.

That late pass should continue to write the legacy downstream fields such as:

- `Target DIL by centroid dict`,
- `Target DIL by surfaces dict`,
- `Nearest DILs info dict`,
- `Bx location in prostate dict`.

Those late fields should remain the downstream contract because existing builders and plots still consume them.

The preparation dict should not be overwritten by the late pass.

Reasoning:

- the early targeter answers intended targeting,
- the late targeter answers achieved targeting,
- for stable workflows they should usually agree,
- when they do not agree, that disagreement is a validation signal rather than a reason to collapse the two namespaces.

### Phase 3B: Add A Target Agreement Validator

Once both targeting layers exist, add a focused target-agreement validator.

This should compare at least:

- the intended target recorded in the preparation dict,
- the realized target recorded in the late legacy targeting fields.

The first implementation should be warning-oriented rather than hard-failing.

Recommended outputs:

- a Rich warning line when intended and realized targets disagree,
- patient ID and biopsy ID in the warning text,
- intended target and realized target identifiers in the message,
- an optional dataframe or summary object for later auditing.

This validator becomes more important when optimizer v2 or any stochastic targeting mode is introduced, because final location is not truly known until after transport and finalization.

### Phase 3C: Add Transport Selection Diagnostics And Rank Retention

For optimal simulated biopsies, do not rely on silent selection of the winning optimizer row.

The transport or optimizer stage should expose enough metadata to confirm that the retained biopsy really corresponds to the intended top-ranked candidate.

The next pass should record and optionally display:

- how many optimal candidates were considered,
- which candidate rank was retained,
- what tie-break rule was applied,
- the retained candidate coordinates.

The Rich UI should emit a concise confirmation line when an optimal biopsy is realized so that it is obvious that rank 1 was kept unless another rank was explicitly requested.

This phase should also leave room for a future `keep_k_optimal` contract.

That contract is useful for both:

- future optimizer-v2 simulated biopsy families,
- existing and future guidance-map flows that may want to expose the top several retained candidates rather than only the best one.

Note that there is already meaningful candidate-rank guidance-map work in `advanced_guidance_map_creator.py`, so this phase should align with that existing surface rather than inventing a conflicting candidate contract.

### Phase 4: Add The Biopsy Point Sampler For Optimizer V2

This phase depends on the planner existing as a stable pre-optimizer seam.

The future biopsy point sampler should consume the planning-side reconstructed biopsy model, not the realized downstream biopsy keys.

That is the correct place for stochastic optimizer-v2 support.

The sampler should be deterministic and reusable.

That means:

- the same sampler definition should be usable on pre-optimizer planning geometry and on post-optimizer realized geometry,
- sampling should be defined cleanly enough that a planning biopsy can be sampled once and then repeatedly transported if that is the efficient strategy for coarse-to-fine or stochastic optimization loops,
- the optimizer should not depend on a second special-purpose sampling scheme that differs from the downstream realized-geometry sampling interpretation.

Target dependency shape:

1. preparation
2. planning
3. shared reconstructed biopsy model builder
4. deterministic biopsy point sampler
5. optimizer v2
6. final transport and realization

### Phase 4A: Shared Reconstructed Model Contract

The planning stage and the realized finalizer must pass through the same reconstruction helper.

That shared helper should remain lower-level than the finalizer.

Why this matters:

- the optimizer needs the same reconstructed biopsy object definition that downstream realization uses,
- but the optimizer does not need interpolation side effects, world-space writeback, or volume calculation on every candidate,
- keeping the builder lower-level avoids forcing optimizer v2 through the full realized finalization path.

The correct split is:

1. build reconstructed biopsy model from a z-slice list,
2. optionally sample from that model,
3. optionally finalize and write the realized downstream fields.

The finalizer should call the builder, not duplicate it.

The planner should call the same builder and store the resulting model in planning namespace.

### Phase 4B: Containment Backend Recommendation

For the next pass, keep the current Delaunay/global-convex containment method for biopsy sampling.

Recommendation:

- do not replace biopsy Delaunay containment with the constant-z-slice point-in-polygon path in this pass,
- treat any future containment swap as an internal sampler backend change, not as a new planner/finalizer architecture.

Reasoning:

- the current reconstructed biopsy is already a straight convex object,
- the existing late sampler already consumes the reconstructed global Delaunay object successfully,
- optimizer v2 only needs the same sampling contract earlier in the pipeline,
- the constant-z-slice polygon route would introduce a second containment contract and another place for geometric drift,
- your custom point-in-polygon route appears to rely on constant z slices, so it is only attractive after an explicit profiling or correctness reason justifies the extra conversion path.

So the recommended order is:

1. reuse the current reconstructed-object definition,
2. reuse Delaunay containment for the first planning-side sampler,
3. only revisit constant-z polygon containment later if profiling or edge-case behavior gives a concrete reason.

### Phase 4C: Replace The Downstream Sampler With The Shared Module

Before the sampler is reused pre-optimizer, extract the current repaired downstream sampler into a dedicated module and switch the downstream MC-prep path to call that shared module.

The first pass should preserve the current downstream sampler semantics exactly.

That means preserving at least:

- the centerline-aligned sampling lattice,
- the start-at-apex behavior,
- the explicit axial end behavior,
- the current point-index contract used by downstream dataframe builders,
- the current Delaunay/global-convex containment backend.

This step freezes the existing sampler behavior in one place before it is reused for optimizer v2.

Validation for this phase should compare before and after for a representative case at the sampled-point level, not only at the geometry level.

Compare at least:

- sampled point arrays,
- sampled point counts,
- biopsy-frame coordinates,
- downstream per-point dataframe row counts.

### Phase 4D: Move Transform Generation Earlier By Splitting Generation From Application

The current code already has an early transform-generation stage and a later transform-application stage, but the contract is still blurred by realized-geometry dependencies.

In this context, transform generation means only generating the trial-wise parameters such as:

- dilation samples,
- rotation samples,
- translation samples,
- biopsy-needle-compartment shift samples.

It does not mean acting on biopsy sample points yet.

The correct refactor is:

1. keep a pure transform-parameter generation stage,
2. keep a later transform-application stage that acts on sampled points.

The remaining blocker to moving generation earlier is that the biopsy-needle-compartment shift generation still reads realized contour length.

That dependency should be rerouted to nominal or planned simulated length from the preparation or planning layer.

Preferred end state:

- generate transform parameters for all biopsies before optimizer-v2 targeting,
- continue to apply those parameters later to sampled points after sampling has been performed.

Interim fallback if needed:

- move real-biopsy transform generation earlier first,
- but do not make that the long-term contract unless simulated-length rerouting becomes unexpectedly expensive.

### Phase 5: Extract Shared Finalization Logic If Still Worthwhile

After the simulated path is modularized and reordered safely, revisit the duplication between:

- `biopsy_processor.py`
- the new simulated biopsy realization/finalization path

At that point, if the shared surface is still large and stable, extract the common z-slice-to-biopsy-fields logic into a dedicated helper.

This is useful, but it is not required before the planner/realizer boundary is proven.

Status update:

- the shared realized finalizer already exists,
- the lower-level reconstructed-biopsy-model builder now also exists,
- the remaining missing piece is the deterministic planning-side biopsy point sampler that consumes the planned reconstructed model.

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
- return planned z-slice geometry in the canonical planning frame if needed immediately,
- build and store the planned reconstructed biopsy model dict for sampling.

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
def build_reconstructed_biopsy_model_for_sampling_from_zslice_list(...):
    ...

def finalize_biopsy_geometry_from_zslice_list(...):
    ...
```

These helpers now serve different layers:

- `build_reconstructed_biopsy_model_for_sampling_from_zslice_list(...)` is the shared lower-level reconstruction seam for planning and realization,
- `finalize_biopsy_geometry_from_zslice_list(...)` is the realized downstream writeback layer.

## What Should Not Happen In Phase 1

The following should be explicitly avoided in the first pass.

### Do not move the planner before the optimizer yet

That is Phase 2, not Phase 1.

### Do not change downstream parser location yet

The downstream parser should continue consuming final realized geometry exactly as it does now.

### Do not replace `biopsy_transporter.py`

The transport primitives already exist. The correct first step is to route through them cleanly, not rewrite them.

### Do not replace Delaunay biopsy containment during the sampler-seam pass

That is a backend optimization question, not the current architectural blocker.

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

For the current rerun, also compare the matching patient outputs against the Mar 3 reference run once the run completes.

If there is drift, stop and resolve it before beginning the phased upgrades below.

## Recommended Immediate Next Upgrade Sequence

The next work should proceed in this order:

1. close validation on the current rerun,
2. compare current outputs against the older matching patients and the Mar 3 reference outputs,
3. add the late realized-targeting pass while keeping the early intended-targeting pass unchanged,
4. add the target-agreement validator and Rich warning output,
5. add transport-selection diagnostics and rank-retention metadata for optimal transports,
6. extract the repaired downstream sampler into a shared module and replace the downstream caller with it,
7. reroute transform-generation length reads to preparation or planning length and move transform generation earlier,
8. then consume the shared sampler from the planning side for optimizer v2.

Guidance-map candidate-rank support should be treated as an active dependency during the transport-selection work rather than as a separate disconnected track.

## Summary Decision

The agreed safe path is:

1. validate the now-modularized planning and realization pipeline against existing outputs,
2. keep intended and realized targeting as distinct contracts,
3. add agreement validation between those two targeting layers,
4. make optimal-transport selection explicit and diagnosable,
5. freeze the current repaired downstream sampler inside a shared module,
6. then move transform generation earlier and reuse the same sampler for optimizer v2.

This plan intentionally prefers correctness and reviewability over collapsing multiple architectural moves into one pass.
# Target DIL Optimization Implementation Plan

## Goal

Add a new target-lesion-specific optimization lane that:

- optimizes each simulated core against one selected target DIL only,
- uses coarse-to-fine voxelization for optimization scoring only,
- retains a final target-specific optimized score for every biopsy,
- keeps the current legacy downstream pipeline intact,
- retains legacy any-DIL `PD_all` as a comparator,
- allows us to verify that the optimizer score and the final downstream target-specific score agree.

## Recommended Methodology Decision

Use `PD_target` as the primary paper endpoint for the new method.

Reason:

- the new optimizer is target-lesion-specific,
- the primary downstream evaluation should match the quantity that was actually optimized,
- `PD_all` is still useful, but as a comparator or sensitivity analysis rather than the primary endpoint.

So the planned outputs are:

- `PD_target`: target-lesion-specific score,
- `PD_all`: legacy any-DIL score,
- agreement check between optimizer target score and final target-specific downstream score.

## Current Anchors In The Code

These are the main objects already in place.

### Upstream target-lesion metadata already exists

In the upstream repo, `biopsy_localization_convex_main.py` already assigns per-biopsy target-lesion metadata to each biopsy object:

- `Target DIL by centroid dict`
- `Target DIL by surfaces dict`
- `Nearest DILs info dict`

These are written onto each biopsy structure object in the main patient structure dictionary.

### Legacy optimizer entry point already exists

In the upstream repo, `biopsy_optimizer.py` currently uses:

- `find_dil_optimal_sampling_position(...)`

This currently optimizes the legacy surrogate based on normal-distribution points contained.

### Structure-specific downstream DIL scores already exist

In the upstream repo, `dataframe_builders.py` already creates per-biopsy, per-structure global tissue statistics from:

- `specific_bx_structure["MC data: compiled sim results dataframe"]`

Those structure-specific global statistics are stored in:

- `Tissue class - Global tissue by structure statistics`

This table already contains one row per biopsy and relative structure, keyed by:

- `Relative structure ROI`
- `Relative structure type`
- `Relative structure index`

### There is already a target-specific extraction helper

In the upstream repo, `dataframe_builders.py` already has:

- `bx_global_score_to_target_dil_3d_radiomic_features_dataframe_builder(...)`

This function already uses `Target DIL by centroid dict` to filter the structure-level global tissue table down to the target lesion row for each biopsy.

This is important because it means the downstream target-specific score is not a new concept. It is already recoverable from the existing structure-level outputs.

## Scope Boundaries

### What we will change

- target-specific optimization scoring,
- optimization orchestration,
- persistence of per-biopsy target-specific optimization metadata,
- export of `PD_target` alongside `PD_all`.

### What we will not change in the first implementation passes

- biopsy geometry creation in `biopsy_creator.py`,
- biopsy transport logic in `biopsy_transporter.py`,
- legacy any-DIL Monte Carlo tumor scoring in `MC_simulator_convex.py`,
- existing downstream voxel binning helpers,
- current QA pipeline behavior in this repo.

This keeps the risky voxelization and downstream geometry logic frozen while we change only the optimization score lane.

## Proposed New Objects

To avoid overloading legacy fields, add new explicit objects rather than reusing ambiguous ones.

### On each biopsy structure object

Add a new dictionary on each biopsy structure in the upstream master structure dictionary:

- `Optimization target score dict`

Proposed contents:

- `Score mode`: `PD_target_voxelized`
- `Target source`: `centroid` or `surface`
- `Target DIL ROI`
- `Target DIL type`
- `Target DIL refnum`
- `Target DIL index`
- `Optimizer voxel size coarse mm`
- `Optimizer voxel size fine mm`
- `Optimizer coarse best score`
- `Optimizer fine best score`
- `Optimizer final selected score`
- `Optimizer selected location`
- `Optimizer search summary`
- `Final geometry rescored target score`
- `Final geometry rescored target score abs diff`
- `Legacy PD_all`

Do not store this inside `MC data: tumor tissue probability` because that object currently means legacy any-DIL tumor scoring.

### New export dataframe

Add a dedicated upstream export dataframe for one row per biopsy:

- `Target DIL optimization summary dataframe`

This dataframe should include:

- biopsy identifiers,
- simulation type,
- target-lesion identifiers,
- `PD_target`,
- `PD_all`,
- optimizer coarse score,
- optimizer fine score,
- final rescored target score,
- optimizer vs final-score difference.

This should be a clean audit table, not hidden inside a radiomics merge helper.

## Implementation Phases

## Phase 1: Expose A Clean `PD_target` Export Without Changing Optimization

### Objective

First prove that we can recover one target-specific downstream score per biopsy from the existing structure-level score tables.

### Files / objects to modify

Upstream repo only:

- `dataframe_builders.py`

Primary objects:

- `Tissue class - Global tissue by structure statistics`
- `bx_global_score_to_target_dil_3d_radiomic_features_dataframe_builder(...)`

### Exact change

Refactor or extend the existing target-score extraction helper so it produces a dedicated per-biopsy audit dataframe with explicit `PD_target` columns rather than only a radiomics-merged helper output.

The new dataframe should include:

- `Patient ID`
- `Bx ID`
- `Bx index`
- `Bx refnum`
- `Simulated bool`
- `Simulated type`
- `Target DIL ROI`
- `Target DIL type`
- `Target DIL refnum`
- `Target DIL index`
- `PD_target mean`
- `PD_target nominal`
- `PD_target std err`

If easy, also include:

- `PD_all`

but this is optional in Phase 1.

### Why this phase comes first

This confirms that the target-specific downstream score can already be extracted cleanly before touching any optimization logic.

### Validation

- exactly one target-specific row per biopsy,
- no fraction mixing in the target mapping,
- no missing target rows for biopsies that have target metadata,
- a small manual spot-check that the extracted target row matches the biopsy target dictionary.

## Phase 2: Create One Shared Target-Specific Scoring Function

### Objective

Create one explicit scoring function that takes a biopsy candidate and one target DIL and returns a deterministic target-specific score.

This scorer is the core of the new method.

### Files / objects to modify

Upstream repo:

- `biopsy_optimizer.py`

Possibly a new helper module if it keeps the code cleaner, but the first pass can remain in `biopsy_optimizer.py`.

### Exact change

Add a new pure scoring function, for example:

- `score_candidate_against_target_dil(...)`

Inputs should include:

- candidate biopsy position,
- candidate biopsy geometry or sample points,
- target DIL geometry,
- selected voxel size,
- any needed prostate-frame or biopsy-frame transforms.

Outputs should include at least:

- `PD_target score`,
- optional overlap counts or voxel counts for debugging.

### Design requirement

This scorer must be reusable.

It should be callable from:

- the optimizer during search,
- a final post-selection rescore step on the chosen biopsy geometry.

That is the cleanest way to make the optimizer score and final target score identical by construction.

### What stays unchanged

- `MC_simulator_convex.py`
- legacy any-DIL tumor scoring
- downstream voxel tables

### Validation

- synthetic sanity cases with obvious full hit, partial hit, and zero hit,
- score monotonicity when moving a test core farther from the target lesion,
- stable result when rerun with the same geometry and voxel size.

## Phase 3: Add Coarse-To-Fine Optimization Using The Shared Target Scorer

### Objective

Use the new target-specific scorer to drive the search, while keeping the legacy optimizer lane available.

### Files / objects to modify

Upstream repo:

- `biopsy_localization_convex_main.py`
- `biopsy_optimizer.py`

Existing anchor objects:

- `find_dil_optimal_sampling_position(...)`
- the optimizer call site in `biopsy_localization_convex_main.py`
- `simulated_biopsy_length_method = 'match real'`

### Exact change

Add a new configurable optimization mode, for example:

- `legacy_any_dil_surrogate`
- `target_dil_voxelized`

Add optimization voxel-size controls, for example:

- `optimizer_voxel_size_coarse_mm`
- `optimizer_voxel_size_fine_mm`

Then:

- bind each optimization run to one target DIL only,
- run a coarse search first,
- refine around the best coarse candidate using the fine voxel size,
- retain the final fine score as the optimized target score.

### Important wiring rule

The optimization target must come from the same biopsy-level target metadata that will be used downstream.

Do not create a separate target-selection pathway for optimization.

### Geometry rule

The coarse-to-fine change affects only scoring resolution during search.

It does not change:

- how the final biopsy geometry is created,
- how the final biopsy is transported,
- how the final biopsy is voxelized downstream.

### Validation

- legacy optimization lane still runs unchanged when selected,
- new lane produces one optimized result per intended biopsy,
- coarse-to-fine and fine-only runs on a tiny test subset choose the same or nearly identical final solution,
- stored fine score is reproducible.

## Phase 4: Persist A Final Rescored `PD_target` On The Selected Geometry

### Objective

After the final biopsy geometry is chosen, rescore that exact final geometry against the same target DIL using the same shared scorer and the fine voxel size.

This is the check that should match the optimizer's retained final score.

### Files / objects to modify

Upstream repo:

- `biopsy_localization_convex_main.py`
- `biopsy_optimizer.py`

### Exact change

After the final selected candidate is materialized as a biopsy, call the same target scorer again and store:

- `Final geometry rescored target score`
- `Optimizer final selected score`
- absolute difference between the two.

### Why this matters

If this step uses the same scorer and same final geometry, then the optimizer score and final rescored target score should agree up to floating-point tolerance.

That gives us a direct audit trail.

### Validation

- difference is zero or near-zero within a predefined tolerance,
- failures are investigated before any downstream analysis changes.

## Phase 5: Export Both `PD_target` And `PD_all`

### Objective

Export both metrics clearly so the paper can use `PD_target` as primary and `PD_all` as comparator.

### Files / objects to modify

Upstream repo:

- `dataframe_builders.py`
- any CSV writer or export registry that persists the relevant cohort tables

### Exact change

Add or extend a clean cohort-level table with these columns:

- biopsy identifiers,
- simulation type,
- target lesion identifiers,
- `PD_target`,
- `PD_all`,
- optimizer coarse score,
- optimizer fine score,
- final rescored target score,
- target-score agreement delta.

### Validation

- one row per biopsy,
- legacy rows still present,
- `PD_target` and `PD_all` both populated where expected.

## Phase 6: Downstream Repo Changes Only After Upstream Output Schema Is Stable

### Objective

Keep this repo frozen until the upstream export format is finalized.

### First-pass decision for this repo

No code changes here in the first implementation passes.

That is intentional.

The current downstream and QA code should not be touched until the upstream target-score export exists and is validated.

### Likely downstream touch points later

This repo:

- `main_pipe.py`
- `statistical_tests_1_quick_and_dirty.py`
- `qa/load.py`
- `qa/stats.py`
- `qa/plot_data.py`
- `qa/deliverables.py`

### Downstream changes to make later

- ingest new `PD_target` fields,
- make `PD_target` the primary analysis metric for the new method,
- keep `PD_all` as secondary comparator,
- add simple agreement checks between optimizer score and exported final target score,
- update figures and tables only after the upstream metric is stable.

## Audit Rules For Every Pass

Each pass should be small, auditable, and validated before moving on.

### Pass order

1. target-score extraction only,
2. shared scorer only,
3. optimizer wiring,
4. final geometry rescore,
5. export cleanup,
6. downstream analysis updates.

### Rules

- do not change the legacy lane while building the new lane,
- do not change downstream voxelization while building the new optimization scorer,
- do not combine multiple conceptual changes in one pass,
- do not update paper figures until `PD_target` has been exported and validated,
- every pass must end with a concrete agreement or row-count check.

## Practical Notes

### Naming

Use explicit names:

- `PD_target`
- `PD_all`
- `Optimization target score dict`

Avoid ambiguous names like `tumor tissue probability` for the new target-specific metric because that label already refers to legacy any-DIL behavior.

### Equality expectation

The optimizer target score and the final target-specific rescore should be identical only if they use:

- the same target lesion,
- the same final biopsy geometry,
- the same shared scoring function,
- the same fine voxel size.

That is why the plan uses a shared scorer rather than two separate implementations that we merely hope will match.

## First Pass To Execute

The safest first code pass is:

1. upstream only,
2. no optimizer changes,
3. expose a dedicated per-biopsy `PD_target` audit dataframe from the existing structure-level score outputs,
4. verify one target row per biopsy and no fraction mixing.

If that pass looks correct, we move to the shared target scorer next.
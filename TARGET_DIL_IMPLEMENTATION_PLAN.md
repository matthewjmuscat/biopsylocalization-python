# Target DIL Optimization Implementation Plan

## Goal

Add a new target-lesion-specific optimization lane that:

- optimizes each simulated core against one selected target DIL only,
- can run as a separate gated lane without disturbing the legacy optimizer,
- can coexist with the legacy lane so that legacy-only, target-only, both, or neither can be selected without code contradictions,
- retains a final target-specific optimized score for every biopsy,
- retains legacy any-DIL `PD_all` as a comparator,
- uses the same enabled uncertainty model as the main pipeline,
- preserves the existing code style and reuses the existing high-performance geometry and containment machinery.

## Implementation Constraints

These constraints are part of the plan, not optional preferences.

### Code style and placement

- Do not perform major refactors just because a different file layout would read more cleanly.
- New logic should be inserted near the analogous legacy logic in the existing files.
- New configurables should be added near the beginning of `biopsy_localization_convex_main.py` alongside the existing optimizer, uncertainty, and simulated-biopsy settings.
- New output objects should follow the current object-storage style on the biopsy dictionaries and cohort dataframe dictionaries rather than introducing a new architecture.

### Reuse of existing machinery

- Reuse the existing custom point-in-polygon and containment machinery.
- Reuse the existing CUDA-kernel-driven containment path already configured through `custom_cuda_kernel_type` and related polygon-handling options.
- Reuse the existing uncertainty-generation and transformation machinery rather than writing a second uncertainty system for the new optimizer.

### Gating and coexistence

- Do not disturb the original legacy optimization loop.
- Build a new separately gated target-DIL optimization lane beside it.
- Both lanes should be able to run in the same execution if desired.
- Their outputs should be stored separately so they can coexist without overwriting one another.

### Collaboration protocol

- Every coding pass should be small.
- Before each coding pass, explicitly state which objects will be touched, what will be inserted, where it will be inserted, and why.
- Each pass should be easy to review manually.

### Identifier discipline

- `Bx index` must be treated as the unique biopsy identifier within a `Patient ID`.
- `Bx ID` must not be assumed unique.
- `Ref #` is useful, but it is secondary to `Patient ID` plus `Bx index` for biopsy identity.
- For relative structures, retain the identifiers that uniquely distinguish structure rows, especially `Relative structure ROI`, `Relative structure type`, and `Relative structure index`.
- New dataframes must preserve the identifier columns needed to tie rows back to a unique biopsy and a unique target structure.

## Recommended Methodology Decision

Use `PD_target` as the primary paper endpoint for the new method.

Reason:

- the new optimizer is target-lesion-specific,
- the primary downstream evaluation should match the quantity that was actually optimized,
- `PD_all` remains useful as a comparator and sensitivity analysis.

So the planned outputs are:

- `PD_target`: target-lesion-specific score,
- `PD_all`: legacy any-DIL score,
- an agreement check between the optimizer target score and the final downstream target-specific score.

## Uncertainty Alignment Principle

The target optimizer should not use a separate simplified uncertainty model unless that is explicitly configured.

The intended behavior is:

- if the paper run uses only rigid translations, the target optimizer should be robust to those same rigid translations,
- if biopsy rotations are enabled, the target optimizer should reflect those biopsy rotations,
- if relative DIL or OAR rotations or translations are enabled, the target optimizer should reflect those as well,
- the optimizer still places a straight core at a chosen nominal location, but the objective is a robust score under the same enabled transform family used later in the main pipeline.

This means the new optimization lane should reuse the current uncertainty and transform flow rather than inventing a separate uncertainty tree.

Relevant current configuration anchors in `biopsy_localization_convex_main.py` already exist, including:

- `biopsy_variation_uncertainty_setting`
- `non_biopsy_variation_uncertainty_setting`
- the default translation sigma lists for biopsy, DIL, and OARs
- the default rotation sigma and mu lists for biopsy, DIL, and OARs
- `modify_generated_uncertainty_template`

## Current Anchors In The Code

These are the main objects already in place.

### Top-of-main configuration anchors already exist

In `biopsy_localization_convex_main.py`, the current code already has the style and placement we should follow for new settings. Important nearby anchors include:

- `voxel_size_for_dil_optimizer_grid`
- `num_normal_dist_points_for_biopsy_optimizer`
- `normal_dist_sigma_factor_biopsy_optimizer`
- `optimal_normal_dist_option`
- `bias_LR_multiplier`
- `bias_AP_multiplier`
- `bias_SI_multiplier`
- `bx_sim_locations_dict`
- `simulated_biopsy_length_method`

Any new target-optimizer configurables should be added in that same region, not moved elsewhere.

### Upstream target-lesion metadata already exists

`biopsy_localization_convex_main.py` already assigns per-biopsy target-lesion metadata to each biopsy object:

- `Target DIL by centroid dict`
- `Target DIL by surfaces dict`
- `Nearest DILs info dict`

These are written onto each biopsy structure object in the main patient structure dictionary.

### Legacy optimizer entry point already exists

`biopsy_optimizer.py` currently uses:

- `find_dil_optimal_sampling_position(...)`

This currently optimizes the legacy surrogate based on normal-distribution points contained.

### Existing geometry and transform machinery should be reused

The current repo already has specialized machinery that should be reused rather than replaced:

- custom point-containment utilities in `point_containment_tools`
- the configured custom CUDA point-in-polygon path controlled by `custom_cuda_kernel_type`
- uncertainty handling through `uncertainty_processor` and `uncertainty_file_writer`
- transform generation through `MC_prepper_funcs.generate_transformations(...)`
- biopsy-only transforms through `MC_prepper_funcs.biopsy_only_transformer(...)`
- relative-structure transforms through `MC_prepper_funcs.biopsy_transformer_to_relative_structures(...)`

The new target optimizer should be built on top of this machinery where practical.

### Structure-specific downstream DIL scores already exist

`dataframe_builders.py` already creates per-biopsy, per-structure global tissue statistics from:

- `specific_bx_structure["MC data: compiled sim results dataframe"]`

Those structure-specific global statistics are stored in:

- `Tissue class - Global tissue by structure statistics`

This table already contains one row per biopsy and relative structure, keyed by:

- `Relative structure ROI`
- `Relative structure type`
- `Relative structure index`

### There is already a target-specific extraction helper

`dataframe_builders.py` already has:

- `bx_global_score_to_target_dil_3d_radiomic_features_dataframe_builder(...)`

This function already uses `Target DIL by centroid dict` to filter the structure-level global tissue table down to the target lesion row for each biopsy.

This matters because the downstream target-specific score is already recoverable from the current structure-level outputs.

## Scope Boundaries

### What we will change

- add a new gated target-DIL optimization lane,
- add the target-optimizer-specific metadata objects,
- add exports for `PD_target` and target-optimizer audit values,
- add comparison outputs that allow target-lane and legacy-lane results to coexist.

### What we will not change in the early passes

- the legacy optimization loop itself,
- biopsy geometry creation in `biopsy_creator.py`,
- biopsy transport logic in `biopsy_transporter.py`,
- the legacy any-DIL Monte Carlo tumor score in `MC_simulator_convex.py`,
- the downstream voxel binning helpers,
- the QA repo.

This keeps the risky parts stable while the new lane is built beside the old one.

## Proposed New Objects

The new objects should be explicit, separate, and able to coexist with legacy outputs.

### On each biopsy structure object

Add a new dictionary on each biopsy structure in the upstream master structure dictionary:

- `Target DIL optimization dict`

Proposed contents:

- `Optimization mode`
- `Target source`
- `Target DIL ROI`
- `Target DIL type`
- `Target DIL refnum`
- `Target DIL index`
- `Target optimizer coarse voxel size mm`
- `Target optimizer fine voxel size mm`
- `Target optimizer robust score coarse`
- `Target optimizer robust score fine`
- `Target optimizer retained score`
- `Target optimizer selected location`
- `Target optimizer uncertainty mode`
- `Target optimizer notes`

This should be separate from:

- `MC data: tumor tissue probability`

because that object currently refers to the legacy any-DIL tumor scoring behavior.

### Separate legacy and target deliverables

Do not overwrite the legacy optimizer outputs.

Instead, add separate deliverable objects for the new lane, for example:

- `Target DIL optimization summary dataframe`

and keep the legacy deliverables intact.

Both should be able to exist at the same time.

### Cohort-level export content

The target-lane summary dataframe should include:

- biopsy identifiers,
- simulation type,
- target-lesion identifiers,
- `PD_target`,
- `PD_all`,
- target-optimizer retained score,
- downstream recovered target-specific score,
- agreement delta,
- lane identifiers so legacy and target outputs can be distinguished cleanly.

## Change Inventory And Movement Classification

We should not treat every planned change as the same kind of work.

For this repo, every planned change should be classified into one of four buckets before coding:

- additive no-move change,
- adjacent sidecar wrapper,
- later reposition candidate,
- high-risk no-move block.

This classification is part of the safety protocol.

### Category A: Additive No-Move Changes

These changes add new objects, new exports, new gates, or new comparison artifacts without changing the execution location of legacy logic.

These are the safest early passes.

Current items in this category:

- dedicated per-biopsy `PD_target` audit export from the existing downstream structure-level tables,
- new top-of-main target-lane flags and settings placed beside the current optimizer settings,
- blank storage slots for target-lane outputs on biopsy dictionaries and cohort dataframe dictionaries,
- retained target-lane score export beside downstream recovered `PD_target`,
- dual export of `PD_target` and legacy `PD_all`,
- a comparison-manifest writer for shadow-validation passes.

### Category B: Adjacent Sidecar Wrapper Candidates

These are blocks where we want to preserve the legacy code path exactly, wrap the existing logic with the same body, and run the wrapper beside the legacy block in shadow mode at the same execution point.

These passes do not move the code.

They only prove that:

- the wrapper sees the same inputs,
- the wrapper emits the same outputs,
- the wrapper performs the same mutations,
- the wrapper preserves any needed downstream state.

Current items in this category:

- `MC_prepper_funcs.generate_transformations(...)`,
- `MC_prepper_funcs.biopsy_only_transformer(...)`,
- `MC_prepper_funcs.biopsy_transformer_to_relative_structures(...)`,
- any later callable seam we introduce around structure-side preparation needed to evaluate target-lane robust scores,
- any legacy block whose code body can be preserved verbatim and whose outputs can be compared cleanly through a manifest.

For every adjacent sidecar wrapper pass, the rule is:

- keep the legacy block live,
- keep the wrapped code body identical unless a difference is explicitly justified,
- run the wrapper on copied inputs or copied touched state,
- emit a comparison manifest,
- do not reposition the call in the same pass.

### Category C: Later Reposition Candidates

These are the steps that may eventually need to run earlier or later than they do now, but should not be moved until the equivalent adjacent sidecar wrapper has already been validated.

Current items in this category:

- uncertainty preparation if exact draw reuse for the optimizer requires access to uncertainty-attached objects before the legacy simulation-prep point,
- transform sampling if exact draw reuse for the optimizer requires sampled transforms before the legacy simulation-prep point,
- any structure-side dilation or alternate-geometry preparation that the target-lane robust score must share with the downstream Monte Carlo path,
- any optimizer-adjacent callable seam that currently depends on state produced later in the pipeline.

For every reposition pass, the rule is:

- first validate the wrapped block at the original location,
- then audit all required inputs at the proposed new location,
- then compare the input-state manifest between old location and new location,
- only then move the call.

### Category D: High-Risk No-Move Blocks

These are blocks that should remain untouched while we are still building confidence in the wrapper and manifest process.

Current items in this category:

- the legacy optimizer loop itself,
- the benchmark-sensitive structure-side geometry and containment path in `MC_simulator_convex.py`,
- biopsy geometry creation in `biopsy_creator.py`,
- biopsy transport logic in `biopsy_transporter.py`,
- the existing downstream any-DIL Monte Carlo scoring behavior.

These blocks may be read, wrapped beside their current execution point, or compared through manifests, but they should not be rewritten or moved in the early passes.

## Shadow Validation Protocol

Every sidecar pass should produce a machine-readable comparison manifest.

At minimum, the manifest should record:

- block name,
- execution location,
- patient and structure identifiers relevant to the block,
- compared object path or output name,
- equality status,
- shape and dtype when arrays are involved,
- exact mismatch summary when equality fails,
- notes on whether row order or key order is part of the contract.

The purpose of the manifest is to let us inspect equality after one real run without guessing.

This protocol applies before any legacy block is removed and before any validated block is repositioned.

## Recommended Execution Order

The practical order should be:

1. Add the comparison-manifest machinery.
2. Complete the additive no-move passes.
3. Shadow-wrap `generate_transformations(...)` beside the legacy call and validate exact equality.
4. Shadow-wrap `biopsy_only_transformer(...)` beside the legacy call and validate exact equality.
5. Shadow-wrap `biopsy_transformer_to_relative_structures(...)` beside the legacy call and validate exact equality.
6. Decide whether the target-lane still requires any call-site repositioning after those validated seams exist.
7. If repositioning is still required, move one validated block at a time with a separate state-input validation pass.

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

Extend the existing target-score extraction path so it produces a dedicated per-biopsy audit dataframe with explicit `PD_target` columns.

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

If straightforward, also include:

- `PD_all`

### Why this phase comes first

This is the smallest auditable insertion and it proves the target-specific downstream score can already be recovered without touching optimization behavior.

### Validation

- exactly one target-specific row per biopsy,
- no fraction mixing in the target mapping,
- no missing target rows for biopsies that have target metadata,
- manual spot-checks using `Patient ID` plus `Bx index` and the target structure identifiers.

## Phase 2: Add Separate Gating And Storage For The New Lane

### Objective

Introduce the new lane in a way that matches the current code style and does not disturb the legacy optimizer path.

### Files / objects to modify

Upstream repo:

- `biopsy_localization_convex_main.py`

Primary objects:

- the top-of-main optimizer configuration block,
- `bx_sim_locations_dict`,
- the existing optimizer invocation region,
- the biopsy structure dictionaries that store simulation outputs.

### Exact change

Add new configurables near the current optimizer block at the beginning of `biopsy_localization_convex_main.py`, for example:

- target-lane enable flags,
- whether legacy and target lanes both run,
- target-lane coarse and fine optimizer voxel sizes,
- target-lane target-source choice,
- target-lane uncertainty-evaluation mode.

The exact variable names can be chosen to match the local naming style, but they should live beside the current optimizer and simulated-biopsy settings.

Also add blank storage slots for the new target-lane outputs into the biopsy dictionaries and cohort dataframe dictionaries without removing or renaming legacy fields.

### Validation

- legacy-only run still works,
- target-only run can be disabled cleanly,
- both-lanes-enabled configuration does not overwrite one lane with the other,
- no downstream consumers break from the presence of new fields.

## Phase 3: Build The Target-DIL Optimizer Path Beside The Legacy Loop

### Objective

Implement a new target-specific optimizer path without changing the legacy optimizer loop.

### Files / objects to modify

Upstream repo:

- `biopsy_localization_convex_main.py`
- `biopsy_optimizer.py`

Existing anchor objects:

- `find_dil_optimal_sampling_position(...)`
- the optimizer call site in `biopsy_localization_convex_main.py`
- `simulated_biopsy_length_method = 'match real'`

### Exact change

Create a new gated target-optimizer path beside the legacy optimizer path.

The new path should:

- bind each optimization run to one target DIL only,
- use coarse-to-fine evaluation if configured,
- reuse the current containment machinery rather than introducing a separate simplified geometry engine,
- preserve the existing local coding style by living near the current optimizer logic.

### Important wiring rule

The target lesion for the new lane must come from the same biopsy-level target metadata already stored on the biopsy object.

Do not create a second disconnected target-selection system.

### Important style rule

Do not rewrite the legacy loop into a generic framework just to support both paths.

Add a second gated path instead.

### Validation

- legacy optimizer behavior is unchanged when only legacy is enabled,
- target-lane results appear only when target lane is enabled,
- both lanes can run in the same execution,
- stored outputs remain separated.

## Phase 4: Make The Target Optimizer Reflect The Same Enabled Uncertainty Model

### Objective

Ensure the target-optimizer score reflects the same enabled uncertainty family as the main pipeline for that run.

### Files / objects to modify

Upstream repo:

- `biopsy_localization_convex_main.py`
- `biopsy_optimizer.py`
- only the minimum additional supporting objects needed to reuse the transform machinery

### Exact change

Wire the target-lane score evaluation so it uses the same enabled transform assumptions already configured in the main pipeline.

That means:

- translation-only runs should stay translation-only,
- biopsy-rotation-enabled runs should include biopsy rotations,
- relative-structure rotation and translation settings should be reflected when they are enabled.

The new lane should reuse the existing uncertainty-generation and transform application path where practical rather than defining a separate optimizer-only uncertainty implementation.

### Validation

- when only translations are enabled, the target lane behaves as a translation-robust optimizer,
- when rotations are enabled, the target lane reflects those rotations,
- the lane respects the current top-of-main uncertainty settings rather than hardcoding a reduced model.

## Phase 5: Retain A Final Target-Lane Score And Compare It To Downstream `PD_target`

### Objective

Retain the final target-lane optimized score and compare it to the downstream recovered target-specific score.

### Files / objects to modify

Upstream repo:

- `biopsy_localization_convex_main.py`
- `dataframe_builders.py`

### Exact change

After the target-lane optimizer selects the final candidate, retain that final target-lane score and export it beside the downstream recovered `PD_target`.

This does not require overwriting legacy `PD_all` behavior.

### Expected relationship

The retained optimizer target score and the downstream recovered target-specific score should be extremely close if the same target lesion and same effective uncertainty definition are being evaluated.

They are not required to match the legacy `PD_all` score because that remains a different quantity.

### Validation

- agreement delta is small and auditable,
- rows can be traced back to unique biopsies and unique target lesions,
- legacy `PD_all` remains available as comparator.

## Phase 6: Export Both `PD_target` And `PD_all`

### Objective

Export both metrics clearly so the paper can use `PD_target` as primary and `PD_all` as comparator.

### Files / objects to modify

Upstream repo:

- `dataframe_builders.py`
- the current CSV-writing or export-registration path that persists cohort tables

### Exact change

Add or extend a clean cohort-level target-lane table with these columns:

- biopsy identifiers,
- simulation type,
- target lesion identifiers,
- `PD_target`,
- `PD_all`,
- target-lane retained optimizer score,
- downstream recovered target score,
- target-score agreement delta,
- lane flags indicating which optimizer lanes were run.

### Validation

- one row per biopsy per lane where expected,
- legacy outputs still present,
- `PD_target` and `PD_all` both populated where expected,
- identifier columns retained for unambiguous joins.

## Phase 7: Downstream Repo Changes Only After Upstream Output Schema Is Stable

### Objective

Keep the QA repo frozen until the upstream export format is finalized.

### First-pass decision for the QA repo

No code changes there in the first implementation passes.

That is intentional.

The current downstream and QA code should not be touched until the upstream target-score export exists and is validated.

## Audit Rules For Every Pass

Each pass should be small, auditable, and validated before moving on.

### Pass order

1. target-score extraction only,
2. separate gating and storage,
3. target-lane optimizer path,
4. uncertainty alignment,
5. score retention and agreement export,
6. export cleanup,
7. downstream analysis updates.

### Rules

- do not change the legacy lane while building the new lane,
- do not refactor large surfaces just to make the code read more cleanly,
- do not replace the existing geometry or containment machinery,
- do not change downstream voxelization while building the target lane,
- do not combine multiple conceptual changes in one pass,
- every pass must end with a concrete agreement check, row-count check, or lane-isolation check.

## Practical Notes

### Naming

Use explicit names where useful, but keep them stylistically compatible with the existing repo.

The important part is separation of legacy and target-lane objects, not forcing a new naming scheme across the codebase.

### Equality expectation

The target-lane optimizer score and the downstream recovered `PD_target` should be nearly identical only if they use:

- the same target lesion,
- the same biopsy identity,
- the same effective uncertainty model,
- the same identifier joins.

That is why identifier discipline and uncertainty alignment are part of the plan.

## First Pass To Execute

The safest first code pass is still:

1. upstream only,
2. no optimizer changes,
3. expose a dedicated per-biopsy `PD_target` audit dataframe from the existing structure-level score outputs,
4. verify one target row per biopsy and no fraction mixing.

After that, the next pass should be the smallest possible insertion of separate gating and blank storage for the new lane in `biopsy_localization_convex_main.py`, without yet changing the legacy optimizer loop.

## Concrete Concurrent-Module Refactor For The New Lane

This section answers the practical question directly.

Yes, the refactor can be made to work regardless of the configured simulated-biopsy-length method, provided we split simulated biopsy length determination from simulated biopsy geometry creation.

That is the key design decision.

The current code couples these two things inside the simulated-biopsy block, but the transform sampler does not need the full simulated biopsy geometry. It only needs the biopsy-length scalar already stored under:

- `Reconstructed biopsy cylinder length (from contour data)`

So the safe refactor is not to move simulated biopsy creation earlier.

The safe refactor is to introduce a small earlier scalar-only prepass that computes and stores the nominal biopsy length needed for MC preparation, then let the existing later simulated-biopsy construction block consume that stored scalar.

Because the current real-biopsy block and simulated-biopsy block both sit after the legacy optimizer in the live file order, this prepass must be a new earlier callable helper.

It should not move the full geometry-building blocks.

It should only materialize the minimal scalar state needed by transform preparation.

### Why this works for every current simulated-length mode

Each current mode can be resolved before simulated biopsy geometry is built.

#### `full`

This is already determined directly from `biopsy_needle_compartment_length`.

No optimizer dependency exists.

#### `real normal`

This only depends on:

- the global mean of real biopsy lengths,
- the global standard deviation of real biopsy lengths,
- the NumPy random draw used for this biopsy.

Those inputs are available after the real-biopsy preprocessing and real-length aggregation pass.

No optimizer dependency exists.

Because this is a moved NumPy draw, the prepass should own that draw explicitly and later simulated-biopsy code should consume the stored scalar rather than drawing again.

#### `real mean`

This only depends on the global mean real biopsy length.

No optimizer dependency exists.

#### `match real`

This depends on:

- the patient-specific map from DIL refnum to real-core lengths,
- the simulated biopsy's relative structure type,
- the simulated biopsy's relative structure refnum.

Those are all available before simulated biopsy geometry is built.

No optimizer dependency exists for the length scalar itself.

### What still depends on the optimizer

Only the simulated biopsy placement for the optimal simulated lane still depends on the legacy optimizer output.

That dependency remains:

- legacy optimizer writes `Biopsy optimization: Optimal biopsy location dataframe` on each DIL,
- `biopsy_transporter.biopsy_transporter_optimal(...)` reads that dataframe,
- simulated optimal biopsy geometry is then transported to that location.

So the correct split is:

- simulated biopsy length can move earlier,
- simulated optimal biopsy placement cannot move ahead of the optimizer.

That is not a contradiction because transform sampling needs the former, not the latter.

## New Minimal State To Introduce

To avoid overloading the existing legacy length field too early, the refactor should introduce one neutral prepass field on biopsy dictionaries, for example:

- `MC prep biopsy cylinder length`

Optional companion metadata field:

- `MC prep biopsy cylinder length method`

Reason:

- for real biopsies, the scalar can be derived early from raw contour z-slice centroids without running the whole later geometry block,
- for simulated biopsies, the scalar may exist before the later geometry is actually constructed,
- the existing field name `Reconstructed biopsy cylinder length (from contour data)` remains semantically tied to the later legacy geometry block.

The transform-generation code can then be updated minimally to use:

1. `MC prep biopsy cylinder length` if present,
2. otherwise fall back to `Reconstructed biopsy cylinder length (from contour data)`.

That keeps the refactor incremental and avoids lying about where the scalar came from.

## Recommended New Module Layout

Because this is a true concurrent lane and not just a few extra helper functions, a small dedicated subfolder is justified.

It should still stay shallow and stylistically compatible with the existing repo.

Recommended location:

- `python_files_dcm_meta_based/target_dil_optimizer/`

Recommended initial contents:

- `__init__.py`
- `length_prep.py`
- `target_lane.py`
- `target_score_helpers.py`

### What each file should own

#### `length_prep.py`

Purpose:

- compute and store the earliest safe biopsy-length objects needed by the new lane and by early transform sampling,
- compute and store the earliest safe biopsy-to-target routing objects needed for one-to-one parent matching,
- do not build biopsy geometry,
- do not move or replace the later biopsy-construction logic.

Recommended functions:

- `collect_real_biopsy_length_statistics(...)`
- `collect_real_biopsy_target_routes(...)`
- `assign_simulated_biopsy_nominal_lengths(...)`

Stored outputs on biopsy dictionaries should stay in the current object-storage style, for example:

- `Simulated biopsy nominal length`
- `Simulated biopsy nominal length method`
- `Matched real biopsy index`
- `Matched real biopsy ROI`
- `Matched real biopsy ref #`
- `Matched real biopsy exists bool`
- `Unmatched DIL extra simulated bool`

The later simulated-biopsy block can then copy that stored value into:

- `Reconstructed biopsy cylinder length (from contour data)`

after geometry is actually built, or can directly reuse the stored scalar to preserve equality with the earlier prepass.

To minimize downstream risk, these new parent-match fields should first be exported only through new `v2` or target-lane dataframes rather than being pushed immediately into broad legacy CSV schemas.

#### `target_lane.py`

Purpose:

- run the new target-specific optimization lane beside the legacy optimizer,
- keep all target-lane storage separate from legacy optimizer storage,
- consume already prepared transforms or transform-ready state rather than regenerating uncertainty independently.

Recommended functions:

- `initialize_target_dil_optimization_slots(...)`
- `run_target_dil_optimizer_lane(...)`
- `store_target_dil_optimizer_outputs(...)`

This file should not replace `biopsy_optimizer.py`.

It should sit beside it as a new lane owner.

#### `target_score_helpers.py`

Purpose:

- extract target-specific score views,
- build agreement tables between retained target-lane score and downstream recovered `PD_target`,
- keep the target-lane dataframe logic isolated from the legacy dataframe builders until the schema stabilizes.

Recommended functions:

- `build_target_lane_summary_dataframe(...)`
- `build_target_lane_agreement_dataframe(...)`

### Why a small subfolder is better than one more giant file

This is one of the few cases where a new subfolder is the cleaner choice even under the "do not refactor broadly" constraint.

Reason:

- the new lane is conceptually separate,
- it will need a small number of helper functions,
- pushing all of that into `biopsy_localization_convex_main.py` or `biopsy_optimizer.py` would make the concurrent design harder to review,
- keeping orchestration in main while putting lane-specific helpers in one shallow subfolder preserves both readability and the existing style.

The main file should still remain the orchestration surface.

The new folder should only hold the new lane's helpers.

## RNG Isolation Requirement

This is the main technical constraint that appears once transform sampling is moved earlier.

The legacy optimizer currently uses global CuPy randomness for its internal normal-distribution sampling.

The current transform-generation helpers also use global CuPy randomness.

So if transform generation is moved ahead of the legacy optimizer without isolating RNG usage, the legacy optimizer will no longer see the same CuPy random stream.

That would silently change legacy behavior.

The clean fix is:

- add an optional RNG argument to the transform-sampling helpers in `cupy_functions.py`, defaulting to the current global behavior,
- let the moved transform-preparation path use its own dedicated CuPy RNG object,
- leave the legacy optimizer on the current global CuPy RNG unless and until we intentionally change that contract.

For the `real normal` simulated-length mode, the same principle applies on the NumPy side:

- the early length-prep helper should own the NumPy draw once,
- the later simulated-biopsy block should reuse the stored scalar,
- do not redraw inline later.

This is not a replay design.

This is isolation of moved random draws so the concurrent refactor does not perturb unrelated legacy randomness.

## Exact Execution Order For The Clean Refactor

The safe sequence is the following.

### Stage 1: Add a new scalar-only real-biopsy length prepass before the optimizer

Do not move the full current real-biopsy geometry block.

Instead, extract the minimum logic needed to compute a real biopsy length from each real biopsy's raw contour z-slice list.

This helper should store:

- `MC prep biopsy cylinder length` for each real biopsy,
- global real-biopsy length mean,
- global real-biopsy length standard deviation,
- patient-and-DIL keyed real-length lookup for `match real`.

This prepass should not do interpolation, point-cloud creation, reconstructed-biopsy creation, or structure-volume calculation.

### Stage 2: Add an early biopsy-to-target routing prepass before simulated construction

If the future design requires one simulated set per real biopsy, then some biopsy-to-target routing must exist before simulated biopsy expansion.

This does not mean the full late `Target DIL by centroid dict` block has to move earlier.

It means a smaller sidecar prepass should resolve and store only the routing state needed to decide:

- which DIL each real biopsy is matched to,
- which real biopsies belong to each DIL,
- whether a DIL has any matched real biopsy at all,
- which parent real biopsy a one-to-one simulated biopsy should inherit its length from.

Recommended design direction for the next constructor pass:

- remain DIL-driven as the primary creation regime,
- use matched real-biopsy identity as a secondary expansion key,
- create one simulated set per matched real biopsy for each eligible DIL,
- optionally, behind a separate gate, also create an additional unmatched-DIL simulated set when a DIL has no matched real biopsy.

This preserves the ability to keep lesion-driven extras while still supporting the new one-to-one counterfactual design.

That unmatched-DIL extra path should not hardwire length behavior.

It should have its own configurable unmatched-extra length method, for example:

- `full`
- `real normal`
- `real mean`
- other explicitly allowed fallback modes if later needed

### Stage 3: Add an early simulated-length prepass before the optimizer

Walk the simulated biopsy objects and store one scalar nominal length per biopsy according to the configured `simulated_biopsy_length_method`.

Store that scalar in:

- `MC prep biopsy cylinder length`

This helper must not create point clouds, centroid lines, interpolations, or transported biopsy geometries.

It only stores the scalar nominal length and metadata about how it was chosen.

For the future one-to-one design, `match real` should be reinterpreted as:

- match the parent real biopsy length exactly for simulated biopsies that have a matched real parent,
- for unmatched-DIL extras, use a separate configurable fallback length method rather than reusing parent-match semantics.

### Stage 4: Move only the uncertainty attachment earlier

Keep the current uncertainty-loading logic, but factor it into a callable helper that can run before the optimizer.

Do not create a second uncertainty system.

Do not move unrelated later simulation consumers in the same pass.

### Stage 5: Sample transforms earlier with an isolated CuPy RNG

At this point the required inputs for transform generation exist:

- every structure has `Uncertainty data`,
- every biopsy has `MC prep biopsy cylinder length`.

Now the existing transform sampler can be moved to this earlier point or wrapped through a new small preparation call.

But this earlier call must use an isolated CuPy RNG so the legacy optimizer's current random stream is not perturbed.

This is the moment where one-time transform generation and reuse becomes safe.

### Stage 6: Run the legacy optimizer where it already lives

Do not move the legacy optimizer in the same pass.

Its outputs are still needed later for optimal simulated-biopsy placement.

### Stage 7: Run the new target optimizer as a concurrent lane

Add a new gated call beside the legacy optimizer call in main.

This new call should live immediately adjacent to the legacy optimizer region, not far away in a different phase of the file.

The concurrent target lane should:

- read target metadata already assigned to the biopsy,
- evaluate candidate biopsy locations against one target DIL only,
- consume the already prepared transform data or transform-ready state,
- write its own target-lane output objects without touching legacy optimizer objects.

### Stage 8: Run the existing real-biopsy geometry block later

Keep the current full real-biopsy geometry block later in the pipeline.

It should continue to build all the legacy geometry and volume products it already builds.

As an audit check, it can confirm that the later legacy length matches the earlier `MC prep biopsy cylinder length` for each real biopsy.

### Stage 9: Run the existing simulated-biopsy geometry block later

Keep the current simulated-biopsy geometry construction and transport block later in the pipeline.

Change only the length source.

Instead of recomputing the simulated biopsy nominal length inline, that block should read the earlier stored scalar from the biopsy dictionary.

For the optimal simulated type, it should still call `biopsy_transporter.biopsy_transporter_optimal(...)` after the legacy optimizer has produced its location dataframe.

### Stage 10: Reuse stored transforms downstream

The downstream transform consumers should use the already stored transform arrays.

They should not trigger a second generation path.

## Minimal Main-File Wiring Changes

The minimum clean changes in `biopsy_localization_convex_main.py` are:

1. add new top-of-main flags for the target lane and for early transform preparation,
2. add one small call to a real-biopsy scalar-length prepass before the optimizer,
3. add one small call to an early biopsy-to-target routing helper that stores parent-match state without moving the full late metadata block,
4. add one small call to `assign_simulated_biopsy_nominal_lengths(...)` before the optimizer,
5. factor uncertainty attachment into a callable helper and invoke it before transform preparation,
6. relocate or wrap the existing transform generation call so it happens after uncertainty plus the new length prepass,
7. give the moved transform-sampling path an isolated CuPy RNG,
8. add one new adjacent call to `run_target_dil_optimizer_lane(...)` beside the legacy optimizer block,
9. change the later simulated-biopsy block to consume the stored nominal length scalar instead of recomputing it inline.

That is the whole refactor shape.

It is targeted, auditable, and does not require a broad rewrite.

## What should not be done

To keep this clean, avoid these directions:

- do not create a second independent uncertainty sampler for the target lane,
- do not rebuild the target lane inside `biopsy_optimizer.py` by heavily generalizing the legacy optimizer first,
- do not move simulated optimal biopsy transport ahead of the legacy optimizer,
- do not regenerate transforms later for the target lane if they were already generated once,
- do not hide the new lane deep inside generic wrappers that make the call order harder to inspect.

## Recommended First Refactor Pass After The Current Planning Pass

If coding starts, the first implementation pass after planning should be:

1. create `python_files_dcm_meta_based/target_dil_optimizer/__init__.py`,
2. create `python_files_dcm_meta_based/target_dil_optimizer/length_prep.py`,
3. add the scalar-only real-biopsy length prepass that fills `MC prep biopsy cylinder length`,
4. add `assign_simulated_biopsy_nominal_lengths(...)` that supports all current length modes and fills the same field for simulated biopsies,
5. wire those helpers into main before the optimizer without changing optimizer behavior yet,
6. confirm that the stored simulated nominal lengths exactly match the lengths the legacy inline code would have selected,
7. confirm that later real-biopsy geometry reconstruction reproduces the same real-biopsy length scalar.

Only after that should the transform-generation call be repositioned.
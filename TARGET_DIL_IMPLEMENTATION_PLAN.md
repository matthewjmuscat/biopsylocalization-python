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
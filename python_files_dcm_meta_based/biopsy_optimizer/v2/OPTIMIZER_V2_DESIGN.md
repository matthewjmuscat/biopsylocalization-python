# Optimizer V2 Design

## Status

The stale `python_files_dcm_meta_based/target_dil_v2/` folder was not the active v2 implementation area.

It contained only `__pycache__` files and has been removed.

The active v2 area remains:

- `python_files_dcm_meta_based/biopsy_optimizer/v2/`

## Purpose

Optimizer v2 should select the top `k` biopsy centroid positions that maximize a target-only stochastic objective under the same enabled transform model used by downstream MC.

This is not the same problem as guidance-map hole ranking.

Guidance-map top-`k` holes are downstream template-hole choices that reach an already optimized sampling position.

Optimizer v2 top-`k` is upstream ranking of biopsy centroid candidates in prostate space.

## Scaling Risk

Optimizer v2 is not just a new scoring path.

If it is enabled alongside legacy v1 simulated-core generation, or if it introduces a new family of simulated cores without disabling the old path, the effective biopsy count can grow materially.

That has at least three consequences:

- more per-biopsy preprocessing state to hold in memory,
- more MC and dataframe output volume,
- more risk that the current one-run-in-one-process execution model hits memory limits before the scientific logic is wrong.

This should be treated as a first-class design constraint, not a late cleanup item.

## Hard Decisions

- The optimizer consumes planning-frame biopsy geometry and planning-frame sampled biopsy points.
- The optimizer does not regenerate a different special-purpose biopsy sampling scheme.
- The optimizer scores against the target DIL only.
- Candidate points outside the target DIL are removed immediately.
- Trialwise candidate evaluation is batched in a single containment call per chunk and stage.
- There is no Python loop over MC trials.
- Hot-path geometry is arrays, not point-cloud objects.
- Point clouds remain debug and visualization tools only.
- Relative-structure dilations must be generated and stored early enough that optimizer v2 and downstream MC can reuse the same transform bank.

## Readiness Snapshot

The repo now contains the first executable optimizer-v2 library slice.

Completed implementation already in code:

1. planning-frame simulated-biopsy geometry is built upstream and stored on each simulated biopsy,
2. planning-frame simulated-biopsy sampled points are already built upstream,
3. target-only fixed-lattice candidate generation and immediate target-interior pruning already exist,
4. shared transform-bank accessors, localization, and containment seams now exist outside optimizer v2,
5. chunked target-only scoring, staged search, winner resolution, and winner-only downstream-comparable validation are implemented in `biopsy_optimizer/v2`,
6. stage-boundary replayable render jobs and carry-down in-memory output dataframe builders are implemented,
7. downstream post-realization biopsy processing is modular enough that optimizer-v2 can hand off to existing realization and downstream MC seams once the live integration layer is wired.

Remaining implementation work is therefore concentrated in:

1. live family declaration and creation in the main simulated-biopsy flow,
2. planner and transport integration for a v2 optimizer family,
3. legacy export-block wiring for the new sidecar outputs,
4. downstream shared-bank reuse validation on real cases,
5. memory-safe activation policy when v2 materially increases simulated biopsy count.

## Canonical Reuse Surfaces

### Planning geometry and biopsy sampling

- `preprocessing.biopsy_processing.simulated_biopsy_planner.build_simulated_biopsy_planning_state(...)`
- `preprocessing.biopsy_processing.simulated_biopsy_planner.build_simulated_biopsy_planning_sample_state(...)`
- `sampling.biopsy_point_sampler.sample_biopsy_points_from_reconstructed_biopsy_model_dict(...)`

### Transform generation

- `MC_prepper_funcs.generate_transformations(...)`
- `cupy_functions.MC_simulator_all_structs_dilations_generator_cupy(...)`
- `cupy_functions.MC_simulator_all_structs_rotations_generator_cupy(...)`
- `cupy_functions.MC_simulator_shift_all_structures_generator_cupy(...)`
- `cupy_functions.MC_simulator_shift_biopsy_structures_uniform_generator_cupy(...)`

### Biopsy self-transforms

- `MC_prepper_funcs.biopsy_only_transformer(...)`

The optimizer should reuse the same underlying transform math, even if it calls a narrower helper surface instead of the full downstream writeback path.

### Localization-transformer recommendation

Optimizer v2 should not invent its own localization or uncertainty-transform math.

The correct reusable surface is a narrow localization-transformer layer built from the existing transformation math.

That reusable layer should:

1. consume the shared transform bank,
2. apply biopsy self-transforms,
3. apply the corresponding relative-structure transforms,
4. return aligned array batches ready for the containment mother function,
5. avoid downstream writeback side effects in the hot path.

This keeps optimizer v2 and downstream stochastic MC on the same uncertainty model even as the uncertainty contract evolves.

That helper should not live inside `biopsy_optimizer/v2` if its contract is genuinely shared.

If the same transformer logic is needed by optimizer v2 and non-optimizer downstream MC, it should live in a shared module near the existing transformation and containment surfaces, with optimizer v2 importing it rather than owning it.

### Relative-structure transforms

- `MC_prepper_funcs.rotate_biopsy_to_relative_structure_points_vectorized(...)`
- `MC_prepper_funcs.translate_biopsy_to_relative_structure_points(...)`

### Relative-structure dilation generation

- `polygon_dilation_helpers_numpy.convert_to_2d_array_and_indices_numpy(...)`
- `polygon_dilation_helpers_numpy.generate_dilated_structures_parallelized(...)`
- `polygon_dilation_helpers_numpy.reconstruct_list_from_2d_array(...)`

### Containment engine

- `custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.custom_point_containment_mother_function(...)`
- `custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.create_containment_results_dataframe_type_2I(...)`
- `custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.create_containment_results_dataframe_type_2II(...)`

### Probability compilation

- existing `Total successes`, `Binomial estimator`, standard-error, and CI formulas from `MC_simulator_convex.py`

Only target-DIL probability compilation is required in optimizer v2.

Sum-to-one tissue-class compilation is not required for the first implementation.

## Required Upstream Staging Change

The optimizer cannot sample its own private transform draws if the goal is agreement with downstream MC.

The required design is:

1. Generate the full transform bank once.
2. Store it on the same structure objects used later by downstream MC.
3. Let optimizer v2 consume prefixes or slices of that stored bank.
4. Let downstream MC consume the same bank later.

The bank-size rule should be:

1. large enough to satisfy the largest effective optimizer-v2 stage prefix that may actually be used,
2. large enough to support any explicit downstream-comparable winner rescore at `N_stochastic_tissue`,
3. treated as one shared ceiling rather than regenerated separately for optimizer and downstream MC.

With final-winner tie-break rescoring enabled, item 1 means more than the raw stage-C trial count.

It must include the largest score-based tie-break prefix that may actually be requested before any geometric fallback is allowed.

So the practical bank-generation rule is:

1. compute the highest optimizer-side prefix implied by the stage-C trial count plus the configured tie-break rescore attempts,
2. compare that against any requested downstream-comparable winner rescore count,
3. generate one shared bank large enough for the larger of those two values.

This must include:

- biopsy self translations,
- biopsy self rotations,
- biopsy self dilations,
- biopsy needle-compartment shifts when enabled,
- target DIL rigid transforms,
- target DIL dilations,
- any additional enabled relative-structure transforms that become part of the contract later.

The optimizer only needs the target DIL subset for scoring, but the transform generation step should still be shared and upstream so that the downstream MC run reuses the same draws.

## Arrays, Not Point Clouds

In the hot path, "cloud" means arrays of point coordinates.

The hot-path objects should be shaped arrays such as:

- sampled biopsy points: `(num_points, 3)`
- trialwise biopsy samples: `(num_trials, num_points, 3)`
- chunked candidate trial samples: `(num_candidates_in_chunk * num_trials, num_points, 3)` conceptually, or an equivalent 3D batched representation

Open3D point clouds should not be part of the optimization loop.

They are too expensive and are only useful for inspection.

## Throughput Packing Principle

The active containment path is already architected as a batched GPU call.

For one mother-function invocation:

- the prepper builds one batched nearest-z structure for the full 3D input batch,
- valid points are flattened,
- the optimized CUDA kernel launches once across all valid points in that batch.

That means the practical throughput target is not "keep trial count low" by itself.

The practical throughput target is:

- pack as much combined point-work as possible into each mother-function call,
- while staying below the memory limit.

For optimizer-v2 implementation, the effective batch budget should be treated as:

- `num_candidates_in_chunk * num_trials_in_stage * num_biopsy_sample_points`

with additional overhead from:

- nearest-z prep arrays,
- valid masks,
- flattened valid-point arrays,
- stacked target-structure geometry arrays.

So the correct implementation instinct is:

- reduce the number of mother-function calls,
- increase packed work per call up to the safe memory limit,
- change `N` and candidate chunk size together rather than treating `N` alone as the cost driver.

This does not mean that `1000` trials is literally free relative to `100`.

It means that, in this architecture, larger `N` can be much cheaper than CPU intuition suggests if it allows fewer total calls and still fits in memory.

The practical limit is still the combined memory footprint of:

- candidate chunks,
- transform-bank slices,
- staged biopsy sample arrays,
- target-structure geometry batches,
- and all downstream retained results that survive past the scoring stage.

So optimizer-v2 design should explicitly separate:

- throughput packing inside one scoring chunk,
- and whole-run memory growth from increasing biopsy count and retained outputs.

## Search Region

The initial search region is the target DIL interior.

For the first implementation, the candidate lattice should follow the verified v1 default spacing.

That means:

- use one initial candidate lattice with `1.0` mm spacing,
- prune that lattice immediately to target-DIL-interior points,
- keep those surviving coordinates fixed while the trial budget increases across stages.

This matches the current v1 configuration, where `voxel_size_for_dil_optimizer_grid = 1` in main and the optimizer builds one lattice before pruning to the DIL interior.

The immediate prune rule is:

- generate a candidate lattice,
- remove every candidate outside the target DIL,
- continue only with surviving target-interior candidates.

This means the first containment call is a cheap pruning pass on lattice points, not the full stochastic objective.

The correct containment surface for that pass is the existing CUDA mother function plus `create_containment_results_dataframe_type_2I(...)`.

## Coarse-To-Fine Search

The optimizer should not evaluate every surviving candidate at full MC budget immediately.

For the first implementation, coarse-to-fine should mean statistical refinement, not geometric multi-resolution.

In other words:

- build one initial `1.0` mm candidate lattice,
- prune to points inside the target DIL,
- start with a small trial prefix,
- repeatedly prune survivors as `N` increases,
- do not regenerate smaller local lattices by default.

Local spatial refinement around top survivors is still a valid later extension, but it should not be the default first-pass behavior.

The recommended first-pass strategy is therefore a staged coarse-to-fine search with three knobs.

## The Three Knobs

The stage schedule should be controlled by three independent knobs:

1. Initial candidate spacing.
2. Trial budget per stage.
3. Survivor budget per stage.

Concrete interpretation:

1. Initial candidate spacing:
   one fixed search lattice spacing used to generate the initial candidate pool.

2. Trial budget per stage:
   small prefix of the shared transform bank for early ranking, then larger prefixes for later ranking.

3. Survivor budget per stage:
   how many candidates survive each stage.

The important point is that these are separate controls.

Do not collapse them into one "resolution" number.

For the first implementation, there is intentionally no fourth knob for local sub-lattice refinement.

That can be added later if the fixed `1.0` mm lattice proves too coarse.

## Nested Trial Prefix Rule

Each finer stage should reuse a larger prefix of one shared transform bank, not draw fresh transforms.

Example:

- coarse stage uses trials `0:10`
- fine stage uses trials `0:40`
- confirmation stage uses trials `0:160`

This gives two benefits:

- no disagreement between optimizer and downstream MC due to different random draws,
- stable stage-to-stage ranking because finer stages refine the same uncertainty sample rather than replacing it.

There is also a throughput benefit:

- when the candidate set has already been reduced, later stages should increase `N` aggressively up to the safe combined batch budget rather than making many small calls.

## Recommended Stage Schedule

The first implementation should use three stages.

## Auto-Packed Stage Budget Recommendation

The optimizer should not throw away explicit stage semantics and replace them with one hardware-dependent chunk-size rule.

A pure "whatever fits in one grandmother chunk" schedule would make:

1. stage meaning opaque,
2. run-to-run behavior depend too strongly on GPU memory availability,
3. validation and cross-run comparisons harder.

The better design is:

1. keep stage structure explicit,
2. treat the user-declared stage trial counts as minimum prefixes,
3. allow the chunk executor to raise the effective prefix when doing so improves batch utilization,
4. keep an explicit ceiling or bound so stage meaning does not become hardware-defined,
5. optionally support an auto mode where each stage declares a min/max prefix and the executor chooses the largest safe prefix within that bound.

So the recommendation is not "hardcoded stage counts forever".

The recommendation is:

1. explicit stage minimums remain the canonical contract,
2. chunk-capacity-aware auto-packing can be added as a controlled option above those minimums,
3. any auto-packed effective trial count must be written to the candidate and ranking manifest so results stay auditable.

## Whole-Dataset Scaling Strategy

Even if candidate-trial scoring is chunked correctly, larger cohorts or multiple simulated-core families can still push the whole run over memory limits.

The long-run design should therefore support at least one, and preferably both, of these surfaces:

### 1. Dataset chunking

Split a larger cohort into smaller execution batches that can be run independently.

Examples:

- by patient,
- by patient-fraction,
- by configured case batches.

This is the simplest way to cap peak memory when the dataset grows.

### 2. Output stitching

Provide a reliable post-run stitching layer that can merge outputs from multiple chunked runs into one cohort-level result surface.

This matters because dataset chunking is only operationally useful if the downstream outputs can be recombined without ad hoc manual work.

The stitching surface should be explicit about:

- which outputs are safe to concatenate directly,
- which outputs require grouped recomputation of cohort statistics,
- which outputs should remain chunk-local only,
- and which manifest fields prove that the stitched runs came from compatible configs.

The likely end state is to support both:

- chunked execution to stay inside memory limits,
- stitched downstream aggregation to recover one cohort view.

Optimizer-v2 design should stay compatible with that future from the start.

Concretely, that means avoiding hidden assumptions that one process sees the whole cohort at once when emitting run manifests, per-biopsy identifiers, or ranked-candidate outputs.

### Stage A: coarse prune and coarse score

- lattice spacing: fixed initial lattice
- trials: very small prefix
- goal: cheaply eliminate obviously weak candidates
- keep: top `M` candidates or top `p%`

Example only:

- spacing `1.0` mm
- trials `8` to `16`
- keep top `5%` to `10%`, or a capped top `M`

These values are only an example.

If profiling shows the GPU path is underfilled at that stage, `N` should be increased until the combined batch is close to the safe memory limit.

### Stage B: survivor rescoring

- score with a larger trial prefix
- keep a much smaller set of best candidates

Example only:

- spacing unchanged
- trials `32` to `64`

Again, these values are examples, not hard targets.

The real control rule is to pack the mother-function call near the safe batch ceiling.

### Stage C: confirmation ranking

- evaluate only the final small survivor set
- use the large intended MC budget
- emit ranked candidate outputs
- carry only one operational winner into downstream transport by default

Example only:

- spacing unchanged
- trials equal to the intended robust scoring budget

The exact numeric defaults should remain configurable.

The important distinction is:

1. stage-C should score and rank the final small survivor set,
2. the ranked output may retain more than one row for audit or review,
3. but the default carry-down into transport should be exactly one winning candidate.

## Call Minimization Rule

The optimizer should avoid designing stage logic that creates many tiny mother-function calls.

That would waste time in:

- repeated prepper work,
- repeated host-device conversions,
- repeated valid-mask construction,
- repeated kernel launches.

The preferred order of operations is:

1. choose the largest safe packed batch,
2. fill that batch with as many candidate-trial-point combinations as possible,
3. call the mother function,
4. repeat only when another packed batch is required.

So, yes, the first implementation should explicitly optimize for fewer larger mother-function calls rather than many smaller ones.

## Optional Later Spatial Refinement

If later testing shows that the fixed `1.0` mm lattice is too coarse, add one optional local spatial-refinement phase.

That later phase should:

- start only from the final small survivor set,
- build a local neighborhood lattice around each survivor,
- use smaller spacing such as `0.5` mm,
- reuse the same staged trial-prefix logic.

This should be treated as a later enhancement, not as a requirement for the first optimizer-v2 slice.

## Candidate Evaluation Contract

Each candidate behaves like a nominal biopsy centroid location, but candidates should not be evaluated one by one in Python.

The correct contract is:

1. Start from one canonical planned biopsy sample array.
2. Translate it to every candidate centroid in the chunk.
3. Apply biopsy self-transforms trialwise.
4. Apply inverse target-DIL rigid transforms trialwise.
5. Keep target-DIL dilations as structure-side trial geometries.
6. Test all candidate-trial biopsy arrays against the matching target-DIL trial structures in one containment call.

This matches the intended "candidate configuration `i`, trial `j` vs DIL trial `j`" design.

## Batched Shape

Conceptually, if a chunk contains `C` candidates and a stage uses `T` trials, the batched test surface is:

- number of test structures = `C * (T + 1)` if nominal is included, otherwise `C * T`
- points per test structure = `num_biopsy_sample_points`

The mapping array should ensure:

- candidate `c`, nominal maps to target structure nominal
- candidate `c`, trial `t` maps to target structure trial `t`

This is exactly the sort of trialwise simultaneous call the current containment machinery was designed to support.

## Probability Compilation Contract

Optimizer v2 computes only against the target DIL.

The first objective should be:

- mean target-DIL binomial estimator across all sampled biopsy points

Equivalent wording:

- average per-point probability that the sampled biopsy volume lies in the target DIL under the enabled uncertainty model

This is the clean target-only robust objective.

Main-facing metric selection should expose reducer choices over the same target-only per-point probability surface.

Recommended first config surface:

1. default: `mean_pd`
2. optional: `max_pd`
3. optional: `min_pd`

Where `pd` here means the target-DIL probability or binomial-estimator surface computed from the shared trial bank.

The important contract is that optimizer v2 always computes the same underlying target-only per-point probability surface first, and only then applies the selected reducer.

If candidate `i` has biopsy-sample-point probability surface `p_hat_{i,p}`, then the current reducer semantics are:

`mean_pd(i) = (1 / P_i) * sum_p p_hat_{i,p}`

`max_pd(i) = max_p p_hat_{i,p}`

`min_pd(i) = min_p p_hat_{i,p}`

So yes: these reducers act on the biopsy-sample-point axis, not on the trial axis.

That is exactly why the uncertainty math is asymmetric.

For `mean_pd`, averaging over biopsy points commutes with averaging over trials, so if `X_{i,t,p}` is the binary trial-point containment indicator then:

`mean_pd(i) = (1 / P_i) * sum_p ((1 / N) * sum_t X_{i,t,p}) = (1 / N) * sum_t ((1 / P_i) * sum_p X_{i,t,p})`

This gives the clean trialwise scalar `Z_{i,t}` used by the first statistical-pruning pass.

For `max_pd` and `min_pd`, that commutation does not hold in general:

`max_p ((1 / N) * sum_t X_{i,t,p}) != (1 / N) * sum_t max_p X_{i,t,p}`

and similarly for `min`.

So `max_pd` and `min_pd` are still well-defined optimizer objectives, but they do not inherit the same simple exact trialwise decomposition that `mean_pd` does.

The candidate dataframe should retain enough components to support future alternative objectives, including:

- total successes by biopsy sample point,
- per-point binomial estimators,
- candidate-level mean binomial estimator,
- candidate-level max binomial estimator if configured,
- candidate-level min and quantiles if later needed,
- distance to target DIL centroid for tie-breaks.

## Ranking Contract

The output should include both:

- a full tested-candidate dataframe
- a ranked candidate dataframe

The ranked dataframe should include at least:

- patient and biopsy identity
- target DIL identity
- candidate rank
- candidate X, Y, Z in prostate frame
- objective value
- objective reducer name
- stage metadata
- number of trials used
- stored transform-bank prefix size used for scoring
- targeted-DIL nominal score under the same reducer
- winning-candidate downstream-comparable target score when that rescore is requested
- downstream-comparable score trial count
- tie-break fields such as distance to target centroid
- tie-break resolution method
- tie-break warning and fallback flags

## Current Output Surfaces

At the contract level, optimizer v2 already carries two distinct dataframe surfaces with different meanings.

1. `tested_candidate_dataframe` is the concatenation of all scored candidate rows across all stages in the `A -> B -> C` run, after any winner-resolution or downstream-comparable annotations are carried down.
2. `ranked_candidate_dataframe` is the final retained set from the last stage after operational winner resolution has been applied.

The live bridge and the current public sidecar-output path do not emit both of those surfaces yet.

They currently emit:

1. a winner summary dataframe,
2. a ranked candidate dataframe built from the final retained set.

That means the current public ranked CSV is not the full candidate-history table.

It is only the final review surface.

Stage-A and stage-B losers already exist in memory during the run, and they are already counted in summary metadata through fields such as `Target optimizer num tested candidate rows`, but those earlier-stage rows are not yet exported as their own standalone audit table.

The completed optimizer-v2 export contract should therefore keep two distinct public tables:

1. a ranked final-candidate table for transport and quick review,
2. a tested-candidate audit table containing every candidate row that was actually scored at every stage.

The recommended default moving forward is to retain and export the full tested-candidate manifest.

That full manifest should be treated as the canonical research and audit surface for optimizer v2.

The final ranked table should remain, but only as the concise operational subset used for transport, quick review, and winner-focused downstream checks.

The ranked final-candidate table should stay intentionally narrow.

It should remain the surface used for transport, operational review, and winner-level downstream comparisons.

The tested-candidate audit table should instead become the main research and debugging surface.

It should be allowed to grow wider and include stage-level prune and uncertainty metadata.

## Adaptive Uncertainty-Aware Staged Pruning Upgrade

The repo now has two optimizer-v2 search-policy lanes.

1. legacy fixed `stage_configs`,
2. adaptive `adaptive_block_config`.

The current live default should be the adaptive `mean_pd` lane.

That lane uses:

1. an explicit `initial_trial_prefix` floor,
2. an explicit appended `trial_block_size` floor,
3. an explicit `max_total_trials` hard ceiling,
4. an optional `max_test_structures_per_call` packing budget that can raise the cumulative round prefix above those floors,
5. one prune decision after each cumulative shared-prefix round,
6. no fixed raw survivor-count boundary inside those adaptive rounds.

The current executable adaptive pass is intentionally a control-flow upgrade first.

It still rescored the active candidate set on the full cumulative prefix at each round.

It does not yet use retained appended-block sufficient statistics to avoid rescoring the earlier prefix.

That retained-block-only update path remains the next upgrade on top of the adaptive control flow.

The agreed completed design should keep the staged and chunked structure, but change the pruning rule.

The optimizer should no longer prune candidates only because they fall outside a raw top-k or top-fraction boundary at a small trial prefix.

Instead, it should prune only candidates that are statistically dominated at that pruning round.

The intended rule is:

1. score the active candidate set using one shared trial prefix or one appended shared trial block,
2. update each candidate's cumulative score summaries,
3. compare candidates using uncertainty-aware score differences,
4. prune only those candidates whose score deficit is large enough that they are no longer statistically competitive,
5. keep statistically indistinguishable candidates alive even when their current point estimate is lower,
6. continue to the next pruning round with a larger shared prefix.

This preserves the throughput advantage of staged batched scoring while avoiding premature elimination of candidates that only appear worse because `N` is still small.

The current executable policy now treats `initial_trial_prefix` and `trial_block_size` as minimums, not rigid per-round counts.

If `max_test_structures_per_call` is configured, the runner computes the largest cumulative shared prefix that still fits the current candidate chunk inside one containment call and uses that larger `N` whenever it exceeds the floor.

If the floor itself already exceeds the per-call structure budget, the floor still wins and the lower-level containment path simply splits the work across multiple mother-function calls.

### Statistical comparison rule

The intended comparison is not an independent-samples comparison between candidate means.

Because all active candidates in one pruning round are evaluated against the same shared transform-bank prefix, the right first comparison object is the paired score difference under shared randomness.

The recommended implementation contract is:

1. partition the stochastic evaluation into appended shared trial blocks,
2. record sufficient block-level statistics for the exact update rule that is going to be used,
3. compare the current leader set against each active challenger using paired block differences,
4. prune only when the challenger's upper bound is still below the leader's lower bound, or equivalently when the paired difference is decisively unfavorable under the configured confidence rule.

This can give exact cumulative updating, but only if the retained block summaries are actually sufficient for the statistic being claimed.

It also means older trial blocks do not need to be rescored once their sufficient statistics have been retained.

#### Exact cumulative updating from retained block statistics

For the current executable optimizer path, the default reducer is `mean_pd`.

Under that reducer, the random quantity should be defined explicitly.

Let:

1. `i` index candidate centroid,
2. `t` index stochastic trial,
3. `p` index sampled biopsy point along the nominal biopsy model,
4. `X_{i,t,p}` be the binary containment indicator for candidate `i`, trial `t`, and biopsy point `p`, where `X_{i,t,p} = 1` if that localized biopsy point lies inside the target DIL and `0` otherwise.

In the current scorer, the raw structured stochastic containment tensor has this semantic shape:

`candidate x trial x biopsy_point`

For one candidate `i` and one biopsy point `p`, the pointwise probability estimate is:

`p_hat_{i,p} = (1 / N) * sum_t X_{i,t,p}`

If candidate `i` has `P_i` biopsy sample points, then the current `mean_pd` selection score is:

`mu_hat_i = (1 / P_i) * sum_p p_hat_{i,p}`

Substituting the pointwise estimator gives:

`mu_hat_i = (1 / (N * P_i)) * sum_t sum_p X_{i,t,p}`

Now define the per-trial candidate score:

`Z_{i,t} = (1 / P_i) * sum_p X_{i,t,p}`

Then the candidate score can be written exactly as:

`mu_hat_i = (1 / N) * sum_t Z_{i,t}`

This is the cleanest quantity for uncertainty-aware staged pruning under the current `mean_pd` objective.

The easiest exact update is therefore the cumulative mean of `Z_{i,t}`.

If block `b` contains `n_b` trials and candidate `i` has exact block score sum

`S_{i,b} = sum_{t in b} Z_{i,t}`

then the cumulative mean score after multiple blocks is exactly:

`mu_i = (sum_b S_{i,b}) / (sum_b n_b)`

Equivalently, if the stored block value is the exact block mean

`m_{i,b} = S_{i,b} / n_b`

then:

`mu_i = (sum_b n_b * m_{i,b}) / (sum_b n_b)`

That part is exact.

For exact uncertainty of one candidate mean, more than the block mean is needed.

At minimum, the retained history should also support the second moment of the same trialwise score quantity.

The simplest retained second-moment statistic is:

`Q_{i,b} = sum_{t in b} Z_{i,t}^2`

From `S_{i,b}` and `Q_{i,b}`, the cumulative first and second moments of `Z_{i,t}` can be updated exactly across appended blocks.

For exact paired leader-versus-challenger uncertainty, the requirement is stronger again.

If `Y_{i,t}` is the candidate-level trial score for candidate `i` at shared trial `t`, then the paired comparison uses:

`D_{i,j,t} = Z_{i,t} - Z_{j,t}`

The exact paired mean is determined by the retained sums of `D_{i,j,t}`.

The exact paired uncertainty is determined by retained second-moment information for `D_{i,j,t}` as well.

One exact retained paired second-moment statistic would be:

`R_{i,j,b} = sum_{t in b} (Z_{i,t} - Z_{j,t})^2`

That means one scalar block mean per candidate is enough for exact cumulative score updating, but it is not enough by itself for exact paired-difference uncertainty.

If the completed optimizer claims exact paired-difference pruning, it must retain sufficient statistics for that paired quantity, either directly as paired-difference block summaries for the comparisons that remain active or through an equivalent retained representation that preserves the needed covariance structure.

The easiest exact representation is to retain the trialwise `Z_{i,t}` vector for the active candidate set at the current pruning horizon.

That is far cheaper than retaining localized biopsy geometry, and it allows any later paired-difference quantity to be reconstructed exactly for those active candidates.

For the current repo, this exact incremental statistical-pruning story should therefore be scoped to the `mean_pd` objective first.

For `max_pd` and `min_pd`, the current reducer is applied after trial-averaging pointwise probabilities, so the same simple linear trialwise decomposition does not hold in the same way.

Those reducers can remain supported in the legacy fixed-stage lane, but the first adaptive uncertainty-aware pruning implementation should target `mean_pd` explicitly.

### Transform-bank growth and stopping rule

The current code still uses a fixed precomputed shared transform bank.

It requests cumulative leading prefixes from that bank and raises an error if a later request exceeds the available number of stored trials.

That is acceptable for the current adaptive block-round implementation as long as the precomputed bank already covers the declared hard ceiling, but the growth policy should still remain explicit.

The recommended contract is:

1. keep an explicit `initial_trial_prefix` floor,
2. keep an explicit `trial_block_size` floor for each appended pruning round,
3. keep an explicit `max_total_trials` hard ceiling,
4. optionally use a `max_test_structures_per_call` budget to pack a larger cumulative prefix when the current chunk size permits,
5. allow the bank to grow only by appending new contiguous shared trial blocks,
6. never replace earlier draws with a fresh unrelated bank,
7. stop early when one candidate is unique or when the active ambiguity set is small enough for final winner resolution,
8. stop at the hard ceiling if statistical separation is still not achieved.

If the hard ceiling is reached without a unique statistically separated winner, the optimizer should not silently pretend certainty.

It should emit an explicit unresolved-ambiguity flag together with the forced-stop trial count and the operational fallback that was used.

Two implementation strategies are acceptable here:

1. provision a conservative full bank up front that already covers `max_total_trials`,
2. append new trial blocks on demand and extend the stored bank in place.

The second strategy is more flexible.

The first is simpler operationally.

In the second strategy, the appended block is not a recycled pass through the original prefix.

It is a genuinely fresh contiguous extension of the same shared bank.

The earlier prefix remains unchanged, and the newly appended trials occupy the next unused block boundary.

Either way, the completed design should still preserve a declared hard ceiling.

An unbounded while-loop with no maximum trial policy is not recommended for production runs.

If on-demand bank append is implemented, the transform-bank tracking contract must also become explicit.

Appending stochastic blocks is not just a matter of making one array longer.

The optimizer and all downstream consumers must be able to prove that every score, every audit row, every winner-only replay, and every downstream-comparable MC surface refers to the same shared uncertainty realization up to the declared prefix length.

That means the stored bank metadata should include, at minimum:

1. a bank identity or run-local bank UUID,
2. the total number of currently realized trials,
3. the cumulative block boundaries,
4. the random-generation policy and uncertainty-model identity used to create the bank,
5. enough metadata to prove that the appended blocks are contiguous extensions of the same bank rather than a fresh replacement bank,
6. the exact prefix length used by each scoring or replay surface.

Any coupled arrays or derivative cached objects that depend on trial count should then be treated as prefix-indexed views over that same tracked bank, not as independent stochastic objects.

The practical rule is simple:

1. append only contiguous new shared draws,
2. never rewrite earlier draws,
3. never silently swap to a fresh unrelated bank,
4. always stamp every scored surface with the exact bank identity and prefix length it used.

### Retained history and debug replay rule

The reason to retain history is not to keep renderer objects.

The real reason is to avoid recomputing stochastic evidence that may be needed later for pruning audits or winner-only debug scenes.

The agreed memory policy should separate three different history classes.

1. lightweight scalar history that should be retained broadly,
2. late-stage finalist debug payload that may be retained selectively,
3. full localized point geometry for the whole search space, which should not be retained.

The lightweight scalar history should be the default retained surface.

That history should include, at minimum:

1. candidate global index,
2. prune round or stage name,
3. cumulative trial count,
4. appended block size,
5. block-level score contribution,
6. cumulative score,
7. uncertainty summary or enough information to reconstruct it,
8. survivor flag,
9. prune reason.

This scalar history is cheap enough to retain for the full active search surface and is the right substrate for later audit tables and reviewer-facing methods summaries.

The winner-containment render path is different.

That path needs localized stochastic success/failure geometry.

The completed design should not retain that full geometry for every candidate across every stage.

That is unnecessary and scales badly.

The preferred rule is:

1. do not retain full localized point geometry for the full candidate space,
2. do not retain Open3D or Plotly objects at all,
3. allow winner-only or late-finalist localized payload retention only for a very small finalist subset if replay avoidance later proves worth the memory cost,
4. otherwise accept winner-only recomputation for debug rendering.

That means recomputing the winner-only containment debug scene remains an acceptable default behavior.

### Multiround render navigation contract

When stage-boundary or round-boundary debug scenes are rendered for optimizer v2, the default live behavior should render all available pruning rounds unless an explicit stage-name filter is provided.

The multiround Open3D viewer should remain usable even when the adaptive search emits more than the old three legacy stages.

The intended navigation contract is:

1. direct digit keys `1-9` for the first nine available rounds,
2. left and right arrow keys for previous and next round navigation,
3. home and end for first and last round navigation,
4. the same layer-toggle keys staying valid regardless of which round is active.

This keeps the render surface aligned with adaptive pruning rounds instead of drifting back toward a fixed three-stage mental model.

### Audit and output contract for pruning rounds

The completed optimizer-v2 output contract should make pruning auditable without requiring re-execution.

That means the tested-candidate audit surface should record not only scores but also pruning decisions.

Recommended additional columns include:

1. prune round index,
2. last stage reached,
3. pruned-at stage or pruning round,
4. prune reason,
5. active-set size before prune,
6. active-set size after prune,
7. survivor threshold for that round,
8. statistical-separation metric used for the decision,
9. whether the candidate remained ambiguity-active,
10. whether the run terminated because of a hard ceiling rather than clean separation.

The final ranked table should remain concise, but it should still carry summary prune metadata that helps interpret the winning set quickly.

Recommended carry-down columns for the final ranked table include:

1. last stage reached,
2. number of pruning rounds survived,
3. final winner-resolution trial count,
4. whether the final decision was statistically unique,
5. whether a forced hard-boundary stop was used.

### Bayesian sidequest recommendation

There is a separate Bayesian-posterior idea that is conceptually related to the same uncertainty-reduction story.

That is interesting, and the repo already contains at least one small Bayesian demo script, but it should not become the first implementation path for the optimizer upgrade.

The first upgrade should remain a paired-difference frequentist pruning rule on top of the existing Monte Carlo score surface.

That recommendation is driven by engineering clarity rather than a claim that Bayesian updating is invalid.

The reasons are:

1. the current optimizer already exposes a Monte Carlo score surface that can be updated exactly from retained block summaries,
2. introducing priors creates new policy questions that are unrelated to the immediate pruning bugbear,
3. a paired-difference rule is easier to audit against downstream MC and easier to explain in manifests and sidecar CSVs,
4. the Bayesian path can remain a later methodological sidequest or reviewer-facing appendix if it becomes scientifically useful.

If a Bayesian pathway is explored later, it should be treated as an optional alternate inference layer over the same retained scalar history rather than as the default operational pruning rule.

## Final Winner Tie-Break Policy

The optimizer should remain score-first.

Nearest-to-centroid should not be the normal selector when multiple candidates share the best final score.

Recommended first policy:

1. detect a final-winner tie using the configured score tolerance,
2. emit a warning to the log and manifest that the final winner is still tied at the current trial prefix,
3. rerun only the tied final candidates with a larger shared trial prefix,
4. allow this tie-break rescore escalation a small fixed number of times, with `2` additional attempts as the recommended default,
5. if the tie still persists after those score-based attempts, fall back to nearest target-DIL centroid,
6. record explicitly that the geometric fallback was used.

This keeps the selection metric score-based first and geometric only as a last resort.

Those larger tie-break prefixes must come from new draws in the shared bank beyond the earlier stage-C prefix.

They should not be produced by reusing the exact same fixed `N` draws, because that cannot change the score estimate.

They also should not be produced by changing the perturbation magnitudes just for tie-breaking, because that would change the optimization objective and break comparability with downstream MC.

The recommended default escalation rule is:

1. start from the stage-C trial prefix,
2. multiply the trial prefix for each tie-break attempt using the shared transform bank,
3. clip each attempt to the actually available shared-bank ceiling,
4. stop early as soon as the winner becomes unique.

The ranked or winner manifest should retain enough metadata to audit this later, including:

1. whether a tie was detected,
2. how many additional tie-break rescoring attempts were needed,
3. the final trial count actually used for winner resolution,
4. whether geometric fallback was invoked.

The intended interpretation of the shared bank here is:

1. stage C might score at prefix `0:N`,
2. the first tie-break attempt might score at prefix `0:2N`,
3. the second tie-break attempt might score at prefix `0:4N`,
4. the additional trials are genuinely new IID draws already present in the larger pre-generated bank.

This is still one shared bank. It is not a fresh independently generated bank per tie-break attempt.

Changing the perturbation scale for tie-breaking is not recommended.

That would answer a different question: robustness under a different uncertainty model.

If that is ever useful, it should be treated as a separate stress-test or sensitivity-analysis mode, not as the normal winner-resolution policy.

## Optional Sidequest: Sensitivity-Robustness Review Path

If this model later needs a stronger reviewer-facing validation story, a useful sidequest is to add one explicit sensitivity-analysis pathway that perturbs the uncertainty model itself and checks whether the optimizer winner is stable.

This is a worthwhile question, but it should remain clearly separated from the main winner-selection path.

Recommended framing:

1. keep the main optimizer winner defined under one fixed declared uncertainty model,
2. run one optional sensitivity mode afterward under altered uncertainty settings,
3. report whether the winner or top-ranked set is stable under that altered model,
4. do not use that altered-model run as the normal tie-break mechanism.

Examples of altered-model sensitivity runs that may be interesting later:

1. larger perturbation magnitudes with the same sampling family,
2. alternate reducer choice over the same target-only probability surface,
3. different biopsy-only versus target-only uncertainty emphasis,
4. stability of the winner under larger transform-bank prefixes.

If implemented, this should emit reviewer-friendly metadata such as:

1. baseline winner,
2. sensitivity-mode winner,
3. overlap of the top-ranked candidate set,
4. magnitude of score movement,
5. whether the baseline winner remains operationally acceptable.

This can strengthen the validation story, but it is not required for the first executable optimizer-v2 slice.

Transport should consume this contract generically.

It should not care whether the row came from legacy optimizer v1 or optimizer v2.

## Downstream Agreement Contract

There are two distinct score surfaces once optimizer-v2 trial counts and downstream tissue trial counts are allowed to differ:

1. the optimizer selection score used to rank candidates,
2. the downstream-comparable winner score computed at the downstream tissue count.

Only the second score needs to agree exactly with downstream MC.

That should become an explicit validation surface.

Recommended first contract:

1. store the optimizer selection score for the winning candidate together with reducer name, target DIL identity, transform-bank metadata, and the optimizer trial count actually used,
2. if downstream tissue scoring uses a different `N_stochastic_tissue`, force one additional winner-only rescore using that same shared transform bank prefix,
3. store that downstream-comparable winner score separately together with its own trial count,
4. when downstream MC later runs with the same `N_stochastic_tissue` and reuses the same stored bank, require exact agreement against the downstream-comparable winner score,
5. emit both the optimizer selection score and the downstream-comparable agreement result to the manifest surface and the Rich UI.

This gives an early warning if the optimizer path and downstream stochastic path silently diverge even though they claim to share the same uncertainty bank.

It also avoids a false requirement that the optimizer selection score must equal the downstream MC score when the user intentionally chooses different trial counts for ranking versus later tissue analysis.

## Computation And Memory Strategy

The main danger is allocating the full `all_candidates x all_trials x all_points x 3` array at once.

That is not acceptable.

The correct strategy is chunked batched evaluation.

### Outer loops that are acceptable

- loop over stages
- loop over candidate chunks
- optionally loop over local refinement neighborhoods

### Loops that are not acceptable in the hot path

- loop over MC trials in Python
- loop over candidate points one candidate at a time for containment

### Memory rule of thumb

The dominant biopsy test array cost is approximately:

`num_candidates_in_chunk * num_trials_in_stage * num_points_per_biopsy * 3 * bytes_per_float`

With `float32`, bytes per float is `4`.

That estimate does not include containment prep arrays, index arrays, or the trialwise target-DIL geometry pack, so real usage is higher.

Therefore chunk size must be chosen conservatively.

### Practical defaults

Start conservative.

Example starting point only:

- coarse stage chunk size: `64` to `256` candidates
- fine stage chunk size: `16` to `64` candidates
- confirmation stage chunk size: whatever fits after profiling, likely `8` to `32`

These should become configurables or auto-tuned limits later.

### Reuse strategy

- build the target-DIL nominal-plus-dilated trial pack once per target DIL per stage budget
- never rebuild that pack per candidate
- reuse one planned biopsy sample array
- reuse one shared transform bank
- free or reuse CuPy buffers between chunks

## Recommended First Implementation Scope

Do not build the entire final optimizer in one pass.

The first implementation slice should be:

1. generate and store the shared transform bank earlier in the pipeline
2. build the target-DIL search lattice and prune it to target-interior candidates
3. implement chunked candidate evaluation for one objective only
4. emit ranked candidate outputs
5. teach transport to consume the ranked candidate contract

The first objective should remain target-only mean binomial estimator.

Do not add multi-core diversity constraints, OAR penalties, or guidance-hole logic in the first slice.

## Non-Goals For The First Slice

- no multi-core coupled optimization
- no template-hole ranking inside optimizer v2
- no sum-to-one tissue-class objective
- no OAR penalty objective
- no large framework refactor around the legacy optimizer

These can be layered later after the core target-only stochastic evaluator is stable.

## Validation Requirements

Every implementation pass should end with concrete checks.

Required validation categories:

1. candidate prune correctness
2. trialwise mapping correctness
3. equality of reused transform bank between optimizer and downstream MC
4. chunked evaluation equivalence versus a tiny unchunked reference case
5. ranking stability when increasing the trial prefix
6. transport compatibility with ranked candidate output

## Current Implementation Status

The active v2 slice is now the code under `python_files_dcm_meta_based/biopsy_optimizer/v2/`.

It currently provides:

1. stage configuration objects and bank-size policy,
2. fixed-lattice target candidate generation,
3. immediate target-interior pruning through the existing CUDA containment mother function,
4. shared transform-bank, localization, and containment seams outside optimizer v2,
5. chunked target-only candidate-trial scoring,
6. staged A -> B -> C search with final winner resolution and winner-only downstream-comparable validation,
7. tested and ranked candidate dataframe emission,
8. replayable render-job construction for stage-boundary and success/failure debug surfaces,
9. carry-down in-memory output dataframe builders for the target lane.

It does not yet provide:

1. live simulated-family declaration through `bx_sim_locations_dict` or an equivalent family registry,
2. planner and transport integration through `simulated_biopsy_planner_processer(...)` and `simulated_biopsy_processer(...)`,
3. legacy export-block wiring for the new target-lane sidecar outputs,
4. full real-case validation of shared-bank reuse and transport handoff,
5. a finalized memory-safe activation policy when total simulated biopsy count grows.

## Visualization Selector Recommendation

The first implementation should not use a growing pile of ad hoc booleans.

Use one explicit visualization-selector object that can be passed through the v2 pipeline.

That selector object should carry:

1. which surfaces to show,
2. exact candidate indices to inspect,
3. random candidate count for spot checks,
4. exact trial indices to inspect,
5. random trial count for spot checks,
6. random seed for reproducibility.

The current v2 module now supports this pattern for the candidate-pool stage.

The scoring stage should reuse the same selector object so that only a small chosen subset of candidate-trial slices gets materialized for plotting.

That is the right way to validate:

1. generated lattice points,
2. immediate target-interior prune results,
3. translated candidate biopsy samples,
4. trialwise transformed biopsy samples,
5. containment results for a small selected subset.

This keeps debugging deterministic and avoids trying to render the full hot-path batch.

## Stage Candidate Render Recommendation

For debug and verification, optimizer v2 should support at least one simultaneous Open3D render of candidate positions at each stage boundary.

Minimum useful scenes:

1. stage-input candidate positions before scoring,
2. stage-output survivor positions after pruning,
3. optional overlay of target DIL geometry and the planned biopsy nominal centroid,
4. optional highlight of the final winner.

These renders should show all candidates in the stage simultaneously, not one candidate at a time.

Implementation recommendation:

1. reuse the repo's existing Open3D plotting surfaces such as `plotting_funcs.plot_geometries(...)`,
2. reuse the same Rich pause/resume behavior already used in render/debug paths so stopwatch timing excludes interactive inspection,
3. keep this visualization outside the hot scoring loop and trigger it only at stage boundaries or for explicitly selected checkpoints.

The existing processed-dataset render/debug surface is a good orchestration model even though the actual scene contents will be optimizer-specific.

## Media Export Recommendation

The repo should support file-based rendering directly from code, not only interactive viewing.

The right surface is not a loose collection of `ScreenCamera_*.json` files referenced from main.

The right surface is a named render-job system.

Each render job should specify:

1. scene type,
2. patient and structure selection,
3. candidate selector,
4. trial selector,
5. camera preset,
6. frame schedule,
7. output resolution,
8. output directory,
9. packaging mode.

Recommended packaging flow:

1. generate deterministic PNG frames to disk,
2. write a manifest alongside those frames,
3. optionally package the frame sequence into MP4 or GIF.

That gives three advantages:

1. reproducible figures without opening windows,
2. reusable frame sets for publications and social media clips,
3. clean output organization instead of camera-state files scattered near code.

Recommended output layout:

1. one root such as `output_data/media_exports/`,
2. one subdirectory per named render job,
3. one manifest file per export,
4. one `frames/` directory plus optional packaged video artifacts.

The same visualization-selector object used for debugging should be reusable here, so a render job can say things like:

1. lattice only,
2. candidate prune only,
3. candidate `17` and trials `0, 3, 7`,
4. top `k` ranked candidates over stage transitions.

Backend recommendation:

1. first implementation should keep using Open3D because the repo already stores and manipulates many Open3D-ready geometries,
2. the render-job surface should still hide the backend behind a small adapter so a later PyVista or VTK renderer can replace it if that produces better publication or social-media output,
3. do not couple the export system to ad hoc camera JSON files.

So the recommendation is not "Open3D forever".

The recommendation is:

1. Open3D first for minimal integration cost,
2. renderer abstraction from day one,
3. deterministic frame export as the real contract.

For optimizer-v2 debugging specifically, the first scene adapter should support stage-boundary candidate clouds as a named scene type rather than treating them as ad hoc one-off Open3D calls from main.

## Repo Config Surface Recommendation

This repo should move away from the current pattern where many runtime choices live at the top of main.

The first cleanup step should be a typed Python config surface, not a giant immediate migration to YAML or JSON.

The recommended path is:

1. create small config dataclasses grouped by responsibility,
2. let main build one root config object,
3. pass narrow config slices to modules,
4. only later add file-backed loading if that still feels useful.

For optimizer v2 specifically, the first config pass should be treated as preparatory runtime policy only.

It should cover things like search policy, stage policy, scoring policy, visualization policy, and downstream-comparable winner rescore policy.

It should not pretend that the repo already has a fully separate optimizer-v2 biopsy family creation pathway wired end to end.

That family-creation and transport integration work is a later integration pass, not a prerequisite for building the clean config surface now.

For future GUI work, the most important design property is that the config surface should be pure data.

That means:

1. no progress-bar objects in config,
2. no file handles or runtime-only objects in config,
3. no derived arrays cached inside config objects,
4. one stable serializable schema that a GUI can edit and hand to the pipeline.

Recommended config domains:

1. optimizer config,
2. visualization config,
3. media export config,
4. MC/runtime config,
5. output-path config.

That is safer than trying to externalize every current top-of-main setting at once.

It also fits the current code style better because many options already behave like structured runtime settings rather than true user-facing text configuration.

Folder reorganization should be done as a controlled follow-up sweep once the new surfaces stabilize.

It should not be mixed aggressively into phase-2 implementation work unless a file move directly improves the seam being built.

## Implementation Phases

The next phases should be treated as separate executable slices.

### Phase 1: candidate pool

This phase is already in place.

Deliverables:

1. fixed `1.0` mm lattice generation,
2. target-interior pruning,
3. candidate-pool visualization.

### Phase 2: upstream transform-bank seam

Goal:

Generate one shared uncertainty bank before optimizer scoring and store it where downstream MC can reuse it later.

Clarification:

The random draw generation for both biopsy and relative structures already exists in `MC_prepper_funcs.generate_transformations(...)`.

So phase 2 is not primarily about inventing a new source of random draws.

Earliest-seam clarification:

There are two different answers here.

As the code works today, the earliest safe place to call `generate_transformations(...)` without changing dependencies is still late in main, after simulated-biopsy geometry has been processed and after uncertainty objects have been attached to all structures.

That is because:

1. `generate_transformations(...)` requires `Uncertainty data` on every structure it touches,
2. the current uncertainty-builder configuration for biopsies uses per-biopsy variation terms such as `Mean centroid variation`,
3. simulated biopsies do not get those variation fields until after planned geometry is transported and finalized,
4. the current biopsy needle-compartment shift generator reads `Reconstructed biopsy cylinder length (from contour data)` directly from each biopsy structure.

So, in the current implementation, `just after multiplicity` is too early.

With a small contract refactor, the pure random-draw bank can move much earlier than it does now.

The realistic early target is:

1. after simulated-biopsy preparation has completed,
2. after the final per-biopsy row set exists post-multiplicity,
3. after uncertainty objects are available in a form that does not depend on later simulated-biopsy geometry fields.

That distinction matters because multiplicity alone is not enough for the biopsy-side draw bank when uniform biopsy-compartment shifts are enabled.

Those shifts need an effective biopsy core length.

For real biopsies, the existing reconstructed contour length is already the right source.

For simulated biopsies, phase 2 should stop requiring the later top-level reconstructed contour field and instead read one earlier effective length source, preferably the prepared nominal length from simulated-biopsy preparation, or secondarily the planned model length from simulated-biopsy planning.

Phase 2 is about:

1. formalizing those existing arrays as one shared transform-bank contract,
2. making the target-specific derived geometry pack available early enough for optimizer-v2,
3. ensuring downstream MC later reads the same stored draws instead of regenerating or reshaping them differently.

Concrete work:

1. start from the existing arrays produced in `generate_transformations(...)`,
2. split the seam question into two contracts:
   - an early pure draw-bank contract,
   - a later derived-geometry-pack contract,
3. formalize biopsy self-transform draws up to `max_trials_for_optimizer_v2` as a reusable bank surface,
4. formalize target rigid-transform draws up to the same trial budget as part of that same bank,
5. move the pure draw bank as early as the dependency contract allows, ideally after simulated-biopsy preparation rather than after downstream MC setup,
6. compute the target nominal-plus-trial structure pack later, at the first point where the necessary planned biopsy arrays and target-relative geometry are both available, because that pack is not currently stored when optimizer-v2 runs,
7. if needed, add one small helper that resolves effective biopsy length for draw generation without requiring later reconstructed simulated-biopsy geometry fields,
8. if needed, split uncertainty-object creation from later geometry-derived biopsy-variation terms so optimizer-v2 can consume the shared draw bank earlier,
9. store those arrays on the same patient or structure objects used later by downstream MC,
10. expose a narrow getter surface that returns prefixes of that stored bank.

This phase should also define the reusable localization-transformer contract described above, even if the actual helper extraction lands one phase later.

Validation:

1. optimizer and downstream MC must read the same stored bank,
2. stage prefixes must be exact leading slices of the full bank,
3. nominal-plus-trial indexing must be unambiguous,
4. the early draw-bank seam must not depend on later simulated-biopsy geometry fields unless that dependency is made explicit by contract.

### Phase 3: candidate-trial batch builder

Goal:

Turn one candidate chunk and one stage trial prefix into the exact batched arrays needed by the containment mother function.

This phase needs a dedicated wrapper surface around the mother-function call.

Do not rely on implicit row ordering alone.

Concrete work:

1. start from one canonical planned sampled biopsy array,
2. translate it to every candidate centroid in the chunk,
3. apply biopsy self-transforms for the stage prefix,
4. apply inverse target rigid transforms for the same stage prefix,
5. pair those test arrays with the stored target nominal-plus-trial structure pack produced in phase 2,
6. emit bookkeeping arrays that preserve candidate index and trial index identity,
7. emit the exact `test_struct_to_relative_struct_1d_mapping_array` needed by the mother function.

This is the phase where the first optimizer-facing localization-transformer helper should likely land.

That helper should be responsible for producing aligned candidate-trial biopsy arrays and aligned target-relative structure arrays without performing scoring, ranking, or downstream writeback.

Important containment constraint:

The current 3D mother-function path does not require identical point coordinates across all test structures.

The important constraint is only:

1. every test structure in the 3D batch must have the same number of biopsy sample points.

So chunking across candidate points is valid, but only if the wrapper preserves the row mapping cleanly on both sides of the mother-function call.

The wrapper should preserve at least:

1. chunk-local test-structure index,
2. candidate global index,
3. candidate chunk index,
4. trial index,
5. nominal-vs-trial flag,
6. mapped target relative-structure index.

## Generalized Chunking Recommendation

There is a worthwhile reusable abstraction here, but it should be narrower than optimizer-v2 semantics.

Do not put candidate-ranking logic into the custom containment package.

Do not make the custom containment package understand optimizer-specific concepts like:

1. candidate centroids,
2. stage schedules,
3. top-k survivor pruning,
4. target-only scoring.

The reusable part is a sanitized row-batch executor.

That generalized executor should:

1. accept already-aligned 3D test arrays,
2. accept the matching relative-structure pack,
3. accept the `test_struct_to_relative_struct_1d_mapping_array`,
4. accept one maximum batch budget such as max test structures or max total test points,
5. slice all aligned inputs consistently,
6. call the mother function repeatedly when needed,
7. concatenate results and preserve caller-supplied metadata.

That would make it reusable across:

1. optimizer-v2 candidate chunks,
2. future batching across biopsies,
3. other row-aligned containment workloads.

What should stay outside that generalized executor:

1. construction of candidate-trial arrays,
2. creation of target trial structures,
3. candidate ranking,
4. optimizer-specific tie-break logic.

So the recommendation is:

1. yes to a generalized chunk executor in the custom package,
2. no to pushing optimizer-v2 semantics down into that package,
3. do it after the optimizer-v2 wrapper has made the required invariants concrete enough to generalize safely.

That also means the grandmother function should not live inside `biopsy_optimizer/v2`.

It belongs with the shared containment infrastructure or an adjacent shared batching helper module that can be reused by non-v2 callers.

If the custom one-to-one CUDA containment stack is migrated into its own standalone repository, the grandmother executor should move with that stack or target that package boundary from the start.

In that future layout, optimizer v2 should depend on the standalone containment package as a consumer, not as the owner of the batching abstraction.

If a grandmother function is added, its job should be exactly this generalized chunk execution layer.

It should own:

1. max safe packed batch sizing,
2. slicing aligned test arrays and relative-structure packs,
3. repeated mother-function invocation,
4. metadata-preserving concatenation.

It should not own:

1. optimizer stage policy,
2. trial-prefix semantics,
3. candidate ranking,
4. target-only reducer selection.

Validation:

1. selected candidate/trial visualizations must match geometric expectations,
2. one tiny batch must agree with a hand-built reference case,
3. mapping from `(candidate, trial)` to relative structure must be exact.

### Phase 4: chunked target-only scoring

Goal:

Run the batched containment call for one chunk and compile one target-only objective.

Concrete work:

1. choose candidate chunk size from the combined batch budget,
2. run one mother-function call for the chunk,
3. aggregate successes per candidate and per biopsy sample point,
4. compute the candidate-level mean target binomial estimator,
5. retain tie-break fields such as distance to target centroid.

Validation:

1. selected candidate/trial containment plots must look correct,
2. chunked results must agree with a tiny unchunked reference,
3. larger stage prefixes must refine rather than destabilize ranking.

### Phase 5: staged pruning and ranking

Goal:

Apply the `A -> B -> C` stage schedule using shared trial prefixes.

Concrete work:

1. score all candidates at stage A,
2. prune survivors,
3. rescore survivors at stage B using a larger prefix,
4. prune again,
5. confirm final ranking at stage C and carry one operational winner forward by default,
6. if requested, force one winner-only rescore at downstream `N_stochastic_tissue` using the same shared bank,
7. emit full and ranked candidate dataframes.

### Phase 6: transport integration

Goal:

Teach transport to consume the ranked v2 candidate dataframe without special-case logic.

Concrete work:

1. map the winning ranked centroid back into the existing transport path,
2. preserve tie-break metadata,
3. keep the transport contract generic across v1 and v2 outputs.

## Immediate Next Coding Pass

The next pass should do two narrow things:

1. wire one live optimizer-v2 simulated family into the planning and transport flow,
2. export the new target-lane sidecar dataframes without mutating legacy CSV contracts.

That turns the already-implemented library slice into a real end-to-end pipeline lane while keeping the export migration additive.
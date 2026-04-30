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

Example only:

- spacing unchanged
- trials equal to the intended robust scoring budget

The exact numeric defaults should remain configurable.

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

The candidate dataframe should retain enough components to support future alternative objectives, including:

- total successes by biopsy sample point,
- per-point binomial estimators,
- candidate-level mean binomial estimator,
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
- stage metadata
- number of trials used
- tie-break fields such as distance to target centroid

Transport should consume this contract generically.

It should not care whether the row came from legacy optimizer v1 or optimizer v2.

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

The first narrow v2 slice is now in `biopsy_optimizer_module_v2.py`.

It currently provides:

1. stage configuration objects,
2. fixed-lattice target candidate generation,
3. immediate target-interior pruning through the existing CUDA containment mother function,
4. visualization selectors for lattice, containment, and selected-candidate inspection.

It does not yet provide:

1. shared transform-bank generation,
2. chunked candidate-trial scoring,
3. ranked candidate dataframe emission,
4. transport integration.

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

Concrete work:

1. identify the earliest safe upstream location where planned sampled biopsy points and target-relative structures are both available,
2. generate biopsy self-transform draws up to `max_trials_for_optimizer_v2`,
3. generate target rigid-transform draws up to the same trial budget,
4. generate target dilation trial structures up to the same trial budget,
5. store those arrays on the same patient/structure objects used later by downstream MC,
6. expose a narrow getter surface that returns prefixes of that stored bank.

Validation:

1. optimizer and downstream MC must read the same stored bank,
2. stage prefixes must be exact leading slices of the full bank,
3. nominal-plus-trial indexing must be unambiguous.

### Phase 3: candidate-trial batch builder

Goal:

Turn one candidate chunk and one stage trial prefix into the exact batched arrays needed by the containment mother function.

Concrete work:

1. start from one canonical planned sampled biopsy array,
2. translate it to every candidate centroid in the chunk,
3. apply biopsy self-transforms for the stage prefix,
4. apply inverse target rigid transforms for the same stage prefix,
5. pair those test arrays with the stored target nominal-plus-trial structure pack,
6. emit bookkeeping arrays that preserve candidate index and trial index identity.

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
5. confirm final ranking at stage C,
6. emit full and ranked candidate dataframes.

### Phase 6: transport integration

Goal:

Teach transport to consume the ranked v2 candidate dataframe without special-case logic.

Concrete work:

1. map the winning ranked centroid back into the existing transport path,
2. preserve tie-break metadata,
3. keep the transport contract generic across v1 and v2 outputs.

## Immediate Next Coding Pass

The next pass should do two narrow things:

1. add the shared pre-optimizer transform-bank generation seam,
2. add target-only chunked candidate-trial scoring on top of the already-implemented candidate-pool surface.

That keeps the first executable slice small and makes the downstream MC agreement problem solvable from the start instead of retrofitted later.
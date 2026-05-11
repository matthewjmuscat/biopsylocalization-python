# Optimizer V2 Performance Plan

## Current Measurement Status

The optimizer-v2 timing breakdown now records these per-stage slices:

- `biopsy_self_transform`
- `relative_structure_localization`
- `flatten_for_containment`
- `containment`
- `score_reduction`
- `tested_candidate_dataframe`

These timings are emitted in the runtime checkpoint `optimizer_v2.structure.search.end` immediately after each target-structure staged search completes inside the per-structure loop.

The runtime logger flushes after writes, so the timing breakdown should be inspectable as soon as the first optimizer-v2 biopsy/target structure finishes its search. We do not need to wait for the full algorithm to complete to inspect the runtime log.

The summary, ranked, and tested optimizer-v2 dataframes are accumulated during the run and are written to CSV later in the top-level export phase. For first-pass performance inspection, the runtime log is the right source of truth.

## Already Landed

- Detailed per-stage timing instrumentation across scoring, runner aggregation, runtime checkpoints, and output dataframes.
- A small hot-path optimization in winner validation: downstream-comparable rescoring no longer builds a tested-candidate dataframe when only scalar winner scores are needed.

## Ranked Performance Recommendations

The estimates below are rough expected wall-clock improvements if the named slice is a major share of the run. They are meant to prioritize investigation order, not to promise exact speedups.

| Rank | Recommendation | Primary code locations | Expected gain | Confidence | Why this is promising |
| --- | --- | --- | --- | --- | --- |
| 1 | Remove the CuPy -> NumPy boundary before containment and keep the aligned batch on-device as long as possible. | `preprocessing/localization_transformer.py` (`build_relative_structure_localized_biopsy_batch`, `flatten_relative_structure_localized_batch_for_containment`), `biopsy_optimizer/v2/scoring.py` (`score_target_candidate_chunk`), `preprocessing/containment_runner.py` (`run_aligned_containment_batch`) | Very high, roughly `1.5x` to `4x` overall if flattening plus containment-boundary work dominates | High | The current optimizer path localizes with CuPy and then explicitly converts to NumPy before containment. That is the clearest avoidable data-movement boundary in the hot path. |
| 2 | Vectorize candidate self-transform and relative-localization work more aggressively to remove candidate-by-candidate Python loops. | `preprocessing/localization_transformer.py` (`build_candidate_biopsy_self_transform_batch`, `build_relative_structure_localized_biopsy_batch`) | High, roughly `1.3x` to `3x` overall if localization dominates | High | Static review showed these helpers still do substantial Python-side per-candidate orchestration before the batched containment call. |
| 3 | Increase work packed into each containment call by retuning chunk size and calibrated batch size after the new timings come back. | `biopsy_optimizer/v2/runner.py`, `biopsy_optimizer/v2/scoring.py`, optimizer-v2 config in `biopsy_localization_convex_main.py` | Medium to high, roughly `1.2x` to `2.5x` overall | Medium-high | Current defaults are conservative, especially `optimizer_v2_max_candidates_per_chunk = 8`. If containment launch/prep overhead is nontrivial, larger chunks should improve throughput until memory becomes the limit. |
| 4 | Make tested/ranked dataframe construction optional or deferred for performance runs, not a default cost of every structure search. | `biopsy_optimizer/v2/scoring.py`, `biopsy_optimizer/v2/runner.py`, `biopsy_optimizer/v2/output.py`, `biopsy_optimizer/v2/live_integration.py` | Medium, roughly `1.1x` to `1.5x` overall | Medium-high | We already removed one unnecessary tested-dataframe build in winner validation and that was clearly safe. The remaining dataframe work is still pure overhead relative to geometric scoring. |
| 5 | Keep score reductions on the device until the final winner extraction, instead of pulling multiple intermediate arrays back to host. | `biopsy_optimizer/v2/scoring.py` | Medium, roughly `1.05x` to `1.3x` overall | Medium | The scoring path still performs repeated `cp.asnumpy(...)` conversions for reductions that could stay device-side longer. |
| 6 | Turn render-heavy and debug-heavy optimizer-v2 defaults off for profiling and normal throughput runs. | optimizer-v2 config block in `biopsy_localization_convex_main.py`, plus `biopsy_optimizer/v2/render.py` and `biopsy_optimizer/v2/live_integration.py` | Case-dependent, from negligible up to roughly `1.1x` to `2x` wall-clock reduction | High | The current defaults request stage-boundary clouds, Plotly export, winner containment debug, and the `both` backend. Even when not the core compute bottleneck, these features pollute timing baselines and extend wall-clock runtime. |
| 7 | Reduce host transfers in candidate-pool pruning and make sure the one-time target-interior mask path stays cheap. | `biopsy_optimizer/v2/candidate_pool.py` (`prune_candidate_lattice_to_target_interior`, `build_target_candidate_pool`) | Low to medium, roughly `1.05x` to `1.2x` overall | Medium | This is probably a smaller slice than staged scoring, but it is still an avoidable CPU boundary and it runs before every structure search. |
| 8 | Only if the new timings show containment itself dominating: profile and optimize the shared containment prepper / grandmother / kernel path. | `preprocessing/containment_runner.py`, `custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.py`, `custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p_grandparents.py` | Potentially very high, roughly `1.5x` to `5x`, but highly uncertain until measured | Low to medium | This could be the biggest remaining lever, but right now the strongest evidence points to optimizer-v2-side orchestration and boundary costs rather than a proven kernel-core bottleneck. |

## Recommendations I Am Most Confident In

These are the items I am most comfortable prioritizing first once we inspect the first timing breakdown:

1. Remove or reduce the device-to-host boundary before containment.
2. Vectorize the localization helpers further.
3. Disable render/debug-heavy defaults for the profiling rerun.
4. Defer tested/ranked dataframe materialization when a run is being used mainly for optimization, not reporting.

## Suggested Decision Sequence

1. Stop the current long run unless we explicitly want it as a coarse baseline.
2. Rerun one short optimizer-v2 case with render/debug-heavy options disabled.
3. Inspect the first `optimizer_v2.structure.search.end` timing breakdown from the runtime log.
4. If `flatten_for_containment` plus `containment` is dominant, prioritize ranks 1 and 3, then inspect the shared containment path.
5. If `biopsy_self_transform` plus `relative_structure_localization` is dominant, prioritize rank 2 first.
6. If `tested_candidate_dataframe` is materially visible, prioritize rank 4 immediately.

## Interpretation Notes

- The expected gains are not additive.
- Ranks 1 and 2 likely overlap because both attack optimizer-v2-side orchestration overhead before the main containment call.
- Rank 8 should wait for evidence from the new timings unless we decide to do a deeper kernel audit in parallel for other reasons.
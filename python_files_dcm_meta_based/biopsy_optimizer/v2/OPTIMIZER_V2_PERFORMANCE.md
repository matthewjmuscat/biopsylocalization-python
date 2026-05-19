# Optimizer V2 Performance Plan

## Current Measurement Status

The optimizer-v2 runtime instrumentation now records all of these slices in the live search path:

- Per-stage scoring slices: `biopsy_self_transform`, `relative_structure_localization`, `flatten_for_containment`, `containment`, `score_reduction`, and `tested_candidate_dataframe`.
- Search-level slices: `stage_total_elapsed_seconds`, `winner_resolution_elapsed_seconds`, `winner_validation_elapsed_seconds`, and `unattributed_search_elapsed_seconds`.
- Grandmother sub-slices: total containment grandmother time, mother-call time, chunk slicing time, chunk concatenation time, inner chunk count, and chunked-call count.
- Isolated downstream rescore benchmark event: `optimizer_v2.structure.winner_validation_benchmark.end`.

These timings are emitted in the runtime checkpoints as soon as each target-structure staged search completes, and the logger flushes after writes. For first-pass performance inspection, the runtime log remains the source of truth.

## Already Landed

- Detailed per-stage timing instrumentation across scoring, runner aggregation, runtime checkpoints, and output dataframes.
- Search-level timing splits that proved where the old unattributed search gap was actually going.
- Grandmother timing-report mode that separates mother-call time from wrapper-only work.
- An isolated winner-validation benchmark path for direct comparison against the in-search downstream rescore.
- A small winner-validation hot-path optimization: downstream-comparable rescoring no longer builds a tested-candidate dataframe when only scalar winner scores are needed.

## Latest Measured Findings

### Live optimizer-v2 reruns

| Run | Search elapsed | Stage total | Winner validation | Unattributed search | Key conclusion |
| --- | --- | --- | --- | --- | --- |
| `MC_sim_out- Date-May-10-2026 Time-22,57,42` | `2335.754 s` | `2207.177 s` | `128.493 s` | `0.082 s` | The old `~124 s` blind spot was almost entirely downstream winner validation. |
| `MC_sim_out- Date-May-11-2026 Time-00,05,47` | `2347.443 s` | `2220.636 s` | `126.720 s` | `0.084 s` | Search time is still overwhelmingly containment-path time. |

For the first completed structure in the second rerun (`181 (F1)` / `Bx_Tr_sim_target_dil_v2` / `DIL_RP`):

- Stage containment grandmother time was `2194.164 s` and stage grandmother mother-call time was `2194.146 s`. Wrapper-only overhead was effectively zero in the measured case.
- Stage grandmother chunk slicing time and concatenation time were both `0.0 s`, and internal grandmother chunking did not trigger (`stage_grandmother_chunked_call_count = 0`).
- Winner-validation chunk total was `122.820 s`, winner-validation grandmother mother-call time was `122.762 s`, isolated winner-validation benchmark time was `118.278 s`, and additional setup outside the chunk score was only about `3.901 s`.
- The stage workload was `1759` candidates x `257` rows = `452,063` test structures. The downstream-comparable winner validation handled `10,001` test structures. Stage search is slower primarily because it is doing about `45.2x` more work, not because `256` is somehow more expensive than `10,000`.
- The repeated call count is coming from optimizer-level chunking, not grandmother fallback chunking. With `optimizer_v2_max_candidates_per_chunk = 8`, the stage executed `220` mother calls.

### Nearest-z helper benchmark on proper-sized toy data

The current mother-function path still uses `nearest_zslice_vals_and_indices_all_structures_3d_point_arr_ver5`. A synthetic benchmark was run against proper chunk-shaped toy data and a grouped host-side searchsorted prototype that preserves the exact output contract.

Benchmark setup:

- Relative-structure slice counts were randomly generated in the range `18` to `64`.
- Stage-like cases used `257` relative structures repeated across `8` candidates, for `2056` test structures per mother call.
- Winner-like case used `10,001` relative structures with a one-to-one mapping, matching the downstream comparable trial count.
- The grouped prototype matched the existing helper outputs exactly under `np.allclose(..., rtol=0.0, atol=0.0)`.

| Scenario | Existing helper | Existing helper time | Grouped prototype time | Measured speedup |
| --- | --- | --- | --- | --- |
| Stage-like, `2056` test structures, `257` relative structures, `25` points per trial | `ver5` | `3.046 s` | `0.0183 s` | `166.0x` |
| Stage-like, `2056` test structures, `257` relative structures, `25` points per trial | `ver7` searchsorted mode | `2.959 s` | `0.0183 s` | `161.3x` |
| Stage-like, `2056` test structures, `257` relative structures, `51` points per trial | `ver5` | `3.054 s` | `0.0217 s` | `140.7x` |
| Stage-like, `2056` test structures, `257` relative structures, `51` points per trial | `ver7` searchsorted mode | `3.016 s` | `0.0217 s` | `138.9x` |
| Winner-like, `10,001` test structures, `10,001` relative structures, `25` points per trial | `ver5` | `15.073 s` | `0.611 s` | `24.7x` |

This does not prove the whole mother function will speed up by the same factor. It does prove that the current nearest-z helper is spending a large amount of time on repeated tiny per-structure GPU transfers and launches, and that the underlying nearest-slice arithmetic itself can be done much faster while preserving exact outputs.

## Revised Ranked Performance Recommendations

The estimates below are rough overall wall-clock expectations if the named slice is a major share of the run. They are meant to prioritize implementation order, not to promise exact gains.

| Rank | Recommendation | Primary code locations | Expected gain | Confidence | Why this is promising now |
| --- | --- | --- | --- | --- | --- |
| 1 | Replace the nearest-z helper with a grouped or cached searchsorted path that preserves the current output contract exactly. | `polygon_dilation_helpers_numpy.py`, `custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.py` | High inside containment, with potentially material stage-level improvement | High for helper-level gain, medium-high for end-to-end gain | Proper-sized toy data showed `24.7x` to `166x` helper-level speedups with exact output agreement. This is now measured evidence, not speculation. |
| 2 | Increase work packed into each containment call by retuning `optimizer_v2_max_candidates_per_chunk` upward from `8` after each change is revalidated. | `biopsy_optimizer/v2/runner.py`, `biopsy_optimizer/v2/scoring.py`, optimizer-v2 config in `biopsy_localization_convex_main.py` | Medium to high, roughly `1.2x` to `2.5x` overall | High | The measured stage executed `220` mother calls with no internal grandmother chunking. Larger optimizer chunks should amortize fixed prepper and helper costs better until GPU memory becomes the real limit. |
| 3 | Cache repeated prepper inputs and device-side metadata inside the mother/prepper path instead of rebuilding and re-uploading identical per-structure inputs on every call. | `custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.py` | Medium to high overall if prepper work is substantial inside mother-call time | Medium-high | The wrapper is cleared. Almost all remaining containment time is inside the mother-function body, where repeated Python packing and array construction still occur. |
| 4 | Remove the CuPy -> NumPy boundary before containment and keep the aligned batch on-device as long as possible. | `preprocessing/localization_transformer.py`, `biopsy_optimizer/v2/scoring.py`, `preprocessing/containment_runner.py` | Medium to high overall | High | This is still a clear avoidable boundary in the optimizer path. It just no longer appears to be the first thing to investigate before the mother-function internals. |
| 5 | Vectorize candidate self-transform and relative-localization work more aggressively to remove candidate-by-candidate Python loops. | `preprocessing/localization_transformer.py` | Medium to high overall if localization remains visible after containment-path fixes | High | This remains a credible optimizer-side improvement, but current live data says containment-path work is the first limiter. |
| 6 | Make tested/ranked dataframe construction optional or deferred for throughput runs. | `biopsy_optimizer/v2/scoring.py`, `biopsy_optimizer/v2/runner.py`, `biopsy_optimizer/v2/output.py`, `biopsy_optimizer/v2/live_integration.py` | Medium, roughly `1.1x` to `1.5x` overall | Medium-high | This is still safe overhead removal, but it is now clearly behind containment-path work in priority. |
| 7 | Keep score reductions on the device until final winner extraction. | `biopsy_optimizer/v2/scoring.py` | Low to medium overall | Medium | This remains worthwhile cleanup, but it is unlikely to beat the measured containment-path opportunities. |
| 8 | Only after the upstream containment fixes are exhausted, and only with explicit approval, inspect raw-kernel FP64 and branch-heavy work inside the custom containment kernel. | `custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.py` | Potentially high, but uncertain and higher risk | Low to medium | The evidence now says the wrapper is not the problem. The raw kernel may still matter, but correctness risk is higher there and the nearest-z/prepper path has much stronger measured evidence first. |

## Recommendations I Am Most Confident In

1. Keep raw-kernel code read-only for now.
2. Replace the nearest-z helper behind the exact same return contract and verify equality against the current helper on recorded or synthetic batches.
3. Rerun one short live optimizer-v2 case immediately after the helper change to see how much of mother-call time actually moves.
4. If containment time drops materially, increase `optimizer_v2_max_candidates_per_chunk` above `8` until memory, not orchestration overhead, becomes the limiting factor.

## Suggested Decision Sequence

1. Leave the raw kernel untouched until the helper-level changes are measured live.
2. Implement the nearest-z redesign as a contract-preserving swap, ideally with an equality check harness against `ver5` during development.
3. Rerun one short optimizer-v2 live case and inspect the first `optimizer_v2.structure.search.end` event plus the winner-validation benchmark event.
4. If stage containment drops, retune chunk packing next.
5. If containment is still dominant after that, cache more prepper inputs inside the mother function.
6. Only then consider raw-kernel edits, and only with explicit approval.

## Interpretation Notes

- The expected gains are not additive.
- The helper benchmark measured only the nearest-z slice, not the entire mother function.
- The grouped redesign is especially attractive in staged search because the relative-structure mapping repeats across candidates inside a chunk.
- Even the one-to-one winner-like toy case favored the host-side searchsorted prototype strongly, which means the current helper overhead is not just a repeated-mapping artifact.
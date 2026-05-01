# Refactor Validation Checklist

Last updated: 2026-04-30

Purpose:

- track the scientific and behavioral validation work for the ongoing refactor,
- preserve completed checks instead of deleting them,
- keep one living place to add new checks or drift findings as they appear.

Conventions:

- `[x]` means the check or setup step is completed.
- `[ ]` means still pending.
- completed items should stay in the file for history unless they are clearly wrong.

## Current Status Snapshot

- [x] first-pass dtype policy module added and wired for optimizer-v1 plus transporter recovery
- [x] output runtime directory lifecycle extracted into a dedicated helper and initialized early for fresh, preprocessed-load, and results-load paths
- [x] focused syntax validation passed for the latest dtype-policy and output-runtime-directory changes
- [x] current Apr 30 validation run output directory identified: `MC_sim_out- Date-Apr-30-2026 Time-15,28,21 - 3 patient validation run`
- [x] current pickle-enabled validation run identified for the next round-trip check: `MC_sim_out- Date-Apr-30-2026 Time-18,22,05`
- [ ] current fresh validation rerun finishes with no exceptions
- [ ] current rerun outputs are compared against the Mar 3 reference run for drift

## What We Can Validate Now

- [x] distinguish exact-equality checks from sanity-and-drift checks
- [x] confirm that exact MC transform-array equality is not currently proven in the live rerun because CuPy RNG replay or saved transform-bank reuse is not instrumented right now
- [ ] complete a sanity-and-drift validation pass on the current rerun
- [ ] complete a pickle round-trip validation pass after the fresh rerun succeeds

## Current Rerun Checklist

- [ ] confirm the current rerun completes without exceptions or missing-field failures
- [x] record the output folder for the current rerun so it can be compared cleanly later
- [x] note the matching patient cases that should be compared against the Mar 3 reference run: `181 (F1)`, `181 (F2)`, `184 (F1)`
- [x] confirm the Apr 30 run and Apr 29 successful 3-patient run produced the same 101 CSV relative paths

### Representative Structural Comparisons

For at least one representative matching case, compare before versus current output for:

- [ ] `Raw contour pts zslice list`
- [ ] `Structure global centroid`
- [ ] `Reconstructed biopsy cylinder length (from contour data)`
- [ ] `Centroid line vec (bx needle base to bx needle tip)`
- [ ] `Reconstructed structure pts arr`

### Targeting And Downstream Checks

- [ ] compare `Target DIL by centroid dict`
- [ ] compare `Target DIL by surfaces dict`
- [ ] compare `Nearest DILs info dict`
- [ ] confirm downstream dataframe builders still consume those fields cleanly
- [ ] confirm later biopsy parsing stages still run without missing-field failures
- [ ] confirm sextant and sampled-biopsy downstream products still operate on realized final geometry

### Important Final-State Statistics And Sanity Checks

These are drift checks, not exact-equality claims, unless later RNG control is added.

- [ ] compare cohort biopsy counts against the Mar 3 reference run
- [ ] compare DIL counts against the Mar 3 reference run
- [ ] compare real versus simulated biopsy length summary behavior
- [ ] compare mean biopsy centroid variation and related global variation summaries
- [ ] compare patient-level tissue-class summary values for obvious drift
- [ ] compare patient-level dosimetry summary values for obvious drift
- [ ] compare patient-level MR summary values for obvious drift
- [ ] note any numbers that look materially wrong rather than merely noisy
- [ ] if serious drift is found, drill into patient-specific outputs before assuming a global regression

## Fresh Run Versus Pickle Round-Trip

### Fresh Run With Pickle Save Enabled

- [ ] rerun with preprocessed pickle export enabled
- [ ] confirm the fresh run with pickle save enabled completes cleanly
- [ ] record the exported preprocessed dataset directory used for the round-trip test

### Load The Preprocessed Dataset

- [ ] run again by loading the exported preprocessed dataset
- [ ] confirm the load-and-rebuild path completes cleanly
- [ ] compare fresh-run versus preprocessed-load outputs for deterministic structural fields
- [ ] compare fresh-run versus preprocessed-load downstream geometry-derived outputs
- [ ] compare processed-dataset render/debug behavior after load-rebuild versus fresh-run behavior

### Optional Results-Bundle Round-Trip

- [ ] export results pickles from a clean run
- [ ] load the exported results bundle in a separate validation run
- [ ] compare downstream dataframe generation versus the fresh-run results path
- [ ] compare downstream figure generation versus the fresh-run results path

## Exact Equality Checks Not Yet Instrumented

These are future instrumentation tasks, not claims for the current rerun.

- [ ] add machine-readable comparison-manifest machinery for transform-side shadow validation
- [ ] shadow-validate `generate_transformations(...)` exact equality under controlled CuPy RNG replay or equivalent saved-transform-bank reuse
- [ ] shadow-validate `biopsy_only_transformer(...)` exact equality
- [ ] shadow-validate `biopsy_transformer_to_relative_structures(...)` exact equality
- [ ] record equality status, shape, dtype, and mismatch summaries in a manifest

## Modular Seams To Keep Watching

- [x] simulated biopsy planning is still called through the modular seam in main
- [x] preprocessed export is routed through the pickle tools module
- [x] preprocessed load-rebuild is routed through the pickle tools module
- [x] processed-dataset render/debug is routed through the modular render/debug surface
- [ ] confirm those seams remain scientifically behavior-neutral on the current validation dataset

## Remaining Core Refactor Work

- [ ] close validation on the current rerun
- [ ] compare current outputs against the older matching patients and the Mar 3 reference outputs
- [ ] add the late realized-targeting pass while keeping the early intended-targeting pass unchanged
- [ ] add the target-agreement validator and Rich warning output
- [ ] add transport-selection diagnostics and rank-retention metadata for optimal transports
- [ ] continue the dataframe utility modularization via dedicated dataframe-focused subfolder and compatibility shim strategy
- [ ] continue the dtype-policy rollout beyond optimizer-v1 and transporter

## V2 Optimizer Checklist

### Current Gaps

- [ ] add shared transform-bank generation
- [ ] add chunked candidate-trial scoring
- [ ] add ranked candidate dataframe emission
- [ ] add transport integration
- [ ] keep v2 activation compatible with memory-safe operation when total simulated biopsy count grows

### Required Validation Categories

- [ ] validate candidate prune correctness
- [ ] validate trialwise mapping correctness
- [ ] validate equality of reused transform bank between optimizer and downstream MC
- [ ] validate chunked evaluation equivalence versus a tiny unchunked reference case
- [ ] validate ranking stability as the trial prefix increases
- [ ] validate transport compatibility with ranked candidate output

## Optional Sidequests

- [ ] write structured validation manifests to disk for the current rerun instead of relying only on manual inspection
- [ ] add a lightweight comparison script for current run versus Mar 3 outputs at patient and cohort levels
- [ ] add a clean canonical biopsy-targeting manifest dataframe for downstream joins
- [ ] add deterministic sampler reuse checks for planning versus realized geometry
- [ ] add a small serious-drift threshold table so clearly bad numbers are flagged immediately
- [ ] add a short validation-notes section at the bottom of this file after each important rerun
- [ ] add a dataset chunking plan for larger cohorts or multi-family simulated-core runs
- [ ] add an output-stitching surface so chunked runs can be recombined into one cohort result safely

## Notes To Add During Validation

- [x] current rerun notes added
- [ ] Mar 3 comparison notes added
- [ ] pickle round-trip notes added
- [ ] v2 validation notes added

## Validation Notes

### 2026-04-30: Apr 30 Three-Patient Validation Run

- Run inspected: `MC_sim_out- Date-Apr-30-2026 Time-15,28,21 - 3 patient validation run`
- User stopped this run during guidance-map production because it was taking too long.
- Even though the run was stopped, the output surfaces already present include `Output CSVs`, `Output figures`, `Raw MC output`, and a pickled-data subdirectory.
- The Apr 30 run and the Apr 29 successful 3-patient run have the same 101 CSV relative paths.
- Only 21 of those 101 common CSVs are hash-identical between Apr 30 and Apr 29.
- Section-level hash comparison versus Apr 29:
	- `Preprocessing`: 27 total, 6 identical, 21 changed
	- `MC simulation`: 51 total, 11 identical, 40 changed
	- `Cohort`: 23 total, 4 identical, 19 changed
- Exact preprocessing matches confirmed across Apr 30 versus Apr 29 for all three current cases:
	- `Selected structures.csv`
	- `Simulated biopsy preparation dataframe.csv`
- Mar 3 reference run contains overlapping patient/fraction CSV subtrees for `181 (F1)`, `181 (F2)`, and `184 (F1)` in `Preprocessing`, `MC simulation`, and `FANOVA simulation`.
- First targeted content check on `Cohort: Nearest DILs to each biopsy.csv`:
	- Apr 30 versus Apr 29 has the same 78 subset rows for the three current cases and the same merge keys.
	- Despite that, 24 matched rows show changes in centroid/surface-distance-related fields.
	- Largest observed absolute diffs in that first pass were about:
		- `BX to DIL centroid (X)`: `1.00000002`
		- `BX to DIL centroid (Y)`: `1.240738`
		- `BX to DIL centroid (Z)`: `3.0`
		- `BX to DIL centroid distance`: `1.8492993`
		- `NN surface-surface distance`: `1.5459466`
- Initial interpretation:
	- some deterministic preprocessing surfaces remain stable,
	- but target-related and optimizer-related preprocessing outputs are still drifting and need deeper inspection before signoff.

### 2026-04-30: Refined Preprocessing Drift Readout

- A refined composite-key comparison on `Nearest DILs info dataframe.csv` used:
	- `Bx ID`
	- `Bx refnum`
	- `Bx index`
	- `Relative DIL ID`
	- `Relative DIL ref num`
	- `Relative DIL index`
- Under that proper key, there were no left-only or right-only rows across Apr 30 versus Apr 29 for the three current cases.
- Under that proper key, all changed `Nearest DILs info` rows were `Simulated type == Optimal DIL`.
- Under that proper key, zero rows changed `Target DIL (by centroids)` or `Target DIL (by surfaces)` flags.
- The observed `Nearest DILs info` drift is therefore in geometry-derived centroid and surface-distance columns, not in target-identity flag flips.
- A keyed comparison on `Biopsy basic spatial features dataframe.csv` also found changed rows only in `Simulated type == Optimal DIL`.
- `DIL centroids optimal targeting dataframe.csv` kept stable centroid target coordinates across Apr 30 versus Apr 29 and only showed small containment-score differences.
- `Optimal DIL targeting dataframe.csv` is the earliest inspected layer where chosen target coordinates themselves drift.
- Those target-coordinate changes were discrete lattice-step moves such as:
	- about `1.0` mm plane-index moves in `X`, `Y`, or `Z`,
	- and one larger inspected case with about `3.0` mm in `Z`.
- Current interpretation after the refined readout:
	- real biopsy rows look stable in the inspected preprocessing CSVs,
	- centroid-target rows look stable in coordinates and only drift in containment counts,
	- the remaining drift signal is confined to the stochastic optimizer-v1 optimal-core path.

### 2026-04-30: Optimizer-v1 Stochasticity Note

- The inspected optimizer-v1 helper path is not deterministic unless RNG is explicitly controlled.
- In `biopsy_optimizer_module_v1_helpers.py`, the optimal-DIL scorer draws containment clouds using `cp.random.normal(...)` and then selects:
	- maximum `Number of normal dist points contained`,
	- then minimum `Dist to DIL centroid`,
	- then `sample()` if ties remain.
- That means small run-to-run containment-score differences are expected.
- Because the chosen optimal lattice point is selected from that score surface, discrete optimal-target coordinate drift is also expected when the RNG is uncontrolled.
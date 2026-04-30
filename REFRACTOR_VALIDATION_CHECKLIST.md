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
- [ ] current fresh validation rerun finishes with no exceptions
- [ ] current rerun outputs are compared against the Mar 3 reference run for drift

## What We Can Validate Now

- [x] distinguish exact-equality checks from sanity-and-drift checks
- [x] confirm that exact MC transform-array equality is not currently proven in the live rerun because CuPy RNG replay or saved transform-bank reuse is not instrumented right now
- [ ] complete a sanity-and-drift validation pass on the current rerun
- [ ] complete a pickle round-trip validation pass after the fresh rerun succeeds

## Current Rerun Checklist

- [ ] confirm the current rerun completes without exceptions or missing-field failures
- [ ] record the output folder for the current rerun so it can be compared cleanly later
- [ ] note the matching patient cases that should be compared against the Mar 3 reference run

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

## Notes To Add During Validation

- [ ] current rerun notes added
- [ ] Mar 3 comparison notes added
- [ ] pickle round-trip notes added
- [ ] v2 validation notes added
# Standalone Containment Package Design

This note describes the intended future extraction of the custom one-to-one GPU point-in-polygon stack into its own repository and installable Python package.

No runtime code is moved by this note.

## Scope

The future package should own the custom CUDA point-containment stack, not just a thin wrapper around one kernel.

That includes:

1. raw CUDA kernels,
2. Python preparation and launch wrappers,
3. the high-level mother-function API,
4. a future generalized batched containment executor,
5. correctness tests, benchmarks, and diagnostic tooling.

The future package should not own biopsy-localization workflow code, optimizer-v1 or optimizer-v2 policy, or repo-specific dataframe/reporting surfaces that only exist for this project.

## Why This Deserves Its Own Repo

The package value is not merely "GPU point in polygon" in the abstract.

The stronger technical claim is:

1. row-aligned or one-to-one containment is a first-class workflow,
2. many existing APIs expose all-points-versus-all-polygons style outputs,
3. in aligned simulation workloads that can waste memory and work,
4. this stack is designed around structured one-to-one or row-aligned testing and the preparation logic needed to support it efficiently.

That makes the containment stack reusable beyond this biopsy-localization repo.

## Recommended End State

Recommended long-term shape:

1. one standalone GitHub repository,
2. one installable Python package,
3. a small public API surface,
4. optional extras for diagnostics and plotting,
5. this repo importing that package as a dependency.

That is preferable to growing an in-tree `custom_PIP` runtime package here.

This folder should remain a planning area only unless there is a temporary local prototyping need.

## Proposed Package Boundaries

Suggested internal layout:

1. `kernels/`
   raw CUDA kernels and launch-specialized helpers
2. `prep/`
   polygon normalization, closure handling, duplicate-point handling, offset/index packing, contiguity checks
3. `core/`
   mother-function orchestration and the stable runtime API
4. `batching/`
   the future generalized batched containment executor
5. `adapters/`
   optional dataframe helpers and convenience formatting layers
6. `diagnostics/`
   logging, debug formatting, plotting helpers, edge-log readers
7. `benchmarks/`
   reproducible performance comparisons
8. `tests/`
   correctness, regression, contract, and benchmark-smoke tests

The key point is that batching belongs with containment, not inside optimizer v2.

## Proposed Public API Direction

The public API should stay smaller than the current monolithic source file.

Recommended first-class surfaces:

1. a mother-function style entry point for aligned containment,
2. explicit prep functions for callers that want lower-level control,
3. a batched executor for row-aligned 3D test arrays,
4. pairwise or row-aligned result metadata helpers,
5. optional dataframe adapters kept outside the hot path.

The package should prefer explicit typed contracts and shape assumptions over convenience magic.

## Testing Plan

The extraction should preserve the existing trusted behavior and make that trust easier to demonstrate.

Minimum test surfaces:

1. curated geometric edge cases:
   boundary points, vertex hits, horizontal and vertical edges, tiny edges, out-of-z-extent points, polygon closure assumptions
2. wrapper contract tests:
   2D versus 3D test-array paths, row-to-structure mapping, shape validation, contiguous-memory requirements
3. batching tests:
   chunked results must match an equivalent unchunked reference on small inputs
4. regression tests:
   freeze representative expected outputs from already-trusted workloads
5. benchmark tests:
   reproducible throughput and memory comparisons across problem sizes

## Benchmarking Direction

The package should benchmark itself against the actual alternative workflow it is meant to beat.

That likely means comparing:

1. aligned or one-to-one containment through this package,
2. all-points-versus-all-polygons style containment plus the cost of extracting only the needed aligned results,
3. different batch sizes and memory ceilings,
4. realistic structured workloads rather than only synthetic random polygons.

The benchmark story is strongest when it measures both throughput and memory behavior.

## Licensing Direction

This parent repo currently uses a non-commercial research license.

The future standalone package must choose its own license deliberately at extraction time.

Real options depend on the intended adoption model:

1. if the goal is broad open adoption with minimal friction, a permissive license such as Apache-2.0 is attractive,
2. if the goal is to preserve a clear commercial pathway, a research or source-available model may be more appropriate,
3. if both matter, dual-licensing is worth considering.

The important point is not to leave the standalone package unlicensed or ambiguously licensed.

It should launch with:

1. a top-level license,
2. an explicit copyright notice,
3. clear wording on commercial versus non-commercial use if relevant,
4. package metadata that matches that license choice.

## Extraction Strategy

Recommended low-risk sequence:

1. leave the current runtime code untouched while optimizer-v2 work is active,
2. document the intended package boundary now,
3. continue building new shared batching abstractions against that future boundary,
4. when extraction starts, split the monolithic containment file by responsibility before polishing public packaging,
5. only after that stabilize the package API and publish it.

This keeps current project momentum while avoiding a rushed packaging job.

## Relationship To This Repo

This biopsy-localization repo should remain a downstream consumer once extraction happens.

That means:

1. no optimizer-v2-specific logic should leak into the standalone containment package,
2. new shared containment helpers should be designed so they can migrate cleanly later,
3. future imports in this repo should target the standalone package boundary rather than reintroducing local coupling.
# custom_PIP

This folder is a planning stub only.

No runtime CUDA containment code has been moved here yet.

## Current Decision

The long-term target is a separate standalone repository and package for the custom one-to-one GPU point-in-polygon stack.

That future package should own:

1. the raw CUDA kernels,
2. the Python prep and wrapper layer,
3. the high-level mother-function API,
4. the future generalized batched containment executor,
5. package-local tests, benchmarks, and diagnostics.

This biopsy-localization repo should eventually consume that package as a dependency rather than continue owning the containment stack directly.

## Why A Separate Repo

The containment stack is broader than optimizer v2 and broader than this repo's biopsy workflow.

Its core value proposition is a row-aligned or one-to-one GPU containment workflow, which is different from the more general all-points-versus-all-polygons style APIs exposed by existing spatial libraries.

That makes it a reusable infrastructure asset, not just an internal helper.

## Near-Term Plan

1. leave the existing containment implementation in place for now,
2. avoid risky runtime refactors while optimizer-v2 work is active,
3. keep new shared batching abstractions designed against the future standalone package boundary,
4. keep a running adoption audit of direct mother-function callers so grandmother migrations can be done later without rediscovering the boundary,
5. when extraction starts, split the current monolithic containment file into kernels, wrappers, batching, adapters, and diagnostics.

## Packaging Direction

Recommended long-term shape:

1. standalone GitHub repository,
2. installable Python package,
3. explicit license at the package root,
4. reproducible correctness tests and benchmarks,
5. this repo importing that package rather than vendoring the implementation.

See [STANDALONE_PACKAGE_DESIGN.md](STANDALONE_PACKAGE_DESIGN.md) for the fuller extraction and licensing plan.

That design note now also records the current two-mode input contract and the current grandmother-adoption audit.

## License Note

This parent repo currently uses a non-commercial research license.

The future standalone package should choose its own license deliberately at extraction time based on the intended adoption model.

It should not launch without:

1. a top-level license,
2. explicit copyright ownership,
3. package metadata that matches the license choice,
4. clear wording on commercial versus non-commercial use if that distinction is retained.
# Pickle Export Boundaries

The repo now treats pickle export as an explicit boundary contract instead of a best-effort dump of whatever happens to be in memory.

## Current boundaries

- `preprocessed` means `post_preprocessing`.
- `results` means `post_results`.

Both modes still sanitize known non-picklable runtime objects such as Open3D meshes and KD-tree style helpers.

## Preprocessed bundle contract

The `preprocessed` bundle is intended to support the existing skip-preprocessing load path.

It keeps:

- structure geometry and derived arrays required to rebuild runtime-only objects
- dose and MR lattices needed by `rebuild_loaded_preprocessed_runtime_objects(...)`
- preprocessing metadata and preprocessing-scope summary dataframes

It explicitly excludes:

- optimizer-v1 and optimizer-v2 outputs under `Biopsy optimization - ...`
- the optimizer-v1 lattice dataframe stored under `Biopsy optimization: Optimal biopsy location (entire cubic lattice) dataframe`
- optimizer-selected transport requests
- Monte Carlo payloads under `MC data:`
- FANOVA payloads under `FANOVA:`

This makes the bundle semantically stable even if the export call occurs after optimizer work has already populated in-memory dictionaries.

## Results bundle contract

The `results` bundle is for plain-data post-run artifacts that are still useful after the live session ends.

It currently keeps optimizer summary/ranked/tested dataframes and transport request metadata, but excludes the live stage-boundary render-job payload.

That render payload should not be treated as a durable replay contract.

## Render replay boundary

The first render manifest should be optimizer-v2 specific rather than a generic rendering manifest for the entire repo.

That keeps the contract narrow and avoids coupling unrelated rendering flows such as:

- post-pickle-load full-dataset rendering
- downstream MC stochastic rendering
- any future non-optimizer rendering modules

Optimizer-v2 replay should be modeled as:

1. existing pickle bundle load and runtime rebuild
2. explicit optimizer-v2 render manifest load
3. shared replay/regeneration functions used by both the in-run broker and the offline renderer

The manifest is the durable contract for replay. The live queued render context is not.

## Config ownership

Pickle load currently mixes two sources of truth:

- config already serialized into `master_structure_info_dict`
- live runtime config passed back into `rebuild_loaded_preprocessed_runtime_objects(...)`

That is the next architecture boundary to make explicit.

The model should be:

### Frozen-with-bundle config

These settings define the meaning of the preprocessed dataset and should travel with the pickle bundle, or be reconstructed from bundle metadata and then enforced during load.

Examples:

- preprocessing interpolation distances
- mesh-rebuild parameters that affect regenerated geometry from stored arrays
- any future optimizer-v2 replay inputs that define candidate generation/search semantics rather than presentation

If these differ at load time, we should either:

- ignore the runtime value and use the bundle value
- or fail loudly with a config mismatch message

### Runtime-overridable config

These settings do not change the semantic contents of the saved dataset and may be supplied by the current session.

Examples:

- display thresholds for dose and MR point clouds
- purely presentational color/visual settings
- export format and render-backend choices for replayed scenes

### Immediate implication

`rebuild_loaded_preprocessed_runtime_objects(...)` currently still receives a mixed bag of both frozen and runtime-overridable settings. That should be split into two explicit config objects later:

1. bundle-frozen rebuild config
2. runtime visualization config

## Next implementation step

The highest-value next steps are:

1. move the preprocessed export call to the real post-preprocessing boundary
2. define the bundle-frozen rebuild config contract for pickle load
3. add the optimizer-v2-specific render manifest and shared replay module
4. only then decide whether a separate `results` bundle is needed
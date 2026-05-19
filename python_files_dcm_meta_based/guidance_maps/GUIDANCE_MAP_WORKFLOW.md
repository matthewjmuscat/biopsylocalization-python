# Guidance Map Workflow Boundary

Purpose:

- make guidance-map planning callable after simulated-core finalization in the current cohort pipeline,
- keep the non-plotting planning tables separate from figure rendering,
- preserve a future caller surface for prospective planning tools that operate on one patient or a small patient set.

## Current Module Surface

- `guidance_maps.config.GuidanceMapPlanningConfig`
- `guidance_maps.planning.precompute_guidance_map_firing_depth_recommendations_for_run(...)`
- `guidance_maps.planning.GuidanceMapPlanningResult`

The planning function writes the same legacy dataframe keys as the inline code it replaced:

- patient table: `Biopsy optimization - Guidance-map firing depth recommendations dataframe`
- cohort table: `Cohort: Guidance-map firing depth recommendations dataframe`

It also records runtime timing under `guidance_maps.precompute` when a `RuntimeLogger` is provided.

## Current Cohort Caller

`biopsy_localization_convex_main.py` calls the planning module after DIL optimization outputs are built and before guidance-map rendering. This preserves the existing behavior: planning tables are generated before any plotter tries to consume precomputed guidance data.

Rendering is still routed through `startup.guidance_map_workflow.render_guidance_maps_for_run(...)`. That render workflow already keeps plotting outside the main pipeline body, but a later cleanup should expose it through this package so callers can import one guidance-map namespace.

## Future Prospective Caller Shape

A standalone prospective planning entrypoint should eventually do this:

1. load one patient or a small patient set through the input-manifest/routing-profile layer,
2. build the same patient geometry dictionaries used by the cohort pipeline,
3. finalize target cores or user-selected target points,
4. call `precompute_guidance_map_firing_depth_recommendations_for_run(...)`,
5. render/export guidance maps through a UI-neutral render service.

The prospective caller should not depend on cohort CSV export, rich progress UI, or retrospective study aggregation. It should only require the prepared patient geometry store, target-selection outputs, and a guidance-map config.

## Next Refactor Steps

- move or wrap `startup.guidance_map_workflow.render_guidance_maps_for_run(...)` behind `guidance_maps.rendering`,
- add a smaller one-patient planning function once the prepared patient context wrapper exists,
- define a guidance-map input contract that lists required patient dictionary keys and dataframe columns,
- keep legacy dataframe names until output-artifact validation proves a canonical schema can be added safely.
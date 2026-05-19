# Output Schema Registry Guide

## What Is Temporary And What Should Last

The Phase 3B and Phase 3C names are transitional validation scaffolding. They exist because the legacy pathway is still authoritative while the patient-scoped pathway proves it can recreate current outputs.

The schema registry is different. The current Python implementation can evolve, but the concept should remain permanent: every durable output table should have a reviewed contract that says what it is, what one row means, how it joins to other tables, where it comes from, how it is validated, and whether it should remain in the core output surface.

When the legacy pathway is replaced, the phase-specific output folders can be renamed into clearer production names. The registry should continue as the output schema contract for the pipeline, GUI, validation reports, and downstream analyses.

## What Gets A Registry Entry

Add or update a registry entry when a dataframe is a durable output surface:

- it is written to disk,
- it is consumed by another repository,
- it is intended for GUI display,
- it is part of a validation report,
- it is a post-run analysis artifact that should be stable across runs.

Do not add registry entries for short-lived scratch dataframes unless they become one of those durable surfaces.

## Patient Fragment Route Pattern

New patient-fragment routes should follow the existing legacy formalism as much as possible while the old pathway is still present.

For each route:

1. Identify the legacy dataframe builder or final dataframe construction block.
2. Extract or wrap a patient-level builder function when the current code only builds the final cohort table.
3. Store the patient-level dataframe under a clear key in the appropriate existing dictionary:
   - preprocessing patient tables: `pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"]`,
   - MC patient tables: `pydicom_item[all_ref_key]["Multi-structure MC simulation output dataframes dict"]`,
   - per-biopsy MC tables: `sp_bx["Output data frames"]`.
4. Use a stable key that corresponds to the registry `source_fragment_table_id` and still maps clearly to the legacy table name.
5. Add or update the Phase 3B/3C exporter iterator only if the dataframe is not already picked up by the generic dictionary traversal.
6. Stitch the patient fragments by the registry `stitch_method`.
7. Validate the stitched table against the legacy final table before changing the registry status from `needs_phase3d_route` or `needs_live_phase3c_validation` to `validated_phase3c`.

This keeps traceability close to the current code style: dataframe builder, named dictionary storage, manifest/export route, registry contract, validation evidence.

## How To Use The Registry During Development

When adding or changing an output table:

1. Decide whether it is a durable output or temporary scratch data.
2. If durable, add or update its `OutputTableSpec` in `schema_registry.py`.
3. Define the exact row grain before choosing keys. Avoid vague grains such as `summary` when the row is really one biopsy, one voxel, one structure, one MC trial, or one dose bin.
4. Use canonical keys for joins. `Bx refnum` and related refnum columns can remain legacy/source attributes, but they should not become canonical join keys.
5. Set `stitch_method` honestly:
   - `concat_rows` for simple patient-row stitching,
   - `concat_current_summary_fragments` for current summary fragments that are already patient scoped,
   - `aggregate_from_long_form` when the final table must be rebuilt from lower-level rows,
   - `join_derived` when the table is a derived join across multiple source tables,
   - `recompute_downstream` for outputs like DVH metrics that should eventually move to a clean post-run service,
   - `manifest_metadata` for run metadata.
6. Set `validation_status` conservatively. Do not mark a table validated until a stitched/rebuilt table has been compared to the legacy output.
7. Run the Phase 3C coverage report and inspect the generated data dictionary.

## Human-Readable Dictionary

The registry is the machine-readable source of truth. The human-readable views are generated from it:

- `output_schema_data_dictionary.csv`,
- `output_schema_data_dictionary.md`.

The Markdown dictionary is intentionally verbose. It includes legacy name and location, scope/family, row grain, canonical key, legacy keys, source stage, source fragment, stitch/build method, aggregation builder, validation status, retention policy, downstream usage, storage/column policy, usage notes, and implementation next step.

Do not hand-edit the generated dictionary. Improve `OutputTableSpec` values or the dictionary renderer instead.

## Documentation Standard Going Forward

New modules and important public functions should have docstrings that explain purpose, inputs, outputs, and validation assumptions. Use short line comments when code is doing something non-obvious, especially around:

- configuration values and defaults,
- legacy compatibility choices,
- canonical key decisions,
- validation comparisons,
- patient-fragment to cohort-stitch boundaries,
- places where a table is retained only for legacy validation.

Avoid comments that repeat the code. Prefer comments that explain why a choice exists or what can safely change later.

## Config Area Reminder

The configuration area still needs the same treatment. A future config pass should document:

- which settings are scientific inputs,
- which settings are runtime/performance controls,
- which settings are validation/debug toggles,
- which settings are legacy compatibility switches,
- which settings should eventually become GUI-facing or manifest-stamped.

That pass should add docstrings or structured config metadata before broad refactoring, because configuration meaning is part of the run contract.

## Recently Wired Route Targets

The first four simple-concat route targets have been wired through patient-level preprocessing dataframe fragments and stitch pairs:

- `Cohort: 3D radiomic features all OAR and DIL structures`,
- `Cohort: Per voxel prostate double sextant classification`,
- `Cohort: Per sample point prostate double sextant classification`,
- `Cohort: Simulated biopsy planned vs realized centroid variation validation`.

The May 19, 2026 full Phase 3C run confirmed all four stitched outputs match the legacy final tables, so their cohort registry specs are now marked `validated_phase3c`.

## May 19, 2026 Simplification Pass

After the sister-repository scan and user review, the current registry was
reduced to 62 specs by removing outputs that should not remain core patient-runner
surfaces:

- `Cohort: DIL global tissue scores and DIL features` - derived join; regenerate downstream if needed.
- `Cohort: tissue volume above threshold` and its biopsy fragment - downstream-calculable threshold summary.
- `Cohort: Bx DVH metrics` and its legacy patient fragment - deprecated old DVH surface superseded by generalized DVH metrics during migration.

`Cohort: Simulated biopsy planned vs realized centroid variation validation`
remains registered and validated, but is marked validation-only and should live
under validation outputs rather than normal cohort CSV export.
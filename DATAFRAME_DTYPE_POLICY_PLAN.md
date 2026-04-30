# Dataframe Dtype Policy Plan

## Goal

Preserve the repo's memory-saving dataframe compression strategy while making numeric semantics explicit and safe for scientific calculations, ranking, filtering, and geometry selection logic.

This plan addresses the failure mode where a low-cardinality numeric column is converted to categorical, then later consumed by code that expects ordered numeric behavior such as `.min()`, `.max()`, sorting, thresholding, or arithmetic.

## Scope

This plan is repo-wide.

The immediate audit focus is:

1. `python_files_dcm_meta_based/dataframe_builders.py`
2. `python_files_dcm_meta_based/biopsy_optimizer/v1/biopsy_optimizer_module_v1.py`
3. `python_files_dcm_meta_based/biopsy_optimizer/v1/biopsy_optimizer_module_v1_helpers.py`
4. `python_files_dcm_meta_based/biopsy_optimizer/v2/biopsy_optimizer_module_v2.py`
5. `python_files_dcm_meta_based/custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.py`
6. `python_files_dcm_meta_based/advanced_guidance_map_creator.py`
7. downstream consumers such as `python_files_dcm_meta_based/biopsy_transporter.py`

## Current Helper Behavior

Current compression is driven by `convert_columns_to_categorical_and_downcast(...)` in `python_files_dcm_meta_based/dataframe_builders.py`.

Current behavior:

1. A column is converted to categorical only when its uniqueness ratio is less than or equal to the threshold.
2. Float columns are excluded from categorical conversion by default via `ignore_types=(np.floating,)`.
3. Columns can be excluded from categorical conversion by explicit name using `do_not_convert_column_names_to_categorical`.
4. Numeric columns that remain numeric are downcasted to smaller integer or float dtypes.

Important consequence:

1. Low-cardinality integer columns can still become categorical.
2. Those columns may still be semantically numeric even if they are not measurements.
3. Float columns remain numeric, but float downcasting can still reduce precision.

## Risk Model

### Categorical Conversion Risk

Categorical conversion usually does not lose the observed values themselves.

The main risk is operational and semantic:

1. unordered categoricals do not support some numeric operations,
2. ranking and filtering logic can fail,
3. future consumers can mistake stored labels for true numeric series,
4. adding new values or comparing across recovered series can become awkward.

### Downcast Risk

Integer downcasting is usually safe for currently observed values.

Float downcasting can lose precision.

This matters most for:

1. coordinates,
2. distances,
3. dose,
4. ADC values,
5. proportions and probabilities,
6. geometry-derived measurements,
7. optimizer objective values.

## Policy Direction

### Primary Policy

Scientific data and algorithm-driving numeric columns must remain numeric.

This includes both:

1. physical or scientific measurements,
2. numeric columns that control selection, ranking, filtering, indexing, or aggregation.

### What Can Remain Categorical

The following are generally safe to keep categorical:

1. patient identifiers,
2. structure identifiers and labels,
3. structure-type labels,
4. template-hole labels,
5. categorical flags or display labels,
6. other grouping keys that are not used arithmetically.

### What Should Usually Stay Numeric

The following should usually be protected from categorical conversion:

1. coordinates,
2. distances,
3. depths,
4. counts used in optimization,
5. ranks,
6. plane indices when used for grouping or ordering math,
7. trial indices when used algorithmically,
8. proportions, probabilities, and scores,
9. voxel or point indices when they participate in numeric joins, sorting, or range logic.

## Where Policy Should Live

Recommended policy shape:

1. schema rules should be defined at top level, not handwritten inline at every call site,
2. call sites should still pass those rules explicitly into the compression helper,
3. do not hide repo-specific schema behavior inside the generic compression helper itself.

Recommended implementation direction:

1. create one shared schema-policy module for dataframe families,
2. place that module in a dedicated dataframe-focused subfolder rather than a generic misc module,
3. if `convert_columns_to_categorical_and_downcast(...)` is relocated, copy it into that new module first and keep a compatibility shim or re-export at the old import path until callers are migrated,
4. store explicit exception lists and precision rules there,
5. import those named constants at producer call sites,
6. pass them explicitly to the compression helper.

This keeps call sites readable while avoiding duplicated ad hoc lists.

The staged move matters because `dataframe_builders.py` is already a dependency surface. A copy-plus-shim migration avoids breaking imports while still letting the repo converge on a cleaner dataframe utilities boundary.

### Preferred Shape

Preferred:

1. `OPTIMIZER_V1_NEVER_CATEGORICAL_COLUMNS`
2. `GUIDANCE_MAP_CANDIDATE_FIRING_NEVER_CATEGORICAL_COLUMNS`
3. `GUIDANCE_MAP_CANDIDATE_GEOMETRY_NEVER_CATEGORICAL_COLUMNS`
4. optional future `*_NEVER_DOWNCAST_COLUMNS`

Not preferred:

1. silently hard-coding special cases inside `convert_columns_to_categorical_and_downcast(...)`
2. scattering one-off inline column lists at every call site.

## Recovery Helper Strategy

Compression should not require whole-dataframe rehydration.

Recommended standard helper family:

1. single-series numeric resolver,
2. single-series integer resolver,
3. small multi-column temporary numeric resolver.

Design rule:

1. helpers should be non-mutating by default,
2. the stored dataframe remains compressed,
3. only the needed series or local working slice is resolved temporarily.

### Intended Uses

Single-series numeric resolver:

1. `.min()`
2. `.max()`
3. thresholding
4. sorting
5. arithmetic on one column

Single-series integer resolver:

1. ranks,
2. counts,
3. row indices,
4. candidate indices,
5. plane indices.

Multi-column resolver:

1. local ranking blocks,
2. geometry selection blocks,
3. calculations requiring several columns to remain aligned.

## Audit Findings

### 1. Core Compression Helper

File:

1. `python_files_dcm_meta_based/dataframe_builders.py`

Findings:

1. the helper already supports explicit name-based categorical exclusions,
2. the helper already protects floats from becoming categorical,
3. the helper does not currently distinguish between `never categorical` and `never downcast`,
4. low-cardinality integer columns remain the main current hazard.

### 2. Optimizer V1 Candidate Dataframes

Files:

1. `python_files_dcm_meta_based/biopsy_optimizer/v1/biopsy_optimizer_module_v1.py`
2. `python_files_dcm_meta_based/biopsy_optimizer/v1/biopsy_optimizer_module_v1_helpers.py`

This is the highest-risk producer audited so far.

Why:

1. multiple optimizer dataframes are compressed immediately after creation,
2. those dataframes are later reused across module boundaries,
3. low-cardinality integer ranking columns are present,
4. downstream code already depends on `.max()` and `.min()` over those columns.

Confirmed high-importance numeric columns in v1 candidate outputs:

1. `Test location (X)`
2. `Test location (Y)`
3. `Test location (Z)`
4. `Test location to DIL centroid (X)`
5. `Test location to DIL centroid (Y)`
6. `Test location to DIL centroid (Z)`
7. `Dist to DIL centroid`
8. `Test location (Prostate centroid origin) (X)`
9. `Test location (Prostate centroid origin) (Y)`
10. `Test location (Prostate centroid origin) (Z)`
11. `Dist to Prostate centroid`
12. `Number of normal dist points contained`
13. `Number of normal dist points tested`
14. `Proportion of normal dist points contained`
15. `X_plane_index`
16. `Y_plane_index`
17. `Z_plane_index`

Recommended v1 producer policy:

1. protect all coordinate, distance, count, proportion, and plane-index columns from categorical conversion,
2. keep label and ID columns eligible for categorical compression,
3. review whether key float measurement columns should also skip float downcast when used downstream for geometry fidelity or cross-run comparison.

Affected v1 dataframe families:

1. `DIL centroid optimal biopsy location dataframe`
2. `Optimal biopsy location dataframe`
3. `Optimal biopsy location (all tested lattice points) dataframe`
4. `Optimal biopsy location (zero lattice) dataframe`
5. `guidance map max-planes dataframe`
6. `All points outside of DILs (zero points) dataframe`
7. `All points within DILs (tested points) dataframe`
8. `Optimal biopsy location (entire cubic lattice) dataframe`
9. `Biopsy optimization - Cumulative projection (all points within prostate) dataframe`

### 3. Guidance-Map Candidate Dataframes

File:

1. `python_files_dcm_meta_based/advanced_guidance_map_creator.py`

Findings:

1. this module already defines explicit numeric schema constants,
2. downstream consumers in the same module already use `pandas.to_numeric(...)` on rank and index columns,
3. the schema metadata exists, but compression policy enforcement is not yet aligned with those schema lists.

Useful existing schema anchors:

1. `GUIDANCE_MAP_LEGACY_FIRING_DF_NUMERIC_COLUMNS`
2. `GUIDANCE_MAP_CANDIDATE_GEOMETRY_CONTEXT_NUMERIC_COLUMNS`
3. `GUIDANCE_MAP_CANDIDATE_FIRING_DF_NUMERIC_COLUMNS`

Confirmed high-importance guidance-map numeric columns:

1. `Relative struct index`
2. `Candidate hole rank`
3. `Candidate hole distance to optimal sampling point (3D) (mm)`
4. `Firing depth row index (per structure)`
5. `Penetration depth (mm)`
6. all coordinate columns in prostate-centroid and transducer-primed frames,
7. all geometric distance and projection-parameter columns,
8. Euler angle columns.

Guidance-map columns that are good categorical or string candidates:

1. `Patient ID`
2. `Relative structure ID`
3. `Relative struct type`
4. `Candidate hole label`
5. `Candidate hole UID`
6. `Optimal template hole`
7. direction and convention label columns.

Recommended guidance-map policy:

1. reuse the existing numeric schema constants as the seed for producer-side `never categorical` lists,
2. keep the existing consumer-side `pandas.to_numeric(...)` pattern because it is already a valid non-mutating recovery pattern,
3. align producer enforcement with the existing schema declarations.

### 4. Optimizer V2 Surfaces

Files:

1. `python_files_dcm_meta_based/biopsy_optimizer/v2/biopsy_optimizer_module_v2.py`
2. `python_files_dcm_meta_based/custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.py`

Findings:

1. v2 is more array-first than v1,
2. its main metadata structure is not a heavily reused ranking dataframe,
3. the main compressed dataframe surface observed in this audit is the containment-results dataframe,
4. the current v2 producer already uses a name-based categorical exclusion for `Pt contained bool`.

Confirmed numeric columns worth protecting in v2 containment outputs when needed downstream:

1. `Relative structure index`
2. `Test pt X`
3. `Test pt Y`
4. `Test pt Z`
5. `Test pt index`
6. `Nearest zslice index`
7. `Nearest zslice zval`
8. associated value columns passed as `associated_value_str`

Current v2 assessment:

1. lower immediate risk than v1,
2. still a candidate for the same policy framework,
3. should adopt the shared policy module rather than remain one-off.

### 5. Existing Recovery Pattern Already Present in Repo

The repo already uses one-column recovery in several places via `pandas.to_numeric(...)`.

This is already effectively the preferred consumer-side pattern.

Implication:

1. new resolver helpers will formalize existing practice,
2. they do not introduce a foreign style,
3. they reduce repetition and make intent explicit.

## Recommended Architecture

### Producer Side

At dataframe creation time:

1. use named schema-policy constants,
2. pass those constants explicitly into the compression helper,
3. do not duplicate inline lists unless the dataframe is genuinely one-off.

### Consumer Side

At computation sites:

1. use standardized non-mutating resolver helpers,
2. recover only the needed series or the needed local group of columns,
3. avoid mutating the stored dataframe back to a wider schema unless there is a compelling reason.

## Proposed Future Helper API

These names are placeholders for the implementation pass.

1. `resolve_numeric_series(...)`
2. `resolve_integer_series(...)`
3. `resolve_numeric_columns(...)`

Desired behavior:

1. accept a dataframe plus column name or column list,
2. return temporary numeric series or a temporary numeric dataframe slice,
3. be explicit about coercion behavior,
4. default to non-mutating behavior,
5. support integer-target and float-target variants.

## Open Design Questions

1. Do we need a second producer-side control for `never downcast`, separate from `never categorical`?
2. Which float columns in geometry-heavy outputs must remain `float64` rather than accepting `float32` downcast?
3. Should some index-like columns remain numeric everywhere for consistency, even if currently used only as identifiers?
4. Should the shared policy module live in `python_files_dcm_meta_based/` as a general utility module, or next to `dataframe_builders.py`?

## Phased Rollout

### Phase 1: Policy Freeze

1. agree on repo-wide dtype principles,
2. freeze first-pass `never categorical` lists for v1, v2, and guidance-map dataframe families,
3. decide whether `never downcast` support is needed immediately.

### Phase 2: Helper Implementation

1. add non-mutating resolver helpers,
2. document their intended usage,
3. copy the existing compression helper into the new dataframe-utilities module and leave a compatibility shim at the old import path,
4. use them in existing consumer sites that already need numeric recovery.

### Phase 3: Producer Enforcement

1. create shared schema-policy constants,
2. wire them into producer call sites,
3. keep call sites explicit by passing named lists rather than hiding policy inside the helper.
4. defer any default heuristic changes in the compression helper until the explicit policy inputs have been rolled out and validated.

### Phase 4: Consumer Cleanup

1. replace ad hoc numeric recovery with standard helpers where useful,
2. keep one-off `pandas.to_numeric(...)` only where that is clearer than helper indirection,
3. verify legacy loaded dataframes still work.

### Phase 5: Validation

1. rerun simulated biopsy transport using optimizer-v1 outputs,
2. compare selection metadata and chosen targets against expected legacy behavior,
3. verify guidance-map rank and firing depth consumers still behave correctly,
4. spot-check precision-sensitive geometry outputs if any float downcast policy changes are made.

## Immediate To-Do List

1. define the first shared dtype-policy constants module,
2. define the resolver helper API,
3. classify optimizer-v1 candidate dataframe columns into `never categorical`, `allowed categorical`, and `precision-sensitive numeric`,
4. classify guidance-map candidate geometry and firing dataframe columns using the existing schema constants,
5. classify v2 containment-result columns and metadata columns,
6. implement producer-side exceptions for the highest-risk v1 dataframes,
7. implement consumer-side resolvers for rank/count/index recovery,
8. update transporter and any similar selection modules to use the standard helper rather than ad hoc local coercion,
9. create the dedicated dataframe-utilities subfolder and move toward it via copy-plus-shim rather than a hard move,
10. decide whether float precision controls are needed in the compression helper,
11. document final policy next to the helper implementation once code lands.

## Working Decision

Pending code implementation, the working direction is:

1. keep the repo's compression strategy,
2. define dtype policy at top level by dataframe family,
3. pass policy explicitly at producer call sites,
4. add non-mutating resolver helpers for consumer-side recovery,
5. treat optimizer-v1 as the highest-priority fix surface.
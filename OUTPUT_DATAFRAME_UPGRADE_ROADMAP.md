# Output Dataframe Upgrade Roadmap

## Goal

Move the output dataframes toward a clean manifest-style contract where biopsy-level and voxel-level tables can be joined downstream by stable UIDs instead of ad hoc multi-column merges.

The upgrade should preserve backward compatibility with current runs while giving future downstream pipelines one canonical join surface.

## Core Principles

1. Keep lineage and achieved state separate.
2. Use stable UIDs as primary join keys.
3. Keep ROI and ref number columns as audit fields, not primary join keys.
4. Add fraction metadata from concrete DICOM fields when available.
5. Resolve import metadata from a config-described DICOM field contract rather than one hard-coded export assumption.
6. If a concrete fraction field is not available, write `NA` rather than inferring a new fraction column from the generated patient UID string.

## Input DICOM Contract Dependency

The dataframe upgrade is downstream of the input DICOM contract.

Current assessment notes live in `INPUT_DICOM_DATA_ASSESSMENT.md`.

The key direction is:

1. keep one default export profile for the data already used in this repo,
2. allow config-defined field selection for patient identity, fraction identity, MR or US routing, and series classification,
3. keep output dataframe columns sourced from the resolved config contract rather than from ad hoc string parsing scattered through the pipeline.

## Fraction Metadata

Current code stores `Fraction number` by parsing the DICOM patient ID string.

That should be treated as legacy behavior.

For the upgraded dataframe pathway, prefer a concrete DICOM fraction source resolved through the input-data config contract.

Confirmed candidate source from the current RTPLAN data:

- `FractionGroupSequence`
- `FractionGroupNumber`

Observed sample behavior:

- RTPLAN contains `FractionGroupSequence[0].FractionGroupNumber = 1`
- the same sample also separates patient identity across DICOM patient name and patient ID fields

Default upgrade rule:

1. use RTPLAN `FractionGroupSequence[*].FractionGroupNumber` when available and unambiguous,
2. otherwise use the next configured fraction field candidate for the active export profile,
3. otherwise use `NA` for the new canonical dataframe fraction column,
4. keep the current legacy parsed `Fraction number` field only as an audit/deprecation field until old joins are retired.

This keeps the default path aligned with the current RTPLAN data while allowing other sites to declare a different authoritative fraction field without rewriting dataframe builders.

## Canonical Join UIDs

The long-term primary keys should be index-driven within a case, not ROI-string driven.

Within a patient fraction, the combination of structure type and index number is the true object identity.

Recommended canonical columns:

- `Case UID`
- `Patient Name`
- `Patient ID (from dicom)`
- `Fraction number (dicom)`
- `Fraction number (legacy parsed)`

Recommended biopsy-level UIDs:

- `Biopsy UID` = `Case UID|Bx ref|Bx index`
- `Matched real biopsy UID` = `Case UID|Bx ref|Matched real biopsy index`
- `Simulated family UID` = `Case UID|Simulated type|Target structure type|Target structure index|Multiplicity base`

Recommended structure-level UIDs:

- `Structure UID` = `Case UID|Struct type|Structure index`

Recommended voxel/point UIDs:

- `Biopsy voxel UID` = `Biopsy UID|Voxel index`
- `Biopsy point UID` = `Biopsy UID|Original pt index`

## Canonical Biopsy Targeting Manifest

Add one new clean cohort dataframe as the authoritative biopsy targeting manifest.

Suggested name:

- `Cohort: Biopsy targeting manifest`

This dataframe should contain one row per biopsy object and expose both intended and realized targeting contracts.

Minimum recommended columns:

- case and patient identity columns
- fraction columns
- `Biopsy UID`
- `Simulated bool`
- `Simulated type`
- `Simulated family UID`
- multiplicity columns
- matched real biopsy columns
- intended target UID and descriptive fields
- realized centroid target UID and descriptive fields
- realized surface target UID and descriptive fields
- intended-vs-realized agreement flags
- transport family and selected transport metadata

This manifest becomes the canonical biopsy-level join surface for downstream study-specific pipelines.

## Canonical Voxel-Level Join Surface

The same UID contract should be extended to voxel-level and point-level tables.

Every biopsy-point or biopsy-voxel dataframe should carry:

- `Case UID`
- `Biopsy UID`
- fraction columns
- `Biopsy voxel UID` or `Biopsy point UID`

This allows voxel-level dosimetry, tissue classification, targeting, and radiomics-derived voxel summaries to be joined downstream without repeated multi-column merge logic.

## Backward Compatibility Path

Phase 1:

- keep existing legacy dataframe columns unchanged,
- add canonical UID columns alongside them,
- add `Fraction number (dicom)` alongside the legacy parsed fraction field.

Phase 2:

- add the clean biopsy targeting manifest,
- make downstream study pipelines prefer `Biopsy UID` and `Simulated family UID`.

Phase 3:

- propagate canonical UIDs to biopsy-level dose, targeting, MR, and DVH tables,
- propagate canonical UIDs to voxel-level and point-level tables.

Phase 4:

- retire fragile multi-column joins in new downstream code,
- keep legacy columns only for audit and compatibility.

## Relationship To The Simulated Biopsy Refactor

This roadmap is intentionally separate from the geometry refactor, but it depends on the same contract split:

- intended target and family lineage come from preparation,
- realized target comes from late realized-targeting,
- transport metadata comes from the transport contract,
- downstream tables should be able to see all three without redefining family identity from achieved location.

That means the dataframe cleanup should not redefine simulated family membership from the late realized targeter.

Family identity remains lineage-driven.

Realized targeting remains an achieved-placement measurement.
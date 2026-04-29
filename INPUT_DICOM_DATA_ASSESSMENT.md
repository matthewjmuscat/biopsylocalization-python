# Input DICOM Data Assessment

## Purpose

This document records the current DICOM input assumptions in the repo and the current observed export pattern in the most recent input data.

It is not the final "how to export your data" guide.

It is the staging note that should keep later data-standardization work consistent while the current focus stays on the optimizer and transport upgrades.

## Current Repo Loader Contract

The current loader logic classifies files mainly from DICOM `Modality` and a few additional tags.

Current hard-coded routing:

- `RTSTRUCT` by `Modality == RTSTRUCT`
- `RTDOSE` by `Modality == RTDOSE`
- `RTPLAN` by `Modality == RTPLAN`
- `US` by `Modality == US`
- `MR T2` by `Modality == MR` and `SeriesDescription == T2`
- `MR ADC` by `Modality == MR` and `SeriesDescription == ADC`
- `US` fallback by `Modality == MR` and empty `MRAcquisitionType`

The current case UID is generated as:

- `PatientName (PatientID)`

Current legacy fraction handling is separate and still parses the patient ID string.

## Current Input Data Assessment

Lightweight scan of the current input data under `Data/Input data` found:

- `3` RTPLAN files
- `3` RTDOSE files
- `3` RTSTRUCT files
- `277` MR files

Observed current RT files:

- RTSTRUCT sample uses `StructureSetLabel = ANON`
- RTPLAN sample uses `RTPlanLabel = ANON`
- RTDOSE sample uses `DoseSummationType = PLAN`, `DoseUnits = GY`, `DoseType = PHYSICAL`
- RTPLAN fraction metadata is present in all sampled current plan files via `FractionGroupSequence[0].FractionGroupNumber`

Observed current MR export behavior:

- all sampled MR files had `Modality = MR`
- all sampled MR files had empty `SeriesDescription`
- all sampled MR files had empty `MRAcquisitionType`
- sampled MR files had `ImageType = [DERIVED, SECONDARY, OTHER]`
- sampled MR files were exported by `Varian Medical Systems` / `Vitesse`
- sampled MR files did not expose a visible private creator in the inspected example

## Immediate Consequence

The current hard-coded MR routing does not generalize to the most recent data export shape.

In the current dataset, the built-in `SeriesDescription == T2` or `SeriesDescription == ADC` assumptions are not satisfied.

The empty-`MRAcquisitionType` fallback also cannot reliably distinguish true MRI from ultrasound-like data exported as MR if the exporting workflow changes again.

This is the same class of issue discussed earlier when some ultrasound-origin images were effectively being routed through MR-labelled exports.

## What Should Be Standardized

Two different things should be separated:

1. the ideal export contract,
2. the repo's ingest contract.

Ideal export contract:

- export each logical data type with stable, explicit DICOM tags when possible,
- avoid depending on folder naming or manual case-specific interpretation,
- keep patient identity and fraction identity explicit in DICOM metadata.

Repo ingest contract:

- do not assume one export style forever,
- define in config how a site exported its data,
- resolve logical file roles from configured tag rules with documented fallbacks.

## Recommended Direction

The safest generalizable strategy is not to hard-code one export layout into the loader.

Instead, define a config-described export profile that says where each required value should be read from.

That config should describe at least:

- how to identify a case
- how to identify a fraction
- how to identify RTSTRUCT, RTPLAN, RTDOSE
- how to identify ultrasound-like images
- how to identify MR series types such as T2 and ADC
- what fallbacks are allowed if the primary field is missing

## Suggested Config Surface

Example shape only:

```yaml
dicom_ingest:
  case_uid:
    strategy: composite
    parts:
      - tag: PatientName
      - literal: " ("
      - tag: PatientID
      - literal: ")"

  file_roles:
    rtstruct:
      match_all:
        - tag: Modality
          equals: RTSTRUCT

    rtplan:
      match_all:
        - tag: Modality
          equals: RTPLAN

    rtdose:
      match_all:
        - tag: Modality
          equals: RTDOSE

    mr_t2:
      match_all:
        - tag: Modality
          equals: MR
        - any_of:
            - tag: SeriesDescription
              equals_any: [T2, T2W]
            - tag: CustomFieldForTrueMRType
              equals: T2

    mr_adc:
      match_all:
        - tag: Modality
          equals: MR
        - any_of:
            - tag: SeriesDescription
              equals_any: [ADC, ADC MAP]
            - tag: CustomFieldForTrueMRType
              equals: ADC

    us_like_image_series:
      match_any:
        - match_all:
            - tag: Modality
              equals: US
        - match_all:
            - tag: Modality
              equals: MR
            - tag: MRAcquisitionType
              equals: ""
            - tag: CustomFieldForExportType
              equals: US

  fraction:
    candidates:
      - sequence: FractionGroupSequence[0].FractionGroupNumber
      - tag: SomeOtherSiteSpecificFractionField
      - parser: none
```

The important point is not the exact YAML.

The important point is that the repo should resolve metadata through a declared export profile rather than silently assuming one hospital or one export tool will always populate the same tags.

## Proposed Fallback Philosophy

1. Use explicit standard DICOM tags first.
2. Use configured custom or site-specific tags second.
3. Use documented string parsing only as a legacy audit fallback.
4. If no reliable source exists, emit `NA` instead of manufacturing certainty.

## What A Future Export README Should Cover

The later user-facing export guide should state:

- required logical file groups
- required patient and fraction identifiers
- preferred DICOM tags for each logical value
- accepted site-specific override fields
- exact fallback order used by the repo
- examples of valid exports from Vitesse or other upstream tools
- validation checks users should run before feeding data into the repo

## Near-Term Implication For Current Refactor Work

This does not require a loader rewrite right now.

The near-term action is only to keep new roadmap and dataframe work compatible with a future config-driven ingest contract.

That means:

- new canonical fraction columns should assume configurable source fields,
- new biopsy and voxel manifests should record resolved import metadata cleanly,
- optimizer and transport upgrades should avoid hard-coding new patient-ID parsing assumptions.
# Input Data Manifest Design

Last updated: 2026-04-30

## Purpose

Define a cheap, reproducibility-focused manifest that records what input data was fed into a run without copying the input DICOM files into the output directory.

This should complement the existing output-side config and run manifests.

The goal is not full archival preservation of the source data.

The goal is to make it easy to answer:

- which patient folders were used,
- which DICOM objects were discovered,
- which logical roles were assigned to those DICOM objects,
- and whether two runs appear to have used the same source dataset at a practical, inspectable level.

## Non-Goals

- copying the input DICOM files into the output tree,
- forcing a cryptographic proof of identity for every run by default,
- turning missing optional DICOM tags into a hard run failure.

## Why This Matters

The current output tree already captures downstream products well, but it is still too easy to lose track of what exact source data produced a given run.

For this pipeline, the most important missing audit surface is the input side:

- patient folder names,
- case IDs derived by the loader,
- DICOM SOP/series/study identifiers,
- and the role-assignment decision that told the algorithm which files counted as RTSTRUCT, RTPLAN, RTDOSE, MR T2, MR ADC, or fallback image inputs.

Folder names alone are not reliable enough to prove identity, but they are still useful as a fast surface-level check and should be recorded as such.

## Recommended Output Location

Write the manifest into the run output directory, alongside the existing run manifests.

Recommended layout:

```text
<specific_output_dir>/
  manifests/
    run_config_manifest.json
    input_manifest_summary.json
    input_case_manifest.csv
    input_dicom_manifest.csv
    input_manifest_warnings.jsonl
```

If the existing manifest folder naming differs, keep the current top-level convention and only add the new files.

## Recommended Files

### 1. `input_manifest_summary.json`

Small run-level summary file.

Suggested fields:

- `manifest_version`
- `generated_utc`
- `run_output_dir`
- `input_root_dir`
- `loader_profile_name` or equivalent ingest-config identifier
- `num_patient_folders_seen`
- `num_case_uids_seen`
- `num_dicom_files_seen`
- `num_role_assignments`
- `dicom_manifest_hash_mode`
- `notes`

This file should be cheap to open and enough to understand what the larger CSV files contain.

### 2. `input_case_manifest.csv`

One row per discovered case or patient-level grouping.

Suggested columns:

- `Input root dir`
- `Patient folder relative path`
- `Patient folder name`
- `Derived case UID`
- `Patient ID`
- `Patient Name`
- `Fraction label or number if known`
- `RTSTRUCT count`
- `RTPLAN count`
- `RTDOSE count`
- `MR count`
- `US count`
- `Other DICOM count`
- `Folder name reliability note`

This is the fastest human-readable answer to "which cases did this run actually use?"

### 3. `input_dicom_manifest.csv`

One row per discovered DICOM file that the loader considered during ingest.

Suggested columns:

- `Relative path from input root`
- `Immediate parent folder`
- `Patient folder relative path`
- `Derived case UID`
- `Patient ID`
- `Patient Name`
- `Study Instance UID`
- `Series Instance UID`
- `SOP Instance UID`
- `Frame of Reference UID`
- `Modality`
- `Series Description`
- `Study Description`
- `Manufacturer`
- `Manufacturer Model Name`
- `Image Type`
- `MRAcquisitionType`
- `Dose Summation Type`
- `Dose Units`
- `RT Plan Label`
- `Structure Set Label`
- `Assigned logical role`
- `Assigned by rule`
- `Assigned role confidence`
- `File size bytes`
- `Modified time`
- `Optional content digest`

This is the authoritative cheap audit surface for what data was fed into the algorithm.

### 4. `input_manifest_warnings.jsonl`

Append-only warning stream for ingest ambiguities.

Each record should be one JSON object.

Typical warning categories:

- missing patient identifiers,
- duplicate SOP Instance UID,
- multiple candidate RTPLAN files for one case,
- ambiguous MR role assignment,
- files skipped because required tags were missing,
- folder naming that conflicts with DICOM-derived identity.

## Recommended Identity Strength Levels

The manifest should support a tiered identity policy so the cheap default stays cheap.

### Tier 1: Cheap Default

Record:

- relative path,
- file size,
- modified time,
- DICOM identifiers,
- loader role assignment.

This is likely sufficient for routine run audit and day-to-day comparisons.

### Tier 2: Stronger Audit

Optionally add a file digest column.

Recommended choices:

- `sha256` if maximum certainty is needed,
- a faster digest if runtime cost becomes noticeable.

This should be configurable, not always-on.

## Reliability Notes To Record Explicitly

The manifest should state these limits plainly:

- folder names are a surface-level check only,
- missing or inconsistent DICOM metadata can weaken identity confidence,
- role assignment depends on the current ingest contract and config profile,
- the manifest proves what the loader saw and assigned, not whether the source export itself was semantically correct.

## Generation Point In The Pipeline

Generate the manifest after file discovery and role assignment are complete, but before heavy geometry reconstruction or simulation begins.

That placement gives two benefits:

- the manifest still gets written even if the run later dies in optimizer, MC, or plotting stages,
- the manifest reflects the actual ingest decisions used by the rest of the run.

Recommended timing:

1. discover candidate files
2. classify files into logical roles
3. emit input manifest files
4. continue into preprocessing and downstream pipeline stages

## Minimal Required Fields For A First Pass

If only a small first implementation is desired, do not block on the full schema above.

The minimum useful first pass is:

- `input_manifest_summary.json`
- `input_case_manifest.csv`
- `input_dicom_manifest.csv` with only:
  - relative path
  - patient folder name
  - derived case UID
  - patient ID
  - modality
  - series description
  - study UID
  - series UID
  - SOP UID
  - assigned logical role
  - file size

That is already enough to make later run-to-run provenance checks much easier.

## Suggested Validation Checks

Once implemented, the manifest should support simple run-comparison checks such as:

- same number of cases discovered,
- same set of derived case UIDs,
- same set of SOP Instance UIDs,
- same logical-role counts per case,
- same patient-folder surface names when expected.

## Relationship To Existing Docs

- `INPUT_DICOM_DATA_ASSESSMENT.md` should stay focused on ingest assumptions and observed export profiles.
- This document should stay focused on what input-side audit artifact should be emitted per run.

## Recommended First Implementation Order

1. emit `input_manifest_summary.json`
2. emit `input_case_manifest.csv`
3. emit a minimal `input_dicom_manifest.csv`
4. add optional digest mode later
5. wire simple run-comparison tooling after the manifest exists
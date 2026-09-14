# DICOM Input Shape

This package records the DICOM input shape expected by the current biopsy localization pipeline.

The current implementation is intentionally conservative: it documents and emits the legacy routing profile used by the existing main pipeline, but it does not change role assignment.

## Tier 1: Current Fixed Profile

The current profile is `legacy_variseed_mim_v1`.

Required core roles per patient/fraction are:

- `RTSTRUCT`
- `RTDOSE`
- `RTPLAN`

The current role rules are:

| Role | DICOM field rule |
| --- | --- |
| `RTSTRUCT` | `Modality == RTSTRUCT` |
| `RTDOSE` | `Modality == RTDOSE` |
| `RTPLAN` | `Modality == RTPLAN` |
| `US` | `Modality == US` |
| `MR_T2` | `Modality == MR` and `SeriesDescription == T2` |
| `MR_ADC` | `Modality == MR` and `SeriesDescription == ADC` |
| `US` fallback | `Modality == MR` and `MRAcquisitionType == ""` |

The fallback exists because this dataset can contain ultrasound files exported by Variseed/MIM with `Modality == MR`. In the current data shape, those files are treated as ultrasound when they are not identified as T2 or ADC and their `MRAcquisitionType` is empty.

The current generated patient UID is:

```text
PatientName + " (" + PatientID + ")"
```

The current legacy fraction parser reads from `PatientID` using configured fraction prefixes and a number regex.

## Location, logical object, and content identity

Experimental sibling directories legitimately contain duplicate physical copies.
These three identities answer different questions:

| Identity | Fields | Meaning |
| --- | --- | --- |
| File location | Source path and resolved filesystem path | Which physical copy did execution consume? |
| Logical DICOM object | SOPInstanceUID and SOPClassUID; StudyInstanceUID / SeriesInstanceUID for grouping | Which DICOM object is represented? |
| Exact content | SHA-256 and byte length | Are the complete file bytes identical? |

`PatientInputPaths.manifest_identity_sha256` binds role/path assignments.
`patient_input_content_v1` additionally binds those locations to bytes and verifies
the exact planned files before and after a worker. Neither is a logical object
deduplication contract. Strict local execution may require path **and** bytes to
match; a future scientific-input equivalence policy may accept another path when
logical DICOM identity and bytes match. Do not weaken the existing mutation checks
to obtain that separate equivalence.

Near-term independent discovery slice (specified, **not implemented** in main):

- Same SOPInstanceUID and SHA-256: one logical routing candidate, retaining every
  source-path alias and its content/location provenance. SOPClassUID must agree.
- Same SOPInstanceUID with different SHA-256: explicit conflict, fail unless a
  named resolution policy has been supplied. Never silently choose one copy.
- Different SOPInstanceUIDs: distinct objects, irrespective of similar filenames.
- Never deduplicate by filename. After object deduplication, role ambiguity is
  still a separate routing decision; distinct plans are not interchangeable.

Current main recursively scans `**/*.dcm`; core role dictionaries still overwrite
earlier paths for the same generated case UID, while image roles append paths.
Thus current main does **not** implement the future duplicate/conflict policy.
The biopsy pass consumes explicit retained role assignments and does not rewrite
discovery. This limitation must remain visible until that independent slice lands.

The optional synthetic-dose fixture tool records source `locations`, `dicom`, and
`content` separately, and handles identical/conflicting copies within its explicit
source job. This is a small validation-tool foundation, not adoption by discovery.

## Reusable synthetic subjects

Keep local patient-derived fixtures under `Input data/synthetic set/`, in a fresh
versioned subdirectory. The generator uses names `SYNTHETIC_DOSE_10 (F2)` and
`SYNTHETIC_DOSE_13_5 (F2)`, new study/series/frame/SOP identities, and remapped
references among copied objects. Pixel bytes and geometry stay unchanged.
Use a distinct `--case-prefix` for additional fixture versions (for example
`SYNTHETIC_DOSE_V2`); a different folder alone does not distinguish generated cases.
Its manifest can be used directly; normal recursive discovery also sees its
`.dcm` files. The subdirectory is **not** an exclusion mechanism: explicitly select
clinical or synthetic case UIDs in work orders. Never include both by accident.

These are private patient-derived fixtures, not de-identified distributable data.
Known external references remain external; this is not a complete DICOM export
validator. Source content and copy transformations are recorded locally.

Useful fixture progression:

1. Now: fixed physical dose with unequal 10.0/13.5 Gy prescriptions, plus explicit
   threshold and missing-prescription controls in synthetic unit tests.
2. Soon, with discovery: identical bytes at two paths; same SOP UID with changed
   bytes; different SOP UIDs with identical filenames. Keep deliberate conflicts
   outside routine discovery trees until negative-test selection exists.
3. Later, when the consuming boundary is tested: changed dose scaling or pixel
   spacing/orientation, missing modality, multiple ADC series/units, translated
   geometry with consistent frame references. Each variant needs a declared
   expected result; avoid accumulating unexplained copies.

See [biopsy validation and fixture commands](../../docs/runtime/BIOPSY_PREPROCESSING_RUNBOOK.md).

## Tier 2: Future Configurable Profile

Future GUI and CLI workflows should allow the user to provide or select a routing profile that defines:

- which DICOM fields identify each role,
- the expected values for MR ADC, MR T2, US, dose, plan, and structure files,
- rule priority and fallback behavior,
- whether ambiguous or missing roles are warnings or errors,
- how patient/case identity is built,
- how fraction-level DICOM groups are identified,
- which roles are required versus optional.

The current `input_routing_profile.json` output is the additive bridge between these tiers. It gives validation runs a durable record of the rules assumed by the pipeline before those rules become user-configurable.

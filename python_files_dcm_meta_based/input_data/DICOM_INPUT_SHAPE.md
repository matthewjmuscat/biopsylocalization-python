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

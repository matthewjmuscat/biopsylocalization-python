# Patient Runner Cohort-Derived Quantities

This note records runtime pathways removed during the patient-runner migration.
They were useful legacy conveniences, but they depended on all-patient state
being visible during the main algorithm. That is incompatible with a robust
single-patient runner.

## Removed Runtime Pathways

### Biopsy Variation: Global Mean

Legacy behavior:

- computed biopsy centroid-variation summaries across all patients retained in
  `master_structure_reference_dict`,
- wrote the selected all-patient mean into
  `master_structure_info_dict["Global"]["Mean biopsy centroid variation"]`,
- allowed uncertainty generation mode `"Global mean"` to add that cohort mean
  into biopsy translation sigmas.

Current behavior:

- `"Global mean"` is no longer a supported uncertainty mode,
- uncertainty generation now supports patient-local/per-biopsy modes only:
  `"Per biopsy max"`, `"Per biopsy mean"`, and `"Default only"`,
- the main pipeline no longer computes or stores all-patient biopsy centroid
  variation summaries.

Future reimplementation:

- if this becomes scientifically useful again, implement it as an explicit
  per-patient sidecar input produced before the main run,
- the sidecar generator should reuse the exact same biopsy contour and centroid
  modules used by the main algorithm,
- sidecar files should be version stamped with code version, config identity,
  DICOM identity, and measurement method.

### Simulated Biopsy Length: Cohort Mean/Normal

Legacy behavior:

- `"real mean"` assigned every simulated biopsy the all-patient mean real biopsy
  contour length,
- `"real normal"` sampled simulated biopsy lengths from the all-patient real
  biopsy length mean and standard deviation,
- `"match real"` fell back to the all-patient mean if no matched or same-DIL
  real biopsy length was available.

Current behavior:

- `"real mean"` and `"real normal"` are no longer supported,
- `"match real"` remains supported but uses only patient-compatible sources:
  matched real biopsy length, then same-patient/same-DIL mean if available,
  then the configured full needle compartment length,
- unsupported length modes now fail loudly instead of silently defaulting.

Future reimplementation:

- cohort-derived or learned length priors should be generated outside the main
  run as optional per-patient sidecar inputs,
- the main runner should read those sidecars as data, not calculate them by
  retaining all patients in memory.

## Input Location Note

Raw DICOM and local input data should remain ignored by git. The migration goal
is to define a clearer input root and input contract inside the project layout,
not to commit patient data. DICOM discovery should remain metadata-driven: tags
and spatial identity are authoritative, while folder structure is only a local
organization aid.
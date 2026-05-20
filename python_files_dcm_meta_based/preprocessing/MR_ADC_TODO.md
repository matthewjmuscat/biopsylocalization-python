# MR ADC Pathway TODO

Notes for a future deliberate MR ADC pathway review. Do not change MR ADC science casually during wrapper/modularization passes.

## Known Concerns

- Bad ADC values need an explicit policy. Current code filters negative values during lattice reconstruction, but exported summaries still include zero ADC values. Confirm whether zero represents valid signal, padding, masked background, or a replacement for bad values.
- May 20, 2026 input audit of `/home/matthew-muscat/Documents/UBC/Research/Data/Input data (for MR ADC run)`:
  - 2,895 DICOM files were metadata-readable.
  - 1,375 files are exact `Modality == MR` and `SeriesDescription == ADC`, grouped as 17 ADC series, one per subject/fraction folder in the current input directory. The DICOM `PatientID` field is not unique enough here (`F1`/`F2`), so use the input folder/runtime UID rather than raw DICOM `PatientID` for patient identity.
  - Exact ADC subject/fraction folders present: `181 (F1)`, `181_F2`, `194 (F1)`, `194 (F2)`, `195 (F1)`, `195 (F2)`, `196 (F1)`, `196 (F2)`, `198 (F2)`, `199 (F1)`, `199 (F2)`, `200 (F1)`, `200 (F2)`, `201 (F1)`, `201 (F2)`, `203 (F1)`, `203 (F2)`. No `198 (F1)` exact ADC series was present in this input directory during the audit.
  - 1,375 files are `Modality == MR` with blank `SeriesDescription`; they are 8-bit `DERIVED\SECONDARY\OTHER` images with no real-world value mapping and raw values 0-255. Treat these as non-quantitative display/secondary images unless proven otherwise.
  - 94 files are exact `SeriesDescription == T2`, currently only one T2 series in this input set.
  - Exact ADC series have raw value ranges from -256 to 4,497. Aggregate exact ADC voxels: 24.49% negative and 5.22% zero before filtering.
  - 9 exact ADC series have `RealWorldValueMappingSequence` with slope near `1e-6` and units labelled `mm2/s`; these are the series with many negative stored voxels. 8 exact ADC series have no RWV mapping; these have no negative voxels but higher zero fractions, up to about 23% in the full image lattice.
  - Current pipeline behavior removes negative ADC voxels via `filter_out_negatives=True`; it does not set negative values to zero. Zero-valued ADC voxels remain available for containment summaries.
  - The current loader stores RWV slope/intercept, but `reconstruct_mr_lattice_with_coordinates_from_dict_v2` does not apply the slope when units are labelled `mm2/s`, `mm²/s`, or `mm²/s (assumed)`. This means exported/current values are stored ADC-scale values such as 900 rather than physical `0.0009 mm2/s`; the default `1e-6` slope for no-RWVM series is also skipped by the same unit check.
- If zero represents bad/no-data ADC, evaluate safer treatments before changing outputs:
  - remove bad/no-data points before containment summaries,
  - interpolate bad points from neighboring valid ADC values,
  - reconstruct onto a standard equally spaced lattice with inverse-distance weighting,
  - record counts/fractions of removed or imputed points in output metadata.
- Multiple ADC series per patient currently resolve by keeping the first series. Keep the warning behavior visible, and later replace first-series selection with an auditable selection rule based on metadata/geometry/series identity.
- MR ADC DICOMs are not fully standardized across inputs. Review SeriesDescription matching, orientation, spacing, coordinate reconstruction, and value scaling before treating this as a finalized pathway.
- Prostate-only ADC summary currently starts with prostate-contained ADC lattice points, then removes points contained in rectum, urethra, and DILs using `Test pt index`. This depends on each containment dataframe being generated from the same ADC lattice in the same order.

## Recommended Direction

- Keep raw/stored ADC values and physical ADC values as separate fields. Apply DICOM real-world value mapping when present, and preserve the original stored value for traceability.
- Do not mix RWVM-scaled and unscaled ADC values in the same downstream feature without an explicit unit/scale column. First fix should be a controlled extractor that returns both `ADC stored value` and `ADC physical mm2/s` when RWVM is available, plus a scale/source flag.
- Exclude invalid voxels from summary denominators instead of replacing them with zero. Start with non-finite, negative, and explicit padding/background values; decide separately whether zero is invalid for each series class after checking prostate-contained zero counts.
- For biopsy-level MR sampling, interpolate from valid neighboring voxels at biopsy points using trilinear interpolation or inverse-distance weighting with a maximum distance/radius and a valid-neighbor count threshold. Return missing/flagged values when local support is poor instead of inventing values from distant tissue.
- Normalize biopsy-sampled ADC and T2 values within patient/fraction using a prostate-only reference distribution after excluding DIL, urethra, rectum, and invalid voxels. Robust options: median/IQR scaling, median/MAD z-score, or percentile/rank within the prostate-only distribution.
- Treat raw T2 intensity as non-quantitative unless these are true T2 maps. For ordinary T2-weighted images, use within-patient normalization/percentiles rather than cross-patient absolute intensities.
- Retain absolute physical ADC summaries where possible, but consider normalized ADC features primary for cross-patient modeling because ADC varies with scanner, sequence, b-values, reconstruction, and vendor handling.

## Existing Validation Anchors

- Patient `194 (F2)` has MR ADC structure-summary outputs in multiple historical runs.
- The May 19, 2026 run and the May 5, 2026 MR ADC + optimizer-v2 baseline both contain the `Prostate_excluding_UDR` row in `MR - ADC - summary statistics by structure dataframe.csv`.
- Future MR ADC changes should compare patient-level preprocessing CSVs and downstream cohort MR tables against these anchors before and after any scientific change.

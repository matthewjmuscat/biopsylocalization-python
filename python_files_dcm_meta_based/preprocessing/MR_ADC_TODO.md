# MR ADC Pathway TODO

Notes for a future deliberate MR ADC pathway review. Do not change MR ADC science casually during wrapper/modularization passes.

## Known Concerns

- Bad ADC values need an explicit policy. Current code filters negative values during lattice reconstruction, but exported summaries still include zero ADC values. Confirm whether zero represents valid signal, padding, masked background, or a replacement for bad values.
- If zero represents bad/no-data ADC, evaluate safer treatments before changing outputs:
  - remove bad/no-data points before containment summaries,
  - interpolate bad points from neighboring valid ADC values,
  - reconstruct onto a standard equally spaced lattice with inverse-distance weighting,
  - record counts/fractions of removed or imputed points in output metadata.
- Multiple ADC series per patient currently resolve by keeping the first series. Keep the warning behavior visible, and later replace first-series selection with an auditable selection rule based on metadata/geometry/series identity.
- MR ADC DICOMs are not fully standardized across inputs. Review SeriesDescription matching, orientation, spacing, coordinate reconstruction, and value scaling before treating this as a finalized pathway.
- Prostate-only ADC summary currently starts with prostate-contained ADC lattice points, then removes points contained in rectum, urethra, and DILs using `Test pt index`. This depends on each containment dataframe being generated from the same ADC lattice in the same order.

## Existing Validation Anchors

- Patient `194 (F2)` has MR ADC structure-summary outputs in multiple historical runs.
- The May 19, 2026 run and the May 5, 2026 MR ADC + optimizer-v2 baseline both contain the `Prostate_excluding_UDR` row in `MR - ADC - summary statistics by structure dataframe.csv`.
- Future MR ADC changes should compare patient-level preprocessing CSVs and downstream cohort MR tables against these anchors before and after any scientific change.

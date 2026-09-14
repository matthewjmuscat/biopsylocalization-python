# Standalone Biopsy Preprocessing

## Scope and evidence

The user-operated five-case anatomical independence gate passed on 2026-09-13
under clean `b123089`, exact 0/0. Its provenance and limits remain in the
[anatomical runbook](ANATOMICAL_INDEPENDENCE_RUNBOOK.md). This pass enables
`biopsy_preprocessing_shadow` with an identically named checkpoint. It adds
synthetic implementation evidence, **not a real-patient biopsy PASS**.

The same lightweight parent launches one fresh patient process. The worker
rehydrates verified `PipelineConfig`, builds only that patient's runtime, and
runs dependency-selected grid → anatomical → preprocessing → artifact writing.
The shared paired validator compares fresh standalone and singleton legacy-input
construction, followed by the same downstream science. It verifies completed
attempts, source/config/environment, exact input bytes, stage inventory, resolved
grid state and numerical products. Sharing algorithms limits this evidence to
input/execution migration parity.

## Scientific boundary inspected

| Operation | Inputs and config | Patient-local mutations / evidence |
| --- | --- | --- |
| Real biopsy reconstruction | RTSTRUCT raw biopsy slices; preprocessing interpolation/kernel/volume settings; biopsy radius | Raw/interpolated contours, centroids, fitted line, reconstructed cylinder points, Delaunay arrays, volume, centroid variation and line samples |
| Simulated preparation | This patient's DIL centroids and real biopsies; `biopsy.simulated.simulated_biopsy_length_method`; `mc.prep.biopsy_needle_compartment_length` | Target selection, multiplicity/matched biopsy, nominal length/source, reindexed biopsy records, patient biopsy-info counts and preparation dataframe |
| Simulated planning | Prepared length; configured canonical line/origin, centroid count and radius; `mc.prep.bx_sample_pts_lattice_spacing` | Canonical local geometry, rotation/translation, Delaunay arrays, planned sample coordinates/bounds/count and preparation decisions |

Raw contour pulling already includes real biopsies and skips simulated ones.
Preparation needs anatomical DIL and biopsy centroids; the current coarse DAG
also runs grid preprocessing before anatomy. No new dose/MR dependency is added
to the biopsy algorithm itself. `SequentialWorkerPool` supplies map/starmap
inside the patient process; there is no nested pool or full-cohort state.

`full` uses configured needle length; `match real` uses matched real biopsy
length, this patient's DIL mean, or the existing full-length fallback. Removed
`real mean` / `real normal` modes already fail because they derive cohort state.
Patient dictionaries remain transitional compatibility storage. This pass does
not create a second mutable product store or serialize those dictionaries.

Uncertainty spreadsheet attachment has no standalone producer/input contract
and remains absent. Realized targeting is still deferred to simulated-biopsy
finalization. Optimization, transforms, realization, classification, MC and
guidance remain fail-closed as standalone live pathways. Planning samples here
are an existing preprocessing product, not later tissue classification.

## Scientific correction: fitted-line samples

The allocation defect identified during the architectural checkpoint `dce3a32`
is corrected in this separate scientific pass. Real-patient biopsy parity remains
**pending**. The checkpoint still captures the corrected field at exact 0/0.

### Origin and chosen contract

Current-branch history locates the faulty preallocation in `e760db5`
(2022-12-07), well before the 2026 helper extraction. Its parent main used an
append loop with 20 intervals and 21 points, including both fitted endpoints.
The change introduced the 0.1 mm spacing limit and an N-row allocation, but kept
`samples[-1]` as though it still meant the last appended, initialized point.
For N > 1 it instead reads the uninitialized final row. This is supported by
main's history, not merely by the separate prototype or remembered intent.

The explicit contract now is: for the existing PCA endpoints A and B, let
L = ||B − A|| and N = ceil(L / 0.1 mm). Store **N + 1** uniformly spaced points
on the closed segment [A, B], with both endpoints included and spacing L/N
at most 0.1 mm, up to floating-point rounding. `np.linspace` initializes every
element without accumulating predecessor error. Endpoint order follows the
existing PCA output; this pass does not impose a new anatomical orientation.

This chooses a well-defined fitted-segment representation. It does **not**
establish that the PCA endpoints are physical biopsy tips. Current
`pca.linear_fitter` centers the segment at the mean slice centroid and uses the
maximum **Euclidean radius** from that mean as its symmetric half-length, rather
than the minimum/maximum projections onto the principal axis. Historical code
cannot by itself recover the author's scientific rationale for that extent.

### Producer and consumer audit

| Route | Use and effect of this correction |
| --- | --- |
| Real reconstruction | Both main and patient adapters call the shared finalizer/builder. The stored line samples change. |
| Planned simulated model | The planner calls the same builder on canonical rings. Its stored line samples change. |
| Realized simulated reconstruction | Transported contours reach the same finalizer in the legacy pathway. It also receives the fix; this standalone stage is not yet enabled. |
| Cylinder, Delaunay and volume inputs | Cylinder transport takes the fitted start, travel vector, ring count and radius. It never uses the other stored line samples. Its start is now read directly from the fitted line. Reconstructed points, rotated slices and volume inputs are unchanged. |
| Length, axis, translation and rotation | Calculated from PCA endpoints/centroids, independently of stored line samples; unchanged. Matched-real planning lengths consequently remain unchanged. |
| Real and planned volume sampling | `sampling/biopsy_point_sampler.py` uses reconstructed points, Delaunay and rotation. The actual lattice sampler and biopsy-frame transformation were compared numerically; coordinates, bounds and counts are unchanged. |
| Classification/MC inputs | Sampled-volume arrays and their transformed coordinates feed double-sextant classification, MC preparation and containment. No current reader of stored line-sample rows 1 onward was found on these routes. These inputs were checked; full classification, GPU volume, MC and dosimetry were not executed. |
| Checkpoints and saved state | The exact checkpoint deliberately retains both real and planned samples. Any historical state retaining this field can contain the faulty array. Old and corrected checkpoints should not be expected to agree on it. |
| Plotting and historical experiments | `plot_general_per_patient` supports this field through `cbfls`; the tracked call is in the prototype. The prototype also uses samples for KD-tree queries, but has its own append-based producer. The analogous nearest-neighbour block in the December 2022 main is inside a triple-quoted inactive block. These findings do not establish what every historical experiment executed. |
| Dead duplicate implementations | The real processor archive and simulated processor's `if False` block retain old code. They are inactive and were not turned into additional production algorithms. |

Scope of plausible historical impact: corrupted stored diagnostic arrays,
displays that consume them, and any external/scratch analysis of those arrays.
The current inspected production dependencies and the numerical comparisons
provide no evidence that this defect changed cylinder-based sampling or its
classification inputs. They do not certify all historic outputs or publications.

### Numerical evidence and separate geometry questions

`validation/test_biopsy_geometry_characterization.py` now runs actual NumPy,
scikit-learn PCA, cylinder transport, Open3D, SciPy Delaunay, the repaired lattice
sampler, and real/planned sampling wrappers on fabricated contours. Allocation
residues include 29, 87, −29, NaN and 1e200. Tests enforce finite samples,
endpoint/count/spacing/collinearity and exact allocation independence.

The old recurrence is retained only as a test negative control. An additional
local experiment loaded the **complete committed helper from dce3a32** and
compared it with the correction, using actual downstream code. Every other
model numeric field, Delaunay points/connectivity/transforms, convex-hull volume,
sample coordinates/bounds/count and biopsy-frame sample coordinates matched
exactly. Representative values below use radius 0.35 mm and sampling step 0.2 mm:

| Fabricated contours | Fitted length (mm) | Cylinder axial span (mm) | Stored line points, old → corrected | Volume sample count |
| --- | ---: | ---: | ---: | ---: |
| Axial | 1.000000 | 0.900000 | 10 → 11 | 45 |
| Oblique | 1.234000 | 1.139077 | 13 → 14 | 54 |
| Reversed slice order | 1.234000 | 1.139077 | 13 → 14 | 54 |
| Asymmetric/bent | 1.207845 | 1.114934 | 13 → 14 | 54 |

Actual canonical planning at nominal length 1.234 mm also retained all 54
volume samples exactly. Tests additionally round-trip the corrected real and
planned line arrays through the checkpoint and detect changes to either field.
The focused geometry/preprocessing/checkpoint/pair/fixture suite passed 45 tests
in the installed scientific environment on 2026-09-14. Native imports succeed
without a GPU here; this is not GPU numerical validation.

Two related scientific questions remain separate from the allocation fix:

- **Cylinder endpoint coverage:** current transport creates N rings with spacing
  L/N, so its span is L − L/N, although the stored cylinder-length field is L.
  The new line-sample count must not be reused as the cylinder ring count
  incidentally. Adding an endpoint ring changes hull/volume/sampling and warrants
  a deliberate scientific geometry pass with those outputs compared.
- **Fitted extent:** in the bent fixture the centroid projection span is
  1.008890 mm, while the symmetric-radius fitted length is 1.207845 mm. Determine
  whether the desired domain contract is that existing extent, projected
  extrema, or independently identified physical endpoints before changing it.
  Degenerate/very short reconstructions and horizontal-axis handling also need
  explicit geometry-domain decisions; this pass does not claim to repair them.

## Checkpoint contract

`biopsy_preprocessing_checkpoint_v1` reuses the anatomical JSON/NPZ engine and
adds a fixed inventory in `validation/biopsy_checkpoint_fields.py`. Each biopsy
has identity/kind, preparation decisions, real geometry where applicable and
planned geometry/sampling where applicable. Real coordinates remain DICOM
patient millimetres; planned coordinates remain canonical local millimetres.
Dataframe order/schema, array dtype/shape, discrete values and native numeric
arrays are compared without sorting, alignment, re-scaling or recomputation.

Missing biopsy inventory, unfinished planning, missing required numerical products
and inconsistent sample counts fail capture/read. Optional absent fields are
listed explicitly. The writer retains Delaunay points/connectivity, not native
objects; it never pickles runtime state. A paired PASS requires exact **0/0** and
the completed-input byte seals. Old anatomical entrypoint names remain compatible;
the second schema is selected explicitly through the same engine/service.

## User-operated real gate

Review and commit/freeze the fitted-line correction before this gate.
Use fresh destinations; do not change inputs or source after preparing provenance.
Environment identity is now v2, so prepare new execution provenance while
retaining the historical scientific snapshot. Do not rewrite v1 reports.

From the repository root:

```bash
PY="/home/matthew-muscat/.local/share/virtualenvs/biopsylocalization-python-a85Yh81c/bin/python"
S="python_files_dcm_meta_based"
DATA="/home/matthew-muscat/Documents/UBC/Research/Data/Output data"
SOURCE="$DATA/anatomical_validation_181_F2_2026-09-10"
DISCOVERY="$DATA/MC_sim_out- Date-Jun-25-2026 Time-11,42,44 - standard-run - inputs-dicom-549_mr-adc-1_mr-t2-0_rtdose-5_rtplan-5_rtstruct-5_us-5/manifests"
WORK="$DATA/biopsy_preprocessing_centroid_fix_2026-09-14"

"$PY" "$S/prepare_patient_scientific_run.py" \
  --scientific-config-snapshot "$SOURCE/provenance/resolved_scientific_config.json" \
  --routing-profile "$DISCOVERY/input_routing_profile.json" \
  --output-dir "$WORK/provenance"

"$PY" "$S/run_patient_scientific_standalone.py" \
  --input-case-manifest "$DISCOVERY/input_case_manifest.csv" \
  --scientific-config-snapshot "$WORK/provenance/resolved_scientific_config.json" \
  --run-compatibility-identity "$WORK/provenance/run_compatibility_identity.json" \
  --output-root "$WORK/plan" \
  --pathway-name biopsy_preprocessing_shadow \
  --checkpoint-name biopsy_preprocessing_shadow \
  --patient-uid '181 (F2)' --capture-input-content

"$PY" "$S/validate_patient_preprocessing.py" \
  --job "$WORK/plan/worker_jobs/patient_0001_181_(F2).json" \
  --output-dir "$WORK/pair_181_F2" --timeout-seconds 3600
```

This is a targeted three-command gate, not a run of the legacy main script.
The first command refreshes source/environment provenance from the retained
scientific snapshot. The second selects the case and writes a sealed job
(planning hashes inputs). The third executes two fresh patient workers. Inspect
`biopsy_preprocessing_pair_summary.json`,
the numerical comparison's biopsy coverage, and both `worker.log` files. Require
real **and** simulated biopsy coverage for the representative case. A changed
field or failed stage is a finding to resolve, not permission to retry with
different tolerances. Add another patient only for a missing mechanism/coverage
case; do not automatically repeat a five-case scheduling campaign.

## Optional unequal-prescription fixture and probe

The builder is implemented and tested using entirely fabricated DICOM files.
No patient-derived fixture was created by the coding agent. Run this locally
yourself when ready, using the explicit F2 source job prepared above:

```bash
SYNTHETIC="/home/matthew-muscat/Documents/UBC/Research/Data/Input data/synthetic set/dose_threshold_v1"
"$PY" "$S/create_synthetic_dose_fixture.py" \
  --source-job "$WORK/plan/worker_jobs/patient_0001_181_(F2).json" \
  --output-dir "$SYNTHETIC"

"$PY" "$S/run_patient_scientific_standalone.py" \
  --input-case-manifest "$SYNTHETIC/manifests/input_case_manifest.csv" \
  --scientific-config-snapshot "$WORK/provenance/resolved_scientific_config.json" \
  --run-compatibility-identity "$WORK/provenance/run_compatibility_identity.json" \
  --output-root "$WORK/synthetic_dose_plan" \
  --pathway-name anatomical_qa --checkpoint-name anatomical_qa \
  --patient-uid 'SYNTHETIC_DOSE_10 (F2)' \
  --patient-uid 'SYNTHETIC_DOSE_13_5 (F2)' --capture-input-content

"$PY" "$S/probe_legacy_dose_order.py" \
  --process-plan "$WORK/synthetic_dose_plan/patient_process_run_plan.json" \
  --output-dir "$WORK/synthetic_dose_probe"
```

The existing probe can now consume a normal process plan directly, so this
mechanism check does not require a full anatomical qualification first. It runs
the unchanged wrapper in forward/reverse order and singleton-reset controls.
For prescription sensitivity, the scientific snapshot must have
`replay.lower_bound_dose_value=None`; an explicit override intentionally removes
prescription dependence. No defaults are changed by the fixture tool.

Expected evidence: identical full dose/gradient lattice and unthresholded points,
different effective lower-bound state under order reversal, potentially changed
thresholded points, and `dependence_observed`, not `unexplained_grid_difference`.
The real five-case dose probe remains historically incomplete. Neither these
fixtures nor the anatomy PASS imply effects on MC/dosimetry results.

The fixture copies all explicitly declared source objects, assigns new
study/series/frame/SOP identities, updates file meta and references among copied
objects, and changes only TARGET prescription and case identity metadata.
Pixel data/geometry remain unchanged. External references remain external; this
is not a full DICOM conformance export or anonymizer. The source data are already
anonymized according to the user. This tool does not remove all retained header
fields or independently certify anonymization. The user's policy is to retain
the derivation trace locally: `synthetic_fixture.json` records the source job,
source content ledger, logical DICOM identities, source-path aliases and UID
remapping for each variant. Data and these source-linked ledgers remain outside
Git. Location, logical identity and content remain separate concepts. For
additional fixture versions use a distinct
`--case-prefix`, for example `SYNTHETIC_DOSE_V2`, and select the emitted case UIDs;
different subdirectory names alone do not prevent case-identity collisions.
Normal recursive discovery sees the files, so work orders
must explicitly select the desired clinical or synthetic subjects. See
[input identity and future fixture recommendations](../../python_files_dcm_meta_based/input_data/DICOM_INPUT_SHAPE.md).

## Next steps and retirement

Next scientific work: pass the representative biopsy gate with the corrected
line samples; resolve the cylinder endpoint/fitted-extent contract as a separate
geometry slice before treating those lengths as established physical extents.
Then qualify transform/optimizer producer outputs
and realized biopsy geometry in bounded slices. Keep classification and guidance
dependencies grounded in actual required products rather than one fixed workflow.

Next independent architecture slice: extract production defaults/config
construction from main with exact snapshot equivalence. Follow with aggressive
removal of replaced main orchestration and typed grid/geometry/biopsy products
that replace dictionary reads. Discovery duplicate/conflict handling is another
near-term independent slice. Neither config extraction nor these designs needs
to wait for a full MC campaign.

The singleton legacy input adapter, historical checkpoint entrypoint names and
legacy dose probe remain transitional. Delete them when replacement numerical
and output gates cover their purposes. Durable content identity, completed
attempt matching, typed scientific config and bounded numeric artifacts remain
useful beyond the migration. No legacy sidecar is retired by this pass.

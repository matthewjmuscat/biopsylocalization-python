# Standalone Biopsy Preprocessing

## Scope and evidence

The user-operated five-case anatomical independence gate passed on 2026-09-13
under clean `b123089`, exact 0/0. Its provenance and limits remain in the
[anatomical runbook](ANATOMICAL_INDEPENDENCE_RUNBOOK.md). The standalone
`biopsy_preprocessing_shadow` pathway and identically named checkpoint are
implemented. The reference-worker guard is fixed in `2852b72`; the accepted
projected-centroid extent and endpoint-inclusive reconstruction are frozen in
`02307c0`. Real-patient geometry characterization for `181 (F2)` is complete and
reviewed; its temporary reporter and test are retired, with CSV/JSON evidence
retained outside the repository. **No real-patient biopsy PASS is claimed yet.**
The next scientific gate is fresh exact paired biopsy preprocessing validation.

The same lightweight parent launches one fresh patient process. The worker
rehydrates verified `PipelineConfig`, builds only that patient's runtime, and
runs dependency-selected grid → anatomical → preprocessing → artifact writing.
The shared paired validator compares fresh standalone and singleton legacy-input
construction, followed by the same downstream science. It verifies completed
attempts, source/config/environment, exact input bytes, stage inventory, resolved
grid state and numerical products. Sharing algorithms limits this evidence to
input/execution migration parity.

## Scientific boundary

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
Patient dictionaries remain transitional compatibility storage; checkpoints
retain bounded products rather than serializing runtime dictionaries.

Uncertainty spreadsheet attachment has no standalone producer/input contract
and remains absent. Realized targeting is still deferred to simulated-biopsy
finalization. Optimization, transforms, realization, classification, MC and
guidance remain fail-closed as standalone live pathways. Planning samples here
are an existing preprocessing product, not later tissue classification.

## Reference-worker boundary

`2852b72` fixes the guard that rejected the first real biopsy pair's reference
lane before science. That attempt produced no numerical comparison and remains
retained as failed execution evidence.

`run_patient_anatomical_reference.py` accepts matching pathway/checkpoint pairs
for `anatomical_qa` and `biopsy_preprocessing_shadow` through the shared
`preprocessing_boundary` selector. The historical filename remains compatible.
The reference lane uses independent singleton legacy input construction in a
fresh worker, followed by the same downstream science and exact 0/0 comparison.

## Accepted biopsy geometry

Contour-slice centroids are the observations. PCA determines an **unsigned**
straight-axis direction; the minimum and maximum centroid projections define
the straightened axial extent. For centroids c_i, their mean m and PCA unit
axis u:

- s_i = (c_i − m) · u.
- A = m + min(s_i) u; B = m + max(s_i) u.
- L = max(s_i) − min(s_i).
- The reconstructed cylinder is a closed physical segment containing A and B.
- N = ceil(L / 0.1 mm), with N + 1 ring positions, including both endpoints;
  axial spacing is L/N and cylinder span is L.

`preprocessing/biopsy_processing/fitted_segment.py` owns this shared calculation.
The existing radius and transverse ring convention are retained.
**Generic `pca.linear_fitter()` remains unchanged.** The model measures straightened
centroid extent, not curved tissue length. The stored `Structure global centroid`
remains the mean of observed centroids and may differ from the cylinder midpoint.

For this transperineal prostate cohort, the signed biopsy frame runs from lower
to higher patient Z; in this acquisition, increasing Z is the needle-tip/superior
direction. This sign comes from the acquisition convention, not PCA or encoded
tip/base metadata, and is not a universal biopsy rule. Degenerate extent and
ambiguous equal-endpoint-Z reconstruction fail explicitly. The
[architecture orientation contract](../architecture/PATIENT_RUNNER_PROCESS_ARCHITECTURE.md#biopsy-geometry-and-orientation-ownership)
records the future orientation policy and legacy naming/schema debt.

The shared reconstruction builder serves real biopsies, canonical simulated
planning and later realized simulated reconstruction. Its corrected geometry
can affect volume/hull inputs, samples, frame origins and downstream
classification or dose/MR queries. Matched-real simulated lengths also consume
the corrected real length. Downstream algorithms retain their existing behavior.
This geometry correction in `02307c0` is separate from the earlier deterministic
centroid line-sample allocation fix in `ec95d279`.

## Preserved 1 mm analysis semantics

Physical reconstruction and the discrete analysis lattice are distinct. The
current sample spacing and analysis voxel length are both 1 mm. In the signed
biopsy frame, sample positions 0, 1, 2, … mm represent the 1 mm analysis voxels
beginning at those positions. Boundary assignment goes upward: z=1 mm belongs
to voxel 2, labelled [1,2] mm.

The physical cylinder includes both endpoints; an exact terminal sample plane
is excluded. A 2 mm cylinder therefore samples z=0,1 mm. Labelled terminal voxel
bounds can extend beyond exact physical extent because analysis is quantized at
1 mm; sample-coordinate range and voxel labels do not redefine cylinder length.
Sampling, frame transforms, voxelization, tables and plotting are unchanged.

Technical note: the existing containment test probes 1e-4 mm forward while
returning the original sample coordinate, so a plane within that margin below
the terminal endpoint can also be excluded.

## Completed real-patient characterization

The old-versus-new geometry characterization for `181 (F2)` has been run and
reviewed. It compared the old helper at `ec95d279` with the accepted geometry now
frozen in `02307c0`, using the same retained contour observations and settings.
The temporary Python reporter and test have been deleted; generated CSV/JSON
evidence remains outside the repository.

| Biopsy | Old fitted length (mm) | New fitted length (mm) | Difference (mm) | Old → new cylinder span (mm) | Sample/voxel count |
| --- | --- | --- | --- | --- | --- |
| Bx_Track RT POST | 15.247336 | 15.135296 | −0.112039 | 15.147680 → 15.135296 | 16 → 16 |
| Bx_Track LT POST | 17.046430 | 17.023435 | −0.022995 | 16.946743 → 17.023435 | 17 → 18 |

Displayed lengths are rounded to six decimal places; differences are reported directly from the characterization output. The PCA extent
correction was small in these two biopsies. LT POST's old cylinder physically stopped below the
17 mm sampling plane despite a nominal fitted length above 17 mm. Reconstruction
through both fitted endpoints restores that plane: its terminal biopsy-frame
sample Z changes from 16 to 17 mm, and the sample/voxel count from 17 to 18.

This reviewed characterization supports the accepted geometry; it does not
establish standalone/reference preprocessing parity. Permanent synthetic tests
cover geometry invariants, deterministic samples, sampling/voxel integration,
checkpoint retention and input-content protection. The real paired gate below
remains outstanding.

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

## Next user-operated real gate

The guard repair and geometry correction are frozen, and characterization is
complete with its temporary source retired. Use fresh destinations and keep
source and inputs stable after preparing provenance. Prepare new environment-v2
execution provenance from the retained scientific snapshot; preserve historical
reports unchanged.

From the repository root:

```bash
PY="/home/matthew-muscat/.local/share/virtualenvs/biopsylocalization-python-a85Yh81c/bin/python"
S="python_files_dcm_meta_based"
DATA="/home/matthew-muscat/Documents/UBC/Research/Data/Output data"
SOURCE="$DATA/anatomical_validation_181_F2_2026-09-10"
DISCOVERY="$DATA/MC_sim_out- Date-Jun-25-2026 Time-11,42,44 - standard-run - inputs-dicom-549_mr-adc-1_mr-t2-0_rtdose-5_rtplan-5_rtstruct-5_us-5/manifests"
WORK="$DATA/biopsy_preprocessing_projected_extent_2026-09-14"

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

The optional builder is tested using entirely fabricated DICOM files. To create
a local patient-derived fixture, use the explicit F2 source job prepared above:

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

The probe consumes a normal process plan directly, so this
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

## Progression and retirement

The next gate is the representative exact paired run above, with real and
simulated biopsy coverage. Further scientific slices and the independent config,
main, typed-state and discovery migration tracks are recorded in the
[roadmap](../roadmap/PATIENT_RUNNER_UPGRADE_ROADMAP.md#september-2026-priorities).

The singleton legacy input adapter, historical checkpoint entrypoint names and
legacy dose probe remain transitional. Retire them when replacement numerical
and output gates cover their purposes. Strict content identity, completed-attempt
matching, typed scientific config and bounded numerical artifacts remain durable
contracts. The completed geometry characterization tool is already retired.

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

## Known scientific defect: centroid samples

`preprocessing/biopsy_processing/biopsy_geometry_helper.py` initializes a NumPy
array, fills row zero, then reads `centroid_line_sample[-1]` before that row has
been initialized. The helper stores the resulting `Centroid line sample pts`
in both real reconstruction and planned models. The focused characterization
test runs the actual helper with deterministic allocation residue and reproduces
different stored samples. It also confirms the reconstructed cylinder remains
unchanged in that synthetic case: its creation uses the initialized first row.
This does not establish absence of effects in every historical downstream use.

The migration deliberately retains this field in exact checkpoint comparison.
It must not be omitted or given a loose tolerance to manufacture a PASS. A
separate, explicit scientific correction is recommended before the real biopsy
acceptance gate; changing the predecessor reference to the previous initialized
row requires its own regression evidence. At this handoff the legacy algorithm
is unchanged and that correction is awaiting the user's scope decision. The
synthetic contract tests use deterministic reconstruction substitutes and do not
claim the allocation defect has been fixed. A real run now is diagnostic only.

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

Review/freeze source first, including any separately approved scientific fix.
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
WORK="$DATA/biopsy_preprocessing_2026-09-14"

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

The first two commands only prepare/plan (planning hashes inputs). The third
executes two patient workers. Inspect `biopsy_preprocessing_pair_summary.json`,
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
is not a full DICOM conformance export or anonymizer. Data stay private and
outside Git. `synthetic_fixture.json` records location, logical identity and
content separately. For additional fixture versions use a distinct
`--case-prefix`, for example `SYNTHETIC_DOSE_V2`, and select the emitted case UIDs;
different subdirectory names alone do not prevent case-identity collisions.
Normal recursive discovery sees the files, so work orders
must explicitly select the desired clinical or synthetic subjects. See
[input identity and future fixture recommendations](../../python_files_dcm_meta_based/input_data/DICOM_INPUT_SHAPE.md).

## Next steps and retirement

Next scientific work: resolve the centroid-sampling defect explicitly, pass the
representative biopsy gate, then qualify transform/optimizer producer outputs
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

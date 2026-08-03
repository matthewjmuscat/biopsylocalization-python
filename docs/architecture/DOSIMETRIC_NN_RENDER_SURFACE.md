# Dosimetric Nearest-Neighbour Render Surface

Last updated: 2026-08-03

## Purpose

This note defines the additive render surface for producing a real-data
dosimetric voxel/lattice figure for the Medical Physics GPR revision while
keeping the long-term patient-runner and GUI architecture clean.

The immediate figure need is to show biopsy voxels or sampled biopsy points in
the HDR dose lattice, including the nearest-neighbour dose-lattice points used
for interpolation and the vectors from biopsy query points to those neighbours.
The same surface should also be useful later as a dose-localization debug tool
and as a domain-specific plugin in a future GUI.

This render surface is the first concrete consumer of the broader patient
context-artifact strategy described in `PATIENT_SCIENTIFIC_CONTEXT_ARTIFACTS.md`.

## Decision

Build this as additive, contract-driven visualization code. It should consume
existing dose-localization contracts and nearest-neighbour outputs. It should
not change dose sampling, inverse-distance weighting, Monte Carlo simulation,
or patient-runner execution semantics.

The render surface should be a domain plugin around the generic render broker,
not a new GUI branch inside `biopsy_localization_convex_main.py`.

The clean near-term placement is:

```text
python_files_dcm_meta_based/
  mc/
    visualization/
      __init__.py
      dose_nn_scene.py
      dose_nn_capture.py
      dose_nn_pyvista.py
      dose_nn_plotly.py
      dose_nn_open3d.py
      dose_nn_selector.py
```

The package name can be adjusted to `mc/rendering/` if that convention becomes
clearer later, but it should stay under `mc/` because the domain being rendered
is Monte Carlo dose localization, not generic UI. The generic broker remains in
`ui/` and should not import MC modules.

## Existing Anchors

The old interactive prototype already exists inside the legacy MC simulator. It
builds the core arrays from `dose_nearest_neighbour_results_dataframe`:

- `Struct test pt vec`
- `Trial num`
- `Dose val (interpolated)`
- `Nearest phys space points`
- `Nearest doses`
- `Nearest distances`

That code path proves the data needed for the figure exists, but it is embedded
in the simulation loop and tied to direct plotting calls. The new surface should
extract the same render payload without making interactive rendering part of
scientific execution.

The better long-term capture inputs are the patient-local dose contracts in
`mc/simulation/per_patient/dose.py`:

- `PatientDoseLatticeContext`
- `PatientDoseBiopsyContext`
- `PatientDoseLocalizationOutputs`

The numerical localizer remains `dosimetric_localizer.dosimetric_localization_dataframe_version(...)`.
That function is an input producer for visualization, not part of the render
plugin.

## Boundary Rules

1. The render surface must be optional and disabled unless explicitly invoked.
2. The render surface must not mutate dose-localization results.
3. The render surface must not change nearest-neighbour search, dose values,
   interpolation weights, or DVH outputs.
4. The render surface may consume a completed nearest-neighbour dataframe, a
   patient-local dose context, or a small saved render scene artifact.
5. The generic render broker must remain toolkit-agnostic and domain-agnostic.
6. Tkinter, Open3D, and Plotly imports should stay in adapter/renderer modules,
   not in numerical dose modules.
7. Real patient-data runs should remain user-operated by default; tests and CI
   should use synthetic tiny scenes.

## Contract Shape

The first stable contract should be a scene object, not a live GUI object.

Suggested core dataclasses:

```text
DoseNNRenderScene
  metadata
  lattice_points
  lattice_doses
  biopsy_points_by_trial
  interpolated_biopsy_doses_by_trial
  nearest_lattice_points_by_trial
  nearest_lattice_doses_by_trial
  nearest_distances_by_trial

DoseNNRenderConfig
  selected_trials
  dose_threshold_min
  dose_threshold_max
  max_lattice_points
  spatial_radius_mm
  biopsy_point_stride
  show_biopsy_points
  show_lattice_points
  show_dose_colorwash
  show_nearest_neighbour_points
  show_nearest_neighbour_vectors
  vector_stride
  camera/view settings

DoseNNRenderBackend
  key
  display_label
  allowed_export_formats
  render(scene, config, export_settings)
```

The backend protocol should start as a small internal static registry rather
than a full third-party plugin manager. That gives the code plugin boundaries
without over-building packaging, discovery, versioning, and dependency policy
before the first figure exists.

Preferred backend modules:

- `dose_nn_pyvista.py` for the primary scientific 3D renderer. PyVista provides
  a Pythonic VTK surface for scalar-colored point clouds, thresholding, clipping,
  arrows/glyphs, scalar bars, camera control, screenshots, and offscreen export.
- `dose_nn_plotly.py` for HTML sharing and lightweight interactive inspection.
- `dose_nn_open3d.py` only as an optional compatibility/inspection backend if it
  remains useful for point-cloud workflows.

ParaView should be treated first as an external export target rather than the
embedded backend. The scene contract should make it straightforward to export
VTK-family files for ParaView later if PyVista is not sufficient for final
publication figure polish.

The selector module should build `RenderBrokerRequest` choice groups and map
broker decisions back to the dose render backend registry. It owns dose-specific
labels, trial choices, and export suggestions. The generic broker should not
learn what a dose lattice, biopsy voxel, trial, or NN vector is.

## GUI Controls

The figure/debug workflow needs dose-domain controls beyond the current generic
broker choice groups. Those controls should belong to the dose selector or scene
config layer, not the broker core unless the same control pattern becomes useful
across multiple domains.

Near-term controls:

- trial selection, including nominal trial 0 and selected MC trials,
- dose-threshold sliders or numeric bounds,
- max displayed lattice points or spatial radius filter,
- biopsy point thinning/stride,
- vector thinning/stride,
- show dose lattice points versus dose colorwash,
- show/hide biopsy query points,
- show/hide nearest-neighbour dose points,
- show/hide vectors from biopsy points to nearest dose-lattice points,
- Plotly export format, size, and scale.

If a richer GUI framework is adopted later, these controls can move into that
adapter while preserving `DoseNNRenderScene`, `DoseNNRenderConfig`, and backend
renderer modules.

## Relationship To The Paused Patient-Runner Work

This detour does not replace the standalone patient-runner plan.

The paused main line remains:

```text
TOML run profile
-> typed resolved run plan
-> standalone parent process
-> one worker process per patient
-> one-patient runtime builder
-> pathway execution
-> patient artifacts and manifests
-> post-run cohort assembly
-> validation against legacy oracle or previous runs
```

Current patient-runner status at the time of this note:

- `run_patient_scientific_standalone.py` can write parent plans and worker job
  packets.
- `run_patient_scientific_worker.py` can load a worker job and write a compact
  result.
- Dry-run worker execution validates the process/job/result boundary.
- Non-dry-run worker execution is intentionally blocked at the missing
  one-patient runtime-state builder.
- The legacy-main live patient-scientific runner remains disabled by default.
- The `from_legacy` bridge remains a validation adapter, not the target primary
  execution path.

The return point after the publication render pass is the one-patient runtime
builder and the run-profile/orchestrator boundary. The dose render work should
not add new dependencies from the patient runner back into legacy main-local
variables.

## Run Location

The primary GUI/render workflow should run post-run or as a standalone utility
over a saved `DoseNNRenderScene` artifact. That is the cleanest fit for the
future GUI: choose a patient, biopsy, trial, thresholds, and backend from a
durable output dataset or captured scene file, then render/export without
re-entering the scientific pipeline.

Once a scene or context artifact exists, this utility should be independently
runnable as many times as needed for figure tuning. Changing display thresholds,
trial selections, vector thinning, camera position, or renderer backend should
not require rerunning the main algorithm.

The only inline runtime role should be optional capture, not interactive
rendering. The raw nearest-neighbour dataframe is intentionally short-lived in
the legacy MC path because it can be large. If the final output dataset does not
retain enough NN detail to reconstruct the figure, an explicit selected run may
capture a small scene artifact while the dataframe exists, then return control
to the post-run renderer. Inline capture should be disabled by default and
should never open render windows inside the MC numerical loop.

## Current Retention Assessment

The normal exported MC CSV/parquet dataset is not sufficient by itself to rebuild
the full nearest-neighbour vector render surface. It retains pointwise and
voxelwise interpolated dose products such as `Point-wise dose output by MC trial
number` and `Voxel-wise dose output by MC trial number`, but those tables do not
retain the nearest dose-lattice point coordinates, nearest-lattice doses, or
nearest-neighbour distances needed to draw the vectors used by the interpolation.

The legacy MC loop currently creates the full `dose_nearest_neighbour_results_dataframe`,
optionally writes it when `raw_data_mc_dosimetry_dump_bool` is enabled, pivots it
to the point-by-trial interpolated dose array, stores that compact dose array on
the biopsy record, and deletes the dataframe. The raw dump is disabled by
default and is documented in `biopsy_localization_convex_main.py` as potentially
hundreds of gigabytes, so it should not become the routine figure workflow.

A results pickle is a transitional reconstruction source because the
results-pickle sanitizer keeps picklable scientific arrays while dropping
runtime-only Open3D/KD-tree objects. In particular, the retained dose-and-gradient
physical-space array can rebuild the dose lattice, and the retained biopsy
sampled/shifted arrays can rebuild the query points. That should allow
nearest-neighbour rows to be recomputed after the run, but it depends on a
legacy pickle workflow rather than the target standalone-run artifact boundary.

The long-term target is not pickle replay. The target is manifest-backed patient
context artifacts: dose lattice arrays, biopsy query geometry, transform
provenance, resolved transform/query arrays, optional nearest-neighbour index
tensors, and selected render-scene artifacts. The dose renderer should consume
those artifacts through stable contracts once they exist.

Historical runs that predate these artifacts will need either a rerun with the
new retention policy or an explicit reconstruction/capture pass when sufficient
legacy retained state exists. The normal pointwise/voxelwise dose tables cannot
be silently upgraded into full NN-vector scene artifacts because they lack the
nearest-neighbour geometry.

The clean target is therefore a dedicated, compact `DoseNNRenderScene` artifact
for selected patient/biopsy/trial cases. That artifact should capture only the
display-scoped lattice subset, biopsy query points, nearest-neighbour points,
nearest-neighbour doses, distances, and provenance needed for rendering. It
should be generated either from retained context artifacts, from a transitional
results pickle by recomputing NN rows, or from an explicit inline capture while
the NN dataframe exists.

## Implementation Phases

### Phase 1: Scene Contract And Synthetic Validation

- Add the `mc/visualization/` package.
- Define scene/config/backend contracts.
- Build a scene from a small synthetic nearest-neighbour dataframe.
- Validate thresholding, trial selection, vector counts, and shape invariants.

### Phase 2: PyVista Scientific Renderer

- Implement a PyVista backend for dose-colored lattice points, biopsy points,
  NN points, optional vectors, scalar bars, camera presets, and screenshots.
- Keep PyVista/VTK imports inside the backend module.
- Validate that the backend can build a non-empty synthetic scene on the local
  workstation before relying on it for real-data figure generation.

### Phase 3: Plotly Sharing Renderer

- Implement a Plotly backend for dose-colored lattice points, biopsy points,
  NN points, and optional vectors.
- Support static export settings through the existing render broker export
  model.
- Keep visual defaults useful for HTML inspection and lightweight exports, but
  do not require Plotly to carry very large point clouds.

### Phase 4: Optional Open3D Inspection Renderer

- Implement an Open3D backend for interactive scene inspection.
- Reuse existing view JSONs only as optional camera presets.
- Keep Open3D imports out of scene contracts and numerical modules.

### Phase 5: Broker Selector

- Build dose-specific choice groups and action handling in `dose_nn_selector.py`.
- Keep arbitrary dose sliders and toggles in the dose selector/config layer.
- Re-enter the broker loop after rendering, matching the optimizer-v2 pattern.

### Phase 6: Real-Data Capture Path

- Add a controlled one-patient, one-biopsy capture path that can produce a
  render scene artifact from existing dose-localization outputs.
- Prefer recomputing from a results pickle or consuming an already-captured
  scene artifact over enabling full raw NN dumps.
- If an inline capture rerun is needed because raw NN data were not retained,
  keep it explicit, selected, and user-operated for real patient data.

## Pushback And Non-Goals

The useful plugin boundary is a renderer/scene boundary, not a new scientific
plugin system. Do not build a broad third-party plugin framework before the
first dose figure is generated.

Do not embed sliders or render windows directly into the MC simulation loop.
The old simulator block can guide the data extraction, but the new design should
move rendering out to replayable scene objects.

Do not use this detour to rewrite patient-runner execution, run profiles, or
input discovery. The only patient-runner-adjacent work allowed here is a small
read-only consumption path for patient-local dose contexts or artifacts.

Do not inspect or batch-run real patient data automatically. The code should be
ready for the user to run on selected real data, while tests stay synthetic.

## Validation Strategy

Validation should start with synthetic scenes because they can be checked without
patient data:

- scene builder rejects missing required dataframe columns,
- selected-trial filtering preserves expected point counts,
- dose thresholding filters lattice points without touching biopsy query points,
- vector generation creates `num_biopsy_points * num_neighbours` segments before
  thinning,
- Plotly backend can build a non-empty figure from the synthetic scene,
- backend modules do not import Tkinter or legacy main.

For real figure generation, record enough provenance beside exported images to
identify patient, biopsy ROI/index, trial selection, threshold settings, code
version, and export settings.
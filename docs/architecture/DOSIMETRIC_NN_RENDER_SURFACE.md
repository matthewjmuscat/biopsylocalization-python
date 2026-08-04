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

## Current Checkpoint

The manifest/accounting detour is resolved enough to resume this render surface.
The codebase now has a code-owned manifest catalog, a post-run presence scanner,
a per-run `run_manifest_index.json` writer, patient-runner batch integration,
and producer-local contract declarations for the patient-runner manifests and
the run manifest index. That is the durable pattern to reuse as new selected
scene/context artifacts are produced.

This does not mean every legacy output boundary now writes a run manifest index.
It means the brittle "manual manifest of manifests" problem has a stable owning
pattern: producer modules declare their manifest contract locally where
practical, run boundaries explicitly record written/skipped/failed manifest
events, and post-run tools can ask the generated index what was actually
produced. Additional legacy or GUI run boundaries should adopt the recorder when
they gain new manifest writers.

The next render pass should continue from the existing renderer-agnostic scene
contract and saved scene-artifact layer, not from raw scientific code. For new
runs, the target source is algorithm-completion context artifacts. If a
historical run did not retain enough context for a real-data figure, the
fallback decision should be explicit: read an already saved selected scene
artifact, reconstruct from a transitional results pickle, or add a
disabled-by-default inline capture hook after approval.

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
`DoseNNRenderScene` is not an exported image. It is renderer-agnostic scene data
that can be consumed by PyVista, Plotly, Open3D, a future GUI, or a batch movie
exporter. Images, HTML views, and movies are derived outputs created from the
scene plus `DoseNNRenderConfig` and backend-specific export settings.

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

The dose-owned GUI control contract is `DoseNNRenderControlSelection`. It is a
toolkit-neutral object that represents what a GUI user selected, then translates
to `DoseNNRenderConfig` and backend-specific settings. The main algorithm should
not construct one of these objects during scientific execution. It should write
the general retained context and manifest summaries needed for the GUI to build
one after the run.

The first concrete adapter for that contract is
`TkDoseNNRenderControlSelectionAdapter`. It is a dose-layer Tkinter dialog that
appears after the generic saved-scene broker selection, uses manifest summaries
to prefill/validate trial and dose-range controls, and returns a
`DoseNNRenderControlSelection`. The generic broker remains unaware of dose
thresholds, lattice/colorwash semantics, reference biopsy points, or NN vectors.

The artifact-backed render bridge should keep the heavy render context
array-first. The retained lattice context owns lattice coordinates and scalar
dose values. A separate dose NN render context artifact owns query rows, trial
IDs, interpolated query doses, nearest-neighbour points, nearest-neighbour
doses, and nearest-neighbour distances. Rebuilding a `DoseNNRenderScene` for
post-run display combines those two artifact handles. Parquet nearest-neighbour
row tables are optional diagnostic or legacy-compatibility views, not the
primary source for broad NN render context.

The GUI-facing bridge should materialize a standard saved-scene artifact from
those retained context handles when a post-run render session needs one. The
existing saved-scene selector can then discover and render the materialized
scene without learning about Zarr layout, lattice artifacts, or NN context
artifact internals. Optional parity checks may rebuild the scene from retained
context and compare it against the runtime scene before the runtime object is
discarded in diagnostic or validation runs.

The same materialization service should support two entry points. Post-run use
reads a patient artifact index, resolves the retained lattice/render-context
artifact refs, writes a standard saved-scene artifact, and launches the existing
selector. Runtime use should call the same service only at an explicit snapshot
boundary requested by run configuration; it should not silently render from or
mutate live scientific objects in the middle of calculation.

## GUI Controls

The figure/debug workflow needs dose-domain controls beyond the current generic
broker choice groups. Those controls should belong to the dose selector or scene
config layer, not the broker core unless the same control pattern becomes useful
across multiple domains.

Near-term controls:

- trial selection, including nominal trial 0 and selected MC trials,
- optional reference biopsy points, especially nominal trial 0, that remain
  visible while rendering selected MC-trial biopsy positions,
- dose-threshold sliders or numeric bounds,
- max displayed lattice points or spatial radius filter,
- biopsy point thinning/stride,
- vector thinning/stride,
- independent dose lattice point and dose colorwash toggles; a colorwash view
  should not require drawing the full lattice point cloud, and NN vectors should
  remain drawable from transformed biopsy positions to the background dose
  lattice/nearest-neighbour targets,
- colorwash style selection between point colorwash, true rectilinear volume
  colorwash when the lattice is complete, and auto fallback,
- show/hide biopsy query points,
- show/hide nearest-neighbour dose points,
- show/hide vectors from biopsy points to nearest dose-lattice points,
- Plotly export format, size, and scale.

The saved-scene render loop should reappear after each render until the user
selects the explicit exit/continue action. This lets the figure be tuned by
changing trial, colorwash, lattice, vector, dose-threshold, and reference-point
settings without rerunning scientific code.

The split is therefore:

```text
patient/runtime artifacts
  retain dose lattice, biopsy query geometry, NN context, trial IDs, ranges, and provenance

dose render controls
  own dose-specific control names, defaults, validation, and translation to render config/settings

generic broker
  owns scene/backend selection and the render-again-or-exit loop

dose Tk control adapter
  collects dose-specific values for the selected saved scene

renderer backend
  consumes the resolved config/settings and writes figures or frames
```

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
- The post-run dose NN context render service can materialize saved-scene
  artifacts from retained lattice/render-context artifact refs in a patient
  artifact index.
- Runtime launching is intentionally not wired yet. When it is needed, it should
  live in the patient-runner MC dose stage after dose-localization outputs are
  finalized and retained artifacts are snapshotted. It should not be wired into
  legacy main or the raw MC numerical loop.

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

The post-run context utility can inspect a patient artifact index before
materialization. `--list-contexts` lists retained dose lattice and dose NN render
context artifact IDs without requiring output paths. Materialization can then
resolve the lattice and render-context refs from artifact metadata when
`--localization-kind` and, when needed, `--biopsy-index` identify a unique pair;
explicit artifact IDs remain available for ambiguous or scripted cases.

Once a scene or context artifact exists, this utility should be independently
runnable as many times as needed for figure tuning. Changing display thresholds,
trial selections, vector thinning, camera position, or renderer backend should
not require rerunning the main algorithm.

The same boundary should support two export modes:

- interactive export, where a GUI button saves the currently viewed rendered
  image, selected scene artifact, or provenance sidecar;
- algorithmic export, where a batch process iterates through many trials or
  camera states to generate figure series or movie frames from retained context
  artifacts.

Large movie-style exports should stream trial windows from context artifacts.
They should not require keeping every trial scene or every rendered frame in
memory at once.

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
should normally be generated from retained context artifacts. Transitional
results-pickle reconstruction or explicit inline capture are fallback paths for
historical runs or selected legacy workflows that predate the context-artifact
surface.

For broad trial coverage, the primary retained artifact should be a compact
context store, not one selected scene per trial. A movie exporter can then slice
or stream the context store by trial, create a prepared scene for the current
frame, render/export it, and release it before moving to the next frame.

## Why Manifest Work Came First

The render surface introduces another manifest-like output: a selected
`DoseNNRenderScene` artifact with JSON metadata and array payloads. Without a
run-level index, post-run tools would have to guess whether that artifact was
written, infer locations from conventions, or rely on a brittle hand-maintained
inventory. That is exactly the failure mode the manifest pass addressed.

The reason was therefore both artifact accounting and scientific reconstruction:

- artifact accounting, because the GUI and publication workflow must know which
  patient, biopsy, trial, scene, and renderer artifacts were produced and where;
- reconstruction, because normal pointwise/voxelwise dose tables do not retain
  nearest-neighbour geometry, so selected scene/context artifacts need explicit
  retention and manifest entries;
- maintainability, because new manifest producers should not require a separate
  manual catalog edit that can drift away from the writer.

For the next dose-render work, the saved scene artifact should be treated as a
normal manifest-backed artifact. When a scene writer is called from a run
boundary, that boundary should record the scene manifest in the run manifest
index. When a scene is rendered later from an already saved artifact, the export
sidecar should record the source scene manifest and display/export settings.

## Implementation Phases

### Phase 1: Scene Contract And Synthetic Validation

- Add the `mc/visualization/` package.
- Define scene/config/backend contracts.
- Build a scene from a small synthetic nearest-neighbour dataframe.
- Validate thresholding, trial selection, vector counts, and shape invariants.
- Status: complete for the renderer-neutral scene/config layer.

### Phase 1B: Selected Scene Artifact IO

- Write selected scene artifacts as JSON metadata plus compressed NumPy arrays.
- Validate round-trip loading, checksum failures, existing-file protection, and
  manifest-only reads on synthetic scenes.
- Status: complete for compact `.npz` selected scenes. Scene manifests now also
  include lightweight GUI summaries such as available trials, per-trial query
  counts, dose ranges, and spatial bounds so selectors can populate controls
  without loading array payloads.

### Phase 1C: Manifest Accounting Boundary

- Catalog the selected scene artifact manifest and run-index manifest surfaces.
- Keep contract declarations beside producers where practical.
- Record written/skipped/failed manifest events at run boundaries rather than by
  hidden global mutation during object construction.
- Status: complete enough to resume rendering; broader legacy run boundaries can
  adopt the same recorder as they gain migrated manifest writers.

### Phase 2: PyVista Scientific Renderer

- Implement a PyVista backend for dose-colored lattice points, biopsy points,
  NN points, optional vectors, scalar bars, camera presets, and screenshots.
- Keep PyVista/VTK imports inside the backend module.
- Validate that the backend can build a non-empty synthetic scene on the local
  workstation before relying on it for real-data figure generation.
- Status: initial synthetic backend complete. `dose_nn_pyvista.py` can build a
  PyVista plotter from `DoseNNPreparedScene`, add named lattice/biopsy/nearest
  neighbour/vector layers, export an offscreen screenshot, and write a provenance
  sidecar.

Offscreen export means PyVista renders to an image buffer/file without opening an
interactive window. It is the right default for tests, CLI use, batch figure
exports, and movie-frame generation. It does not prevent a future interactive
GUI from using an on-screen PyVista plotter for live inspection.

Detailed next pass:

1. Status: added `dose_nn_pyvista.py` with lazy PyVista imports, a small renderer settings
  dataclass, and functions to build a `pyvista.Plotter` from
  `DoseNNPreparedScene`.
2. Status: render lattice points with dose scalars, biopsy query points, nearest-neighbour
  points, optional line/vector geometry, scalar bars, stable colors, and camera
  presets suitable for screenshot export.
3. Status: added synthetic tests that exercise the backend when PyVista is available
  or skip clearly when the optional renderer dependency cannot initialize.
4. Status: added an export function that writes screenshots plus a small provenance JSON
  recording source scene identity, config, backend, camera, image size, and code
  version when available.
5. Status: added a saved-scene CLI/service function that reads a
  `DoseNNRenderScene` artifact, applies render config, and exports a figure
  without touching the scientific pipeline.
6. Next: connect the renderer/export path to retained
  context artifacts for new runs. Historical runs that lack those artifacts can
  use an existing saved scene, transitional results-pickle reconstruction, or an
  approved selected inline capture hook.
7. Status: the PyVista backend now supports explicit point and volume colorwash
  modes. Point colorwash is a translucent scalar-colored point layer over the
  lattice. Volume colorwash is a true PyVista volume render and requires a
  complete three-dimensional rectilinear lattice. Saved-scene CLI controls can
  render lattice only, colorwash only, or both.
8. Status: the PyVista service can export bounded per-trial screenshot frame
  sequences plus a manifest recording selected trials, FPS, render settings, and
  frame/provenance paths. Direct MP4/GIF writing remains a later GUI/export pass
  because the local environment does not currently include a video-writer
  dependency such as `imageio`.
9. Status: colorwash, full dose lattice points, nearest-neighbour points,
  nearest-neighbour vectors, and biopsy points are independently selectable in
  the renderer-neutral config. CLI colorwash renders hide the full lattice point
  actor by default for readability, but `--show-lattice-points` can overlay it.
  NN vectors remain drawable without full lattice points.
10. Status: frame and screenshot renders can keep reference biopsy points visible,
  defaulting to nominal trial 0 through the CLI when requested. This supports
  comparing transformed MC-trial biopsy positions against a fixed nominal
  reference during figure/movie construction.

### Phase 3: Broker Selector And Dose GUI Controls

- Build dose-specific choice groups and action handling in `dose_nn_selector.py`.
- Keep arbitrary dose sliders and toggles in the dose selector/config layer.
- Re-enter the broker loop after rendering, matching the optimizer-v2 pattern.
- Status: initial saved-scene selector layer added. It can discover retained
  scene artifacts from manifests, build stable scene options, and dispatch
  selected saved scenes through the PyVista render service. The generic broker
  now supports PyVista as a selectable backend, preserving the existing
  Open3D/Plotly `both` alias for optimizer-v2. The remaining GUI work is richer
  dose-specific controls and export settings, not basic backend selection. The
  loop already reopens after each render and the saved-scene selector labels the
  terminal action as "Exit renderer".
- Status: added `DoseNNRenderControlSelection` in `dose_nn_render_controls.py`.
  This is the first dose-owned control contract for GUI-selected trials,
  reference biopsy visibility, dose thresholds, lattice/colorwash/vector layers,
  colorwash style, opacity, stride controls, axes, and scalar bar settings. The
  saved-scene selector can now accept that control selection and resolve it per
  scene using manifest trial summaries.
- Status: added `TkDoseNNRenderControlSelectionAdapter` in
  `dose_nn_tk_render_controls.py` and
  `run_saved_dose_nn_scene_controlled_selector_session(...)`. The current GUI
  flow is two-step but cleanly separated: the generic broker selects a saved
  scene/backend, then the dose-specific Tk control dialog collects trial,
  threshold, colorwash, lattice, vector, and reference-point settings for that
  scene before rendering. If the control dialog is cancelled, no render is
  emitted and the broker loop continues.
- Remaining: improve figure-tuning ergonomics around output naming, camera/view
  presets, screenshot dimensions, and eventual movie/video writer controls. The
  current adapter is intentionally a boundary-setting first pass, not the final
  publication GUI.

### Phase 4: Zarr-Backed Retained Scientific Context

This is the scientific artifact-production and retention phase. It should not be
deferred behind optional renderer backends. For large patient context arrays,
Zarr-backed stores are the target implementation from the start, with JSON
manifests carrying schema, shape, dtype, units, chunking, retention policy,
provenance, and path entries.

Initial dose-render context contracts should be code-owned datatypes rather than
new entries appended into the legacy master structure reference dictionary. A
minimal first set is:

- dose lattice context artifact: physical lattice coordinates or regular-grid
  model, dose values, optional gradients, coordinate frame, units, and source
  artifact references;
- biopsy query context artifact: patient/biopsy identifiers, trial numbers,
  original point indices, query points by trial, and transform provenance links;
- dose NN context artifact: nearest lattice indices, nearest distances,
  interpolation outputs, neighbour count, algorithm/config identity, and
  retention level;
- patient artifact manifest entries: stable artifact IDs and relative paths
  under the intentional per-patient subtree.

These should be implemented as a family of dataclasses and artifact specs, not
as one new all-containing patient object. A lightweight patient context/index may
tie them together by artifact ID and path, but the large arrays should remain in
Zarr-backed stores and be accessed through typed readers or handles. The render
scene constructor should request the dose lattice, biopsy query, and dose NN
contexts it needs, slice them by patient/biopsy/trial, and build a
`DoseNNRenderScene` only for the selected view.

For placement, the shared artifact specs and patient artifact index should live
near `output_artifacts`, the dose-specific context dataclasses/writers should
live under `mc/simulation/per_patient/`, patient-runner orchestration should
call those writers from its artifact-store layer, and the visualization package
should own only the post-run context-to-scene reader/constructor. This keeps the
scientific writer beside the dose stage and the GUI/render code downstream of
the retained artifact boundary.

At runtime, these objects may flow through the patient runner as typed stage
outputs, but they should not be a new mutable master dict. The accumulating
state should be the patient artifact index and manifest events. Large arrays are
written once to Zarr-backed stores; later render or dataframe code reads slices
through typed handles.

Producer ownership should target the standalone per-patient runner and its
stage-level output modules. Legacy pathways may call a compatibility adapter to
write equivalent artifacts during migration or validation, but the adapter
should construct the same typed artifacts and manifests rather than mutating the
legacy master dictionary into a new storage API. The master/reference dict can
remain an input/oracle source for transitional parity checks, not the durable
artifact boundary.

Validation for this phase should prove writer/reader round trips, manifest path
and checksum integrity, shape/dtype/unit preservation, and selected parity
against legacy localizer outputs on synthetic or user-operated sampled cases.

### Phase 5: Context-To-Scene And Derived View Constructors

- Build `DoseNNRenderScene` objects on demand from retained context artifacts.
- Reconstruct nearest-neighbour coordinates by indexing the retained dose
  lattice with retained nearest-lattice indices rather than storing repeated
  coordinate triples as the primary context.
- Stream or slice trial windows from Zarr stores for movie/frame workflows.
- Add dataframe constructors for human-readable views such as selected NN rows
  or pointwise dose tables, with constructor manifests that record source
  artifact IDs and filters.
- Keep compact `.npz` selected render-scene artifacts as publication/debug
  derivatives, not as the only durable scientific context.

### Phase 6: Real-Data Figure Generation And Migration

- Use the retained context artifacts to generate selected real-data figures in a
  user-operated pass.
- For historical runs that lack context artifacts, use an existing saved scene,
  a transitional results-pickle reconstruction utility, or an approved selected
  inline capture hook.
- Compare selected standalone context artifacts and derived scenes against the
  legacy oracle during migration.
- Expand retention-policy controls into the run profile once validation is
  credible.

### Optional Later Backends: Plotly And Open3D

- Implement a Plotly backend for dose-colored lattice points, biopsy points,
  NN points, and optional vectors when HTML sharing becomes important.
- Implement an Open3D inspection backend only if it remains useful for point
  cloud workflows.
- Keep both optional backends behind the same scene/config contracts; neither is
  a prerequisite for retained scientific context artifacts.

Known render-backend asymmetry: the generic broker can now carry PyVista
decisions and the dose surface has a PyVista backend, but optimizer-v2 still has
only Open3D and Plotly render implementations. Adding a PyVista optimizer-v2
renderer is feasible but should be treated as a separate future upgrade, not a
dependency of the dose-render figure path.

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
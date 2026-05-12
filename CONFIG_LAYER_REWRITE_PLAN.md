# Config Layer Rewrite Plan

Last updated: 2026-05-12

## Purpose

Define the next config pass as part of the repo-wide config rewrite rather than as a narrow pickle-load fix.

This plan is meant to align:

- future GUI compatibility
- cleaner startup orchestration
- pickle/load boundary correctness
- modular domain-level config passing

## Existing repo direction

The current docs already point in the right direction.

- `GUI_AND_STARTUP_ARCHITECTURE_PLAN.md` says the repo should expose stable orchestration helpers that a GUI can call.
- `python_files_dcm_meta_based/biopsy_optimizer/v2/OPTIMIZER_V2_DESIGN.md` says the first cleanup step should be a typed Python config surface, built as pure-data dataclasses, with one root config object and narrow slices passed to modules.

This rewrite should extend that idea to the broader repo rather than inventing a parallel config model.

## Core principles

1. Config should be pure data.
2. Config should be typed with frozen dataclasses.
3. `main()` should build one root config object.
4. Domain modules should receive only the config slice they need.
5. Artifact loads must distinguish frozen-with-artifact config from current runtime config.
6. `master_structure_info_dict` should not remain the primary config authority.

## Dataclass recommendation

Yes, the config layer should use dataclasses.

Recommended pattern:

- module-level `@dataclass(frozen=True)` types
- only pure-data fields such as `str`, `bool`, `int`, `float`, `Path`, tuples, enums, and other frozen dataclasses
- validation in `__post_init__` when needed

These objects are picklable as long as their fields are picklable and the class is defined at module scope.

That means the following are acceptable inside config:

- `Path`
- tuples of primitive values
- nested frozen dataclasses
- simple dictionaries where needed during transition

The following should not live inside config even if they happen to be picklable or technically serializable:

- progress bars
- live UI adapters
- file handles
- pools, queues, or thread/process executors
- Open3D objects
- derived arrays cached for performance
- giant runtime dataframes

## Root config shape

The config rewrite is larger than just the two pickle-load objects.

Those two objects were only the first boundary split needed for artifact loading.

The larger target should look roughly like this:

```text
PipelineConfig
    StartupConfig
    InputConfig
    OutputConfig
    RuntimeUIConfig
    PreprocessingConfig
    SamplingConfig
    OptimizerConfig
        OptimizerV1Config
        OptimizerV2Config
    SimulationConfig
        MonteCarloConfig
        FanovaConfig
    VisualizationConfig
    GuidanceMapConfig
    ArtifactConfig
```

This does not mean every one of these needs to be fully implemented at once.

It means the rewrite should aim toward one root object composed from clearly named domain configs.

## Existing typed config surfaces to reuse

The repo already has useful precedents.

- `NonBiopsyStructurePreprocessingConfig`
- `GuidanceMapRenderConfig`
- optimizer-v2 config dataclasses in `python_files_dcm_meta_based/biopsy_optimizer/v2/config.py`

These should be reused or wrapped into the root config rather than replaced blindly.

## Artifact-load config ownership

The pickle/load path exposed a broader repo problem: configuration currently has mixed ownership.

Some settings are effectively baked into the dataset meaning, while others are only runtime display choices.

That split must become explicit.

### Frozen-with-artifact config

This config defines the meaning of a saved artifact and should travel with the artifact as a config snapshot.

Examples:

- preprocessing interpolation distances
- geometry-rebuild parameters that affect regenerated meshes
- sampling lattice definitions that downstream stochastic modules depend on
- any future optimizer replay policy that defines search semantics rather than presentation

If these values differ at load time, the runtime should either:

- ignore the runtime value and use the frozen value,
- or fail loudly with a config mismatch.

### Runtime-overridable config

This config may change from session to session without changing the meaning of the saved data.

Examples:

- dose and MR thresholding for display
- color and figure styling
- render backend and export format choices
- UI timeout behavior
- whitelist/filter choices for replay surfaces

## Canonical config storage

The old pattern of copying selected config values into `master_structure_info_dict["Global"]` should be treated as transitional config tracking, not the end state.

The cleaner model is:

1. `PipelineConfig` is the authoritative runtime config object.
2. Artifacts store an explicit config snapshot in export metadata.
3. `master_structure_info_dict` keeps scientific run metadata and derived status fields, not the canonical config source.

That means fields such as "MC info", "Random info", and "Preprocessing info" should gradually move toward one of two roles:

- derived run metadata,
- or explicit config snapshot fields copied from the root config during export.

They should not stay as a partial parallel config system.

## Render manifest scope

The first render manifest should be optimizer-v2 specific.

Do not make the first manifest a generic rendering manifest for the entire repo.

Optimizer-v2 replay, post-load full-dataset rendering, and downstream stochastic rendering should remain separate contracts unless and until they prove they really share one stable schema.

## Recommended phased implementation

Yes, this should be done in phases.

Trying to rewrite the entire repo config surface in one pass would create too much behavioral risk.

### Phase 1: establish root config scaffolding

Deliverables:

1. create a dedicated config package under `python_files_dcm_meta_based/`
2. define a root `PipelineConfig`
3. define a small first set of domain dataclasses and adapter builders
4. have `main()` build the root config without changing behavior yet

Candidate first domains:

- `StartupConfig`
- `PreprocessingConfig`
- `OptimizerV2RuntimeConfig` or wrapper around existing optimizer-v2 configs
- `GuidanceMapConfig`
- `ArtifactConfig`

### Phase 2: artifact boundary split

Deliverables:

1. move preprocessed export to the real post-preprocessing boundary
2. define a frozen config snapshot schema for preprocessed bundle export
3. split pickle-load rebuild inputs into:
   - `FrozenPreprocessedBundleConfig`
   - `RuntimeReplayConfig`
4. enforce mismatch handling for frozen config at load time

This phase is the first place where the config rewrite directly fixes the pickle/load mixing problem.

### Phase 3: migrate domain call sites to narrow config slices

Deliverables:

1. preprocessing modules accept preprocessing config objects only
2. guidance-map workflow accepts guidance/media export config only
3. optimizer-v2 live bridge accepts optimizer/runtime/render config slices only
4. top-of-main scalar settings shrink materially

### Phase 4: stochastic and downstream modules

Deliverables:

1. Monte Carlo config dataclasses
2. FANOVA config dataclasses
3. downstream render/replay config separation
4. remove remaining ad hoc config-tracking duplication where practical

### Phase 5: GUI-facing config surface

Deliverables:

1. stable serializable root config schema
2. GUI-editable config adapter
3. artifact snapshot compatibility checks
4. optional file-backed config loading if still useful

## Immediate recommended next step

The next implementation slice should be:

1. add the root config scaffolding,
2. wrap existing typed configs into that root,
3. split frozen artifact config from runtime replay config,
4. then build the optimizer-v2-specific render manifest on top of that cleaner boundary.

That keeps the config rewrite aligned with both the GUI plan and the pickle/replay work without trying to solve the entire repo in one patch.
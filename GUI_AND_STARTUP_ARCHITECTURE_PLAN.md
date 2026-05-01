# GUI And Startup Architecture Plan

Last updated: 2026-04-30

## Purpose

Define a clean orchestration boundary for future GUI work without pushing UI-specific logic deeper into the scientific pipeline.

This repository is large enough that run bootstrap and dataset-loading logic should have an explicit home. The goal is not a broad mechanical breakup of `main()`. The goal is to introduce a stable orchestration layer that future tools can call.

## Current Direction

Use explicit package names for orchestration and UI adapters instead of a vague catch-all such as `operations`.

- `python_files_dcm_meta_based/startup/`
  Owns run bootstrap and startup workflow helpers.
- `python_files_dcm_meta_based/ui/`
  Owns UI-specific adapters such as Tk file dialogs.
- existing domain packages and modules
  Continue to own scientific processing, data IO, and rendering logic.

## Why This Split

Future GUI surfaces should not need to know how pickle bundles are loaded, how run output directories are created, or which low-level modules need to be called in sequence.

Instead, the layering should be:

1. UI surface gathers inputs.
2. Startup workflow bootstraps the run session.
3. Domain modules perform preprocessing, simulation, MR, dose, plotting, and export work.

This keeps GUI code thinner and makes the same orchestration helpers reusable from a CLI, a notebook-driven workflow, or multiple domain-specific GUIs.

## Initial Package Shape

```text
python_files_dcm_meta_based/
    startup/
        pickle_bundle_run_loader.py
    ui/
        tk_file_dialogs.py
```

The first extracted orchestration seam is the shared pickle-bundle startup path used by:

- the preprocessed-dataset load flow.

That helper owns:

- prompting for the two pickle files,
- loading the reference and info dictionaries,
- bootstrapping per-run output directories,
- returning the loaded session state to the caller.

The old in-pipeline results-bundle reload path is being retired. This pipeline should compute, write tables/manifests, and stop. Any later result consumption should be handled by downstream tools or a future GUI surface, not by re-entering the main pipeline in a special results-load mode.

## Architectural Rules

1. `startup/` should orchestrate workflow order, not implement scientific kernels.
2. `ui/` should stay thin and replaceable.
3. low-level data IO should remain near the relevant storage modules.
4. domain modules should remain callable without a GUI.
5. `main()` should gradually move toward orchestration and stage calls, not direct dialog wiring.

## Near-Term Extractions

1. Move more startup-only branches out of `main()` into `startup/`.
2. Route remaining Tk dialog calls through `ui/` helpers.
3. Define a normalized startup/session object once more branches are extracted.
4. Separate future GUI entrypoints from the scientific pipeline core.

## GUI Direction

This structure supports either of the following without forcing the choice now:

- one overarching application that exposes the full pipeline,
- several narrower tools such as an MR-focused tool, a dose-focused tool, or a validation-focused tool.

In either case, the GUI layer should call stable orchestration helpers rather than duplicating bootstrap logic.
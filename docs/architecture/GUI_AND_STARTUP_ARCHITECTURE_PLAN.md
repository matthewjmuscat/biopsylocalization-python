# GUI And Startup Architecture Plan

Last updated: 2026-05-20

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

That does not mean post-MC checkpointing is useless. A future GUI may still need a cleaner post-simulation artifact contract for:

- regenerating result summaries without rerunning the full pipeline,
- regenerating interactive visualizations from one finished run,
- exporting polished GUI-driven figures or tables.

If that returns, it should be implemented as an explicit post-MC snapshot or manifest/audit artifact surface, not as a legacy-style mutable results pickle that re-enters `main()` mid-pipeline.

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

## Product-Ready Boundary Direction

The future GUI should consume the scientific pipeline through stable contracts,
not by reaching into legacy mutable dictionaries. The durable public boundary
should be input manifests, run/session configuration, patient artifacts, schema
registry entries, validation reports, and orchestration helpers that can run
without a GUI.

This keeps the core repository usable as a validated research/developer engine
while allowing a separate application layer to own product-specific workflow,
deployment, visualization polish, and user interaction. Any business, licensing,
or patent strategy belongs in private notes and counsel review; the public code
direction is simply to keep the scientific core UI-neutral and adapter-driven.

The public scientific repository should therefore expose typed Python contracts
first: `PipelineConfig`, patient-runner configs, input manifests, patient
artifacts, and validation reports. A private GUI or product repository can later
own JSON forms, wizard flows, branded defaults, deployment packaging, and user
experience polish, but those layers should compile down to the public typed
contracts rather than importing main-local variables or legacy dictionaries.

Until scientific-shadow parity is credible, JSON should be treated as a
resolved-config snapshot or future adapter view, not as the primary run-config
authority. This avoids creating a public JSON schema around config groupings that
are still being validated.

## Selectable Stage Direction

The GUI-facing architecture should support both a single full workflow and
separate task-focused tools. Near-term module extraction should therefore avoid
blending distinct scientific domains into one wrapper when those domains could
be useful as independent product actions.

Examples of future selectable actions:

- map dose,
- map MR,
- perform targeting,
- perform QA or validation,
- run the full patient/cohort pipeline.

The selection can be driven by available input data, explicit user choice, or a
combination of both. A GUI mother application can still compose these smaller
stage surfaces into guided workflows, but the domain modules themselves should
remain independently callable.

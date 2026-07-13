# Patient-Runner Process Architecture

Last updated: 2026-07-08

This note defines the target execution architecture for moving the patient
runner outside the legacy all-patient runtime. It is the process and memory
companion to `PATIENT_RUNNER_OUTPUT_ARCHITECTURE.md`: patient execution should
produce durable patient artifacts and manifests, while cohort assembly and
validation consume those artifacts after the fact.

## Core Decision

The current `from_legacy` patient runner remains a validation and migration
adapter. It is useful because it can carve one-patient views from a completed
legacy runtime and compare patient-runner artifacts against the legacy oracle.
It should not become the permanent primary execution path.

The primary patient runner should be a standalone parent/worker architecture:

```text
run profile / typed run config
-> parent orchestrator
   -> patient worker process: load/build one patient, run pathway, write artifacts, exit
   -> patient worker process: load/build one patient, run pathway, write artifacts, exit
   -> ...
-> run manifest from worker manifests
-> optional post-run cohort assembly
-> optional validation against a completed oracle or previous run
```

The parent process must not hold a full cohort scientific runtime. It should
hold only run configuration, patient inventory, scheduling state, worker attempt
records, manifest paths, and lightweight status summaries.

## Why Process Isolation

One patient per worker process is the clean memory boundary.

In a same-process loop, Python objects may become unreachable after a patient
finishes, but native allocators, NumPy/pandas memory, CUDA contexts, and GPU/RMM
memory pools may still keep memory reserved for reuse. This can make a
sequential run look like it is leaking or growing even when ordinary Python
references have been dropped.

When each patient runs in its own process, process exit releases the Python
heap, native memory, CUDA context, and allocator pools owned by that worker. The
parent keeps the cohort run alive without inheriting the worker's heavy memory.

Process isolation also gives a failure boundary. A patient worker can fail,
timeout, or exceed memory without corrupting already completed patients. The
parent records the failed attempt and then follows the configured failure
policy: stop, continue, or retry.

## Current Bridge Versus Target Runner

Current bridge mode:

```text
legacy main builds full master dictionaries
-> patient runner receives full legacy dictionaries
-> bridge carves shallow one-patient views
-> patient artifacts and manifests are written
-> post-run assembly/validation can consume those artifacts
```

This is appropriate for oracle parity and migration checkpoints, but it can keep
cohort-scale legacy state alive for the duration of the process.

The legacy main path should remain runnable with all patient-runner hooks off.
There are two distinct legacy-adjacent controls:

- `patient_runner_validation_mode = "disabled"` disables the main-facing
   validation hook. In disabled mode the hook returns a skipped result before it
   wraps legacy state or runs patient-runner validation.
- `patient_scientific_runner_mode = "disabled"` skips the live
   patient-scientific runner block inside legacy main. Any production legacy-only
   run should use this value until the standalone primary runner is ready.

Current migration status:

- `run_patient_scientific_standalone.py` writes a parent plan and one JSON
   worker job packet per selected manifest patient.
- `run_patient_scientific_worker.py` loads one worker job packet and writes a
   worker result JSON. Its dry-run mode validates the process/job/result boundary
   without touching patient data.
- Non-dry-run worker execution intentionally reports the missing
   `one_patient_runtime_state_builder` boundary until the patient-local runtime
   builder is implemented.
- The legacy-main live patient-scientific runner default is disabled; the
   from-legacy bridge remains available as an explicit validation adapter.

The long-term removal path should be conservative. First, make both legacy hooks
default to disabled for ordinary legacy runs. Second, move new patient-runner
execution to the standalone parent/worker entrypoint. Third, keep the
`from_legacy` bridge available as an explicit validation adapter until primary
runner parity is proven. Only after that should legacy-main imports, config
fields, and call sites be deleted or moved behind a smaller validation-only
entrypoint.

Target primary mode:

```text
discovery builds patient inventory and run config
-> parent chooses patient jobs
-> worker loads/builds only one patient runtime state
-> worker runs selected dependency-checked pathway
-> worker writes patient artifacts and manifests
-> worker exits
-> parent writes batch/run manifest from worker records
-> assembly reads artifacts from disk
```

The target primary runner should not require
`master_structure_reference_dict` or `master_structure_info_dict` to already
exist for the full cohort. Legacy-shaped dictionaries may still exist inside a
single worker while a migrated stage needs them, but that state should be
patient-local and process-local.

## Paired Oracle And Standalone Runs

The future run profile should allow standalone patient execution and legacy
oracle execution in the same resolved plan. This is not an either/or choice:
intrarun and interrun validation need both modes to remain available.

When both are requested, the preferred execution order is:

```text
run profile
-> resolve config and patient inventory
-> standalone patient runner first
   -> one worker process per selected patient
   -> patient artifacts and manifests
-> legacy oracle runner second
   -> separate process
   -> legacy cohort outputs
-> post-run cohort assembly
-> validation
```

The patient runner should run before the legacy oracle so legacy all-patient
memory does not persist into patient execution. The parent orchestrator should
remain light: it records config, inventory, job order, status, and paths. It
should launch both the standalone runner and the legacy oracle as subprocess
jobs rather than importing heavy scientific state into the parent.

During migration, `biopsy_localization_convex_main.py` remains the legacy oracle
entrypoint and can still produce input manifests. The standalone runner may
initially consume those manifests, but the long-term owner should be shared
input/config services that can be called by the legacy oracle, standalone CLI,
validation tooling, and a future GUI.

## Process Roles

### Parent Orchestrator

The parent owns run-level coordination only:

- parse human TOML profile and build typed Python run config,
- discover or load the patient inventory,
- resolve pathway, stage graph, output root, and resource policy,
- write a plan/provenance record before execution,
- launch one worker per patient attempt,
- collect exit code, timeout state, retry count, elapsed time, and manifest
  paths,
- write the batch/run manifest,
- optionally call post-run cohort assembly after all selected patients finish.

The parent should not pass heavy patient objects through Python function calls
or queues. Worker communication should be path/config/status oriented.

### Patient Worker

The worker owns one patient attempt:

- receive a typed worker job or resolved JSON job packet,
- load/build one patient runtime state from the patient input manifest,
- preflight required core input paths such as RTSTRUCT, RTDOSE, and RTPLAN
   before scientific stages begin,
- adapt to legacy-shaped patient-local state only at compatibility boundaries,
- run the dependency-checked pathway,
- write patient artifacts, patient manifests, and patient logs,
- return a compact status summary,
- exit so the operating system releases worker memory.

The worker should treat artifact writing as the durable success boundary. If a
later patient fails, completed patient artifacts remain valid evidence.

### Post-Run Assembly

Cohort assembly is outside patient execution. It consumes patient/batch
manifests and artifact contracts, reconstructs eligible cohort tables, and
writes assembly evidence. This keeps stitching repeatable without rerunning
scientific stages.

## Configuration Boundary

Use the same config separation used by the validation work:

- TOML is the human-authored run profile.
- Typed Python config/dataclasses are the runtime authority.
- JSON manifests and summaries are generated provenance.

A future primary run profile should describe user-facing execution choices such
as patient selection, pathway, output root, failure policy, worker backend, and
resource limits. It should not become a second loose scientific-config system.
Scientific parameters should still flow through typed config contracts such as
`PipelineConfig` and patient-runner scientific config adapters.

The run profile is an orchestration config, not a replacement for scientific
config. It should own:

- input root or input manifest paths,
- patient/fraction selection,
- pathway/checkpoint selection,
- requested execution jobs such as `standalone_patient_runner`,
  `legacy_oracle`, `post_run_assembly`, and `validation`,
- execution order, failure policy, retry policy, and resource limits,
- output root and provenance labels.

It should not copy every scientific knob from `biopsy_localization_convex_main.py`.
Those values should move gradually into `PipelineConfig` and related typed
domain configs, then the run profile should point at that resolved scientific
config.

## Worker Backends

The reference production backend should be sequential subprocess execution:

```text
for patient in selected_patients:
    launch patient worker process
    wait for worker result
    record manifest/status
```

This is deterministic, simple to validate, and gives the memory reset that the
same-process batch runner cannot guarantee.

A bounded process-pool backend can come later:

```text
launch up to N patient worker processes
record each completed worker result
start the next queued patient when resources are available
```

The pool size should be bounded by measured memory and GPU/device policy, not
by CPU count alone. Thread-based scientific execution should remain avoided for
heavy stages that touch shared legacy dictionaries, Open3D, optimizer state,
MC simulation, CUDA, or other native memory surfaces.

## Failure And Retry Policy

Every patient attempt should produce enough evidence to decide what happened:

- patient UID,
- attempt number,
- worker PID,
- start/end timestamps,
- elapsed seconds,
- exit code,
- timeout flag,
- input-preflight status for required core DICOM paths,
- exception summary when available,
- peak memory when available,
- GPU/device assignment when relevant,
- patient manifest path,
- artifact count and output root.

The Jun28 split-A 185 failure is the motivating example for this contract. The
legacy all-patient path completed, but the from-legacy patient-runner replay
failed when it reran raw-contour pulling from RTSTRUCT paths that were no longer
available. The standalone runner should surface that condition as a worker input
preflight failure before raw-contour pulling, and should keep the failed worker
result separate from completed patient artifacts.

The first supported policies should be:

- `stop_on_failure`: stop the batch after the first failed patient,
- `continue_on_failure`: keep completed artifacts and run remaining patients,
- `retry_then_continue`: retry failed patients up to a configured limit, then
  continue or stop according to a final policy.

This handles patient-specific failures without invalidating the completed run
evidence.

## Memory Policy

The process architecture mainly solves same-process retention. It does not, by
itself, reduce the peak memory required by one very heavy patient. Single-patient
peak memory should be handled by stage-level safeguards, resource limits,
chunking, CUDA/RMM pool policy, or fallback modes when those become necessary.

For the current migration, the priority memory contract is:

- no full-cohort scientific runtime in the parent,
- one patient runtime per worker,
- no heavy patient objects returned from worker to parent,
- worker exits after writing artifacts and manifests,
- post-run assembly reads from disk.

## Implementation Sequence

1. Keep the current `from_legacy` bridge as the validation adapter and document
   that it is not the primary memory architecture.
2. Define typed contracts for `PatientRunPlan`, `PatientWorkerJob`,
   `PatientWorkerResult`, and process resource/failure policy.
3. Add a plan-only CLI path that resolves selected patients, pathway, output
   roots, and worker jobs without launching workers.
4. Add a one-patient worker entrypoint that accepts a resolved job packet,
   builds one patient-local runtime state, runs the selected pathway, writes
   artifacts/manifests, and exits.
5. Add a sequential subprocess parent backend that launches one worker at a
   time and writes a batch manifest from returned worker summaries.
6. Replace full legacy-dictionary startup in the primary runner with patient
   inventory plus one-patient input builders. Legacy-shaped state may remain
   inside the worker while old stage adapters still require it.
7. Validate primary-runner output by assembling artifacts from disk and comparing
   against completed legacy/oracle outputs and same-subset split/full runs.
8. Add bounded process-pool execution only after the sequential subprocess path
   has passing validation and resource logging.

## Phased Upgrade Plan And Risk Controls

The migration should proceed in phases so we can keep momentum without running
large validation jobs after every small edit. The principle is cheap structural
checks after every phase, targeted one-patient checks at patient-runtime
boundaries, and full validation only after behavior-capable boundaries change.

| Phase | Scope | Confidence | Silent mistake risk | Main protections |
| --- | --- | --- | --- | --- |
| 0 | Retain usable validation profiles and disable invalid historical gates | High | Low | Path audits, validation dry-runs, explicit notes for invalid Jun28 split evidence |
| 1 | Add TOML run-profile contract and typed plan-only loader | High | Low to medium | Parse tests, dry-run resolved JSON snapshots, no scientific execution |
| 2 | Add light parent orchestrator that orders standalone, legacy, assembly, validation jobs | High | Low to medium | Plan-only output, subprocess command dry-runs, no heavy state in parent |
| 3 | Extend standalone manifest-first worker planning and preflight | High | Low | Synthetic manifests, path preflight, patient UID filtering tests, worker result JSON checks |
| 4 | Implement one-patient runtime builder from manifest and typed config | Medium | Medium to high | One-patient parity against from-legacy adapter, stage manifests, artifact row/count checks, fail-closed missing-input handling |
| 5 | Extract input discovery/inventory from `main` into shared services | Medium | Medium to high | Compare generated manifests byte/field-wise against legacy-main manifests before running science |
| 6 | Add paired patient-first-then-legacy execution mode | Medium | Medium | Separate subprocesses, output-root separation, run manifest state machine, memory/status logging |
| 7 | Migrate scientific config groups out of `main` | Medium | High if broad | One config group at a time, default-equivalence snapshots, retained legacy regression profiles |
| 8 | GUI/product backend boundary | Medium | Low for science if kept API-driven | GUI calls public CLI/API only, JSON/TOML contract tests, no GUI dependence on `main` globals |

Phase 4 and Phase 7 are the highest-risk areas. Phase 4 can silently change
which patient objects or input files feed the scientific stages. Phase 7 can
silently change defaults. Both should be split into small, reviewable passes
with explicit before/after config or manifest evidence.

Recommended validation cadence:

- After phases 1 to 3: run parser, dry-run, synthetic manifest, and worker JSON
   checks only.
- During phase 4: use one controlled patient first, then a small retained subset.
- Before phase 5 replaces discovery behavior: compare only generated input
   manifests against legacy-main manifests.
- After phases 4, 5, and 6: run post-run assembly and parity on the retained
   Jun25/Jun26-style surfaces or a fresh small subset.
- Run a full validation gate only after a phase changes scientific execution,
   input discovery, or config defaults that affect scientific values.

## Validation Gates

The migration should be validated in layers:

- plan-only profile resolves the expected patients and pathway,
- one-patient worker writes the same artifacts as the from-legacy bridge for a
  controlled patient,
- sequential subprocess batch writes manifests and artifacts for a small subset,
- post-run assembly reconstructs cohort surfaces from worker artifacts,
- intrarun parity compares assembled patient-runner output with a completed
  legacy/oracle run,
- full-vs-split reconstruction confirms the same patient set produces equivalent
  cohort surfaces when run as one batch or as split batches.

The Jun25/Jun26 and Jun28 TOML validation profiles are examples of the final two
gates once completed output folders exist.

## Guardrails

- Do not move raw CUDA/kernel math while building this process layer.
- Do not make the parent process a hidden holder of full legacy dictionaries.
- Do not return dataframes, point clouds, or arrays from worker to parent.
- Do not mix process architecture changes with scientific behavior changes.
- Do not make post-run assembly part of one patient's scientific execution.
- Keep the legacy oracle runnable until primary-runner parity is proven on the
  contracted output surface.

## End State

The desired end state is a primary patient runner that can run a full cohort as
a sequence of isolated patient jobs, preserve successful patient artifacts when
other patients fail, scale naturally into bounded parallel workers, and validate
by assembling artifact manifests from disk. The legacy pathway remains valuable
as an oracle during migration, but it is no longer required to launch the
patient runner.
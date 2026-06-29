# Patient-Runner Process Architecture

Last updated: 2026-06-28

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
- exception summary when available,
- peak memory when available,
- GPU/device assignment when relevant,
- patient manifest path,
- artifact count and output root.

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
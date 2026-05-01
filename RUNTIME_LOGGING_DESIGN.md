# Runtime Logging Design

Last updated: 2026-04-30

## Purpose

Define a cheap but critical runtime logging surface that writes to disk during execution so it is possible to tell where the program died, especially in cases of silent crashes, memory pressure, or long-running stages.

This design is intentionally biased toward robustness and triage value over sophistication.

## Main Problems To Solve

- some runs fail or terminate without leaving a clear traceback in the terminal history,
- long runs can die after hours of work,
- memory-related failures are hard to localize after the fact,
- the current monolithic entrypoint makes it difficult to infer the last successful stage unless that state is written to disk.

## Design Goals

- always-on and cheap enough for normal runs,
- safe to append during long-running execution,
- flushed often enough that the last successful step is preserved,
- readable by humans first,
- structured enough for later machine parsing,
- modular enough to survive the ongoing refactor.

## Non-Goals

- full debug tracing of every function call,
- high-volume per-point or per-voxel logging,
- replacing proper exception handling,
- solving every crash by itself.

## Recommended Log Surface

Use a small set of complementary files rather than one overloaded file.

Recommended layout inside the run output directory:

```text
<specific_output_dir>/
  logs/
    run.log
    events.jsonl
    run_status.json
```

## File Roles

### 1. `run.log`

Human-readable append-only text log.

Use this for quick inspection in the shell or editor.

Each line should contain at least:

- UTC timestamp,
- elapsed seconds,
- level,
- phase,
- patient or structure context when available,
- message.

Example shape:

```text
2026-04-30T18:22:05Z | +0000.0s | INFO  | run_start | output_dir=...
2026-04-30T18:22:11Z | +0006.2s | INFO  | preprocessing.structure_referencer.start | patient=181 (F1)
2026-04-30T18:45:02Z | +1377.1s | INFO  | optimizer_v1.guidance_map.start | patient=181 (F1) | dil=DIL_RP
2026-04-30T18:45:09Z | +1384.2s | WARN  | memory.snapshot | rss_mb=10432 | gpu_used_mb=22118
```

### 2. `events.jsonl`

Structured append-only event stream.

One JSON object per line.

This is the durable machine-readable source for later tooling.

Suggested core fields:

- `timestamp_utc`
- `elapsed_sec`
- `level`
- `event_type`
- `phase`
- `run_id`
- `pid`
- `patient_uid`
- `structure_id`
- `structure_ref_type`
- `structure_index`
- `message`
- `details`

The `details` object can hold event-specific values without exploding the top-level schema.

### 3. `run_status.json`

Single small checkpoint file that is overwritten in place.

This is the most important crash-localization surface.

At any moment it should answer:

- what run is this,
- what phase is active,
- which patient or structure is being processed,
- what was the last completed checkpoint,
- when was the file last updated,
- what were the most recent memory stats.

If the process dies, this file will usually still point to the last known good stage.

Suggested fields:

- `run_id`
- `status`
- `started_utc`
- `last_update_utc`
- `current_phase`
- `current_patient_uid`
- `current_structure_id`
- `last_completed_checkpoint`
- `last_completed_checkpoint_utc`
- `rss_mb`
- `gpu_used_mb`
- `gpu_free_mb`
- `output_dir`
- `argv`

## Event Categories

The first implementation should only log stage boundaries and critical checkpoints.

Recommended event categories:

- `run_start`
- `run_finish`
- `run_abort`
- `phase_start`
- `phase_end`
- `patient_start`
- `patient_end`
- `structure_start`
- `structure_end`
- `checkpoint`
- `warning`
- `error`
- `exception`
- `memory_snapshot`
- `pickle_export_start`
- `pickle_export_end`
- `pickle_load_start`
- `pickle_load_end`

## Critical Checkpoints For This Pipeline

The first pass should log only the high-value steps where runs historically get stuck or die.

Recommended checkpoints:

1. run initialization complete
2. input discovery complete
3. structure referencer start and end
4. simulated biopsy preparation start and end
5. optimizer-v1 start and end per patient or per DIL
6. guidance-map generation start and end per DIL
7. MC simulation start and end per patient
8. dataframe export start and end
9. figure export start and end
10. pickle export start and end
11. preprocessed pickle load start and end
12. results bundle load start and end

## Memory-Focused Fields

Because the logging surface is partly motivated by memory issues, memory snapshots should be built in from the beginning.

Recommended values when available:

- process RSS in MB,
- process VMS in MB,
- CuPy memory pool used bytes,
- CuPy memory pool total bytes,
- device free and total memory,
- optional system available RAM.

Memory snapshots do not need to be continuous.

A cheap first pass is enough:

- at run start,
- before and after optimizer-v1,
- before and after guidance-map generation,
- before and after MC simulation,
- at any warning or exception.

## Flush Strategy

The logging surface only helps if it survives abnormal termination.

Recommended behavior:

- line-buffered append for `run.log`,
- line-buffered append for `events.jsonl`,
- explicit flush after every event,
- `fsync` on critical checkpoints and on warnings or errors,
- overwrite and flush `run_status.json` on every major checkpoint.

Do not rely on process exit to flush the important log state.

## Failure Semantics

If an exception is caught, emit:

- an `exception` event to `events.jsonl`,
- a human-readable error line to `run.log`,
- a final `status = failed` update to `run_status.json`.

If the process is killed hard and cannot write a final failure event, the stale `run_status.json` still identifies the last active phase.

## Recommended Minimal First Implementation

The first implementation should stay small and centralized.

Phase 1:

- create a single runtime logging helper module,
- wire it only in the main entrypoint,
- emit `run.log`, `events.jsonl`, and `run_status.json`,
- cover only the major checkpoints listed above.

This gives immediate value without waiting for deeper modularization.

Phase 2:

- add richer patient and structure context,
- add per-module helper wrappers,
- add automatic duration capture,
- add optional memory snapshots from more locations.

Phase 3:

- add lightweight log readers or summary scripts,
- integrate with manifest output,
- add warning aggregation.

## Recommended Event Schema For `events.jsonl`

Example event:

```json
{
  "timestamp_utc": "2026-04-30T18:45:02Z",
  "elapsed_sec": 1377.1,
  "level": "INFO",
  "event_type": "phase_start",
  "phase": "optimizer_v1.guidance_map",
  "run_id": "MC_sim_out- Date-Apr-30-2026 Time-18,22,05",
  "pid": 412331,
  "patient_uid": "181 (F1)",
  "structure_id": "DIL_RP",
  "structure_ref_type": "DIL ref",
  "structure_index": 0,
  "message": "Starting guidance-map generation",
  "details": {
    "output_dir": "...",
    "rss_mb": 10432,
    "gpu_used_mb": 22118
  }
}
```

## What Not To Log

Avoid writing:

- giant arrays,
- full dataframes,
- per-point containment details,
- repeated identical status messages inside tight loops,
- raw DICOM payloads.

Those will slow the run and make the log unreadable.

## Relationship To The Input Manifest

The input manifest and the runtime log should complement each other.

- the input manifest explains what data the run consumed,
- the runtime log explains how far the run got and where it failed.

Together they are enough to answer most postmortem questions for failed or suspicious runs.

## Suggested Validation For The Logging Surface

After implementation, validate these cases deliberately:

1. normal successful run writes all three files and closes with `run_finish`
2. handled exception writes failure state and traceback summary
3. manual termination still leaves a useful `run_status.json`
4. long optimizer or MC stages update status often enough to localize a hang
5. logging overhead is negligible relative to pipeline runtime
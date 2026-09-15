# Standalone optimization migration gate

**Status: optimization standalone boundary implemented / gate pending.** No real
patient optimizer qualification has been run for this implementation. Anatomy
and biopsy remain qualified only under their previously recorded source/config
identities; their live checkpoint pathways remain available.

Implementation checks: 129 focused CPU/synthetic tests passed, including the
existing anatomy/biopsy/independence suites, input seals, config rehydration,
worker/reference dispatch and new uncertainty/optimizer contracts. The existing
temporary-worker independence fixture requires the source directory on
`PYTHONPATH`. The new transform test runs the real generator bodies with a NumPy
backend substitute; it proves stream ownership, not NumPy/CuPy bit equivalence.
The documented config/provenance preparation was also exercised in a temporary
directory with a synthetic routing policy. No real DICOM or GPU optimizer ran.

## Execution and scientific scope

`optimization_shadow` runs grid → anatomy → biopsy preparation/planning →
transform generation → optimizer-v1 → optimizer-v2, then writes artifacts and
exits. Simulated-biopsy finalization, realized targeting, classification, MC
simulation and the guidance stage are not enabled. Existing optimizer-v1 guidance
fragments are optimizer outputs and are retained.

Both lanes start in fresh processes. Standalone builds singleton input fragments;
reference calls the existing singleton legacy input builder. Both then use the
same patient scientific adapters. A PASS establishes migration/input-execution
parity, not independent verification of the optimizer algorithms or a new
patient-order/cohort qualification. The historical reference script/builder names
remain compatible.

The missing prerequisite was uncertainty preparation. The existing generator
combines registry components with per-biopsy real/planned variation; the
standalone worker previously attached no uncertainty data. The new headless
patient bridge calls that unchanged generator after planning and preserves the
legacy CSV write/read conversion in memory, including pandas float parsing.
The three choices now live in `preprocessing.uncertainty`, using
`UncertaintyPreparationConfig`: per-biopsy mean, anatomy defaults only, and
quadrature interpreted as two sigma. Main reads the same typed choices. Edited
external spreadsheets remain unsupported in standalone execution; their input
sealing is a separate future contract. CSV coercion of a numeric-only patient UID
fails explicitly rather than silently attaching zero rows.

Production optimizer-v1 uses `dil dimension driven` clouds of 10,000 points and
its existing patient-derived seed. It does not consume the generated transform
banks. V2 consumes biopsy and target-DIL transform prefixes, including available
needle-compartment shifts. Generation retains the existing maximum of MC,
optimizer and stochastic-targeting requirements: 10,000 samples with current
production config. V2's adaptive search ceiling is 256; its downstream-comparable
winner evaluation still uses the configured 10,000 containment trials. This does
not execute the later MC simulation stage. No distributions, seeds, geometry,
search, scoring or winner-selection algorithms were changed.

## Capacity is part of reproducible execution

V2's capacity-packed adaptive policy can change the trial prefix used before
pruning. For example, 10 active candidates in chunks of 10 with budgets 170 and
330 produce prefixes 16 and 32 respectively. Capacity cannot be classified as
universally harmless resource metadata.

With current production counts, a derived chunk at the explicit fallback budget
is 399: `4,000,000 // (10,000 + 1)`. This allows the first adaptive prefix to reach
the 256-trial ceiling; the discovery of capacity coupling does not establish that
these production defaults currently produce different winners on different GPUs.

The newly exposed worker requires a positive explicit
`optimizer.optimizer_v2.capacity.max_test_structures_per_call` and explicit
transform/v1 seeds. Candidate chunk size is resolved before patient loading by
the unchanged pure V2 rule, now shared with preflight. Both resolved values are
passed into the stage and checked in checkpoint evidence. No lane calibrates on
behalf of the other. The ordinary production builder retains its auto-calibrating
defaults; the recipe below deliberately freezes the existing fallback budget for
this gate. A PASS is specific to that frozen config. Separating resource chunking
from a scientifically fixed pruning schedule is a later scientific design change.

## Checkpoint contract

`optimization_checkpoint_v1` extends the existing JSON/NPZ engine. Fixed field
contracts live in `validation/optimization_checkpoint_fields.py`.

| Surface | Retained state |
| --- | --- |
| Earlier boundaries | Anatomy, grids and biopsy reconstruction/planning evidence |
| Every registered structure | ROI/ref/index/type; translation, rotation and dilation banks; biopsy uniform compartment shifts where enabled; six uncertainty parameter arrays and their frame/distribution labels |
| Executed state | Generated count; actual transform/v1 base and patient-derived seeds, scopes and policy version; v1 DIL/v2 target counts; resolved structure and candidate budgets |
| V1 per DIL | Centroid/optimal/all-tested/zero location tables, contained lattice, max-plane guidance fragment |
| V1 patient | Outside/inside/entire overlapped lattice tables and cumulative projection fragment |
| V2 patient | Summary, ranked and all-tested candidate tables, including winner/search audit values |
| Producer routing | Configured biopsy transport family plus V2 transport family, target vector, source and selection metadata for the later realization stage |

Validation copies CuPy arrays to host without changing runtime state. Shapes,
dtypes, table/index/category schemas, row/rank order, categorical values and
numeric values are retained. Explicitly enumerated 25 V2 timing column names
are removed from tables and selection metadata. Their names are reported in
coverage; no broad substring filter suppresses future fields. Caches, render
objects, timers and GPU memory observations are not scientific evidence.

Capture/read reject missing required products, incorrect counts/seeds/capacities,
missing transport outputs and missing non-fallback ranked/tested results. The
paired service and optimization comparator both enforce **abs_tol=0, rel_tol=0**.
Existing source/environment/schema/config and before/after input seals remain.
The shared `capture_validation_checkpoint` switch selects the job's checkpoint;
the two historical capture switches remain compatible.

The added uncertainty policy changes the scientific config payload. Removing only
that new subtree reproduces the previously accepted SHA
`7d8ee8231b2573fab3c081de68ea77fdee1f855553235010ca8104caeaddd70c`.
Fresh default production SHA is
`1aa188841f404dbf8aeb414babfb89697eebeab0cc2b82912900a0177b2db725`, before
pinning capacity. Historical snapshots remain readable evidence but snapshots
missing the policy cannot rehydrate into the expanded tree with the same SHA.
Do not relabel them or weaken verification. Build fresh config/provenance for
new executions, including any renewed anatomy/biopsy gate.

## User-operated representative gate

Run only after reviewing and freezing this pass. Keep source/config/inputs fixed
through both lanes. This is a targeted paired run, **not a run of main**. No
historical output directory is reused. Commands below start at the repository
root. Choose a new `WORK` if the example destination already exists.

```bash
PY="/home/matthew-muscat/.local/share/virtualenvs/biopsylocalization-python-a85Yh81c/bin/python"
S="python_files_dcm_meta_based"
DATA="/home/matthew-muscat/Documents/UBC/Research/Data/Output data"
DISCOVERY="$DATA/MC_sim_out- Date-Jun-25-2026 Time-11,42,44 - standard-run - inputs-dicom-549_mr-adc-1_mr-t2-0_rtdose-5_rtplan-5_rtstruct-5_us-5/manifests"
WORK="$DATA/optimization_shadow_181_F2_2026-09-14_01"
```

1. Generate a fresh resolved snapshot through typed Python config. This explicitly
   selects the configured fallback budget for the gate, preserving all other
   scientific values. It writes no patient data and imports no main or CUDA code.

```bash
"$PY" - "$WORK" <<'PY'
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0, "python_files_dcm_meta_based")
from config.production import build_production_pipeline_config
from config.snapshots import build_pipeline_scientific_config_snapshot, write_pipeline_config_snapshot
from patient_runner.optimization_execution import resolve_fixed_optimization_execution
work = Path(sys.argv[1])
work.mkdir(parents=True, exist_ok=False)
config = build_production_pipeline_config()
v2 = config.optimizer.optimizer_v2
capacity = replace(v2.capacity,
    max_test_structures_per_call=v2.capacity.fallback_max_test_structures_per_call,
    auto_calibrate_max_test_structures_per_call=False,
    verify_calibrated_max_test_structures_per_call=False)
config = replace(config, optimizer=replace(config.optimizer, optimizer_v2=replace(v2, capacity=capacity)))
print("Fixed structure budget / candidate chunk:", resolve_fixed_optimization_execution(config))
snapshot = build_pipeline_scientific_config_snapshot(config)
print("Scientific SHA:", snapshot.config_sha256)
print(write_pipeline_config_snapshot(snapshot, work / "config/resolved_scientific_config.json"))
PY
```

Require exit 0 and budgets `(4000000, 399)`. The JSON is generated evidence of
this explicit typed configuration, not an editable source of scientific defaults.

2. Capture frozen source/environment/routing provenance.

```bash
"$PY" "$S/prepare_patient_scientific_run.py" \
  --scientific-config-snapshot "$WORK/config/resolved_scientific_config.json" \
  --routing-profile "$DISCOVERY/input_routing_profile.json" \
  --output-dir "$WORK/provenance"
```

Require exit 0 and `preparation_record.json`; inspect the printed new provenance
paths and the recorded scientific SHA before continuing.

3. Plan exactly the characterized F2 patient and seal its input bytes.

```bash
"$PY" "$S/run_patient_scientific_standalone.py" \
  --input-case-manifest "$DISCOVERY/input_case_manifest.csv" \
  --scientific-config-snapshot "$WORK/provenance/resolved_scientific_config.json" \
  --run-compatibility-identity "$WORK/provenance/run_compatibility_identity.json" \
  --output-root "$WORK/plan" \
  --pathway-name optimization_shadow --checkpoint-name optimization_shadow \
  --patient-uid '181 (F2)' --capture-input-content
```

Require exit 0. Inspect `plan/worker_jobs/patient_0001_181_(F2).json`: one patient,
matching pathway/checkpoint, five planned scientific stages, input-content ledger.
This command plans; it does not launch science.

4. Run the two fresh GPU worker lanes sequentially, then compare exactly.

```bash
"$PY" "$S/validate_patient_preprocessing.py" \
  --job "$WORK/plan/worker_jobs/patient_0001_181_(F2).json" \
  --checkpoint-name optimization_shadow \
  --output-dir "$WORK/pair_181_F2" --timeout-seconds 14400
```

The timeout is four hours **per lane**, an execution ceiling rather than a runtime
estimate. Inspect `pair_181_F2/optimization_pair_summary.json` and
`optimization_comparison.json`: both workers exit 0, `resolved_state_passed` and
`passed` are true, optimization coverage is complete, and tolerances are both 0.
Both lane `worker.log` files explain failures. A failure or timeout is retained
as failed evidence; use a new destination for a rerun.

After a representative exact PASS, the next gate is
`post_optimizer_biopsy_realization_shadow`, which consumes the retained producer
request and finalizes geometry. Later sampling/classification, MC and guidance
remain distinct work. Typed product boundaries should progressively replace
master dictionaries and permit deletion of corresponding main responsibilities;
this validation checkpoint is transitional evidence, not the new runtime model.

# Anatomical Order and Split Qualification

## Scope and present status

The five-case singleton anatomical input-migration gate passed under commit
`0f2a6361bfb6dee8c27a0dc375c2297657b4e368`. The user subsequently completed this
forward/reverse/split qualification on **2026-09-13: PASS**, from clean commit
`b123089747c7101bc0748369b56e4228ddd61e8d` on `feat/patient_runner`.
Only the user executes patient science. Keep historical sources and reports read-only.

The retained five cases were `181 (F1)`, `181 (F2)`, `184 (F1)`, `184 (F2)`,
and `194 (F2)` (ADC). Split A was `181 (F1)` and `181 (F2)`; split B was the
complement. All five fresh singleton input-builder pairs and standalone
forward/reverse/split-union comparisons passed at exact **0/0**, including
input bytes, config/source/environment identity, planned stages, and resolved
grid state. No order-dependent runtime state was observed at this checkpoint
for these cases and schedules.

Prepared provenance reported with the user-operated PASS:

| Dimension | Recorded value |
| --- | --- |
| Source SHA | `ef3ac9818dbad1ad1b6be3a5b9218a8cfe1a5d8af28a89d04558dd026dabf9d5` |
| Compatibility SHA | `1f8f417f4db1d9b5f4375b97c9bf010832eec4ca399a950d67a421b4f480b0ef` |
| Input policy SHA | `78dbb8c537b8dec9fabb28ffb188be83d421308d2bc73aa827366885ee19fb80` |
| Runtime environment SHA (v1) | `35f4bebe54e7682618881b308d88155870b22ade30d03ea7261dcec4c174b0b4` |
| Scientific config SHA | `7d8ee8231b2573fab3c081de68ea77fdee1f855553235010ca8104caeaddd70c` |
| Output schema / policy | `phase3d_output_schema_registry_v1` / `strict_exact_v1` |

This proves tested anatomical input-migration and scheduling parity. It does
**not** establish biopsy preprocessing, optimization, realization, classification,
dosimetry/MC, guidance, untested patients, concurrency, or independent truth of
shared algorithms.

The optional real legacy-dose probe **did not complete**. Repeated environment
checks exposed import-order-dependent distribution discovery after a legacy
import added vendored packages to `sys.path`. That provenance defect does not
invalidate the qualification PASS and is not a completed dose characterization.
New code captures environment v2 from interpreter installation roots; historical
v1 hashes remain readable and unchanged, and do not match new v2 runs. Prepare
fresh current provenance for future work rather than relabelling old reports.

The recipe runs fresh singleton pairs, one combined forward order, one combined
reverse order, and disjoint split B then A. For five patients this is **25 patient
subprocesses**: 10 in singleton pairs, 10 combined, and 5 split. Numerical
comparison runs in separate child processes, one patient pair at a time.

Forward/reverse qualification is proportionate because ordinary execution already
uses a fresh process per patient; no previous patient's scientific state enters
the next worker. Synthetic tests exercise every permutation of three patients,
actual subprocess/finalization paths, and the actual legacy threshold mechanism.
Additional cyclic real-patient schedules are diagnostic follow-ups if position
effects arise, not a recurring ritual. This gate claims independence for the
tested schedules, not all hypothetical cohorts or concurrency policies.

## What is reused and what is temporary

| Boundary | Ownership and lifetime |
| --- | --- |
| Byte hashing and patient content ledger | `input_data/content_identity.py`; reusable provenance, no DICOM decoding |
| Before/after content verification | Existing ordinary worker, opt-in via planned content identity; reusable |
| Effective dose state and ADC selection/units | Existing grid-stage manifest metadata, `patient_grid_state_v1`; reusable evidence, not a new mutable runtime store |
| Completed job/result/manifest verification and patient matching | `post_run/completed_patients.py`; reusable for future checkpoint comparators |
| Numerical anatomy evidence | Existing `anatomical_checkpoint_v1` JSON/NPZ and comparator; unchanged schema |
| Forward/reverse/split recipe | `validation/anatomical_independence.py`; bounded anatomical qualification, not a framework |
| Legacy input singleton and dose-order probe | Validation-only, removable when corresponding replacement gates no longer require legacy behavior |

The parent holds configuration, jobs, paths, ledgers, and statuses. Workers load
one patient's state, execute existing scientific adapters, write artifacts, and
exit. Numerical readers run afterward outside the orchestration parent. A future
CLI/API/GUI should call the ordinary planner/worker and completed-artifact services;
it should not call the legacy probe or depend on this qualification recipe.

The byte ledger contains ordered role/path assignments, resolved paths, lengths,
and SHA-256 values for **all declared files**, including ADC/US/T2. Its identity is
patient-scoped: cohort selection is not a merge dimension. The worker hashes
before input construction and after scientific work/artifact writing, before
sealing success. Drift fails the attempt. Ledgers remain local, containing source
paths; no patient data are uploaded. Before/after checks cannot prove absence of
every transient modification, so keep inputs immutable throughout execution.
Missing optional modalities are permitted; declared-but-missing files fail.

The ledger is an additive opt-in worker contract. Existing v3 jobs and the paired
validator still work without one. This qualification requires content evidence;
the generic completed-patient matcher refuses unverified identities. Adoption by
ordinary profiles can follow without changing scientific defaults or creating an
anatomy-specific input identity. Existing role/path fingerprints retain their meaning.

## Freeze and prepare

Run from the repository root on the scientific workstation. Freeze the completed
source changes first (commit or an otherwise immutable worktree) and keep the
environment fixed. New code requires fresh provenance; do not edit/relabel old
identity files. The source scientific values can be reused unchanged.

The following paths correspond to the retained local evidence inspected during
design. Choose a different fresh `WORK` directory for every new attempt.

```bash
PY="/home/matthew-muscat/.local/share/virtualenvs/biopsylocalization-python-a85Yh81c/bin/python"
S="python_files_dcm_meta_based"
DATA="/home/matthew-muscat/Documents/UBC/Research/Data/Output data"
SOURCE="$DATA/anatomical_validation_181_F2_2026-09-10"
DISCOVERY="$DATA/MC_sim_out- Date-Jun-25-2026 Time-11,42,44 - standard-run - inputs-dicom-549_mr-adc-1_mr-t2-0_rtdose-5_rtplan-5_rtstruct-5_us-5/manifests"
WORK="$DATA/anatomical_independence_2026-09-13"

"$PY" "$S/prepare_patient_scientific_run.py" \
  --scientific-config-snapshot "$SOURCE/provenance/resolved_scientific_config.json" \
  --routing-profile "$DISCOVERY/input_routing_profile.json" \
  --output-dir "$WORK/provenance"
```

This reuses science but captures current code/environment. No normal main run or
main runner-switch changes are required. Initial default extraction from main
remains transitional; this phase does not duplicate or migrate scientific defaults.

## Run qualification

```bash
"$PY" "$S/validate_patient_independence.py" run \
  --input-case-manifest "$DISCOVERY/input_case_manifest.csv" \
  --scientific-config-snapshot "$WORK/provenance/resolved_scientific_config.json" \
  --run-compatibility-identity "$WORK/provenance/run_compatibility_identity.json" \
  --output-dir "$WORK/qualification" \
  --patient-uid '181 (F1)' --patient-uid '181 (F2)' \
  --patient-uid '184 (F1)' --patient-uid '184 (F2)' --patient-uid '194 (F2)' \
  --split-a-patient-uid '181 (F1)' --split-a-patient-uid '181 (F2)' \
  --timeout-seconds 3600
```

`qualification_plan.json` records all schedules and the frozen jobs/content
ledgers. `singletons/` contains the existing paired-validator outputs;
`runs/forward`, `runs/reverse`, `runs/split_b`, and `runs/split_a` contain ordinary
standalone jobs, worker/patient results, batch manifests, and run indexes.
`comparisons/` contains per-patient numerical reports and comparison summaries.
The top-level `anatomical_independence_summary.json` is the qualification verdict.

Each completed order and the exact split union is compared against the fresh
standalone singleton references. Singleton pairs separately reconfirm the legacy
input-migration gate under the new source identity. The split-B set now includes
184 F2; it is not identical to the historical four-case split experiment.

Timeout/worker/input failures leave a non-passing summary and retained logs.
Configuration/content failures before execution may occur before the destination
is created. Failed attempts are not numerical discrepancies; both are non-PASS.
Never select retries by newest filename or reuse a partial output destination.

## Characterize the legacy dose wrapper separately

For this equal-threshold five-case set, real-data probing is supplemental
characterization, not an additional requirement for the standalone PASS verdict.
The synthetic controls already exercise the actual carry mechanism. Run the
following command when retaining real wrapper traces is useful, or when a
standalone-versus-cohort difference needs localization.

```bash
"$PY" "$S/probe_legacy_dose_order.py" \
  --qualification-plan "$WORK/qualification/qualification_plan.json" \
  --output-dir "$WORK/legacy_dose_probe"
```

This validation-only process invokes the unchanged cohort wrapper through a
streaming input adapter; it holds one patient's scientific products at a time
plus scalar wrapper state. A delegating observer records actual helper inputs,
returns, and grid/point arrays. It never substitutes a copied recurrence. Fresh
legacy singleton input is loaded per patient; content/provenance remain required.
No legacy kernel/wrapper source is modified. The process is not a production
worker and does not characterize every legacy cohort wrapper.

The report distinguishes full dose/gradient grid, unfiltered points, effective
threshold, and filtered points. Expected carry is a finding, not a reason to
modify the oracle. Changes in the full/unfiltered products remain unexplained.

All three retained dose-containing cases resolved to 13.5. Equal thresholds can
conceal numerical order effects; the report states this limitation. Synthetic
controls with unequal prescriptions, missing prescription/fallback zero, explicit
threshold override, and dose-absent patients exercise sensitivity. Do not alter
real inputs/config merely to force the effect. Singleton-reset controls retain
the effective state and numerical products for localization.

## Acceptance and interpretation

- **Standalone PASS:** all five fresh singleton pairs pass; all ordinary plans
  complete in their declared orders; every required stage/artifact/checkpoint
  is present; exact patient sets and input byte identities match; strict config,
  source, environment, bootstrap policy and output schema identities match; all
  numerical and resolved-state comparisons pass at exact 0/0.
- **Numerical FAIL:** any unexplained field, shape/dtype, identity, modality,
  numerical, or resolved-state difference. No tolerance relaxation is offered.
- **Invalid/incomplete:** missing/corrupt evidence, input drift, failed/unlaunched
  workers, timeout, duplicate/omitted patients, incompatible provenance, or absent
  content-verification seals. These also exit nonzero.
- **Legacy characterization:** independently reports dependence observed, not
  observed in tested inputs, or unexplained grid differences. Completing a probe
  does not establish legacy independence. Synthetic sensitivity must pass.

Standalone should match its fresh legacy singleton input reference and its own
other executions. A characterized legacy cohort effect is not automatically a
standalone migration defect or a scientifically desired behavior. No mismatch is
silently waived; separate verdicts preserve what was actually demonstrated.

Historical June25 20:55 and June26 completed runs remain read-only four-case
table regression evidence. Their inputs/routing agree and retained reports pass
19 tables, but they lack current strict provenance and anatomical checkpoints.
The June25 11:42 five-case discovery inventory is usable; completed output
evidence was not found for that run. June28 split evidence remains invalid due
to changed inputs. Never mix these runs into the new strict qualification.

## Continuing migration

The subsequent [biopsy preprocessing pass](BIOPSY_PREPROCESSING_RUNBOOK.md) enables
that bounded standalone pathway for migration validation. Optimizer, realization,
classification, MC, output and guidance pathways remain fail-closed in normal
standalone live workers. Full table assembly parity, whole-main behavior,
parallel GPU scheduling and patient resume remain separate gates. Shared
anatomical algorithm correctness is also not established by migration parity.

Follow [Project North Stars](../architecture/PROJECT_NORTH_STARS.md): continue
config-default extraction, removal of main responsibilities, replacement of
master dictionaries through typed stage products, and actual producer-based
pathway dependencies. Delete legacy singleton/probe adapters when corresponding
replacement numerical gates are sufficient; keep reusable provenance/matching.

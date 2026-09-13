"""Bounded anatomical qualification using the existing process and pair runners.

The experiment owns forward/reverse/singleton/split schedules and reports only.
Numerical comparison and legacy pairs run in separate child processes; the
parent never imports a checkpoint reader or retains scientific patient state.
This recipe is validation scaffolding, not a general experiment framework.
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Sequence


def write_new_json(path: Path, value: dict) -> None:
    """Write a new evidence record exclusively; never overwrite a prior attempt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def qualification_orders(patient_uids: Sequence[str], split_a: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Resolve a small qualification design, preserving caller-specified order."""
    patients, first = tuple(patient_uids), tuple(split_a)
    if len(patients) < 2 or len(set(patients)) != len(patients):
        raise ValueError("qualification requires at least two distinct patients")
    if not first or len(set(first)) != len(first) or not set(first) < set(patients):
        raise ValueError("split A must be a nonempty distinct proper subset")
    return {
        "forward": patients, "reverse": patients[::-1],
        "split_b": tuple(uid for uid in patients if uid not in first), "split_a": first,
    }


def _arm_plan(base, name: str, uids: tuple[str, ...], destination: Path):
    jobs = {job.patient_case.patient_uid: job for job in base.worker_jobs}
    root, run_id = destination / name, "anatomical-independence-" + name
    ordered_jobs = tuple(replace(
        jobs[uid], output_root=root, run_id=run_id,
        job_id=f"patient_{index:04d}_{jobs[uid].patient_case.safe_patient_uid}",
        metadata={**jobs[uid].metadata, "patient_index": index},
    ) for index, uid in enumerate(uids, 1))
    return replace(base, output_root=root, run_id=run_id, worker_jobs=ordered_jobs)


def _child(arguments: list[str], log_path: Path, *, timeout: float) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, *arguments], stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError("validation child failed; see " + str(log_path))


def run_anatomical_independence(
    *, input_case_manifest: Path, scientific_config_snapshot: Path,
    run_compatibility_identity: Path, output_dir: Path,
    patient_uids: Sequence[str], split_a: Sequence[str], timeout_seconds: float = 3600,
) -> dict:
    """User-operated exact anatomical qualification; every execution is fresh.

    Runs N singleton pairs, two full orders and a disjoint split (5N patient
    subprocesses total). Failures retain a non-passing summary; no auto-resume,
    retries, tolerance relaxation, or historical source rewriting occurs.
    """
    from patient_runner.process_runner import build_patient_process_run_plan, run_patient_process_plan, write_patient_worker_job_packets

    orders = qualification_orders(patient_uids, split_a)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout must be positive and finite")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError("qualification requires an absent output directory")
    # One content ledger per patient is fixed before any lane starts.
    base = build_patient_process_run_plan(
        input_case_manifest_path=Path(input_case_manifest), output_root=destination / "source",
        pathway_name="anatomical_qa", checkpoint_name="anatomical_qa",
        patient_uids=orders["forward"], execution_mode="live_workers", timeout_seconds=timeout_seconds,
        scientific_config_snapshot_path=Path(scientific_config_snapshot),
        run_compatibility_identity_path=Path(run_compatibility_identity),
        capture_input_content=True, metadata={"capture_anatomical_checkpoint": True},
    )
    destination.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parents[1]
    report = {"schema_version": "anatomical_independence_v1", "passed": False,
              "abs_tol": 0, "rel_tol": 0, "orders": orders, "singleton_pairs": [], "runs": {}, "comparisons": {}}
    write_new_json(destination / "qualification_plan.json", {
        "schema_version": "anatomical_independence_plan_v1", "orders": orders,
        "base_plan": base.as_mapping(), "abs_tol": 0, "rel_tol": 0,
    })
    try:
        source_jobs = write_patient_worker_job_packets(base)
        reference_jobs = []
        for index, (job, path) in enumerate(zip(base.worker_jobs, source_jobs), 1):
            pair_dir = destination / "singletons" / f"{index:04d}"
            report["singleton_pairs"].append(str(pair_dir / "anatomical_pair_summary.json"))
            _child([str(scripts / "validate_patient_anatomical.py"), "--job", str(path),
                    "--output-dir", str(pair_dir), "--abs-tol", "0", "--rel-tol", "0",
                    "--timeout-seconds", str(timeout_seconds)],
                   destination / "logs" / f"pair_{index:04d}.log", timeout=2 * timeout_seconds + 300)
            pair = json.loads((pair_dir / "anatomical_pair_summary.json").read_text())
            if pair.get("passed") is not True:
                raise ValueError("singleton pair did not pass: " + job.patient_case.patient_uid)
            reference_job = pair_dir / "standalone" / "worker_jobs" / (job.job_id + "_standalone.json")
            reference_jobs.append(reference_job)
        candidates = {}
        for name, uids in orders.items():
            arm = _arm_plan(base, name, uids, destination / "runs")
            results = run_patient_process_plan(arm)
            report["runs"][name] = {"output_root": str(arm.output_root),
                                    "planned_order": list(uids),
                                    "actual_order": [r.worker_job.patient_case.patient_uid for r in results]}
            if len(results) != len(uids) or any(not r.succeeded or r.dry_run for r in results):
                raise ValueError("incomplete qualification run: " + name)
            candidates[name] = [job.job_path for job in arm.worker_jobs]
        candidates["split_union"] = candidates.pop("split_b") + candidates.pop("split_a")
        for name, paths in candidates.items():
            request = destination / "comparisons" / name / "request.json"
            write_new_json(request, {"reference_jobs": list(map(str, reference_jobs)), "candidate_jobs": list(map(str, paths))})
            summary_path = request.parent / "comparison_summary.json"
            report["comparisons"][name] = str(summary_path)
            _child([str(scripts / "validate_patient_independence.py"), "compare", "--request", str(request)],
                   request.parent / "comparison.log", timeout=timeout_seconds)
            comparison = json.loads(summary_path.read_text())
            if comparison.get("passed") is not True:
                raise ValueError("anatomical independence comparison failed: " + name)
        report["passed"] = True
    except Exception as exc:
        report["error"] = str(exc)
    write_new_json(destination / "anatomical_independence_summary.json", report)
    return report


def compare_completed_anatomical_jobs(request_path: Path) -> dict:
    """Post-run numerical child: compare one patient pair at a time, at exact 0/0.

    Reuses the anatomical checkpoint schema and comparator. Scientific-state
    values are compared from stage manifests, not added to a new runtime model.
    """
    from post_run.completed_patients import match_completed_patients
    from validation.anatomical_checkpoint import compare_anatomical_checkpoints, validate_anatomical_checkpoint_identity

    request_path = Path(request_path)
    report = {"schema_version": "anatomical_independence_comparison_v1", "passed": False, "patients": []}
    try:
        request = json.loads(request_path.read_text())
        matches = match_completed_patients(request["reference_jobs"], request["candidate_jobs"])
        for index, (left, right) in enumerate(matches, 1):
            paths, states = [], []
            for completed in (left, right):
                job, patient = completed.job, completed.patient
                if job.pathway_name != "anatomical_qa" or job.checkpoint_name != "anatomical_qa":
                    raise ValueError("anatomical comparison requires anatomical_qa")
                checkpoint = job.patient_output_root / "validation/anatomical/anatomical_checkpoint.json"
                validate_anatomical_checkpoint_identity(
                    checkpoint, patient_uid=job.patient_case.patient_uid,
                    scientific_config_sha256=job.metadata["scientific_config_snapshot_fingerprint_sha256"],
                    expected_metadata={key: patient.metadata[key] for key in (
                        "worker_job_id", "run_id", "run_compatibility_identity", "input_content_identity",
                        "patient_input_manifest_identity_sha256",
                    )},
                )
                state = next(stage for stage in patient.stage_results if stage.stage_name == "grid_preprocessing").metadata["resolved_scientific_state"]
                if state.get("schema_version") != "patient_grid_state_v1" or "dose" not in state or "mr_adc" not in state:
                    raise ValueError("missing resolved grid-state evidence")
                states.append(state)
                paths.append(checkpoint)
            details = request_path.parent / f"patient_{index:04d}.json"
            comparison = compare_anatomical_checkpoints(*paths, abs_tol=0, rel_tol=0, output_path=details)
            report["patients"].append({"patient_uid": left.job.patient_case.patient_uid,
                "numerical_passed": comparison["passed"], "state_passed": states[0] == states[1],
                "comparison_path": str(details), "coverage": comparison["coverage"]["comparison"]})
        report["passed"] = bool(report["patients"]) and all(p["numerical_passed"] and p["state_passed"] for p in report["patients"])
    except Exception as exc:
        report["error"] = str(exc)
    write_new_json(request_path.parent / "comparison_summary.json", report)
    return report

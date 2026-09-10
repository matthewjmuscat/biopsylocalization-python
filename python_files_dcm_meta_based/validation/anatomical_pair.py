"""Two-process anatomical migration validation from one immutable worker job.

This service executes patient data only when invoked explicitly by the user.
Both lanes retain identical input roles and science, but build state independently.
It does not claim independence of shared scientific algorithms or cohort order.
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import traceback


def run_anatomical_pair(*, job_path: Path, output_dir: Path, abs_tol: float, rel_tol: float, timeout_seconds: float) -> dict:
    """Run fresh standalone/reference processes, then compare numeric evidence.

    Tolerances must be chosen before examining results. Output must be fresh.
    Patient scientific failures yield ``passed=False`` and durable lane reports;
    invalid configuration fails before launch. Historical artifacts are read-only.
    """
    for name, value in (("abs_tol", abs_tol), ("rel_tol", rel_tol), ("timeout_seconds", timeout_seconds)):
        if not math.isfinite(value) or value < 0 or (name == "timeout_seconds" and value == 0):
            raise ValueError(name + " must be finite and nonnegative (timeout strictly positive)")
    from patient_runner.process_runner import load_patient_worker_job, launch_worker_job_file, write_patient_worker_result
    from patient_runner.process_finalization import read_validated_worker_patient_result
    from validation.anatomical_checkpoint import compare_anatomical_checkpoints, validate_anatomical_checkpoint_identity

    source = load_patient_worker_job(job_path)
    if source.pathway_name != "anatomical_qa" or source.checkpoint_name != "anatomical_qa":
        raise ValueError("paired validation requires an anatomical_qa job")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError("paired validation output directory must be fresh")
    destination.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "anatomical_pair_v1", "passed": False,
        "patient_uid": source.patient_case.patient_uid,
        "source_job_path": str(Path(job_path).resolve()),
        "abs_tol": abs_tol, "rel_tol": rel_tol,
        "scope": "fresh subprocess per input lane with shared patient anatomical stage adapters",
        "lanes": {},
    }
    checkpoints = {}
    for lane in ("standalone", "legacy_input"):
        job = replace(
            source, output_root=destination / lane, job_id=source.job_id + "_" + lane,
            metadata={**source.metadata, "capture_anatomical_checkpoint": True, "validation_lane": lane},
        )
        job.job_path.parent.mkdir(parents=True, exist_ok=True)
        job.job_path.write_text(json.dumps(job.as_mapping(), indent=2) + "\n", encoding="utf-8")
        script = None if lane == "standalone" else Path(__file__).resolve().parents[1] / "run_patient_anatomical_reference.py"
        try:
            log_path = job.output_root / "worker.log"
            result = launch_worker_job_file(job.job_path, timeout_seconds=timeout_seconds, worker_script_path=script, log_path=log_path)
            write_patient_worker_result(result)
            lane_report = {"succeeded": result.succeeded, "worker_result_path": str(job.result_path), "log_path": str(log_path), "exit_code": result.exit_code}
            if result.succeeded:
                read_validated_worker_patient_result(job, result)
                checkpoint = job.patient_output_root / "validation" / "anatomical" / "anatomical_checkpoint.json"
                validate_anatomical_checkpoint_identity(
                    checkpoint, patient_uid=job.patient_case.patient_uid,
                    scientific_config_sha256=job.metadata["scientific_config_snapshot_fingerprint_sha256"],
                    expected_metadata={
                        "worker_job_id": job.job_id, "run_id": job.run_id,
                        "run_compatibility_identity": job.metadata["run_compatibility_identity"],
                    },
                )
                checkpoints[lane] = checkpoint
                lane_report["checkpoint_path"] = str(checkpoint)
            report["lanes"][lane] = lane_report
        except Exception as exc:
            report["lanes"][lane] = {"succeeded": False, "error": str(exc), "traceback": traceback.format_exc()}
    if len(checkpoints) == 2:
        comparison_path = destination / "anatomical_comparison.json"
        try:
            comparison = compare_anatomical_checkpoints(
                checkpoints["legacy_input"], checkpoints["standalone"],
                abs_tol=abs_tol, rel_tol=rel_tol, output_path=comparison_path,
            )
            report.update(passed=comparison["passed"], comparison_path=str(comparison_path), coverage=comparison["coverage"])
        except Exception as exc:
            report["comparison_error"] = str(exc)
    (destination / "anatomical_pair_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
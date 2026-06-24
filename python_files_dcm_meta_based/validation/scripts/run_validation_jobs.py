from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


VALIDATION_DIR = Path(__file__).resolve().parents[1]
PYTHON_ROOT = VALIDATION_DIR.parent
REPO_ROOT = PYTHON_ROOT.parent
DEFAULT_CONFIG_PATH = VALIDATION_DIR / "configs" / "validation_jobs.json"

SCRIPT_REGISTRY = {
    "validate_run_against_baseline": PYTHON_ROOT / "validate_run_against_baseline.py",
    "compare_cohort_runs": PYTHON_ROOT / "compare_cohort_runs.py",
    "compare_run_csv_outputs": PYTHON_ROOT / "compare_run_csv_outputs.py",
    "compare_patient_runner_parity": PYTHON_ROOT / "compare_patient_runner_parity.py",
    "compare_reconstructed_cohort_runs": PYTHON_ROOT / "compare_reconstructed_cohort_runs.py",
}


@dataclass(frozen=True, slots=True)
class ValidationJobResult:
    group_name: str
    job_name: str
    script_name: str
    returncode: int
    command: tuple[str, ...]

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        config = json.load(file_obj)
    if not isinstance(config, dict):
        raise TypeError("validation config root must be a JSON object")
    return config


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _resolve_named_path(config: Mapping[str, Any], key_name: str) -> Path:
    named_runs = config.get("runs", {})
    named_paths = config.get("paths", {})
    if key_name in named_runs:
        return _resolve_path(named_runs[key_name])
    if key_name in named_paths:
        return _resolve_path(named_paths[key_name])
    raise KeyError(f"No run/path named {key_name!r} in validation config")


def _resolve_job_path(config: Mapping[str, Any], job: Mapping[str, Any], field_name: str) -> Path | None:
    if field_name in job and job[field_name]:
        return _resolve_path(job[field_name])
    run_key_name = f"{field_name}_run"
    if run_key_name in job and job[run_key_name]:
        return _resolve_named_path(config, str(job[run_key_name]))
    path_key_name = f"{field_name}_path"
    if path_key_name in job and job[path_key_name]:
        return _resolve_named_path(config, str(job[path_key_name]))
    return None


def _append_common_options(args: list[str], job: Mapping[str, Any], defaults: Mapping[str, Any]) -> None:
    output_dir = job.get("output_dir")
    if output_dir:
        args.extend(["--output-dir", str(_resolve_path(output_dir))])

    abs_tol = job.get("abs_tol", defaults.get("abs_tol"))
    rel_tol = job.get("rel_tol", defaults.get("rel_tol"))
    if abs_tol is not None:
        args.extend(["--abs-tol", str(abs_tol)])
    if rel_tol is not None:
        args.extend(["--rel-tol", str(rel_tol)])


def _build_validate_run_args(config: Mapping[str, Any], job: Mapping[str, Any], defaults: Mapping[str, Any]) -> list[str]:
    baseline = _resolve_job_path(config, job, "baseline")
    candidate = _resolve_job_path(config, job, "candidate")
    if baseline is None or candidate is None:
        raise ValueError("validate_run_against_baseline jobs require baseline and candidate")
    args = [str(baseline), str(candidate)]
    _append_common_options(args, job, defaults)
    top_n = job.get("top_n", defaults.get("top_n"))
    if top_n is not None:
        args.extend(["--top-n", str(top_n)])
    for pattern in job.get("diagnostic_column_regex", ()):
        args.extend(["--diagnostic-column-regex", str(pattern)])
    return args


def _build_two_run_compare_args(config: Mapping[str, Any], job: Mapping[str, Any], defaults: Mapping[str, Any]) -> list[str]:
    baseline = _resolve_job_path(config, job, "baseline")
    candidate = _resolve_job_path(config, job, "candidate")
    if baseline is None or candidate is None:
        raise ValueError(f"{job.get('script')} jobs require baseline and candidate")
    args = [str(baseline), str(candidate)]
    _append_common_options(args, job, defaults)
    return args


def _build_patient_runner_parity_args(config: Mapping[str, Any], job: Mapping[str, Any], defaults: Mapping[str, Any]) -> list[str]:
    legacy_output = _resolve_job_path(config, job, "legacy_output")
    patient_runner_output = _resolve_job_path(config, job, "patient_runner_output")
    if legacy_output is None or patient_runner_output is None:
        raise ValueError("compare_patient_runner_parity jobs require legacy_output and patient_runner_output")
    args = [str(legacy_output), str(patient_runner_output)]
    _append_common_options(args, job, defaults)

    assembled_table_dir = job.get("assembled_table_dir")
    if assembled_table_dir:
        args.extend(["--assembled-table-dir", str(_resolve_path(assembled_table_dir))])
    for table_name in job.get("final_table_names", ()):
        args.extend(["--final-table-name", str(table_name)])
    if bool(job.get("recursive_csv", False)):
        args.append("--recursive-csv")
    if bool(job.get("skip_assembled_cohort", False)):
        args.append("--skip-assembled-cohort")
    return args


def _resolve_job_path_list(config: Mapping[str, Any], job: Mapping[str, Any], field_name: str) -> tuple[Path, ...]:
    direct_values = job.get(field_name, ())
    if direct_values:
        if isinstance(direct_values, str):
            raise TypeError(f"{field_name} must be a list when provided")
        return tuple(_resolve_path(value) for value in direct_values)

    run_values = job.get(f"{field_name}_runs", ())
    if run_values:
        if isinstance(run_values, str):
            raise TypeError(f"{field_name}_runs must be a list when provided")
        return tuple(_resolve_named_path(config, str(value)) for value in run_values)

    path_values = job.get(f"{field_name}_paths", ())
    if path_values:
        if isinstance(path_values, str):
            raise TypeError(f"{field_name}_paths must be a list when provided")
        return tuple(_resolve_named_path(config, str(value)) for value in path_values)

    return ()


def _build_reconstructed_cohort_args(config: Mapping[str, Any], job: Mapping[str, Any], defaults: Mapping[str, Any]) -> list[str]:
    reference = _resolve_job_path(config, job, "reference_patient_runner_output")
    if reference is None:
        raise ValueError("compare_reconstructed_cohort_runs jobs require reference_patient_runner_output")
    split_outputs = _resolve_job_path_list(config, job, "split_patient_runner_outputs")
    if not split_outputs:
        raise ValueError("compare_reconstructed_cohort_runs jobs require split_patient_runner_outputs")

    args = [str(reference), *(str(path) for path in split_outputs)]
    _append_common_options(args, job, defaults)
    for table_name in job.get("final_table_names", ()):
        args.extend(["--final-table-name", str(table_name)])
    if bool(job.get("allow_patient_set_mismatch", False)):
        args.append("--allow-patient-set-mismatch")
    return args


def _build_script_args(config: Mapping[str, Any], job: Mapping[str, Any], defaults: Mapping[str, Any]) -> list[str]:
    script_name = str(job.get("script", "")).strip()
    if script_name == "validate_run_against_baseline":
        return _build_validate_run_args(config, job, defaults)
    if script_name in {"compare_cohort_runs", "compare_run_csv_outputs"}:
        return _build_two_run_compare_args(config, job, defaults)
    if script_name == "compare_patient_runner_parity":
        return _build_patient_runner_parity_args(config, job, defaults)
    if script_name == "compare_reconstructed_cohort_runs":
        return _build_reconstructed_cohort_args(config, job, defaults)
    raise ValueError(f"Unsupported validation script: {script_name!r}")


def _job_enabled(job: Mapping[str, Any]) -> bool:
    return bool(job.get("enabled", True))


def _iter_enabled_jobs(config: Mapping[str, Any]) -> Sequence[tuple[str, str, Mapping[str, Any]]]:
    jobs: list[tuple[str, str, Mapping[str, Any]]] = []
    run_groups = config.get("run_groups", {})
    if not isinstance(run_groups, dict):
        raise TypeError("run_groups must be a JSON object")
    for group_name, group_config in run_groups.items():
        if not isinstance(group_config, dict):
            raise TypeError(f"run group {group_name!r} must be a JSON object")
        if not bool(group_config.get("enabled", True)):
            continue
        group_jobs = group_config.get("jobs", [])
        if not isinstance(group_jobs, list):
            raise TypeError(f"run group {group_name!r} jobs must be a list")
        for index, job in enumerate(group_jobs, start=1):
            if not isinstance(job, dict):
                raise TypeError(f"job {index} in group {group_name!r} must be a JSON object")
            if not _job_enabled(job):
                continue
            job_name = str(job.get("name") or f"job_{index}")
            jobs.append((str(group_name), job_name, job))
    return tuple(jobs)


def _run_job(config: Mapping[str, Any], group_name: str, job_name: str, job: Mapping[str, Any]) -> ValidationJobResult:
    defaults = config.get("defaults", {})
    script_name = str(job.get("script", "")).strip()
    script_path = SCRIPT_REGISTRY.get(script_name)
    if script_path is None:
        raise ValueError(f"Unsupported validation script: {script_name!r}")
    if not script_path.is_file():
        raise FileNotFoundError(script_path)

    script_args = _build_script_args(config, job, defaults)
    command = (sys.executable, str(script_path), *script_args)
    print(f"\n[validation-jobs] {group_name}/{job_name}: {script_name}")
    print("[validation-jobs] command:", " ".join(command))

    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PYTHON_ROOT) if not current_pythonpath else f"{PYTHON_ROOT}{os.pathsep}{current_pythonpath}"
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    return ValidationJobResult(
        group_name=group_name,
        job_name=job_name,
        script_name=script_name,
        returncode=int(completed.returncode),
        command=tuple(command),
    )


def _write_run_summary(config_path: Path, results: Sequence[ValidationJobResult]) -> Path:
    output_path = REPO_ROOT / "validation_outputs" / "configured_validation_last_run.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "configured_validation_run_v1",
        "generated_utc": _utc_now_iso(),
        "config_path": str(config_path),
        "job_count": len(results),
        "failed_job_count": sum(1 for result in results if not result.succeeded),
        "jobs": [
            {
                "group_name": result.group_name,
                "job_name": result.job_name,
                "script_name": result.script_name,
                "returncode": result.returncode,
                "succeeded": result.succeeded,
                "command": list(result.command),
            }
            for result in results
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    config_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = _read_config(config_path)
    jobs = _iter_enabled_jobs(config)
    if not jobs:
        print(f"[validation-jobs] no enabled jobs in {config_path}")
        return 0

    print(f"[validation-jobs] config: {config_path}")
    print(f"[validation-jobs] enabled jobs: {len(jobs)}")
    results = [_run_job(config, group_name, job_name, job) for group_name, job_name, job in jobs]
    summary_path = _write_run_summary(config_path, results)
    failed_results = [result for result in results if not result.succeeded]
    print(f"\n[validation-jobs] wrote summary: {summary_path}")
    print(f"[validation-jobs] completed jobs: {len(results)} | failed jobs: {len(failed_results)}")
    return 1 if failed_results else 0


if __name__ == "__main__":
    raise SystemExit(main())

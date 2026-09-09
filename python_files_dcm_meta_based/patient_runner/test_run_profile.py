from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from .run_profile import load_patient_orchestration_profile
from config.snapshots import PipelineConfigSnapshot
from config.snapshots import canonical_sha256
from config.snapshots import read_pipeline_config_snapshot
from config.snapshots import write_pipeline_config_snapshot
from output_artifacts.run_compatibility import RunCompatibilityIdentity
from output_artifacts.run_compatibility import write_run_compatibility_identity


_CASE_MANIFEST_COLUMNS = (
    "Patient UID (generated)",
    "Patient Name",
    "Patient ID (from dicom)",
    "Fraction number (legacy parsed)",
    "Has RTSTRUCT",
    "Has RTDOSE",
    "Has RTPLAN",
    "Core RTSTRUCT/RTDOSE/RTPLAN complete",
    "RTSTRUCT path",
    "RTDOSE path",
    "RTPLAN path",
    "Num US files",
    "Num MR T2 files",
    "Num MR ADC files",
    "US paths",
    "MR T2 paths",
    "MR ADC paths",
)


class PatientOrchestrationProfileTests(unittest.TestCase):
    def test_profile_resolves_paths_and_compiles_existing_process_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = _write_case_manifest(root, ("P001", "P002"))
            config_snapshot_path = root.joinpath("scientific_config.json")
            config_payload = {"mc": {"trials": 10}}
            write_pipeline_config_snapshot(
                PipelineConfigSnapshot(
                    config_type="config.PipelineConfig.scientific",
                    config=config_payload,
                    config_sha256=canonical_sha256(config_payload),
                ),
                config_snapshot_path,
            )
            compatibility_identity_path = _write_compatibility_identity(root, config_snapshot_path)
            profile_path = _write_profile(
                root,
                execution_mode="plan_only",
                patient_uids=("P002",),
                scientific_config_snapshot="scientific_config.json",
                run_compatibility_identity="run_compatibility_identity.json",
            )

            profile = load_patient_orchestration_profile(profile_path)
            plan = profile.build_process_run_plan()
            payload = plan.as_mapping()

        self.assertEqual(profile.input_case_manifest_path, manifest_path)
        self.assertEqual(tuple(job.patient_case.patient_uid for job in plan.worker_jobs), ("P002",))
        self.assertEqual(plan.execution_mode, "plan_only")
        self.assertEqual(plan.retention_level, "context")
        self.assertEqual(plan.timeout_seconds, 30.0)
        self.assertEqual(plan.scientific_config_snapshot_path, config_snapshot_path)
        self.assertEqual(plan.run_compatibility_identity_path, compatibility_identity_path)
        self.assertEqual(len(payload["metadata"]["profile_source_fingerprint_sha256"]), 64)
        self.assertEqual(len(payload["metadata"]["input_case_manifest_fingerprint_sha256"]), 64)
        self.assertEqual(len(payload["metadata"]["scientific_config_snapshot_fingerprint_sha256"]), 64)
        self.assertEqual(len(payload["metadata"]["scientific_config_snapshot_file_sha256"]), 64)
        self.assertEqual(len(payload["worker_commands"]), 1)
        self.assertNotIn("--dry-run", payload["worker_commands"][0])

    def test_profile_rejects_checkpoint_pathway_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_case_manifest(root, ("P001",))
            profile_path = _write_profile(
                root,
                execution_mode="plan_only",
                patient_uids=("P001",),
                pathway_name="current_dosimetry_shadow",
                checkpoint_name="anatomical_qa",
            )

            with self.assertRaisesRegex(ValueError, "checkpoint and pathway disagree"):
                load_patient_orchestration_profile(profile_path)

    def test_profile_rejects_unknown_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_case_manifest(root, ("P001",))
            profile_path = _write_profile(
                root,
                execution_mode="plan_only",
                patient_uids=("P001",),
                pathway_name="anatomical_qa",
                checkpoint_name="not_a_checkpoint",
            )

            with self.assertRaisesRegex(ValueError, "Unsupported patient scientific runner checkpoint"):
                load_patient_orchestration_profile(profile_path)

    def test_profile_rejects_missing_scientific_config_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_case_manifest(root, ("P001",))
            profile_path = _write_profile(
                root,
                execution_mode="plan_only",
                patient_uids=("P001",),
                scientific_config_snapshot="missing.json",
            )

            with self.assertRaises(FileNotFoundError):
                load_patient_orchestration_profile(profile_path)

    def test_profile_rejects_unknown_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_case_manifest(root, ("P001",))
            profile_path = _write_profile(
                root,
                execution_mode="plan_only",
                patient_uids=("P001",),
            )
            profile_text = profile_path.read_text(encoding="utf-8")
            profile_path.write_text(
                profile_text.replace('max_workers = 1', 'max_workers = 1\nmax_workerz = 2'),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unsupported fields"):
                load_patient_orchestration_profile(profile_path)

    def test_live_profile_requires_scientific_config_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_case_manifest(root, ("P001",))
            profile_path = _write_profile(
                root,
                execution_mode="live_workers",
                patient_uids=("P001",),
            )

            with self.assertRaisesRegex(ValueError, "requires scientific_config.snapshot"):
                load_patient_orchestration_profile(profile_path)

    def test_live_profile_requires_run_compatibility_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_case_manifest(root, ("P001",))
            config_snapshot_path = root.joinpath("scientific_config.json")
            config_payload = {"mc": {"trials": 10}}
            write_pipeline_config_snapshot(
                PipelineConfigSnapshot(
                    config_type="config.PipelineConfig.scientific",
                    config=config_payload,
                    config_sha256=canonical_sha256(config_payload),
                ),
                config_snapshot_path,
            )
            profile_path = _write_profile(
                root,
                execution_mode="live_workers",
                patient_uids=("P001",),
                scientific_config_snapshot="scientific_config.json",
            )

            with self.assertRaisesRegex(ValueError, "requires scientific_config.run_compatibility_identity"):
                load_patient_orchestration_profile(profile_path)

    def test_profile_cli_runs_cpu_only_dry_run_workers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_case_manifest(root, ("P001",))
            profile_path = _write_profile(
                root,
                execution_mode="dry_run_workers",
                patient_uids=("P001",),
            )
            script_path = Path(__file__).resolve().parents[1].joinpath("run_patient_scientific_standalone.py")

            completed = subprocess.run(
                (sys.executable, str(script_path), "--profile", str(profile_path)),
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            plan_path = root.joinpath("output", "patient_process_run_plan.json")
            with plan_path.open("r", encoding="utf-8") as plan_file:
                plan_payload = json.load(plan_file)

        self.assertEqual(plan_payload["execution_mode"], "dry_run_workers")
        self.assertEqual(plan_payload["requested_jobs"], ["standalone_patient_runner"])
        self.assertIn("--dry-run", plan_payload["worker_commands"][0])


def _write_profile(
    root: Path,
    *,
    execution_mode: str,
    patient_uids: tuple[str, ...],
    pathway_name: str = "anatomical_qa",
    checkpoint_name: str = "anatomical_qa",
    scientific_config_snapshot: str = "",
    run_compatibility_identity: str = "",
) -> Path:
    profile_path = root.joinpath("patient_run.toml")
    patient_values = ", ".join('"{}"'.format(patient_uid) for patient_uid in patient_uids)
    snapshot_section = ""
    scientific_config_lines = []
    if scientific_config_snapshot:
        scientific_config_lines.append('snapshot = "{}"'.format(scientific_config_snapshot))
    if run_compatibility_identity:
        scientific_config_lines.append(
            'run_compatibility_identity = "{}"'.format(run_compatibility_identity)
        )
    if scientific_config_lines:
        snapshot_section = "\n[scientific_config]\n{}\n".format("\n".join(scientific_config_lines))
    profile_path.write_text(
        """schema_version = "patient_orchestration_profile_v1"
description = "Synthetic standalone profile"
enabled = true

[run]
run_id = "synthetic-profile"
output_root = "output"
pathway = "{pathway_name}"
checkpoint = "{checkpoint_name}"

[inputs]
case_manifest = "input_case_manifest.csv"

[selection]
patient_uids = [{patient_values}]

[execution]
mode = "{execution_mode}"
requested_jobs = ["standalone_patient_runner"]
failure_policy = "stop_on_failure"
max_workers = 1
timeout_seconds = 30

[artifacts]
retention_level = "context"
{snapshot_section}
[metadata]
purpose = "unit_test"
""".format(
            pathway_name=pathway_name,
            checkpoint_name=checkpoint_name,
            patient_values=patient_values,
            execution_mode=execution_mode,
            snapshot_section=snapshot_section,
        ),
        encoding="utf-8",
    )
    return profile_path


def _write_compatibility_identity(root: Path, config_snapshot_path: Path) -> Path:
    snapshot = read_pipeline_config_snapshot(config_snapshot_path)
    path = root.joinpath("run_compatibility_identity.json")
    write_run_compatibility_identity(
        RunCompatibilityIdentity(
            scientific_config_sha256=snapshot.config_sha256,
            code_source_sha256="synthetic-code-source",
            input_policy_sha256=canonical_sha256({"policy": "synthetic"}),
            runtime_environment_sha256="synthetic-runtime-environment",
            output_schema_registry_version="synthetic-output-schema",
        ),
        path,
    )
    return path


def _write_case_manifest(root: Path, patient_uids: tuple[str, ...]) -> Path:
    manifest_path = root.joinpath("input_case_manifest.csv")
    rows = []
    for patient_uid in patient_uids:
        core_paths = {
            role: root.joinpath("inputs", patient_uid, "{}.dcm".format(role))
            for role in ("rtstruct", "rtdose", "rtplan")
        }
        for core_path in core_paths.values():
            core_path.parent.mkdir(parents=True, exist_ok=True)
            core_path.touch()
        rows.append(
            {
                "Patient UID (generated)": patient_uid,
                "Patient Name": patient_uid,
                "Patient ID (from dicom)": patient_uid,
                "Fraction number (legacy parsed)": "1",
                "Has RTSTRUCT": "true",
                "Has RTDOSE": "true",
                "Has RTPLAN": "true",
                "Core RTSTRUCT/RTDOSE/RTPLAN complete": "true",
                "RTSTRUCT path": core_paths["rtstruct"],
                "RTDOSE path": core_paths["rtdose"],
                "RTPLAN path": core_paths["rtplan"],
                "Num US files": "0",
                "Num MR T2 files": "0",
                "Num MR ADC files": "0",
                "US paths": "",
                "MR T2 paths": "",
                "MR ADC paths": "",
            }
        )
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=_CASE_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


if __name__ == "__main__":
    unittest.main()

"""Disposable characterization of the unchanged legacy dose cohort wrapper.

A validation process streams one patient at a time through the actual wrapper.
A scoped delegating observer records helper arguments, return values and grid
products; it neither changes the helper nor reproduces its threshold recurrence.
No ordinary worker imports this module. This is not whole-legacy-main parity.
"""

from __future__ import annotations

from collections.abc import MutableMapping
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping, Any
from unittest.mock import patch

from config.snapshots import canonical_sha256
from input_data.content_identity import fingerprint_file
from patient_runner.resolved_state import dose_threshold_inputs
from validation.anatomical_independence import write_new_json


class _Progress:
    def add_task(self, *args, **kwargs):
        return 0

    def update(self, *args, **kwargs):
        pass


def capture_legacy_dose_order(
    *, patient_uids: Iterable[str], load_patient: Callable[[str], Mapping],
    config: Any, output_dir: Path, patient_metadata: Mapping[str, dict] | None = None,
    verify_after: Callable[[str], None] | None = None,
) -> dict:
    """Observe one real wrapper invocation with O(1)-patient scientific memory.

    ``load_patient`` transfers an exclusively owned fresh mutable patient mapping.
    The stream clears it after capture/verification, before loading the next
    patient, so even the wrapper's prior loop binding cannot retain its arrays.
    Only scalar state and artifact paths accumulate. Rendering is rejected. Evidence is
    raw native arrays (dose/gradient lattice and point XYZ in mm), never a cohort
    runtime pickle. Errors propagate and leave no success report.
    """
    import numpy as np
    from preprocessing import dose_grid_processing as legacy

    uids = tuple(patient_uids)
    if not uids or len(set(uids)) != len(uids):
        raise ValueError("legacy probe requires distinct explicit patient UIDs")
    if config.show_3d_dose_renderings or config.show_3d_dose_renderings_thresholded:
        raise ValueError("legacy probe requires rendering disabled")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    current = {}
    original = legacy.build_dose_grid_runtime_objects_for_patient

    def observe(patient, effective_config, *args):
        state = dose_threshold_inputs(patient, effective_config)
        effective = original(patient, effective_config, *args)
        state["effective_lower_bound"] = effective
        dose = patient[config.dose_ref]
        arrays = {
            "dose_gradient_grid": np.asarray(dose["Dose and gradient phys space and pixel 3d arr"]),
            "unthresholded_points": np.asarray(dose["Dose grid point cloud"].points),
            "thresholded_points": np.asarray(dose["Dose grid point cloud thresholded"].points),
        }
        if arrays["dose_gradient_grid"].ndim != 3 or arrays["dose_gradient_grid"].shape[-1] != 14:
            raise ValueError("legacy dose lattice does not have its established 14-column layout")
        for name in ("unthresholded_points", "thresholded_points"):
            if arrays[name].ndim != 2 or arrays[name].shape[-1] != 3:
                raise ValueError("legacy cloud must contain point XYZ")
        if any(array.dtype.kind not in "biuf" for array in arrays.values()):
            raise ValueError("legacy probe accepts real numeric arrays only")
        array_path = output_dir / f"patient_{current['index']:04d}.npz"
        with array_path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        current.update(state=state, arrays_file=array_path.name,
                       arrays_sha256=fingerprint_file(array_path)["sha256"],
                       array_contracts={name: {"dtype": a.dtype.str, "shape": list(a.shape)} for name, a in arrays.items()})
        return effective

    class StreamingPatients:
        def items(self):
            for index, uid in enumerate(uids, 1):
                current.clear()
                current.update(index=index, patient_uid=uid, metadata=dict((patient_metadata or {}).get(uid, {})))
                patient = load_patient(uid)
                if not isinstance(patient, MutableMapping):
                    raise TypeError("legacy probe loader must transfer an owned mutable mapping")
                current["state"] = dose_threshold_inputs(patient, config)
                yield uid, patient
                if verify_after is not None:
                    verify_after(uid)
                current["input_verification_after"] = verify_after is not None
                records.append(dict(current))
                # The wrapper still has a loop binding until next() returns.
                # Release this completed private mapping before constructing its successor.
                patient.clear()
                del patient

    progress = _Progress()
    with patch.object(legacy, "build_dose_grid_runtime_objects_for_patient", observe):
        result = legacy.build_dose_grids_for_cohort(
            StreamingPatients(), {"Global": {"Num cases": len(uids)}}, config, progress, progress, None,
        )
    report = {"schema_version": "legacy_dose_order_v1", "patient_order": list(uids),
              "initial_lower_bound": config.lower_bound_dose_value,
              "final_lower_bound": result.lower_bound_dose_value, "patients": records,
              "scope": "unchanged dose cohort wrapper; streaming singleton legacy input",
              "units": {"points": "DICOM patient mm", "grid": "indices; mm; native dose; dose/mm; dimensionless"}}
    report["report_sha256"] = canonical_sha256(report)
    write_new_json(output_dir / "legacy_dose_order.json", report)
    return report


def compare_legacy_dose_orders(reference_dir: Path, candidate_dir: Path, *, patient_uids: tuple[str, ...] | None = None) -> dict:
    """Compare recorded real values at exact 0/0; label unaffected-grid failures.

    Threshold-state or filtered-cloud changes characterize dependence. Any change
    in the full dose/gradient grid or unfiltered coordinates remains unexplained.
    This diagnosis never supplies a standalone numerical acceptance exemption.
    """
    import numpy as np
    from validation.anatomical_checkpoint import _compare_array

    loaded = []
    for root in map(Path, (reference_dir, candidate_dir)):
        report = json.loads((root / "legacy_dose_order.json").read_text())
        digest = report.pop("report_sha256")
        if report["schema_version"] != "legacy_dose_order_v1" or canonical_sha256(report) != digest:
            raise ValueError("invalid legacy probe report digest/schema")
        records = {p["patient_uid"]: p for p in report["patients"]}
        if len(records) != len(report["patients"]) or list(records) != report["patient_order"]:
            raise ValueError("legacy probe patient inventory mismatch")
        loaded.append((root, report, records))
    left_root, left_report, left = loaded[0]
    right_root, right_report, right = loaded[1]
    if patient_uids is not None:
        if not patient_uids or len(set(patient_uids)) != len(patient_uids) or not set(patient_uids) <= left.keys() or not set(patient_uids) <= right.keys():
            raise ValueError("legacy probe requested patients are absent or duplicated")
        left = {uid: left[uid] for uid in patient_uids}
        right = {uid: right[uid] for uid in patient_uids}
    if left.keys() != right.keys() or not left or left_report["initial_lower_bound"] != right_report["initial_lower_bound"]:
        raise ValueError("legacy probe patient sets or initial thresholds differ")
    result = {"schema_version": "legacy_dose_order_comparison_v1", "patients": [], "characterization_complete": True}
    for uid in left:
        a, b = left[uid], right[uid]
        if a["metadata"] != b["metadata"] or a["input_verification_after"] != b["input_verification_after"]:
            raise ValueError("legacy probe patient provenance differs")
        if a["state"]["dose_present"] != b["state"]["dose_present"]:
            raise ValueError("legacy probe dose presence differs")
        products = []
        if a["state"]["dose_present"]:
            arrays = []
            for root, record in ((left_root, a), (right_root, b)):
                path = root / record["arrays_file"]
                if path.resolve().parent != root.resolve() or path.is_symlink() or fingerprint_file(path)["sha256"] != record["arrays_sha256"]:
                    raise ValueError("invalid legacy probe array artifact")
                with np.load(path, allow_pickle=False) as archive:
                    values = {name: archive[name] for name in archive.files}
                if set(values) != {"dose_gradient_grid", "unthresholded_points", "thresholded_points"}:
                    raise ValueError("legacy probe array inventory differs")
                if {name: {"dtype": v.dtype.str, "shape": list(v.shape)} for name, v in values.items()} != record["array_contracts"]:
                    raise ValueError("legacy probe array contract differs")
                arrays.append(values)
            products = [{"field": name, **_compare_array(arrays[0][name], arrays[1][name], abs_tol=0, rel_tol=0, exact=True)} for name in arrays[0]]
        state_changed = a["state"].get("effective_lower_bound") != b["state"].get("effective_lower_bound")
        unexpected = any(not item["passed"] for item in products if item["field"] != "thresholded_points")
        changed = state_changed or any(not item["passed"] for item in products)
        classification = "unexplained_grid_difference" if unexpected else "dependence_observed" if changed else "not_observed_in_tested_inputs"
        result["patients"].append({"patient_uid": uid, "classification": classification,
                                   "helper_input_threshold_changed": a["state"]["configured_lower_bound"] != b["state"]["configured_lower_bound"],
                                   "reference_state": a["state"], "candidate_state": b["state"], "products": products})
        result["characterization_complete"] &= not unexpected
    return result


def run_legacy_dose_probe(qualification_plan: Path, output_dir: Path) -> dict:
    """User-operated disposable probe from the frozen qualification plan.

    Checks current source/environment/config and patient content before loading
    each singleton, then verifies content again after processing. Runs forward,
    reverse and singleton-reset controls. Historical artifacts are never written.
    """
    from config.rehydration import rehydrate_pipeline_scientific_config_snapshot
    from config.snapshots import read_pipeline_config_snapshot
    from input_data.content_identity import INPUT_CONTENT_KEY, verify_patient_input_content
    from patient_runner.process_runner import PatientWorkerJob, _validate_worker_compatibility_identity
    from validation.anatomical_execution import build_legacy_input_anatomical_runtime
    from preprocessing.dose_grid_processing import DoseGridProcessingConfig

    payload = json.loads(Path(qualification_plan).read_text())
    if payload.get("schema_version") != "anatomical_independence_plan_v1":
        raise ValueError("expected an anatomical qualification plan")
    jobs = [PatientWorkerJob.from_mapping(p) for p in payload["base_plan"]["worker_jobs"]]
    if not jobs or len({j.patient_case.patient_uid for j in jobs}) != len(jobs):
        raise ValueError("probe requires distinct planned patients")
    snapshots = []
    for job in jobs:
        snapshot = read_pipeline_config_snapshot(job.scientific_config_snapshot_path)
        if (snapshot.config_sha256 != job.metadata["scientific_config_snapshot_fingerprint_sha256"]
                or fingerprint_file(job.scientific_config_snapshot_path)["sha256"] != job.metadata["scientific_config_snapshot_file_sha256"]):
            raise ValueError("probe scientific snapshot differs from planned identity")
        _validate_worker_compatibility_identity(job, scientific_config_sha256=snapshot.config_sha256)
        snapshots.append(snapshot)
    if any(s.to_dict() != snapshots[0].to_dict() or j.metadata["run_compatibility_identity"] != jobs[0].metadata["run_compatibility_identity"] for s, j in zip(snapshots, jobs)):
        raise ValueError("probe jobs are incompatible")
    config = rehydrate_pipeline_scientific_config_snapshot(snapshots[0])
    dose_config = DoseGridProcessingConfig(
        config.legacy_refs.dose_ref, config.legacy_refs.plan_ref,
        config.replay.lower_bound_dose_value, config.replay.lower_bound_dose_gradient_value,
        config.grid_preprocessing.show_3d_dose_renderings, config.grid_preprocessing.show_3d_dose_renderings_thresholded,
    )
    jobs_by_uid = {job.patient_case.patient_uid: job for job in jobs}
    metadata = {uid: {"run_compatibility_identity": job.metadata["run_compatibility_identity"],
                      INPUT_CONTENT_KEY: job.metadata[INPUT_CONTENT_KEY]} for uid, job in jobs_by_uid.items()}

    def verify(uid):
        job = jobs_by_uid[uid]
        _validate_worker_compatibility_identity(job, scientific_config_sha256=snapshots[0].config_sha256)
        verify_patient_input_content(job.metadata[INPUT_CONTENT_KEY], job.patient_inputs)

    def load(uid):
        verify(uid)
        job = jobs_by_uid[uid]
        runtime = build_legacy_input_anatomical_runtime(patient_case=job.patient_case, patient_inputs=job.patient_inputs, pipeline_config=config)
        return runtime.runtime_state.pydicom_item

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": "legacy_dose_probe_v1", "characterization_complete": False}
    try:
        uids = tuple(jobs_by_uid)
        for name, order in (("forward", uids), ("reverse", uids[::-1])):
            capture_legacy_dose_order(patient_uids=order, load_patient=load, config=dose_config,
                                      output_dir=output_dir / name, patient_metadata=metadata, verify_after=verify)
        comparison = compare_legacy_dose_orders(output_dir / "forward", output_dir / "reverse")
        write_new_json(output_dir / "forward_vs_reverse.json", comparison)
        # Singleton controls establish the reset condition without a cohort store.
        singleton_records = []
        singleton_comparisons = []
        for index, uid in enumerate(uids, 1):
            singleton_dir = output_dir / "singletons" / f"{index:04d}"
            record = capture_legacy_dose_order(patient_uids=(uid,), load_patient=load, config=dose_config,
                output_dir=singleton_dir, patient_metadata=metadata, verify_after=verify)
            singleton_records.append({"patient_uid": uid, "state": record["patients"][0]["state"]})
            for order in ("forward", "reverse"):
                comparison_path = singleton_dir / (order + "_vs_singleton.json")
                check = compare_legacy_dose_orders(singleton_dir, output_dir / order, patient_uids=(uid,))
                write_new_json(comparison_path, check)
                singleton_comparisons.append(str(comparison_path))
                comparison["characterization_complete"] &= check["characterization_complete"]
        thresholds = [r["state"].get("effective_lower_bound") for r in singleton_records if r["state"]["dose_present"]]
        result.update(characterization_complete=comparison["characterization_complete"],
            comparison_path=str(output_dir / "forward_vs_reverse.json"), singleton_states=singleton_records,
            singleton_comparison_paths=singleton_comparisons,
            distinct_singleton_thresholds=len(set(thresholds)),
            sensitivity_note="Equal singleton thresholds cannot disprove the known carry mechanism; use unequal-prescription synthetic controls.")
    except Exception as exc:
        result["error"] = str(exc)
    write_new_json(output_dir / "legacy_dose_probe_summary.json", result)
    return result

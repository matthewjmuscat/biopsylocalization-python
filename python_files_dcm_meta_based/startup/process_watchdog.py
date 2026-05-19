from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Optional

import psutil

try:
    import GPUtil
except Exception:  # pragma: no cover - optional dependency surface
    GPUtil = None

_STOP_REQUESTED = False


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return repr(value)


def _handle_stop_signal(signum, frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _write_jsonl(log_file, payload: dict[str, Any]) -> None:
    log_file.write(json.dumps(_json_safe(payload), sort_keys=True) + "\n")
    log_file.flush()
    os.fsync(log_file.fileno())


def _read_status_snapshot(status_path: Optional[Path]) -> dict[str, Any]:
    if status_path is None or not status_path.exists():
        return {}
    try:
        status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status_read_error": repr(exc)}

    return {
        "runtime_status": status_payload.get("status"),
        "current_phase": status_payload.get("current_phase"),
        "current_patient_uid": status_payload.get("current_patient_uid"),
        "current_structure_id": status_payload.get("current_structure_id"),
        "current_structure_ref_type": status_payload.get("current_structure_ref_type"),
        "current_structure_index": status_payload.get("current_structure_index"),
        "last_completed_checkpoint": status_payload.get("last_completed_checkpoint"),
        "last_completed_checkpoint_utc": status_payload.get("last_completed_checkpoint_utc"),
        "run_status_last_update_utc": status_payload.get("last_update_utc"),
    }


def _collect_gpu_snapshot() -> dict[str, Any]:
    if GPUtil is None:
        return {}
    try:
        gpus = GPUtil.getGPUs()
    except Exception as exc:
        return {"gpu_snapshot_error": repr(exc)}
    if len(gpus) == 0:
        return {}
    return {
        "gpu_free_mb": round(float(gpus[0].memoryFree), 3),
        "gpu_used_mb": round(float(gpus[0].memoryUsed), 3),
        "gpu_total_mb": round(float(gpus[0].memoryTotal), 3),
    }


def _collect_target_snapshot(
    *,
    target_pid: int,
    expected_start_time: Optional[float],
) -> dict[str, Any]:
    try:
        process = psutil.Process(target_pid)
    except psutil.NoSuchProcess:
        return {"target_alive": False, "target_exit_reason": "pid_not_found"}
    except Exception as exc:
        return {"target_alive": False, "target_exit_reason": "pid_lookup_error", "error": repr(exc)}

    try:
        actual_start_time = float(process.create_time())
    except Exception:
        actual_start_time = None

    if (
        expected_start_time is not None
        and actual_start_time is not None
        and abs(actual_start_time - float(expected_start_time)) > 0.001
    ):
        return {
            "target_alive": False,
            "target_exit_reason": "pid_reused",
            "expected_process_start_time": expected_start_time,
            "observed_process_start_time": actual_start_time,
        }

    child_count = 0
    child_rss_bytes = 0
    child_vms_bytes = 0
    try:
        children = process.children(recursive=True)
    except Exception:
        children = []

    for child_process in children:
        try:
            child_memory_info = child_process.memory_info()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        child_count += 1
        child_rss_bytes += int(child_memory_info.rss)
        child_vms_bytes += int(child_memory_info.vms)

    try:
        with process.oneshot():
            memory_info = process.memory_info()
            status = process.status()
            thread_count = process.num_threads()
            cpu_percent = process.cpu_percent(interval=None)
    except psutil.NoSuchProcess:
        return {"target_alive": False, "target_exit_reason": "pid_disappeared_during_sample"}
    except Exception as exc:
        return {"target_alive": True, "target_sample_error": repr(exc)}

    virtual_memory = psutil.virtual_memory()
    swap_memory = psutil.swap_memory()
    rss_bytes = int(memory_info.rss)
    vms_bytes = int(memory_info.vms)

    return {
        "target_alive": True,
        "target_status": status,
        "process_start_time": actual_start_time,
        "rss_mb": round(rss_bytes / (1024 ** 2), 3),
        "vms_mb": round(vms_bytes / (1024 ** 2), 3),
        "child_process_count": int(child_count),
        "child_rss_mb": round(child_rss_bytes / (1024 ** 2), 3),
        "child_vms_mb": round(child_vms_bytes / (1024 ** 2), 3),
        "tree_rss_mb": round((rss_bytes + child_rss_bytes) / (1024 ** 2), 3),
        "tree_vms_mb": round((vms_bytes + child_vms_bytes) / (1024 ** 2), 3),
        "num_threads": int(thread_count),
        "cpu_percent": float(cpu_percent),
        "system_available_ram_mb": round(virtual_memory.available / (1024 ** 2), 3),
        "system_used_ram_mb": round(virtual_memory.used / (1024 ** 2), 3),
        "system_ram_percent": float(virtual_memory.percent),
        "swap_used_mb": round(swap_memory.used / (1024 ** 2), 3),
        "swap_percent": float(swap_memory.percent),
    }


def _build_event(
    *,
    event_type: str,
    target_pid: int,
    started_monotonic: float,
    sample_index: int,
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp_utc": _utc_now_iso(),
        "elapsed_sec": round(time.monotonic() - started_monotonic, 3),
        "event_type": event_type,
        "watchdog_pid": os.getpid(),
        "target_pid": int(target_pid),
        "sample_index": int(sample_index),
        "details": details,
    }


def run_watchdog(
    *,
    target_pid: int,
    log_path: Path,
    status_path: Optional[Path],
    interval_sec: float,
    expected_start_time: Optional[float],
) -> int:
    signal.signal(signal.SIGTERM, _handle_stop_signal)
    signal.signal(signal.SIGINT, _handle_stop_signal)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_monotonic = time.monotonic()
    sample_index = 0

    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        _write_jsonl(
            log_file,
            _build_event(
                event_type="watchdog_start",
                target_pid=target_pid,
                started_monotonic=started_monotonic,
                sample_index=sample_index,
                details={
                    "interval_sec": float(interval_sec),
                    "expected_process_start_time": expected_start_time,
                    "status_path": None if status_path is None else str(status_path),
                },
            ),
        )

        while not _STOP_REQUESTED:
            sample_index += 1
            target_snapshot = _collect_target_snapshot(
                target_pid=target_pid,
                expected_start_time=expected_start_time,
            )
            status_snapshot = _read_status_snapshot(status_path)
            details = {}
            details.update(target_snapshot)
            details.update(_collect_gpu_snapshot())
            details.update(status_snapshot)
            _write_jsonl(
                log_file,
                _build_event(
                    event_type="watchdog_sample",
                    target_pid=target_pid,
                    started_monotonic=started_monotonic,
                    sample_index=sample_index,
                    details=details,
                ),
            )
            if not bool(target_snapshot.get("target_alive")):
                _write_jsonl(
                    log_file,
                    _build_event(
                        event_type="target_disappeared",
                        target_pid=target_pid,
                        started_monotonic=started_monotonic,
                        sample_index=sample_index,
                        details=details,
                    ),
                )
                return 0

            sleep_until = time.monotonic() + float(interval_sec)
            while not _STOP_REQUESTED and time.monotonic() < sleep_until:
                time.sleep(min(0.25, sleep_until - time.monotonic()))

        _write_jsonl(
            log_file,
            _build_event(
                event_type="watchdog_stop_requested",
                target_pid=target_pid,
                started_monotonic=started_monotonic,
                sample_index=sample_index,
                details=_read_status_snapshot(status_path),
            ),
        )
    return 0


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a running pipeline process from a sibling process.")
    parser.add_argument("--pid", type=int, required=True, help="PID of the process to monitor.")
    parser.add_argument("--log-path", type=Path, required=True, help="JSONL file to append samples to.")
    parser.add_argument("--status-path", type=Path, default=None, help="Optional runtime run_status.json path.")
    parser.add_argument("--interval-sec", type=float, default=5.0, help="Sampling interval in seconds.")
    parser.add_argument(
        "--parent-start-time",
        type=float,
        default=None,
        help="Expected psutil create_time for PID-reuse detection.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    return run_watchdog(
        target_pid=int(args.pid),
        log_path=Path(args.log_path),
        status_path=None if args.status_path is None else Path(args.status_path),
        interval_sec=float(args.interval_sec),
        expected_start_time=args.parent_start_time,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

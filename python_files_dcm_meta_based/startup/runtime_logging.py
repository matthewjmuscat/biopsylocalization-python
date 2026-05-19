from __future__ import annotations

import atexit
from datetime import datetime, timezone
import faulthandler
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any
from typing import Mapping
from typing import Optional
from typing import Sequence
import uuid

import psutil

try:
    import GPUtil
except Exception:  # pragma: no cover - optional dependency surface
    GPUtil = None


_ACTIVE_RUNTIME_LOGGER: Optional["RuntimeLogger"] = None
_PREVIOUS_EXCEPTHOOK = sys.excepthook


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return repr(value)


class RuntimeLogger:
    def __init__(
        self,
        base_output_dir: Path,
        *,
        run_id: Optional[str] = None,
        argv: Optional[Sequence[str]] = None,
    ):
        self.base_output_dir = Path(base_output_dir)
        self.run_id = str(run_id or self._build_default_run_id())
        self.argv = list(sys.argv if argv is None else argv)
        self.pid = os.getpid()
        self.started_utc = _utc_now_iso()
        self._started_monotonic = time.monotonic()
        self._current_phase: Optional[str] = None
        self._current_patient_uid: Optional[str] = None
        self._current_structure_id: Optional[str] = None
        self._current_structure_ref_type: Optional[str] = None
        self._current_structure_index: Optional[int] = None
        self._last_completed_checkpoint: Optional[str] = None
        self._last_completed_checkpoint_utc: Optional[str] = None
        self._status = "running"
        self._finalized = False
        self._specific_output_dir: Optional[Path] = None

        self._staging_logs_dir = self.base_output_dir.joinpath("_runtime_logs", self.run_id)
        self._logs_dir = self._staging_logs_dir
        self._logs_dir.mkdir(parents=True, exist_ok=True)

        self._run_log_path = self._logs_dir.joinpath("run.log")
        self._events_path = self._logs_dir.joinpath("events.jsonl")
        self._status_path = self._logs_dir.joinpath("run_status.json")
        self._run_log_file = self._run_log_path.open("a", encoding="utf-8", buffering=1)
        self._events_file = self._events_path.open("a", encoding="utf-8", buffering=1)
        self._native_fault_log_file = None
        self._native_fault_log_path = None
        self._enable_native_fault_logging()

        self._write_status(force_fsync=True)
        self.log_event(
            level="INFO",
            event_type="run_start",
            phase="run_start",
            message="Initialized runtime logger.",
            details={
                "base_output_dir": str(self.base_output_dir),
                "logs_dir": str(self._logs_dir),
                "native_fault_log_path": None
                if self._native_fault_log_path is None
                else str(self._native_fault_log_path),
                "argv": self.argv,
            },
            write_status=True,
            force_fsync=True,
        )

    def _build_default_run_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        return "runtime-{}-pid{}-{}".format(timestamp, os.getpid(), uuid.uuid4().hex[:8])

    @property
    def logs_dir(self) -> Path:
        return self._logs_dir

    @property
    def current_phase(self) -> Optional[str]:
        return self._current_phase

    @property
    def is_finalized(self) -> bool:
        return self._finalized

    def attach_output_dir(self, specific_output_dir: Path) -> Path:
        specific_output_dir = Path(specific_output_dir)
        final_logs_dir = specific_output_dir.joinpath("logs")
        if self._logs_dir == final_logs_dir:
            return final_logs_dir

        final_logs_dir.mkdir(parents=True, exist_ok=True)
        self._flush_files(force_fsync=True)

        new_run_log_path = final_logs_dir.joinpath("run.log")
        new_events_path = final_logs_dir.joinpath("events.jsonl")
        new_status_path = final_logs_dir.joinpath("run_status.json")
        new_native_fault_log_path = final_logs_dir.joinpath("native_fault.log")

        if self._run_log_path.exists() and not new_run_log_path.exists():
            shutil.copyfile(self._run_log_path, new_run_log_path)
        if self._events_path.exists() and not new_events_path.exists():
            shutil.copyfile(self._events_path, new_events_path)
        if self._status_path.exists():
            shutil.copyfile(self._status_path, new_status_path)
        if self._native_fault_log_path is not None and self._native_fault_log_path.exists():
            if self._native_fault_log_file is not None:
                self._native_fault_log_file.flush()
            shutil.copyfile(self._native_fault_log_path, new_native_fault_log_path)

        old_run_log_file = self._run_log_file
        old_events_file = self._events_file
        self._run_log_file = new_run_log_path.open("a", encoding="utf-8", buffering=1)
        self._events_file = new_events_path.open("a", encoding="utf-8", buffering=1)

        old_run_log_file.close()
        old_events_file.close()

        self._logs_dir = final_logs_dir
        self._run_log_path = new_run_log_path
        self._events_path = new_events_path
        self._status_path = new_status_path
        self._specific_output_dir = specific_output_dir
        self._enable_native_fault_logging()

        self.log_event(
            level="INFO",
            event_type="checkpoint",
            phase="run_output_dir.attach",
            message="Attached runtime logger to the specific output directory.",
            details={
                "specific_output_dir": str(specific_output_dir),
                "logs_dir": str(final_logs_dir),
                "staging_logs_dir": str(self._staging_logs_dir),
            },
            write_status=True,
            force_fsync=True,
        )
        return final_logs_dir

    def info(self, phase: Optional[str], message: str, **context: Any) -> None:
        self.log_event(
            level="INFO",
            event_type="checkpoint",
            phase=phase,
            message=message,
            **context,
        )

    def warning(self, phase: Optional[str], message: str, **context: Any) -> None:
        self.log_event(
            level="WARN",
            event_type="warning",
            phase=phase,
            message=message,
            write_status=True,
            force_fsync=True,
            **context,
        )

    def error(self, phase: Optional[str], message: str, **context: Any) -> None:
        self.log_event(
            level="ERROR",
            event_type="error",
            phase=phase,
            message=message,
            write_status=True,
            force_fsync=True,
            **context,
        )

    def checkpoint(
        self,
        phase: str,
        message: str,
        *,
        patient_uid: Optional[str] = None,
        structure_id: Optional[str] = None,
        structure_ref_type: Optional[str] = None,
        structure_index: Optional[int] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._last_completed_checkpoint = str(phase)
        self._last_completed_checkpoint_utc = _utc_now_iso()
        self.log_event(
            level="INFO",
            event_type="checkpoint",
            phase=phase,
            message=message,
            patient_uid=patient_uid,
            structure_id=structure_id,
            structure_ref_type=structure_ref_type,
            structure_index=structure_index,
            details=details,
            write_status=True,
            force_fsync=True,
        )

    def phase_start(
        self,
        phase: str,
        message: str,
        *,
        patient_uid: Optional[str] = None,
        structure_id: Optional[str] = None,
        structure_ref_type: Optional[str] = None,
        structure_index: Optional[int] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._set_context(
            phase=phase,
            patient_uid=patient_uid,
            structure_id=structure_id,
            structure_ref_type=structure_ref_type,
            structure_index=structure_index,
        )
        self.log_event(
            level="INFO",
            event_type="phase_start",
            phase=phase,
            message=message,
            patient_uid=patient_uid,
            structure_id=structure_id,
            structure_ref_type=structure_ref_type,
            structure_index=structure_index,
            details=details,
            write_status=True,
            force_fsync=True,
        )

    def phase_end(
        self,
        phase: str,
        message: str,
        *,
        patient_uid: Optional[str] = None,
        structure_id: Optional[str] = None,
        structure_ref_type: Optional[str] = None,
        structure_index: Optional[int] = None,
        details: Optional[Mapping[str, Any]] = None,
        clear_phase: bool = False,
    ) -> None:
        self._last_completed_checkpoint = str(phase)
        self._last_completed_checkpoint_utc = _utc_now_iso()
        self.log_event(
            level="INFO",
            event_type="phase_end",
            phase=phase,
            message=message,
            patient_uid=patient_uid,
            structure_id=structure_id,
            structure_ref_type=structure_ref_type,
            structure_index=structure_index,
            details=details,
            write_status=True,
            force_fsync=True,
        )
        if clear_phase:
            self._current_phase = None

    def memory_snapshot(
        self,
        phase: Optional[str],
        message: str,
        *,
        level: str = "INFO",
        patient_uid: Optional[str] = None,
        structure_id: Optional[str] = None,
        structure_ref_type: Optional[str] = None,
        structure_index: Optional[int] = None,
        details: Optional[Mapping[str, Any]] = None,
        force_fsync: bool = True,
    ) -> None:
        snapshot = self._collect_memory_snapshot()
        merged_details = dict(snapshot)
        if details is not None:
            merged_details.update(dict(details))
        self.log_event(
            level=level,
            event_type="memory_snapshot",
            phase=phase,
            message=message,
            patient_uid=patient_uid,
            structure_id=structure_id,
            structure_ref_type=structure_ref_type,
            structure_index=structure_index,
            details=merged_details,
            write_status=True,
            force_fsync=force_fsync,
        )

    def log_exception(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback,
        *,
        phase: Optional[str] = None,
    ) -> None:
        traceback_text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        self._status = "failed"
        self.log_event(
            level="ERROR",
            event_type="exception",
            phase=phase,
            message="Unhandled exception: {}".format(exc_value),
            details={
                "exception_type": getattr(exc_type, "__name__", str(exc_type)),
                "traceback": traceback_text,
            },
            write_status=True,
            force_fsync=True,
        )

    def mark_aborted(self, message: str, *, phase: Optional[str] = None) -> None:
        if self._finalized:
            return
        self._status = "aborted"
        self.log_event(
            level="WARN",
            event_type="run_abort",
            phase=phase,
            message=message,
            write_status=True,
            force_fsync=True,
        )
        self.close()

    def mark_completed(self, message: str = "Run completed successfully.") -> None:
        if self._finalized:
            return
        self._status = "completed"
        self._last_completed_checkpoint = "run_finish"
        self._last_completed_checkpoint_utc = _utc_now_iso()
        self.log_event(
            level="INFO",
            event_type="run_finish",
            phase="run_finish",
            message=message,
            write_status=True,
            force_fsync=True,
        )
        self.close()

    def close(self) -> None:
        if self._finalized:
            return
        self._disable_native_fault_logging()
        self._flush_files(force_fsync=True)
        self._run_log_file.close()
        self._events_file.close()
        self._finalized = True

    def log_event(
        self,
        *,
        level: str,
        event_type: str,
        phase: Optional[str],
        message: str,
        patient_uid: Optional[str] = None,
        structure_id: Optional[str] = None,
        structure_ref_type: Optional[str] = None,
        structure_index: Optional[int] = None,
        details: Optional[Mapping[str, Any]] = None,
        write_status: bool = False,
        force_fsync: bool = False,
    ) -> None:
        timestamp_utc = _utc_now_iso()
        elapsed_sec = round(time.monotonic() - self._started_monotonic, 3)
        resolved_phase = str(phase or self._current_phase or "unknown")
        resolved_patient_uid = patient_uid if patient_uid is not None else self._current_patient_uid
        resolved_structure_id = structure_id if structure_id is not None else self._current_structure_id
        resolved_structure_ref_type = (
            structure_ref_type if structure_ref_type is not None else self._current_structure_ref_type
        )
        resolved_structure_index = (
            structure_index if structure_index is not None else self._current_structure_index
        )

        event_payload = {
            "timestamp_utc": timestamp_utc,
            "elapsed_sec": elapsed_sec,
            "level": str(level).upper(),
            "event_type": str(event_type),
            "phase": resolved_phase,
            "run_id": self.run_id,
            "pid": self.pid,
            "patient_uid": _json_safe(resolved_patient_uid),
            "structure_id": _json_safe(resolved_structure_id),
            "structure_ref_type": _json_safe(resolved_structure_ref_type),
            "structure_index": _json_safe(resolved_structure_index),
            "message": str(message),
            "details": _json_safe(details or {}),
        }

        context_parts = []
        if resolved_patient_uid is not None:
            context_parts.append("patient={}".format(resolved_patient_uid))
        if resolved_structure_id is not None:
            context_parts.append("structure={}".format(resolved_structure_id))
        if resolved_structure_ref_type is not None:
            context_parts.append("struct_ref={}".format(resolved_structure_ref_type))
        if resolved_structure_index is not None:
            context_parts.append("struct_index={}".format(resolved_structure_index))
        for detail_key, detail_value in (details or {}).items():
            scalar_detail = _json_safe(detail_value)
            if isinstance(scalar_detail, (bool, int, float, str)) or scalar_detail is None:
                context_parts.append("{}={}".format(detail_key, scalar_detail))

        log_line = (
            "{} | +{:08.3f}s | {:<5} | {}".format(
                timestamp_utc,
                elapsed_sec,
                str(level).upper(),
                resolved_phase,
            )
        )
        if len(context_parts) > 0:
            log_line = "{} | {}".format(log_line, " | ".join(context_parts))
        log_line = "{} | {}\n".format(log_line, str(message))

        self._run_log_file.write(log_line)
        self._events_file.write(json.dumps(event_payload, sort_keys=True) + "\n")
        self._flush_files(force_fsync=force_fsync)
        if write_status:
            self._write_status(force_fsync=force_fsync)

    def _set_context(
        self,
        *,
        phase: Optional[str] = None,
        patient_uid: Optional[str] = None,
        structure_id: Optional[str] = None,
        structure_ref_type: Optional[str] = None,
        structure_index: Optional[int] = None,
    ) -> None:
        if phase is not None:
            self._current_phase = str(phase)
        if patient_uid is not None:
            self._current_patient_uid = str(patient_uid)
        if structure_id is not None:
            self._current_structure_id = str(structure_id)
        if structure_ref_type is not None:
            self._current_structure_ref_type = str(structure_ref_type)
        if structure_index is not None:
            self._current_structure_index = int(structure_index)

    def _collect_memory_snapshot(self) -> dict[str, Any]:
        process = psutil.Process(self.pid)
        memory_info = process.memory_info()
        virtual_memory = psutil.virtual_memory()

        snapshot = {
            "rss_mb": round(memory_info.rss / (1024 ** 2), 3),
            "vms_mb": round(memory_info.vms / (1024 ** 2), 3),
            "system_available_ram_mb": round(virtual_memory.available / (1024 ** 2), 3),
        }
        if GPUtil is not None:
            try:
                gpus = GPUtil.getGPUs()
            except Exception:
                gpus = []
            if len(gpus) > 0:
                snapshot.update(
                    {
                        "gpu_free_mb": round(float(gpus[0].memoryFree), 3),
                        "gpu_used_mb": round(float(gpus[0].memoryUsed), 3),
                        "gpu_total_mb": round(float(gpus[0].memoryTotal), 3),
                    }
                )
        return snapshot

    def _build_status_payload(self) -> dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "status": self._status,
            "started_utc": self.started_utc,
            "last_update_utc": _utc_now_iso(),
            "current_phase": self._current_phase,
            "current_patient_uid": self._current_patient_uid,
            "current_structure_id": self._current_structure_id,
            "current_structure_ref_type": self._current_structure_ref_type,
            "current_structure_index": self._current_structure_index,
            "last_completed_checkpoint": self._last_completed_checkpoint,
            "last_completed_checkpoint_utc": self._last_completed_checkpoint_utc,
            "output_dir": None if self._specific_output_dir is None else str(self._specific_output_dir),
            "logs_dir": str(self._logs_dir),
            "native_fault_log_path": None
            if self._native_fault_log_path is None
            else str(self._native_fault_log_path),
            "argv": self.argv,
        }
        payload.update(self._collect_memory_snapshot())
        return payload

    def _enable_native_fault_logging(self) -> None:
        native_fault_log_path = self._logs_dir.joinpath("native_fault.log")
        native_fault_log_file = native_fault_log_path.open("a", encoding="utf-8", buffering=1)
        previous_native_fault_log_file = self._native_fault_log_file
        try:
            faulthandler.enable(file=native_fault_log_file, all_threads=True)
        except Exception:
            native_fault_log_file.close()
            return

        self._native_fault_log_file = native_fault_log_file
        self._native_fault_log_path = native_fault_log_path
        if previous_native_fault_log_file is not None:
            try:
                previous_native_fault_log_file.close()
            except Exception:
                pass

    def _disable_native_fault_logging(self) -> None:
        native_fault_log_file = self._native_fault_log_file
        self._native_fault_log_file = None
        self._native_fault_log_path = None
        if native_fault_log_file is None:
            return
        try:
            faulthandler.disable()
        except Exception:
            pass
        try:
            native_fault_log_file.close()
        except Exception:
            pass

    def _write_status(self, *, force_fsync: bool = False) -> None:
        status_payload = _json_safe(self._build_status_payload())
        with self._status_path.open("w", encoding="utf-8") as status_file:
            json.dump(status_payload, status_file, indent=2, sort_keys=True)
            status_file.write("\n")
            status_file.flush()
            if force_fsync:
                os.fsync(status_file.fileno())

    def _flush_files(self, *, force_fsync: bool = False) -> None:
        self._run_log_file.flush()
        self._events_file.flush()
        if force_fsync:
            os.fsync(self._run_log_file.fileno())
            os.fsync(self._events_file.fileno())


def install_runtime_logger(runtime_logger: RuntimeLogger) -> RuntimeLogger:
    global _ACTIVE_RUNTIME_LOGGER
    _ACTIVE_RUNTIME_LOGGER = runtime_logger
    sys.excepthook = _runtime_logging_excepthook
    return runtime_logger


def get_active_runtime_logger() -> Optional[RuntimeLogger]:
    return _ACTIVE_RUNTIME_LOGGER


def _runtime_logging_excepthook(exc_type, exc_value, exc_traceback):
    runtime_logger = _ACTIVE_RUNTIME_LOGGER
    if runtime_logger is not None and not runtime_logger.is_finalized:
        try:
            runtime_logger.log_exception(exc_type, exc_value, exc_traceback)
            runtime_logger.close()
        except Exception:
            pass
    _PREVIOUS_EXCEPTHOOK(exc_type, exc_value, exc_traceback)


def _runtime_logging_atexit_handler() -> None:
    runtime_logger = _ACTIVE_RUNTIME_LOGGER
    if runtime_logger is None or runtime_logger.is_finalized:
        return
    try:
        runtime_logger.mark_aborted(
            "Process exited without an explicit runtime completion marker.",
            phase=runtime_logger.current_phase,
        )
    except Exception:
        pass


atexit.register(_runtime_logging_atexit_handler)
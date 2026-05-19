from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import Any, Iterable, Optional


DEFAULT_OUTPUT_ROOT = Path.home().joinpath("Documents", "UBC", "Research", "Data", "Output data")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"read_error": repr(exc)}


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    yield {"json_parse_error": line[:200]}
    except FileNotFoundError:
        return


def _tail_jsonl(path: Path, count: int = 5) -> list[dict[str, Any]]:
    events = list(_iter_jsonl(path))
    if count <= 0:
        return events
    return events[-count:]


def _max_detail_value(path: Path, detail_key: str) -> Optional[tuple[float, dict[str, Any]]]:
    best_value = None
    best_event = None
    for event in _iter_jsonl(path):
        details = event.get("details") or {}
        value = details.get(detail_key)
        if value is None:
            continue
        try:
            numeric_value = float(value)
        except Exception:
            continue
        if best_value is None or numeric_value > best_value:
            best_value = numeric_value
            best_event = event
    if best_value is None or best_event is None:
        return None
    return best_value, best_event


def _resolve_logs_dir(path: Path) -> Path:
    path = Path(path)
    if path.name == "logs" and path.is_dir():
        return path
    if path.joinpath("logs").is_dir():
        return path.joinpath("logs")
    if path.joinpath("run_status.json").exists():
        return path
    raise FileNotFoundError("Could not resolve logs directory from {}".format(path))


def _latest_run_dir(output_root: Path) -> Path:
    candidates = [
        path for path in output_root.iterdir()
        if path.is_dir() and (path.name.startswith("MC_sim_out-") or path.name == "_runtime_logs")
    ]
    if not candidates:
        raise FileNotFoundError("No runtime output directories found under {}".format(output_root))
    newest_root = max(candidates, key=lambda path: path.stat().st_mtime)
    if newest_root.name != "_runtime_logs":
        return newest_root

    staging_candidates = [path for path in newest_root.iterdir() if path.is_dir()]
    if not staging_candidates:
        return newest_root
    return max(staging_candidates, key=lambda path: path.stat().st_mtime)


def _format_context(event: dict[str, Any]) -> str:
    details = event.get("details") or {}
    parts = []
    for key in ("phase", "patient_uid", "structure_id"):
        if event.get(key) is not None:
            parts.append("{}={}".format(key, event.get(key)))
    for key in (
        "runtime_status",
        "current_phase",
        "current_patient_uid",
        "current_structure_id",
        "last_completed_checkpoint",
        "target_alive",
        "target_exit_reason",
        "rss_mb",
        "tree_rss_mb",
        "system_available_ram_mb",
        "gpu_used_mb",
    ):
        if key in details:
            parts.append("{}={}".format(key, details[key]))
    if parts:
        return "; ".join(parts)
    return "no context"


def summarize_logs(logs_dir: Path, *, tail_count: int = 5) -> str:
    logs_dir = _resolve_logs_dir(logs_dir)
    lines = []
    lines.append("Runtime log summary")
    lines.append("logs_dir: {}".format(logs_dir))

    status_path = logs_dir.joinpath("run_status.json")
    if status_path.exists():
        status = _read_json(status_path)
        lines.append("status: {}".format(status.get("status")))
        lines.append("phase: {}".format(status.get("current_phase")))
        lines.append("patient: {}".format(status.get("current_patient_uid")))
        lines.append("structure: {}".format(status.get("current_structure_id")))
        lines.append("last_checkpoint: {}".format(status.get("last_completed_checkpoint")))
        lines.append("last_update_utc: {}".format(status.get("last_update_utc")))
        for key in ("rss_mb", "system_available_ram_mb", "gpu_used_mb", "gpu_free_mb"):
            if key in status:
                lines.append("{}: {}".format(key, status.get(key)))
    else:
        lines.append("status: missing run_status.json")

    native_fault_path = logs_dir.joinpath("native_fault.log")
    if native_fault_path.exists():
        lines.append("native_fault_bytes: {}".format(native_fault_path.stat().st_size))
    else:
        lines.append("native_fault_bytes: missing")

    for file_name in ("events.jsonl", "process_watchdog.jsonl"):
        path = logs_dir.joinpath(file_name)
        if not path.exists():
            lines.append("{}: missing".format(file_name))
            continue
        events = list(_iter_jsonl(path))
        lines.append("{}: {} lines, {} bytes".format(file_name, len(events), path.stat().st_size))
        for key in ("rss_mb", "tree_rss_mb", "system_available_ram_mb", "gpu_used_mb"):
            max_value = _max_detail_value(path, key)
            if max_value is None:
                continue
            value, event = max_value
            lines.append("  max_{}: {} at {} {}".format(
                key,
                round(value, 3),
                event.get("timestamp_utc"),
                event.get("phase") or event.get("event_type"),
            ))
        tail = events[-tail_count:] if tail_count > 0 else []
        for event in tail:
            lines.append("  tail: {} {} {}".format(
                event.get("timestamp_utc"),
                event.get("event_type"),
                _format_context(event),
            ))

    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize runtime logs for a biopsy-localization run.")
    parser.add_argument("path", nargs="?", type=Path, help="Run directory or logs directory to summarize.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root used when no path is provided.",
    )
    parser.add_argument("--tail", type=int, default=5, help="Number of tail events to print per JSONL file.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    target_path = args.path if args.path is not None else _latest_run_dir(args.output_root)
    print(summarize_logs(target_path, tail_count=int(args.tail)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

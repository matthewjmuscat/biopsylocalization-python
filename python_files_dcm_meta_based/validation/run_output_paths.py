from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VALIDATION_OUTPUT_ROOT = REPO_ROOT / "validation_outputs"


def resolve_existing_output_dir(requested_path: str | Path) -> Path:
    requested_path = Path(requested_path).expanduser()
    if requested_path.is_dir():
        return requested_path

    parent = requested_path.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"Parent directory does not exist: {parent}")

    candidates = sorted(
        path
        for path in parent.glob(f"{requested_path.name}*")
        if path.is_dir()
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not resolve output directory '{requested_path}'. No matching folder prefix was found."
        )
    if len(candidates) > 1:
        raise FileNotFoundError(
            "Ambiguous output directory prefix "
            f"'{requested_path}'. Matching candidates: {[str(path) for path in candidates]}"
        )
    return candidates[0]


def cohort_csv_dir(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    cohort_dir = output_dir / "Output CSVs" / "Cohort"
    if not cohort_dir.is_dir():
        raise FileNotFoundError(f"Missing cohort CSV directory: {cohort_dir}")
    return cohort_dir


def discover_cohort_csvs(output_dir: str | Path) -> dict[str, Path]:
    cohort_dir = cohort_csv_dir(output_dir)
    return {
        str(path.relative_to(cohort_dir)): path
        for path in sorted(cohort_dir.rglob("*.csv"))
    }
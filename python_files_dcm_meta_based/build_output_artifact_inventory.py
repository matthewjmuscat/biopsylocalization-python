from __future__ import annotations

import argparse
from pathlib import Path

from output_artifacts import build_output_artifact_inventory
from output_artifacts import summarize_output_artifact_inventory
from output_artifacts import write_output_artifact_inventory
from validation import resolve_existing_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reusable Phase 2 inventory of output artifacts for a completed run.",
    )
    parser.add_argument("run_dir", help="Completed run output directory or a run name under the default Output data root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where inventory CSV and summary JSON should be written.",
    )
    parser.add_argument(
        "--include-other-files",
        action="store_true",
        help="Include files outside the known table/figure/log/manifest extensions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_existing_output_dir(args.run_dir)
    if args.output_dir is None:
        safe_run_name = run_dir.name.replace("/", "_").replace(" ", "_").replace(":", "_").replace(",", "_")
        output_dir = Path("validation_outputs").joinpath("output_inventory", safe_run_name)
    else:
        output_dir = args.output_dir

    inventory_df = build_output_artifact_inventory(
        run_dir,
        include_other_files=args.include_other_files,
    )
    inventory_path, summary_path = write_output_artifact_inventory(inventory_df, output_dir)
    summary = summarize_output_artifact_inventory(inventory_df)
    print(f"[inventory] wrote {inventory_path}")
    print(f"[inventory] wrote {summary_path}")
    print(
        "[inventory] artifacts={artifact_count} | tables={table_count} | classes={output_class_counts}".format(
            **summary,
        )
    )


if __name__ == "__main__":
    main()
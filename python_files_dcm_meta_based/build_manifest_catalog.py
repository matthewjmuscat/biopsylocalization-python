from __future__ import annotations

import argparse
import json
from pathlib import Path

from output_artifacts.manifest_catalog import summarize_manifest_catalog
from output_artifacts.manifest_catalog import write_manifest_catalog
from output_artifacts.manifest_catalog import write_manifest_presence_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a CSV, JSON summary, and Markdown report for known manifest contracts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation_outputs/manifest_catalog"),
        help="Directory where manifest catalog reports should be written.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional completed run directory to inspect for cataloged manifest presence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog_path, summary_path, markdown_path = write_manifest_catalog(args.output_dir)
    summary = summarize_manifest_catalog()
    print(f"[manifest-catalog] wrote {catalog_path}")
    print(f"[manifest-catalog] wrote {summary_path}")
    print(f"[manifest-catalog] wrote {markdown_path}")
    print(
        "[manifest-catalog] contracts={manifest_contract_count} | scopes={scope_counts} | statuses={lifecycle_status_counts}".format(
            **summary,
        )
    )
    if args.run_dir is not None:
        presence_path, presence_summary_path = write_manifest_presence_report(args.run_dir, args.output_dir)
        with presence_summary_path.open("r", encoding="utf-8") as file_obj:
            presence_summary = json.load(file_obj)
        print(f"[manifest-presence] wrote {presence_path}")
        print(f"[manifest-presence] wrote {presence_summary_path}")
        print(
            "[manifest-presence] found={found_manifest_contract_count}/{manifest_contract_count} | statuses={presence_status_counts}".format(
                **presence_summary,
            )
        )


if __name__ == "__main__":
    main()
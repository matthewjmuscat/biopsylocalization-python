from __future__ import annotations

import argparse
from pathlib import Path

from output_artifacts.manifest_catalog import summarize_manifest_catalog
from output_artifacts.manifest_catalog import write_manifest_catalog


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


if __name__ == "__main__":
    main()
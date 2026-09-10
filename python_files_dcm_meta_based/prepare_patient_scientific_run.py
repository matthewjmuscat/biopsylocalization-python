"""CPU-safe CLI for preparing new provenance from a verified scientific snapshot.

The startup service owns validation and provenance writes. This entrypoint does
not import legacy main, accept patient data, or plan/execute scientific work.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from startup.standalone_preparation import DEFAULT_REPOSITORY_PATH
from startup.standalone_preparation import prepare_patient_scientific_run


def main(argv: Sequence[str] | None = None) -> int:
    """Parse explicit preparation inputs and print the new provenance paths.

    Writes only through the startup service. Help needs no GPU or GUI imports.
    Argument errors exit with status 2; preparation failures exit with status 1
    and a concise diagnostic. Successful preparation returns 0, not a run result.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Prepare NEW run provenance from an existing verified scientific config snapshot. "
            "Reuse resolved science while capturing current source/environment identity; "
            "do not execute patient jobs or rewrite historical provenance."
        ),
    )
    parser.add_argument(
        "--scientific-config-snapshot",
        type=Path,
        required=True,
        help="Existing resolved scientific config snapshot to verify and rehydrate.",
    )
    parser.add_argument(
        "--routing-profile",
        type=Path,
        required=True,
        help="Explicit routing profile JSON: shared input policy, NOT patient data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Absent or empty NEW provenance directory, outside the source snapshot directory.",
    )
    parser.add_argument(
        "--repository-path",
        type=Path,
        default=DEFAULT_REPOSITORY_PATH,
        help="Execution repository/build path (default: repository containing this script).",
    )
    args = parser.parse_args(argv)
    try:
        result = prepare_patient_scientific_run(
            scientific_config_snapshot_path=args.scientific_config_snapshot,
            routing_profile_path=args.routing_profile,
            output_dir=args.output_dir,
            repository_path=args.repository_path,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(1, "preparation failed: {}\n".format(exc))
    print(json.dumps({
        "preparation_record_path": result.preparation_record_path.as_posix(),
        **result.provenance.manifest_metadata(),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
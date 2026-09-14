"""Disposable legacy dose-state characterization; never a production worker."""

import argparse
from pathlib import Path


def main(argv=None) -> int:
    """Execute only the unchanged dose wrapper with explicit frozen patient jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--qualification-plan", type=Path)
    source.add_argument("--process-plan", type=Path,
                        help="Fresh anatomical process plan with input content identities; no full qualification required.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    from validation.legacy_dose_order import run_legacy_dose_probe

    result = run_legacy_dose_probe(args.qualification_plan or args.process_plan, args.output_dir)
    print("Characterization complete" if result["characterization_complete"] else "FAIL: " + result.get("error", "unexplained grid difference"))
    return 0 if result["characterization_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

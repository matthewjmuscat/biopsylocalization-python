"""Run standalone versus legacy-input anatomical validation in fresh processes."""

import argparse
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True, help="One prepared anatomical worker job JSON.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh validation directory.")
    parser.add_argument("--abs-tol", type=float, required=True)
    parser.add_argument("--rel-tol", type=float, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    from validation.anatomical_pair import run_anatomical_pair

    report = run_anatomical_pair(job_path=args.job, output_dir=args.output_dir, abs_tol=args.abs_tol, rel_tol=args.rel_tol, timeout_seconds=args.timeout_seconds)
    print("Anatomical comparison: {}".format("PASS" if report["passed"] else "FAIL"))
    print(args.output_dir / "anatomical_pair_summary.json")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
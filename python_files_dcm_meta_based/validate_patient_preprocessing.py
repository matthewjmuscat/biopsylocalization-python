"""Compare a standalone biopsy-preprocessing worker with a fresh legacy-input worker.

Reuses the existing paired service and checkpoint engine. Invoke only for an
explicit user-operated scientific validation; it is not normal worker dispatch.
"""

import argparse
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    from validation.anatomical_pair import run_anatomical_pair

    report = run_anatomical_pair(job_path=args.job, output_dir=args.output_dir,
                                abs_tol=0, rel_tol=0, timeout_seconds=args.timeout_seconds,
                                checkpoint_name="biopsy_preprocessing_shadow")
    print("Biopsy preprocessing comparison: " + ("PASS" if report["passed"] else "FAIL"))
    print(args.output_dir / "biopsy_preprocessing_pair_summary.json")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

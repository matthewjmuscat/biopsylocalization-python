"""User-operated anatomical independence qualification and post-run comparison."""

import argparse
from pathlib import Path


def main(argv=None) -> int:
    """Expose the bounded qualification recipe; help performs no patient I/O."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run fresh singleton pairs, forward/reverse cohorts and splits")
    for flag in ("input-case-manifest", "scientific-config-snapshot", "run-compatibility-identity", "output-dir"):
        run.add_argument("--" + flag, type=Path, required=True)
    run.add_argument("--patient-uid", action="append", required=True)
    run.add_argument("--split-a-patient-uid", action="append", required=True)
    run.add_argument("--timeout-seconds", type=float, default=3600)
    compare = commands.add_parser("compare", help="Compare completed patient jobs without executing science")
    compare.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    from validation.anatomical_independence import run_anatomical_independence, compare_completed_anatomical_jobs

    if args.command == "compare":
        report = compare_completed_anatomical_jobs(args.request)
    else:
        report = run_anatomical_independence(
            input_case_manifest=args.input_case_manifest, scientific_config_snapshot=args.scientific_config_snapshot,
            run_compatibility_identity=args.run_compatibility_identity, output_dir=args.output_dir,
            patient_uids=args.patient_uid, split_a=args.split_a_patient_uid, timeout_seconds=args.timeout_seconds,
        )
    print("PASS" if report["passed"] else "FAIL: " + report.get("error", "see numerical comparison reports"))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

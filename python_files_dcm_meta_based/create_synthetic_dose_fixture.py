"""Create private synthetic prescription variants from one explicitly selected F2 job."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-job", type=Path, required=True)
    parser.add_argument("--case-prefix", default="SYNTHETIC_DOSE",
                        help="Distinct SYNTHETIC_ name prefix for this fixture version; uppercase letters/digits/underscores.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Absent directory, e.g. Input data/synthetic set/dose_threshold_v1")
    args = parser.parse_args(argv)
    from validation.synthetic_dose_fixture import build_synthetic_dose_fixture

    result = build_synthetic_dose_fixture(source_job=args.source_job, output_dir=args.output_dir,
                                         case_prefix=args.case_prefix)
    print(result["input_case_manifest"])
    for subject in result["subjects"]:
        print(subject["patient_uid"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

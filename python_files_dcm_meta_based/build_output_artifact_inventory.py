from __future__ import annotations

import argparse
from pathlib import Path

from output_artifacts import build_output_artifact_inventory
from output_artifacts import build_patient_artifact_manifest
from output_artifacts import build_patient_stitch_plan
from output_artifacts import build_output_table_contracts
from output_artifacts import run_shadow_stitch_validation
from output_artifacts import summarize_patient_artifacts
from output_artifacts import summarize_output_artifact_inventory
from output_artifacts import summarize_output_table_contracts
from output_artifacts import summarize_shadow_stitch_validation
from output_artifacts import write_patient_artifact_outputs
from output_artifacts import write_output_artifact_inventory
from output_artifacts import write_output_table_contracts
from output_artifacts import write_shadow_stitch_validation
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
    parser.add_argument(
        "--build-shadow-stitches",
        action="store_true",
        help="Recreate selected final cohort tables from patient fragments and compare them against current final outputs.",
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
    contracts_df = build_output_table_contracts(inventory_df)
    contracts_path, contracts_summary_path = write_output_table_contracts(contracts_df, output_dir)
    patient_manifest_df = build_patient_artifact_manifest(inventory_df, contracts_df)
    stitch_plan_df = build_patient_stitch_plan(contracts_df)
    patient_manifest_path, stitch_plan_path, patient_summary_path = write_patient_artifact_outputs(
        patient_manifest_df,
        stitch_plan_df,
        output_dir,
    )
    summary = summarize_output_artifact_inventory(inventory_df)
    contracts_summary = summarize_output_table_contracts(contracts_df)
    patient_summary = summarize_patient_artifacts(patient_manifest_df, stitch_plan_df)
    print(f"[inventory] wrote {inventory_path}")
    print(f"[inventory] wrote {summary_path}")
    print(f"[contracts] wrote {contracts_path}")
    print(f"[contracts] wrote {contracts_summary_path}")
    print(f"[patient-artifacts] wrote {patient_manifest_path}")
    print(f"[patient-artifacts] wrote {stitch_plan_path}")
    print(f"[patient-artifacts] wrote {patient_summary_path}")
    print(
        "[inventory] artifacts={artifact_count} | tables={table_count} | classes={output_class_counts}".format(
            **summary,
        )
    )
    print(
        "[contracts] contracts={contract_count} | lifetime_classes={proposed_lifetime_class_counts} | prune_flags={pruning_assessment_counts}".format(
            **contracts_summary,
        )
    )
    print(
        "[patient-artifacts] patients={patient_count} | artifacts={patient_artifact_count} | stitch_readiness={stitch_readiness_counts}".format(
            **patient_summary,
        )
    )
    if args.build_shadow_stitches:
        validation_df = run_shadow_stitch_validation(inventory_df, output_dir)
        validation_path, validation_summary_path = write_shadow_stitch_validation(validation_df, output_dir)
        validation_summary = summarize_shadow_stitch_validation(validation_df)
        print(f"[shadow-stitch] wrote {validation_path}")
        print(f"[shadow-stitch] wrote {validation_summary_path}")
        print(
            "[shadow-stitch] pairs={validation_pair_count} | statuses={validation_status_counts}".format(
                **validation_summary,
            )
        )


if __name__ == "__main__":
    main()
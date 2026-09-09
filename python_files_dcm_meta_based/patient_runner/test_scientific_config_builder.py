"""Synthetic tests for patient-runner scientific config builder options."""

from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from patient_runner.contracts import LegacyRuntimeKeys
from patient_runner.contracts import PatientBatchRunConfig
from patient_runner.contracts import PatientRunConfig
from patient_runner.contracts import PatientStageName
from patient_runner.scientific_config import PatientMCSimulationScientificConfig
from patient_runner.scientific_config import PatientRunnerScientificConfig
from patient_runner.scientific_config_builder import PatientRunnerScientificConfigBuildContext
from patient_runner.scientific_config_builder import _build_mc_simulation_config
from patient_runner.scientific_config_builder import build_patient_runner_scientific_config
from patient_runner.scientific_runner import PatientScientificRunConfig
from patient_runner.scientific_runner import summarize_patient_scientific_run_config


class PatientRunnerScientificConfigBuilderTests(unittest.TestCase):
    def test_stage_scoped_builder_constructs_anatomical_checkpoint_only(self) -> None:
        pipeline_config = SimpleNamespace(
            structure_registry=SimpleNamespace(
                structs_referenced_dict={"Prostate ref": {}},
                structs_referenced_list=("Prostate ref",),
                structs_referenced_list_generalized=("Prostate ref",),
                structs_referenced_list_generalized_unique_structs=("Prostate ref",),
            )
        )
        grid_config = SimpleNamespace(enabled=True)
        anatomical_config = SimpleNamespace(enabled=True)

        with patch(
            "patient_runner.scientific_config_builder._build_grid_preprocessing_config",
            return_value=grid_config,
        ) as build_grid, patch(
            "patient_runner.scientific_config_builder._build_anatomical_preprocessing_config",
            return_value=anatomical_config,
        ) as build_anatomical:
            config = build_patient_runner_scientific_config(
                pipeline_config,
                stage_names=(
                    PatientStageName.GRID_PREPROCESSING,
                    PatientStageName.ANATOMICAL_PREPROCESSING,
                ),
            )

        build_grid.assert_called_once()
        build_anatomical.assert_called_once()
        self.assertIs(config.grid_preprocessing, grid_config)
        self.assertIs(config.anatomical_preprocessing, anatomical_config)
        self.assertIsNone(config.preprocessing)
        self.assertIsNone(config.optimization)
        self.assertIsNone(config.mc_simulation)

    def test_mc_simulation_builder_threads_dose_context_artifact_options(self) -> None:
        context = PatientRunnerScientificConfigBuildContext(
            persist_dose_context_artifacts=True,
            persist_dose_nn_render_context_artifacts=False,
            dose_context_artifact_localization_kinds=("Dose", "dose-gradient"),
            launch_dose_nn_render_selector_after_persisting_artifacts=True,
            dose_nn_render_selector_biopsy_index=3,
        )

        config = _build_mc_simulation_config(_minimal_pipeline_config(), context)

        self.assertTrue(config.persist_dose_context_artifacts)
        self.assertFalse(config.persist_dose_nn_render_context_artifacts)
        self.assertEqual(config.dose_context_artifact_localization_kinds, ("dose", "dose_gradient"))
        self.assertTrue(config.launch_dose_nn_render_selector_after_persisting_artifacts)
        self.assertEqual(config.dose_nn_render_selector_biopsy_index, 3)

    def test_run_plan_summary_exposes_persisting_artifact_options(self) -> None:
        run_config = PatientScientificRunConfig(
            batch_config=PatientBatchRunConfig(
                patient_config=PatientRunConfig(
                    output_root=Path("synthetic_output"),
                    legacy_keys=_legacy_keys(),
                ),
            ),
            scientific_config=PatientRunnerScientificConfig(
                mc_simulation=PatientMCSimulationScientificConfig(
                    persist_dose_context_artifacts=True,
                    persist_dose_nn_render_context_artifacts=True,
                    launch_dose_nn_render_selector_after_persisting_artifacts=True,
                    dose_nn_render_selector_biopsy_index=2,
                ),
            ),
        )

        summary = summarize_patient_scientific_run_config(run_config)

        self.assertTrue(summary["persisting_artifacts"]["dose_context"]["persist"])
        self.assertTrue(
            summary["persisting_artifacts"]["dose_context"]["launch_selector_after_persisting_artifacts"]
        )
        self.assertEqual(summary["persisting_artifacts"]["dose_context"]["selector_biopsy_index"], 2)


def _minimal_pipeline_config() -> SimpleNamespace:
    return SimpleNamespace(
        legacy_refs=SimpleNamespace(mr_adc_ref="MR_ADC_ref"),
        mc=SimpleNamespace(
            counts=SimpleNamespace(
                perform_mc_containment_sim=False,
                perform_mc_dose_sim=False,
                perform_mc_mr_sim=False,
                num_mc_containment_simulations_input=0,
                num_mc_dose_simulations_input=0,
                num_mc_mr_simulations_input=0,
            ),
            prep=SimpleNamespace(bx_sample_pts_lattice_spacing=1.0),
        ),
    )


def _legacy_keys() -> LegacyRuntimeKeys:
    return LegacyRuntimeKeys(
        all_ref_key="All ref",
        bx_ref="Bx ref",
        by_patient_key="By patient",
        global_key="Global",
        global_num_cases_key="Global num cases",
    )


if __name__ == "__main__":
    unittest.main()
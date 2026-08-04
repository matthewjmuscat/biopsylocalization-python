"""Synthetic tests for patient-runner scientific config builder options."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from patient_runner.scientific_config_builder import PatientRunnerScientificConfigBuildContext
from patient_runner.scientific_config_builder import _build_mc_simulation_config


class PatientRunnerScientificConfigBuilderTests(unittest.TestCase):
    def test_mc_simulation_builder_threads_dose_context_artifact_options(self) -> None:
        context = PatientRunnerScientificConfigBuildContext(
            write_dose_context_artifacts=True,
            write_dose_nn_render_context_artifacts=False,
            dose_context_artifact_localization_kinds=("Dose", "dose-gradient"),
        )

        config = _build_mc_simulation_config(_minimal_pipeline_config(), context)

        self.assertTrue(config.write_dose_context_artifacts)
        self.assertFalse(config.write_dose_nn_render_context_artifacts)
        self.assertEqual(config.dose_context_artifact_localization_kinds, ("dose", "dose_gradient"))


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


if __name__ == "__main__":
    unittest.main()
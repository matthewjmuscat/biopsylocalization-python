from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from preprocessing.structure_reference_bootstrap import PatientStructureReferenceBootstrapFragment

from patient_runner.contracts import PatientCase
from patient_runner.inputs import PatientInputPaths
from patient_runner.runtime_builder import build_standalone_patient_runtime
from patient_runner.worker_resources import SequentialWorkerPool


class StandalonePatientRuntimeBuilderTests(unittest.TestCase):
    def test_composes_one_patient_state_and_stage_context(self) -> None:
        patient_case = PatientCase(patient_uid="Synthetic-F2")
        patient_inputs = PatientInputPaths(
            patient_uid=patient_case.patient_uid,
            rtstruct=Path("synthetic/rtstruct.dcm"),
            rtdose=Path("synthetic/rtdose.dcm"),
            rtplan=Path("synthetic/rtplan.dcm"),
            mr_adc=(Path("synthetic/mr-adc-1.dcm"),),
        )
        pipeline_config = _pipeline_config()
        fragment = PatientStructureReferenceBootstrapFragment(
            patient_uid=patient_case.patient_uid,
            patient_reference_dict={"patient shell": True},
            patient_info_dict={"patient info": True},
            metadata={"num_total_structures": 5},
        )
        master_info = {
            "Global": {"Num cases": 1},
            "By patient": {patient_case.patient_uid: fragment.patient_info_dict},
        }

        with patch(
            "patient_runner.runtime_builder.build_patient_structure_reference_bootstrap_fragment_from_path_and_config",
            return_value=fragment,
        ) as build_fragment, patch(
            "patient_runner.runtime_builder.attach_patient_dose_reference_from_path",
            return_value=True,
        ) as attach_dose, patch(
            "patient_runner.runtime_builder.attach_patient_plan_reference_from_path",
        ) as attach_plan, patch(
            "patient_runner.runtime_builder.attach_patient_mr_adc_references_from_paths",
        ) as attach_mr, patch(
            "patient_runner.runtime_builder.assemble_structure_reference_info_for_run",
            return_value=master_info,
        ):
            runtime = build_standalone_patient_runtime(
                patient_case=patient_case,
                patient_inputs=patient_inputs,
                pipeline_config=pipeline_config,
                metadata={
                    "runtime_builder": "untrusted override",
                    "patient_input_manifest_identity_sha256": "untrusted override",
                },
            )

        build_fragment.assert_called_once()
        attach_dose.assert_called_once()
        attach_plan.assert_called_once()
        attach_mr.assert_called_once()
        self.assertIs(runtime.runtime_state.pydicom_item, fragment.patient_reference_dict)
        self.assertEqual(runtime.runtime_state.master_structure_info_dict, master_info)
        self.assertEqual(runtime.runtime_state.legacy_keys.all_ref_key, "All ref")
        self.assertEqual(runtime.runtime_state.legacy_keys.bx_ref, "Bx ref")
        self.assertEqual(runtime.runtime_state.legacy_keys.by_patient_key, "By patient")
        self.assertEqual(runtime.runtime_state.legacy_keys.global_key, "Global")
        self.assertEqual(runtime.runtime_state.legacy_keys.global_num_cases_key, "Num cases")
        self.assertEqual(runtime.metadata["runtime_builder"], "standalone_patient_runtime_v1")
        self.assertEqual(
            runtime.metadata["patient_input_manifest_identity_sha256"],
            patient_inputs.manifest_identity_sha256,
        )
        self.assertIsInstance(runtime.config_build_context.parallel_pool, SequentialWorkerPool)
        self.assertEqual(
            runtime.config_build_context.rtstruct_dicom_paths_by_patient_uid,
            {patient_case.patient_uid: patient_inputs.rtstruct},
        )


def _pipeline_config() -> SimpleNamespace:
    refs = SimpleNamespace(
        all_ref_key="All ref",
        bx_ref="Bx ref",
        by_patient_key="By patient",
        global_key="Global",
        global_num_cases_key="Num cases",
        oar_ref="OAR ref",
        dil_ref="DIL ref",
        rectum_ref_key="Rectum ref",
        urethra_ref_key="Urethra ref",
        dose_ref="Dose ref",
        plan_ref="Plan ref",
        mr_adc_ref="MR ADC ref",
    )
    return SimpleNamespace(
        legacy_refs=refs,
        bootstrap=SimpleNamespace(
            simulated_biopsies=SimpleNamespace(locations={}),
        ),
        structure_registry=SimpleNamespace(),
        preprocessing=SimpleNamespace(
            interp_inter_slice_dist=1.0,
            interp_intra_slice_dist=1.0,
        ),
    )


if __name__ == "__main__":
    unittest.main()
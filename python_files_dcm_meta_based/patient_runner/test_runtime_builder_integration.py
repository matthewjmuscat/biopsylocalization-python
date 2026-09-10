"""Synthetic DICOM integration at the runtime boundary; no scientific stages run."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, RTDoseStorage, RTPlanStorage
from pydicom.uid import MRImageStorage, RTStructureSetStorage, generate_uid

from config.bootstrap import PatientBootstrapConfig
from config.bootstrap import SimulatedBiopsyBootstrapPolicy, StructureDataRemovalPolicy
from config.pipeline import LegacyReferenceConfig, PipelineConfig, PreprocessingConfig
from config.rehydration import rehydrate_pipeline_scientific_config_snapshot
from config.snapshots import build_pipeline_scientific_config_snapshot
from config.snapshots import read_pipeline_config_snapshot, write_pipeline_config_snapshot
from config.test_snapshots import _build_real_pipeline_config
from patient_runner.contracts import LegacyPatientRuntimeState, LegacyRuntimeKeys, PatientCase
from patient_runner.inputs import PatientInputPaths
from patient_runner.runtime_builder import StandalonePatientRuntime
from patient_runner.runtime_builder import build_standalone_patient_runtime
from patient_runner.worker_resources import SequentialWorkerPool


_ADC_MAPPED_SERIES_UID = "1.2.826.0.1.3680043.8.498.20260909.1"
_ADC_FALLBACK_SERIES_UID = "1.2.826.0.1.3680043.8.498.20260909.2"


class StandalonePatientRuntimeIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        source_config = _build_real_pipeline_config()
        patient_uids = ("Synthetic (F1)", "Synthetic (F2)")
        source_config = replace(
            source_config,
            preprocessing=replace(
                source_config.preprocessing,
                interpolation=replace(
                    source_config.preprocessing.interpolation,
                    interp_inter_slice_dist=0.75,
                    interp_intra_slice_dist=0.25,
                ),
            ),
            bootstrap=PatientBootstrapConfig(
                removals=StructureDataRemovalPolicy(
                    biopsy={patient_uid: ("Bx remove",) for patient_uid in patient_uids},
                    dil={patient_uid: ("DIL remove",) for patient_uid in patient_uids},
                ),
                simulated_biopsies=SimulatedBiopsyBootstrapPolicy(
                    locations={
                        "Target DIL v2": {
                            "Create": True,
                            "Relative to struct type": "DIL ref",
                            "Identifier string": "synthetic_target",
                            "Transport family": "identity",
                        },
                    },
                    fraction_numbers_to_create=(2,),
                ),
            ),
            structure_registry=replace(
                source_config.structure_registry,
                structs_referenced_dict={
                    **source_config.structure_registry.structs_referenced_dict,
                    "DIL ref": {"Contour names": ("DIL",)},
                },
            ),
        )
        self.snapshot = build_pipeline_scientific_config_snapshot(source_config)
        self.snapshot_path = self.root / "resolved_scientific_config.json"
        write_pipeline_config_snapshot(self.snapshot, self.snapshot_path)
        self.config = rehydrate_pipeline_scientific_config_snapshot(self.snapshot_path)
        self.assertIs(type(self.config), PipelineConfig)
        self.assertIs(type(self.config.bootstrap), PatientBootstrapConfig)
        self.assertIs(type(self.config.bootstrap.removals), StructureDataRemovalPolicy)
        self.assertIs(type(self.config.bootstrap.simulated_biopsies), SimulatedBiopsyBootstrapPolicy)
        self.assertIs(type(self.config.legacy_refs), LegacyReferenceConfig)
        self.assertIs(type(self.config.preprocessing), PreprocessingConfig)
        self.assertEqual(self.config.bootstrap.simulated_biopsies.fraction_numbers_to_create, (2,))
        self.assertEqual(
            self.config.structure_registry.structs_referenced_dict["OAR ref"]["PCD color"],
            [1.0, 0.0, 0.0],
        )
        self._assert_config_unchanged()

    def test_f2_rehydration_bootstrap_dose_plan_and_run_info(self) -> None:
        inputs = self._write_inputs(fraction_number=2)
        self.assertTrue(inputs.core_paths_all_present)
        self.assertEqual(inputs.mr_adc, ())

        runtime = self._build_runtime(inputs)

        self._assert_runtime_contract(runtime, inputs, fraction_number=2)
        self._assert_dose(runtime.runtime_state.pydicom_item["Dose ref"], inputs.patient_uid)
        self._assert_config_unchanged()

    def test_f1_skips_unreadable_dose_but_attaches_plan(self) -> None:
        """Test dose omission at runtime, independently of worker path preflight."""
        inputs = replace(
            self._write_inputs(fraction_number=1),
            rtdose=self.root / "never-created-rtdose.dcm",
        )
        self.assertEqual(inputs.missing_core_roles, ("rtdose",))

        runtime = self._build_runtime(inputs)

        self._assert_runtime_contract(runtime, inputs, fraction_number=1)
        self.assertNotIn("Dose ref", runtime.runtime_state.pydicom_item)
        self.assertIs(runtime.metadata["dose_reference_attached"], False)
        self._assert_config_unchanged()

    def test_optional_adc_normalization_preserves_slice_order_and_series(self) -> None:
        mapped_paths = self._write_adc_series(mapped=True)
        fallback_paths = self._write_adc_series(mapped=False)
        paths = (mapped_paths[0], fallback_paths[0], mapped_paths[1])
        inputs = self._write_inputs(
            fraction_number=2,
            mr_adc_manifest=f"  {paths[0]} | | {paths[1]} | {paths[2]} |  ",
        )
        self.assertEqual(inputs.mr_adc, paths)
        without_adc = replace(inputs, mr_adc=())
        self.assertNotEqual(inputs.manifest_identity_sha256, without_adc.manifest_identity_sha256)

        for label, selected_inputs in (
            ("absent before ADC", without_adc),
            ("normalized ADC series", inputs),
            ("absent after ADC", without_adc),
        ):
            with self.subTest(adc=label):
                runtime = self._build_runtime(selected_inputs)
                self._assert_runtime_contract(runtime, selected_inputs, fraction_number=2)
                if selected_inputs.mr_adc:
                    self._assert_adc(
                        runtime.runtime_state.pydicom_item["MR ADC ref"],
                        selected_inputs.patient_uid,
                        mapped=True,
                        fallback=True,
                    )
                else:
                    self.assertNotIn("MR ADC ref", runtime.runtime_state.pydicom_item)
        self._assert_config_unchanged()

    def test_adc_without_mapping_retains_default_units_and_single_slice_shapes(self) -> None:
        paths = self._write_adc_series(mapped=False)
        inputs = self._write_inputs(fraction_number=2, mr_adc_manifest=str(paths[0]))

        runtime = self._build_runtime(inputs)

        self._assert_runtime_contract(runtime, inputs, fraction_number=2)
        self._assert_adc(
            runtime.runtime_state.pydicom_item["MR ADC ref"], inputs.patient_uid,
            mapped=False, fallback=True,
        )
        self._assert_config_unchanged()

    def test_repeated_calls_and_interleaved_patients_have_fresh_mutable_state(self) -> None:
        paths = self._write_adc_series(mapped=True)
        inputs = self._write_inputs(
            fraction_number=2, mr_adc_manifest=" | ".join(str(path) for path in paths),
        )
        first = self._build_runtime(inputs)
        second = self._build_runtime(inputs)
        self._assert_independent_runtimes(first, second)

        first_patient = first.runtime_state.pydicom_item
        first_patient["Dose ref"]["Dose pixel arr"][0, 0, 0] = 999
        first_patient["Dose ref"]["Image position patient"][0] = 999.0
        first_patient["Plan ref"]["Prescription doses dict"]["TARGET"] = -1.0
        first_patient["OAR ref"][0]["Nearest neighbours objects"].append("changed")
        first_patient["Bx ref"][0]["Output data frames"]["Differential DVH by MC trial"] = "changed"
        first_patient["All ref"]["Multi-structure pre-processing output dataframes dict"][
            "Selected structures"
        ] = "changed"
        first_adc = first_patient["MR ADC ref"][_ADC_MAPPED_SERIES_UID]
        for key in (
            "Pixel arr (all slices)", "Pixel spacing", "RWVSlope (all slices)",
            "RWVIntercept (all slices)", "Image orientation patient",
            "Image position patient (all slices)",
        ):
            first_adc[key].flat[0] = 999
        first_info = first.runtime_state.master_structure_info_dict
        first_info["Global"]["Preprocessing info"]["Preprocessing performed"] = True
        first_info["By patient"][inputs.patient_uid]["Bx ref"]["Num structs"] = -1
        first.runtime_state.metadata["bootstrap"]["num_total_structures"] = -1
        first.config_build_context.rtstruct_dicom_paths_by_patient_uid[inputs.patient_uid] = (
            self.root / "changed.dcm"
        )

        other_inputs = self._write_inputs(fraction_number=1)
        other = self._build_runtime(other_inputs)
        self._assert_runtime_contract(other, other_inputs, fraction_number=1)
        third = self._build_runtime(inputs)
        self._assert_independent_runtimes(second, third)
        for runtime in (second, third):
            self._assert_runtime_contract(runtime, inputs, fraction_number=2)
            self._assert_dose(runtime.runtime_state.pydicom_item["Dose ref"], inputs.patient_uid)
            self._assert_adc(
                runtime.runtime_state.pydicom_item["MR ADC ref"], inputs.patient_uid,
                mapped=True, fallback=False,
            )
        self._assert_config_unchanged()

    def _write_inputs(
        self, *, fraction_number: int, mr_adc_manifest: str | None = None,
    ) -> PatientInputPaths:
        directory = self.root / f"F{fraction_number}"
        directory.mkdir()
        structure = _dataset(RTStructureSetStorage, "RTSTRUCT", fraction_number)
        structure.StructureSetLabel = "SYNTHETIC"
        structure.StructureSetROISequence = Sequence(
            [_roi(name, number) for number, name in enumerate(
                ("Prostate", "DIL target", "DIL remove", "Bx keep", "Bx remove", "Rectum", "Urethra"),
                start=1,
            )]
        )
        structure.save_as(directory / "rtstruct.dcm", enforce_file_format=True)

        dose = _dataset(RTDoseStorage, "RTDOSE", fraction_number)
        _set_pixels(dose, _dose_pixels())
        dose.DoseGridScaling = 0.01
        dose.DoseUnits = "GY"
        dose.DoseType = "PHYSICAL"
        dose.DoseSummationType = "PLAN"
        dose.PixelSpacing = [1.5, 2.5]
        dose.GridFrameOffsetVector = [0.0, 4.0]
        dose.ImageOrientationPatient = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0]
        dose.ImagePositionPatient = [10.0, -20.0, 30.0]
        dose.save_as(directory / "rtdose.dcm", enforce_file_format=True)

        plan = _dataset(RTPlanStorage, "RTPLAN", fraction_number)
        plan.RTPlanLabel = "SYNTHETIC"
        target = Dataset()
        target.DoseReferenceNumber = 1
        target.DoseReferenceType = "TARGET"
        target.TargetPrescriptionDose = 15.5
        organ_at_risk = Dataset()
        organ_at_risk.DoseReferenceNumber = 2
        organ_at_risk.DoseReferenceType = "ORGAN_AT_RISK"
        organ_at_risk.TargetPrescriptionDose = 7.25
        plan.DoseReferenceSequence = Sequence([target, organ_at_risk])
        plan.save_as(directory / "rtplan.dcm", enforce_file_format=True)
        return PatientInputPaths.from_case_manifest_row({
            "Patient UID (generated)": f"Synthetic (F{fraction_number})",
            "RTSTRUCT path": str(directory / "rtstruct.dcm"),
            "RTDOSE path": str(directory / "rtdose.dcm"),
            "RTPLAN path": str(directory / "rtplan.dcm"),
            "US paths": None,
            "MR T2 paths": None,
            "MR ADC paths": mr_adc_manifest,
        })

    def _write_adc_series(self, *, mapped: bool) -> tuple[Path, ...]:
        if mapped:
            slices = (
                ("adc-z-first.dcm", 11, [30.0, -40.0, 8.5], 2e-6, -1e-6),
                ("adc-a-second.dcm", 21, [30.0, -40.0, 4.5], 3e-6, 1e-6),
            )
        else:
            slices = (("adc-fallback.dcm", 31, [-12.0, 20.0, 5.0], 0.0, 0.0),)
        paths = []
        for filename, first_pixel, position, slope, intercept in slices:
            dataset = _dataset(MRImageStorage, "MR", 2)
            dataset.SeriesInstanceUID = _ADC_MAPPED_SERIES_UID if mapped else _ADC_FALLBACK_SERIES_UID
            dataset.ImageType = ["DERIVED", "PRIMARY", "ADC"]
            dataset.PixelSpacing = [0.7, 0.9]
            dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
            dataset.ImagePositionPatient = position
            _set_pixels(dataset, np.arange(first_pixel, first_pixel + 6, dtype="<u2").reshape(2, 3))
            if mapped:
                dataset.SliceThickness = 4.0
                units = Dataset()
                units.CodeValue = "mm2/s"
                units.CodingSchemeDesignator = "UCUM"
                units.CodeMeaning = "square millimeters per second"
                mapping = Dataset()
                mapping.RealWorldValueSlope = slope
                mapping.RealWorldValueIntercept = intercept
                mapping.RealWorldValueFirstValueMapped = 0
                mapping.RealWorldValueLastValueMapped = 65535
                mapping.LUTLabel = "ADC"
                mapping.MeasurementUnitsCodeSequence = Sequence([units])
                dataset.RealWorldValueMappingSequence = Sequence([mapping])
            path = self.root / filename
            dataset.save_as(path, enforce_file_format=True)
            paths.append(path)
        return tuple(paths)

    def _build_runtime(self, inputs: PatientInputPaths) -> StandalonePatientRuntime:
        return build_standalone_patient_runtime(
            patient_case=PatientCase(patient_uid=inputs.patient_uid),
            patient_inputs=inputs,
            pipeline_config=self.config,
            metadata={"test_label": "Phase2C"},
        )

    def _assert_config_unchanged(self) -> None:
        self.assertRegex(self.snapshot.config_sha256, r"^[0-9a-f]{64}$")
        for snapshot in (
            build_pipeline_scientific_config_snapshot(self.config),
            read_pipeline_config_snapshot(self.snapshot_path),
        ):
            self.assertEqual(snapshot.config_sha256, self.snapshot.config_sha256)
            self.assertEqual(snapshot.config, self.snapshot.config)

    def _assert_runtime_contract(
        self,
        runtime: StandalonePatientRuntime,
        inputs: PatientInputPaths,
        *,
        fraction_number: int,
    ) -> None:
        has_dose = fraction_number != 1
        has_simulated_biopsy = fraction_number == 2
        biopsy_count = 1 + int(has_simulated_biopsy)
        biopsy_types = {"Real": 1}
        if has_simulated_biopsy:
            biopsy_types["Target DIL v2"] = 1
        state = runtime.runtime_state
        self.assertIs(type(state), LegacyPatientRuntimeState)
        self.assertEqual(set(state.master_structure_reference_dict), {inputs.patient_uid})
        self.assertEqual(state.legacy_keys, LegacyRuntimeKeys(
            all_ref_key="All ref",
            bx_ref="Bx ref",
            by_patient_key="By patient",
            global_key="Global",
            global_num_cases_key="Num cases",
        ))
        patient = state.pydicom_item
        identity = {
            "Patient UID (generated)": inputs.patient_uid,
            "Patient ID (from dicom)": f"F{fraction_number}",
            "Patient Name": "Synthetic^Runtime",
            "Fraction number": fraction_number,
        }
        expected_keys = set(identity) | {
            "Bx ref", "OAR ref", "DIL ref", "Rectum ref", "Urethra ref",
            "All ref", "Ready to plot data list", "Plan ref",
        }
        if has_dose:
            expected_keys.add("Dose ref")
        if inputs.mr_adc:
            expected_keys.add("MR ADC ref")
        self.assertEqual(set(patient), expected_keys)
        self.assertEqual({key: patient[key] for key in identity}, identity)
        self.assertIsNone(patient["Ready to plot data list"])
        for ref, name, number in (
            ("OAR ref", "Prostate", 1), ("DIL ref", "DIL target", 2),
            ("Rectum ref", "Rectum", 6), ("Urethra ref", "Urethra", 7),
        ):
            with self.subTest(structure_family=ref):
                self.assertEqual(len(patient[ref]), 1)
                record = patient[ref][0]
                self.assertEqual(
                    (record["ROI"], record["Ref #"], record["Index number"], record["Struct type"]),
                    (name, number, 0, ref),
                )
                self.assertIsNone(record["Raw contour pts"])
                self.assertIsNone(record["Reconstructed structure pts arr"])
                self.assertEqual(record["Nearest neighbours objects"], [])
        biopsies = state.biopsy_structures
        expected_biopsies = [("Bx keep", 4, 0, False, "Real")]
        if has_simulated_biopsy:
            expected_biopsies.append((
                "Bx_Tr_synthetic_target DIL target", "synthetic_target DIL target",
                1, True, "Target DIL v2",
            ))
            self.assertIsNone(biopsies[1]["Simulated biopsy transport request dict"])
            self.assertEqual(biopsies[1]["Relative structure ref #"], 2)
        self.assertEqual([
            (record["ROI"], record["Ref #"], record["Index number"],
             record["Simulated bool"], record["Simulated type"])
            for record in biopsies
        ], expected_biopsies)
        self.assertEqual(set(patient["All ref"]), {
            "Multi-structure information dict (not for csv output)",
            "Multi-structure pre-processing output dataframes dict",
            "Multi-structure MC simulation output dataframes dict",
        })
        self.assertEqual(state.master_structure_info_dict, {
            "Global": {
                "Num cases": 1,
                "Num unique patient names": 1,
                "Num structures": 4 + biopsy_count,
                "Num biopsies": biopsy_count,
                "Num biopsies by bx type dict": biopsy_types,
                "Num DILs": 1,
                "Bx types list": ["Real", "Target DIL v2"],
                "Preprocessing info": {
                    "Interslice interp dist": 0.75,
                    "Intraslice interp dist": 0.25,
                    "Preprocessing performed": False,
                },
                "MC info": {
                    "Num MC containment simulations": None,
                    "Num MC dose simulations": None,
                    "Num MC MR simulations": None,
                    "Num optimizer v2 transform samples": None,
                    "Num stochastic targeting transform samples": None,
                    "Num sample pts per BX core": None,
                    "BX sample pt lattice spacing (mm)": None,
                    "BX sample pt volume element (mm^3)": None,
                    "Max of num MC simulations": None,
                    "Max of generated transform samples": None,
                    "MC sim performed": False,
                    "MC containment sim performed": False,
                    "MC dose sim performed": False,
                    "MC MR sim performed": False,
                },
                "Random info": {
                    "Transform generation random seed": None,
                    "Optimizer v1 random seed": None,
                },
                "Patient specific guidance map figures directory dict": None,
                "Guidance map figures dir": None,
                "Specific output dir": None,
            },
            "By patient": {inputs.patient_uid: {
                **identity,
                "Bx ref": {
                    "Num structs": biopsy_count,
                    "Num sim structs": int(has_simulated_biopsy),
                    "Num real structs": 1,
                    "Biopsy type counts": biopsy_types,
                },
                "OAR ref": {"Num structs": 1},
                "DIL ref": {"Num structs": 1},
                "Rectum ref": {"Num structs": 1},
                "Urethra ref": {"Num structs": 1},
                "All ref": {"Total num structs": 4 + biopsy_count},
            }},
        })
        self.assertEqual(patient["Plan ref"], {
            "Plan ID": inputs.patient_uid + "20260102",
            "Study date": "20260102",
            "Dose units": "Gy",
            "Prescription doses dict": {"TARGET": 15.5, "ORGAN_AT_RISK": 7.25},
        })
        expected_metadata = {
            "test_label": "Phase2C",
            "runtime_builder": "standalone_patient_runtime_v1",
            "patient_input_manifest_identity_sha256": inputs.manifest_identity_sha256,
            "dose_reference_attached": has_dose,
            "plan_reference_attached": True,
            "mr_adc_reference_attached": bool(inputs.mr_adc),
            "bootstrap": {
                "num_real_biopsies": 1,
                "num_simulated_biopsies": int(has_simulated_biopsy),
                "num_total_structures": 4 + biopsy_count,
            },
        }
        self.assertEqual(runtime.metadata, expected_metadata)
        self.assertEqual(state.metadata, expected_metadata)
        self.assertIsInstance(runtime.config_build_context.parallel_pool, SequentialWorkerPool)
        self.assertEqual(runtime.config_build_context.rtstruct_dicom_paths_by_patient_uid,
                         {inputs.patient_uid: inputs.rtstruct})
        self.assertEqual(runtime.config_build_context.metadata, {
            "runtime_builder": "standalone_patient_runtime_v1",
            "patient_input_manifest_identity_sha256": inputs.manifest_identity_sha256,
        })

    def _assert_dose(self, dose: dict, patient_uid: str) -> None:
        np.testing.assert_array_equal(dose["Dose pixel arr"], _dose_pixels())
        self.assertEqual(dose["Dose pixel arr"].dtype, np.dtype("<u2"))
        self.assertEqual({key: value for key, value in dose.items() if key != "Dose pixel arr"}, {
            "Dose ID": patient_uid + "20260102",
            "Study date": "20260102",
            "Dose pixel data": _dose_pixels().tobytes(),
            "Pixel spacing": [1.5, 2.5],
            "Dose grid scaling": 0.01,
            "Dose units": "GY",
            "Dose type": "PHYSICAL",
            "Grid frame offset vector": [0.0, 4.0],
            "Image orientation patient": [0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
            "Image position patient": [10.0, -20.0, 30.0],
            "Dose and gradient phys space and pixel 3d arr": None,
            "Dose grid point cloud": None,
            "Dose grid point cloud thresholded": None,
            "Dose grid gradient point cloud": None,
            "Dose grid gradient point cloud thresholded": None,
            "KDtree": None,
            "KDtree gradient": None,
        })

    def _assert_adc(
        self, references: dict, patient_uid: str, *, mapped: bool, fallback: bool,
    ) -> None:
        common = {
            "MR ADC ID": patient_uid + "20260102",
            "Study date": "20260102",
            "Pixel spacing": np.array([0.7, 0.9]),
            "Image orientation patient": np.array([1.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
            "MR ADC phys space Nx4 arr": None,
            "MR ADC phys space Nx4 arr (filtered, non-negative)": None,
            "MR ADC grid point cloud": None,
            "MR ADC grid point cloud thresholded": None,
            "KDtree": None,
        }
        expected_series = {}
        if mapped:
            expected_series[_ADC_MAPPED_SERIES_UID] = {
                **common,
                "Series instance UID": _ADC_MAPPED_SERIES_UID,
                "Pixel arr (all slices)": np.array([
                    [[11, 21], [12, 22], [13, 23]],
                    [[14, 24], [15, 25], [16, 26]],
                ], dtype="<u2"),
                "Units": "square millimeters per second",
                "RWVSlope (all slices)": np.array([2e-6, 3e-6]),
                "RWVIntercept (all slices)": np.array([-1e-6, 1e-6]),
                "RWV Units": "ADC",
                "Slice thickness": 4.0,
                "Image position patient (all slices)": np.array([
                    [30.0, -40.0, 8.5], [30.0, -40.0, 4.5],
                ]),
            }
        if fallback:
            expected_series[_ADC_FALLBACK_SERIES_UID] = {
                **common,
                "Series instance UID": _ADC_FALLBACK_SERIES_UID,
                "Pixel arr (all slices)": np.array([[31, 32, 33], [34, 35, 36]], dtype="<u2"),
                "Units": "mm\u00b2/s (assumed)",
                "RWVSlope (all slices)": np.array([1e-6]),
                "RWVIntercept (all slices)": np.array([0.0]),
                "RWV Units": "mm\u00b2/s (assumed)",
                "Slice thickness": -1,
                "Image position patient (all slices)": np.array([-12.0, 20.0, 5.0]),
            }
        self.assertEqual(set(references), set(expected_series))
        for series_uid, expected in expected_series.items():
            actual = references[series_uid]
            self.assertEqual(set(actual), set(expected))
            for key, value in expected.items():
                with self.subTest(series_uid=series_uid, field=key):
                    if isinstance(value, np.ndarray):
                        self.assertIsInstance(actual[key], np.ndarray)
                        np.testing.assert_array_equal(actual[key], value)
                        self.assertEqual(actual[key].dtype, value.dtype)
                    else:
                        self.assertEqual(actual[key], value)

    def _assert_independent_runtimes(
        self, first: StandalonePatientRuntime, second: StandalonePatientRuntime,
    ) -> None:
        self.assertIsNot(first, second)
        self.assertIsNot(first.runtime_state, second.runtime_state)
        self.assertIsNot(first.config_build_context, second.config_build_context)
        self.assertIsNot(first.config_build_context.parallel_pool, second.config_build_context.parallel_pool)
        for first_tree, second_tree in (
            (first.runtime_state.master_structure_reference_dict, second.runtime_state.master_structure_reference_dict),
            (first.runtime_state.master_structure_info_dict, second.runtime_state.master_structure_info_dict),
            (first.runtime_state.metadata, second.runtime_state.metadata),
            (first.metadata, second.metadata),
            (first.config_build_context.metadata, second.config_build_context.metadata),
            (first.config_build_context.rtstruct_dicom_paths_by_patient_uid,
             second.config_build_context.rtstruct_dicom_paths_by_patient_uid),
        ):
            self._assert_independent_tree(first_tree, second_tree)

    def _assert_independent_tree(self, first: object, second: object) -> None:
        """Compare all nested values while rejecting shared containers or array buffers."""
        if isinstance(first, dict) and isinstance(second, dict):
            self.assertIsNot(first, second)
            self.assertEqual(set(first), set(second))
            for key in first:
                with self.subTest(independent_key=key):
                    self._assert_independent_tree(first[key], second[key])
        elif isinstance(first, list) and isinstance(second, list):
            self.assertIsNot(first, second)
            self.assertEqual(len(first), len(second))
            for first_value, second_value in zip(first, second):
                self._assert_independent_tree(first_value, second_value)
        elif isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
            self.assertIsNot(first, second)
            self.assertFalse(np.shares_memory(first, second))
            np.testing.assert_array_equal(first, second)
        else:
            self.assertEqual(first, second)


def _dataset(sop_class_uid: str, modality: str, fraction_number: int) -> Dataset:
    dataset = Dataset()
    dataset.file_meta = FileMetaDataset()
    dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset.file_meta.MediaStorageSOPClassUID = sop_class_uid
    dataset.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    dataset.SOPClassUID = sop_class_uid
    dataset.SOPInstanceUID = dataset.file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = modality
    dataset.PatientName = "Synthetic^Runtime"
    dataset.PatientID = f"F{fraction_number}"
    dataset.StudyDate = "20260102"
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    return dataset


def _roi(name: str, number: int) -> Dataset:
    roi = Dataset()
    roi.ROIName = name
    roi.ROINumber = number
    return roi


def _set_pixels(dataset: Dataset, pixels: np.ndarray) -> None:
    dataset.Rows, dataset.Columns = pixels.shape[-2:]
    if pixels.ndim == 3:
        dataset.NumberOfFrames = pixels.shape[0]
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = pixels.astype("<u2").tobytes()


def _dose_pixels() -> np.ndarray:
    return np.array([
        [[1, 2, 3], [4, 5, 6]],
        [[101, 102, 103], [104, 105, 106]],
    ], dtype="<u2")


if __name__ == "__main__":
    unittest.main()
from __future__ import annotations

from types import SimpleNamespace
import unittest

from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from config.bootstrap import PatientBootstrapConfig
from config.bootstrap import SimulatedBiopsyBootstrapPolicy
from config.bootstrap import StructureContourPolicy
from config.bootstrap import StructureDataRemovalPolicy
from preprocessing.structure_reference_bootstrap import build_patient_structure_reference_bootstrap_fragment
from preprocessing.structure_reference_bootstrap import build_patient_structure_reference_bootstrap_fragment_from_config


class PatientStructureReferenceBootstrapTests(unittest.TestCase):
    def test_typed_adapter_matches_direct_patient_bootstrap(self) -> None:
        patient_uid = "Synthetic (F2)"
        structure_item = _structure_dataset()
        locations = {
            "Target DIL v2": {
                "Create": True,
                "Relative to struct type": "DIL ref",
                "Transport family": "identity",
                "Identifier string": "sim_target_dil_v2",
            }
        }
        bootstrap_config = PatientBootstrapConfig(
            removals=StructureDataRemovalPolicy(
                biopsy={patient_uid: ("Bx remove",)},
                dil={patient_uid: ("DIL remove",)},
            ),
            contours=StructureContourPolicy(),
            simulated_biopsies=SimulatedBiopsyBootstrapPolicy(
                locations=locations,
                fraction_numbers_to_create="all",
                fraction_prefixes=("f", "fraction", ""),
            ),
        )
        legacy_refs = SimpleNamespace(
            all_ref_key="All ref",
            bx_ref="Bx ref",
            oar_ref="OAR ref",
            dil_ref="DIL ref",
            rectum_ref_key="Rectum ref",
            urethra_ref_key="Urethra ref",
        )
        structure_registry = SimpleNamespace(
            structs_referenced_dict={
                "Bx ref": {"Contour names": ("Bx",)},
                "OAR ref": {"Contour names": ("Prostate",)},
                "DIL ref": {"Contour names": ("DIL",)},
                "Rectum ref": {"Contour names": ("Rectum",)},
                "Urethra ref": {"Contour names": ("Urethra",)},
            }
        )

        typed_result = build_patient_structure_reference_bootstrap_fragment_from_config(
            patient_uid=patient_uid,
            structure_item=structure_item,
            bootstrap_config=bootstrap_config,
            legacy_refs=legacy_refs,
            structure_registry=structure_registry,
        )
        direct_result = build_patient_structure_reference_bootstrap_fragment(
            patient_uid=patient_uid,
            structure_item=structure_item,
            data_removals_dict_bx=bootstrap_config.removals.biopsy,
            data_removals_dict_prostate=bootstrap_config.removals.prostate,
            data_removals_dict_dil=bootstrap_config.removals.dil,
            data_removals_dict_urethra=bootstrap_config.removals.urethra,
            data_removals_dict_rectum=bootstrap_config.removals.rectum,
            OAR_list=bootstrap_config.contours.oar,
            DIL_list=bootstrap_config.contours.dil,
            Bx_list=bootstrap_config.contours.biopsy,
            st_ref_list=("Bx ref", "OAR ref", "DIL ref", "Rectum ref", "Urethra ref"),
            structs_referenced_dict=structure_registry.structs_referenced_dict,
            all_ref_key="All ref",
            mr_global_multi_structure_output_dataframe_str=bootstrap_config.mr_global_structure_table_name,
            mr_global_by_voxel_multi_structure_output_dataframe_str=bootstrap_config.mr_global_voxel_table_name,
            bx_sim_locations_dict=bootstrap_config.simulated_biopsies.locations,
            rectum_list=bootstrap_config.contours.rectum,
            urethra_list=bootstrap_config.contours.urethra,
            simulated_biopsy_fraction_numbers_to_create=(
                bootstrap_config.simulated_biopsies.fraction_numbers_to_create
            ),
            fraction_prefixes=bootstrap_config.simulated_biopsies.fraction_prefixes,
        )

        self.assertEqual(typed_result.patient_reference_dict, direct_result.patient_reference_dict)
        self.assertEqual(typed_result.patient_info_dict, direct_result.patient_info_dict)
        self.assertEqual(typed_result.metadata, direct_result.metadata)
        self.assertEqual(typed_result.metadata["num_real_biopsies"], 1)
        self.assertEqual(typed_result.metadata["num_simulated_biopsies"], 1)


def _structure_dataset() -> Dataset:
    dataset = Dataset()
    dataset.PatientName = "Synthetic"
    dataset.PatientID = "F2"
    dataset.StructureSetROISequence = Sequence(
        [
            _roi("Prostate", 1),
            _roi("DIL target", 2),
            _roi("DIL remove", 3),
            _roi("Bx keep", 4),
            _roi("Bx remove", 5),
            _roi("Rectum", 6),
            _roi("Urethra", 7),
        ]
    )
    return dataset


def _roi(name: str, number: int) -> Dataset:
    roi = Dataset()
    roi.ROIName = name
    roi.ROINumber = number
    return roi


if __name__ == "__main__":
    unittest.main()

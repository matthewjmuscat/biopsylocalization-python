from __future__ import annotations

import unittest

from .bootstrap import PatientBootstrapConfig
from .bootstrap import SimulatedBiopsyBootstrapPolicy
from .bootstrap import StructureContourPolicy
from .bootstrap import StructureDataRemovalPolicy
from .snapshots import canonical_sha256


class PatientBootstrapConfigTests(unittest.TestCase):
    def test_policy_normalizes_mutable_sequences_to_stable_values(self) -> None:
        config = PatientBootstrapConfig(
            removals=StructureDataRemovalPolicy(
                biopsy={"P001": ["Bx bad"]},
                dil={"P001": ["DIL bad"]},
            ),
            contours=StructureContourPolicy(
                oar=["Prostate"],
                dil=["DIL"],
                biopsy=["Bx"],
                rectum=["Rectum"],
                urethra=["Urethra"],
            ),
            simulated_biopsies=SimulatedBiopsyBootstrapPolicy(
                locations={
                    "Target DIL v2": {
                        "Create": 1,
                        "Relative to struct type": "DIL ref",
                        "Transport family": "identity",
                        "Identifier string": "sim_target_dil_v2",
                    }
                },
                fraction_numbers_to_create=[1, 2],
                fraction_prefixes=["f", "fraction", ""],
            ),
        )

        self.assertEqual(config.removals.biopsy["P001"], ("Bx bad",))
        self.assertEqual(config.contours.oar, ("Prostate",))
        self.assertEqual(config.simulated_biopsies.fraction_numbers_to_create, (1, 2))
        self.assertTrue(config.simulated_biopsies.locations["Target DIL v2"]["Create"])

    def test_equivalent_policy_inputs_have_same_fingerprint(self) -> None:
        first = StructureDataRemovalPolicy(biopsy={"P001": ["Bx bad"]})
        second = StructureDataRemovalPolicy(biopsy={"P001": ("Bx bad",)})

        self.assertEqual(canonical_sha256(first), canonical_sha256(second))

    def test_simulated_biopsy_location_requires_identity_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "Identifier string"):
            SimulatedBiopsyBootstrapPolicy(
                locations={
                    "Target DIL v2": {
                        "Create": True,
                        "Relative to struct type": "DIL ref",
                    }
                }
            )


if __name__ == "__main__":
    unittest.main()

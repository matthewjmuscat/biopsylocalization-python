from __future__ import annotations

import unittest

from .state_isolation import POST_DISCOVERY_DEEPCOPY_SOURCE
from .state_isolation import copy_isolated_legacy_runtime_state_from_snapshot


class PatientRunnerStateIsolationTests(unittest.TestCase):
    def test_snapshot_state_is_deep_copied(self) -> None:
        reference_snapshot = {"P001": {"nested": {"values": [1, 2]}}}
        info_snapshot = {"Global": {"Num cases": 1}}

        isolated = copy_isolated_legacy_runtime_state_from_snapshot(
            reference_snapshot,
            info_snapshot,
            operation_name="synthetic runner",
        )
        isolated.master_structure_reference_dict["P001"]["nested"]["values"].append(3)
        isolated.master_structure_info_dict["Global"]["Num cases"] = 99

        self.assertEqual(reference_snapshot["P001"]["nested"]["values"], [1, 2])
        self.assertEqual(info_snapshot["Global"]["Num cases"], 1)
        self.assertEqual(isolated.source, POST_DISCOVERY_DEEPCOPY_SOURCE)

    def test_missing_snapshot_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "will not fall back"):
            copy_isolated_legacy_runtime_state_from_snapshot(
                None,
                None,
                operation_name="synthetic runner",
            )

    def test_partial_snapshot_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "found only one"):
            copy_isolated_legacy_runtime_state_from_snapshot(
                {"P001": {}},
                None,
                operation_name="synthetic runner",
            )


if __name__ == "__main__":
    unittest.main()

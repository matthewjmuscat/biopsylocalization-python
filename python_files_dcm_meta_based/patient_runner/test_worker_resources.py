from __future__ import annotations

import unittest

from patient_runner.worker_resources import SequentialWorkerPool


def _add(left: int, right: int) -> int:
    return left + right


class SequentialWorkerPoolTests(unittest.TestCase):
    def test_map_preserves_order_and_returns_list(self) -> None:
        result = SequentialWorkerPool().map(str.upper, ("first", "second"), chunksize=1)

        self.assertEqual(result, ["FIRST", "SECOND"])

    def test_starmap_unpacks_arguments_and_returns_list(self) -> None:
        result = SequentialWorkerPool().starmap(_add, ((1, 2), (10, 20)), chunksize=2)

        self.assertEqual(result, [3, 30])


if __name__ == "__main__":
    unittest.main()
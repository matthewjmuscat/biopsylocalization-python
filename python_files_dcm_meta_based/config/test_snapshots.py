from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import tempfile
import unittest

import numpy as np

from biopsy_optimizer.v2.config import build_default_optimizer_v2_search_config
from .pipeline import ArtifactConfig
from .pipeline import GuidanceMapConfig
from .pipeline import OptimizerRuntimeConfig
from .pipeline import PipelineConfig
from .pipeline import PreprocessingConfig
from .pipeline import PreprocessingGeometryConfig
from .pipeline import PreprocessingInterpolationConfig
from .pipeline import PreprocessingKernelExecutionConfig
from .pipeline import RandomSeedConfig
from .pipeline import RuntimeReplayConfig
from .pipeline import RuntimeUIConfig
from .pipeline import StructureRegistryConfig
from .rehydration import rehydrate_pipeline_scientific_config_snapshot
from .snapshots import PipelineConfigSnapshot
from .snapshots import build_pipeline_config_snapshot
from .snapshots import build_pipeline_scientific_config_snapshot
from .snapshots import canonical_sha256
from .snapshots import read_pipeline_config_snapshot
from .snapshots import write_pipeline_config_snapshot


class _Mode(str, Enum):
    ENABLED = "enabled"


@dataclass(frozen=True)
class _NestedConfig:
    threshold: float
    labels: tuple[str, ...]


@dataclass(frozen=True)
class _SyntheticPipelineConfig:
    name: str
    nested: _NestedConfig
    options: dict[str, int]
    output_path: Path
    mode: _Mode = _Mode.ENABLED


def _build_real_pipeline_config() -> PipelineConfig:
    search_config = build_default_optimizer_v2_search_config()
    return PipelineConfig(
        ui=RuntimeUIConfig(),
        artifacts=ArtifactConfig(),
        preprocessing=PreprocessingConfig(
            interpolation=PreprocessingInterpolationConfig(1.0, 1.0, 1.0),
            geometry=PreprocessingGeometryConfig(1.0, 10, 1.0, 1.0, 1.0),
            kernel_execution=PreprocessingKernelExecutionConfig(
                100,
                100,
                "auto-close-if-open",
                True,
                False,
                "kernel",
            ),
        ),
        replay=RuntimeReplayConfig(None, 0.0, None, None, 1.0),
        guidance_maps=GuidanceMapConfig(),
        optimizer=OptimizerRuntimeConfig(optimizer_v2_search_config=search_config),
        random_seeds=RandomSeedConfig(None, None),
        structure_registry=StructureRegistryConfig(
            structs_referenced_dict={"OAR ref": {"PCD color": np.array([1.0, 0.0, 0.0])}},
            structs_referenced_list=("OAR ref",),
            structs_referenced_list_generalized=("OAR ref",),
            structs_referenced_list_generalized_unique_structs=("OAR ref",),
        ),
    )


class PipelineConfigSnapshotTests(unittest.TestCase):
    def test_mapping_order_does_not_change_config_fingerprint(self) -> None:
        first = _SyntheticPipelineConfig(
            name="synthetic",
            nested=_NestedConfig(threshold=1.5, labels=("a", "b")),
            options={"first": 1, "second": 2},
            output_path=Path("outputs/run"),
        )
        second = _SyntheticPipelineConfig(
            name="synthetic",
            nested=_NestedConfig(threshold=1.5, labels=("a", "b")),
            options={"second": 2, "first": 1},
            output_path=Path("outputs/run"),
        )

        self.assertEqual(
            build_pipeline_config_snapshot(first).config_sha256,
            build_pipeline_config_snapshot(second).config_sha256,
        )

    def test_snapshot_round_trip_verifies_payload(self) -> None:
        config = _SyntheticPipelineConfig(
            name="synthetic",
            nested=_NestedConfig(threshold=1.5, labels=("a", "b")),
            options={"first": 1},
            output_path=Path("outputs/run"),
        )
        snapshot = build_pipeline_config_snapshot(config)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory).joinpath("resolved_pipeline_config.json")
            write_pipeline_config_snapshot(snapshot, path)
            loaded = read_pipeline_config_snapshot(path)

        self.assertEqual(loaded, snapshot)
        self.assertEqual(loaded.config["output_path"], "outputs/run")
        self.assertEqual(loaded.config["mode"], "enabled")

    def test_snapshot_rejects_tampered_payload(self) -> None:
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            PipelineConfigSnapshot(
                config_type="synthetic.Config",
                config={"value": 2},
                config_sha256=canonical_sha256({"value": 1}),
            )

    def test_snapshot_rejects_runtime_objects(self) -> None:
        @dataclass(frozen=True)
        class _InvalidConfig:
            runtime_resource: object

        with self.assertRaisesRegex(TypeError, "unsupported config value"):
            build_pipeline_config_snapshot(_InvalidConfig(runtime_resource=object()))

    def test_scientific_snapshot_excludes_orchestration_fields(self) -> None:
        @dataclass(frozen=True)
        class _PipelineLikeConfig:
            preprocessing: str = "preprocessing"
            replay: str = "replay"
            guidance_maps: str = "guidance"
            optimizer: str = "optimizer"
            random_seeds: str = "seeds"
            mc: str = "mc"
            legacy_refs: str = "refs"
            structure_registry: str = "registry"
            bootstrap: str = "bootstrap"
            grid_preprocessing: str = "grid"
            biopsy: str = "biopsy"
            ui: str = "first-ui"
            patient_selection: tuple[str, ...] = ("P001",)

        first = _PipelineLikeConfig()
        second = _PipelineLikeConfig(ui="second-ui", patient_selection=("P002",))

        self.assertEqual(
            build_pipeline_scientific_config_snapshot(first).config_sha256,
            build_pipeline_scientific_config_snapshot(second).config_sha256,
        )

    def test_real_pipeline_config_snapshot_supports_numpy_structure_values(self) -> None:
        snapshot = build_pipeline_scientific_config_snapshot(_build_real_pipeline_config())
        rehydrated = rehydrate_pipeline_scientific_config_snapshot(snapshot)

        self.assertEqual(snapshot.config["structure_registry"]["structs_referenced_dict"]["OAR ref"]["PCD color"], [1.0, 0.0, 0.0])
        self.assertEqual(len(snapshot.config_sha256), 64)
        self.assertIsInstance(rehydrated, PipelineConfig)
        self.assertIsInstance(rehydrated.preprocessing, PreprocessingConfig)
        self.assertEqual(
            build_pipeline_scientific_config_snapshot(rehydrated).config_sha256,
            snapshot.config_sha256,
        )

    def test_rehydration_rejects_changed_scientific_field_contract(self) -> None:
        snapshot = build_pipeline_scientific_config_snapshot(_build_real_pipeline_config())
        altered_config = dict(snapshot.config)
        altered_config["unknown_scientific_field"] = {}
        altered_snapshot = PipelineConfigSnapshot(
            config_type=snapshot.config_type,
            config=altered_config,
            config_sha256=canonical_sha256(altered_config),
        )

        with self.assertRaisesRegex(ValueError, "supported contract"):
            rehydrate_pipeline_scientific_config_snapshot(altered_snapshot)

    def test_rehydration_rejects_non_pipeline_scientific_snapshot(self) -> None:
        snapshot = build_pipeline_scientific_config_snapshot(_build_real_pipeline_config())
        altered_snapshot = PipelineConfigSnapshot(
            config_type="synthetic.Config.scientific",
            config=snapshot.config,
            config_sha256=snapshot.config_sha256,
        )

        with self.assertRaisesRegex(ValueError, "not a PipelineConfig scientific snapshot"):
            rehydrate_pipeline_scientific_config_snapshot(altered_snapshot)


if __name__ == "__main__":
    unittest.main()

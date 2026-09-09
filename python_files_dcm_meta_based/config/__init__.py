"""Public configuration contracts with lazy domain imports.

Lightweight orchestration and provenance tools must be able to inspect config
snapshots without importing scientific preprocessing, visualization, or CUDA
dependencies. Concrete PipelineConfig classes remain available through the same
public names and are loaded only when requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_PIPELINE_EXPORTS = frozenset(
    {
        "ArtifactConfig",
        "BiopsyGeometryConfig",
        "BiopsyRuntimeConfig",
        "FROZEN_PREPROCESSED_BUNDLE_CONFIG_METADATA_KEY",
        "FrozenPreprocessedBundleConfig",
        "GridPreprocessingConfig",
        "GuidanceMapConfig",
        "LegacyReferenceConfig",
        "MCCountsConfig",
        "MCDebugConfig",
        "MCOutputDumpConfig",
        "MCPrepConfig",
        "MCSimulationCoreConfig",
        "MCTissueClassificationConfig",
        "MCVisualizationConfig",
        "MonteCarloConfig",
        "OptimizerV1RuntimeConfig",
        "OptimizerV2CapacityConfig",
        "OptimizerV2DiagnosticsConfig",
        "OptimizerV2PlotlyExportConfig",
        "OptimizerV2RenderConfig",
        "OptimizerV2RuntimeConfig",
        "OptimizerRuntimeConfig",
        "OutputValidationConfig",
        "PatientScientificRunnerExecutionConfig",
        "PatientRunnerValidationHookConfig",
        "PipelineConfig",
        "PreprocessingConfig",
        "PreprocessingDebugConfig",
        "PreprocessingGeometryConfig",
        "PreprocessingInterpolationConfig",
        "PreprocessingKernelExecutionConfig",
        "RandomSeedConfig",
        "RuntimeReplayConfig",
        "RuntimeUIConfig",
        "SamplingClassificationConfig",
        "SimulatedBiopsyConfig",
        "StructureRegistryConfig",
        "ValidationSidecarConfig",
    }
)

_BOOTSTRAP_EXPORTS = frozenset(
    {
        "PatientBootstrapConfig",
        "SimulatedBiopsyBootstrapPolicy",
        "StructureContourPolicy",
        "StructureDataRemovalPolicy",
    }
)

_SNAPSHOT_EXPORTS = frozenset(
    {
        "PIPELINE_CONFIG_SNAPSHOT_SCHEMA_VERSION",
        "PIPELINE_SCIENTIFIC_CONFIG_FIELD_NAMES",
        "PipelineConfigSnapshot",
        "build_pipeline_config_snapshot",
        "build_pipeline_scientific_config_snapshot",
        "canonical_json_value",
        "canonical_sha256",
        "read_pipeline_config_snapshot",
        "write_pipeline_config_snapshot",
    }
)


def __getattr__(name: str) -> Any:
    if name in _BOOTSTRAP_EXPORTS:
        return getattr(import_module(".bootstrap", __name__), name)
    if name in _PIPELINE_EXPORTS:
        return getattr(import_module(".pipeline", __name__), name)
    if name in _SNAPSHOT_EXPORTS:
        return getattr(import_module(".snapshots", __name__), name)
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))


__all__ = sorted((*_BOOTSTRAP_EXPORTS, *_PIPELINE_EXPORTS, *_SNAPSHOT_EXPORTS))
"""Dependency graph and pathway presets for patient-runner science."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from .contracts import PatientStageName


class PatientScientificPathwayName(str, Enum):
    """Named scientific graph slices selected intentionally by run config."""

    ANATOMICAL_QA = "anatomical_qa"
    BIOPSY_PREPROCESSING_SHADOW = "biopsy_preprocessing_shadow"
    OPTIMIZATION_SHADOW = "optimization_shadow"
    CURRENT_DOSIMETRY_SHADOW = "current_dosimetry_shadow"
    FULL_CURRENT_PIPELINE_SHADOW = "full_current_pipeline_shadow"


@dataclass(frozen=True, slots=True)
class PatientScientificStageDependency:
    """Dependency declaration for one patient-runner scientific stage node."""

    stage_name: PatientStageName | str
    required_stage_names: Sequence[PatientStageName | str] = ()
    summary: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_name", PatientStageName(self.stage_name))
        object.__setattr__(
            self,
            "required_stage_names",
            tuple(PatientStageName(stage_name) for stage_name in self.required_stage_names),
        )
        object.__setattr__(self, "summary", str(self.summary).strip())
        object.__setattr__(self, "metadata", dict(self.metadata))


# Full intended graph-node order. Some nodes are not standalone adapters yet.
DEFAULT_PATIENT_SCIENTIFIC_GRAPH_ORDER = (
    PatientStageName.GRID_PREPROCESSING,
    PatientStageName.ANATOMICAL_PREPROCESSING,
    PatientStageName.PREPROCESSING,
    PatientStageName.TRANSFORM_GENERATION,
    PatientStageName.OPTIMIZATION,
    PatientStageName.SIMULATED_BIOPSY_FINALIZATION,
    PatientStageName.SAMPLING_CLASSIFICATION,
    PatientStageName.MC_PREP,
    PatientStageName.MC_SIMULATION,
    PatientStageName.GUIDANCE,
)

# Current runner adapter order used by build_patient_scientific_stages.
DEFAULT_PATIENT_SCIENTIFIC_EXECUTABLE_STAGE_ORDER = (
    PatientStageName.GRID_PREPROCESSING,
    PatientStageName.ANATOMICAL_PREPROCESSING,
    PatientStageName.PREPROCESSING,
    PatientStageName.TRANSFORM_GENERATION,
    PatientStageName.OPTIMIZATION,
    PatientStageName.MC_PREP,
    PatientStageName.MC_SIMULATION,
    PatientStageName.GUIDANCE,
)

# Dependencies among currently executable runner adapters.
DEFAULT_PATIENT_SCIENTIFIC_STAGE_DEPENDENCIES = (
    PatientScientificStageDependency(
        stage_name=PatientStageName.GRID_PREPROCESSING,
        summary="Builds dose/MR grid artifacts used by downstream anatomical and MC work.",
    ),
    PatientScientificStageDependency(
        stage_name=PatientStageName.ANATOMICAL_PREPROCESSING,
        required_stage_names=(PatientStageName.GRID_PREPROCESSING,),
        summary="Builds patient anatomical structure state after prerequisite grid artifacts.",
    ),
    PatientScientificStageDependency(
        stage_name=PatientStageName.PREPROCESSING,
        required_stage_names=(PatientStageName.ANATOMICAL_PREPROCESSING,),
        summary="Builds biopsy-facing preprocessing state after anatomical structures exist.",
    ),
    PatientScientificStageDependency(
        stage_name=PatientStageName.TRANSFORM_GENERATION,
        required_stage_names=(PatientStageName.PREPROCESSING,),
        summary="Builds transform-bank samples before optimizer and MC transform consumers.",
    ),
    PatientScientificStageDependency(
        stage_name=PatientStageName.OPTIMIZATION,
        required_stage_names=(PatientStageName.TRANSFORM_GENERATION,),
        summary="Runs optimizer stages after transform samples are available for search behavior.",
    ),
    PatientScientificStageDependency(
        stage_name=PatientStageName.MC_PREP,
        required_stage_names=(PatientStageName.TRANSFORM_GENERATION,),
        summary="Applies MC transform state after transform-bank generation.",
    ),
    PatientScientificStageDependency(
        stage_name=PatientStageName.MC_SIMULATION,
        required_stage_names=(PatientStageName.MC_PREP,),
        summary="Runs current MC/dosimetry simulation after MC transform preparation.",
    ),
    PatientScientificStageDependency(
        stage_name=PatientStageName.GUIDANCE,
        required_stage_names=(PatientStageName.ANATOMICAL_PREPROCESSING, PatientStageName.PREPROCESSING),
        summary="Builds guidance outputs from selected anatomical and biopsy products.",
    ),
)

# Dependencies among full graph nodes, including planned adapter splits.
DEFAULT_PATIENT_SCIENTIFIC_GRAPH_STAGE_DEPENDENCIES = (
    PatientScientificStageDependency(
        stage_name=PatientStageName.GRID_PREPROCESSING,
        summary="Builds dose/MR grid artifacts used by downstream anatomical and MC work.",
    ),
    PatientScientificStageDependency(
        stage_name=PatientStageName.ANATOMICAL_PREPROCESSING,
        required_stage_names=(PatientStageName.GRID_PREPROCESSING,),
        summary="Builds patient anatomical structure state after prerequisite grid artifacts.",
    ),
    PatientScientificStageDependency(
        stage_name=PatientStageName.PREPROCESSING,
        required_stage_names=(PatientStageName.ANATOMICAL_PREPROCESSING,),
        summary="Builds biopsy-facing preprocessing state after anatomical structures exist.",
    ),
    PatientScientificStageDependency(
        stage_name=PatientStageName.TRANSFORM_GENERATION,
        required_stage_names=(PatientStageName.PREPROCESSING,),
        summary="Builds transform-bank samples before optimizer and MC transform consumers.",
    ),
    PatientScientificStageDependency(
        stage_name=PatientStageName.OPTIMIZATION,
        required_stage_names=(PatientStageName.TRANSFORM_GENERATION,),
        summary="Runs optimizer stages after transform samples are available for search behavior.",
    ),
    PatientScientificStageDependency(
        stage_name=PatientStageName.SIMULATED_BIOPSY_FINALIZATION,
        required_stage_names=(PatientStageName.OPTIMIZATION,),
        summary="Finalizes optimized simulated biopsy geometry after optimizer stages.",
        metadata={"adapter_status": "planned"},
    ),
    PatientScientificStageDependency(
        stage_name=PatientStageName.SAMPLING_CLASSIFICATION,
        required_stage_names=(PatientStageName.SIMULATED_BIOPSY_FINALIZATION,),
        summary="Stores sampled-biopsy products and classification fragments after finalized biopsy geometry.",
        metadata={"adapter_status": "planned_split_from_preprocessing"},
    ),
    PatientScientificStageDependency(
        stage_name=PatientStageName.MC_PREP,
        required_stage_names=(PatientStageName.SAMPLING_CLASSIFICATION,),
        summary="Builds MC transform state from finalized biopsy and anatomical structures.",
    ),
    PatientScientificStageDependency(
        stage_name=PatientStageName.MC_SIMULATION,
        required_stage_names=(PatientStageName.MC_PREP,),
        summary="Runs current MC/dosimetry simulation after MC transform preparation.",
    ),
    PatientScientificStageDependency(
        stage_name=PatientStageName.GUIDANCE,
        required_stage_names=(PatientStageName.ANATOMICAL_PREPROCESSING, PatientStageName.PREPROCESSING),
        summary="Builds guidance outputs from selected anatomical and biopsy products.",
    ),
)

# Current executable adapter slices for each named pathway.
DEFAULT_PATIENT_SCIENTIFIC_PATHWAYS = {
    PatientScientificPathwayName.ANATOMICAL_QA: (
        PatientStageName.GRID_PREPROCESSING,
        PatientStageName.ANATOMICAL_PREPROCESSING,
    ),
    PatientScientificPathwayName.BIOPSY_PREPROCESSING_SHADOW: (
        PatientStageName.GRID_PREPROCESSING,
        PatientStageName.ANATOMICAL_PREPROCESSING,
        PatientStageName.PREPROCESSING,
    ),
    PatientScientificPathwayName.OPTIMIZATION_SHADOW: (
        PatientStageName.GRID_PREPROCESSING,
        PatientStageName.ANATOMICAL_PREPROCESSING,
        PatientStageName.PREPROCESSING,
        PatientStageName.TRANSFORM_GENERATION,
        PatientStageName.OPTIMIZATION,
    ),
    PatientScientificPathwayName.CURRENT_DOSIMETRY_SHADOW: (
        PatientStageName.GRID_PREPROCESSING,
        PatientStageName.ANATOMICAL_PREPROCESSING,
        PatientStageName.PREPROCESSING,
        PatientStageName.TRANSFORM_GENERATION,
        PatientStageName.OPTIMIZATION,
        PatientStageName.MC_PREP,
        PatientStageName.MC_SIMULATION,
    ),
    PatientScientificPathwayName.FULL_CURRENT_PIPELINE_SHADOW: DEFAULT_PATIENT_SCIENTIFIC_EXECUTABLE_STAGE_ORDER,
}

# Full graph-node slices for each named pathway.
DEFAULT_PATIENT_SCIENTIFIC_GRAPH_PATHWAYS = {
    PatientScientificPathwayName.ANATOMICAL_QA: (
        PatientStageName.GRID_PREPROCESSING,
        PatientStageName.ANATOMICAL_PREPROCESSING,
    ),
    PatientScientificPathwayName.BIOPSY_PREPROCESSING_SHADOW: (
        PatientStageName.GRID_PREPROCESSING,
        PatientStageName.ANATOMICAL_PREPROCESSING,
        PatientStageName.PREPROCESSING,
    ),
    PatientScientificPathwayName.OPTIMIZATION_SHADOW: (
        PatientStageName.GRID_PREPROCESSING,
        PatientStageName.ANATOMICAL_PREPROCESSING,
        PatientStageName.PREPROCESSING,
        PatientStageName.TRANSFORM_GENERATION,
        PatientStageName.OPTIMIZATION,
    ),
    PatientScientificPathwayName.CURRENT_DOSIMETRY_SHADOW: (
        PatientStageName.GRID_PREPROCESSING,
        PatientStageName.ANATOMICAL_PREPROCESSING,
        PatientStageName.PREPROCESSING,
        PatientStageName.TRANSFORM_GENERATION,
        PatientStageName.OPTIMIZATION,
        PatientStageName.SIMULATED_BIOPSY_FINALIZATION,
        PatientStageName.SAMPLING_CLASSIFICATION,
        PatientStageName.MC_PREP,
        PatientStageName.MC_SIMULATION,
    ),
    PatientScientificPathwayName.FULL_CURRENT_PIPELINE_SHADOW: DEFAULT_PATIENT_SCIENTIFIC_GRAPH_ORDER,
}

_DEPENDENCIES_BY_STAGE_NAME = {
    dependency.stage_name: dependency for dependency in DEFAULT_PATIENT_SCIENTIFIC_STAGE_DEPENDENCIES
}

_GRAPH_DEPENDENCIES_BY_STAGE_NAME = {
    dependency.stage_name: dependency for dependency in DEFAULT_PATIENT_SCIENTIFIC_GRAPH_STAGE_DEPENDENCIES
}


def resolve_patient_scientific_stage_names(
    stage_names: Sequence[PatientStageName | str],
    *,
    allow_duplicates: bool = False,
) -> tuple[PatientStageName, ...]:
    """Validate and normalize patient scientific stage names."""
    resolved_names = tuple(PatientStageName(stage_name) for stage_name in stage_names)
    if not allow_duplicates and len(set(resolved_names)) != len(resolved_names):
        raise ValueError("stage_names cannot contain duplicates")
    return resolved_names


def resolve_patient_scientific_pathway_name(
    pathway_name: PatientScientificPathwayName | str,
) -> PatientScientificPathwayName:
    """Validate and normalize one scientific pathway name."""
    return PatientScientificPathwayName(pathway_name)


def patient_scientific_pathway_stage_names(
    pathway_name: PatientScientificPathwayName | str,
) -> tuple[PatientStageName, ...]:
    """Return currently executable stage adapters for a named pathway."""
    return DEFAULT_PATIENT_SCIENTIFIC_PATHWAYS[resolve_patient_scientific_pathway_name(pathway_name)]


def patient_scientific_pathway_graph_stage_names(
    pathway_name: PatientScientificPathwayName | str,
) -> tuple[PatientStageName, ...]:
    """Return the full split graph-node slice for a named scientific pathway."""
    return DEFAULT_PATIENT_SCIENTIFIC_GRAPH_PATHWAYS[resolve_patient_scientific_pathway_name(pathway_name)]


def executable_patient_scientific_pathway_stage_names(
    pathway_name: PatientScientificPathwayName | str,
    *,
    satisfied_stage_names: Sequence[PatientStageName | str] = (),
) -> tuple[PatientStageName, ...]:
    """Return pathway stages that still need execution after satisfied stages."""
    satisfied_names = set(resolve_patient_scientific_stage_names(satisfied_stage_names))
    return tuple(
        stage_name
        for stage_name in patient_scientific_pathway_stage_names(pathway_name)
        if stage_name not in satisfied_names
    )


def validate_patient_scientific_stage_dependencies(
    stage_names: Sequence[PatientStageName | str],
    *,
    satisfied_stage_names: Sequence[PatientStageName | str] = (),
) -> tuple[PatientStageName, ...]:
    """Validate that requested executable scientific stages obey dependency order.

    ``satisfied_stage_names`` is for controlled states such as loaded
    preprocessed bundles where upstream products are already present and should
    not be rerun.
    """
    return _validate_stage_dependencies(
        stage_names,
        dependencies_by_stage_name=_DEPENDENCIES_BY_STAGE_NAME,
        satisfied_stage_names=satisfied_stage_names,
    )


def validate_patient_scientific_graph_dependencies(
    stage_names: Sequence[PatientStageName | str],
    *,
    satisfied_stage_names: Sequence[PatientStageName | str] = (),
) -> tuple[PatientStageName, ...]:
    """Validate a full split scientific graph-node slice."""
    return _validate_stage_dependencies(
        stage_names,
        dependencies_by_stage_name=_GRAPH_DEPENDENCIES_BY_STAGE_NAME,
        satisfied_stage_names=satisfied_stage_names,
    )


def _validate_stage_dependencies(
    stage_names: Sequence[PatientStageName | str],
    *,
    dependencies_by_stage_name: Mapping[PatientStageName, PatientScientificStageDependency],
    satisfied_stage_names: Sequence[PatientStageName | str] = (),
) -> tuple[PatientStageName, ...]:
    resolved_stage_names = resolve_patient_scientific_stage_names(stage_names)
    satisfied_names = set(resolve_patient_scientific_stage_names(satisfied_stage_names))
    seen_names = set(satisfied_names)
    missing_by_stage: dict[PatientStageName, tuple[PatientStageName, ...]] = {}

    for stage_name in resolved_stage_names:
        dependency = dependencies_by_stage_name.get(stage_name)
        if dependency is not None:
            missing_required_names = tuple(
                required_stage_name
                for required_stage_name in dependency.required_stage_names
                if required_stage_name not in seen_names
            )
            if missing_required_names:
                missing_by_stage[stage_name] = missing_required_names
        seen_names.add(stage_name)

    if missing_by_stage:
        missing_summary = "; ".join(
            f"{stage_name.value} requires {', '.join(required.value for required in missing_required_names)}"
            for stage_name, missing_required_names in missing_by_stage.items()
        )
        raise ValueError("invalid patient scientific stage dependency selection: " + missing_summary)

    return resolved_stage_names


def validate_patient_scientific_pathway_dependencies(
    pathway_name: PatientScientificPathwayName | str,
    *,
    satisfied_stage_names: Sequence[PatientStageName | str] = (),
) -> tuple[PatientStageName, ...]:
    """Validate and return the currently executable slice for a named pathway."""
    stage_names = patient_scientific_pathway_stage_names(pathway_name)
    return validate_patient_scientific_stage_dependencies(
        stage_names,
        satisfied_stage_names=satisfied_stage_names,
    )


def validate_patient_scientific_pathway_graph_dependencies(
    pathway_name: PatientScientificPathwayName | str,
    *,
    satisfied_stage_names: Sequence[PatientStageName | str] = (),
) -> tuple[PatientStageName, ...]:
    """Validate and return the full split graph slice for a named pathway."""
    stage_names = patient_scientific_pathway_graph_stage_names(pathway_name)
    return validate_patient_scientific_graph_dependencies(
        stage_names,
        satisfied_stage_names=satisfied_stage_names,
    )


def summarize_patient_scientific_dependency_graph() -> tuple[dict[str, Any], ...]:
    """Return JSON-ready summaries of the split scientific stage graph."""
    return tuple(
        {
            "stage_name": dependency.stage_name.value,
            "required_stage_names": tuple(stage_name.value for stage_name in dependency.required_stage_names),
            "summary": dependency.summary,
            "metadata": dict(dependency.metadata),
        }
        for dependency in DEFAULT_PATIENT_SCIENTIFIC_GRAPH_STAGE_DEPENDENCIES
    )


def summarize_patient_scientific_pathways() -> tuple[dict[str, Any], ...]:
    """Return JSON-ready summaries of named scientific pathway presets."""
    return tuple(
        {
            "pathway_name": pathway_name.value,
            "stage_names": tuple(stage_name.value for stage_name in stage_names),
            "graph_stage_names": tuple(
                stage_name.value for stage_name in patient_scientific_pathway_graph_stage_names(pathway_name)
            ),
        }
        for pathway_name, stage_names in DEFAULT_PATIENT_SCIENTIFIC_PATHWAYS.items()
    )
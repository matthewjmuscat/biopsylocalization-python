"""Scientific tranche recipes for opt-in patient-runner orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from .contracts import PatientStageName
from .runner import PatientStage
from .scientific_config import PatientRunnerScientificConfig
from .scientific_stages import build_patient_scientific_stages


class PatientScientificTrancheName(str, Enum):
    """Stable names for patient-runner scientific orchestration tranches."""

    COMPATIBILITY_BOOTSTRAP = "compatibility_bootstrap"
    GRID_PREPROCESSING = "grid_preprocessing"
    ANATOMICAL_PREPROCESSING = "anatomical_preprocessing"
    BIOPSY_PREPROCESSING = "biopsy_preprocessing"
    PRE_OPTIMIZER_TRANSFORM_AND_OPTIMIZATION = "pre_optimizer_transform_and_optimization"
    POST_OPTIMIZER_BIOPSY_REALIZATION = "post_optimizer_biopsy_realization"
    SAMPLING_CLASSIFICATION = "sampling_classification"
    MC_PREP_AND_SIMULATION = "mc_prep_and_simulation"
    OUTPUT_GUIDANCE_ASSEMBLY_PARITY = "output_guidance_assembly_parity"


@dataclass(frozen=True, slots=True)
class PatientScientificTranche:
    """One orchestration recipe entry for patient-runner scientific shadow work."""

    tranche_name: PatientScientificTrancheName | str
    display_name: str
    summary: str
    planned_surfaces: Sequence[str] = ()
    implemented_stage_names: Sequence[PatientStageName | str] = ()
    owns_patient_discovery: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tranche_name", PatientScientificTrancheName(self.tranche_name))
        object.__setattr__(self, "display_name", _non_empty_string(self.display_name, "display_name"))
        object.__setattr__(self, "summary", _non_empty_string(self.summary, "summary"))
        object.__setattr__(
            self,
            "planned_surfaces",
            tuple(_non_empty_string(value, "planned_surfaces item") for value in self.planned_surfaces),
        )
        object.__setattr__(
            self,
            "implemented_stage_names",
            tuple(PatientStageName(stage_name) for stage_name in self.implemented_stage_names),
        )
        object.__setattr__(self, "owns_patient_discovery", bool(self.owns_patient_discovery))
        if self.owns_patient_discovery:
            raise ValueError("patient-runner scientific tranches must not own patient discovery")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def has_implemented_stages(self) -> bool:
        return bool(self.implemented_stage_names)


def _non_empty_string(value: Any, field_name: str) -> str:
    resolved_value = str(value).strip()
    if resolved_value == "":
        raise ValueError(f"{field_name} cannot be empty")
    return resolved_value


DEFAULT_PATIENT_SCIENTIFIC_TRANCHES = (
    PatientScientificTranche(
        tranche_name=PatientScientificTrancheName.COMPATIBILITY_BOOTSTRAP,
        display_name="Compatibility Bootstrap",
        summary="Consumes discovered patient inputs and creates one-patient runtime/reference state.",
        planned_surfaces=(
            "discovered patient case inputs",
            "legacy key names",
            "one-patient runtime/reference/info state",
        ),
    ),
    PatientScientificTranche(
        tranche_name=PatientScientificTrancheName.GRID_PREPROCESSING,
        display_name="Grid Preprocessing",
        summary="Builds dose/MR grid artifacts before anatomical structure processing that depends on them.",
        planned_surfaces=(
            "dose-grid runtime objects",
            "MR ADC input normalization",
            "MR ADC grid runtime objects",
            "patient-local lattice/grid/KD-tree artifacts",
        ),
        implemented_stage_names=(PatientStageName.GRID_PREPROCESSING,),
    ),
    PatientScientificTranche(
        tranche_name=PatientScientificTrancheName.ANATOMICAL_PREPROCESSING,
        display_name="Anatomical Preprocessing",
        summary="Processes patient anatomical structures after prerequisite grid artifacts are available.",
        planned_surfaces=(
            "raw contour pulling",
            "selected and unique structures",
            "OAR/prostate, rectum, urethra, and DIL preprocessing",
            "prostate-only MR ADC structure summary",
        ),
        implemented_stage_names=(PatientStageName.ANATOMICAL_PREPROCESSING,),
    ),
    PatientScientificTranche(
        tranche_name=PatientScientificTrancheName.BIOPSY_PREPROCESSING,
        display_name="Biopsy Preprocessing",
        summary="Runs currently available biopsy-facing preprocessing adapters.",
        planned_surfaces=(
            "real-biopsy geometry processing",
            "simulated-biopsy preparation",
            "simulated-biopsy planning",
            "uncertainty attachment",
            "realized targeting",
        ),
        implemented_stage_names=(PatientStageName.PREPROCESSING,),
    ),
    PatientScientificTranche(
        tranche_name=PatientScientificTrancheName.PRE_OPTIMIZER_TRANSFORM_AND_OPTIMIZATION,
        display_name="Pre-Optimizer Transform And Optimization",
        summary="Runs transform-bank generation and patient optimizer stages.",
        planned_surfaces=(
            "transform-bank generation",
            "optimizer-v1",
            "optimizer-v2",
        ),
        implemented_stage_names=(PatientStageName.TRANSFORM_GENERATION, PatientStageName.OPTIMIZATION),
    ),
    PatientScientificTranche(
        tranche_name=PatientScientificTrancheName.POST_OPTIMIZER_BIOPSY_REALIZATION,
        display_name="Post-Optimizer Biopsy Realization",
        summary="Finalizes simulated biopsies and post-optimizer biopsy checks after optimizer-v2.",
        planned_surfaces=(
            "simulated-biopsy finalization",
            "planned-vs-realized centroid validation",
            "post-optimizer biopsy annotations",
        ),
        implemented_stage_names=(PatientStageName.SIMULATED_BIOPSY_FINALIZATION,),
    ),
    PatientScientificTranche(
        tranche_name=PatientScientificTrancheName.SAMPLING_CLASSIFICATION,
        display_name="Sampling And Classification",
        summary="Stores sampled-biopsy outputs and classification fragments before MC simulation.",
        planned_surfaces=(
            "sampled-biopsy processing",
            "optimizer-v2 sampling audit annotation",
            "double-sextant sample-point fragments",
            "run-level per-voxel double-sextant assembly",
        ),
    ),
    PatientScientificTranche(
        tranche_name=PatientScientificTrancheName.MC_PREP_AND_SIMULATION,
        display_name="MC Prep And Simulation",
        summary="Runs MC prep transforms and patient MC simulation stages immediately before MC outputs.",
        planned_surfaces=(
            "BX-only/self transforms",
            "relative-structure transforms",
            "convex containment/dose simulation",
            "MR ADC localization",
            "downstream MC annotations",
        ),
        implemented_stage_names=(PatientStageName.MC_PREP, PatientStageName.MC_SIMULATION),
    ),
    PatientScientificTranche(
        tranche_name=PatientScientificTrancheName.OUTPUT_GUIDANCE_ASSEMBLY_PARITY,
        display_name="Output Guidance Assembly And Parity",
        summary="Writes patient outputs and runs guidance, assembly, and parity surfaces outside science modules.",
        planned_surfaces=(
            "patient artifact writing",
            "guidance-map precompute/render handoff",
            "cohort table assembly",
            "post-run parity",
        ),
        implemented_stage_names=(PatientStageName.GUIDANCE,),
    ),
)

DEFAULT_PATIENT_SCIENTIFIC_TRANCHE_ORDER = tuple(
    tranche.tranche_name for tranche in DEFAULT_PATIENT_SCIENTIFIC_TRANCHES
)

_DEFAULT_TRANCHES_BY_NAME = {
    tranche.tranche_name: tranche for tranche in DEFAULT_PATIENT_SCIENTIFIC_TRANCHES
}


def default_patient_scientific_tranches() -> tuple[PatientScientificTranche, ...]:
    """Return the canonical tranche recipes in legacy-compatible order."""
    return DEFAULT_PATIENT_SCIENTIFIC_TRANCHES


def resolve_patient_scientific_tranche_names(
    tranche_names: Sequence[PatientScientificTrancheName | str] = DEFAULT_PATIENT_SCIENTIFIC_TRANCHE_ORDER,
) -> tuple[PatientScientificTrancheName, ...]:
    """Validate and normalize patient scientific tranche names."""
    resolved_names = tuple(PatientScientificTrancheName(tranche_name) for tranche_name in tranche_names)
    if len(set(resolved_names)) != len(resolved_names):
        raise ValueError("tranche_names cannot contain duplicates")
    return resolved_names


def get_patient_scientific_tranche(
    tranche_name: PatientScientificTrancheName | str,
) -> PatientScientificTranche:
    """Return one canonical tranche recipe by name."""
    resolved_name = PatientScientificTrancheName(tranche_name)
    return _DEFAULT_TRANCHES_BY_NAME[resolved_name]


def iter_patient_scientific_tranches(
    tranche_names: Sequence[PatientScientificTrancheName | str] = DEFAULT_PATIENT_SCIENTIFIC_TRANCHE_ORDER,
) -> tuple[PatientScientificTranche, ...]:
    """Return selected tranche recipes in requested order."""
    return tuple(
        get_patient_scientific_tranche(tranche_name)
        for tranche_name in resolve_patient_scientific_tranche_names(tranche_names)
    )


def patient_scientific_tranche_stage_names(
    tranche_names: Sequence[PatientScientificTrancheName | str] = DEFAULT_PATIENT_SCIENTIFIC_TRANCHE_ORDER,
) -> tuple[PatientStageName, ...]:
    """Return currently implemented stage names implied by the selected tranches."""
    stage_names: list[PatientStageName] = []
    for tranche in iter_patient_scientific_tranches(tranche_names):
        for stage_name in tranche.implemented_stage_names:
            if stage_name not in stage_names:
                stage_names.append(stage_name)
    return tuple(stage_names)


def build_patient_scientific_stages_for_tranches(
    scientific_config: PatientRunnerScientificConfig,
    *,
    tranche_names: Sequence[PatientScientificTrancheName | str] = DEFAULT_PATIENT_SCIENTIFIC_TRANCHE_ORDER,
    include_artifact_writing: bool = True,
    satisfied_stage_names: Sequence[PatientStageName | str] = (),
    validate_dependencies: bool = True,
) -> tuple[PatientStage, ...]:
    """Build currently implemented patient stages from selected tranche recipes."""
    stage_order = patient_scientific_tranche_stage_names(tranche_names)
    return build_patient_scientific_stages(
        scientific_config,
        include_artifact_writing=include_artifact_writing,
        stage_order=stage_order,
        satisfied_stage_names=satisfied_stage_names,
        validate_dependencies=validate_dependencies,
    )


def summarize_patient_scientific_tranches(
    tranche_names: Sequence[PatientScientificTrancheName | str] = DEFAULT_PATIENT_SCIENTIFIC_TRANCHE_ORDER,
) -> tuple[dict[str, Any], ...]:
    """Return JSON-ready tranche summaries for docs, manifests, or validation logs."""
    return tuple(_tranche_summary(tranche) for tranche in iter_patient_scientific_tranches(tranche_names))


def _tranche_summary(tranche: PatientScientificTranche) -> dict[str, Any]:
    return {
        "tranche_name": tranche.tranche_name.value,
        "display_name": tranche.display_name,
        "summary": tranche.summary,
        "planned_surfaces": tuple(tranche.planned_surfaces),
        "implemented_stage_names": tuple(stage_name.value for stage_name in tranche.implemented_stage_names),
        "owns_patient_discovery": tranche.owns_patient_discovery,
        "metadata": dict(tranche.metadata),
    }


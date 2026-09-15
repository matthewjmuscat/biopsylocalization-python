"""Typed policy for the existing independent normal uncertainty preparation.

Component magnitudes remain in StructureRegistryConfig. This contract controls
their existing quadrature combination, not new distributions or correlations.
"""

from dataclasses import dataclass
from typing import Literal, get_args

BiopsyVariationMode = Literal["Per biopsy max", "Per biopsy mean", "Default only"]
AnatomyVariationMode = Literal["Default only"]
QuadratureInterpretation = Literal["sigma", "two sigma"]


@dataclass(frozen=True)
class UncertaintyPreparationConfig:
    """Production policy shared by main and workers; no cohort-derived modes.

    ``two sigma`` halves quadrature-combined deviations. Means retain the legacy
    quadrature calculation. Biopsy variation uses real/planned geometry.
    """

    biopsy_variation_uncertainty_setting: BiopsyVariationMode = "Per biopsy mean"
    non_biopsy_variation_uncertainty_setting: AnatomyVariationMode = "Default only"
    use_added_in_quad_errors_as: QuadratureInterpretation = "two sigma"

    def __post_init__(self) -> None:
        for name, domain in (
            ("biopsy_variation_uncertainty_setting", BiopsyVariationMode),
            ("non_biopsy_variation_uncertainty_setting", AnatomyVariationMode),
            ("use_added_in_quad_errors_as", QuadratureInterpretation),
        ):
            if getattr(self, name) not in get_args(domain):
                raise ValueError(f"{name} must be one of {get_args(domain)}")

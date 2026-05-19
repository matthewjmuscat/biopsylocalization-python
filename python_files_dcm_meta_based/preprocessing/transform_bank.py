"""Shared transform-bank contracts and accessors.

This module sits outside optimizer-v2 because the stored transform draws are
shared by optimizer scoring and downstream MC surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple


OPTIMIZER_V2_TRANSFORM_SAMPLE_COUNT_KEY = "Num optimizer v2 transform samples"
STOCHASTIC_TARGETING_TRANSFORM_SAMPLE_COUNT_KEY = "Num stochastic targeting transform samples"
MAX_GENERATED_TRANSFORM_SAMPLES_KEY = "Max of generated transform samples"

MC_NORMAL_DILATION_SAMPLES_KEY = "MC data: Generated normal dist random samples dilations arr"
MC_NORMAL_ROTATION_SAMPLES_KEY = "MC data: Generated normal dist random samples rotations arr"
MC_NORMAL_TRANSLATION_SAMPLES_KEY = "MC data: Generated normal dist random samples arr"
MC_BX_NEEDLE_COMPARTMENT_DISTANCE_SAMPLES_KEY = (
    "MC data: Generated uniform dist (biopsy needle compartment) random distance (z_needle) samples arr"
)


@dataclass(frozen=True)
class SharedTransformBankPrefix:
    """A leading trial prefix from one structure's stored transform bank.

    Arrays are returned in the same backend and dtype they were stored with so
    callers do not accidentally trigger host-device copies here.
    """

    dilation_samples: Any
    rotation_samples: Any
    translation_samples: Any
    requested_num_trials: int
    available_num_trials: int
    needle_compartment_distance_samples: Optional[Any] = None


def resolve_required_generated_transform_samples(
    mc_info: Mapping[str, Any],
    num_mc_containment_simulations_input: int,
    num_mc_dose_simulations_input: int,
    num_mc_mr_simulations_input: int,
) -> Tuple[int, int]:
    """Resolve the shared transform-bank ceiling needed by current runtime settings."""
    _validate_non_negative_count(num_mc_containment_simulations_input, "num_mc_containment_simulations_input")
    _validate_non_negative_count(num_mc_dose_simulations_input, "num_mc_dose_simulations_input")
    _validate_non_negative_count(num_mc_mr_simulations_input, "num_mc_mr_simulations_input")

    max_num_mc_simulations = max(
        num_mc_dose_simulations_input,
        num_mc_containment_simulations_input,
        num_mc_mr_simulations_input,
    )
    num_optimizer_v2_transform_samples = int(mc_info.get(OPTIMIZER_V2_TRANSFORM_SAMPLE_COUNT_KEY) or 0)
    num_stochastic_targeting_transform_samples = int(
        mc_info.get(STOCHASTIC_TARGETING_TRANSFORM_SAMPLE_COUNT_KEY) or 0
    )
    max_generated_transform_samples = max(
        max_num_mc_simulations,
        num_optimizer_v2_transform_samples,
        num_stochastic_targeting_transform_samples,
    )

    return max_num_mc_simulations, max_generated_transform_samples


def get_structure_transform_bank_prefix(
    structure_dict: Mapping[str, Any],
    num_trials: int,
) -> SharedTransformBankPrefix:
    """Return a leading prefix of one structure's shared transform draws."""
    _validate_non_negative_count(num_trials, "num_trials")

    available_num_trials = _resolve_consistent_available_num_trials(
        structure_dict,
        (
            MC_NORMAL_DILATION_SAMPLES_KEY,
            MC_NORMAL_ROTATION_SAMPLES_KEY,
            MC_NORMAL_TRANSLATION_SAMPLES_KEY,
        ),
    )

    return SharedTransformBankPrefix(
        dilation_samples=_slice_trial_prefix(
            structure_dict,
            MC_NORMAL_DILATION_SAMPLES_KEY,
            num_trials,
            available_num_trials,
        ),
        rotation_samples=_slice_trial_prefix(
            structure_dict,
            MC_NORMAL_ROTATION_SAMPLES_KEY,
            num_trials,
            available_num_trials,
        ),
        translation_samples=_slice_trial_prefix(
            structure_dict,
            MC_NORMAL_TRANSLATION_SAMPLES_KEY,
            num_trials,
            available_num_trials,
        ),
        requested_num_trials=num_trials,
        available_num_trials=available_num_trials,
    )


def get_biopsy_transform_bank_prefix(
    biopsy_structure_dict: Mapping[str, Any],
    num_trials: int,
    require_needle_compartment_shift: bool = False,
) -> SharedTransformBankPrefix:
    """Return a leading prefix of one biopsy structure's shared transform draws."""
    transform_bank_prefix = get_structure_transform_bank_prefix(biopsy_structure_dict, num_trials)

    needle_compartment_distance_samples = None
    if require_needle_compartment_shift or biopsy_structure_dict.get(MC_BX_NEEDLE_COMPARTMENT_DISTANCE_SAMPLES_KEY) is not None:
        available_num_trials = _resolve_consistent_available_num_trials(
            biopsy_structure_dict,
            (MC_BX_NEEDLE_COMPARTMENT_DISTANCE_SAMPLES_KEY,),
            expected_num_trials=transform_bank_prefix.available_num_trials,
        )
        needle_compartment_distance_samples = _slice_trial_prefix(
            biopsy_structure_dict,
            MC_BX_NEEDLE_COMPARTMENT_DISTANCE_SAMPLES_KEY,
            num_trials,
            available_num_trials,
        )

    return SharedTransformBankPrefix(
        dilation_samples=transform_bank_prefix.dilation_samples,
        rotation_samples=transform_bank_prefix.rotation_samples,
        translation_samples=transform_bank_prefix.translation_samples,
        requested_num_trials=transform_bank_prefix.requested_num_trials,
        available_num_trials=transform_bank_prefix.available_num_trials,
        needle_compartment_distance_samples=needle_compartment_distance_samples,
    )


def _resolve_consistent_available_num_trials(
    structure_dict: Mapping[str, Any],
    key_names: Tuple[str, ...],
    expected_num_trials: Optional[int] = None,
) -> int:
    available_num_trials = None

    for key_name in key_names:
        structure_samples = _get_required_structure_samples(structure_dict, key_name)
        structure_num_trials = _resolve_num_trials(structure_samples, key_name)

        if expected_num_trials is not None and structure_num_trials != expected_num_trials:
            raise ValueError(
                "stored transform sample count mismatch for {}: expected {}, found {}".format(
                    key_name,
                    expected_num_trials,
                    structure_num_trials,
                )
            )

        if available_num_trials is None:
            available_num_trials = structure_num_trials
        elif structure_num_trials != available_num_trials:
            raise ValueError(
                "stored transform sample count mismatch across transform-bank arrays: {} has {}, expected {}".format(
                    key_name,
                    structure_num_trials,
                    available_num_trials,
                )
            )

    if available_num_trials is None:
        raise ValueError("key_names cannot be empty")

    return available_num_trials


def _slice_trial_prefix(
    structure_dict: Mapping[str, Any],
    key_name: str,
    num_trials: int,
    available_num_trials: int,
):
    if num_trials > available_num_trials:
        raise ValueError(
            "requested {} trials from {}, but only {} are available".format(
                num_trials,
                key_name,
                available_num_trials,
            )
        )

    structure_samples = _get_required_structure_samples(structure_dict, key_name)
    return structure_samples[:num_trials]


def _get_required_structure_samples(structure_dict: Mapping[str, Any], key_name: str):
    structure_samples = structure_dict.get(key_name)
    if structure_samples is None:
        raise ValueError("missing stored transform-bank array: {}".format(key_name))
    return structure_samples


def _resolve_num_trials(structure_samples: Any, key_name: str) -> int:
    if hasattr(structure_samples, "shape") and len(structure_samples.shape) > 0:
        return int(structure_samples.shape[0])

    try:
        return len(structure_samples)
    except TypeError as exc:
        raise ValueError("stored transform-bank array {} does not expose a trial dimension".format(key_name)) from exc


def _validate_non_negative_count(count: int, count_name: str) -> None:
    if count < 0:
        raise ValueError("{} cannot be negative".format(count_name))


__all__ = [
    "MAX_GENERATED_TRANSFORM_SAMPLES_KEY",
    "MC_BX_NEEDLE_COMPARTMENT_DISTANCE_SAMPLES_KEY",
    "MC_NORMAL_DILATION_SAMPLES_KEY",
    "MC_NORMAL_ROTATION_SAMPLES_KEY",
    "MC_NORMAL_TRANSLATION_SAMPLES_KEY",
    "OPTIMIZER_V2_TRANSFORM_SAMPLE_COUNT_KEY",
    "STOCHASTIC_TARGETING_TRANSFORM_SAMPLE_COUNT_KEY",
    "SharedTransformBankPrefix",
    "get_biopsy_transform_bank_prefix",
    "get_structure_transform_bank_prefix",
    "resolve_required_generated_transform_samples",
]
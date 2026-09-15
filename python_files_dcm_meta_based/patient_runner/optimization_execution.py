"""Reproducible execution preflight for the optimizer standalone boundary.

Adaptive pruning consumes capacity-packed trial prefixes. Hardware calibration
therefore cannot independently resolve the two lanes of an exact migration gate.
The existing typed capacity option must carry an explicit fixed structure budget.
This is a restriction on newly exposed worker execution, not a scientific change
to the legacy optimizer or its auto-calibrating production defaults.
"""

from biopsy_optimizer.v2.config import resolve_optimizer_v2_max_candidates_per_chunk


def resolve_fixed_optimization_execution(pipeline_config) -> tuple[int, int]:
    """Return structure/candidate budgets without GPU, patient, or RNG state.

    Reject implicit hardware calibration and unseeded execution before science.
    A None candidate limit is resolved by the unchanged pure optimizer rule.
    """
    optimizer = pipeline_config.optimizer.optimizer_v2
    budget = optimizer.capacity.max_test_structures_per_call
    if type(budget) is not int or budget < 1:
        raise ValueError(
            "optimization_shadow requires an explicit positive integer "
            "optimizer.optimizer_v2.capacity.max_test_structures_per_call; "
            "capacity affects adaptive trial prefixes and pruning"
        )
    seeds = pipeline_config.random_seeds
    if seeds.transform_generation_random_seed is None or seeds.optimizer_v1_random_seed is None:
        raise ValueError("optimization_shadow requires explicit transform and optimizer-v1 seeds")
    chunk, _ = resolve_optimizer_v2_max_candidates_per_chunk(
        requested_max_candidates_per_chunk=optimizer.capacity.max_candidates_per_chunk,
        resolved_max_test_structures_per_call=budget,
        search_config=optimizer.search_config,
        downstream_comparable_trial_count=(
            pipeline_config.mc.counts.num_mc_containment_simulations_input or None
        ),
        include_nominal=True,
    )
    return budget, chunk

from __future__ import annotations

import hashlib
from typing import Any, Mapping, MutableMapping

import cupy as cp


RANDOM_SEED_POLICY_SCHEMA_VERSION = "runtime_random_seed_policy_v1"
RANDOM_INFO_KEY = "Random info"
TRANSFORM_GENERATION_RANDOM_SEED_KEY = "Transform generation random seed"
TRANSFORM_GENERATION_SEED_SCOPE_KEY = "Transform generation seed scope"
OPTIMIZER_V1_RANDOM_SEED_KEY = "Optimizer v1 random seed"
OPTIMIZER_V1_SEED_SCOPE_KEY = "Optimizer v1 seed scope"
OPTIMIZER_V1_RESOLVED_PATIENT_SEEDS_KEY = "Optimizer v1 resolved patient seeds"
RANDOM_SEED_POLICY_SCHEMA_KEY = "Random seed policy schema version"

TRANSFORM_GENERATION_SEED_SCOPE = "cohort_stream"
OPTIMIZER_V1_SEED_SCOPE = "patient_uid_derived"
UINT32_MODULUS = 2 ** 32


def _global_random_info(master_structure_info_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    return master_structure_info_dict.setdefault("Global", {}).setdefault(RANDOM_INFO_KEY, {})


def _optional_int(seed: Any) -> int | None:
    if seed is None:
        return None
    return int(seed)


def _stable_derived_seed(base_seed: int, *components: Any) -> int:
    payload = "|".join([str(int(base_seed)), *(str(component) for component in components)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % UINT32_MODULUS


def configure_runtime_random_seed_settings(master_structure_info_dict: MutableMapping[str, Any],
                                           transform_generation_random_seed: int | None,
                                           optimizer_v1_random_seed: int | None) -> MutableMapping[str, Any]:
    random_info = _global_random_info(master_structure_info_dict)
    random_info[RANDOM_SEED_POLICY_SCHEMA_KEY] = RANDOM_SEED_POLICY_SCHEMA_VERSION
    random_info[TRANSFORM_GENERATION_RANDOM_SEED_KEY] = _optional_int(transform_generation_random_seed)
    random_info[TRANSFORM_GENERATION_SEED_SCOPE_KEY] = TRANSFORM_GENERATION_SEED_SCOPE
    random_info[OPTIMIZER_V1_RANDOM_SEED_KEY] = _optional_int(optimizer_v1_random_seed)
    random_info[OPTIMIZER_V1_SEED_SCOPE_KEY] = OPTIMIZER_V1_SEED_SCOPE
    random_info.setdefault(OPTIMIZER_V1_RESOLVED_PATIENT_SEEDS_KEY, {})
    return random_info


def build_transform_generation_rng(master_structure_info_dict: MutableMapping[str, Any]) -> cp.random.RandomState:
    random_info = _global_random_info(master_structure_info_dict)
    transform_generation_random_seed = random_info.get(TRANSFORM_GENERATION_RANDOM_SEED_KEY)
    if transform_generation_random_seed is None:
        return cp.random.RandomState()
    return cp.random.RandomState(int(transform_generation_random_seed))


def resolve_optimizer_v1_patient_seed(optimizer_v1_random_seed: int | None,
                                      patient_uid: str) -> int | None:
    base_seed = _optional_int(optimizer_v1_random_seed)
    if base_seed is None:
        return None
    return _stable_derived_seed(base_seed, "optimizer_v1", "patient", patient_uid)


def build_optimizer_v1_patient_rng(master_structure_info_dict: MutableMapping[str, Any],
                                   patient_uid: str,
                                   *,
                                   optimizer_v1_random_seed: int | None = None) -> tuple[cp.random.RandomState | None, dict[str, Any]]:
    random_info = _global_random_info(master_structure_info_dict)
    random_info.setdefault(RANDOM_SEED_POLICY_SCHEMA_KEY, RANDOM_SEED_POLICY_SCHEMA_VERSION)
    random_info.setdefault(TRANSFORM_GENERATION_SEED_SCOPE_KEY, TRANSFORM_GENERATION_SEED_SCOPE)
    random_info[OPTIMIZER_V1_SEED_SCOPE_KEY] = OPTIMIZER_V1_SEED_SCOPE

    base_seed = _optional_int(optimizer_v1_random_seed)
    if base_seed is None:
        base_seed = _optional_int(random_info.get(OPTIMIZER_V1_RANDOM_SEED_KEY))
    else:
        random_info[OPTIMIZER_V1_RANDOM_SEED_KEY] = base_seed

    resolved_seed = resolve_optimizer_v1_patient_seed(base_seed, str(patient_uid))
    metadata = {
        "random_seed_policy_schema_version": RANDOM_SEED_POLICY_SCHEMA_VERSION,
        "optimizer_v1_random_seed_base": base_seed,
        "optimizer_v1_random_seed_scope": OPTIMIZER_V1_SEED_SCOPE,
        "optimizer_v1_resolved_patient_seed": resolved_seed,
        "optimizer_v1_seeded": resolved_seed is not None,
    }
    if resolved_seed is None:
        return None, metadata

    resolved_patient_seeds = random_info.setdefault(OPTIMIZER_V1_RESOLVED_PATIENT_SEEDS_KEY, {})
    resolved_patient_seeds[str(patient_uid)] = int(resolved_seed)
    return cp.random.RandomState(int(resolved_seed)), metadata


def random_seed_policy_metadata(*,
                                transform_generation_random_seed: int | None,
                                optimizer_v1_random_seed: int | None) -> dict[str, Any]:
    return {
        "schema_version": RANDOM_SEED_POLICY_SCHEMA_VERSION,
        "transform_generation_random_seed": _optional_int(transform_generation_random_seed),
        "transform_generation_seed_scope": TRANSFORM_GENERATION_SEED_SCOPE,
        "optimizer_v1_random_seed": _optional_int(optimizer_v1_random_seed),
        "optimizer_v1_seed_scope": OPTIMIZER_V1_SEED_SCOPE,
    }


def runtime_random_seed_manifest_metadata(master_structure_info_dict: Mapping[str, Any]) -> dict[str, Any]:
    global_info = master_structure_info_dict.get("Global", {})
    random_info = dict(global_info.get(RANDOM_INFO_KEY, {}))
    resolved_patient_seeds = random_info.get(OPTIMIZER_V1_RESOLVED_PATIENT_SEEDS_KEY, {})
    return {
        "schema_version": random_info.get(RANDOM_SEED_POLICY_SCHEMA_KEY, RANDOM_SEED_POLICY_SCHEMA_VERSION),
        "transform_generation_random_seed": random_info.get(TRANSFORM_GENERATION_RANDOM_SEED_KEY),
        "transform_generation_seed_scope": random_info.get(TRANSFORM_GENERATION_SEED_SCOPE_KEY),
        "optimizer_v1_random_seed": random_info.get(OPTIMIZER_V1_RANDOM_SEED_KEY),
        "optimizer_v1_seed_scope": random_info.get(OPTIMIZER_V1_SEED_SCOPE_KEY),
        "optimizer_v1_resolved_patient_seeds": dict(resolved_patient_seeds),
    }
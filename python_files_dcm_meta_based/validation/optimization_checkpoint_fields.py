"""Bounded transform/optimizer evidence added to the migration checkpoint engine.

Retains products and resolved scientific execution, never mutable caches. Arrays
are copied to host only here. Timing columns are excluded by explicit name from
v2 tables and transport selection metadata; row/rank order stays unchanged.
"""

import numpy as np
import pandas as pd
from biopsy_optimizer.v1.output_keys import (
    OPTIMIZER_V1_DIL_OUTPUT_KEYS, OPTIMIZER_V1_MULTI_STRUCTURE_INFORMATION_KEYS,
    OPTIMIZER_V1_MULTI_STRUCTURE_PREPROCESSING_KEYS,
)
from biopsy_optimizer.v2.output_keys import (
    TARGET_DIL_OPTIMIZER_V2_SUMMARY_DF_KEY, TARGET_DIL_OPTIMIZER_V2_RANKED_DF_KEY,
    TARGET_DIL_OPTIMIZER_V2_TESTED_DF_KEY,
)
from preprocessing.transform_bank import (
    MC_NORMAL_DILATION_SAMPLES_KEY, MC_NORMAL_ROTATION_SAMPLES_KEY,
    MC_NORMAL_TRANSLATION_SAMPLES_KEY, MC_BX_NEEDLE_COMPARTMENT_DISTANCE_SAMPLES_KEY,
)
from .anatomical_checkpoint import AnatomicalFieldSpec as Field, _mapping

# Audited v2 output columns, not a substring rule that could hide future fields.
V2_TIMING_COLUMNS = frozenset({
    'Target optimizer cumulative stage biopsy self-transform elapsed seconds',
    'Target optimizer cumulative stage containment elapsed seconds',
    'Target optimizer cumulative stage elapsed seconds',
    'Target optimizer cumulative stage flatten for containment elapsed seconds',
    'Target optimizer cumulative stage relative structure localization elapsed seconds',
    'Target optimizer cumulative stage score reduction elapsed seconds',
    'Target optimizer cumulative stage tested candidate dataframe elapsed seconds',
    'Target optimizer final stage biopsy self-transform elapsed seconds',
    'Target optimizer final stage chunk scoring elapsed seconds',
    'Target optimizer final stage containment elapsed seconds',
    'Target optimizer final stage flatten for containment elapsed seconds',
    'Target optimizer final stage ranking elapsed seconds',
    'Target optimizer final stage relative structure localization elapsed seconds',
    'Target optimizer final stage score reduction elapsed seconds',
    'Target optimizer final stage tested candidate dataframe elapsed seconds',
    'Target optimizer final stage total elapsed seconds',
    'Target optimizer stage biopsy self-transform elapsed seconds',
    'Target optimizer stage chunk scoring elapsed seconds',
    'Target optimizer stage containment elapsed seconds',
    'Target optimizer stage flatten for containment elapsed seconds',
    'Target optimizer stage ranking elapsed seconds',
    'Target optimizer stage relative structure localization elapsed seconds',
    'Target optimizer stage score reduction elapsed seconds',
    'Target optimizer stage tested candidate dataframe elapsed seconds',
    'Target optimizer stage total elapsed seconds',
})

V2_TABLES = (TARGET_DIL_OPTIMIZER_V2_SUMMARY_DF_KEY,
             TARGET_DIL_OPTIMIZER_V2_RANKED_DF_KEY, TARGET_DIL_OPTIMIZER_V2_TESTED_DF_KEY)
TRANSFORM_FIELDS = (
    Field(MC_NORMAL_DILATION_SAMPLES_KEY, axes=("trial", "xy_z"), shape=(None, 2), units="mm"),
    Field(MC_NORMAL_ROTATION_SAMPLES_KEY, axes=("trial", "xyz"), shape=(None, 3), units="radians"),
    Field(MC_NORMAL_TRANSLATION_SAMPLES_KEY, axes=("trial", "xyz"), shape=(None, 3), units="mm"),
    Field(MC_BX_NEEDLE_COMPARTMENT_DISTANCE_SAMPLES_KEY, axes=("trial",), shape=(None,), units="mm; biopsy frame"),
)
UNCERTAINTY_FIELDS = tuple(
    Field("uncertainty_data_" + suffix, axes=("component",), shape=(size,), units=units)
    for suffix, size, units in (
        ("mean_arr", 3, "mm"), ("sigma_arr", 3, "mm"),
        ("dilations_mean_arr", 2, "mm"), ("dilations_sigma_arr", 2, "mm"),
        ("rotations_mean_arr", 3, "radians"), ("rotations_sigma_arr", 3, "radians"),
    )
)
TRANSFORM_STATE_KEYS = (
    "num_generated_transform_samples", "random_seed_policy_schema_version",
    "transform_generation_random_seed_base", "transform_generation_random_seed_scope",
    "transform_generation_resolved_patient_seed", "transform_generation_seeded",
)
OPTIMIZATION_STATE_KEYS = (
    "optimizer_v1_dil_count", "optimizer_v2_target_structure_count",
    "optimizer_v2_resolved_max_test_structures_per_call", "optimizer_v2_resolved_max_candidates_per_chunk",
    "optimizer_v1_random_seed_policy_schema_version", "optimizer_v1_optimizer_v1_random_seed_base",
    "optimizer_v1_optimizer_v1_random_seed_scope", "optimizer_v1_optimizer_v1_resolved_patient_seed",
    "optimizer_v1_optimizer_v1_seeded",
)


def _host(value):
    if hasattr(value, "__cuda_array_interface__"):
        import cupy as cp
        return cp.asnumpy(value)
    return value


def _scientific_table(table):
    if table is None:
        return None
    if not isinstance(table, pd.DataFrame):
        raise TypeError("optimizer output must be a dataframe")
    return table.drop(columns=[name for name in table.columns if name in V2_TIMING_COLUMNS])


def capture_optimization(capture, patient, config, state):
    """Capture fixed structure products, both optimizers and their executed state."""
    refs = config.legacy_refs
    for section, keys in (("transform_generation", TRANSFORM_STATE_KEYS),
                          ("optimization", OPTIMIZATION_STATE_KEYS)):
        values = state.get(section, {})
        for key in keys:
            capture.field(["optimization", "execution", section, key], Field(key, "label"), values.get(key))
    for family in config.structure_registry.structs_referenced_list:
        for index, record in enumerate(patient.get(family, ())):
            path = ["optimization", "structures", family, str(index)]
            for key in ("ROI", "Ref #", "Index number", "Struct type"):
                capture.field([*path, key], Field(key, "label"), record.get(key))
            for spec in TRANSFORM_FIELDS:
                capture.field([*path, spec.key], spec, _host(record.get(spec.key)))
            uncertainty = record.get("Uncertainty data")
            for spec in UNCERTAINTY_FIELDS:
                capture.field([*path, spec.key], spec, None if uncertainty is None else getattr(uncertainty, spec.key))
            info = {} if uncertainty is None else uncertainty.uncertainty_data_info_dict
            for key in ("Frame of reference", "Distribution"):
                capture.field([*path, "uncertainty", key], Field(key, "label"), info.get(key))
            if family == refs.dil_ref:
                for key in OPTIMIZER_V1_DIL_OUTPUT_KEYS:
                    spec = (Field(key, axes=("point", "xyz"), shape=(None, 3), units="mm")
                            if key.endswith("only in dil") else Field(key, "table"))
                    capture.field([*path, "v1", key], spec, record.get(key))
            if family == refs.bx_ref:
                capture.field([*path, "configured_transport_family"], Field("Transport family", "label"),
                              record.get("Transport family"))
                request = _mapping(record.get("Simulated biopsy transport request dict"))
                for key in ("Transport family", "Transport source"):
                    capture.field([*path, "transport", key], Field(key, "label"), request.get(key))
                capture.field([*path, "transport", "Target vector"],
                              Field("Target vector", axes=("xyz",), shape=(3,), units="mm; DICOM patient frame"),
                              request.get("Target vector"))
                selection = request.get("Selection metadata")
                table = None if selection is None else pd.DataFrame([dict(selection)])
                capture.field([*path, "transport", "Selection metadata"], Field("Selection metadata", "table"),
                              _scientific_table(table))
    all_ref = _mapping(patient.get(refs.all_ref_key))
    info = _mapping(all_ref.get("Multi-structure information dict (not for csv output)"))
    tables = _mapping(all_ref.get("Multi-structure pre-processing output dataframes dict"))
    for key in OPTIMIZER_V1_MULTI_STRUCTURE_INFORMATION_KEYS:
        capture.field(["optimization", "v1", key], Field(key, "table"), info.get(key))
    for key in OPTIMIZER_V1_MULTI_STRUCTURE_PREPROCESSING_KEYS:
        capture.field(["optimization", "v1", key], Field(key, "table"), tables.get(key))
    for key in V2_TABLES:
        capture.field(["optimization", "v2", key], Field(key, "table"), _scientific_table(tables.get(key)))


def expected_optimization_inventory(capture, items, config):
    """Reconstruct descriptors from bounded family inventories, never supplied contracts."""
    patient = {}
    for family in config.structure_registry.structs_referenced_list:
        path = ("biopsies", "count") if family == config.legacy_refs.bx_ref else ("families", family)
        count = items[path].get("value", {}).get("value")
        if type(count) is not int or count < 0 or count > len(items):
            raise ValueError("optimization requires explicit structure family inventory")
        patient[family] = [{} for _ in range(count)]
    capture_optimization(capture, patient, config, {})


def optimization_coverage(items, arrays, config, patient_uid):
    """Check required products and executed counts/seeds/capacities against config."""
    from patient_runner.optimization_execution import resolve_fixed_optimization_execution
    from patient_runner.scientific_config_builder import _resolve_explicit_transform_counts
    from random_seed_policy import (
        RANDOM_SEED_POLICY_SCHEMA_VERSION, TRANSFORM_GENERATION_SEED_SCOPE, OPTIMIZER_V1_SEED_SCOPE,
        resolve_transform_generation_patient_seed, resolve_optimizer_v1_patient_seed,
    )
    entries = {tuple(item["path"]): item for item in items}
    missing = []
    def label(path):
        return entries[path].get("value", {}).get("value")
    def require(path):
        item = entries[path]
        if item["state"] != "present":
            missing.append(list(path))
        return item
    n = _resolve_explicit_transform_counts(config)[1]
    budget, chunk = resolve_fixed_optimization_execution(config)
    seeds = config.random_seeds
    expected_state = {
        "transform_generation": dict(zip(TRANSFORM_STATE_KEYS, (
            n, RANDOM_SEED_POLICY_SCHEMA_VERSION, seeds.transform_generation_random_seed,
            TRANSFORM_GENERATION_SEED_SCOPE,
            resolve_transform_generation_patient_seed(seeds.transform_generation_random_seed, patient_uid), True))),
    }
    dil_count = label(("families", config.legacy_refs.dil_ref))
    biopsy_count = label(("biopsies", "count"))
    targets = [i for i in range(biopsy_count) if
               label(("biopsies", str(i), "Simulated bool")) is True and
               label(("biopsies", str(i), "Simulated type")) == config.biopsy.simulated.optimizer_simulated_type]
    expected_state["optimization"] = dict(zip(OPTIMIZATION_STATE_KEYS, (
        dil_count, len(targets), budget, chunk, RANDOM_SEED_POLICY_SCHEMA_VERSION,
        seeds.optimizer_v1_random_seed, OPTIMIZER_V1_SEED_SCOPE,
        resolve_optimizer_v1_patient_seed(seeds.optimizer_v1_random_seed, patient_uid), True)))
    for section, values in expected_state.items():
        for key, expected in values.items():
            path = ("optimization", "execution", section, key)
            if label(path) != expected:
                missing.append(list(path))
    for path, item in entries.items():
        if path[:2] in (("optimization", "v1"),):
            require(path)
        if path[:2] != ("optimization", "structures") or len(path) < 5:
            continue
        family, index, key = path[2:5]
        if len(path) == 5 and key in {s.key for s in TRANSFORM_FIELDS + UNCERTAINTY_FIELDS}:
            uniform = key == MC_BX_NEEDLE_COMPARTMENT_DISTANCE_SAMPLES_KEY
            needed = not uniform or (family == config.legacy_refs.bx_ref and
                                    config.mc.prep.simulate_uniform_bx_shifts_due_to_bx_needle_compartment)
            if needed:
                required = require(path)
                array = arrays.get(required.get("payload"))
                if array is None or not np.isfinite(array).all():
                    missing.append(list(path))
                elif key in {s.key for s in TRANSFORM_FIELDS} and len(array) != n:
                    missing.append(list(path))
        elif key == "configured_transport_family":
            if label(("biopsies", index, "Simulated bool")) is True:
                require(path)
        elif len(path) == 5 or (key == "v1" and len(path) == 6) or (key == "uncertainty" and len(path) == 6):
            require(path)
        elif key == "transport" and len(path) == 6 and int(index) in targets:
            require(path)
    if targets:
        summary = require(("optimization", "v2", TARGET_DIL_OPTIMIZER_V2_SUMMARY_DF_KEY))
        if summary.get("table_schema", {}).get("shape", [0])[0] != len(targets):
            missing.append(["optimization", "v2", "summary row count"])
        nonfallback = False
        for index in targets:
            path = ("optimization", "structures", config.legacy_refs.bx_ref, str(index), "transport")
            source = label((*path, "Transport source"))
            if source not in ("target_dil_optimizer_v2", "target_dil_optimizer_v2:target_centroid_fallback"):
                missing.append([*path, "Transport source"])
            nonfallback |= source == "target_dil_optimizer_v2"
            if label((*path, "Transport family")) != "identity":
                missing.append([*path, "Transport family"])
            vector = arrays.get(entries[(*path, "Target vector")].get("payload"))
            if vector is None or not np.isfinite(vector).all():
                missing.append([*path, "Target vector"])
            selection = entries[(*path, "Selection metadata")].get("table_schema", {}).get("shape", [0, 0])
            if selection[0] != 1 or selection[1] == 0:
                missing.append([*path, "Selection metadata"])
        if nonfallback:
            for key in (TARGET_DIL_OPTIMIZER_V2_RANKED_DF_KEY, TARGET_DIL_OPTIMIZER_V2_TESTED_DF_KEY):
                table = require(("optimization", "v2", key))
                if table.get("table_schema", {}).get("shape", [0])[0] == 0:
                    missing.append(["optimization", "v2", key])
    return {"complete": not missing, "missing_required": missing,
            "generated_sample_count": n, "optimizer_v1_dil_count": dil_count,
            "optimizer_v2_target_count": len(targets),
            "resolved_structure_budget": budget, "resolved_candidate_chunk": chunk,
            "timing_columns_excluded": sorted(V2_TIMING_COLUMNS)}

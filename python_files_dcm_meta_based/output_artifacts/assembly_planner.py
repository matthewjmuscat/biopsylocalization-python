from __future__ import annotations

"""Registry-derived assembly plans for patient-fragment cohort outputs.

The planner keeps output assembly policy with the output contract layer. Patient
runner code can consume these plans without hardcoding table names, while the
registry remains the durable place to declare row grain, source fragments, and
retention policy.
"""

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from legacy_data_keys import legacy_data_keys

from .schema_registry import OutputSchemaRegistry
from .schema_registry import OutputTableSpec
from .stitch_validation import ShadowStitchPair


OUTPUT_ASSEMBLY_PLANNER_SCHEMA_VERSION = "output_assembly_planner_v1"

DIRECT_REGISTRY_STITCH_METHODS = frozenset(
    {
        "concat_rows",
        "concat_current_summary_fragments",
        "recompute_downstream",
    }
)
CURRENT_DIRECT_ASSEMBLY_METHOD = "concat_patient_fragments"
ORDER_MODE_SOURCE_FRAGMENT = "source_fragment_order"
ORDER_MODE_COLUMN_SORT = "column_sort"
ORDER_MODES = frozenset({ORDER_MODE_SOURCE_FRAGMENT, ORDER_MODE_COLUMN_SORT})

LEGACY_STRUCTURE_RECORD_KEYS = legacy_data_keys.structure_record
LEGACY_VALIDATION_ORDER_OVERRIDES: Mapping[str, tuple[str, ...]] = {
    "Cohort: Per voxel prostate double sextant classification": (
        "Patient ID",
        "Bx ID",
        "Bx index",
        "Voxel index",
        LEGACY_STRUCTURE_RECORD_KEYS.simulated_type_key,
        LEGACY_STRUCTURE_RECORD_KEYS.simulated_bool_key,
        "Bx refnum",
    ),
}


@dataclass(frozen=True, slots=True)
class OutputRowOrderPolicy:
    """Row ordering policy for one assembly context.

    Validation order is allowed to preserve legacy construction order. Production
    order can be cleaner and canonical once parity is proven and downstream
    users accept the change.
    """

    policy_id: str
    order_mode: str
    columns: tuple[str, ...] = ()
    missing_column_policy: str = "ignore_missing"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.policy_id.strip():
            raise ValueError("policy_id cannot be empty")
        if self.order_mode not in ORDER_MODES:
            raise ValueError(f"Unsupported row order mode: {self.order_mode!r}")
        object.__setattr__(self, "columns", tuple(self.columns))

    def to_row_prefix(self, prefix: str) -> dict[str, Any]:
        """Return CSV/JSON-friendly fields with a stable prefix."""

        return {
            f"{prefix}_order_policy_id": self.policy_id,
            f"{prefix}_order_mode": self.order_mode,
            f"{prefix}_order_columns": " | ".join(self.columns),
            f"{prefix}_missing_column_policy": self.missing_column_policy,
            f"{prefix}_order_notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class OutputAssemblyPlan:
    """Contract-level plan for assembling one final cohort table."""

    final_table_id: str
    final_table_name: str
    source_table_id: str
    source_table_name: str
    source_output_section: str
    file_extension: str
    stitch_method: str
    registry_stitch_method: str
    identity_key: tuple[str, ...]
    validation_order_policy: OutputRowOrderPolicy
    production_order_policy: OutputRowOrderPolicy
    columns_policy: str
    validation_csv_index: bool
    production_csv_index: bool
    has_multiindex_columns: bool | None
    validation_status: str
    retention_policy: str
    notes: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "final_table_name",
            "source_table_name",
            "source_output_section",
            "file_extension",
            "stitch_method",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} cannot be empty")
        object.__setattr__(self, "identity_key", tuple(self.identity_key))

    def order_policy(self, context: str = "validation") -> OutputRowOrderPolicy:
        """Return the row-order policy for a named assembly context."""

        if context == "validation":
            return self.validation_order_policy
        if context == "production":
            return self.production_order_policy
        raise ValueError(f"Unsupported assembly order context: {context!r}")

    def to_shadow_stitch_pair(self) -> ShadowStitchPair:
        """Return the transitional stitch-pair shape used by older callers."""

        row_order_columns = (
            self.validation_order_policy.columns
            if self.validation_order_policy.order_mode == ORDER_MODE_COLUMN_SORT
            else ()
        )
        return ShadowStitchPair(
            final_table_name=self.final_table_name,
            source_table_name=self.source_table_name,
            source_output_section=self.source_output_section,
            file_extension=self.file_extension,
            stitch_method=self.stitch_method,
            row_order_columns=row_order_columns,
        )

    def to_row(self) -> dict[str, Any]:
        """Return a CSV/JSON-friendly representation of the assembly plan."""

        row = asdict(self)
        row["identity_key"] = " | ".join(self.identity_key)
        row.pop("validation_order_policy")
        row.pop("production_order_policy")
        row.update(self.validation_order_policy.to_row_prefix("validation"))
        row.update(self.production_order_policy.to_row_prefix("production"))
        row["schema_version"] = OUTPUT_ASSEMBLY_PLANNER_SCHEMA_VERSION
        if self.has_multiindex_columns is None:
            row["has_multiindex_columns"] = "unknown_until_runtime"
        return row


def _specs_by_id(registry: OutputSchemaRegistry) -> dict[str, OutputTableSpec]:
    return {spec.table_id: spec for spec in registry.specs}


def _cohort_spec_for_pair(registry: OutputSchemaRegistry, pair: ShadowStitchPair) -> OutputTableSpec | None:
    return registry.match_spec(pair.final_table_name, "Output CSVs/Cohort", pair.file_extension)


def _source_spec_for_pair(registry: OutputSchemaRegistry, pair: ShadowStitchPair) -> OutputTableSpec | None:
    return registry.match_spec(pair.source_table_name, pair.source_output_section, pair.file_extension)


def _validation_order_policy(final_table_name: str,
                             explicit_columns: Sequence[str] = ()) -> OutputRowOrderPolicy:
    columns = tuple(explicit_columns) or LEGACY_VALIDATION_ORDER_OVERRIDES.get(final_table_name, ())
    if columns:
        return OutputRowOrderPolicy(
            policy_id="legacy_validation_column_sort",
            order_mode=ORDER_MODE_COLUMN_SORT,
            columns=columns,
            notes="Explicit legacy validation order for this final cohort artifact.",
        )
    return OutputRowOrderPolicy(
        policy_id="legacy_validation_source_fragment_order",
        order_mode=ORDER_MODE_SOURCE_FRAGMENT,
        notes="Preserve patient fragment order while proving parity with the legacy cohort builder.",
    )


def _production_order_policy(identity_key: Sequence[str]) -> OutputRowOrderPolicy:
    return OutputRowOrderPolicy(
        policy_id="canonical_identity_key_order",
        order_mode=ORDER_MODE_COLUMN_SORT,
        columns=tuple(identity_key),
        notes="Canonical production order candidate; switch only after validation parity is green.",
    )


def _plan_from_specs(final_spec: OutputTableSpec | None,
                     source_spec: OutputTableSpec | None,
                     *,
                     final_table_name: str,
                     source_table_name: str,
                     source_output_section: str,
                     file_extension: str,
                     stitch_method: str = CURRENT_DIRECT_ASSEMBLY_METHOD,
                     row_order_columns: Sequence[str] = ()) -> OutputAssemblyPlan:
    identity_key = ()
    if final_spec is not None:
        identity_key = final_spec.canonical_primary_key
    elif source_spec is not None:
        identity_key = source_spec.canonical_primary_key

    registry_stitch_method = final_spec.stitch_method if final_spec is not None else stitch_method
    columns_policy = final_spec.columns_policy if final_spec is not None else "legacy_preserve"
    has_multiindex_columns = final_spec.has_multiindex_columns if final_spec is not None else None
    validation_status = final_spec.validation_status if final_spec is not None else "unregistered"
    retention_policy = final_spec.retention_policy if final_spec is not None else "unknown"
    notes = final_spec.notes if final_spec is not None else "No final cohort spec matched this assembly plan."

    return OutputAssemblyPlan(
        final_table_id=final_spec.table_id if final_spec is not None else "",
        final_table_name=final_table_name,
        source_table_id=source_spec.table_id if source_spec is not None else "",
        source_table_name=source_table_name,
        source_output_section=source_output_section,
        file_extension=file_extension,
        stitch_method=stitch_method,
        registry_stitch_method=registry_stitch_method,
        identity_key=identity_key,
        validation_order_policy=_validation_order_policy(final_table_name, row_order_columns),
        production_order_policy=_production_order_policy(identity_key),
        columns_policy=columns_policy,
        validation_csv_index=True,
        production_csv_index=False,
        has_multiindex_columns=has_multiindex_columns,
        validation_status=validation_status,
        retention_policy=retention_policy,
        notes=notes,
    )


def _registry_direct_pairs(registry: OutputSchemaRegistry) -> tuple[tuple[OutputTableSpec, OutputTableSpec], ...]:
    specs_by_id = _specs_by_id(registry)
    pairs: list[tuple[OutputTableSpec, OutputTableSpec]] = []
    for final_spec in registry.specs:
        if final_spec.artifact_scope != "cohort":
            continue
        if final_spec.stitch_method not in DIRECT_REGISTRY_STITCH_METHODS:
            continue
        if not final_spec.source_fragment_table_id:
            continue
        source_spec = specs_by_id.get(final_spec.source_fragment_table_id)
        if source_spec is None:
            raise ValueError(
                "Missing source fragment spec "
                f"{final_spec.source_fragment_table_id!r} for {final_spec.table_id!r}"
            )
        pairs.append((final_spec, source_spec))
    return tuple(pairs)


def build_output_assembly_plans(registry: OutputSchemaRegistry | None = None,
                                stitch_pairs: Sequence[ShadowStitchPair] | None = None) -> tuple[OutputAssemblyPlan, ...]:
    """Build assembly plans from registry links or transitional stitch pairs.

    When `stitch_pairs` is omitted, the planner discovers direct cohort assembly
    surfaces from registry specs whose final table points at a source fragment.
    Transitional callers can still pass `ShadowStitchPair` values; the returned
    plans are enriched with registry identity and ordering policy.
    """

    registry = OutputSchemaRegistry() if registry is None else registry
    if stitch_pairs is not None:
        return tuple(
            _plan_from_specs(
                _cohort_spec_for_pair(registry, pair),
                _source_spec_for_pair(registry, pair),
                final_table_name=pair.final_table_name,
                source_table_name=pair.source_table_name,
                source_output_section=pair.source_output_section,
                file_extension=pair.file_extension,
                stitch_method=pair.stitch_method,
                row_order_columns=pair.row_order_columns,
            )
            for pair in stitch_pairs
        )

    return tuple(
        _plan_from_specs(
            final_spec,
            source_spec,
            final_table_name=final_spec.legacy_table_name,
            source_table_name=source_spec.legacy_table_name,
            source_output_section=source_spec.legacy_output_section,
            file_extension=source_spec.file_extension,
            stitch_method=CURRENT_DIRECT_ASSEMBLY_METHOD,
            row_order_columns=LEGACY_VALIDATION_ORDER_OVERRIDES.get(final_spec.legacy_table_name, ()),
        )
        for final_spec, source_spec in _registry_direct_pairs(registry)
    )


def build_shadow_stitch_pairs_from_output_assembly_plans(
    plans: Sequence[OutputAssemblyPlan] | None = None,
) -> tuple[ShadowStitchPair, ...]:
    """Return transitional stitch pairs derived from assembly plans."""

    resolved_plans = build_output_assembly_plans() if plans is None else tuple(plans)
    return tuple(plan.to_shadow_stitch_pair() for plan in resolved_plans)


def output_assembly_plan_rows(plans: Sequence[OutputAssemblyPlan] | None = None) -> tuple[dict[str, Any], ...]:
    """Return assembly plans as serializable rows for reports and debugging."""

    resolved_plans = build_output_assembly_plans() if plans is None else tuple(plans)
    return tuple(plan.to_row() for plan in resolved_plans)


__all__ = [
    "CURRENT_DIRECT_ASSEMBLY_METHOD",
    "DIRECT_REGISTRY_STITCH_METHODS",
    "LEGACY_VALIDATION_ORDER_OVERRIDES",
    "ORDER_MODE_COLUMN_SORT",
    "ORDER_MODE_SOURCE_FRAGMENT",
    "OUTPUT_ASSEMBLY_PLANNER_SCHEMA_VERSION",
    "OutputAssemblyPlan",
    "OutputRowOrderPolicy",
    "build_output_assembly_plans",
    "build_shadow_stitch_pairs_from_output_assembly_plans",
    "output_assembly_plan_rows",
]
"""Build one standalone worker's patient-local scientific runtime state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from preprocessing.structure_reference_bootstrap import (
    assemble_structure_reference_info_for_run,
)
from preprocessing.structure_reference_bootstrap import (
    attach_patient_dose_reference_from_path,
)
from preprocessing.structure_reference_bootstrap import (
    attach_patient_mr_adc_references_from_paths,
)
from preprocessing.structure_reference_bootstrap import (
    attach_patient_plan_reference_from_path,
)
from preprocessing.structure_reference_bootstrap import (
    build_patient_structure_reference_bootstrap_fragment_from_path_and_config,
)

from .contracts import LegacyPatientRuntimeState
from .contracts import LegacyRuntimeKeys
from .contracts import PatientCase
from .inputs import PatientInputPaths
from .scientific_config_builder import PatientRunnerScientificConfigBuildContext
from .worker_resources import SequentialWorkerPool


@dataclass(frozen=True, slots=True)
class StandalonePatientRuntime:
    """Patient-local mutable state plus resources required to configure stages."""

    runtime_state: LegacyPatientRuntimeState
    config_build_context: PatientRunnerScientificConfigBuildContext
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_state, LegacyPatientRuntimeState):
            raise TypeError("runtime_state must be a LegacyPatientRuntimeState")
        if not isinstance(self.config_build_context, PatientRunnerScientificConfigBuildContext):
            raise TypeError("config_build_context must be a PatientRunnerScientificConfigBuildContext")
        object.__setattr__(self, "metadata", dict(self.metadata))


def build_standalone_patient_runtime(
    *,
    patient_case: PatientCase,
    patient_inputs: PatientInputPaths,
    pipeline_config: Any,
    parallel_pool: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> StandalonePatientRuntime:
    """Load explicit patient inputs into the established patient state contract.

    This boundary performs input adaptation only. It does not execute a
    scientific stage and does not mutate any cohort- or main-owned dictionary.
    """
    if not isinstance(patient_case, PatientCase):
        raise TypeError("patient_case must be a PatientCase")
    if not isinstance(patient_inputs, PatientInputPaths):
        raise TypeError("patient_inputs must be a PatientInputPaths")
    if patient_inputs.patient_uid != patient_case.patient_uid:
        raise ValueError("patient_inputs.patient_uid must match patient_case.patient_uid")
    if patient_inputs.rtstruct is None or patient_inputs.rtdose is None or patient_inputs.rtplan is None:
        raise ValueError("standalone patient runtime requires RTSTRUCT, RTDOSE, and RTPLAN paths")

    refs = pipeline_config.legacy_refs
    fragment = build_patient_structure_reference_bootstrap_fragment_from_path_and_config(
        patient_uid=patient_case.patient_uid,
        structure_item_path=patient_inputs.rtstruct,
        bootstrap_config=pipeline_config.bootstrap,
        legacy_refs=refs,
        structure_registry=pipeline_config.structure_registry,
    )
    patient_reference_dict = fragment.patient_reference_dict
    dose_attached = attach_patient_dose_reference_from_path(
        patient_reference_dict,
        patient_uid=patient_case.patient_uid,
        dose_item_path=patient_inputs.rtdose,
        ds_ref=refs.dose_ref,
    )
    attach_patient_plan_reference_from_path(
        patient_reference_dict,
        patient_uid=patient_case.patient_uid,
        plan_item_path=patient_inputs.rtplan,
        pln_ref=refs.plan_ref,
    )
    if patient_inputs.mr_adc:
        attach_patient_mr_adc_references_from_paths(
            patient_reference_dict,
            patient_uid=patient_case.patient_uid,
            mr_adc_item_paths=patient_inputs.mr_adc,
            mr_adc_ref=refs.mr_adc_ref,
        )

    master_structure_reference_dict = {
        patient_case.patient_uid: patient_reference_dict,
    }
    master_structure_info_dict = assemble_structure_reference_info_for_run(
        (fragment,),
        st_ref_list=(
            refs.bx_ref,
            refs.oar_ref,
            refs.dil_ref,
            refs.rectum_ref_key,
            refs.urethra_ref_key,
        ),
        all_ref_key=refs.all_ref_key,
        bx_sim_locations_dict=pipeline_config.bootstrap.simulated_biopsies.locations,
        interp_inter_slice_dist=pipeline_config.preprocessing.interp_inter_slice_dist,
        interp_intra_slice_dist=pipeline_config.preprocessing.interp_intra_slice_dist,
    )
    legacy_keys = LegacyRuntimeKeys(
        all_ref_key=refs.all_ref_key,
        bx_ref=refs.bx_ref,
        by_patient_key=refs.by_patient_key,
        global_key=refs.global_key,
        global_num_cases_key=refs.global_num_cases_key,
    )
    resolved_metadata = {
        **dict(metadata or {}),
        "runtime_builder": "standalone_patient_runtime_v1",
        "patient_input_manifest_identity_sha256": patient_inputs.manifest_identity_sha256,
        "dose_reference_attached": dose_attached,
        "plan_reference_attached": True,
        "mr_adc_reference_attached": bool(patient_inputs.mr_adc),
        "bootstrap": dict(fragment.metadata),
    }
    runtime_state = LegacyPatientRuntimeState(
        patient_case=patient_case,
        master_structure_reference_dict=master_structure_reference_dict,
        master_structure_info_dict=master_structure_info_dict,
        legacy_keys=legacy_keys,
        metadata=resolved_metadata,
    )
    config_build_context = PatientRunnerScientificConfigBuildContext(
        rtstruct_dicom_paths_by_patient_uid={patient_case.patient_uid: patient_inputs.rtstruct},
        parallel_pool=parallel_pool if parallel_pool is not None else SequentialWorkerPool(),
        metadata={
            "runtime_builder": "standalone_patient_runtime_v1",
            "patient_input_manifest_identity_sha256": patient_inputs.manifest_identity_sha256,
        },
    )
    return StandalonePatientRuntime(
        runtime_state=runtime_state,
        config_build_context=config_build_context,
        metadata=resolved_metadata,
    )


__all__ = ["StandalonePatientRuntime", "build_standalone_patient_runtime"]
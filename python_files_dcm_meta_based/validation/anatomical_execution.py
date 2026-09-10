"""Validation-only input adapter and evidence hook for anatomical runs.

The legacy input lane calls the unchanged cohort input builder with one patient.
Both lanes subsequently use the same established anatomical stage adapters: this
checks input migration and execution parity, not independent algorithm truth.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any


def build_legacy_input_anatomical_runtime(*, patient_case, patient_inputs, pipeline_config, metadata=None):
    """Build fresh singleton state through the unchanged legacy input function.

    Called only by the validation child process after worker preflight. No main
    globals, serialized runtime pickle, or standalone bootstrap state is reused.
    """
    from preprocessing.structure_referencer import structure_referencer
    from presentation import LegacyPresentationContext
    from patient_runner.contracts import LegacyPatientRuntimeState, LegacyRuntimeKeys
    from patient_runner.runtime_builder import StandalonePatientRuntime
    from patient_runner.scientific_config_builder import PatientRunnerScientificConfigBuildContext
    from patient_runner.worker_resources import SequentialWorkerPool

    refs = pipeline_config.legacy_refs
    bootstrap = pipeline_config.bootstrap
    uid = patient_case.patient_uid
    presentation = LegacyPresentationContext.null()
    reference, info = structure_referencer(
        data_removals_dict_bx=bootstrap.removals.biopsy,
        data_removals_dict_prostate=bootstrap.removals.prostate,
        data_removals_dict_dil=bootstrap.removals.dil,
        data_removals_dict_urethra=bootstrap.removals.urethra,
        data_removals_dict_rectum=bootstrap.removals.rectum,
        structure_dcm_dict={uid: patient_inputs.rtstruct},
        dose_dcm_dict={uid: patient_inputs.rtdose},
        plan_dcm_dict={uid: patient_inputs.rtplan},
        US_dcms_dict={uid: list(patient_inputs.us)} if patient_inputs.us else {},
        MR_T2_dcms_dict={uid: list(patient_inputs.mr_t2)} if patient_inputs.mr_t2 else {},
        MR_ADC_dcms_dict={uid: list(patient_inputs.mr_adc)} if patient_inputs.mr_adc else {},
        OAR_list=bootstrap.contours.oar, DIL_list=bootstrap.contours.dil,
        Bx_list=bootstrap.contours.biopsy,
        st_ref_list=[refs.bx_ref, refs.oar_ref, refs.dil_ref, refs.rectum_ref_key, refs.urethra_ref_key],
        structs_referenced_dict=pipeline_config.structure_registry.structs_referenced_dict,
        ds_ref=refs.dose_ref, pln_ref=refs.plan_ref, mr_adc_ref=refs.mr_adc_ref,
        mr_t2_ref=refs.mr_t2_ref, us_ref=refs.us_ref, all_ref_key=refs.all_ref_key,
        mr_global_multi_structure_output_dataframe_str=bootstrap.mr_global_structure_table_name,
        mr_global_by_voxel_multi_structure_output_dataframe_str=bootstrap.mr_global_voxel_table_name,
        bx_sim_locations_dict=bootstrap.simulated_biopsies.locations,
        rectum_list=bootstrap.contours.rectum, urethra_list=bootstrap.contours.urethra,
        interp_inter_slice_dist=pipeline_config.preprocessing.interp_inter_slice_dist,
        interp_intra_slice_dist=pipeline_config.preprocessing.interp_intra_slice_dist,
        simulated_biopsy_fraction_numbers_to_create=bootstrap.simulated_biopsies.fraction_numbers_to_create,
        fraction_prefixes=bootstrap.simulated_biopsies.fraction_prefixes,
        important_info=presentation.important_info, live_display=presentation.live_display,
    )
    runtime = LegacyPatientRuntimeState(
        patient_case, reference, info,
        LegacyRuntimeKeys(refs.all_ref_key, refs.bx_ref, refs.by_patient_key, refs.global_key, refs.global_num_cases_key),
        metadata={**(metadata or {}), "runtime_builder": "legacy_single_patient_input_validation"},
    )
    context = PatientRunnerScientificConfigBuildContext(
        rtstruct_dicom_paths_by_patient_uid={uid: patient_inputs.rtstruct},
        parallel_pool=SequentialWorkerPool(),
    )
    return StandalonePatientRuntime(runtime, context)


def with_anatomical_checkpoint(stages: tuple, pipeline_config: Any) -> tuple:
    """Decorate the anatomical stage with fail-closed, opt-in evidence writing.

    Checkpoint files are validation evidence, not cohort dataframe fragments.
    Capture errors fail the stage through the existing runner error handling.
    """
    from patient_runner.runner import PatientStage
    from patient_runner.contracts import PatientStageName
    from validation.anatomical_checkpoint import write_anatomical_checkpoint

    wrapped = []
    found = False
    for stage in stages:
        if stage.stage_name != PatientStageName.ANATOMICAL_PREPROCESSING:
            wrapped.append(stage)
            continue
        found = True
        original_runner = stage.runner

        def capture(runtime_state, config, *, runner=original_runner):
            result = runner(runtime_state, config)
            if not result.succeeded:
                return result
            path = write_anatomical_checkpoint(
                runtime_state=runtime_state, pipeline_config=pipeline_config,
                output_dir=config.patient_output_dir(runtime_state.patient_case) / "validation" / "anatomical",
                metadata={**runtime_state.metadata, "stage_metadata": dict(result.metadata)},
            )
            return replace(result, metadata={**result.metadata, "anatomical_checkpoint_path": str(path)})

        wrapped.append(PatientStage(stage.stage_name, capture))
    if not found:
        raise ValueError("anatomical checkpoint requires the anatomical preprocessing stage")
    return tuple(wrapped)
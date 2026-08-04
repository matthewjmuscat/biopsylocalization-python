"""Patient-runner persistence for retained MC dose context artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from output_artifacts.context_contracts import ArtifactRef
from output_artifacts.context_contracts import PATIENT_ARTIFACT_INDEX_FILENAME
from output_artifacts.context_contracts import PatientArtifactIndex
from output_artifacts.context_contracts import write_patient_artifact_index

from mc.simulation.per_patient.dose_context_artifacts import build_patient_dose_biopsy_query_context_artifact_plan
from mc.simulation.per_patient.dose_context_artifacts import build_patient_dose_lattice_context_artifact_plan
from mc.simulation.per_patient.dose_context_artifacts import build_patient_dose_localization_context_artifact_plan
from mc.simulation.per_patient.dose_context_artifacts import patient_dose_biopsy_query_context_array_payload
from mc.simulation.per_patient.dose_context_artifacts import patient_dose_lattice_context_array_payload
from mc.simulation.per_patient.dose_context_artifacts import patient_dose_localization_context_array_payload
from mc.simulation.per_patient.dose_context_artifacts import write_patient_dose_context_zarr_arrays
from mc.visualization.dose_nn_context_bridge import write_dose_nn_render_context_zarr_artifact
from mc.visualization.dose_nn_scene import DoseNNSceneMetadata
from mc.visualization.dose_nn_scene import build_dose_nn_render_scene_from_dataframe


@dataclass(frozen=True, slots=True)
class PatientDoseContextPersistenceSummary:
    """Summary of retained dose context artifacts written for one patient."""

    artifact_index_path: Path
    artifact_refs: tuple[ArtifactRef, ...]
    output_paths: tuple[Path, ...]


@dataclass(slots=True)
class PatientDoseContextArtifactPersister:
    """Persist dose context artifacts at the MC dose localization finalization boundary."""

    patient_uid: str
    output_root: Path | str
    run_id: str = ""
    localization_kinds: Sequence[str] = ("dose",)
    write_query_context: bool = True
    write_localization_values_context: bool = True
    write_render_context: bool = True
    overwrite: bool = False
    _artifact_refs_by_id: dict[str, ArtifactRef] = field(default_factory=dict, init=False)
    _written_paths_by_artifact_id: dict[str, Path] = field(default_factory=dict, init=False)
    _persisted_lattice_kinds: set[str] = field(default_factory=set, init=False)
    _persisted_query_biopsy_indices: set[int] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self.patient_uid = str(self.patient_uid).strip()
        if self.patient_uid == "":
            raise ValueError("patient_uid cannot be empty")
        self.output_root = Path(self.output_root)
        self.run_id = str(self.run_id).strip()
        self.localization_kinds = tuple(_normal_token(localization_kind) for localization_kind in self.localization_kinds)
        if len(self.localization_kinds) == 0:
            raise ValueError("localization_kinds cannot be empty")
        self.write_query_context = bool(self.write_query_context)
        self.write_localization_values_context = bool(self.write_localization_values_context)
        self.write_render_context = bool(self.write_render_context)
        self.overwrite = bool(self.overwrite)

    @property
    def artifact_index_path(self) -> Path:
        """Return the patient context artifact index path."""
        return Path(self.output_root).joinpath("context", PATIENT_ARTIFACT_INDEX_FILENAME)

    @property
    def output_paths(self) -> tuple[Path, ...]:
        """Return written artifact paths including the patient artifact index."""
        paths = tuple(self._written_paths_by_artifact_id.values())
        if self._artifact_refs_by_id:
            return (*paths, self.artifact_index_path)
        return paths

    @property
    def artifact_refs(self) -> tuple[ArtifactRef, ...]:
        """Return artifact refs recorded in the patient artifact index."""
        return tuple(self._artifact_refs_by_id.values())

    def persist_localization_finalization(self, finalization: Any) -> None:
        """Persist context from a dose-localization finalization callback payload."""
        self.persist_dose_localization_context(
            lattice_context=finalization.lattice_context,
            biopsy_context=finalization.biopsy_context,
            localization_outputs=finalization.localization_outputs,
        )

    def persist_dose_localization_context(
        self,
        *,
        lattice_context: Any,
        biopsy_context: Any,
        localization_outputs: Any,
    ) -> PatientDoseContextPersistenceSummary:
        """Persist retained artifacts for one finalized biopsy dose-localization result."""
        localization_kind = _normal_token(localization_outputs.localization_kind)
        if localization_kind not in self.localization_kinds:
            return self.summary()

        if localization_kind not in self._persisted_lattice_kinds:
            lattice_plan = build_patient_dose_lattice_context_artifact_plan(lattice_context)
            self._remember_written_artifacts(
                lattice_plan.artifact_refs,
                write_patient_dose_context_zarr_arrays(
                    lattice_plan,
                    patient_dose_lattice_context_array_payload(lattice_context),
                    self.output_root,
                    overwrite=self.overwrite,
                ),
            )
            self._persisted_lattice_kinds.add(localization_kind)

        biopsy_index = int(biopsy_context.biopsy_index)
        if self.write_query_context and biopsy_index not in self._persisted_query_biopsy_indices:
            query_plan = build_patient_dose_biopsy_query_context_artifact_plan(biopsy_context)
            self._remember_written_artifacts(
                query_plan.artifact_refs,
                write_patient_dose_context_zarr_arrays(
                    query_plan,
                    patient_dose_biopsy_query_context_array_payload(biopsy_context),
                    self.output_root,
                    overwrite=self.overwrite,
                ),
            )
            self._persisted_query_biopsy_indices.add(biopsy_index)

        if self.write_localization_values_context:
            localization_plan = build_patient_dose_localization_context_artifact_plan(
                localization_outputs,
                patient_uid=self.patient_uid,
                biopsy_index=biopsy_index,
            )
            self._remember_written_artifacts(
                localization_plan.artifact_refs,
                write_patient_dose_context_zarr_arrays(
                    localization_plan,
                    patient_dose_localization_context_array_payload(localization_outputs),
                    self.output_root,
                    overwrite=self.overwrite,
                ),
            )

        if self.write_render_context and localization_kind == "dose":
            render_scene = build_dose_nn_render_scene_from_dataframe(
                localization_outputs.nearest_neighbour_dataframe,
                lattice_points=lattice_context.physical_coordinates,
                lattice_doses=lattice_context.sampled_values,
                metadata=DoseNNSceneMetadata(
                    patient_uid=self.patient_uid,
                    biopsy_roi=str(biopsy_context.roi),
                    biopsy_index=biopsy_index,
                    localization_kind=localization_kind,
                    result_column=str(localization_outputs.result_column),
                    source_label="patient_runner_mc_dose_finalization",
                ),
                result_column=localization_outputs.result_column,
            )
            render_plan, written_paths = write_dose_nn_render_context_zarr_artifact(
                render_scene,
                self.output_root,
                overwrite=self.overwrite,
            )
            self._remember_written_artifacts(render_plan.artifact_refs, written_paths)

        self._write_index()
        return self.summary()

    def summary(self) -> PatientDoseContextPersistenceSummary:
        """Return the current persistence summary without writing additional artifacts."""
        return PatientDoseContextPersistenceSummary(
            artifact_index_path=self.artifact_index_path,
            artifact_refs=self.artifact_refs,
            output_paths=self.output_paths,
        )

    def _remember_written_artifacts(
        self,
        artifact_refs: Sequence[ArtifactRef],
        written_paths_by_artifact_id: Mapping[str, Path],
    ) -> None:
        refs_by_id = {artifact_ref.artifact_id: artifact_ref for artifact_ref in artifact_refs}
        for artifact_id, written_path in written_paths_by_artifact_id.items():
            self._artifact_refs_by_id[artifact_id] = refs_by_id[artifact_id]
            self._written_paths_by_artifact_id[artifact_id] = Path(written_path)

    def _write_index(self) -> Path:
        index = PatientArtifactIndex(
            patient_uid=self.patient_uid,
            run_id=self.run_id,
            artifacts=self.artifact_refs,
            metadata={"producer": "patient_runner.dose_context_persistence"},
        )
        return write_patient_artifact_index(index, self.artifact_index_path, overwrite=True)


def _normal_token(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")
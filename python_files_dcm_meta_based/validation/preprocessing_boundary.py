"""Names for the supported migration checkpoints; no scientific imports.

This is a bounded selector shared by workers and validation commands, not a
pathway registration framework. Scientific dependencies remain in patient_runner.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessingBoundary:
    """On-disk evidence and successful stage at which it must be captured."""

    name: str
    stage_name: str
    directory: str
    schema_version: str
    capture_flag: str

    @property
    def manifest_name(self) -> str:
        return self.directory + "_checkpoint.json"

    @property
    def arrays_name(self) -> str:
        return self.directory + "_arrays.npz"


def preprocessing_boundary(name: str) -> PreprocessingBoundary:
    """Reject unimplemented checkpoints instead of guessing their products."""
    if name == "anatomical_qa":
        return PreprocessingBoundary(name, "anatomical_preprocessing", "anatomical",
                                     "anatomical_checkpoint_v1", "capture_anatomical_checkpoint")
    if name == "biopsy_preprocessing_shadow":
        return PreprocessingBoundary(name, "preprocessing", "biopsy_preprocessing",
                                     "biopsy_preprocessing_checkpoint_v1", "capture_biopsy_preprocessing_checkpoint")
    if name == "optimization_shadow":
        return PreprocessingBoundary(name, "optimization", "optimization",
                                     "optimization_checkpoint_v1", "capture_validation_checkpoint")
    raise ValueError("unsupported preprocessing checkpoint: " + str(name))


def requested_checkpoint_captures(metadata, checkpoint_name: str) -> tuple[str, ...]:
    """Resolve the common capture switch and historical flags before science."""
    result = []
    for flag, name in (
        ("capture_anatomical_checkpoint", "anatomical_qa"),
        ("capture_biopsy_preprocessing_checkpoint", "biopsy_preprocessing_shadow"),
        ("capture_validation_checkpoint", checkpoint_name),
    ):
        enabled = metadata.get(flag, False)
        if type(enabled) is not bool:
            raise TypeError(flag + " must be a boolean")
        if enabled:
            preprocessing_boundary(name)
            if name != "anatomical_qa" and name != checkpoint_name:
                raise ValueError("checkpoint capture requires matching pathway/checkpoint: " + name)
            if name not in result:
                result.append(name)
    return tuple(result)

"""Names for the two supported migration checkpoints; no scientific imports.

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
    raise ValueError("unsupported preprocessing checkpoint: " + str(name))

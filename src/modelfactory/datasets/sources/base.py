"""Abstract base for a labelled medical-image cohort source.

A `DatasetSource` is the thin adapter between a particular on-disk format
(PDDCA NRRD, DICOM RTSTRUCT, MSD NIfTI, Slicer .seg.nrrd, …) and the
nnUNetv2 raw-dataset writer in `modelfactory.datasets.convert`.

The orchestrator never reads source files directly — it asks the source
to discover() its cases and then load_image / load_mask per case. New
formats are added by writing one Source subclass; the orchestrator,
spec catalog, and per-fold training pipeline don't change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import SimpleITK as sitk


@dataclass(frozen=True)
class CaseRef:
    """Opaque pointer to one case in a source.

    The source attaches whatever it needs to its CaseRefs (image_path may
    be a single file for NRRD, a directory for DICOM series, etc.). The
    orchestrator only reads `case_id` (for output filenames) and
    `patient_id` (for split stratification), plus passes the CaseRef back
    to the source when it asks for image/mask data.
    """

    case_id: str
    patient_id: str
    image_path: Path
    label_paths: dict[str, Path] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetSource(ABC):
    """A source of labelled medical-image cases for one or more datasets.

    Source implementations are stateless w.r.t. any single conversion run;
    they can be instantiated with config (root paths, filters) and reused
    for multiple DatasetSpecs.
    """

    #: short identifier used in DatasetSpec.source_constraints lookup
    source_type: str = "base"

    @abstractmethod
    def discover(self, structures: Sequence[str]) -> list[CaseRef]:
        """Return CaseRefs that contain all of the requested structures.

        Caller passes the canonical structure names from the DatasetSpec;
        each Source is responsible for translating those to its own naming
        via the spec's StructureMapping.aliases dict before doing I/O.

        A case is included only if every requested structure can be loaded.
        Missing-structure cases are logged but not returned.
        """

    @abstractmethod
    def load_image(self, case: CaseRef) -> sitk.Image:
        """Load the case's primary image (3D CT in HU, or whatever the
        modality dictates). The returned image's geometry defines the
        reference frame for `load_mask`.
        """

    @abstractmethod
    def load_mask(
        self,
        case: CaseRef,
        canonical_name: str,
        ref_image: sitk.Image,
    ) -> sitk.Image:
        """Load one binary mask for the named structure.

        Must return a uint8 SimpleITK image with the same Size, Spacing,
        Origin and Direction as `ref_image`. Sources that read masks in a
        different frame are responsible for resampling (nearest-neighbour)
        before returning.
        """


__all__ = ["CaseRef", "DatasetSource"]

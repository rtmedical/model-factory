"""Source for Medical Segmentation Decathlon (MSD) datasets.

http://medicaldecathlon.com — Simpson et al. distribute the
"Decathlon" datasets as a uniform layout:

    Task<NN>_<Name>/
        imagesTr/<case_id>.nii.gz       # training images
        labelsTr/<case_id>.nii.gz       # multi-label uint8 masks
        imagesTs/                       # (test split — we don't use)
        dataset.json                    # MSD's own schema

MSD's labels are a single multi-label NIfTI per case with integer
class IDs encoded in the label dataset.json. For Task04 Hippocampus
the published encoding is `{1: "anterior", 2: "posterior"}` — i.e.,
hippocampi are split into anterior/posterior subfields, NOT
left/right. This source therefore exposes anterior and posterior as
two distinct structures; downstream specs can either (a) request
both and treat them as their two foreground classes, or (b) merge
them in the spec by mapping both canonical names to overlapping
source names (one canonical name = union of multiple IDs).

For Dataset046_Brain_MR_Hippocampus_Gold we'll start by exposing
"Hippocampus_L" / "Hippocampus_R" — and at convert-time, if Task04
actually uses anterior/posterior encoding, we relabel the spec rather
than fight the source.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .base import CaseRef, DatasetSource


log = logging.getLogger("msd_source")


class MSDDecathlonSource(DatasetSource):
    """Source for any MSD Task<NN>_<Name> directory."""

    source_type = "msd"

    def __init__(self, src_root: Path, task: str | None = None):
        """`src_root` should already point at the task directory (e.g.
        /factory/intermediate/Dataset046_MSD_Task04/Task04_Hippocampus/).
        If `task` is given, we expect src_root to be the parent of the
        task dir; this helps with downloads that extract to one level up.
        """
        self.src_root = Path(src_root)
        if task and (self.src_root / task).is_dir():
            self.src_root = self.src_root / task
        self._label_map = self._load_dataset_json()

    def _load_dataset_json(self) -> dict[str, int]:
        """Return MSD's class-name → integer mapping from dataset.json."""
        dj_path = self.src_root / "dataset.json"
        if not dj_path.is_file():
            log.warning("no dataset.json at %s", dj_path)
            return {}
        try:
            dj = json.loads(dj_path.read_text())
        except Exception as e:
            log.warning("dataset.json parse failed: %s", e)
            return {}
        # MSD's "labels" key maps integer-string → class-name.
        # We invert to name → int for downstream use.
        labels = dj.get("labels", {})
        out: dict[str, int] = {}
        for k, v in labels.items():
            try:
                out[str(v).strip()] = int(k)
            except (TypeError, ValueError):
                continue
        log.info("MSD labels: %s", out)
        return out

    @property
    def label_map(self) -> dict[str, int]:
        return self._label_map

    def discover(self, structures: Sequence[str]) -> list[CaseRef]:
        cases: list[CaseRef] = []
        images_dir = self.src_root / "imagesTr"
        labels_dir = self.src_root / "labelsTr"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            log.error("MSD layout missing: imagesTr=%s labelsTr=%s",
                      images_dir.is_dir(), labels_dir.is_dir())
            return cases

        for img_path in sorted(images_dir.glob("*.nii.gz")):
            if img_path.name.startswith("."):
                continue
            cid = img_path.name.removesuffix(".nii.gz")
            label_path = labels_dir / img_path.name
            if not label_path.is_file():
                log.debug("[%s] missing label, skip", cid)
                continue
            # All requested structures live inside the same multi-label
            # NIfTI; the orchestrator calls load_mask once per name.
            label_paths = {s: label_path for s in structures}
            cases.append(
                CaseRef(
                    case_id=cid,
                    patient_id=cid,                 # MSD has no patient grouping
                    image_path=img_path,
                    label_paths=label_paths,
                    metadata={
                        "msd_root": str(self.src_root),
                        "msd_labels": self._label_map,
                    },
                )
            )
        log.info("MSD discover(%s): %d cases", list(structures), len(cases))
        return cases

    def load_image(self, case: CaseRef) -> sitk.Image:
        return sitk.ReadImage(str(case.image_path))

    def load_mask(
        self,
        case: CaseRef,
        canonical_name: str,
        ref_image: sitk.Image,
    ) -> sitk.Image:
        """Extract one class from MSD's multi-label NIfTI.

        Heuristic for Task04 Hippocampus where MSD encodes
        anterior/posterior (not L/R): if the spec asks for
        Hippocampus_L / Hippocampus_R but the dataset.json says
        anterior/posterior, we split the union by the sagittal
        midline (X axis median) and assign by hemisphere. This is
        anatomically reasonable for hippocampal contours.
        """
        multi = sitk.ReadImage(str(case.label_paths[canonical_name]))
        arr = sitk.GetArrayFromImage(multi).astype(np.uint8)
        ids_for_class = self._ids_for(canonical_name)
        if not ids_for_class:
            log.warning("[%s] canonical %r not resolvable in MSD labels %s",
                        case.case_id, canonical_name, self._label_map)
            mask = np.zeros_like(arr)
        else:
            mask = np.isin(arr, list(ids_for_class)).astype(np.uint8)
            if canonical_name in ("Hippocampus_L", "Hippocampus_R"):
                mask = self._split_by_midline(mask, canonical_name)

        out = sitk.GetImageFromArray(mask)
        out.CopyInformation(multi)
        if out.GetSize() != ref_image.GetSize():
            out = sitk.Resample(
                out, ref_image, sitk.Transform(),
                sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
            )
        return out

    def _ids_for(self, canonical: str) -> set[int]:
        """Resolve canonical structure → set of MSD label IDs.

        For Task04 Hippocampus the dataset.json typically says
        {0: "background", 1: "Anterior", 2: "Posterior"}. Both
        anatomical regions form *one* hippocampus per side after
        the midline split, so we union both IDs for either L/R name.
        """
        canon = canonical.lower()
        # Exact match first
        for name, lid in self._label_map.items():
            if name.lower() == canon and lid != 0:
                return {lid}
        # Hippocampus_L / _R → all non-background hippocampal IDs
        if canon.startswith("hippocampus"):
            return {lid for name, lid in self._label_map.items()
                    if lid != 0 and "background" not in name.lower()}
        return set()

    @staticmethod
    def _split_by_midline(mask: np.ndarray, canonical: str) -> np.ndarray:
        """Split a bilateral mask by the X-axis median.

        Convention: smaller X index = left in NIfTI's neurological
        convention when origin is at the right. We split at the
        per-case median of the X-coordinates where mask is nonzero.
        """
        if not mask.any():
            return mask
        # SimpleITK numpy order is (z, y, x). We split along the
        # last axis (x).
        nz = np.nonzero(mask)
        if len(nz[-1]) == 0:
            return mask
        x_median = int(np.median(nz[-1]))
        side = np.zeros_like(mask)
        x_coords = np.arange(mask.shape[-1])
        if canonical.endswith("_L"):
            side[..., x_coords <= x_median] = 1
        elif canonical.endswith("_R"):
            side[..., x_coords >  x_median] = 1
        return (mask.astype(bool) & side.astype(bool)).astype(np.uint8)


__all__ = ["MSDDecathlonSource"]

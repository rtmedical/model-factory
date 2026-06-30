"""PDDCA NRRD source (head & neck manual contours).

PDDCA on-disk layout:
    <src_root>/PDDCA-<version>_part{1,2,3,4}/<case_id>/img.nrrd
    <src_root>/PDDCA-<version>_part{1,2,3,4}/<case_id>/structures/<Name>.nrrd

Each structure is a binary uint8 NRRD aligned to img.nrrd. Case IDs are
shared across PDDCA versions (same patient, re-contoured). For training
we pin to one version (typically 1.4.1 — newest contours, all 9 structures).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import SimpleITK as sitk

from .base import CaseRef, DatasetSource


log = logging.getLogger("pddca_source")


class PDDCASource(DatasetSource):
    """PDDCA NRRD source pinned to one version."""

    source_type = "pddca"

    def __init__(self, src_root: Path, version: str = "1.4.1"):
        self.src_root = Path(src_root)
        self.version = version

    def _part_dirs(self) -> list[Path]:
        return sorted(self.src_root.glob(f"PDDCA-{self.version}_part*"))

    def discover(self, structures: Sequence[str]) -> list[CaseRef]:
        required = list(structures)
        cases: list[CaseRef] = []
        skipped: dict[str, list[str]] = {}
        seen: set[str] = set()

        for part_dir in self._part_dirs():
            for case_dir in sorted(part_dir.iterdir()):
                if not case_dir.is_dir() or not (case_dir / "img.nrrd").is_file():
                    continue
                cid = case_dir.name
                if cid in seen:
                    log.debug("duplicate case %s across parts; keeping first", cid)
                    continue
                seen.add(cid)

                struct_dir = case_dir / "structures"
                label_paths: dict[str, Path] = {}
                missing: list[str] = []
                for name in required:
                    nrrd = struct_dir / f"{name}.nrrd"
                    if nrrd.is_file():
                        label_paths[name] = nrrd
                    else:
                        missing.append(name)
                if missing:
                    skipped[cid] = missing
                    continue

                cases.append(
                    CaseRef(
                        case_id=cid,
                        patient_id=cid,                 # PDDCA case == patient
                        image_path=case_dir / "img.nrrd",
                        label_paths=label_paths,
                        metadata={"pddca_version": self.version},
                    )
                )

        log.info(
            "PDDCA-%s discover(%s): %d cases ok, %d skipped",
            self.version, required, len(cases), len(skipped),
        )
        if skipped:
            sample = list(skipped.items())[:3]
            log.info("  skip samples: %s", sample)
        return cases

    def load_image(self, case: CaseRef) -> sitk.Image:
        return sitk.ReadImage(str(case.image_path))

    def load_mask(
        self,
        case: CaseRef,
        canonical_name: str,
        ref_image: sitk.Image,
    ) -> sitk.Image:
        nrrd = case.label_paths[canonical_name]
        img = sitk.ReadImage(str(nrrd))
        if img.GetSize() != ref_image.GetSize():
            img = sitk.Resample(
                img, ref_image, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
            )
        elif img.GetPixelID() != sitk.sitkUInt8:
            img = sitk.Cast(img, sitk.sitkUInt8)
        return img


__all__ = ["PDDCASource"]

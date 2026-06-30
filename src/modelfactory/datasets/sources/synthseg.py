"""Source for brain MR T1s with SynthSeg-generated per-structure binary masks.

The label-generation step (see `modelfactory.labelers.fomo_synthseg`) runs
SynthSeg over a curated subject set, extracts the four canonical OAR
classes (Brainstem, Hippocampus_L, Hippocampus_R, Cerebellum), applies
largest-connected-component + QC filtering, and writes per-subject
images and per-structure binary masks in this layout:

    intermediate_root/
        <subject_id>/
            image.nii.gz                  # T1, original geometry
            structures/
                Brainstem.nii.gz          # uint8 binary
                Hippocampus_L.nii.gz
                Hippocampus_R.nii.gz
                Cerebellum.nii.gz
            qc.json                       # SynthSeg QC score + per-class volumes

The source is intentionally generic over the structure names — it loads
whatever the spec asks for, and discover() filters out subjects missing
any requested structure (i.e., a subject whose SynthSeg run produced a
fragmented brainstem and was dropped during post-processing won't
appear in discover()'s output).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import SimpleITK as sitk

from .base import CaseRef, DatasetSource


log = logging.getLogger("synthseg_source")


class SynthSegSource(DatasetSource):
    """Source for SynthSeg-labelled brain MR cohorts (e.g. FOMO300K subsets)."""

    source_type = "synthseg"

    def __init__(
        self,
        intermediate_root: Path,
        modality: str = "T1w",
        extra_roots: Sequence[Path] | None = None,
    ):
        """SynthSeg-labelled brain MR source.

        Parameters
        ----------
        intermediate_root:
            Primary structures tree (typically `synthseg_base` for sub-cortical
            datasets, or `synthseg_parc` for cortical datasets).
        modality:
            Carried into per-case metadata (T1w / T2w / etc.).
        extra_roots:
            Optional additional structures trees. When a requested canonical
            structure is missing from the primary tree, each extra root is
            searched in order. Used by D063 Brain_MR_FullBrain_Generalist to
            fuse the `synthseg_base` (sub-cortical) and `synthseg_parc`
            (cortical) trees into one 34-class spec. The cohort is the
            INTERSECTION of subjects present in every tree.
        """
        self.intermediate_root = Path(intermediate_root)
        self.modality = modality
        self.extra_roots = [Path(r) for r in (extra_roots or [])]

    def _trees(self) -> list[Path]:
        """All structures trees to search, primary first."""
        return [self.intermediate_root, *self.extra_roots]

    def discover(self, structures: Sequence[str]) -> list[CaseRef]:
        required = list(structures)
        cases: list[CaseRef] = []
        skipped: dict[str, list[str]] = {}

        for tree in self._trees():
            if not tree.is_dir():
                log.error("structures tree missing: %s", tree)
                return cases

        # Cohort = subjects present in EVERY tree (intersection).
        subj_sets = [
            {p.name for p in tree.iterdir() if p.is_dir()}
            for tree in self._trees()
        ]
        common = sorted(set.intersection(*subj_sets)) if subj_sets else []

        for subj_name in common:
            primary_subj = self.intermediate_root / subj_name
            image_path = primary_subj / "image.nii.gz"
            if not image_path.is_file():
                continue

            label_paths: dict[str, Path] = {}
            missing: list[str] = []
            for s in required:
                # Search each tree in order; first hit wins.
                found = None
                for tree in self._trees():
                    p = tree / subj_name / "structures" / f"{s}.nii.gz"
                    if p.is_file():
                        found = p
                        break
                if found is not None:
                    label_paths[s] = found
                else:
                    missing.append(s)
            if missing:
                skipped[subj_name] = missing
                continue

            cases.append(
                CaseRef(
                    case_id=subj_name,
                    patient_id=subj_name,
                    image_path=image_path,
                    label_paths=label_paths,
                    metadata={
                        "modality": self.modality,
                        "label_source": "SynthSeg-2.0-robust",
                    },
                )
            )

        log.info(
            "SynthSeg discover(%s): %d cases ok, %d skipped",
            required, len(cases), len(skipped),
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
        img = sitk.ReadImage(str(case.label_paths[canonical_name]))
        if img.GetSize() != ref_image.GetSize():
            img = sitk.Resample(
                img, ref_image, sitk.Transform(),
                sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
            )
        elif img.GetPixelID() != sitk.sitkUInt8:
            img = sitk.Cast(img, sitk.sitkUInt8)
        return img


__all__ = ["SynthSegSource"]

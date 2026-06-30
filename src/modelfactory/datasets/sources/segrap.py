"""SegRap2023 source — H&N OAR + NPC GTV cohort, MICCAI 2023 challenge.

Provenance: SegRap 2023 challenge data from Sichuan Cancer Hospital & Institute,
released by HiLab-git at https://segrap2023.grand-challenge.org/dataset/.
Licence per the release's `dataset_task001.json`: CC-BY-SA-4.0 (same tier as
MSD; share-alike applies to derived weights).

On-disk layout once unzipped by `scripts/download_segrap2023.sh`:

    <src_root>/
        SegRap2023_Training_Set_120cases/
            segrap_0000/                    # IDs 0000..0119
                image.nii.gz                # non-contrast head-and-neck CT
                image_contrast.nii.gz       # pre-aligned contrast-enhanced CT
                Brain.nii.gz                # 45 OAR binary masks (flat, no Mask/ subdir)
                BrainStem.nii.gz
                ...
                GTVp.nii.gz                 # primary NPC GTV
                GTVnd.nii.gz                # nodal GTV
        SegRap2023_Training_Set_120cases_Update_Labels/
            segrap_0000/
                Brain.nii.gz                # 45 OAR masks only (corrected; no GTV, no image)
                ...
        SegRap2023_Validation_Set_20cases/  # SKIPPED — images only, no labels.
            head-neck-ct/segrap_0120.mha
            head-neck-contrast-enhanced-ct/segrap_0120.mha
            readme.txt.txt

Label-source policy. For OAR canonicals we prefer the `_Update_Labels`
directory (corrected per-organ masks released after the challenge) and
fall back to the main `_Training_Set_120cases` directory if a file is
missing there. GTVs only live in the main directory; OAR specs never
need them, so the policy is internal to load_mask.

Multi-phase handling. The two CT phases are pre-aligned, so the same
mask files apply to both. We expose each phase as an independent CaseRef
whose case_id is suffixed with `_nc` or `_ce`, while keeping patient_id
at the bare `segrap_NNNN`. The orchestrator's `_write_splits` groups by
patient_id, so a patient's NC and CE volumes always land in the same
fold — no test-time leakage.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .base import CaseRef, DatasetSource


log = logging.getLogger("segrap_source")


# Suffixes used to distinguish a patient's two CT phases as separate cases.
NC_SUFFIX = "_nc"
CE_SUFFIX = "_ce"

# Filenames inside each segrap_<NNNN>/ directory of the main training set.
_PHASE_FILENAME = {
    NC_SUFFIX: "image.nii.gz",
    CE_SUFFIX: "image_contrast.nii.gz",
}

# Directory names under src_root.
_MAIN_DIR = "SegRap2023_Training_Set_120cases"
_UPDATED_DIR = "SegRap2023_Training_Set_120cases_Update_Labels"

# GTV labels only exist in the main training dir; never look for them in
# the Update_Labels mirror.
_GTV_NAMES = frozenset({"GTVp", "GTVnd"})


class SegRapSource(DatasetSource):
    """SegRap2023 NIfTI source for the training set's two CT phases.

    Args:
        src_root: directory containing the unzipped SegRap2023_*_Set_* folders.
        phase: one of "both" (default — emit NC and CE cases per patient),
            "contrast" (CE only), or "noncontrast" (NC only).
        prefer_updated_labels: if True (default), look in
            `_Update_Labels/segrap_NNNN/<name>.nii.gz` first for OAR masks
            and fall back to the main training dir. Set False to pin to the
            originally-released labels.
        splits: kept for API compatibility with the spec's source_constraints;
            the adapter ignores anything other than "Training" because the
            validation set ships without labels.
    """

    source_type = "segrap"

    def __init__(
        self,
        src_root: Path,
        phase: str = "both",
        prefer_updated_labels: bool = True,
        splits: Sequence[str] = ("Training",),
        **_ignored: object,
    ):
        self.src_root = Path(src_root)
        phase = phase.lower()
        if phase not in ("both", "contrast", "noncontrast"):
            raise ValueError(f"phase must be both|contrast|noncontrast, got {phase!r}")
        self.phase = phase
        self.prefer_updated_labels = prefer_updated_labels

        usable = [s for s in splits if s.lower() == "training"]
        if not usable:
            usable = ["Training"]
        if list(splits) != usable:
            log.info(
                "SegRap2023 validation set has no labels — ignoring non-Training splits %s",
                [s for s in splits if s not in usable],
            )
        self.splits = tuple(usable)

        self._main_root = self.src_root / _MAIN_DIR
        self._updated_root = self.src_root / _UPDATED_DIR

    # ── path resolution ──────────────────────────────────────────────────

    def _patient_dirs(self) -> list[Path]:
        if not self._main_root.is_dir():
            log.error("missing main training root: %s", self._main_root)
            return []
        return sorted(p for p in self._main_root.iterdir()
                      if p.is_dir() and p.name.startswith("segrap_"))

    def _phases_for_case(self) -> tuple[str, ...]:
        if self.phase == "contrast":
            return (CE_SUFFIX,)
        if self.phase == "noncontrast":
            return (NC_SUFFIX,)
        return (NC_SUFFIX, CE_SUFFIX)

    def _mask_path(self, segrap_name: str, patient_dir: Path) -> Path | None:
        """Resolve where a single OAR/GTV mask lives, applying the
        prefer-updated-labels policy. Returns None if neither dir has it."""
        if segrap_name in _GTV_NAMES:
            candidate = patient_dir / f"{segrap_name}.nii.gz"
            return candidate if candidate.is_file() else None

        if self.prefer_updated_labels:
            updated = self._updated_root / patient_dir.name / f"{segrap_name}.nii.gz"
            if updated.is_file():
                return updated
        main = patient_dir / f"{segrap_name}.nii.gz"
        if main.is_file():
            return main
        if not self.prefer_updated_labels:
            updated = self._updated_root / patient_dir.name / f"{segrap_name}.nii.gz"
            if updated.is_file():
                return updated
        return None

    # ── discovery ────────────────────────────────────────────────────────

    def discover(self, structures: Sequence[str]) -> list[CaseRef]:
        required = list(structures)
        phases = self._phases_for_case()
        cases: list[CaseRef] = []
        skipped: dict[str, list[str]] = {}

        for patient_dir in self._patient_dirs():
            patient_id = patient_dir.name

            label_paths: dict[str, Path] = {}
            label_origins: dict[str, str] = {}
            missing: list[str] = []
            for segrap_name in required:
                p = self._mask_path(segrap_name, patient_dir)
                if p is None:
                    missing.append(segrap_name)
                    continue
                label_paths[segrap_name] = p
                # Walk up to the parent split dir to identify which release
                # (original vs Update_Labels) supplied this mask.
                label_origins[segrap_name] = (
                    "Update_Labels" if _UPDATED_DIR in p.parts else "original"
                )
            if missing:
                skipped[patient_id] = missing
                continue

            updated_count = sum(1 for v in label_origins.values() if v == "Update_Labels")
            for suffix in phases:
                img_path = patient_dir / _PHASE_FILENAME[suffix]
                if not img_path.is_file():
                    log.debug("[%s] phase %s missing %s; skipping phase",
                              patient_id, suffix, img_path.name)
                    continue
                cases.append(
                    CaseRef(
                        case_id=f"{patient_id}{suffix}",
                        patient_id=patient_id,
                        image_path=img_path,
                        label_paths=dict(label_paths),
                        metadata={
                            "segrap_phase": "noncontrast" if suffix == NC_SUFFIX else "contrast",
                            "segrap_split": _MAIN_DIR,
                            "labels_from_update_set": updated_count,
                            "labels_from_original_set": len(label_paths) - updated_count,
                        },
                    )
                )

        log.info(
            "SegRap discover(%s; phase=%s; updated=%s): %d cases, %d patients skipped",
            required, self.phase, self.prefer_updated_labels,
            len(cases), len(skipped),
        )
        if skipped:
            sample = list(skipped.items())[:3]
            log.info("  skip samples (patient → missing structures): %s", sample)
        return cases

    # ── I/O ──────────────────────────────────────────────────────────────

    def load_image(self, case: CaseRef) -> sitk.Image:
        return sitk.ReadImage(str(case.image_path))

    def load_mask(
        self,
        case: CaseRef,
        canonical_name: str,
        ref_image: sitk.Image,
    ) -> sitk.Image:
        mask_path = case.label_paths[canonical_name]
        img = sitk.ReadImage(str(mask_path))
        if img.GetSize() != ref_image.GetSize():
            img = sitk.Resample(
                img, ref_image, sitk.Transform(),
                sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
            )
        elif img.GetPixelID() != sitk.sitkUInt8:
            img = sitk.Cast(img, sitk.sitkUInt8)
        # Defensive: ensure values are {0, 1}. SegRap masks are already
        # binary uint8, but coerce in case a resample left intermediate
        # values from a non-NN path.
        arr = sitk.GetArrayFromImage(img)
        if arr.dtype != np.uint8 or arr.max() > 1:
            arr = (arr > 0).astype(np.uint8)
            out = sitk.GetImageFromArray(arr)
            out.CopyInformation(img)
            img = out
        return img


__all__ = ["SegRapSource", "NC_SUFFIX", "CE_SUFFIX"]

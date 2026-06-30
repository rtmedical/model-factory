"""Source for the Longitudinal-CT (Tübingen) melanoma whole-body CT dataset.

Küstner et al, Sci Data 2026. FDAT DOI 10.57754/FDAT.75kj1-64747. CC-BY-4.0.

On-disk layout (assumed; see docs/longitudinal_ct_tubingen.md for full schema):

    <src_root>/
        inputsTr/
            <patient>.csv                      # one row per lesion (lesion_id, lesion_type, ...)
            <patient>_<BL|FU>_<NN>.json        # "Points of interest" (centroids — unused here)
            <patient>_<BL|FU>_img_<NN>.nii.gz  # CT volume
            <patient>_<BL|FU>_mask_<NN>.nii.gz # BL masks (only)
        targetsTr/
            <patient>_FU_mask_<NN>.nii.gz      # FU masks

Where:
- <patient> is a 10-char hex hash
- <NN> is a series index within a timepoint (one timepoint may have multiple
  series, e.g. a chest scan and an abdomen scan)

The mask voxels encode the per-patient `lesion_id` (1..N) — NOT the anatomy
class. The CSV's `lesion_type` column maps `lesion_id` → anatomy string.
`load_mask` joins them via the canonical → CSV-type lookup in `_longct_codes`.

Single case = one (image, mask) pair = one `(patient, BL|FU, NN)` triple.
"""
from __future__ import annotations

import csv
import logging
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from ._longct_codes import CANONICAL_TO_CSV_TYPES, WILDCARD_CANONICAL
from .base import CaseRef, DatasetSource


log = logging.getLogger("longct_source")

# <patient>_<BL|FU>_img_<NN>.nii.gz
IMG_RE = re.compile(r"^([0-9a-f]+)_(BL|FU)_img_(\d+)\.nii\.gz$")


class LongitudinalCTSource(DatasetSource):
    """Source adapter for the Tübingen Longitudinal-CT melanoma cohort."""

    source_type = "longct"

    def __init__(self, src_root: Path):
        self.src_root = Path(src_root)
        self.inputs_dir = self.src_root / "inputsTr"
        self.targets_dir = self.src_root / "targetsTr"
        if not self.inputs_dir.is_dir():
            raise FileNotFoundError(f"missing inputsTr/ under {self.src_root}")
        if not self.targets_dir.is_dir():
            raise FileNotFoundError(f"missing targetsTr/ under {self.src_root}")
        # patient_id → {(timepoint, series_idx): {lesion_id: lesion_type}}
        self._csv_cache: dict[str, dict[tuple[str, int], dict[int, str]]] = {}

    # ── CSV loading (cached) ─────────────────────────────────────────────

    def _load_patient_csv(self, patient_id: str) -> dict[tuple[str, int], dict[int, str]]:
        cached = self._csv_cache.get(patient_id)
        if cached is not None:
            return cached
        path = self.inputs_dir / f"{patient_id}.csv"
        out: dict[tuple[str, int], dict[int, str]] = {}
        if not path.is_file():
            log.warning("[%s] missing CSV at %s", patient_id, path)
            self._csv_cache[patient_id] = out
            return out
        with path.open() as f:
            for row in csv.DictReader(f):
                lid_raw = (row.get("lesion_id") or "").strip()
                ltype = (row.get("lesion_type") or "").strip()
                if not lid_raw or not ltype:
                    continue
                try:
                    lid = int(lid_raw)
                except ValueError:
                    continue
                for side, key in (("BL", "img_id_bl"), ("FU", "img_id_fu")):
                    raw = (row.get(key) or "").strip()
                    if not raw:
                        continue
                    try:
                        si = int(raw)
                    except ValueError:
                        continue
                    out.setdefault((side, si), {})[lid] = ltype
        self._csv_cache[patient_id] = out
        return out

    # ── DatasetSource API ────────────────────────────────────────────────

    def discover(self, structures: Sequence[str]) -> list[CaseRef]:
        """Walk inputsTr/ images, pair each with its mask (BL: inputsTr, FU: targetsTr).

        `structures` is the canonical-name list from the DatasetSpec. We don't
        per-structure-filter at discovery time; the orchestrator's
        `empty_label` check drops cases whose union of requested structures
        is all-zero.
        """
        cases: list[CaseRef] = []
        n_skipped = 0
        for img_path in sorted(self.inputs_dir.glob("*_img_*.nii.gz")):
            m = IMG_RE.match(img_path.name)
            if not m:
                continue
            patient_id, timepoint, series_str = m.group(1), m.group(2), m.group(3)
            series_idx = int(series_str)
            mask_name = f"{patient_id}_{timepoint}_mask_{series_str}.nii.gz"
            mask_path = (
                self.inputs_dir / mask_name if timepoint == "BL"
                else self.targets_dir / mask_name
            )
            if not mask_path.is_file():
                n_skipped += 1
                log.debug("[%s_%s_%s] missing mask: %s", patient_id, timepoint, series_str, mask_path)
                continue
            csv_for_patient = self._load_patient_csv(patient_id)
            lid_lookup = csv_for_patient.get((timepoint, series_idx), {})
            case_id = f"{patient_id}_{timepoint}_{series_str}"
            cases.append(
                CaseRef(
                    case_id=case_id,
                    patient_id=patient_id,
                    image_path=img_path,
                    label_paths={
                        # one shared mask for every requested structure
                        # (per-class extraction happens in load_mask)
                        s: mask_path for s in structures
                    },
                    metadata={
                        "timepoint": timepoint,
                        "series_idx": series_idx,
                        "n_lesions_in_csv": len(lid_lookup),
                        "csv_path": str(self.inputs_dir / f"{patient_id}.csv"),
                    },
                )
            )
        log.info(
            "longct discover(%s): %d cases (skipped %d missing-mask)",
            list(structures), len(cases), n_skipped,
        )
        return cases

    def load_image(self, case: CaseRef) -> sitk.Image:
        return sitk.ReadImage(str(case.image_path))

    def load_mask(
        self,
        case: CaseRef,
        canonical_name: str,
        ref_image: sitk.Image,
    ) -> sitk.Image:
        """Extract a binary mask for one canonical anatomy class.

        Three paths:
          1. canonical_name == WILDCARD_CANONICAL ("AnyMetastasis") or "*"
             → binary union of every non-zero voxel.
          2. canonical_name is a key in CANONICAL_TO_CSV_TYPES → join the
             CSV to find lesion_ids whose lesion_type matches; binary mask
             of voxels whose value is in that set.
          3. Otherwise → raise KeyError so the orchestrator emits
             `missing_struct` for that case.
        """
        mask_path = case.label_paths[canonical_name]
        raw_img = sitk.ReadImage(str(mask_path))
        raw = sitk.GetArrayFromImage(raw_img)

        if canonical_name in (WILDCARD_CANONICAL, "*"):
            mask = (raw != 0).astype(np.uint8)
        else:
            csv_types = CANONICAL_TO_CSV_TYPES.get(canonical_name)
            if csv_types is None:
                raise KeyError(
                    f"unknown canonical name {canonical_name!r} for longct source; "
                    f"add it to CANONICAL_TO_CSV_TYPES in _longct_codes.py"
                )
            csv_for_patient = self._load_patient_csv(case.patient_id)
            timepoint = case.metadata["timepoint"]
            series_idx = int(case.metadata["series_idx"])
            lid_lookup = csv_for_patient.get((timepoint, series_idx), {})
            wanted_lids = [
                lid for lid, ltype in lid_lookup.items() if ltype in csv_types
            ]
            if not wanted_lids:
                mask = np.zeros_like(raw, dtype=np.uint8)
            else:
                mask = np.isin(raw, wanted_lids).astype(np.uint8)

        out = sitk.GetImageFromArray(mask)
        out.CopyInformation(raw_img)
        if out.GetSize() != ref_image.GetSize():
            out = sitk.Resample(
                out, ref_image, sitk.Transform(),
                sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
            )
        return out


__all__ = ["LongitudinalCTSource"]

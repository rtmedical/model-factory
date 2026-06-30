"""TotalSegmentator v2 dataset source.

The TS v2 release (Zenodo 10047292, CC-BY-4.0) ships 1228 CT cases at:

    <src_root>/
        meta.csv                              # scan-level metadata
        s0000/  ct.nii.gz                     # 3D CT in HU
                segmentations/<name>.nii.gz   # 117 binary uint8 masks, one per structure

Per-structure NIfTI files are binary (label values 0/1). For cases whose
FOV doesn't include a structure (e.g. iliac vessels on a head-only CT),
the file is still present but all-zero. The convert orchestrator's
`empty_label` filter drops cases whose entire selected structure set is
all-zero, so a TS-organs spec on head-only CTs naturally skips them.

The `meta.csv` is parsed and attached to each CaseRef's metadata. This
lets downstream splits stratify by manufacturer / institute / study_type
(important: 88% Siemens, 73% one institute — a single-source bias to
manage at validation time).
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Sequence
from pathlib import Path

import SimpleITK as sitk

from .base import CaseRef, DatasetSource


log = logging.getLogger("totalseg_source")


class TotalSegSource(DatasetSource):
    """Source for the TotalSegmentator v2 dataset (Zenodo bundle)."""

    source_type = "totalseg"

    def __init__(self, src_root: Path, meta_csv: Path | None = None):
        self.src_root = Path(src_root)
        # default: look for meta.csv next to the case dirs
        if meta_csv is None:
            cand = self.src_root / "meta.csv"
            meta_csv = cand if cand.is_file() else None
        self._meta = self._load_meta(meta_csv) if meta_csv else {}

    @staticmethod
    def _load_meta(csv_path: Path) -> dict[str, dict[str, str]]:
        try:
            with csv_path.open(encoding="utf-8-sig") as f:
                reader = csv.DictReader(f, delimiter=";")
                out = {row["image_id"]: row for row in reader if row.get("image_id")}
            log.info("loaded TS meta.csv: %d rows", len(out))
            return out
        except Exception as e:
            log.warning("could not load %s: %s", csv_path, e)
            return {}

    def discover(self, structures: Sequence[str]) -> list[CaseRef]:
        required = list(structures)
        cases: list[CaseRef] = []
        missing_struct_count: dict[str, int] = {}

        for case_dir in sorted(self.src_root.iterdir()):
            if not case_dir.is_dir() or not case_dir.name.startswith("s"):
                continue
            ct = case_dir / "ct.nii.gz"
            seg_dir = case_dir / "segmentations"
            if not ct.is_file() or not seg_dir.is_dir():
                continue

            label_paths: dict[str, Path] = {}
            missing: list[str] = []
            for s in required:
                nii = seg_dir / f"{s}.nii.gz"
                if nii.is_file():
                    label_paths[s] = nii
                else:
                    missing.append(s)
            if missing:
                for m in missing:
                    missing_struct_count[m] = missing_struct_count.get(m, 0) + 1
                continue

            row = self._meta.get(case_dir.name, {})
            cases.append(
                CaseRef(
                    case_id=case_dir.name,
                    patient_id=case_dir.name,        # TS v2: 1 series = 1 patient
                    image_path=ct,
                    label_paths=label_paths,
                    metadata={
                        "ts_split": row.get("split", "train"),
                        "study_type": row.get("study_type", ""),
                        "institute": row.get("institute", ""),
                        "gender": row.get("gender", ""),
                        "manufacturer": row.get("manufacturer", ""),
                        "scanner_model": row.get("scanner_model", ""),
                        "pathology": row.get("pathology", ""),
                    },
                )
            )

        log.info(
            "TS v2 discover(%d structures): %d cases ok",
            len(required), len(cases),
        )
        if missing_struct_count:
            log.info("  per-structure missing-file counts: %s", missing_struct_count)
        return cases

    def load_image(self, case: CaseRef) -> sitk.Image:
        return sitk.ReadImage(str(case.image_path))

    def load_mask(
        self,
        case: CaseRef,
        canonical_name: str,
        ref_image: sitk.Image,
    ) -> sitk.Image:
        nii = case.label_paths[canonical_name]
        img = sitk.ReadImage(str(nii))
        if img.GetSize() != ref_image.GetSize():
            img = sitk.Resample(
                img, ref_image, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
            )
        elif img.GetPixelID() != sitk.sitkUInt8:
            img = sitk.Cast(img, sitk.sitkUInt8)
        return img


__all__ = ["TotalSegSource"]

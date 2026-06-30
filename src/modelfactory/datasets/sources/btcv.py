"""Source for BTCV (Beyond the Cranial Vault) abdominal CT atlas.

BTCV Multi-Atlas Labeling 2015 (Synapse 3193805, CC-BY-3.0/TCIA) ships
50 portal-venous abdominal CT cases with one multi-label uint8 NIfTI
per case encoding 13 organ classes. The on-disk layout is::

    <src_root>/
        img/img####.nii.gz       # 3D CT in HU
        label/label####.nii.gz   # 3D uint8 multi-label mask

Per-case correspondence is via the numeric stem: ``img0001.nii.gz`` ↔
``label0001.nii.gz``. The integer label codes are::

    1  spleen                    8   aorta
    2  kidney_right              9   inferior_vena_cava
    3  kidney_left              10   portal_vein_and_splenic_vein
    4  gallbladder              11   pancreas
    5  esophagus                12   adrenal_gland_right
    6  liver                    13   adrenal_gland_left
    7  stomach

These names are the source-side aliases that specs declare via
``aliases={"btcv": "<name>"}`` in their StructureMapping entries.

Why a dedicated source (rather than reusing MSD's adapter)? BTCV's
class IDs are *hard-coded by the challenge* — there is no per-case
``dataset.json`` shipped with the cohort, so the adapter has to ship
the mapping itself. That's the only meaningful difference from the
MSD shape.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .base import CaseRef, DatasetSource


log = logging.getLogger("btcv_source")


#: Canonical BTCV integer label → snake_case structure name.
#: Order matches the challenge's official labelling guide.
BTCV_ID_TO_NAME: dict[int, str] = {
    1:  "spleen",
    2:  "kidney_right",
    3:  "kidney_left",
    4:  "gallbladder",
    5:  "esophagus",
    6:  "liver",
    7:  "stomach",
    8:  "aorta",
    9:  "inferior_vena_cava",
    10: "portal_vein_and_splenic_vein",
    11: "pancreas",
    12: "adrenal_gland_right",
    13: "adrenal_gland_left",
}
BTCV_NAME_TO_ID: dict[str, int] = {v: k for k, v in BTCV_ID_TO_NAME.items()}


_IMG_RE = re.compile(r"^img(?P<num>\d+)\.nii(?:\.gz)?$")


class BTCVSource(DatasetSource):
    """Source for the BTCV Multi-Atlas Labeling 2015 abdominal cohort."""

    source_type = "btcv"

    def __init__(self, src_root: Path):
        self.src_root = Path(src_root)
        # Accept either a flat layout (img/, label/) or the original
        # Synapse archive layout (RawData/Training/img, RawData/Training/label).
        for candidate in (self.src_root, self.src_root / "RawData" / "Training"):
            if (candidate / "img").is_dir() and (candidate / "label").is_dir():
                self.images_dir = candidate / "img"
                self.labels_dir = candidate / "label"
                break
        else:
            raise FileNotFoundError(
                f"BTCV layout not found under {self.src_root}; "
                f"expected img/ and label/ subdirs"
            )

    def discover(self, structures: Sequence[str]) -> list[CaseRef]:
        unknown = [s for s in structures if s not in BTCV_NAME_TO_ID]
        if unknown:
            raise ValueError(
                f"BTCV does not provide structures {unknown}; "
                f"available: {sorted(BTCV_NAME_TO_ID)}"
            )

        cases: list[CaseRef] = []
        for img_path in sorted(self.images_dir.iterdir()):
            m = _IMG_RE.match(img_path.name)
            if m is None:
                continue
            num = m.group("num")
            label_path = self.labels_dir / f"label{num}.nii.gz"
            if not label_path.is_file():
                log.debug("[%s] no label, skip", img_path.name)
                continue
            cid = f"btcv_{num}"
            label_paths = {s: label_path for s in structures}
            cases.append(
                CaseRef(
                    case_id=cid,
                    patient_id=cid,        # BTCV: one scan per subject
                    image_path=img_path,
                    label_paths=label_paths,
                    metadata={"btcv_num": num},
                )
            )
        log.info("BTCV discover(%s): %d cases", list(structures), len(cases))
        return cases

    def load_image(self, case: CaseRef) -> sitk.Image:
        return sitk.ReadImage(str(case.image_path))

    def load_mask(
        self,
        case: CaseRef,
        canonical_name: str,
        ref_image: sitk.Image,
    ) -> sitk.Image:
        lid = BTCV_NAME_TO_ID.get(canonical_name)
        if lid is None:
            raise KeyError(f"unknown BTCV structure {canonical_name!r}")
        multi = sitk.ReadImage(str(case.label_paths[canonical_name]))
        arr = sitk.GetArrayFromImage(multi)
        mask = (arr == lid).astype(np.uint8)
        out = sitk.GetImageFromArray(mask)
        out.CopyInformation(multi)
        if out.GetSize() != ref_image.GetSize():
            out = sitk.Resample(
                out, ref_image, sitk.Transform(),
                sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
            )
        return out


__all__ = ["BTCVSource", "BTCV_ID_TO_NAME", "BTCV_NAME_TO_ID"]

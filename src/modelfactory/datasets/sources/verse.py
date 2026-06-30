"""Source for VerSe (Vertebrae Segmentation) 2020 challenge data.

VerSe 2020 (Sekuboyina et al., Zenodo 4153679 / 4153982 / 4505493,
CC-BY-4.0) ships 374 CT cases across three splits (training,
validation, test), each with a multi-label uint8 NIfTI per case
labelling individual vertebrae by integer ID. The on-disk layout
follows the BIDS-derivatives convention::

    <src_root>/
        rawdata/sub-verseNNN/sub-verseNNN_ct.nii.gz
        derivatives/sub-verseNNN/sub-verseNNN_seg-vert_msk.nii.gz

Per-spec ``src_root`` should point at the directory containing
``rawdata/`` and ``derivatives/``. If your archive ships the three
official splits as siblings (``dataset-01training/``, ``-02validation/``,
``-03test/``), point at one of them or build a flat-tree symlink set.

VerSe's vertebra-ID convention (matches Sekuboyina 2021):

    1-7    C1, C2, ..., C7         (cervical)
    8-19   T1, T2, ..., T12        (thoracic)
    20-24  L1, L2, ..., L5         (lumbar)
    25     T13                     (rare transitional)
    26     L6                      (rare transitional)
    27     sacrum (S1)             (some VerSe releases only)
    28     cocygeus                (very rare)

Specs reference vertebrae by snake_case canonical names like
``vertebra_C1`` (a ``_verse`` helper in ``specs.py`` is the entry
point).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .base import CaseRef, DatasetSource


log = logging.getLogger("verse_source")


def _cervical_ids() -> dict[str, int]:
    return {f"vertebra_C{i}": i for i in range(1, 8)}


def _thoracic_ids() -> dict[str, int]:
    return {f"vertebra_T{i}": 7 + i for i in range(1, 13)}


def _lumbar_ids() -> dict[str, int]:
    return {f"vertebra_L{i}": 19 + i for i in range(1, 6)}


VERSE_NAME_TO_ID: dict[str, int] = {
    **_cervical_ids(),
    **_thoracic_ids(),
    **_lumbar_ids(),
    "vertebra_T13": 25,
    "vertebra_L6":  26,
    "vertebra_S1":  27,
}
VERSE_ID_TO_NAME: dict[int, str] = {v: k for k, v in VERSE_NAME_TO_ID.items()}


_SUBJECT_RE = re.compile(r"^sub-(?P<subj>verse\d+[A-Za-z]?)$")


class VerSeSource(DatasetSource):
    """Source for the VerSe 2020 vertebral segmentation cohort."""

    source_type = "verse"

    def __init__(self, src_root: Path):
        self.src_root = Path(src_root)
        self.rawdata_dir = self.src_root / "rawdata"
        self.derivatives_dir = self.src_root / "derivatives"
        if not self.rawdata_dir.is_dir() or not self.derivatives_dir.is_dir():
            raise FileNotFoundError(
                f"VerSe layout not found under {self.src_root}; "
                f"expected rawdata/ and derivatives/ subdirs"
            )

    def discover(self, structures: Sequence[str]) -> list[CaseRef]:
        unknown = [s for s in structures if s not in VERSE_NAME_TO_ID]
        if unknown:
            raise ValueError(
                f"VerSe does not provide structures {unknown}; "
                f"available: {sorted(VERSE_NAME_TO_ID)}"
            )

        cases: list[CaseRef] = []
        for subj_dir in sorted(self.rawdata_dir.iterdir()):
            m = _SUBJECT_RE.match(subj_dir.name)
            if m is None or not subj_dir.is_dir():
                continue
            subj = m.group("subj")
            ct = subj_dir / f"sub-{subj}_ct.nii.gz"
            seg = self.derivatives_dir / f"sub-{subj}" / f"sub-{subj}_seg-vert_msk.nii.gz"
            if not ct.is_file() or not seg.is_file():
                log.debug("[%s] missing files: ct=%s seg=%s", subj, ct.is_file(), seg.is_file())
                continue
            label_paths = {s: seg for s in structures}
            cases.append(
                CaseRef(
                    case_id=subj,
                    patient_id=subj,
                    image_path=ct,
                    label_paths=label_paths,
                    metadata={},
                )
            )
        log.info("VerSe discover(%s): %d cases", list(structures), len(cases))
        return cases

    def load_image(self, case: CaseRef) -> sitk.Image:
        return sitk.ReadImage(str(case.image_path))

    def load_mask(
        self,
        case: CaseRef,
        canonical_name: str,
        ref_image: sitk.Image,
    ) -> sitk.Image:
        lid = VERSE_NAME_TO_ID.get(canonical_name)
        if lid is None:
            raise KeyError(f"unknown VerSe structure {canonical_name!r}")
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


__all__ = ["VerSeSource", "VERSE_NAME_TO_ID", "VERSE_ID_TO_NAME"]

"""LUNA16 dataset source (lung CT + nodule annotations).

The LUNA16 release (Zenodo 3723295 + 4121926, CC-BY-3.0 inherited from
LIDC-IDRI) ships 888 CT cases as MHD+RAW pairs across 10 subset
directories, plus per-nodule annotations in CSV:

    <src_root>/
        subset0/  <seriesuid>.mhd      <seriesuid>.raw
        subset1/  ...
        ...
        subset9/  ...
        annotations.csv                 # confirmed positives
        candidates.csv                  # positives + negatives (~750k rows)
        candidates_V2.csv               # smaller curated candidate set
        seg-lungs-LUNA16/<seriesuid>.mhd  # lung segmentation masks

`annotations.csv` rows are
    seriesuid, coordX, coordY, coordZ, diameter_mm
in WORLD coordinates (same frame as the image). Multiple rows per case
when there are multiple nodules.

This adapter exposes one canonical structure: "Nodule". For each case,
load_mask synthesises a spherical mask at every annotation centroid by
converting world coordinates to voxel indices via the image's origin
and spacing, then drawing a sphere of radius diameter/2 mm. Overlapping
nodule spheres in the same case are merged (logical OR) — they're the
same class.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .base import CaseRef, DatasetSource


log = logging.getLogger("luna16_source")


class LunaSource(DatasetSource):
    """Source for LUNA16 MHD/RAW CT + annotation CSV."""

    source_type = "luna16"

    def __init__(
        self,
        src_root: Path,
        annotations_csv: Path | None = None,
        subset_glob: str = "subset*",
    ):
        self.src_root = Path(src_root)
        if annotations_csv is None:
            annotations_csv = self.src_root / "annotations.csv"
        self.annotations_csv = Path(annotations_csv)
        self.subset_glob = subset_glob
        self._annotations = self._load_annotations()

    def _load_annotations(self) -> dict[str, list[tuple[float, float, float, float]]]:
        """Read annotations.csv → dict[seriesuid → list[(x, y, z, diameter_mm)]]."""
        out: dict[str, list[tuple[float, float, float, float]]] = {}
        if not self.annotations_csv.is_file():
            log.error("annotations.csv not found at %s", self.annotations_csv)
            return out
        with self.annotations_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row["seriesuid"]
                try:
                    nodule = (
                        float(row["coordX"]),
                        float(row["coordY"]),
                        float(row["coordZ"]),
                        float(row["diameter_mm"]),
                    )
                except (KeyError, ValueError):
                    continue
                out.setdefault(sid, []).append(nodule)
        log.info(
            "loaded %d nodules across %d cases from %s",
            sum(len(v) for v in out.values()), len(out), self.annotations_csv.name,
        )
        return out

    def discover(self, structures: Sequence[str]) -> list[CaseRef]:
        required = list(structures)
        if required != ["Nodule"]:
            log.warning(
                "LunaSource only exposes 'Nodule'; got %r — will produce "
                "empty masks for the others", required,
            )

        cases: list[CaseRef] = []
        for subset_dir in sorted(self.src_root.glob(self.subset_glob)):
            if not subset_dir.is_dir():
                continue
            for mhd in sorted(subset_dir.glob("*.mhd")):
                sid = mhd.stem  # the series UID is the filename without .mhd
                nodules = self._annotations.get(sid, [])
                if not nodules:
                    # CSV-negative case (no nodule annotation) — keep it
                    # anyway so nnUNet sees background-only examples.
                    pass
                cases.append(
                    CaseRef(
                        case_id=sid,
                        patient_id=sid,         # 1 series = 1 patient in LUNA16
                        image_path=mhd,
                        label_paths={s: mhd for s in required},  # mask is synthesised
                        metadata={
                            "subset": subset_dir.name,
                            "n_nodules": len(nodules),
                        },
                    )
                )
        log.info("LUNA16 discover: %d cases (across %d subsets)",
                 len(cases), len(list(self.src_root.glob(self.subset_glob))))
        return cases

    def load_image(self, case: CaseRef) -> sitk.Image:
        return sitk.ReadImage(str(case.image_path))

    def load_mask(
        self,
        case: CaseRef,
        canonical_name: str,
        ref_image: sitk.Image,
    ) -> sitk.Image:
        """Synthesise a binary spherical-nodule mask for this case.

        For non-"Nodule" requests we return an all-zero mask of the same
        shape so the orchestrator's empty_label filter handles it.
        """
        size = ref_image.GetSize()             # (X, Y, Z)
        spacing = ref_image.GetSpacing()       # (sx, sy, sz) in mm
        origin = ref_image.GetOrigin()         # (ox, oy, oz) in mm world
        # Numpy convention is (Z, Y, X).
        arr = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)

        if canonical_name == "Nodule":
            nodules = self._annotations.get(case.case_id, [])
            for (wx, wy, wz, dia_mm) in nodules:
                # World → voxel index (assumes axis-aligned LUNA16 images,
                # which is true for the released MHD files).
                cx = int(round((wx - origin[0]) / spacing[0]))
                cy = int(round((wy - origin[1]) / spacing[1]))
                cz = int(round((wz - origin[2]) / spacing[2]))
                rmm = dia_mm / 2.0
                rx = max(1, int(np.ceil(rmm / max(spacing[0], 1e-6))))
                ry = max(1, int(np.ceil(rmm / max(spacing[1], 1e-6))))
                rz = max(1, int(np.ceil(rmm / max(spacing[2], 1e-6))))
                # Index window
                x0, x1 = max(0, cx - rx), min(size[0], cx + rx + 1)
                y0, y1 = max(0, cy - ry), min(size[1], cy + ry + 1)
                z0, z1 = max(0, cz - rz), min(size[2], cz + rz + 1)
                if x0 >= x1 or y0 >= y1 or z0 >= z1:
                    continue
                zs, ys, xs = np.meshgrid(
                    np.arange(z0, z1), np.arange(y0, y1), np.arange(x0, x1),
                    indexing="ij",
                )
                # Ellipsoid in voxel-space (one axis per spatial dim).
                inside = (
                    ((xs - cx) * spacing[0]) ** 2
                    + ((ys - cy) * spacing[1]) ** 2
                    + ((zs - cz) * spacing[2]) ** 2
                ) <= rmm * rmm
                arr[z0:z1, y0:y1, x0:x1][inside] = 1

        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(ref_image)
        return out


__all__ = ["LunaSource"]

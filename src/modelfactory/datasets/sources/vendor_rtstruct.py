"""Per-case DICOM CT + RTSTRUCT source for the an external auto-segmentation tool drop.

Layout (as found under ``<vendor-export-root>/dataset/``)::

    <root>/<case_id>/
        ct/0001.dcm …          ← DICOM CT series, one slice per file
        AC-<uid>.dcm           ← single RT Structure Set at the case root
        meta.json              ← {series_ser, rtstruct, n_roi}  (NO structure_names[])

Distinct from ``ClinicalRTStructSource`` (which it subclasses) in two ways:

  * the RTSTRUCT is a loose ``AC-*.dcm`` at the case root, not under a
    ``rtstruct/`` subdirectory;
  * ``meta.json`` carries no structure list, so we read the ROI names straight
    from the RTSTRUCT's ``StructureSetROISequence`` (no pixel data) at discover
    time.

The ROI names in this drop are already TG-263 (``Heart``, ``A_LAD``,
``Valve_Aortic`` …), and the spec passes those exact names through
``StructureMapping.name_in("vendor_rtstruct")`` (canonical == on-disk,
or a per-source alias for names with illegal characters such as
``Pericardium_Inf+A_Pulm``). So both ``discover()`` and ``load_mask()`` work by
identity — no alias dictionary is consulted. ``load_image`` is inherited
unchanged (it reads the ``ct/`` DICOM series).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pydicom
import SimpleITK as sitk

from .base import CaseRef
from .clinical_rtstruct import ClinicalRTStructSource


log = logging.getLogger("vendor_rtstruct_source")


class VendorRTStructSource(ClinicalRTStructSource):
    """Per-case ``ct/`` + ``AC-*.dcm`` RTSTRUCT cohort, matched by ROI name.

    Config (from ``DatasetSpec.source_constraints["vendor_rtstruct"]``)::

        root:           directory containing <case_id>/{ct/, AC-*.dcm, meta.json}
        # optional, accepted for symmetry with the parent (unused here):
        aliases, partial_label
    """

    source_type = "vendor_rtstruct"

    @staticmethod
    def _find_rtstruct(case_dir):
        """Return the RTSTRUCT path for a case dir, or None.

        Prefer the RTSTRUCT ``AC-*.dcm`` at the case root; fall back to any
        root-level ``*.dcm`` (the CT slices live under ``ct/``, so they never
        collide with this glob).
        """
        ac = sorted(case_dir.glob("AC-*.dcm"))
        if ac:
            return ac[0]
        loose = sorted(p for p in case_dir.glob("*.dcm") if p.is_file())
        return loose[0] if loose else None

    @staticmethod
    def _roi_names(rt_file) -> set[str]:
        """Read ROI names from an RTSTRUCT's StructureSetROISequence (no pixels)."""
        ds = pydicom.dcmread(str(rt_file), force=True)
        names: set[str] = set()
        for roi in getattr(ds, "StructureSetROISequence", []) or []:
            name = getattr(roi, "ROIName", None)
            if name:
                names.add(str(name))
        return names

    # ── DatasetSource interface ────────────────────────────────────────────

    def discover(self, structures: Sequence[str]) -> list[CaseRef]:
        """Return CaseRefs whose RTSTRUCT contains the requested ROI names.

        ``structures`` are the *on-disk* ROI names (post ``name_in`` resolution),
        so membership is an exact-name check against the RTSTRUCT's
        StructureSetROISequence. AND-filter by default; OR-filter when
        ``partial_label`` is set (mirrors the parent).
        """
        required = list(structures)
        cases: list[CaseRef] = []
        skipped_no_rtstruct = 0
        skipped_no_ct = 0
        skipped_missing: dict[str, int] = {s: 0 for s in required}

        if not self.root.is_dir():
            log.error("cohort root missing: %s", self.root)
            return cases

        for case_dir in sorted(self.root.iterdir()):
            if not case_dir.is_dir():
                continue
            ct_dir = case_dir / "ct"
            if not ct_dir.is_dir():
                skipped_no_ct += 1
                continue
            rt_file = self._find_rtstruct(case_dir)
            if rt_file is None:
                skipped_no_rtstruct += 1
                continue

            try:
                present = self._roi_names(rt_file)
            except Exception as e:
                log.warning("[%s] cannot read RTSTRUCT %s: %s", case_dir.name, rt_file.name, e)
                continue

            present_required = [s for s in required if s in present]
            if self.partial_label:
                if not present_required:
                    for m in required:
                        skipped_missing[m] += 1
                    continue
            else:
                missing = [s for s in required if s not in present]
                if missing:
                    for m in missing:
                        skipped_missing[m] += 1
                    continue

            cid = case_dir.name
            cases.append(
                CaseRef(
                    case_id=cid,
                    patient_id=cid,
                    image_path=ct_dir,
                    label_paths={"_rtstruct": rt_file},
                    metadata={
                        "license": "internal-clinical-RTM-2026",
                        "label_source": "external-rtstruct",
                    },
                )
            )

        log.info(
            "discover(%s): %d cases ok | skipped: %d no-ct, %d no-rtstruct, missing-roi %s",
            required, len(cases), skipped_no_ct, skipped_no_rtstruct, skipped_missing,
        )
        return cases

    def load_mask(
        self,
        case: CaseRef,
        canonical_name: str,
        ref_image: sitk.Image,
    ) -> sitk.Image:
        """Rasterize one ROI by its exact on-disk name via rt_utils.

        ``canonical_name`` here is the on-disk RTSTRUCT ROI name (the spec's
        ``name_in`` already mapped to it), so we look it up verbatim rather than
        walking the alias index the parent uses. Geometry handling (axis order +
        nearest-neighbour resample to the reference grid) matches the parent.
        """
        from rt_utils import RTStructBuilder

        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=str(case.image_path),
            rt_struct_path=str(case.label_paths["_rtstruct"]),
        )

        try:
            m = rtstruct.get_roi_mask_by_name(canonical_name)
            mask_arr = np.asarray(m, dtype=bool)
        except Exception as e:
            log.warning(
                "[%s] ROI %r not rasterizable in RTSTRUCT: %s",
                case.case_id, canonical_name, e,
            )
            empty = np.zeros(ref_image.GetSize()[::-1], dtype=np.uint8)
            img = sitk.GetImageFromArray(empty)
            img.CopyInformation(ref_image)
            return img

        # rt-utils returns (rows, cols, slices) = (Y, X, Z); SimpleITK's numpy
        # buffer is (Z, Y, X).
        arr = mask_arr.transpose(2, 0, 1).astype(np.uint8)

        ref_size_zyx = ref_image.GetSize()[::-1]
        if arr.shape != ref_size_zyx:
            log.warning(
                "[%s] mask shape %s != ref shape %s for %r — resampling",
                case.case_id, arr.shape, ref_size_zyx, canonical_name,
            )
            tmp = sitk.GetImageFromArray(arr)
            tmp.SetSpacing(ref_image.GetSpacing())
            tmp.SetOrigin(ref_image.GetOrigin())
            tmp.SetDirection(ref_image.GetDirection())
            return sitk.Resample(
                tmp, ref_image, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
            )

        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(ref_image)
        return out


__all__ = ["VendorRTStructSource"]

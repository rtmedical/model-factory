"""Per-case DICOM CT + RTSTRUCT source for the internal clinical archive.

Layout (as found under ``/data/<cohort>/`` and, when its upload completes,
``/data/<cohort>/``)::

    <root>/<pseudo_id>/
        ct/<CT_NNNN.dcm>            ← DICOM CT series, one slice per file
        rtstruct/RTSTRUCT.dcm       ← single RT Structure Set
        metadata.json               ← {pseudo_id, region, ct_series_uid,
                                       rtstruct_series_uid, structure_names[]}

The pseudo_id IS the patient stratification key (one case per patient),
so unlike ``rtstruct.py`` we do not need a SeriesUID→PatientID CSV.

Distinct from ``rtstruct.RTStructDicomSource`` because:
  * pairing is by directory name, not by SeriesUID lookup against a TCIA
    tree;
  * structure names in the RTSTRUCT are a mix of TG-263 and Portuguese
    clinical names (sometimes both in the same case), so we apply an alias
    dict (``modelfactory.datasets.aliases``) during discover() AND when
    looking up ROIs in load_mask();
  * planning-derivative names ending in _0.5, _PRV05, _3MM, _OTM, "z*otm",
    "_prv*" are intentionally NOT aliased — they're not the OAR itself.

Cohort survey (run ``scripts/inspect_clinical_cohort.py``) is the
recommended pre-flight before submitting conversions.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .base import CaseRef, DatasetSource
from ..aliases import CLINICAL_RTSTRUCT_ALIASES, reverse_alias_index


log = logging.getLogger("clinical_rtstruct_source")


class ClinicalRTStructSource(DatasetSource):
    """Per-case DICOM CT + RTSTRUCT cohort with alias-aware ROI lookup.

    Config (from ``DatasetSpec.source_constraints["clinical_rtstruct"]``)::

        root:           directory containing <pseudo_id>/{ct,rtstruct,metadata.json}/
        # optional:
        aliases:        dict[on-disk → canonical] override; defaults to
                        CLINICAL_RTSTRUCT_ALIASES from modelfactory.datasets.aliases
    """

    source_type = "clinical_rtstruct"

    def __init__(
        self,
        root: Path,
        aliases: dict[str, str] | None = None,
        partial_label: bool = False,
    ):
        self.root = Path(root)
        self.aliases: dict[str, str] = dict(
            aliases if aliases is not None else CLINICAL_RTSTRUCT_ALIASES
        )
        self.reverse_aliases: dict[str, list[str]] = reverse_alias_index(self.aliases)
        # populated during discover(): {case_id: {canonical: on_disk_variant}}
        self._case_variants: dict[str, dict[str, str]] = {}
        # Partial-label generalist: keep a case if it contours AT LEAST ONE
        # requested structure (the AND-filter is relaxed to OR). The convert
        # worker then paints only the present structures and records them; the
        # partial-label trainer masks each sample's loss to its annotated
        # channels. See specs.DatasetSpec.partial_label.
        self.partial_label = partial_label

    # ── DatasetSource interface ────────────────────────────────────────────

    def discover(self, structures: Sequence[str]) -> list[CaseRef]:
        """Return CaseRefs that contain ALL requested canonical structures.

        Caller passes canonical names (post-spec resolution); we walk each
        per-case metadata.json, apply the alias dict to map the on-disk
        structure_names into canonical form, and check membership.
        """
        required = list(structures)
        cases: list[CaseRef] = []
        skipped_no_meta = 0
        skipped_no_rtstruct = 0
        skipped_missing: dict[str, int] = {s: 0 for s in required}

        if not self.root.is_dir():
            log.error("cohort root missing: %s", self.root)
            return cases

        for case_dir in sorted(self.root.iterdir()):
            if not case_dir.is_dir():
                continue
            meta_path = case_dir / "metadata.json"
            ct_dir = case_dir / "ct"
            rtstruct_dir = case_dir / "rtstruct"

            if not meta_path.is_file() or not ct_dir.is_dir():
                skipped_no_meta += 1
                continue

            rt_files = sorted(rtstruct_dir.glob("*.dcm")) if rtstruct_dir.is_dir() else []
            if not rt_files:
                skipped_no_rtstruct += 1
                continue
            rt_file = rt_files[0]

            try:
                meta = json.loads(meta_path.read_text())
            except Exception as e:
                log.warning("[%s] cannot read metadata.json: %s", case_dir.name, e)
                continue

            on_disk_names: list[str] = list(meta.get("structure_names") or [])
            # Build {canonical: on_disk_name} for the canonical structures present
            present: dict[str, str] = {}
            for name in on_disk_names:
                canonical = self.aliases.get(name)
                if canonical and canonical not in present:
                    present[canonical] = name

            present_required = [s for s in required if s in present]
            if self.partial_label:
                # OR-filter: keep the case if it has any requested structure.
                if not present_required:
                    for m in required:
                        skipped_missing[m] = skipped_missing.get(m, 0) + 1
                    continue
            else:
                # AND-filter: every requested structure must be present.
                missing = [s for s in required if s not in present]
                if missing:
                    for m in missing:
                        skipped_missing[m] = skipped_missing.get(m, 0) + 1
                    continue

            pseudo_id = case_dir.name
            self._case_variants[pseudo_id] = present
            cases.append(
                CaseRef(
                    case_id=pseudo_id,
                    patient_id=pseudo_id,
                    image_path=ct_dir,
                    label_paths={"_rtstruct": rt_file},
                    metadata={
                        "region": meta.get("region"),
                        "icd_code": meta.get("icd_code"),
                        "site": meta.get("site"),
                        "ct_series_uid": meta.get("ct_series_uid"),
                        "rtstruct_series_uid": meta.get("rtstruct_series_uid"),
                        "ct_creation_date": meta.get("ct_creation_date"),
                        "license": "internal-clinical-RTM-2026",
                    },
                )
            )

        log.info(
            "discover(%s): %d cases ok | skipped: %d no-meta, %d no-rtstruct, missing-struct %s",
            required, len(cases), skipped_no_meta, skipped_no_rtstruct, skipped_missing,
        )
        return cases

    def load_image(self, case: CaseRef) -> sitk.Image:
        reader = sitk.ImageSeriesReader()
        file_names = reader.GetGDCMSeriesFileNames(str(case.image_path))
        if not file_names:
            raise RuntimeError(f"no DICOM files under {case.image_path}")
        reader.SetFileNames(file_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        return reader.Execute()

    def load_mask(
        self,
        case: CaseRef,
        canonical_name: str,
        ref_image: sitk.Image,
    ) -> sitk.Image:
        """Build a binary mask for the named canonical structure.

        Resolves canonical → on-disk variant via:
          1) the per-case variant table populated in discover() (preferred —
             matches the exact name that case's RTSTRUCT actually has);
          2) failing that, every reverse-alias variant for this canonical
             in turn (covers the worker-pool case where discover()'s side
             effect isn't visible because we're in a fresh process).
        """
        from rt_utils import RTStructBuilder

        candidates: list[str] = []
        on_disk = self._case_variants.get(case.case_id, {}).get(canonical_name)
        if on_disk:
            candidates.append(on_disk)
        # Always include the full reverse-alias list as fallback (the worker
        # process won't have populated _case_variants; that side-effect lives
        # in the discover-time source instance only).
        for variant in self.reverse_aliases.get(canonical_name, []):
            if variant not in candidates:
                candidates.append(variant)

        if not candidates:
            raise KeyError(f"no on-disk variant known for canonical {canonical_name!r}")

        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=str(case.image_path),
            rt_struct_path=str(case.label_paths["_rtstruct"]),
        )

        last_err: Exception | None = None
        mask_arr: np.ndarray | None = None
        for variant in candidates:
            try:
                m = rtstruct.get_roi_mask_by_name(variant)
                mask_arr = np.asarray(m, dtype=bool)
                break
            except Exception as e:
                last_err = e
                continue

        if mask_arr is None:
            log.warning(
                "[%s] no candidate variant for %r matched in RTSTRUCT (tried %s); last err: %s",
                case.case_id, canonical_name, candidates, last_err,
            )
            empty = np.zeros(ref_image.GetSize()[::-1], dtype=np.uint8)
            img = sitk.GetImageFromArray(empty)
            img.CopyInformation(ref_image)
            return img

        # rt-utils returns (rows, cols, slices) = (Y, X, Z). SimpleITK
        # numpy buffer is (Z, Y, X). Match it.
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
            tmp = sitk.Resample(
                tmp, ref_image, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
            )
            return tmp

        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(ref_image)
        return out


__all__ = ["ClinicalRTStructSource"]

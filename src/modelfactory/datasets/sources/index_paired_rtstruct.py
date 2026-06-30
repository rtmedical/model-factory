"""Index-paired DICOM series + RTSTRUCT source.

For cohorts where each case is a numbered directory and the image series and
its RTSTRUCT live in sibling trees, paired by directory index (NOT by
SeriesInstanceUID and NOT carrying a metadata.json)::

    <root>/<image_subdir>/0001/<NNNN.dcm>      ← DICOM image series (one slice/file)
    <root>/<rtstruct_subdir>/0001/AC-*.dcm     ← one RT Structure Set per case
    <root>/lineage.json                         ← optional; per-index case/patient id

This is the layout produced by running an external auto-segmentation tool over an
existing nnUNet dataset's DICOM (e.g. Dataset063's ``dicomTr/`` + ``rtstructTr/``):
the RTSTRUCT RTSTRUCTs carry complementary OARs (eyes, lens, optic
nerve/chiasm) not in the original label set.

Distinct from:
  * ``rtstruct.RTStructDicomSource`` — pairs by SeriesUID against a TCIA tree + series.csv;
  * ``clinical_rtstruct.ClinicalRTStructSource`` — pairs by dir name but REQUIRES a
    per-case ``metadata.json`` listing ``structure_names``.

Here there is no metadata.json, so we read the ROI names straight from each
RTSTRUCT's StructureSetROISequence at discover time. ROI names are mapped to
factory-canonical names via the shared ``CLINICAL_RTSTRUCT_ALIASES`` table
(``OpticNrv_L``→``OpticNerve_L``, ``OpticChiasm``→``Chiasm``, ``Eye_*``/``Lens_*``
already canonical), with a defensive strip of RTSTRUCT version/derivative
suffixes (``_MR7``, ``_MR06``, ``_OTM``, ``_PRV05``, ``_3MM``) before lookup.
Rasterization reuses the same RTStructBuilder path as ``clinical_rtstruct``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .base import CaseRef, DatasetSource
from ..aliases import CLINICAL_RTSTRUCT_ALIASES, canonical_for, reverse_alias_index


log = logging.getLogger("index_paired_rtstruct_source")


class IndexPairedRTStructSource(DatasetSource):
    """Index-paired DICOM series + RTSTRUCT cohort with alias-aware ROI lookup.

    Config (from ``DatasetSpec.source_constraints["index_paired_rtstruct"]``)::

        root:             directory holding <image_subdir>/ and <rtstruct_subdir>/
        image_subdir:     subdir of per-case DICOM image series (default "dicomTr")
        rtstruct_subdir:  subdir of per-case RTSTRUCT files     (default "rtstructTr")
        # optional:
        aliases:          on-disk → canonical override; defaults to CLINICAL_RTSTRUCT_ALIASES
        rtstruct_glob:    glob for the RTSTRUCT file in each case dir (default "*.dcm")
        image_glob:       when set, each case's image is a single NIfTI file
                          <image_subdir>/<case>/<image_glob> (e.g. "mri.nii.gz")
                          rather than a DICOM series directory. Used for the
                          TotalSegmentator-MRI cohort (src/s####/mri.nii.gz +
                          rtstruct_86/s####/AC-*.dcm); the NIfTI carries the same
                          geometry the RTSTRUCT polygons reference, so the
                          self-rasterizer maps them straight on.
        partial_label:    when True, relax the AND-filter — keep any case that
                          contours ≥1 requested structure (not all). Required for
                          the partial-label generalist/specialists, where per-case
                          RTSTRUCT coverage varies widely.
    """

    source_type = "index_paired_rtstruct"

    def __init__(
        self,
        root: Path,
        image_subdir: str = "dicomTr",
        rtstruct_subdir: str = "rtstructTr",
        aliases: dict[str, str] | None = None,
        rtstruct_glob: str = "*.dcm",
        image_glob: str | None = None,
        partial_label: bool = False,
    ):
        self.root = Path(root)
        self.image_root = self.root / image_subdir
        self.rtstruct_root = self.root / rtstruct_subdir
        self.rtstruct_glob = rtstruct_glob
        self.image_glob = image_glob
        self.partial_label = partial_label
        self.aliases: dict[str, str] = dict(
            aliases if aliases is not None else CLINICAL_RTSTRUCT_ALIASES
        )
        self.reverse_aliases: dict[str, list[str]] = reverse_alias_index(self.aliases)
        # {dir_name: (case_id, patient_id)} from lineage.json, if present.
        self._idmap: dict[str, tuple[str, str]] = self._load_lineage()
        # Cache the last parsed RTSTRUCT: load_mask is called once per structure
        # for the same case, so parse the file once per case, not 7×.
        self._rt_cache: tuple[str, dict[str, list]] | None = None

    # ── helpers ────────────────────────────────────────────────────────────

    def _load_lineage(self) -> dict[str, tuple[str, str]]:
        """Map each numeric case dir → (case_id, patient_id) from lineage.json.

        lineage.json is a list aligned to the imagesTr order; entry i maps to
        case dir f"{i+1:04d}". Falls back to the dir name for both ids when
        lineage is absent/unreadable so the source still works standalone.
        """
        path = self.root / "lineage.json"
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text())
        except Exception as e:  # noqa: BLE001
            log.warning("lineage.json unreadable (%s); using dir names as ids", e)
            return {}
        out: dict[str, tuple[str, str]] = {}
        if isinstance(data, list):
            for i, entry in enumerate(data):
                if not isinstance(entry, dict):
                    continue
                dir_name = f"{i + 1:04d}"
                cid = str(entry.get("case_id") or dir_name)
                pid = str(entry.get("patient_id") or cid)
                out[dir_name] = (cid, pid)
        elif isinstance(data, dict):
            for k, entry in data.items():
                if not isinstance(entry, dict):
                    continue
                cid = str(entry.get("case_id") or k)
                pid = str(entry.get("patient_id") or cid)
                out[str(k)] = (cid, pid)
        return out

    def _canonical_for(self, roi_name: str) -> str | None:
        """Resolve an on-disk ROI name to a factory-canonical name (or None)."""
        return canonical_for(roi_name, self.aliases)

    @staticmethod
    def _read_roi_names(rt_file: Path) -> list[str]:
        import pydicom  # lazy: only needed at discover time, in the trainer image

        ds = pydicom.dcmread(str(rt_file), stop_before_pixels=True, force=True)
        return [str(r.ROIName) for r in getattr(ds, "StructureSetROISequence", [])]

    # ── DatasetSource interface ────────────────────────────────────────────

    def discover(self, structures: Sequence[str]) -> list[CaseRef]:
        """Return CaseRefs that contain ALL requested canonical structures.

        Pairs <image_subdir>/NNNN ↔ <rtstruct_subdir>/NNNN by directory name,
        reads each RTSTRUCT's ROI names, maps them to canonical via the alias
        table (+ suffix strip), and keeps a case only if every requested
        canonical is present. The per-case canonical→on-disk variant map is
        carried in CaseRef.metadata["variants"] so the (separate-process)
        load_mask worker can resolve the exact ROI name.
        """
        required = list(structures)
        cases: list[CaseRef] = []
        skipped_no_rtstruct = 0
        skipped_no_roi = 0
        skipped_no_image = 0
        skipped_no_overlap = 0
        skipped_missing: dict[str, int] = {s: 0 for s in required}
        seen_case_ids: set[str] = set()

        if not self.image_root.is_dir() or not self.rtstruct_root.is_dir():
            log.error(
                "missing image_root %s or rtstruct_root %s",
                self.image_root, self.rtstruct_root,
            )
            return cases

        for img_dir in sorted(self.image_root.iterdir()):
            if not img_dir.is_dir():
                continue
            dir_name = img_dir.name

            # Image path: a single NIfTI per case (image_glob) or a DICOM series
            # directory. For NIfTI mode the CaseRef carries the file path so
            # load_image reads it directly.
            if self.image_glob:
                matches = sorted(img_dir.glob(self.image_glob))
                if not matches:
                    skipped_no_image += 1
                    continue
                image_path = matches[0]
            else:
                image_path = img_dir

            rt_dir = self.rtstruct_root / dir_name
            rt_files = sorted(rt_dir.glob(self.rtstruct_glob)) if rt_dir.is_dir() else []
            if not rt_files:
                skipped_no_rtstruct += 1
                continue
            rt_file = rt_files[0]

            try:
                roi_names = self._read_roi_names(rt_file)
            except Exception as e:  # noqa: BLE001
                log.warning("[%s] cannot read RTSTRUCT ROIs: %s", dir_name, e)
                skipped_no_roi += 1
                continue

            # {canonical: [on_disk_name, ...]} — ALL on-disk variants that map
            # to this canonical (e.g. Urethra_MR + HDR_Urethra_MR), in file
            # order. load_mask picks the first NON-EMPTY one, so a template
            # duplicate that is declared-but-empty on a case (common with the
            # brachy HDR_/sequence TRUFI_ copies) doesn't shadow the populated
            # variant and cost coverage.
            present: dict[str, list[str]] = {}
            for name in roi_names:
                canonical = self._canonical_for(name)
                if canonical:
                    bucket = present.setdefault(canonical, [])
                    if name not in bucket:
                        bucket.append(name)

            missing = [s for s in required if s not in present]
            if missing:
                for m in missing:
                    skipped_missing[m] = skipped_missing.get(m, 0) + 1
                if not self.partial_label:
                    # Strict AND-filter: every requested structure must be present.
                    continue
                # Partial-label: keep the case if it contours ≥1 requested
                # structure; the converter paints only those and records the
                # per-case annotation set so the trainer masks the rest.
                if len(missing) == len(required):
                    skipped_no_overlap += 1
                    continue

            cid, pid = self._idmap.get(dir_name, (dir_name, dir_name))
            # Guard against a non-unique lineage case_id across indices.
            if cid in seen_case_ids:
                cid = f"{cid}_{dir_name}"
            seen_case_ids.add(cid)

            cases.append(
                CaseRef(
                    case_id=cid,
                    patient_id=pid,
                    image_path=image_path,
                    label_paths={"_rtstruct": rt_file},
                    metadata={
                        "variants": present,          # canonical → on-disk ROI name
                        "src_dir": dir_name,
                        "label_quality": "silver",
                        "label_source": "external-rtstruct",
                    },
                )
            )

        log.info(
            "discover(%s): %d cases ok | mode=%s partial_label=%s | skipped: %d no-image, "
            "%d no-rtstruct, %d unreadable-roi, %d no-overlap | missing-struct %s",
            required, len(cases), "nifti" if self.image_glob else "dicom-series",
            self.partial_label, skipped_no_image, skipped_no_rtstruct, skipped_no_roi,
            skipped_no_overlap, skipped_missing,
        )
        return cases

    def load_image(self, case: CaseRef) -> sitk.Image:
        # NIfTI-per-case mode: image_path is a file, read it directly.
        if case.image_path.is_file():
            return sitk.ReadImage(str(case.image_path))
        reader = sitk.ImageSeriesReader()
        file_names = reader.GetGDCMSeriesFileNames(str(case.image_path))
        if not file_names:
            raise RuntimeError(f"no DICOM files under {case.image_path}")
        reader.SetFileNames(file_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        return reader.Execute()

    def _parse_rtstruct(self, rt_path: Path) -> dict[str, list[np.ndarray]]:
        """{roi_name → [planar contour (N,3) patient-mm arrays]}, cached per file.

        We rasterize contours ourselves rather than via rt_utils because the
        RTSTRUCT-on-HCP DICOM lack ``FrameOfReferenceUID`` on both the image
        slices and the RTSTRUCT (rt_utils hard-requires it). The contours are
        still valid patient-coordinate polygons referencing the dicomTr series,
        so they map cleanly onto the SimpleITK image geometry.
        """
        if self._rt_cache and self._rt_cache[0] == str(rt_path):
            return self._rt_cache[1]
        import pydicom

        ds = pydicom.dcmread(str(rt_path), force=True)
        num2name = {
            int(r.ROINumber): str(r.ROIName)
            for r in getattr(ds, "StructureSetROISequence", [])
        }
        out: dict[str, list[np.ndarray]] = {}
        for rc in getattr(ds, "ROIContourSequence", []):
            name = num2name.get(int(getattr(rc, "ReferencedROINumber", -1)))
            if not name:
                continue
            polys: list[np.ndarray] = []
            for c in getattr(rc, "ContourSequence", []) or []:
                if getattr(c, "ContourGeometricType", "") != "CLOSED_PLANAR":
                    continue
                pts = np.asarray(c.ContourData, dtype=np.float64).reshape(-1, 3)
                if len(pts) >= 3:
                    polys.append(pts)
            out[name] = polys
        self._rt_cache = (str(rt_path), out)
        return out

    def load_mask(
        self,
        case: CaseRef,
        canonical_name: str,
        ref_image: sitk.Image,
    ) -> sitk.Image:
        """Binary mask for a canonical structure, rasterized from RTSTRUCT contours.

        Resolves canonical → on-disk ROI name via CaseRef.metadata["variants"]
        (survives to worker processes) then the reverse-alias list, then fills
        each CLOSED_PLANAR contour onto the reference grid (XOR per slice so
        nested contours cut holes, per the DICOM even-odd convention).
        """
        from skimage.draw import polygon as sk_polygon

        candidates: list[str] = []
        variants = (case.metadata or {}).get("variants") or {}
        on_disk = variants.get(canonical_name) or []
        # `variants` now carries a list of raw on-disk names per canonical;
        # tolerate the legacy str form too.
        if isinstance(on_disk, str):
            on_disk = [on_disk]
        candidates.extend(on_disk)
        for variant in self.reverse_aliases.get(canonical_name, []):
            if variant not in candidates:
                candidates.append(variant)
        if not candidates:
            raise KeyError(f"no on-disk variant known for canonical {canonical_name!r}")

        parsed = self._parse_rtstruct(case.label_paths["_rtstruct"])
        # Prefer the first candidate that is present AND has contours; fall back
        # to a present-but-empty one (yields an empty mask → un-annotated).
        roi_name = next(
            (c for c in candidates if parsed.get(c)), None
        ) or next((c for c in candidates if c in parsed), None)

        size_x, size_y, size_z = ref_image.GetSize()
        mask = np.zeros((size_z, size_y, size_x), dtype=np.uint8)

        if roi_name is None:
            log.warning(
                "[%s] no candidate for %r in RTSTRUCT (tried %s; have %s)",
                case.case_id, canonical_name, candidates, list(parsed)[:8],
            )
            out = sitk.GetImageFromArray(mask)
            out.CopyInformation(ref_image)
            return out

        for poly in parsed[roi_name]:
            idx = np.array(
                [ref_image.TransformPhysicalPointToContinuousIndex(
                    (float(p[0]), float(p[1]), float(p[2]))) for p in poly]
            )
            z = int(round(float(np.mean(idx[:, 2]))))
            if z < 0 or z >= size_z:
                continue
            rr, cc = sk_polygon(idx[:, 1], idx[:, 0], shape=(size_y, size_x))
            mask[z, rr, cc] ^= 1  # XOR: nested contours on a slice cut holes

        out = sitk.GetImageFromArray(mask)
        out.CopyInformation(ref_image)
        return out


__all__ = ["IndexPairedRTStructSource"]

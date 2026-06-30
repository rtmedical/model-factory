"""DICOM CT + RTSTRUCT source for cancer-archive RTSTRUCT cohorts.

Pairing convention (cancer-archive layout, 2026-05):
    rtstruct_root/<CT_SeriesInstanceUID>/AC-<uid>.dcm
    tcia_root/dicom/<collection>/<patient>/<study>/<CT_SeriesInstanceUID>/*.dcm

The directory name under `rtstruct_root` is *the CT series's* UID, not the
RTSTRUCT's own UID. Verified by reading the RTSTRUCT's
ReferencedFrameOfReferenceSequence → it matches the parent directory name.

`series.csv` (sibling of the dicom/ tree) provides
SeriesInstanceUID → PatientID for split stratification. We use PatientID
to ensure all series from the same patient land in the same fold.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .base import CaseRef, DatasetSource


log = logging.getLogger("rtstruct_source")


class RTStructDicomSource(DatasetSource):
    """Source for paired CT-DICOM + RTSTRUCT-DICOM cases.

    Config (from DatasetSpec.source_constraints["rtstruct"]):
        rtstruct_root:  directory containing <SeriesUID>/AC-*.dcm subdirs
        tcia_root:      directory containing dicom/.../<SeriesUID>/*.dcm
        series_csv:     CSV with SeriesInstanceUID,PatientID,... columns
    """

    source_type = "rtstruct"

    def __init__(
        self,
        rtstruct_root: Path | str | Sequence[Path | str],
        tcia_root: Path | str | Sequence[Path | str],
        series_csv: Path | str | Sequence[Path | str],
        volume_ranges: dict[str, list[float]] | None = None,
        allow_manifest: Path | None = None,
        partial_label: bool = False,
    ):
        # Multi-tree: each of the three roots may be a single path OR a parallel
        # list of paths. Passing lists unions several cancer-archive buckets
        # (e.g. abdome/f_ct + abdome/m_ct) into ONE dataset — case ids stay
        # unique because they embed the PatientID+SeriesUID, and split
        # stratification is by PatientID across buckets. See specs Dataset161.
        self._rt_roots = self._as_path_list(rtstruct_root)
        self._tcia_roots = self._as_path_list(tcia_root)
        self._series_csvs = self._as_path_list(series_csv)
        if not (len(self._rt_roots) == len(self._tcia_roots) == len(self._series_csvs)):
            raise ValueError(
                "rtstruct_root, tcia_root, series_csv must have equal length "
                f"(got {len(self._rt_roots)}, {len(self._tcia_roots)}, "
                f"{len(self._series_csvs)})"
            )
        # First bucket backs the single-value attributes + cached properties,
        # so single-tree behaviour (and any external reference) is unchanged.
        self.rtstruct_root = self._rt_roots[0]
        self.tcia_root = self._tcia_roots[0]
        self.series_csv = self._series_csvs[0]
        # QC gates (both optional; absent ⇒ behaviour identical to pre-QC).
        # volume_ranges: {source_roi_name: [min_cc, max_cc]} — a case is dropped
        # if any required structure's estimated contour volume is out of range.
        self.volume_ranges = dict(volume_ranges or {})
        # allow_manifest: text file of allowed SeriesUIDs or PatientIDs (one per
        # line); when set, only listed cases pass discovery.
        self.allow_ids = self._load_manifest(allow_manifest)
        # partial_label: relax the AND-filter to OR — keep a case that contours
        # AT LEAST ONE declared structure (converter paints only those present;
        # nnUNetTrainerPartialLabelMLflow masks the absent channels). Mirrors
        # ClinicalRTStructSource. Default False ⇒ strict AND (unchanged).
        self.partial_label = partial_label
        self._ct_dir_index: dict[str, Path] | None = None
        self._series_meta: dict[str, dict[str, str]] | None = None

    @staticmethod
    def _load_manifest(p: Path | None) -> set[str] | None:
        if p is None:
            return None
        p = Path(p)
        if not p.is_file():
            log.warning("allow_manifest not found at %s — no allow-list applied", p)
            return None
        ids = {ln.strip() for ln in p.read_text().splitlines() if ln.strip()}
        log.info("loaded allow-manifest: %d ids from %s", len(ids), p)
        return ids

    @staticmethod
    def _as_path_list(
        value: Path | str | Sequence[Path | str],
    ) -> list[Path]:
        """Normalise a single path or a sequence of paths to list[Path]."""
        if isinstance(value, (str, Path)):
            return [Path(value)]
        return [Path(v) for v in value]

    # ── index builders (lazy, one-shot) ───────────────────────────────────

    def _build_ct_index(self, tcia_root: Path | None = None) -> dict[str, Path]:
        """Walk tcia_root/dicom/<collection>/<patient>/<study>/<series>/.

        Returns {SeriesInstanceUID: directory_with_dcm_files}.
        """
        index: dict[str, Path] = {}
        dicom_root = (tcia_root if tcia_root is not None else self.tcia_root) / "dicom"
        if not dicom_root.is_dir():
            log.error("dicom root missing: %s", dicom_root)
            return index
        # depth-4 search: collection / patient / study / series
        for collection in dicom_root.iterdir():
            if not collection.is_dir():
                continue
            for patient in collection.iterdir():
                if not patient.is_dir():
                    continue
                for study in patient.iterdir():
                    if not study.is_dir():
                        continue
                    for series in study.iterdir():
                        if series.is_dir():
                            # directory name is the SeriesInstanceUID
                            index[series.name] = series
        log.info("indexed %d CT series under %s", len(index), dicom_root)
        return index

    def _load_series_meta(self, series_csv: Path | None = None) -> dict[str, dict[str, str]]:
        meta: dict[str, dict[str, str]] = {}
        csv_path = series_csv if series_csv is not None else self.series_csv
        if not csv_path.is_file():
            log.warning("series.csv missing: %s — splits will fall back to SeriesUID", csv_path)
            return meta
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row.get("SeriesInstanceUID")
                if uid:
                    meta[uid] = row
        log.info("loaded series.csv with %d rows", len(meta))
        return meta

    @property
    def ct_dir_index(self) -> dict[str, Path]:
        if self._ct_dir_index is None:
            self._ct_dir_index = self._build_ct_index()
        return self._ct_dir_index

    @property
    def series_meta(self) -> dict[str, dict[str, str]]:
        if self._series_meta is None:
            self._series_meta = self._load_series_meta()
        return self._series_meta

    # ── DatasetSource interface ───────────────────────────────────────────

    def discover(self, structures: Sequence[str]) -> list[CaseRef]:
        # Resolve canonical → source-name once; the spec's aliases live above
        # this layer and are applied by the orchestrator before calling us.
        required = list(structures)
        cases: list[CaseRef] = []
        for rt_root, tcia_root, series_csv in zip(
            self._rt_roots, self._tcia_roots, self._series_csvs
        ):
            cases.extend(self._discover_bucket(rt_root, tcia_root, series_csv, required))
        return cases

    def _discover_bucket(
        self,
        rtstruct_root: Path,
        tcia_root: Path,
        series_csv: Path,
        required: list[str],
    ) -> list[CaseRef]:
        """Discover cases under a single (rtstruct, tcia, series_csv) tree."""
        cases: list[CaseRef] = []
        skipped_no_ct = 0
        skipped_missing = 0
        skipped_bad_rt = 0
        skipped_volume = 0
        skipped_manifest = 0

        # Per-bucket indexes (the SeriesUID namespace is bucket-local).
        ct_dir_index = self._build_ct_index(tcia_root)
        series_meta = self._load_series_meta(series_csv)

        for ct_uid_dir in sorted(rtstruct_root.iterdir()):
            if not ct_uid_dir.is_dir():
                continue
            ct_uid = ct_uid_dir.name
            rt_files = list(ct_uid_dir.glob("AC-*.dcm"))
            if not rt_files:
                continue
            rt_file = rt_files[0]

            ct_dir = ct_dir_index.get(ct_uid)
            if ct_dir is None:
                skipped_no_ct += 1
                log.debug("no CT series for %s", ct_uid)
                continue

            present = self._list_rt_roi_volumes(rt_file, volume_for=set(self.volume_ranges))
            if present is None:
                skipped_bad_rt += 1
                continue
            present_required = [s for s in required if s in present]
            if self.partial_label:
                # OR-filter: keep the case if it contours AT LEAST ONE declared
                # structure. The converter paints only the present ones and the
                # partial-label trainer masks the absent channels per case.
                if not present_required:
                    skipped_missing += 1
                    continue
            else:
                missing = [s for s in required if s not in present]
                if missing:
                    skipped_missing += 1
                    log.debug("[%s] missing %s", ct_uid, missing)
                    continue

            # Volume-range QC gate: drop the case if any PRESENT required
            # structure's estimated contour volume falls outside its [min_cc,
            # max_cc]. Catches degenerate (near-empty) and runaway (whole-pelvis)
            # blobs, including residual wrong-sex template artifacts. Only
            # present structures are gated (in partial mode some are absent).
            out_of_range = [
                s for s in present_required
                if s in self.volume_ranges
                and not (self.volume_ranges[s][0] <= present[s] <= self.volume_ranges[s][1])
            ]
            if out_of_range:
                skipped_volume += 1
                log.debug(
                    "[%s] volume-gate dropped %s (cc=%s)", ct_uid, out_of_range,
                    {s: round(present[s], 1) for s in out_of_range},
                )
                continue

            meta = series_meta.get(ct_uid, {})
            patient_id = meta.get("PatientID", ct_uid)

            # Allow-list gate (optional): accept either the series UID or the
            # patient id, so a manifest can be at either granularity.
            if self.allow_ids is not None and ct_uid not in self.allow_ids and patient_id not in self.allow_ids:
                skipped_manifest += 1
                continue

            cases.append(
                CaseRef(
                    case_id=_make_case_id(patient_id, ct_uid),
                    patient_id=patient_id,
                    image_path=ct_dir,
                    label_paths={"_rtstruct": rt_file},
                    metadata={
                        "series_uid": ct_uid,
                        "collection": meta.get("collection_id"),
                        "license": meta.get("license_short_name"),
                        "manufacturer": meta.get("Manufacturer"),
                        "sex": meta.get("PatientSex"),
                    },
                )
            )

        log.info(
            "discover(%s) [bucket %s]: %d cases ok, %d no-CT, %d missing-structure, "
            "%d bad-RT, %d volume-gate, %d not-in-manifest",
            required, rtstruct_root.name, len(cases), skipped_no_ct, skipped_missing,
            skipped_bad_rt, skipped_volume, skipped_manifest,
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
        # Imported lazily so the host can run discover() without rt-utils
        from rt_utils import RTStructBuilder

        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=str(case.image_path),
            rt_struct_path=str(case.label_paths["_rtstruct"]),
        )
        try:
            mask = rtstruct.get_roi_mask_by_name(canonical_name)
        except Exception as e:
            log.warning("[%s] rt-utils failed on %r: %s", case.case_id, canonical_name, e)
            # Return an empty mask matching ref geometry so the case can still
            # be skipped at the orchestrator level.
            empty = np.zeros(ref_image.GetSize()[::-1], dtype=np.uint8)
            img = sitk.GetImageFromArray(empty)
            img.CopyInformation(ref_image)
            return img

        # rt-utils returns shape (rows, cols, slices) = (Y, X, Z) boolean.
        # SimpleITK numpy buffer is (Z, Y, X). Transpose accordingly.
        arr = np.asarray(mask, dtype=bool).transpose(2, 0, 1).astype(np.uint8)

        # Sanity check geometry — rt-utils builds its raster against the
        # same DICOM series we passed, so size should match the CT.
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

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _list_rt_roi_volumes(
        rt_file: Path, volume_for: set[str] | None = None
    ) -> dict[str, float] | None:
        """Return {roi_name: approx_volume_cc} for ROIs with ACTUAL contour data.

        RTSTRUCT often declares ROIs in StructureSetROISequence without any
        contour points (empty ROI). Those raise AttributeError in rt-utils, so
        we keep only ROIs with a non-empty ContourSequence carrying >0 points —
        the dict KEYS are the populated set (same membership semantics as the
        old _list_rt_rois).

        Volume is estimated from the planar contour geometry (shoelace polygon
        area per slice × median inter-slice spacing); no rasterization. It is a
        sanity-gate estimate, NOT QA-grade volumetry.

        `volume_for`: if given, compute the (expensive) shoelace volume ONLY for
        these ROI names; all other populated ROIs register presence with a 0.0
        placeholder. This keeps discovery fast on the full-body CT template
        (~260 ROIs/case) where only a handful of structures are gated. Pass
        None (the audit tool) to compute volumes for every ROI.

        Returns None on read failure.
        """
        try:
            import pydicom
            ds = pydicom.dcmread(str(rt_file), stop_before_pixels=True)
            number_to_name = {
                s.ROINumber: s.ROIName for s in ds.StructureSetROISequence
            }
            out: dict[str, float] = {}
            for rc in ds.ROIContourSequence:
                seq = getattr(rc, "ContourSequence", None)
                if not seq:
                    continue
                name = number_to_name.get(rc.ReferencedROINumber)
                if not name:
                    continue
                # cheap presence test: at least one contour carries points
                if not any(int(getattr(c, "NumberOfContourPoints", 0)) > 0 for c in seq):
                    continue
                if volume_for is not None and name not in volume_for:
                    out.setdefault(name, 0.0)  # presence only; skip shoelace
                    continue
                area_mm2 = 0.0
                zs: set[float] = set()
                for c in seq:
                    npts = int(getattr(c, "NumberOfContourPoints", 0))
                    if npts < 3:
                        continue
                    data = c.ContourData  # flat [x,y,z, x,y,z, ...]
                    xs = data[0::3]
                    ys = data[1::3]
                    zs.add(round(float(data[2]), 1))
                    # shoelace area of the planar polygon (mm^2)
                    acc = 0.0
                    for i in range(npts):
                        j = (i + 1) % npts
                        acc += xs[i] * ys[j] - xs[j] * ys[i]
                    area_mm2 += abs(acc) * 0.5
                thick = _median_spacing(sorted(zs))
                out[name] = out.get(name, 0.0) + area_mm2 * thick / 1000.0
            return out
        except Exception as e:
            log.warning("failed to read %s: %s", rt_file, e)
            return None


def _median_spacing(zs: list[float], default: float = 3.0) -> float:
    """Median gap between distinct sorted contour z-planes (slice thickness).

    Falls back to `default` mm when fewer than two planes are present (single-
    slice ROI). Only used for the discovery volume-sanity estimate.
    """
    if len(zs) < 2:
        return default
    gaps = sorted(zs[i + 1] - zs[i] for i in range(len(zs) - 1))
    mid = gaps[len(gaps) // 2]
    return mid if mid > 0 else default


def _make_case_id(patient_id: str, series_uid: str) -> str:
    """Build a filesystem-safe case id. nnUNet uses these for filenames.

    Format: <PatientID>_<short_series_hash>. Short hash keeps filenames
    bounded while still letting us trace back to the source series.
    """
    short = series_uid.split(".")[-1][:8] if "." in series_uid else series_uid[:8]
    safe_pid = patient_id.replace("/", "_").replace(" ", "_")
    return f"{safe_pid}_{short}"


__all__ = ["RTStructDicomSource"]

"""Ingest a reviewer-uploaded DICOM series or NIfTI volume as a QA case.

The QA viewer lets a reviewer bring their own test data: a DICOM series
(loose ``.dcm`` files or a ``.zip``/``.tar`` of them) or a NIfTI volume
(``.nii`` / ``.nii.gz``, one file per model input channel). We convert it to
the cohort's on-disk layout — ``image_000X.nii.gz`` per channel under
``<cohort_root>/<region>/upload_<sha>/`` — so the *entire* existing pipeline
(``/api/predict`` → preprocess cold-path → seg → meshes → cache → verdicts →
GT-edit) works on it unchanged.

No new dependencies: DICOM→NIfTI uses SimpleITK's GDCM series reader and NIfTI
validation uses nibabel, both already in the qa-viewer image. Conversion is
server-side, so the JS bundle is untouched.

This module only *materializes* the case dir + returns its `CaseRecord`; the
caller (the API endpoint, under a lock) merges it into ``manifest.json`` so
manifest writes stay serialized with donations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

from modelfactory.qa.cohort import CaseRecord

logger = logging.getLogger(__name__)

UPLOADED_SOURCE = "uploaded"

_NIFTI_SUFFIXES = (".nii", ".nii.gz")
_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tgz", ".tar.gz", ".tar.bz2")


class UploadError(ValueError):
    """Raised for a malformed / unconvertible / channel-mismatched upload.

    Subclass of ValueError so the API layer can map it to a 4xx with the
    message surfaced to the reviewer.
    """


def ingest_upload(
    staged_dir: Path,
    *,
    model_id: str,
    region: str,
    expected_channels: int,
    cohort_root: Path,
    uploaded_by: str = "",
    original_filename: str = "",
) -> CaseRecord:
    """Convert the files staged in `staged_dir` into a cohort case.

    `staged_dir` holds the raw uploaded files (filenames preserved). Detects
    DICOM vs NIfTI, converts every input channel to ``image_000X.nii.gz``,
    validates the channel count against the model, and materializes a stable
    ``upload_<sha>`` case under ``cohort_root/<region>/``.

    Idempotent: re-uploading identical bytes maps to the same case_id; the
    existing dir is reused and `model_id` is unioned into its
    `compatible_models`.

    Returns the `CaseRecord` (does NOT write the manifest — the caller merges
    under its lock). Raises `UploadError` on any validation failure.
    """
    # Content-address by the RAW uploaded bytes (not the converted output):
    # nibabel/SimpleITK writers can embed a timestamp in the gzip/NIfTI
    # header, so hashing the output would mint a new case_id on every
    # re-upload and defeat idempotency.
    sha = _hash_dir(staged_dir)[:12]

    with tempfile.TemporaryDirectory(prefix="qa-upload-") as tmp:
        work = Path(tmp)
        channels = _convert_to_channels(staged_dir, work)
        if not channels:
            raise UploadError(
                "could not read an image from the upload — expected a NIfTI "
                "volume (.nii/.nii.gz) or a DICOM series (.dcm files or a "
                ".zip/.tar of them)"
            )
        if expected_channels and len(channels) != expected_channels:
            raise UploadError(
                f"model expects {expected_channels} input channel(s) but the "
                f"upload produced {len(channels)} — upload one file per "
                f"channel (ordered), or use a single-channel model"
            )

        case_short = f"upload_{sha}"
        case_id = f"{region}/{case_short}"
        case_dir = cohort_root / region / case_short
        case_dir.mkdir(parents=True, exist_ok=True)

        image_rel: list[str] = []
        for channel_idx, produced in enumerate(channels):
            dst_name = f"image_{channel_idx:04d}.nii.gz"
            dst = case_dir / dst_name
            if not dst.exists():
                shutil.copyfile(produced, dst)
            image_rel.append(f"{region}/{case_short}/{dst_name}")

    # Provenance sidecar. On a re-upload, union the new model into
    # compatible_models so the same volume can QA more than one model.
    src_json = case_dir / "source.json"
    compatible = [model_id]
    if src_json.is_file():
        try:
            prior = json.loads(src_json.read_text())
            compatible = sorted(set(prior.get("compatible_models", [])) | {model_id})
        except (OSError, json.JSONDecodeError):
            pass
    src_json.write_text(json.dumps({
        "source_dataset": UPLOADED_SOURCE,
        "source_case_stem": original_filename or case_short,
        "kind": "upload",
        "original_filename": original_filename,
        "uploaded_by": uploaded_by,
        "compatible_models": compatible,
    }, indent=2))

    logger.info(
        "ingested upload %s (%d channel(s)) for model %s by %r",
        case_id, len(image_rel), model_id, uploaded_by,
    )
    return CaseRecord(
        case_id=case_id,
        region=region,
        source_dataset=UPLOADED_SOURCE,
        source_case_stem=original_filename or case_short,
        image_paths=image_rel,
        groundtruth_path=None,
        compatible_models=compatible,
        uploaded=True,
    )


# ── conversion ────────────────────────────────────────────────────────────


def _convert_to_channels(staged_dir: Path, work: Path) -> list[Path]:
    """Produce a list of ``.nii.gz`` channel paths from the staged upload.

    NIfTI uploads → one channel per file (sorted by name). DICOM uploads
    (archive or loose ``.dcm``) → a single channel from the largest series.
    """
    files = [p for p in staged_dir.iterdir() if p.is_file()]
    niftis = sorted(p for p in files if _is_nifti(p.name))
    archives = [p for p in files if _is_archive(p.name)]

    if niftis and not archives:
        return [_normalize_nifti(p, work / f"ch_{i:04d}.nii.gz")
                for i, p in enumerate(niftis)]

    # DICOM: extract any archives into a series dir, else read loose files
    # already in staged_dir.
    if archives:
        series_dir = work / "dicom"
        series_dir.mkdir(parents=True, exist_ok=True)
        for arc in archives:
            _extract_archive(arc, series_dir)
        return [_dicom_series_to_nifti(series_dir, work / "ch_0000.nii.gz")]

    return [_dicom_series_to_nifti(staged_dir, work / "ch_0000.nii.gz")]


def _normalize_nifti(src: Path, out: Path) -> Path:
    """Validate a NIfTI volume and re-save it as gzip ``.nii.gz``.

    Re-saving normalizes loose ``.nii`` to ``.nii.gz`` and guarantees a clean
    affine/header for the predictor's NibabelIO reader. Header-only checks
    (no full voxel load) keep this cheap.
    """
    import nibabel as nib  # lazy

    try:
        img = nib.load(str(src))
    except Exception as exc:  # noqa: BLE001 — nibabel raises a zoo of errors
        raise UploadError(f"{src.name} is not a readable NIfTI: {exc}") from exc
    if len(img.shape) < 3:
        raise UploadError(
            f"{src.name} is {len(img.shape)}-D; a 3-D volume is required"
        )
    # Drop any trailing singleton/time axes so the predictor sees a 3-D image.
    img3d = img.slicer[:, :, :] if len(img.shape) > 3 else img
    nib.save(img3d, str(out))
    return out


def _dicom_series_to_nifti(dicom_dir: Path, out: Path) -> Path:
    """Read the largest DICOM series under `dicom_dir` and write it to `out`.

    Uses SimpleITK's GDCM series reader (handles standard CT/MR series,
    sorts by ImagePositionPatient). On a multi-series archive the series with
    the most slices wins; the choice is logged.
    """
    import SimpleITK as sitk  # lazy

    reader = sitk.ImageSeriesReader()
    # Recurse: archives often nest the series a few dirs deep.
    search_dir = _deepest_dicom_dir(dicom_dir)
    series_ids = reader.GetGDCMSeriesIDs(str(search_dir))
    if not series_ids:
        raise UploadError(
            "no DICOM series found in the upload — is it a CT/MR image "
            "series (not an RTSTRUCT / report)?"
        )
    best_id = max(
        series_ids,
        key=lambda sid: len(reader.GetGDCMSeriesFileNames(str(search_dir), sid)),
    )
    file_names = reader.GetGDCMSeriesFileNames(str(search_dir), best_id)
    if len(series_ids) > 1:
        logger.info(
            "upload has %d DICOM series; picked %s (%d slices)",
            len(series_ids), best_id, len(file_names),
        )
    if len(file_names) < 3:
        raise UploadError(
            f"DICOM series has only {len(file_names)} slice(s) — too few for "
            f"a 3-D volume"
        )
    reader.SetFileNames(file_names)
    try:
        image = reader.Execute()
    except Exception as exc:  # noqa: BLE001 — SimpleITK/GDCM error surface
        raise UploadError(f"failed to read DICOM series: {exc}") from exc
    sitk.WriteImage(image, str(out))
    return out


def _deepest_dicom_dir(root: Path) -> Path:
    """If an extracted archive has a single nested dir chain, descend it so
    GDCM's series scan starts where the slices actually live. Stops at the
    first dir that contains files or branches."""
    cur = root
    for _ in range(8):  # bounded — guards against pathological trees
        entries = [p for p in cur.iterdir()] if cur.is_dir() else []
        files = [p for p in entries if p.is_file()]
        subdirs = [p for p in entries if p.is_dir()]
        if files or len(subdirs) != 1:
            return cur
        cur = subdirs[0]
    return cur


def _extract_archive(arc: Path, dest: Path) -> None:
    """Safely extract a .zip/.tar archive into `dest` (rejects path escapes)."""
    if arc.name.endswith(".zip"):
        with zipfile.ZipFile(arc) as zf:
            for member in zf.namelist():
                _guard_member(member, dest)
            zf.extractall(dest)
        return
    if tarfile.is_tarfile(arc):
        with tarfile.open(arc) as tf:
            for member in tf.getmembers():
                _guard_member(member.name, dest)
            tf.extractall(dest)  # noqa: S202 — members guarded above
        return
    raise UploadError(f"{arc.name}: unsupported archive format")


def _guard_member(name: str, dest: Path) -> None:
    """Reject archive members that would escape `dest` (zip-slip)."""
    target = (dest / name).resolve()
    if not str(target).startswith(str(dest.resolve())):
        raise UploadError(f"archive member escapes extraction dir: {name!r}")


def _hash_dir(staged_dir: Path) -> str:
    """Stable sha256 over the raw uploaded files (content-addressed id).

    Hashes file bytes in name order, mixing the filename in so two single-file
    uploads with identical bytes but different names don't collide.
    """
    h = hashlib.sha256()
    for p in sorted(
        (q for q in staged_dir.iterdir() if q.is_file()), key=lambda x: x.name,
    ):
        h.update(p.name.encode())
        h.update(b"\0")
        h.update(p.read_bytes())
    return h.hexdigest()


def _is_nifti(name: str) -> bool:
    return name.lower().endswith(_NIFTI_SUFFIXES)


def _is_archive(name: str) -> bool:
    return name.lower().endswith(_ARCHIVE_SUFFIXES)

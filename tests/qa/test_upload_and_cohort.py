"""Host-runnable tests for the upload + multi-case QA features.

No GPU / nnUNet / Redis. The cohort additive-topup logic and the NIfTI
upload-ingest path are pure (nibabel only); the seed-from-prediction +
upload endpoints are exercised through a FastAPI TestClient with fixtures on
a throwaway COHORT_ROOT. DICOM ingest is skipped when SimpleITK is absent
(host) and runs inside the qa-viewer image.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Point the QA api's import-time roots at a throwaway dir BEFORE importing it.
# setdefault so we co-exist with test_crossval if it imported api first; the
# endpoint tests below read api.* roots so they work regardless of which dir won.
_ROOT = Path(tempfile.mkdtemp(prefix="qa-upload-test-"))
os.environ.setdefault("QA_FACTORY_ROOT", str(_ROOT))
os.environ.setdefault("QA_COHORT_ROOT", str(_ROOT / "qa-cohort"))
os.environ.setdefault("QA_RESULTS_ROOT", str(_ROOT / "results"))
os.environ.setdefault("QA_PREPROCESSED_ROOT", str(_ROOT / "preprocessed"))
os.environ.setdefault("QA_DATASETS_ROOT", str(_ROOT / "datasets"))
os.environ.setdefault("QA_VERDICTS_DB", str(_ROOT / "qa.sqlite"))
os.environ.setdefault("QA_WEB_STATIC_DIR", str(_ROOT / "noweb"))

import nibabel as nib  # noqa: E402

from modelfactory.qa import cohort  # noqa: E402
from modelfactory.qa.upload import UploadError, _guard_member, ingest_upload  # noqa: E402


def _write_nifti(path: Path, shape=(8, 8, 4)) -> None:
    arr = np.zeros(shape, dtype=np.int16)
    arr[2:5, 2:5, 1:3] = 1
    nib.save(nib.Nifti1Image(arr, affine=np.eye(4)), str(path))


def _make_dataset(datasets_root: Path, name: str, stems: list[str]) -> None:
    """A minimal nnUNet-raw dataset with dummy single-channel cases. No
    dataset.json → neither MR nor CT, so the axial filter is a no-op and we
    don't need valid image headers."""
    images = datasets_root / name / "imagesTr"
    labels = datasets_root / name / "labelsTr"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for s in stems:
        (images / f"{s}_0000.nii.gz").write_bytes(b"img-" + s.encode())
        (labels / f"{s}.nii.gz").write_bytes(b"gt-" + s.encode())


# ── cohort additive top-up ──────────────────────────────────────────────────


def test_additive_topup_keeps_case_ids_stable():
    tmp = Path(tempfile.mkdtemp(prefix="cohort-additive-"))
    datasets_root = tmp / "datasets"
    results_root = tmp / "results"  # empty → no trained models
    output = tmp / "qa-cohort"
    ds = "Dataset900_AdditiveTest"
    stems = ["case_a", "case_b", "case_c", "case_d", "case_e"]
    _make_dataset(datasets_root, ds, stems)

    def topup(n):
        return cohort.build_cohort_for_dataset(
            ds, datasets_root=datasets_root, results_root=results_root,
            output_root=output, region="abdomen_ct", n_pick=n, trained_models=[],
        )

    def manifest_cases():
        data = json.loads((output / "manifest.json").read_text())
        return {c["case_id"]: c["source_case_stem"]
                for c in data["cases"] if c["source_dataset"] == ds}

    # First build → exactly one case at index 001.
    new1 = topup(1)
    assert len(new1) == 1
    cases1 = manifest_cases()
    assert len(cases1) == 1
    first_id = next(iter(cases1))
    assert first_id.endswith("_case_001")
    first_stem = cases1[first_id]

    # Grow to 3 → two NEW cases; case_001 must keep the SAME source stem.
    new3 = topup(3)
    assert len(new3) == 2
    cases3 = manifest_cases()
    assert len(cases3) == 3
    assert cases3[first_id] == first_stem, "existing case_id was reassigned!"
    assert {cid.rsplit("_", 1)[1] for cid in cases3} == {"001", "002", "003"}
    assert len(set(cases3.values())) == 3, "stems must be distinct (no reuse)"

    # Re-running at the same target is a no-op.
    assert topup(3) == []

    # Asking for more than the pool tops up to the pool size, then stops.
    new_more = topup(10)
    cases_full = manifest_cases()
    assert len(cases_full) == 5
    assert len(new_more) == 2
    assert len(set(cases_full.values())) == 5
    assert topup(10) == []


def test_merge_into_manifest_preserves_uploaded_compatibility():
    tmp = Path(tempfile.mkdtemp(prefix="cohort-upload-merge-"))
    output = tmp / "qa-cohort"
    output.mkdir(parents=True)
    rec = cohort.CaseRecord(
        case_id="abdomen_ct/upload_deadbeef",
        region="abdomen_ct",
        source_dataset="uploaded",
        source_case_stem="scan.nii.gz",
        image_paths=["abdomen_ct/upload_deadbeef/image_0000.nii.gz"],
        groundtruth_path=None,
        compatible_models=["Dataset901_X::trainer__plans__3d_fullres"],
        uploaded=True,
    )
    # Some unrelated trained model exists, but for a DIFFERENT dataset.
    trained = [{"model_id": "Dataset902_Y::t__p__3d_fullres",
                "dataset_name": "Dataset902_Y"}]
    cohort._merge_into_manifest(output, [rec], trained)
    data = json.loads((output / "manifest.json").read_text())
    saved = next(c for c in data["cases"] if c["case_id"] == rec.case_id)
    assert saved["compatible_models"] == rec.compatible_models
    assert saved["uploaded"] is True


# ── upload ingest (NIfTI) ───────────────────────────────────────────────────


def test_ingest_upload_nifti_creates_case():
    tmp = Path(tempfile.mkdtemp(prefix="ingest-nifti-"))
    staged = tmp / "staged"
    staged.mkdir()
    _write_nifti(staged / "scan.nii.gz")
    cohort_root = tmp / "qa-cohort"
    mid = "Dataset903_Z::trainer__plans__3d_fullres"

    rec = ingest_upload(
        staged, model_id=mid, region="abdomen_ct", expected_channels=1,
        cohort_root=cohort_root, uploaded_by="tester", original_filename="scan.nii.gz",
    )
    assert rec.uploaded is True
    assert rec.source_dataset == "uploaded"
    assert rec.compatible_models == [mid]
    assert rec.groundtruth_path is None
    assert rec.case_id.startswith("abdomen_ct/upload_")
    assert rec.image_paths == [f"{rec.case_id}/image_0000.nii.gz"]

    case_dir = cohort_root / "abdomen_ct" / rec.case_id.split("/", 1)[1]
    assert (case_dir / "image_0000.nii.gz").is_file()
    src = json.loads((case_dir / "source.json").read_text())
    assert src["kind"] == "upload"
    assert src["original_filename"] == "scan.nii.gz"

    # Idempotent: same bytes → same case_id; a second model is unioned in.
    mid2 = "Dataset903_Z::other__plans__3d_fullres"
    rec2 = ingest_upload(
        staged, model_id=mid2, region="abdomen_ct", expected_channels=1,
        cohort_root=cohort_root, original_filename="scan.nii.gz",
    )
    assert rec2.case_id == rec.case_id
    assert set(rec2.compatible_models) == {mid, mid2}


def test_ingest_upload_channel_mismatch():
    tmp = Path(tempfile.mkdtemp(prefix="ingest-mismatch-"))
    staged = tmp / "staged"
    staged.mkdir()
    _write_nifti(staged / "scan.nii.gz")
    with pytest.raises(UploadError, match="expects 2 input channel"):
        ingest_upload(
            staged, model_id="D::c", region="abdomen_ct", expected_channels=2,
            cohort_root=tmp / "qa-cohort",
        )


def test_zip_slip_guard():
    tmp = Path(tempfile.mkdtemp(prefix="zipslip-"))
    with pytest.raises(UploadError, match="escapes"):
        _guard_member("../../etc/passwd", tmp)
    # A normal member is fine.
    _guard_member("series/img001.dcm", tmp)


# ── endpoints (TestClient) ──────────────────────────────────────────────────


def _client():
    from fastapi.testclient import TestClient

    from modelfactory.qa import api
    return TestClient(api.app), api


def _seed_uploaded_case(api, region: str, case_short: str, model_id: str) -> str:
    """Materialize an uploaded case dir + manifest entry. Returns full case_id."""
    full = f"{region}/{case_short}"
    case_dir = api.COHORT_ROOT / region / case_short
    case_dir.mkdir(parents=True, exist_ok=True)
    _write_nifti(case_dir / "image_0000.nii.gz")
    (case_dir / "source.json").write_text(json.dumps({
        "source_dataset": "uploaded", "kind": "upload",
        "compatible_models": [model_id],
    }))
    manifest = api.COHORT_ROOT / "manifest.json"
    data = {"version": 2, "regions": [region], "trained_models": [], "cases": []}
    if manifest.is_file():
        data = json.loads(manifest.read_text())
    data["cases"] = [c for c in data["cases"] if c["case_id"] != full] + [{
        "case_id": full, "region": region, "source_dataset": "uploaded",
        "source_case_stem": "scan", "image_paths": [f"{full}/image_0000.nii.gz"],
        "groundtruth_path": None, "compatible_models": [model_id], "uploaded": True,
    }]
    manifest.write_text(json.dumps(data))
    return full


def test_seed_from_prediction_unblocks_gt_edit():
    client, api = _client()
    region, case_short = "abdomen_ct", "upload_seedtest"
    mid = "Dataset904_Seed::trainer__plans__3d_fullres"
    full = _seed_uploaded_case(api, region, case_short, mid)

    # A finished prediction for this case on disk.
    pred_id = "predseedtest1"
    pdir = api.PREDICTIONS_ROOT / pred_id
    pdir.mkdir(parents=True, exist_ok=True)
    _write_nifti(pdir / "seg.nii.gz")
    (pdir / "status.json").write_text(json.dumps({
        "prediction_id": pred_id, "status": "done", "case_id": full,
    }))

    resp = client.post(
        f"/api/cases/{full}/groundtruth/seed-from-prediction",
        json={"prediction_id": pred_id},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["groundtruth_path"] == f"{full}/label_groundtruth.nii.gz"
    gt = api.COHORT_ROOT / region / case_short / "label_groundtruth.nii.gz"
    assert gt.is_file()
    # Manifest patched so the UI sees GT now exists.
    data = json.loads((api.COHORT_ROOT / "manifest.json").read_text())
    saved = next(c for c in data["cases"] if c["case_id"] == full)
    assert saved["groundtruth_path"] == f"{full}/label_groundtruth.nii.gz"


def test_seed_rejects_wrong_case_prediction():
    client, api = _client()
    region, case_short = "abdomen_ct", "upload_wrongcase"
    full = _seed_uploaded_case(api, region, case_short,
                               "Dataset905_W::t__p__3d_fullres")
    pred_id = "predwrong1"
    pdir = api.PREDICTIONS_ROOT / pred_id
    pdir.mkdir(parents=True, exist_ok=True)
    _write_nifti(pdir / "seg.nii.gz")
    (pdir / "status.json").write_text(json.dumps({
        "prediction_id": pred_id, "status": "done",
        "case_id": "abdomen_ct/some_other_case",
    }))
    resp = client.post(
        f"/api/cases/{full}/groundtruth/seed-from-prediction",
        json={"prediction_id": pred_id},
    )
    assert resp.status_code == 409, resp.text


def test_upload_endpoint_nifti_roundtrip():
    client, api = _client()
    # A model dir resolvable to a region with a single CT channel.
    ds = "Dataset906_UploadEP"
    cfg = "trainer__plans__3d_fullres"
    ds_json = json.dumps({
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "organ": 1},
        "tags": {"region": "abdomen", "modality": "CT"},
    })
    model_dir = api.RESULTS_ROOT / ds / cfg
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "dataset.json").write_text(ds_json)  # → channel count
    # Region resolution (tier 2 of _region_for) reads tags.region from the
    # dataset.json under the DATASETS root, not the model dir.
    ds_dir = api._datasets_root() / ds
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "dataset.json").write_text(ds_json)  # → region = abdomen_ct

    tmp = Path(tempfile.mkdtemp(prefix="upload-ep-"))
    nii = tmp / "myscan.nii.gz"
    _write_nifti(nii)

    with nii.open("rb") as fh:
        resp = client.post(
            "/api/cohort/uploads",
            data={"model_id": f"{ds}::{cfg}", "reviewer": "gustavo"},
            files={"files": ("myscan.nii.gz", fh, "application/gzip")},
        )
    assert resp.status_code == 201, resp.text
    case = resp.json()
    assert case["uploaded"] is True
    assert case["region"] == "abdomen_ct"
    assert case["compatible_models"] == [f"{ds}::{cfg}"]
    assert case["case_id"].startswith("abdomen_ct/upload_")
    case_dir = api.COHORT_ROOT / "abdomen_ct" / case["case_id"].split("/", 1)[1]
    assert (case_dir / "image_0000.nii.gz").is_file()

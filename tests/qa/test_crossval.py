"""Host-runnable tests for the cross-validation QA feature.

No GPU / nnUNet / Redis: the pure helpers (OOF routing, aggregation, rollup,
HTML/CSV rendering) are exercised directly, and the orchestrator is exercised
end-to-end with the single GPU-bound call (`_execute_one_prediction`) stubbed.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

from modelfactory.qa import crossval, report

# QA api reads its roots from env at import time — point them at a throwaway
# dir BEFORE the api import (deferred to inside the orchestrator test). crossval
# and report are env-independent, so they import normally above.
_ROOT = Path(tempfile.mkdtemp(prefix="qa-cv-test-"))
os.environ.setdefault("QA_FACTORY_ROOT", str(_ROOT))
os.environ.setdefault("QA_COHORT_ROOT", str(_ROOT / "qa-cohort"))
os.environ.setdefault("QA_RESULTS_ROOT", str(_ROOT / "results"))
os.environ.setdefault("QA_PREPROCESSED_ROOT", str(_ROOT / "preprocessed"))
os.environ.setdefault("QA_VERDICTS_DB", str(_ROOT / "qa.sqlite"))
os.environ.setdefault("QA_WEB_STATIC_DIR", str(_ROOT / "noweb"))


# ── OOF routing ─────────────────────────────────────────────────────────────


def _write_splits(tmp_path: Path) -> Path:
    splits = [
        {"train": ["b", "c"], "val": ["a"]},
        {"train": ["a", "c"], "val": ["b"]},
        {"train": ["a", "b"], "val": ["c"]},
    ]
    p = tmp_path / "splits_final.json"
    p.write_text(json.dumps(splits))
    return p


def test_oof_in_val(tmp_path):
    p = _write_splits(tmp_path)
    r = crossval.oof_fold_for_case(p, "b", [0, 1, 2])
    assert r.oof_fold == 1 and r.resolvable and r.reason == "oof"


def test_oof_external_case(tmp_path):
    p = _write_splits(tmp_path)
    r = crossval.oof_fold_for_case(p, "zzz", [0, 1, 2])
    assert r.oof_fold is None and not r.resolvable and r.reason == "external"


def test_oof_fold_untrained(tmp_path):
    p = _write_splits(tmp_path)
    # "c" is held out by fold 2, but fold 2 has no checkpoint yet.
    r = crossval.oof_fold_for_case(p, "c", [0, 1])
    assert r.oof_fold is None and r.reason == "oof_fold_untrained"


def test_oof_missing_splits(tmp_path):
    r = crossval.oof_fold_for_case(tmp_path / "nope.json", "a", [0, 1, 2])
    assert r.reason == "no_splits" and not r.resolvable


# ── aggregation ──────────────────────────────────────────────────────────────


def _entries():
    def fold(k, oof, md, a, b, hd=None):
        return {
            "kind": "fold", "fold": k, "is_oof": oof, "mean_fg_dice": md,
            "metrics": [
                {"label": 1, "label_name": "A", "dice": a, "hd95_mm": hd},
                {"label": 2, "label_name": "B", "dice": b, "hd95_mm": None},
            ],
        }
    return [
        fold(0, False, 0.90, 0.95, 0.85),
        fold(1, True, 0.80, 0.90, 0.70, 3.0),
        fold(2, False, 0.88, 0.92, 0.84),
        {"kind": "ensemble", "fold": None, "is_oof": False, "mean_fg_dice": 0.93,
         "metrics": [
             {"label": 1, "label_name": "A", "dice": 0.96, "hd95_mm": 1.5},
             {"label": 2, "label_name": "B", "dice": 0.90, "hd95_mm": None},
         ]},
    ]


def test_aggregate_oof_headline():
    agg = crossval.aggregate_cv(_entries(), oof_fold=1)
    assert agg["headline_kind"] == "oof"
    assert abs(agg["headline_mean_fg_dice"] - 0.80) < 1e-9
    assert abs(agg["ensemble_mean_fg_dice"] - 0.93) < 1e-9
    a = next(p for p in agg["per_label"] if p["label"] == 1)
    assert abs(a["dice_mean"] - (0.95 + 0.90 + 0.92) / 3) < 1e-9
    assert a["dice_min"] == 0.90 and a["dice_max"] == 0.95
    assert a["oof_dice"] == 0.90  # fold 1's value for label A
    assert a["dice_std"] is not None and a["dice_std"] > 0


def test_aggregate_no_oof_falls_back():
    agg = crossval.aggregate_cv(_entries(), oof_fold=None)
    assert agg["headline_kind"] == "cross_fold_mean"
    assert abs(agg["headline_mean_fg_dice"] - (0.90 + 0.80 + 0.88) / 3) < 1e-9


# ── rollup + rendering ──────────────────────────────────────────────────────


def _done_cv():
    entries = _entries()
    return {
        "cv_run_id": "r1", "model_id": "Dataset063_X::t__p__c",
        "case_id": "brain_mr/d063_case_001", "source_case_stem": "a",
        "region": "brain_mr", "dataset_name": "Dataset063_X", "status": "done",
        "compute_hd95": "oof_and_ensemble", "oof_fold": 1, "oof_resolvable": True,
        "oof_reason": "oof", "available_folds": [0, 1, 2], "gt_revision": None,
        "label_map": {"background": 0, "A": 1, "B": 2},
        "updated_at": "2026-05-29T00:00:00+00:00",
        "entries": entries, "aggregate": crossval.aggregate_cv(entries, 1),
    }


def test_build_model_report_coverage():
    cv = _done_cv()
    rep = crossval.build_model_report(
        cv["model_id"], cv["dataset_name"], [cv],
        ["brain_mr/d063_case_001", "brain_mr/d063_case_002"],
    )
    assert rep["n_cases_with_cv"] == 1
    assert rep["n_cases_total"] == 2
    assert rep["n_with_oof"] == 1
    assert abs(rep["honest_mean_fg_dice"] - 0.80) < 1e-9
    assert any(c["status"] == "pending" for c in rep["cases"])


def test_render_case_html_self_contained():
    html = report.render_case_html(_done_cv())
    assert html.startswith("<!doctype html>")
    assert "<script" not in html and "googleapis" not in html  # offline, no JS
    assert "★" in html and "Out-of-fold" in html


def test_render_case_csv_rows():
    csv_text = report.render_case_csv(_done_cv())
    lines = csv_text.strip().splitlines()
    assert "is_oof" in lines[0]
    assert len(lines) - 1 == 8  # 4 entries × 2 labels


def test_render_rollup_html_and_csv():
    cv = _done_cv()
    rep = crossval.build_model_report(cv["model_id"], cv["dataset_name"], [cv], [])
    html = report.render_rollup_html(rep)
    assert "Rollup" in html and "CV runs:" in html and "<script" not in html
    csv_text = report.render_rollup_csv(rep)
    assert csv_text.startswith("# model_id,") and "per_fold" in csv_text


# ── orchestrator integration (GPU call stubbed) ─────────────────────────────


def test_orchestrator_end_to_end(monkeypatch):
    from modelfactory.qa import api

    dataset = "Dataset063_Brain_MR_FullBrain_Generalist"
    cfg = "nnUNetTrainerMLflow__nnUNetResEncUNetLPlans__3d_fullres"
    model_id = f"{dataset}::{cfg}"
    case_id = "brain_mr/d063_case_001"
    stem = "HCP_Wu_Minn-sub-0041"

    model_dir = api.RESULTS_ROOT / dataset / cfg
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "dataset.json").write_text(
        json.dumps({"labels": {"background": 0, "Brainstem": 1}})
    )
    for k in (0, 1, 2):
        fd = model_dir / f"fold_{k}"
        fd.mkdir(exist_ok=True)
        (fd / "checkpoint_best.pth").write_bytes(b"x")

    splits_dir = api.PREPROCESSED_ROOT / dataset
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "splits_final.json").write_text(json.dumps([
        {"train": [stem], "val": ["other0"]},
        {"train": ["other0"], "val": [stem]},   # fold 1 holds the case out
        {"train": [stem], "val": ["other2"]},
    ]))

    api.COHORT_ROOT.mkdir(parents=True, exist_ok=True)
    (api.COHORT_ROOT / "manifest.json").write_text(json.dumps({
        "version": 2, "regions": ["brain_mr"],
        "cases": [{"case_id": case_id, "source_case_stem": stem,
                   "compatible_models": [model_id]}],
    }))
    case_dir = api.COHORT_ROOT / "brain_mr" / "d063_case_001"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "image_0000.nii.gz").write_bytes(b"x")

    # Stub the only GPU-bound call: return canned per-fold metrics. The
    # compute_hd95 flag is echoed into the metric so we can assert the policy.
    async def fake_exec(*, prediction_id, out_dir, model_id, case_id, model_dir,
                        folds, case_dir, plan_hash, compute_hd95, do_meshes):
        fold = folds[0] if len(folds) == 1 else None
        md = {0: 0.90, 1: 0.80, 2: 0.88}.get(fold, 0.93)
        return {
            "prediction_id": prediction_id, "status": "done",
            "metrics": [{"label": 1, "label_name": "Brainstem", "dice": md,
                         "hd95_mm": (2.0 if compute_hd95 else None),
                         "n_voxels_gt": 900, "n_voxels_pred": 890}],
            "mean_fg_dice": md, "elapsed_s": 1.0, "error": None,
        }

    async def fake_submit(**_kw):
        return None

    monkeypatch.setattr(api, "_execute_one_prediction", fake_exec)
    monkeypatch.setattr(api.predict_queue, "submit", fake_submit)

    cv_run_id = "testrun01"
    cv_dir = api.CROSSVAL_ROOT / cv_run_id
    cv_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(api._run_crossval_background(
        cv_run_id=cv_run_id, cv_dir=cv_dir, model_id=model_id, case_id=case_id,
        model_dir=model_dir, case_dir=case_dir, plan_hash="ph",
        reviewer="tester", compute_hd95="oof_and_ensemble",
    ))

    cv = json.loads((cv_dir / "cv.json").read_text())
    assert cv["status"] == "done"
    assert cv["oof_fold"] == 1 and cv["oof_resolvable"]
    assert cv["folds_done"] == 4  # 3 folds + ensemble
    oof = [e for e in cv["entries"] if e.get("is_oof")]
    assert len(oof) == 1 and oof[0]["fold"] == 1
    # HD95 policy "oof_and_ensemble": OOF fold gets hd95, a non-OOF fold doesn't.
    assert oof[0]["metrics"][0]["hd95_mm"] == 2.0
    non_oof = next(e for e in cv["entries"] if e["kind"] == "fold" and not e["is_oof"])
    assert non_oof["metrics"][0]["hd95_mm"] is None
    assert cv["aggregate"]["headline_kind"] == "oof"
    assert abs(cv["aggregate"]["headline_mean_fg_dice"] - 0.80) < 1e-9

"""FastAPI backend for the model-factory QA viewer.

This module is the entire server side: it serves the Next.js static export
at `/` AND the JSON API at `/api/*` from the same uvicorn process. One pod,
one image, one container.

Endpoints under /api/* serve:
  - the cohort manifest (which cases x which trained models)
  - the case images and groundtruth NIfTI streams (read directly off NFS)
  - inference requests, scheduled against a per-process predictor LRU cache
  - the resulting segmentation NIfTI + per-label metrics
  - QA verdicts (accept/reject/needs_review) persisted in SQLite on NFS

The pod is single-replica, single-worker — one CUDA context (whole GPU 0),
one in-process predictor cache. Inference is serialized via a global asyncio
lock (predictor.predict_logits is not safe to call concurrently on the same
network).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import functools
import json
import logging
import math
import os
import re
import shutil
import tempfile
import time
import uuid
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from modelfactory.qa import cache as result_cache
from modelfactory.qa.gt_corrections import GroundTruthStore
from modelfactory.qa.queue import queue as predict_queue
from modelfactory.qa.schedule import (
    PLANNED_STATUSES,
    ScheduleStore,
    project_schedule,
)
from modelfactory.qa.themes import PALETTE, ModelThemeStore
from modelfactory.qa.verdicts import (
    REJECT_REASONS,
    VERDICT_VALUES,
    VerdictStore,
    approval_status_for,
)

logger = logging.getLogger("qa-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


# ── paths ────────────────────────────────────────────────────────────────

FACTORY_ROOT = Path(os.environ.get("QA_FACTORY_ROOT", "/factory"))
COHORT_ROOT = Path(os.environ.get("QA_COHORT_ROOT", FACTORY_ROOT / "qa-cohort"))
RESULTS_ROOT = Path(os.environ.get("QA_RESULTS_ROOT", FACTORY_ROOT / "results"))
# nnUNet preprocessed root — where convert.py writes splits_final.json
# (preprocessed/<Dataset###_X>/splits_final.json). This is NOT the QA npz
# cache under COHORT_ROOT/preprocessed; the names collide but the roots differ.
PREPROCESSED_ROOT = Path(
    os.environ.get("QA_PREPROCESSED_ROOT", FACTORY_ROOT / "preprocessed")
)
PREDICTIONS_ROOT = COHORT_ROOT / "predictions"
# Cross-validation run manifests. One dir per cv_run_id holding cv.json; the
# child single-fold + ensemble predictions live under PREDICTIONS_ROOT (so the
# existing seg/mesh/metrics routes + GT-recompute scan resolve them unchanged).
CROSSVAL_ROOT = COHORT_ROOT / "crossval"
VERDICTS_DB = Path(os.environ.get("QA_VERDICTS_DB", COHORT_ROOT / "qa.sqlite"))
# Number of parallel training slots the factory runs (GPUs 3-7 MIG = 10
# 3g.40gb slices today). Used only to *project* when queued trainings will
# start/finish on the calendar — the qa pod can't see live MIG/Kueue state,
# so this is a configured approximation, not an allocator.
TRAINING_SLOTS = int(os.environ.get("QA_TRAINING_SLOTS", "10"))
WEB_STATIC_DIR = Path(os.environ.get("QA_WEB_STATIC_DIR", "/opt/qa-viewer/web"))


def _ensure_predictions_root() -> None:
    """Create the predictions output dir on demand (avoids ImportError outside the pod)."""
    PREDICTIONS_ROOT.mkdir(parents=True, exist_ok=True)

# Verdicts persistence is opened lazily so this module imports cleanly in
# tests without an NFS mount. The GT-corrections and model-themes stores
# share the same SQLite file (separate tables).
_verdict_store: VerdictStore | None = None
_gt_store: GroundTruthStore | None = None
_theme_store: ModelThemeStore | None = None
_schedule_store: ScheduleStore | None = None


def _get_verdicts() -> VerdictStore:
    global _verdict_store
    if _verdict_store is None:
        _verdict_store = VerdictStore(VERDICTS_DB)
    return _verdict_store


def _get_schedule() -> ScheduleStore:
    global _schedule_store
    if _schedule_store is None:
        _schedule_store = ScheduleStore(VERDICTS_DB)
    return _schedule_store


def _get_gt_store() -> GroundTruthStore:
    global _gt_store
    if _gt_store is None:
        _gt_store = GroundTruthStore(VERDICTS_DB)
    return _gt_store


def _get_themes() -> ModelThemeStore:
    global _theme_store
    if _theme_store is None:
        _theme_store = ModelThemeStore(VERDICTS_DB)
    return _theme_store


# Per-case asyncio locks for the metric-recompute background task. Keyed
# by "<region>/<case>" to allow recompute of one case to proceed while
# another case's predictions / edits run concurrently.
_recompute_locks: dict[str, asyncio.Lock] = {}
# Snapshots of recompute progress so the frontend can poll a small JSON
# blob instead of trying to count metrics_vN.json files itself.
_recompute_status: dict[str, dict] = {}

# Cross-validation in-flight guard, keyed "<model_id>::<case_id>". A single
# event loop makes plain dict access race-free; this rejects a duplicate CV
# on the same (model, case) with 409 instead of double-spending the GPU.
_crossval_inflight: dict[str, str] = {}


# Soft caps on GT-edit uploads. A 512x512x300 uint8 labelmap is ~75 MB;
# anything beyond `MAX_GT_BYTES` is almost certainly a misformed request.
MAX_GT_BYTES = 512 * 1024 * 1024  # 512 MB

# Cap on a reviewer-uploaded test case (DICOM series .zip or NIfTI). A full
# CT series .zip is typically 50-300 MB; 2 GiB leaves headroom for large
# multi-channel volumes while bounding NFS churn from accidental huge files.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB


# ── app ──────────────────────────────────────────────────────────────────

app = FastAPI(title="model-factory QA API", version="0.9.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Cross-origin isolation headers — required for SharedArrayBuffer, which
# Cornerstone3D's volume loader uses to decode volumes in parallel without
# copying buffers between the main thread and the decoder workers. Without
# these headers, the browser disables SAB and the viewer fails with
# "SharedArrayBuffer is NOT supported in your browser".
#
# Notes:
#   - Page must ALSO be served from a secure context (HTTPS, or http://localhost
#     / 127.0.0.1) for the browser to honor these headers — that's a browser
#     rule, independent of the headers themselves.
#   - COEP=require-corp blocks cross-origin sub-resources that don't send a
#     CORP header. We're fully same-origin (FastAPI serves both / and /api/*
#     from one host; next/font and brand PNGs are local). If a future change
#     adds external CDN assets, switch this to `credentialless` or self-host.
@app.middleware("http")
async def _cross_origin_isolation(request, call_next):
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    _apply_static_cache_headers(request, response)
    return response


def _apply_static_cache_headers(request, response) -> None:
    """Bust browser caching of `index.html` while caching hashed chunks forever.

    Next.js static export writes content-hashed filenames under
    `_next/static/*` — `polyfills-{hash}.js`, `chunks/{hash}.js`, etc.
    These are immutable: the same hash means the same bytes forever, so
    a long max-age is safe. `index.html` (and any other entry HTML) is
    NOT hashed — every deploy can change which chunk hashes it points
    at, so if the browser caches it, a stale visitor's bundle breaks the
    next deploy with `ChunkLoadError: Loading chunk 927 failed`.

    FastAPI's StaticFiles handler doesn't set Cache-Control by default,
    so we add it here. /api/* responses already set their own headers
    upstream — skip them.
    """
    if response.status_code >= 400:
        return
    path = request.url.path
    if path.startswith("/api/"):
        return
    if path.startswith("/_next/static/"):
        # Hashed in the filename → cache forever, never revalidate.
        response.headers.setdefault(
            "Cache-Control", "public, max-age=31536000, immutable",
        )
        return
    if path == "/" or path.endswith(".html"):
        # Entry HTML — must always re-validate; otherwise stale visitors
        # try to load chunks whose hashes have rotated out.
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return


# Lazy predictor cache — only built on first /api/predict so the import path
# of /api/healthz / /api/models stays torch-free during local dev.
_predictor_cache = None

# Cohort-manifest serialisation. manifest.json is read-modify-written by
# donations, uploads, and GT seeding alike; they must share ONE lock or a
# concurrent pair would lose an update (the atomic tmp-rename prevents
# corruption, not lost writes). Each critical section is short — the slow
# parts (disk copy, DICOM→NIfTI conversion) run BEFORE the lock is taken.
_donate_lock = asyncio.Lock()


def _get_cache():
    global _predictor_cache
    if _predictor_cache is None:
        from modelfactory.inference.predictor_cache import PredictorCache
        # Default 6 so a full 5-fold cross-validation run (5 single-fold
        # predictors + 1 ensemble) stays resident without mid-run LRU
        # thrash. The cross-val orchestrator also raises max_size to
        # max(current, available_folds + 1) before a run, so a future
        # 10-fold model widens the cache instead of re-thrashing.
        _predictor_cache = PredictorCache(
            max_size=int(os.environ.get("QA_PREDICTOR_CACHE_SIZE", "6"))
        )
    return _predictor_cache


# ── schemas ──────────────────────────────────────────────────────────────


ModelStatus = Literal["training", "done", "stopped", "failed"]


class FoldProgress(BaseModel):
    """Per-fold detail for one model. nnUNetv2 trains 5-fold cross-
    validation: each fold trains on 4/5 of the data and validates on the
    held-out 1/5, then we can ensemble across folds at inference time.
    Folds are independent jobs, so it's common to have fold 0 done while
    folds 1+2 are mid-training — the catalog UI surfaces each fold's
    state separately so a completed fold doesn't mask a live one.
    """
    fold: int
    status: ModelStatus
    current_epoch: int | None = None
    total_epochs: int | None = None
    val_mean_fg_dice: float | None = None
    has_checkpoint_best: bool = False
    # ── live training-rate + ETA (populated only for `training` folds) ──
    # Median wall-clock seconds per epoch over the most-recent window,
    # derived from the newest `training_log_*.txt` "Epoch time: X s"
    # lines (reliable + positive) with a metrics.jsonl `ts`-delta
    # fallback. None when the fold isn't actively producing epochs or
    # no timing source is parseable.
    sec_per_epoch: float | None = None
    # UTC ISO8601. When the fold began producing epochs (metrics.jsonl
    # run_start ts when present, else derived as now − current·rate) and
    # when the most-recent epoch landed (newest training_log mtime).
    started_at: str | None = None
    last_epoch_at: str | None = None
    # Snapshot remaining-seconds and the absolute UTC finish instant at
    # scan time. The frontend ticks a live countdown toward `est_finish`
    # between the 30 s catalog refetches, so the schedule stays "real
    # time" without hammering the API.
    eta_seconds: float | None = None
    est_finish: str | None = None


class ModelInfo(BaseModel):
    model_id: str
    dataset_name: str
    configuration: str
    trainer: str
    plans: str
    region: str | None
    available_folds: list[int]
    model_dir: str
    val_mean_fg_dice: float | None = None
    last_modified: str | None = None
    # Derived from filesystem heuristics over each fold's checkpoint +
    # training log freshness. See _compute_model_status().
    status: ModelStatus = "done"
    # Latest training epoch observed in any fold's metrics.jsonl (max
    # across folds) plus the trainer's configured epoch count. Both
    # `None` if no metrics.jsonl exists yet (fresh fold) or the read
    # fails. UI surfaces this as "epoch X / Y" or a progress bar.
    current_epoch: int | None = None
    total_epochs: int | None = None
    # Per-structure mean dice from the most recent fold's
    # validation/summary.json — {label_name: mean_dice across cases}.
    # None for in-flight training (no validation pass yet) and for
    # historical folds missing summary.json.
    per_class_dice: dict[str, float] | None = None
    # Dragonfly cache hits for the model's compatible QA-cohort cases —
    # how many of `cohort_size` have a cached inference. Both 0 when
    # cohort manifest is absent or Redis is unreachable.
    cached_count: int = 0
    cohort_size: int = 0
    # Model-level QA decision DERIVED from the per-case verdict tallies
    # (verdicts.approval_status_for): "approved" | "rejected" | "pending".
    # A flag like cached_count — drives the green/red card + ✓/✗ badge.
    approval_status: str = "pending"
    # Per-fold rollup — present when at least one fold dir exists on
    # disk. Sorted by fold index. The catalog renders one progress row
    # per fold so a 5-fold CV run with fold 0 done and fold 1+2 at
    # epoch 19 reads as exactly that, instead of collapsing to a single
    # ambiguous "epoch 1000 / 1000".
    folds: list[FoldProgress] = []


class CaseInfo(BaseModel):
    case_id: str
    region: str
    source_dataset: str
    source_case_stem: str
    image_paths: list[str]
    groundtruth_path: str | None
    compatible_models: list[str]
    # True for ad-hoc cases uploaded via the QA viewer (DICOM/NIfTI). They
    # have no donated GT (groundtruth_path stays null) until a reviewer seeds
    # one from the model's prediction. Optional w/ default so manifests
    # written before this field still parse.
    uploaded: bool = False


class CohortResponse(BaseModel):
    version: int
    regions: list[str]
    cases: list[CaseInfo]
    trained_models: list[ModelInfo]


class PredictRequest(BaseModel):
    model_id: str
    case_id: str
    use_folds: str | list[int] = Field(
        default="best",
        description="'best' = one fold (lowest available index), "
                    "'all' = every fold under the model, or an explicit list.",
    )
    # Reviewer-name string carried into the queue widget so a second user
    # sees "QA: <reviewer> · Dataset090 …" instead of an anonymous slot.
    # Free-text; matches the reviewer field on verdicts.
    reviewer: str = ""


class CrossvalRequest(BaseModel):
    model_id: str
    case_id: str
    reviewer: str = ""
    # HD95 is the expensive metric (~1 min/label brute-force). For a 5-fold CV
    # that is 6 runs, so the default keeps the fold spread Dice-only:
    #   "none"             — no HD95 anywhere (MVP default; fold-agreement via
    #                        Dice mean±std is the headline signal)
    #   "oof_and_ensemble" — HD95 only on the unbiased OOF fold + the ensemble
    #   "all"              — HD95 on every fold (slow; opt-in)
    compute_hd95: Literal["none", "oof_and_ensemble", "all"] = "none"


class LabelMetricOut(BaseModel):
    label: int
    label_name: str
    dice: float | None
    hd95_mm: float | None
    n_voxels_gt: int
    n_voxels_pred: int


class PredictAcceptedResponse(BaseModel):
    """Returned by POST /api/predict — inference runs asynchronously.

    Clients poll `status_url` until `status` is `done` or `error`. The async
    pattern keeps the HTTP round-trip short (sub-second) so any proxy in the
    chain (Cloudflare, pfSense HAProxy, etc.) doesn't time out the inference
    on long cases (TTA + 0.25 step on ResEncXL can hit 60-90 s).
    """
    prediction_id: str
    status: str  # "queued" — or "done" when this is a cache hit
    status_url: str
    # True when the (plan_hash, model, folds, case) tuple already had a
    # finished prediction in Redis. The seg + meshes are on disk; the
    # client can skip polling and read /status once for the full payload.
    from_cache: bool = False
    # Queue observability so the second user knows they're waiting. 0
    # means "next up", N>0 means N predictions ahead. `queue_depth` is
    # the total in-flight count at submission time.
    position_in_queue: int | None = None
    queue_depth: int = 0
    eta_s: float | None = None


class PostprocessingInfo(BaseModel):
    """What nnUNetv2 actually applied to logits before writing the seg NIfTI.

    Populated at the `seg_ready` flip; surfaced read-only in the
    InferencePanel so reviewers can spot when inference settings drift
    between runs (e.g. someone toggled mirroring off and dice dropped).
    Currently every predictor in this pod uses the same hard-coded
    settings — see PREDICTOR_FLAGS in modelfactory.inference.run — but
    storing them per prediction makes the data already-correct the day
    they become configurable.
    """
    # Runtime predictor flags
    test_time_augmentation: bool
    gaussian_tile_blending: bool
    tile_step_size: float
    perform_everything_on_device: bool
    # Geometry (per case)
    network_spacing: list[float]
    original_spacing: list[float]
    resampling_order_seg: int
    # Pipeline order (informational, fixed by nnUNetv2 internals).
    pipeline: list[str]
    # Dataset-level postprocessing (None for current models — would be
    # populated if we ran `nnUNetv2_determine_postprocessing`).
    region_class_order: list[list[int]] | None = None
    keep_largest_component: dict[str, bool] | None = None
    has_postprocessing_pkl: bool = False


class PredictionStatus(BaseModel):
    prediction_id: str
    # State machine:
    #   queued → running → seg_ready → done
    # `seg_ready` means the segmentation is on disk and label_map/seg_url/
    # elapsed_s/used_preprocessed_cache are populated — the viewer can
    # render the overlay. Metrics arrive on the subsequent `done` flip
    # (or, if metrics computation failed, `done` with metrics=null and
    # metrics_error set — never `error`, which is reserved for inference
    # failures).
    status: str  # "queued" | "running" | "seg_ready" | "done" | "error"
    model_id: str
    case_id: str
    folds: list[int]
    started_at: str
    updated_at: str
    # Populated from `seg_ready` onward
    elapsed_s: float | None = None
    used_preprocessed_cache: bool | None = None
    label_map: dict[str, int] | None = None
    seg_url: str | None = None
    metrics_url: str | None = None
    # Populated when status == "done" (may be null if metrics_error set)
    metrics: list[LabelMetricOut] | None = None
    metrics_error: str | None = None
    # Populated when status == "error" (inference failure)
    error_type: str | None = None
    error_message: str | None = None
    # Populated from seg_ready onward when the metadata is reachable.
    postprocessing: PostprocessingInfo | None = None
    # The GT revision the latest metrics were computed against. None when
    # the case has no corrections (metrics computed against the original
    # cohort GT file).
    active_gt_revision: int | None = None
    # Backend mesh precompute progresses independently of `status`. Stays
    # at `status="done"` so the existing polling loop terminates; mesh
    # state is read by the 3D canvas separately.
    #   None → no mesh precompute attempted (legacy prediction)
    #   "pending" → mesh job started but not yet finished
    #   "ready"  → all per-label .vtp files on disk
    #   "failed" → see `meshes_error`
    meshes_status: str | None = None
    meshes_elapsed_s: float | None = None
    meshes_by_label: dict[str, str] | None = None
    meshes_error: str | None = None
    # Live queue position. Set while the prediction is in
    # PredictQueue (i.e. between submit and remove). Stays at None once
    # the prediction has been removed from the queue (status `done` or
    # `error`). `0` = next to start; `>0` = N ahead.
    position_in_queue: int | None = None
    queue_depth: int | None = None


class QueueEntryOut(BaseModel):
    """One row in GET /api/queue."""

    prediction_id: str
    model_id: str
    case_id: str
    reviewer: str
    state: str  # "queued" | "running"
    submitted_at: str
    started_at: str | None = None
    position_in_queue: int
    eta_s: float | None = None


class QueueResponse(BaseModel):
    depth: int
    in_flight: list[QueueEntryOut]


class DonateCaseRequest(BaseModel):
    """POST /api/cohort/cases body. Triggers the build_cohort_for_dataset
    pipeline for one dataset on demand so the frontend can fix the
    "no compatible cases" dead-end without a CLI hop."""

    model_id: str | None = None
    dataset_name: str | None = None
    region: str | None = None
    n_pick: int = 1


class DonateCaseResponse(BaseModel):
    region: str
    dataset_name: str
    new_cases: list[CaseInfo]
    already_existed: bool


class SeedGtRequest(BaseModel):
    """POST .../groundtruth/seed-from-prediction body. Copies a model
    prediction into an uploaded case's `label_groundtruth.nii.gz` so the
    reviewer can correct it in the editor and save it as a training label."""

    prediction_id: str


class GroundTruthRevisionOut(BaseModel):
    id: int
    region: str
    case_id: str
    revision: int
    path: str
    base_prediction_id: str | None
    reviewer: str
    notes: str
    status: str
    created_at: str


class RecomputeStatusOut(BaseModel):
    pending: int
    completed: int
    total: int
    error: str | None = None


class ModelThemeOut(BaseModel):
    model_id: str
    color_key: str
    updated_by: str
    updated_at: str


class ModelThemeIn(BaseModel):
    color_key: str
    updated_by: str = ""


class VerdictRequest(BaseModel):
    prediction_id: str
    model_id: str
    case_id: str
    verdict: str = Field(description=f"One of: {', '.join(VERDICT_VALUES)}")
    notes: str = ""
    reviewer: str = ""
    fold_choice: str = "best"
    mean_dice: float | None = None
    # Structured failure category — only honored when verdict == "reject".
    # One of REJECT_REASONS, or "" for an unspecified reject.
    reject_reason: str = ""


class VerdictOut(BaseModel):
    id: int
    prediction_id: str
    model_id: str
    case_id: str
    verdict: str
    notes: str
    reviewer: str
    fold_choice: str
    mean_dice: float | None
    created_at: str
    reject_reason: str = ""
    # Approve-and-next plumbing: the next compatible case for this model
    # that this reviewer has not yet verdicted. None when the reviewer
    # has worked through every compatible case (or no compatible cases
    # exist).
    next_case_id: str | None = None


class VerdictSummaryOut(BaseModel):
    model_id: str
    accept: int
    reject: int
    needs_review: int
    total: int
    last_at: str | None
    last_verdict: str | None
    # Derived model decision + per-reason reject breakdown (see
    # verdicts.approval_status_for).
    approval_status: str = "pending"
    reject_reasons: dict[str, int] | None = None


# ── endpoints ────────────────────────────────────────────────────────────


@app.get("/api/healthz")
def healthz():
    cache_state = _predictor_cache.loaded() if _predictor_cache else []
    return {
        "status": "ok",
        "time": dt.datetime.now(dt.timezone.utc).isoformat(),
        "loaded_models": cache_state,
        "factory_root_exists": FACTORY_ROOT.is_dir(),
        "cohort_manifest_exists": (COHORT_ROOT / "manifest.json").is_file(),
    }


@app.get("/api/livez")
def livez() -> dict[str, bool]:
    # Pure liveness — no I/O, no cache access, no NFS stat. The probe must
    # succeed whenever the asyncio loop can pick up *any* request, regardless
    # of GPU/predictor/NFS state. nnUNet predict offloaded to asyncio.to_thread
    # still holds the GIL inside the worker thread (torch.load, numpy resample,
    # NIfTI I/O), starving the event loop for seconds at a time. /api/healthz
    # remains for richer human checks; k8s probes use this.
    return {"ok": True}


@app.get("/api/models", response_model=list[ModelInfo])
def list_models():
    """Filesystem scan of /factory/results/<Dataset>/<config>/fold_N/."""
    from modelfactory.qa.cohort import _discover_trained_models, _region_for

    datasets_root = _datasets_root()
    entries = _discover_trained_models(RESULTS_ROOT, datasets_root=datasets_root)

    # Per-model QA decision, derived from the verdict tallies. One grouped
    # query up front (same load-once pattern as the cohort manifest below),
    # then a dict lookup per model — no extra work in the per-model loop.
    approval_by_model: dict[str, str] = {}
    try:
        for s in _get_verdicts().summary():
            approval_by_model[s.model_id] = s.approval_status
    except Exception as exc:  # noqa: BLE001 — opportunistic, never fatal
        logger.warning("verdict summary unavailable for approval flags: %s", exc)

    # Cohort manifest is opportunistic: when present, we can tell the UI
    # how many of each model's compatible cases are warm in Dragonfly.
    # When absent (fresh deploy, cohort not yet prepared), cached_count
    # and cohort_size both stay 0 and the badge hides itself client-side.
    cohort_cases: list[dict] = []
    manifest_path = COHORT_ROOT / "manifest.json"
    if manifest_path.is_file():
        try:
            cohort_cases = json.loads(manifest_path.read_text()).get("cases", []) or []
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("cohort manifest unreadable, skipping cached_count: %s", exc)

    out: list[ModelInfo] = []
    for e in entries:
        config_dir = Path(e["model_dir"])
        last_mod = None
        try:
            stat = config_dir.stat()
            last_mod = dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat()
        except OSError:
            pass

        # Resolve dataset label map once per model so we can name per-class
        # dice rows in the summary.json fallback below.
        int_to_name = {
            v: k for k, v in _load_label_map(config_dir).items() if k != "background"
        }

        # val_dice + per_class_dice resolution. Try the live-training
        # source first (metrics.jsonl from MFACTORY_METRICS_JSONL=1), then
        # fall back to nnUNet's always-present validation/summary.json so
        # finished historical models stop showing "—".
        val_dice: float | None = None
        per_class: dict[str, float] | None = None
        for fold in sorted(e["available_folds"], reverse=True):
            jsonl_path = config_dir / f"fold_{fold}" / "metrics.jsonl"
            v = _last_mean_fg_dice(jsonl_path)
            if v is not None and val_dice is None:
                val_dice = v
            summary_mean, summary_per_class = _read_validation_summary(
                config_dir, fold, int_to_name
            )
            if val_dice is None and summary_mean is not None:
                val_dice = summary_mean
            if per_class is None and summary_per_class:
                per_class = summary_per_class
            if val_dice is not None and per_class is not None:
                break

        # Per-fold rollup. Walks every fold_N directory on disk (not
        # just those with checkpoint_best.pth), so in-flight folds —
        # e.g. fold 1 + 2 actively training while fold 0 sits done —
        # show up explicitly instead of being collapsed into the
        # discovery-based available_folds list. The outer-level status
        # and current_epoch come from `_summary_from_folds`, which
        # prefers the live-most fold so a finished fold 0 doesn't mask
        # epoch-19 fold-1/2 progress.
        total_epochs = _trainer_total_epochs(config_dir)
        fold_progress = _build_fold_progress(config_dir, total_epochs)
        rollup_status, current_epoch = _summary_from_folds(fold_progress)

        # Dragonfly cache lookup over the model's compatible QA cohort
        # cases. cache.get_inference returns None when Redis is down or
        # the URL is unset — see cache.py:18 ("opportunistic, not load
        # bearing").
        cached_count = 0
        cohort_size = 0
        if cohort_cases:
            model_id = e["model_id"]
            compatible = [
                str(c["case_id"]) for c in cohort_cases
                if model_id in (c.get("compatible_models") or [])
            ]
            cohort_size = len(compatible)
            if cohort_size:
                try:
                    plan_hash = result_cache.plan_hash_for_model(config_dir)
                    folds_tuple = tuple(e["available_folds"])
                    for cid in compatible:
                        if result_cache.get_inference(
                            plan_hash, model_id, folds_tuple, cid
                        ):
                            cached_count += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "cached-count lookup failed for %s: %s", e["model_id"], exc
                    )

        # Per the new fold walk, expose every fold dir on disk — not
        # just the ones with `checkpoint_best.pth`. The discovery pass
        # in cohort.py uses checkpoint_best.pth as its hit; we union the
        # in-flight folds so the UI doesn't report "1 fold" for a model
        # that's actually mid-flight on 3.
        all_folds = sorted({fp.fold for fp in fold_progress} | set(e["available_folds"]))

        out.append(ModelInfo(
            model_id=e["model_id"],
            dataset_name=e["dataset_name"],
            configuration=e["configuration"],
            trainer=e["trainer"],
            plans=e["plans"],
            region=_region_for(e["dataset_name"], datasets_root=datasets_root),
            available_folds=all_folds,
            model_dir=e["model_dir"],
            val_mean_fg_dice=val_dice,
            last_modified=last_mod,
            status=rollup_status,
            current_epoch=current_epoch,
            total_epochs=total_epochs,
            per_class_dice=per_class,
            cached_count=cached_count,
            cohort_size=cohort_size,
            approval_status=approval_by_model.get(e["model_id"], "pending"),
            folds=fold_progress,
        ))
    return out


# ── future-trainings pipeline (scheduler) ───────────────────────────────────
#
# The live calendar above renders folds physically training in results/. The
# planned queue below is the other half — what's coming next — so the home
# week-Gantt can show queued bars alongside live ones. Projection (when each
# queued fold starts/finishes) is computed at read time from the live folds'
# est_finish + a configured slot count; the store holds only intent.


class PlannedTrainingOut(BaseModel):
    id: str
    dataset_key: str
    dataset_name: str
    fold: int
    trainer: str
    plans: str
    priority: int
    status: str  # planned | submitted | cancelled
    est_duration_hours: float | None = None
    submitted_by: str = ""
    notes: str = ""
    created_at: str
    # Projected at read time (not stored).
    scheduled_start: str | None = None
    est_finish: str | None = None
    eta_seconds: float | None = None


class PlannedTrainingCreate(BaseModel):
    dataset_key: str
    dataset_name: str
    fold: int = 0
    trainer: str = "nnUNetTrainerMLflow"
    plans: str = "nnUNetResEncUNetLPlans"
    priority: int = 0
    est_duration_hours: float | None = None
    submitted_by: str = ""
    notes: str = ""


class PlannedTrainingUpdate(BaseModel):
    priority: int | None = None
    notes: str | None = None
    status: str | None = None  # planned | submitted | cancelled
    est_duration_hours: float | None = None


def _project_planned() -> list[PlannedTrainingOut]:
    """Reconcile the queue against live results/, then project ETAs.

    Reuses list_models() so the planned bars share the exact live-training
    finish estimates the calendar already draws.
    """
    store = _get_schedule()
    models = list_models()
    live_keys = {
        f"{m.dataset_name}::{f.fold}"
        for m in models
        for f in m.folds
        if f.status in ("training", "done")
    }
    store.reconcile(live_keys)

    now_ms = dt.datetime.now(dt.timezone.utc).timestamp() * 1000.0
    running_finish_ms: list[float] = []
    for m in models:
        for f in m.folds:
            if f.status == "training" and f.est_finish:
                try:
                    running_finish_ms.append(
                        dt.datetime.fromisoformat(f.est_finish).timestamp() * 1000.0
                    )
                except ValueError:
                    pass

    projected = project_schedule(
        store.list_all(status="planned"),
        running_finish_ms=running_finish_ms,
        slots=TRAINING_SLOTS,
        now_ms=now_ms,
    )
    return [PlannedTrainingOut(**p.__dict__) for p in projected]


@app.get("/api/planned-trainings", response_model=list[PlannedTrainingOut])
def list_planned_trainings():
    """Queued (not-yet-started) trainings with projected start/finish."""
    return _project_planned()


@app.post("/api/planned-trainings", response_model=PlannedTrainingOut, status_code=201)
def create_planned_training(req: PlannedTrainingCreate):
    p = _get_schedule().add(**req.model_dump())
    return PlannedTrainingOut(**p.__dict__)


@app.patch("/api/planned-trainings/{planned_id}", response_model=PlannedTrainingOut)
def patch_planned_training(planned_id: str, req: PlannedTrainingUpdate):
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if "status" in updates and updates["status"] not in PLANNED_STATUSES:
        raise HTTPException(status_code=422, detail=f"bad status: {updates['status']}")
    p = _get_schedule().update(planned_id, **updates)
    if p is None:
        raise HTTPException(status_code=404, detail="planned training not found")
    return PlannedTrainingOut(**p.__dict__)


@app.delete("/api/planned-trainings/{planned_id}", status_code=204)
def delete_planned_training(planned_id: str):
    if not _get_schedule().delete(planned_id):
        raise HTTPException(status_code=404, detail="planned training not found")
    return Response(status_code=204)


@app.get("/api/cohort", response_model=CohortResponse)
def get_cohort():
    manifest_path = COHORT_ROOT / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"no cohort manifest at {manifest_path}; "
                   "run `modelfactory qa cohort prepare`",
        )
    raw = json.loads(manifest_path.read_text())
    models = list_models()
    return CohortResponse(
        version=raw["version"],
        regions=raw["regions"],
        cases=[CaseInfo(**c) for c in raw["cases"]],
        trained_models=models,
    )


@app.get("/api/cases/{region}/{case_id}/image")
def case_image(region: str, case_id: str, channel: int = 0):
    case_dir = _resolve_case_dir(region, case_id)
    name = f"image_{channel:04d}.nii.gz"
    path = case_dir / name
    if not path.is_file():
        raise HTTPException(404, f"no channel {channel} for {region}/{case_id}")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=name,
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/api/cases/{region}/{case_id}/groundtruth")
def case_groundtruth(region: str, case_id: str, revision: str = "active"):
    """Stream the active (or a specific) GT NIfTI back to the viewer.

    `revision`:
      - "active" (default) — the currently-active revision, or
        `label_groundtruth.nii.gz` when the case has no corrections yet.
      - an integer string — that specific historical revision number.

    Cache-Control is `no-cache` because the active revision changes on
    activation; the prior `max-age=3600` would have hidden new
    corrections from the browser for an hour.
    """
    case_dir = _resolve_case_dir(region, case_id)
    path = _resolve_gt_path(region, case_id, case_dir, revision)
    if not path.is_file():
        raise HTTPException(404, "no groundtruth for this case")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=path.name,
        headers={"Cache-Control": "no-cache"},
    )


@app.get(
    "/api/cases/{region}/{case_id}/groundtruth/revisions",
    response_model=list[GroundTruthRevisionOut],
)
def list_gt_revisions(region: str, case_id: str):
    _resolve_case_dir(region, case_id)  # path-traversal guard
    revs = _get_gt_store().list_for_case(region, f"{region}/{case_id}")
    return [GroundTruthRevisionOut(**r.__dict__) for r in revs]


@app.post(
    "/api/cases/{region}/{case_id}/groundtruth/edits",
    response_model=GroundTruthRevisionOut,
    status_code=201,
)
async def post_gt_edit(
    region: str,
    case_id: str,
    labelmap: UploadFile = File(...),
    sidecar: UploadFile = File(...),
):
    """Persist a reviewer's GT edits as a new sidecar revision.

    The multipart body has two parts:
      - `labelmap.bin` (octet-stream): raw voxel bytes in Cornerstone's
        Fortran (x-y-z) traversal order, dtype matching the original GT
        (typically uint8).
      - `sidecar.json`: geometry + dtype + optimistic-concurrency token.

    The geometry must match the original NIfTI (sub-mm tolerance). The
    saved file reuses ``orig.affine`` / ``orig.header`` so the corrected
    NIfTI is voxel-aligned with the prediction NIfTI for downstream
    metric computation.
    """
    import nibabel as nib  # type: ignore[import-not-found]
    import numpy as np

    case_dir = _resolve_case_dir(region, case_id)

    raw_sidecar = await sidecar.read()
    try:
        meta = json.loads(raw_sidecar)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"sidecar.json is not valid JSON: {exc}") from exc

    if meta.get("schema_version") != 1:
        raise HTTPException(400, "unsupported sidecar schema_version")

    raw_labelmap = await labelmap.read()
    if len(raw_labelmap) > MAX_GT_BYTES:
        raise HTTPException(413, f"labelmap exceeds {MAX_GT_BYTES} bytes")

    gt_orig_path = case_dir / "label_groundtruth.nii.gz"
    if not gt_orig_path.is_file():
        raise HTTPException(409, "case has no original groundtruth to anchor edits")

    orig = nib.load(str(gt_orig_path))
    expected_shape = tuple(int(x) for x in orig.shape)
    expected_dtype = orig.header.get_data_dtype()
    expected_zooms = tuple(float(z) for z in orig.header.get_zooms()[:3])

    sidecar_shape = tuple(int(x) for x in meta.get("dimensions") or ())
    if sidecar_shape != expected_shape:
        raise HTTPException(
            409,
            f"sidecar shape {sidecar_shape} != original GT shape {expected_shape}",
        )
    sidecar_zooms = tuple(float(z) for z in meta.get("spacing") or ())
    if any(abs(a - b) > 1e-4 for a, b in zip(sidecar_zooms, expected_zooms)):
        raise HTTPException(
            409,
            f"sidecar spacing {sidecar_zooms} != original GT spacing {expected_zooms}",
        )

    # Optimistic concurrency: reject if the active revision has advanced
    # past whatever the operator started from. Operator must reload and
    # reapply.
    source_rev = meta.get("source_revision", "active")
    cur_active = _get_gt_store().get_active(region, f"{region}/{case_id}")
    if cur_active is not None:
        cur_id = cur_active.id
        # source can be "active" (meaning "what was active when I loaded"
        # — which we don't know precisely, so accept), an int revision
        # number, or "original".
        if isinstance(source_rev, int) and cur_active.id != source_rev:
            raise HTTPException(
                409,
                f"newer GT revision {cur_active.revision} exists "
                f"(id={cur_id}); reload and reapply your edits",
            )

    # Cast the uploaded buffer to the original dtype. uint16 promotions
    # are silently truncated only if no value exceeds 255 — otherwise
    # reject so the operator isn't surprised by a quietly-corrupted GT.
    nbytes_per_voxel = np.dtype(expected_dtype).itemsize
    expected_bytes = nbytes_per_voxel * int(np.prod(expected_shape))
    if len(raw_labelmap) != expected_bytes:
        raise HTTPException(
            400,
            f"labelmap byte count {len(raw_labelmap)} != "
            f"expected {expected_bytes} for {expected_shape} {expected_dtype}",
        )
    buf = np.frombuffer(raw_labelmap, dtype=expected_dtype).reshape(
        expected_shape, order="F",
    )

    rev = _get_gt_store().next_revision_number(region, f"{region}/{case_id}")
    out_name = f"label_corrected_v{rev}.nii.gz"
    out_path = case_dir / out_name
    new_img = nib.Nifti1Image(buf.copy(), affine=orig.affine, header=orig.header)
    new_img.set_data_dtype(buf.dtype)
    nib.save(new_img, str(out_path))

    saved = _get_gt_store().record_active(
        region=region,
        case_id=f"{region}/{case_id}",
        revision=rev,
        path=f"{region}/{case_id}/{out_name}",
        base_prediction_id=meta.get("base_prediction_id"),
        reviewer=str(meta.get("reviewer", "")),
        notes=str(meta.get("notes", "")),
    )

    # Kick off recompute so subsequent fetches see fresh metrics. Fire-
    # and-forget; the frontend can poll /recompute-status for progress.
    asyncio.create_task(_recompute_metrics_for_case(region, case_id))

    return GroundTruthRevisionOut(**saved.__dict__)


@app.post(
    "/api/cases/{region}/{case_id}/groundtruth/revisions/{rev_id}/activate",
    response_model=GroundTruthRevisionOut,
)
async def activate_gt_revision(region: str, case_id: str, rev_id: int):
    _resolve_case_dir(region, case_id)  # path-traversal guard
    try:
        rev = _get_gt_store().activate(rev_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    if rev.region != region or rev.case_id != f"{region}/{case_id}":
        raise HTTPException(404, "revision is not for this case")
    asyncio.create_task(_recompute_metrics_for_case(region, case_id))
    return GroundTruthRevisionOut(**rev.__dict__)


@app.get(
    "/api/cases/{region}/{case_id}/groundtruth/recompute-status",
    response_model=RecomputeStatusOut,
)
def get_recompute_status(region: str, case_id: str):
    _resolve_case_dir(region, case_id)  # path-traversal guard
    key = f"{region}/{case_id}"
    state = _recompute_status.get(key)
    if state is None:
        return RecomputeStatusOut(pending=0, completed=0, total=0)
    return RecomputeStatusOut(**state)


@app.post(
    "/api/cases/{region}/{case_id}/groundtruth/seed-from-prediction",
    response_model=CaseInfo,
    status_code=201,
)
async def seed_gt_from_prediction(region: str, case_id: str, req: SeedGtRequest):
    """Seed an uploaded case's ground truth from a model prediction.

    The GT-edit pipeline anchors corrections on `label_groundtruth.nii.gz`
    (see `post_gt_edit`), which uploaded cases lack. Copying the prediction's
    seg into it lets the reviewer open the Cornerstone editor on the model's
    output, fix it, and save it as a training label — closing the
    "QC → corrected GT → retrain" loop for ad-hoc data.

    Restricted to uploaded cases so a donated cohort case's real GT is never
    clobbered. Returns the updated CaseInfo (now with `groundtruth_path` set)
    so the UI can reveal the GT overlay + correct-GT controls.
    """
    case_dir = _resolve_case_dir(region, case_id)
    full_case_id = f"{region}/{case_id}"

    if not _SAFE_SEGMENT.match(req.prediction_id):
        raise HTTPException(400, "bad prediction_id")
    pred_dir = (PREDICTIONS_ROOT / req.prediction_id).resolve()
    if not str(pred_dir).startswith(str(PREDICTIONS_ROOT.resolve())):
        raise HTTPException(400, "bad prediction_id")
    seg_path = pred_dir / "seg.nii.gz"
    status_path = pred_dir / "status.json"
    if not seg_path.is_file() or not status_path.is_file():
        raise HTTPException(404, f"no prediction {req.prediction_id}")

    # The prediction must belong to this case — otherwise its geometry won't
    # match the case image and downstream GT-edit shape checks would fail.
    try:
        status = json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(400, f"unreadable prediction status: {exc}") from exc
    if status.get("case_id") != full_case_id:
        raise HTTPException(
            409,
            f"prediction {req.prediction_id} is for {status.get('case_id')}, "
            f"not {full_case_id}",
        )

    gt_path = case_dir / "label_groundtruth.nii.gz"
    if gt_path.is_file() and not _is_uploaded_case(case_dir):
        raise HTTPException(
            409,
            "case already has ground truth; seed-from-prediction is only for "
            "uploaded cases",
        )

    gt_rel = f"{full_case_id}/label_groundtruth.nii.gz"
    async with _donate_lock:
        await asyncio.to_thread(shutil.copyfile, seg_path, gt_path)
        await asyncio.to_thread(
            _patch_manifest_case, full_case_id, groundtruth_path=gt_rel,
        )

    info = _case_info_from_manifest(full_case_id)
    if info is None:
        info = CaseInfo(
            case_id=full_case_id, region=region, source_dataset="uploaded",
            source_case_stem="", image_paths=[], groundtruth_path=gt_rel,
            compatible_models=[], uploaded=True,
        )
    return info


@app.get("/api/queue", response_model=QueueResponse)
def get_queue():
    """Snapshot of the predict queue for the header widget.

    Cheap (in-memory). Polled by the frontend every ~5 s; an SSE version
    could replace the poll but the bandwidth + cardinality here is
    negligible — typically 0-2 entries — so polling is fine.
    """
    return QueueResponse(
        depth=predict_queue.depth(),
        in_flight=[
            QueueEntryOut(
                prediction_id=e.prediction_id,
                model_id=e.model_id,
                case_id=e.case_id,
                reviewer=e.reviewer,
                state=e.state,
                submitted_at=e.submitted_at,
                started_at=e.started_at,
                position_in_queue=idx,
                eta_s=e.eta_s,
            )
            for idx, e in enumerate(predict_queue.in_flight())
        ],
    )


@app.post(
    "/api/cohort/cases",
    response_model=DonateCaseResponse,
    status_code=201,
)
async def donate_case(req: DonateCaseRequest):
    """On-demand cohort case for a model that has none yet.

    The CLI verb `modelfactory qa cohort prepare` walks every trained
    dataset and donates one case from each. New datasets land in the
    catalogue immediately on training completion but their cohort case
    isn't materialized until the CLI is re-run — which means clicking
    them in the UI shows a dead-end. This endpoint runs the same
    `build_cohort_for_dataset` pipeline against one dataset on demand
    so the UI can offer a Donate button.

    Either `model_id` or `dataset_name` must be set. When `model_id` is
    set, the dataset is derived from its `Dataset###_*::trainer__…`
    prefix.
    """
    from modelfactory.qa.cohort import (
        DatasetNotFoundError,
        build_cohort_for_dataset,
        existing_cohort_cases,
    )

    dataset_name = req.dataset_name
    if not dataset_name and req.model_id:
        if "::" not in req.model_id:
            raise HTTPException(400, "model_id is not in Dataset###_*::… form")
        dataset_name = req.model_id.split("::", 1)[0]
    if not dataset_name:
        raise HTTPException(400, "must supply dataset_name or model_id")
    if not _SAFE_SEGMENT.match(dataset_name):
        raise HTTPException(400, "bad dataset_name")

    datasets_root = Path(
        os.environ.get("QA_DATASETS_ROOT", FACTORY_ROOT / "datasets")
    )

    async with _donate_lock:
        try:
            new = await asyncio.to_thread(
                build_cohort_for_dataset,
                dataset_name,
                datasets_root=datasets_root,
                results_root=RESULTS_ROOT,
                output_root=COHORT_ROOT,
                region=req.region,
                n_pick=max(1, min(req.n_pick, 10)),
            )
        except DatasetNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    # Warm the preprocessed-npz cache in the background so the reviewer's
    # first inference on this case isn't the slow single-threaded cold
    # resample (see _prestage_preprocess_async).
    _prestage_preprocess_async(dataset_name)

    # `build_cohort_for_dataset` is additive and returns ONLY the cases it
    # materialized this call. An empty list means either the dataset is
    # already at the requested count, or it has no usable cases on disk —
    # disambiguate by checking what's already in the manifest.
    if not new:
        existing = existing_cohort_cases(COHORT_ROOT, dataset_name)
        if not existing:
            raise HTTPException(404, f"no usable cases on disk for {dataset_name}")
        return DonateCaseResponse(
            region=existing[0].region,
            dataset_name=dataset_name,
            new_cases=[CaseInfo(**r.__dict__) for r in existing],
            already_existed=True,
        )

    return DonateCaseResponse(
        region=new[0].region,
        dataset_name=dataset_name,
        new_cases=[CaseInfo(**r.__dict__) for r in new],
        already_existed=False,
    )


async def _stage_upload_files(
    files: list[UploadFile], staged_dir: Path,
) -> str:
    """Stream uploaded parts to `staged_dir`, enforcing MAX_UPLOAD_BYTES.

    Returns the first part's (sanitized) filename for provenance. Filenames
    are reduced to their basename so a malicious ``../`` can't escape.
    """
    total = 0
    first_name = ""
    for uf in files:
        name = Path(uf.filename or "upload.bin").name or "upload.bin"
        if not first_name:
            first_name = name
        dst = staged_dir / name
        with dst.open("wb") as out:
            while True:
                chunk = await uf.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413, f"upload exceeds {MAX_UPLOAD_BYTES} bytes",
                    )
                out.write(chunk)
    if total == 0:
        raise HTTPException(400, "empty upload")
    return first_name


@app.post("/api/cohort/uploads", response_model=CaseInfo, status_code=201)
async def upload_case(
    model_id: str = Form(...),
    reviewer: str = Form(""),
    files: list[UploadFile] = File(...),
):
    """Ingest a reviewer-uploaded DICOM series or NIfTI volume as an ad-hoc
    QA case for `model_id`.

    Converts server-side (SimpleITK for DICOM, nibabel for NIfTI — both
    already in the image) into the cohort case layout, so `/api/predict` +
    the viewer work on it unchanged. The case has no ground truth until the
    reviewer seeds one from the model's prediction
    (`/groundtruth/seed-from-prediction`). Returns the new case so the UI can
    auto-select it and run inference.
    """
    from modelfactory.qa.cohort import (
        _discover_trained_models,
        _merge_into_manifest,
    )
    from modelfactory.qa.upload import UploadError, ingest_upload

    model_dir = _resolve_model_dir(model_id)
    region = _model_region(model_dir.parent.name)
    if region is None:
        raise HTTPException(
            409,
            f"cannot determine region for {model_id} — set tags.region in "
            f"its dataset.json",
        )
    expected_channels = _model_channel_count(model_dir)
    datasets_root = _datasets_root()

    with tempfile.TemporaryDirectory(prefix="qa-stage-") as tmp:
        staged = Path(tmp)
        first_name = await _stage_upload_files(files, staged)

        try:
            record = await asyncio.to_thread(
                ingest_upload,
                staged,
                model_id=model_id,
                region=region,
                expected_channels=expected_channels,
                cohort_root=COHORT_ROOT,
                uploaded_by=reviewer,
                original_filename=first_name,
            )
        except UploadError as exc:
            raise HTTPException(400, str(exc)) from exc

        def _persist() -> None:
            trained_models = _discover_trained_models(
                RESULTS_ROOT, datasets_root=datasets_root,
            )
            _merge_into_manifest(COHORT_ROOT, [record], trained_models)

        async with _donate_lock:
            await asyncio.to_thread(_persist)

    # Warm the preprocessed-npz cache for this model in the background so the
    # first inference on the upload skips the slow in-process cold resample.
    _prestage_preprocess_async(model_dir.parent.name)

    return CaseInfo(**record.__dict__)


@app.post("/api/predict", response_model=PredictAcceptedResponse, status_code=202)
async def predict(req: PredictRequest):
    """Kick off inference asynchronously; return immediately with a prediction_id.

    The full inference (preprocess + sliding-window TTA + export + metrics)
    typically takes 30-90 s for ResEncXL 3d_fullres on whole-H100 GPU 0.
    Holding the HTTP connection open that long invites proxy resets
    (Cloudflare 100 s, HAProxy `timeout server` defaults to 60 s, etc.), so
    we 202 immediately and let the client poll `/api/predictions/{id}/status`.
    """
    # Input validation — 4xx HTTPException raises pass through unchanged.
    case_dir = _resolve_case_dir(*req.case_id.split("/", 1))
    model_dir = _resolve_model_dir(req.model_id)
    folds = _resolve_folds(req.use_folds, model_dir)
    if not folds:
        raise HTTPException(400, f"no folds available for {req.model_id}")

    region = _model_region(model_dir.parent.name)
    case_region = req.case_id.split("/", 1)[0]
    if region is not None and region != case_region:
        raise HTTPException(
            409,
            f"region mismatch: model is {region}, case is {case_region}",
        )

    _ensure_predictions_root()

    # Redis-backed result cache: if we already produced a valid seg for
    # this (plan_hash, model, folds, case) tuple, return that prediction_id
    # immediately and skip the inference run. plan_hash is keyed off
    # plans.json + dataset_fingerprint.json bytes, so retraining the model
    # automatically invalidates the cache.
    plan_hash = result_cache.plan_hash_for_model(model_dir)
    cached_id = _cached_prediction_id(plan_hash, req.model_id, folds, req.case_id)
    if cached_id is not None:
        logger.info(
            "cache hit for model=%s case=%s folds=%s → %s",
            req.model_id, req.case_id, folds, cached_id,
        )
        return PredictAcceptedResponse(
            prediction_id=cached_id,
            status="done",
            status_url=f"/api/predictions/{cached_id}/status",
            from_cache=True,
        )

    prediction_id = uuid.uuid4().hex[:12]
    out_dir = PREDICTIONS_ROOT / prediction_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Register in the shared predict queue BEFORE writing status.json so
    # the first poll already sees position_in_queue + queue_depth.
    entry = await predict_queue.submit(
        prediction_id=prediction_id,
        model_id=req.model_id,
        case_id=req.case_id,
        reviewer=getattr(req, "reviewer", "") or "",
    )
    position = predict_queue.position(prediction_id) or 0
    depth = predict_queue.depth()

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    _write_status(out_dir, {
        "prediction_id": prediction_id,
        "status": "queued",
        "model_id": req.model_id,
        "case_id": req.case_id,
        "folds": list(folds),
        "started_at": now,
        "updated_at": now,
        "position_in_queue": position,
        "queue_depth": depth,
    })

    asyncio.create_task(_run_predict_background(
        prediction_id=prediction_id,
        out_dir=out_dir,
        req=req,
        model_dir=model_dir,
        folds=folds,
        case_dir=case_dir,
        plan_hash=plan_hash,
    ))

    return PredictAcceptedResponse(
        prediction_id=prediction_id,
        status="queued",
        status_url=f"/api/predictions/{prediction_id}/status",
        position_in_queue=position,
        queue_depth=depth,
        eta_s=entry.eta_s,
    )


@app.get("/api/predictions/{prediction_id}/status", response_model=PredictionStatus)
def get_prediction_status(prediction_id: str):
    """Return the current status JSON for a prediction.

    Polled by the frontend every ~1.5 s after POST /api/predict. The status
    transitions are queued → running → (done | error). The done payload
    mirrors the legacy synchronous PredictResponse fields so the UI
    consumes a single shape.
    """
    if not _SAFE_SEGMENT.match(prediction_id):
        raise HTTPException(400, "bad prediction id")
    path = PREDICTIONS_ROOT / prediction_id / "status.json"
    if not path.is_file():
        raise HTTPException(404, f"no prediction {prediction_id}")
    raw = json.loads(path.read_text())
    # Always overlay the live queue position — the on-disk status.json
    # might be a few seconds stale but the queue is authoritative.
    live_pos = predict_queue.position(prediction_id)
    if live_pos is not None:
        raw["position_in_queue"] = live_pos
    raw.setdefault("queue_depth", predict_queue.depth())
    return PredictionStatus(**raw)


@app.get("/api/predictions/{prediction_id}/events")
async def prediction_events(prediction_id: str):
    """Server-Sent Events stream of status updates for one prediction.

    Frontend opens an EventSource on this endpoint instead of polling
    `/status` every 1.5 s. Each `data:` frame is the full status JSON.
    Stream closes once the prediction reaches a terminal status (`done`
    or `error`); the client can drop the EventSource and stop tracking.

    Implementation: re-reads status.json every 1 s (mtime-aware) and
    emits a frame when the contents change. Cheap enough for the
    expected concurrency (1-3 reviewers) and avoids needing a real
    pub/sub mechanism inside the worker process.
    """
    from fastapi.responses import StreamingResponse

    if not _SAFE_SEGMENT.match(prediction_id):
        raise HTTPException(400, "bad prediction id")
    status_path = PREDICTIONS_ROOT / prediction_id / "status.json"

    async def _gen():
        last_payload: str | None = None
        # Initial flush: send the current status (or a 404-style frame).
        # 60-minute upper bound — well past any single inference.
        deadline = time.monotonic() + 60 * 60
        terminal = {"done", "error"}
        while time.monotonic() < deadline:
            if not status_path.is_file():
                yield "event: missing\ndata: {}\n\n"
                await asyncio.sleep(2.0)
                continue
            try:
                raw = json.loads(status_path.read_text())
            except (OSError, json.JSONDecodeError):
                await asyncio.sleep(1.0)
                continue
            live_pos = predict_queue.position(prediction_id)
            if live_pos is not None:
                raw["position_in_queue"] = live_pos
            raw.setdefault("queue_depth", predict_queue.depth())
            payload = json.dumps(raw)
            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload
            status_value = raw.get("status")
            if status_value in terminal and raw.get("meshes_status") in (None, "ready", "failed"):
                # Terminal: client can close the connection cleanly.
                yield "event: close\ndata: {}\n\n"
                return
            await asyncio.sleep(1.0)

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",  # disable buffering at proxies if present
        },
    )


async def _status_heartbeat(out_dir: Path, stop: asyncio.Event, interval: float = 30.0) -> None:
    """Tick status.json.updated_at every `interval` seconds.

    CT inference (ResEncXL on multi-segment SegRap models) can run ~5 min
    inline; without status updates during that window, the Caddy / pfSense
    a reverse proxy in front of the viewer may drop the long-poll as idle
    and the browser sees `ERR_NETWORK_CHANGED` / `Failed to fetch`. A 30 s
    heartbeat lands well inside the 60 s proxy idle budget and inside the
    frontend's 12-retry transient-error tolerance, so the cycle just looks
    like normal polling all the way through.
    """
    while True:
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
            return
        except asyncio.TimeoutError:
            try:
                _update_status(out_dir)  # rewrites updated_at only
            except Exception as e:  # noqa: BLE001 — disk hiccups are non-fatal
                logger.warning("status heartbeat write failed: %s", e)


def _mean_fg_dice(metrics_dump: list[dict] | None) -> float | None:
    """Mean foreground dice over a metrics dump (skipping None/NaN labels).

    Matches how MetricsBlock derives its hero number on the frontend, so the
    CV per-fold mean dice and the verdict's recorded mean_dice agree.
    """
    if not metrics_dump:
        return None
    vals = [m["dice"] for m in metrics_dump if m.get("dice") is not None]
    return float(mean(vals)) if vals else None


async def _execute_one_prediction(
    *,
    prediction_id: str,
    out_dir: Path,
    model_id: str,
    case_id: str,
    model_dir: Path,
    folds: tuple[int, ...],
    case_dir: Path,
    plan_hash: str,
    compute_hd95: bool = True,
    do_meshes: bool = True,
) -> dict:
    """Run ONE (folds-tuple) prediction end-to-end for `case_id`.

    Drives this prediction's own status.json + result-cache + (optional) mesh
    lifecycle and returns a summary dict:
        {prediction_id, status, metrics, mean_fg_dice, elapsed_s, error}

    Lock discipline (R3): the GPU `predict_queue.gpu_lock` is acquired HERE,
    around the inference phase only, one fold at a time. Callers — the
    `_run_predict_background` wrapper and the cross-validation orchestrator —
    MUST NOT hold `gpu_lock` when calling this (it is a non-reentrant
    asyncio.Lock); the orchestrator holds only its own per-(model,case)
    mutex. The queue `submit` is the caller's responsibility (done before
    spawning); `mark_started` + `remove` happen here.

    `compute_hd95` is persisted into status.json at seg_ready so a later GT
    edit re-scores this prediction with the same HD95 policy. `do_meshes`
    gates the best-effort surface-mesh precompute (CV fold sub-runs other
    than the OOF fold skip it to save time — the overlay still renders via
    in-browser marching cubes).
    """
    seg_path = out_dir / "seg.nii.gz"
    # Must mirror `preprocess_cohort_for_model` EXACTLY, which writes to
    # `out_root / case["case_id"]` using the FULL region-prefixed case id
    # (e.g. "pelvis_ct/d154_case_001"). Stripping the region here pointed
    # the cache lookup at `.../<config>/d154_case_001/case.npz` while the
    # pre-stage wrote `.../<config>/pelvis_ct/d154_case_001/case.npz`, so the
    # fast path NEVER hit and every pre-staged/donated case silently fell back
    # to the ~10-min in-process cold preprocess (looked like a hang).
    cache_dir = (
        COHORT_ROOT / "preprocessed"
        / model_dir.parent.name / model_dir.name
        / case_id
    )
    npz: Path | None = cache_dir / "case.npz"
    pkl: Path | None = cache_dir / "case.pkl"
    # Freshness guard mirroring the pre-stage writer's own reuse check: only
    # trust the cache when its `plan_hash.txt` matches the model's current
    # plan hash. A stale npz (model re-preprocessed since) falls back to the
    # in-process path rather than being silently reused with wrong spacing.
    hash_file = cache_dir / "plan_hash.txt"
    fresh = (
        hash_file.is_file()
        and hash_file.read_text().strip() == plan_hash
    )
    if not fresh:
        npz = pkl = None
    raw_images = [
        case_dir / f"image_{i:04d}.nii.gz"
        for i in range(_count_channels(case_dir))
    ]

    # Phase 1: inference — fail here means status="error". Phase 2: metrics
    # — always reach status="done" (metrics may be null + metrics_error
    # set). The split lets the frontend render the Cornerstone overlay
    # the moment seg.nii.gz lands, instead of waiting for hd95 to finish
    # for every label (~1 min/label brute-force).
    label_map = _load_label_map(model_dir)
    # Heartbeat covers BOTH the inference and the metrics phases — the
    # HD95 brute-force on a 15-label model can also push past the proxy
    # idle budget on its own. The heartbeat is cancelled in the outer
    # `finally` below so it always tears down, even on early returns.
    hb_stop = asyncio.Event()
    hb_task = asyncio.create_task(_status_heartbeat(out_dir, hb_stop))
    elapsed_for_history: float | None = None
    metrics_dump: list[dict] | None = None
    metrics_error: str | None = None
    result: dict | None = None
    try:
        try:
            async with predict_queue.gpu_lock:
                await predict_queue.mark_started(prediction_id)
                _update_status(
                    out_dir,
                    status="running",
                    position_in_queue=0,
                    queue_depth=predict_queue.depth(),
                )
                result = await asyncio.to_thread(
                    _run_prediction,
                    model_dir=model_dir,
                    folds=folds,
                    raw_images=raw_images,
                    seg_path=seg_path,
                    npz=npz,
                    pkl=pkl,
                )
                elapsed_for_history = float(result.get("elapsed_s") or 0.0)
        except Exception as exc:  # noqa: BLE001 — capture anything torch/nnUNet can throw
            logger.exception(
                "background predict failed for model=%s case=%s folds=%s",
                model_id, case_id, folds,
            )
            _update_status(
                out_dir,
                status="error",
                error_type=type(exc).__name__,
                error_message=str(exc),
                position_in_queue=None,
            )
            return {
                "prediction_id": prediction_id,
                "status": "error",
                "metrics": None,
                "mean_fg_dice": None,
                "elapsed_s": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

        # Inference is done; flip to seg_ready so the UI can render the
        # overlay while metrics catch up. Postprocessing metadata is built
        # here so the panel can render it the moment the seg is up.
        postproc = _postprocessing_info_for(
            model_dir=model_dir,
            configuration=model_dir.name.split("__")[-1] if "__" in model_dir.name else "3d_fullres",
            pkl_path=pkl,
        )
        _update_status(
            out_dir,
            status="seg_ready",
            elapsed_s=result["elapsed_s"],
            used_preprocessed_cache=result["used_preprocessed_cache"],
            label_map=label_map,
            seg_url=f"/api/predictions/{prediction_id}/seg",
            metrics_url=f"/api/predictions/{prediction_id}/metrics",
            postprocessing=postproc.model_dump() if postproc else None,
            # Persist the HD95 policy so a later GT-edit recompute respects it.
            compute_hd95=compute_hd95,
        )

        # Cache the seg now (not in the `done` branch below) so even if
        # metrics fail, a re-click on the same (model, case, folds) tuple
        # still returns this prediction_id instantly.
        result_cache.set_inference(
            plan_hash, model_id, folds, case_id, prediction_id,
        )

        # Metrics vs groundtruth, if the cohort case has one. Use the active
        # revision so corrections produce updated metrics on the next run.
        # Compute in a worker thread so the event loop stays responsive for
        # status polls. Failure here is non-fatal — the seg is fine, just
        # no per-label numbers.
        region, inner = case_id.split("/", 1)
        gt = _resolve_active_gt(region, inner, case_dir)
        active_rev = _get_gt_store().get_active(region, case_id)
        rev_num = active_rev.revision if active_rev else None
        if gt.is_file():
            try:
                metrics_payload = await asyncio.to_thread(
                    _compute_metrics, seg_path, gt, label_map, compute_hd95,
                )
                metrics_dump = [m.model_dump() for m in metrics_payload]
                (out_dir / "metrics.json").write_text(json.dumps(metrics_dump, indent=2))
                if rev_num:
                    (out_dir / f"metrics_v{rev_num}.json").write_text(
                        json.dumps(metrics_dump, indent=2)
                    )
                # Cache metrics keyed on (plan, model, folds, case, gt_rev).
                # A GT correction bumps `rev_num` and produces a fresh
                # cache key, so old metrics are not served against a new GT.
                result_cache.set_metrics(
                    plan_hash, model_id, folds, case_id,
                    rev_num, metrics_dump,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "metrics computation failed for model=%s case=%s folds=%s",
                    model_id, case_id, folds,
                )
                metrics_error = f"{type(exc).__name__}: {exc}"

        _update_status(
            out_dir,
            status="done",
            metrics=metrics_dump,
            metrics_error=metrics_error,
            active_gt_revision=rev_num,
        )

        # Phase 3: backend mesh precompute. Surface-mesh extraction for
        # every non-background label, written as one .vtp per label under
        # out_dir/mesh/. The 3D fullscreen viewer fetches these directly
        # instead of running marching cubes in the browser.
        #
        # IMPORTANT: we do NOT touch `status` here — it stays at `done`.
        # The frontend's polling loop (`runPredictUntilSeg`, `pollMetrics`)
        # only treats `seg_ready` and `done` as terminal, so any value
        # beyond that hangs the poll loop until SEG_TIMEOUT_MS. Mesh
        # progress lives in `meshes_status` + `meshes_elapsed_s` +
        # `meshes_by_label` so the 3D canvas can read it independently.
        if do_meshes:
            _update_status(out_dir, meshes_status="pending")
            try:
                mesh_out = out_dir / "mesh"
                mesh_result = await asyncio.to_thread(
                    _precompute_meshes_sync, seg_path, mesh_out, label_map,
                )
                _update_status(
                    out_dir,
                    meshes_status="ready",
                    meshes_elapsed_s=mesh_result["elapsed_s"],
                    meshes_by_label=mesh_result["by_label"],
                )
                result_cache.set_meshes(
                    plan_hash, model_id, folds, case_id, prediction_id,
                )
            except Exception as exc:  # noqa: BLE001 — mesh precompute is best-effort
                logger.exception(
                    "mesh precompute failed for prediction=%s", prediction_id,
                )
                _update_status(
                    out_dir,
                    meshes_status="failed",
                    meshes_error=f"{type(exc).__name__}: {exc}",
                )
    finally:
        hb_stop.set()
        try:
            await hb_task
        except Exception:  # noqa: BLE001
            pass
        # Remove from the queue regardless of success/failure so the next
        # submission's position math is correct. Feeding `elapsed_s` into
        # the history keeps ETA accurate for the same model_id.
        await predict_queue.remove(prediction_id, elapsed_s=elapsed_for_history)
        _update_status(out_dir, position_in_queue=None)

    return {
        "prediction_id": prediction_id,
        "status": "done",
        "metrics": metrics_dump,
        "mean_fg_dice": _mean_fg_dice(metrics_dump),
        "elapsed_s": (result or {}).get("elapsed_s"),
        "error": metrics_error,
    }


async def _run_predict_background(
    *,
    prediction_id: str,
    out_dir: Path,
    req: PredictRequest,
    model_dir: Path,
    folds: tuple[int, ...],
    case_dir: Path,
    plan_hash: str,
) -> None:
    """Long-running inference task for POST /api/predict.

    Thin wrapper around `_execute_one_prediction` (the GPU-bound core, shared
    with the cross-validation orchestrator). Behaviour of /api/predict is
    unchanged: HD95 on, meshes on.
    """
    await _execute_one_prediction(
        prediction_id=prediction_id,
        out_dir=out_dir,
        model_id=req.model_id,
        case_id=req.case_id,
        model_dir=model_dir,
        folds=folds,
        case_dir=case_dir,
        plan_hash=plan_hash,
        compute_hd95=True,
        do_meshes=True,
    )


@app.get("/api/predictions/{prediction_id}/seg")
def get_seg(prediction_id: str):
    path = PREDICTIONS_ROOT / prediction_id / "seg.nii.gz"
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="application/octet-stream",
                        filename="seg.nii.gz")


@app.get("/api/predictions/{prediction_id}/mesh/{seg_idx}")
def get_mesh(prediction_id: str, seg_idx: int):
    """Serve a pre-computed per-label surface mesh as VTK XML PolyData.

    Returns 404 when the mesh isn't on disk (precompute hasn't finished,
    failed, or the label has no foreground voxels). The frontend handles
    404 by falling back to in-browser marching cubes for that one label.
    """
    if not _SAFE_SEGMENT.match(prediction_id):
        raise HTTPException(400, "bad prediction id")
    if seg_idx < 0 or seg_idx > 1024:
        raise HTTPException(400, "bad segment index")
    path = PREDICTIONS_ROOT / prediction_id / "mesh" / f"{seg_idx}.vtp"
    if not path.is_file():
        raise HTTPException(404)
    # URL is content-addressable by (prediction_id, seg_idx); a successful
    # mesh body never mutates once written. Long-lived immutable cache.
    return FileResponse(
        path,
        media_type="application/xml",
        filename=f"mesh-{seg_idx}.vtp",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


def _precompute_meshes_sync(
    seg_path: Path, out_dir: Path, label_map: dict[str, int]
) -> dict:
    """Thread-pool entry — wraps the meshes module for `asyncio.to_thread`."""
    from modelfactory.inference.meshes import precompute_meshes

    result = precompute_meshes(seg_path, out_dir, label_map)
    return {
        "elapsed_s": result.elapsed_s,
        "by_label": {str(k): str(v.name) for k, v in result.by_label.items()},
    }


@app.get("/api/predictions/{prediction_id}/metrics")
def get_metrics(prediction_id: str):
    path = PREDICTIONS_ROOT / prediction_id / "metrics.json"
    if not path.is_file():
        raise HTTPException(404)
    return JSONResponse(json.loads(path.read_text()))


@app.post("/api/verdicts", response_model=VerdictOut)
def post_verdict(req: VerdictRequest):
    if req.verdict not in VERDICT_VALUES:
        raise HTTPException(
            400, f"verdict must be one of {VERDICT_VALUES}; got {req.verdict!r}"
        )
    # A reason is optional, but if supplied on a reject it must be a known
    # taxonomy key so the per-reason rollup stays clean.
    if req.verdict == "reject" and req.reject_reason and req.reject_reason not in REJECT_REASONS:
        raise HTTPException(
            400,
            f"reject_reason must be one of {REJECT_REASONS}; got {req.reject_reason!r}",
        )
    v = _get_verdicts().record(
        prediction_id=req.prediction_id,
        model_id=req.model_id,
        case_id=req.case_id,
        verdict=req.verdict,
        notes=req.notes,
        reviewer=req.reviewer,
        fold_choice=req.fold_choice,
        mean_dice=req.mean_dice,
        reject_reason=req.reject_reason,
    )
    next_case_id = _next_case_for_reviewer(
        model_id=req.model_id,
        reviewer=req.reviewer,
        current_case_id=req.case_id,
    )
    return VerdictOut(**v.__dict__, next_case_id=next_case_id)


@app.get("/api/verdicts", response_model=list[VerdictOut])
def list_verdicts(model_id: str | None = None, case_id: str | None = None, limit: int = 100):
    store = _get_verdicts()
    if model_id and case_id:
        rows = store.list_for_case(model_id, case_id)
    elif model_id:
        rows = store.list_for_model(model_id, limit=limit)
    else:
        # No filter — return the most recent verdicts across all models.
        rows = []
        for s in store.summary():
            rows.extend(store.list_for_model(s.model_id, limit=10))
        rows.sort(key=lambda r: r.created_at, reverse=True)
        rows = rows[:limit]
    return [VerdictOut(**r.__dict__) for r in rows]


@app.get("/api/verdicts/summary", response_model=list[VerdictSummaryOut])
def verdicts_summary():
    return [VerdictSummaryOut(**s.__dict__) for s in _get_verdicts().summary()]


# ── model card themes ────────────────────────────────────────────────────


@app.get("/api/model-themes", response_model=dict[str, ModelThemeOut])
def list_model_themes():
    """Return all per-model card-color overrides keyed by model_id."""
    return {
        mid: ModelThemeOut(**t.__dict__)
        for mid, t in _get_themes().all().items()
    }


@app.post("/api/model-themes/{model_id:path}", response_model=ModelThemeOut)
def set_model_theme(model_id: str, body: ModelThemeIn):
    if body.color_key not in PALETTE:
        raise HTTPException(
            400, f"color_key must be one of {PALETTE}; got {body.color_key!r}"
        )
    saved = _get_themes().set(model_id, body.color_key, body.updated_by)
    return ModelThemeOut(**saved.__dict__)


@app.delete("/api/model-themes/{model_id:path}")
def delete_model_theme(model_id: str):
    _get_themes().delete(model_id)
    return {"ok": True}


# ── helpers ──────────────────────────────────────────────────────────────


_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")


def _resolve_case_dir(region: str, case_id: str) -> Path:
    if not _SAFE_SEGMENT.match(region) or not _SAFE_SEGMENT.match(case_id):
        raise HTTPException(400, "bad case id")
    path = (COHORT_ROOT / region / case_id).resolve()
    if not str(path).startswith(str(COHORT_ROOT.resolve())):
        raise HTTPException(400, "bad case id")
    if not path.is_dir():
        raise HTTPException(404, f"no case {region}/{case_id}")
    return path


def _resolve_model_dir(model_id: str) -> Path:
    try:
        dataset_name, config_name = model_id.split("::", 1)
    except ValueError as exc:
        raise HTTPException(400, "model_id must be 'Dataset_X::trainer__plans__cfg'") from exc
    if not _SAFE_SEGMENT.match(dataset_name) or not _SAFE_SEGMENT.match(config_name):
        raise HTTPException(400, "bad model id")
    path = (RESULTS_ROOT / dataset_name / config_name).resolve()
    if not str(path).startswith(str(RESULTS_ROOT.resolve())):
        raise HTTPException(400, "bad model id")
    if not path.is_dir():
        raise HTTPException(404, f"no model {model_id}")
    return path


def _resolve_folds(spec: str | list[int], model_dir: Path) -> tuple[int, ...]:
    available = sorted(
        int(d.name.removeprefix("fold_"))
        for d in model_dir.iterdir()
        if d.is_dir() and re.fullmatch(r"fold_\d+", d.name)
        and (d / "checkpoint_best.pth").is_file()
    )
    if isinstance(spec, list):
        return tuple(f for f in spec if f in available)
    if spec == "best":
        return (available[0],) if available else ()
    if spec == "all":
        return tuple(available)
    raise HTTPException(400, f"bad use_folds: {spec}")


def _cached_prediction_id(
    plan_hash: str,
    model_id: str,
    folds: tuple[int, ...],
    case_id: str,
) -> str | None:
    """Return a cached prediction_id whose seg + status are still on disk.

    Returns None on a miss, or when the cache pointer is stale (seg cleaned
    up off disk) — in which case the stale pointer is dropped so the next
    lookup doesn't pay for it again. Shared by POST /api/predict and the
    cross-validation orchestrator so a CV fold sub-run that was already
    computed (single-fold or ensemble) is reused instantly.
    """
    cached_id = result_cache.get_inference(plan_hash, model_id, folds, case_id)
    if cached_id is None:
        return None
    cached_seg = PREDICTIONS_ROOT / cached_id / "seg.nii.gz"
    cached_status = PREDICTIONS_ROOT / cached_id / "status.json"
    if cached_seg.is_file() and cached_status.is_file():
        return cached_id
    result_cache.invalidate_inference(plan_hash, model_id, folds, case_id)
    return None


def _count_channels(case_dir: Path) -> int:
    return len(list(case_dir.glob("image_*.nii.gz")))


def _write_status(out_dir: Path, payload: dict) -> None:
    """Atomically write the status JSON for one prediction."""
    tmp = out_dir / "status.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(out_dir / "status.json")


def _update_status(out_dir: Path, **fields) -> None:
    """Merge `fields` into the existing status.json (writes updated_at)."""
    path = out_dir / "status.json"
    if path.is_file():
        current = json.loads(path.read_text())
    else:
        current = {}
    current.update(fields)
    current["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _write_status(out_dir, current)


def _load_label_map(model_dir: Path) -> dict[str, int]:
    # nnUNet writes plans.json / dataset.json / dataset_fingerprint.json
    # inside the config dir itself (alongside fold_N/), not at the dataset
    # dir level. Prefer that, fall back to the dataset-level path for older
    # layouts.
    ds_json = model_dir / "dataset.json"
    if not ds_json.is_file():
        ds_json = model_dir.parent / "dataset.json"
    if not ds_json.is_file():
        return {"background": 0}
    raw = json.loads(ds_json.read_text())
    labels = raw.get("labels", {})
    # nnUNet allows multi-int labels; flatten to first int.
    out: dict[str, int] = {}
    for name, v in labels.items():
        if isinstance(v, list):
            out[name] = int(v[0])
        else:
            out[name] = int(v)
    return out


def _model_channel_count(model_dir: Path) -> int:
    """Number of input channels the model expects, from dataset.json
    `channel_names`. Falls back to 1 when metadata is missing."""
    ds_json = model_dir / "dataset.json"
    if not ds_json.is_file():
        ds_json = model_dir.parent / "dataset.json"
    if not ds_json.is_file():
        return 1
    try:
        raw = json.loads(ds_json.read_text())
    except (OSError, json.JSONDecodeError):
        return 1
    ch = raw.get("channel_names") or {}
    return len(ch) if ch else 1


def _model_region(dataset_name: str) -> str | None:
    from modelfactory.qa.cohort import _region_for
    return _region_for(dataset_name, datasets_root=_datasets_root())


def _datasets_root() -> Path:
    """Same env-overridable datasets root the donate handler uses."""
    return Path(os.environ.get("QA_DATASETS_ROOT", FACTORY_ROOT / "datasets"))


def _prestage_preprocess_async(dataset_name: str) -> None:
    """Best-effort background pre-stage of the preprocessed `.npz` for a
    dataset's trained model(s).

    The first inference on a freshly-donated/uploaded case runs nnUNet's
    resample+normalize INLINE on the GPU click (the cold path) — single
    threaded, minutes on a large high-res CT (e.g. a 0.65 mm 768² H&N scan).
    Writing the npz ahead of time lets the next click take the warm path
    (load npz → forward pass only). Fire-and-forget; the cold path still
    works if this hasn't finished when the reviewer clicks Run.
    """
    async def _run() -> None:
        try:
            from modelfactory.qa.cohort import _discover_trained_models
            from modelfactory.qa.preprocess import preprocess_cohort_for_model

            datasets_root = _datasets_root()
            models = [
                m
                for m in _discover_trained_models(RESULTS_ROOT, datasets_root=datasets_root)
                if m["dataset_name"] == dataset_name
            ]
            for m in models:
                await asyncio.to_thread(
                    preprocess_cohort_for_model, Path(m["model_dir"]), COHORT_ROOT,
                )
                logger.info("pre-staged preprocessing for %s", m["model_id"])
        except Exception as exc:  # noqa: BLE001 — best-effort warm-up, never fatal
            logger.warning("pre-stage preprocessing failed for %s: %s", dataset_name, exc)

    asyncio.create_task(_run())


def _next_case_for_reviewer(
    *,
    model_id: str,
    reviewer: str,
    current_case_id: str,
) -> str | None:
    """Pick the next compatible case this reviewer hasn't yet verdicted.

    "Compatible" = listed in `cohort.manifest.json` under
    `compatible_models` for the model_id. We scan the manifest from disk
    rather than re-running `_discover_trained_models` so this is fast
    even with a large catalogue.

    Iteration is deterministic (sorted by case_id) so two clicks of
    Approve in a row produce a predictable ordering. The current case
    is skipped explicitly so a reviewer who hasn't verdicted any other
    case still advances off the one they just reviewed.

    Returns None when every compatible case has been verdicted by this
    reviewer, or when no compatible cases exist.
    """
    manifest_path = COHORT_ROOT / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        raw = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    compatible_case_ids = sorted(
        str(c["case_id"]) for c in raw.get("cases", [])
        if model_id in (c.get("compatible_models") or [])
    )
    if not compatible_case_ids:
        return None
    rev = (reviewer or "").strip().lower()
    verdicted_by_rev: set[str] = set()
    if rev:
        for v in _get_verdicts().list_for_model(model_id, limit=1000):
            if (v.reviewer or "").strip().lower() == rev:
                verdicted_by_rev.add(v.case_id)
    for cid in compatible_case_ids:
        if cid == current_case_id:
            continue
        if cid in verdicted_by_rev:
            continue
        return cid
    # Reviewer has worked through every compatible case — return the
    # next case after the current one anyway so they can do a second
    # pass if desired. Helpful for "approve-and-next" UX so the button
    # never goes dead.
    for cid in compatible_case_ids:
        if cid != current_case_id:
            return cid
    return None


def _last_mean_fg_dice(metrics_jsonl: Path) -> float | None:
    if not metrics_jsonl.is_file():
        return None
    try:
        last = None
        for line in metrics_jsonl.read_text().splitlines():
            if not line.strip():
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
        if last is None:
            return None
        return last.get("mean_fg_dice") or last.get("ema_fg_dice")
    except OSError:
        return None


def _read_validation_summary(
    config_dir: Path,
    fold: int,
    int_to_name: dict[int, str],
) -> tuple[float | None, dict[str, float] | None]:
    """Aggregate nnUNetv2's `fold_<n>/validation/summary.json` into
    (mean_fg_dice, {label_name: mean_dice across cases}).

    nnUNet 2.5 pre-aggregates both `foreground_mean` (scalar block) and
    `mean` (per-label block) at the top level — use those directly when
    present. Falls back to hand-aggregating `metric_per_case` for older
    schemas where it's a list of `{"metrics": {...}, ...}` entries or a
    dict keyed by case-id.
    """
    summary = config_dir / f"fold_{fold}" / "validation" / "summary.json"
    if not summary.is_file():
        return None, None
    try:
        raw = json.loads(summary.read_text())
    except (OSError, json.JSONDecodeError):
        return None, None
    if not isinstance(raw, dict):
        return None, None

    def _name_for(label_id: str | int) -> str:
        try:
            return int_to_name.get(int(label_id), str(label_id))
        except (TypeError, ValueError):
            return str(label_id)

    # Preferred path: nnUNet 2.5 ships the means it already computed.
    mean_block = raw.get("mean")
    fg_block = raw.get("foreground_mean")
    per_class: dict[str, float] = {}
    if isinstance(mean_block, dict):
        for label_id, metrics in mean_block.items():
            if not isinstance(metrics, dict):
                continue
            d = metrics.get("Dice")
            if not isinstance(d, (int, float)) or math.isnan(float(d)):
                continue
            per_class[_name_for(label_id)] = float(d)
    fg_mean: float | None = None
    if isinstance(fg_block, dict):
        v = fg_block.get("Dice")
        if isinstance(v, (int, float)) and not math.isnan(float(v)):
            fg_mean = float(v)
    if per_class or fg_mean is not None:
        if fg_mean is None and per_class:
            fg_mean = mean(per_class.values())
        return fg_mean, (per_class or None)

    # Fallback: aggregate from `metric_per_case` ourselves. Tolerate both
    # the modern list-of-entries shape and the older dict-by-case-id shape.
    mpc = raw.get("metric_per_case")
    metric_blocks: list[dict] = []
    if isinstance(mpc, list):
        for entry in mpc:
            if isinstance(entry, dict):
                m = entry.get("metrics")
                if isinstance(m, dict):
                    metric_blocks.append(m)
    elif isinstance(mpc, dict):
        for v in mpc.values():
            if isinstance(v, dict):
                metric_blocks.append(v)
    sums: dict[str, list[float]] = defaultdict(list)
    for metrics_block in metric_blocks:
        for label_id, metrics in metrics_block.items():
            if not isinstance(metrics, dict):
                continue
            d = metrics.get("Dice")
            if not isinstance(d, (int, float)) or math.isnan(float(d)):
                continue
            sums[str(label_id)].append(float(d))
    if not sums:
        return None, None
    per_class = {_name_for(k): mean(v) for k, v in sums.items()}
    flat = [d for vs in sums.values() for d in vs]
    return (mean(flat) if flat else None), per_class


def _last_epoch_from_metrics(metrics_jsonl: Path) -> int | None:
    """Read the most recent `epoch` field from a metrics.jsonl tail."""
    if not metrics_jsonl.is_file():
        return None
    try:
        last_epoch: int | None = None
        for line in metrics_jsonl.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            epoch = entry.get("epoch")
            if isinstance(epoch, (int, float)):
                last_epoch = int(epoch)
        return last_epoch
    except OSError:
        return None


def _latest_epoch_across_folds(
    config_dir: Path,
    available_folds: list[int],
) -> int | None:
    """Max epoch across all folds (including a potentially-mid-train one).

    For a 5-fold model halfway through fold 3, available_folds is [0,1,2]
    (the ones with checkpoint_best.pth) — but we also want to surface
    fold 3's current epoch. Probe up to max(available)+1 so the
    just-started fold is included.
    """
    if not available_folds:
        # No checkpointed fold yet — still try fold_0 in case training
        # has started writing metrics.jsonl but no checkpoint exists.
        probe = [0]
    else:
        probe = list(range(max(available_folds) + 2))
    best: int | None = None
    for fold in probe:
        candidate = config_dir / f"fold_{fold}" / "metrics.jsonl"
        v = _last_epoch_from_metrics(candidate)
        if v is None:
            continue
        if best is None or v > best:
            best = v
    return best


def _walk_fold_dirs(config_dir: Path) -> list[int]:
    """Return every fold_N directory on disk (sorted by N), regardless of
    whether the fold has produced a checkpoint yet. The discovery pass in
    `cohort._discover_trained_models` only finds folds with
    `checkpoint_best.pth`, which silently hides in-flight folds 1+2 when
    fold 0 finished first. This helper closes that gap so the catalog
    can show per-fold progress for actively-training folds.
    """
    if not config_dir.is_dir():
        return []
    out: list[int] = []
    for child in config_dir.iterdir():
        if not child.is_dir():
            continue
        m = re.match(r"fold_(\d+)$", child.name)
        if not m:
            continue
        out.append(int(m.group(1)))
    return sorted(out)


def _fold_status(
    fold_dir: Path,
    *,
    has_best: bool,
    latest_epoch: int | None,
    total_epochs: int | None,
    now: float,
) -> "ModelStatus":
    """Per-fold status. Same four states as the model-level status, but
    judged per fold instead of in aggregate so folds 1+2 mid-training
    don't get drowned out by a completed fold 0.
    """
    # Live training: any training_log_*.txt touched in the last hour.
    for log in fold_dir.glob("training_log_*.txt"):
        try:
            if now - log.stat().st_mtime < 3600:
                return "training"
        except OSError:
            continue
    # checkpoint_final.pth is written at on_train_end → the fold trained to
    # completion, even if checkpoint_best.pth was later deleted on transfer.
    if (fold_dir / "checkpoint_final.pth").is_file():
        return "done"
    if has_best:
        if (
            latest_epoch is not None
            and total_epochs
            and latest_epoch >= total_epochs * 0.9
        ):
            return "done"
        return "stopped"
    # No usable checkpoint and no recent log activity.
    # Was there *any* training history? (logs or partial checkpoints)
    has_history = any(fold_dir.glob("training_log_*.txt")) or any(
        fold_dir.glob("checkpoint_*.pth")
    )
    return "stopped" if has_history else "failed"


def _read_tail(path: Path, nbytes: int = 65536) -> str:
    """Read the last `nbytes` of a (possibly large) text file, dropping the
    first partial line. Training logs grow to ~MBs over 1000 epochs; we
    only need the recent tail to estimate the current epoch rate."""
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > nbytes:
                fh.seek(size - nbytes)
            raw = fh.read()
    except OSError:
        return ""
    text = raw.decode("utf-8", errors="replace")
    if size > nbytes:
        # First line is almost certainly truncated mid-record — drop it.
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1 :]
    return text


_EPOCH_TIME_RE = re.compile(r"Epoch time:\s*([0-9]+(?:\.[0-9]+)?)\s*s")


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _newest_training_log(fold_dir: Path) -> Path | None:
    logs = [p for p in fold_dir.glob("training_log_*.txt") if p.is_file()]
    if not logs:
        return None
    return max(logs, key=lambda p: p.stat().st_mtime if p.exists() else 0.0)


def _epoch_rate_from_log(fold_dir: Path, window: int = 20) -> float | None:
    """Median seconds-per-epoch over the last `window` epochs, parsed from
    the newest training_log's "Epoch time: X s" lines. These are the most
    reliable rate source — the metrics.jsonl `epoch_time_s` field has been
    observed to log negative values on some finished runs."""
    log = _newest_training_log(fold_dir)
    if log is None:
        return None
    times = [float(m) for m in _EPOCH_TIME_RE.findall(_read_tail(log))]
    # Drop implausible values (a corrupt/negative read, or a multi-hour
    # stall that would skew the median away from steady-state rate).
    times = [t for t in times if 0.0 < t < 3 * 3600]
    if not times:
        return None
    return _median(times[-window:])


def _epoch_rate_from_metrics(metrics_jsonl: Path, window: int = 20) -> float | None:
    """Fallback rate source: diff consecutive train-phase `ts` timestamps
    in metrics.jsonl. Used when no training_log is present."""
    if not metrics_jsonl.is_file():
        return None
    stamps: list[float] = []
    try:
        for line in _read_tail(metrics_jsonl).splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("phase") != "train":
                continue
            ts = rec.get("ts")
            if not isinstance(ts, str):
                continue
            try:
                stamps.append(
                    dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                )
            except ValueError:
                continue
    except OSError:
        return None
    deltas = [b - a for a, b in zip(stamps, stamps[1:]) if 0.0 < (b - a) < 3 * 3600]
    if not deltas:
        return None
    return _median(deltas[-window:])


def _metrics_run_start(metrics_jsonl: Path) -> str | None:
    """UTC ISO ts of the run_start record (when the fold began), if present.
    Cheap — it's the first line of metrics.jsonl."""
    if not metrics_jsonl.is_file():
        return None
    try:
        with metrics_jsonl.open() as fh:
            first = fh.readline().strip()
    except OSError:
        return None
    if not first:
        return None
    try:
        rec = json.loads(first)
    except json.JSONDecodeError:
        return None
    ts = rec.get("ts")
    if isinstance(ts, str):
        try:
            # Normalize to a UTC ISO string the frontend's Date() parses.
            return (
                dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
                .astimezone(dt.timezone.utc)
                .isoformat()
            )
        except ValueError:
            return None
    return None


def _fold_training_timing(
    fold_dir: Path,
    *,
    current_epoch: int | None,
    total_epochs: int | None,
    now_dt: dt.datetime,
) -> dict[str, object | None]:
    """Live training-rate + ETA for a single actively-training fold.

    All returned timestamps are UTC ISO8601 so the browser's Date() can
    compute a timezone-correct countdown. `est_finish` is absolute and is
    what the frontend ticks toward. It is anchored on the *last epoch
    boundary* (`last_epoch_at`), NOT on request-time `now`: epochs are
    minutes long, /api/models refetches every 30 s, and the client ticks a
    1 s countdown toward est_finish. Anchoring on `now` would re-project the
    finish ~30 s further out on every refetch that didn't advance the epoch,
    making the countdown visibly bounce. Anchoring on the epoch boundary
    keeps it stable between epochs so the countdown decreases monotonically.
    """
    empty: dict[str, object | None] = {
        "sec_per_epoch": None,
        "started_at": None,
        "last_epoch_at": None,
        "eta_seconds": None,
        "est_finish": None,
    }
    metrics = fold_dir / "metrics.jsonl"
    rate = _epoch_rate_from_log(fold_dir)
    if rate is None:
        rate = _epoch_rate_from_metrics(metrics)
    if rate is None:
        return empty

    # When the most-recent epoch landed — the newest log's mtime is a
    # reliable UTC anchor (fromtimestamp(..., UTC)); fall back to now.
    last_epoch_at = now_dt
    log = _newest_training_log(fold_dir)
    if log is not None:
        try:
            last_epoch_at = dt.datetime.fromtimestamp(log.stat().st_mtime, dt.timezone.utc)
        except OSError:
            pass

    # `current_epoch` is the 0-based index of the last COMPLETED epoch (the
    # trainer logs `self.current_epoch` at the end of each train/val pass,
    # before nnUNet increments it), so (current_epoch + 1) epochs are done
    # as of `last_epoch_at` and `total_epochs − (current_epoch + 1)` remain.
    completed_epochs = (current_epoch + 1) if current_epoch is not None else None

    # When the fold started: prefer the run_start record, else derive it
    # backwards from the last epoch boundary (stable across refetches —
    # deriving from `now` would drift the bar's left edge every poll).
    started_at = _metrics_run_start(metrics)
    if started_at is None and completed_epochs is not None:
        started_at = (
            last_epoch_at - dt.timedelta(seconds=rate * max(0, completed_epochs))
        ).isoformat()

    eta_seconds: float | None = None
    est_finish: str | None = None
    if completed_epochs is not None and total_epochs:
        remaining = max(0, total_epochs - completed_epochs)
        est_dt = last_epoch_at + dt.timedelta(seconds=remaining * rate)
        est_finish = est_dt.isoformat()
        # Snapshot remaining seconds relative to request time (floored — a
        # stalled-but-recent run can project a finish slightly in the past).
        eta_seconds = max(0.0, (est_dt - now_dt).total_seconds())

    return {
        "sec_per_epoch": round(rate, 2),
        "started_at": started_at,
        "last_epoch_at": last_epoch_at.isoformat(),
        "eta_seconds": eta_seconds,
        "est_finish": est_finish,
    }


def _build_fold_progress(
    config_dir: Path,
    total_epochs: int | None,
) -> list[FoldProgress]:
    """One FoldProgress per fold_N on disk. Walks every fold dir so
    in-flight folds (no checkpoint_best.pth yet) still surface."""
    now = time.time()
    now_dt = dt.datetime.now(dt.timezone.utc)
    out: list[FoldProgress] = []
    for fold in _walk_fold_dirs(config_dir):
        fold_dir = config_dir / f"fold_{fold}"
        has_best = (fold_dir / "checkpoint_best.pth").is_file()
        latest_epoch = _last_epoch_from_metrics(fold_dir / "metrics.jsonl")
        val_dice = _last_mean_fg_dice(fold_dir / "metrics.jsonl")
        status = _fold_status(
            fold_dir,
            has_best=has_best,
            latest_epoch=latest_epoch,
            total_epochs=total_epochs,
            now=now,
        )
        # Timing is only meaningful (and only worth the log parse) for a
        # fold that's actively producing epochs.
        timing: dict[str, object | None] = {}
        if status == "training":
            timing = _fold_training_timing(
                fold_dir,
                current_epoch=latest_epoch,
                total_epochs=total_epochs,
                now_dt=now_dt,
            )
        out.append(
            FoldProgress(
                fold=fold,
                status=status,
                current_epoch=latest_epoch,
                total_epochs=total_epochs,
                val_mean_fg_dice=val_dice,
                has_checkpoint_best=has_best,
                sec_per_epoch=timing.get("sec_per_epoch"),  # type: ignore[arg-type]
                started_at=timing.get("started_at"),  # type: ignore[arg-type]
                last_epoch_at=timing.get("last_epoch_at"),  # type: ignore[arg-type]
                eta_seconds=timing.get("eta_seconds"),  # type: ignore[arg-type]
                est_finish=timing.get("est_finish"),  # type: ignore[arg-type]
            )
        )
    return out


def _summary_from_folds(folds: list[FoldProgress]) -> tuple["ModelStatus", int | None]:
    """Pick the model-level status + current_epoch from the per-fold
    rollup. Priority: any training fold dominates (it's the live signal
    the reviewer cares about); next stopped/failed only if no fold is
    done; otherwise done. current_epoch reports the live-most fold so
    "epoch 19 / 1000" shows when fold 1 is actively training even
    though fold 0 finished at epoch 1000.
    """
    if not folds:
        return "failed", None
    training = [f for f in folds if f.status == "training"]
    if training:
        # Among multiple training folds (e.g. 1 + 2 in parallel), show
        # the lowest epoch — that's the worst-case "where am I" signal.
        epochs = [f.current_epoch for f in training if f.current_epoch is not None]
        return "training", (min(epochs) if epochs else None)
    done = [f for f in folds if f.status == "done"]
    stopped = [f for f in folds if f.status == "stopped"]
    failed = [f for f in folds if f.status == "failed"]
    if done and not stopped and not failed:
        epochs = [f.current_epoch for f in done if f.current_epoch is not None]
        return "done", (max(epochs) if epochs else None)
    if done or stopped:
        # Mix of done + stopped → reads as "partially complete" → stopped.
        epochs = [
            f.current_epoch for f in folds if f.current_epoch is not None
        ]
        return "stopped", (max(epochs) if epochs else None)
    return "failed", None


def _trainer_total_epochs(config_dir: Path) -> int | None:
    """Read the configured trainer epoch count.

    nnUNetv2 hard-codes 1000 in `nnUNetTrainer.num_epochs`. Some of our
    subclasses (e.g. `nnUNetTrainerMLflow`) inherit that default. The
    value is not exported into plans.json, but it does land in
    `progress.png` / the training log header. Cheapest reliable source:
    grep the most-recent training log for the `Epoch 0/N` banner.
    Falls back to the nnUNet default if no log is found.
    """
    pattern = re.compile(r"Epoch\s+\d+/(\d+)")
    for fold_dir in sorted(config_dir.glob("fold_*")):
        # Newest log first — training runs append a timestamped file.
        logs = sorted(
            fold_dir.glob("training_log_*.txt"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )
        for log in logs[:1]:
            try:
                head = log.read_text(errors="replace")[:65536]
            except OSError:
                continue
            m = pattern.search(head)
            if m:
                return int(m.group(1))
    return 1000  # nnUNetTrainer.num_epochs default


def _resolve_checkpoint_name(model_dir: Path, folds: tuple[int, ...]) -> str:
    """Pick the checkpoint to run inference from.

    Prefer `checkpoint_best.pth` when every requested fold has it; otherwise
    fall back to `checkpoint_final.pth` — which is all that remains after a
    model is transferred to a downstream product (best is deleted to save space).
    nnUNet writes `checkpoint_final.pth` at `on_train_end`, so it is a complete,
    inference-ready checkpoint.
    """
    if folds and all(
        (model_dir / f"fold_{f}" / "checkpoint_best.pth").is_file() for f in folds
    ):
        return "checkpoint_best.pth"
    return "checkpoint_final.pth"


def _run_prediction(*, model_dir: Path, folds: tuple[int, ...],
                    raw_images: list[Path], seg_path: Path,
                    npz: Path | None, pkl: Path | None) -> dict:
    from modelfactory.inference.predictor_cache import ModelKey
    from modelfactory.inference.run import run_inference

    key = ModelKey(model_dir=model_dir, folds=folds,
                   checkpoint_name=_resolve_checkpoint_name(model_dir, folds))
    predictor = _get_cache().get(key)
    result = run_inference(
        predictor=predictor,
        raw_image_paths=raw_images,
        output_seg_path=seg_path,
        preprocessed_npz=npz,
        preprocessed_pkl=pkl,
    )
    return {
        "elapsed_s": result.elapsed_s,
        "used_preprocessed_cache": result.used_preprocessed_cache,
    }


def _resolve_gt_path(
    region: str,
    case_id: str,
    case_dir: Path,
    revision: str,
) -> Path:
    """Resolve a `revision` parameter from /api/cases/.../groundtruth to a path.

    - `"active"` → currently-active revision row, falling back to the
      legacy `label_groundtruth.nii.gz` if none exists.
    - integer string → `label_corrected_v{n}.nii.gz` for that revision.
    """
    if revision == "active":
        return _resolve_active_gt(region, case_id, case_dir)
    if not revision.isdigit():
        raise HTTPException(400, "bad revision parameter")
    return case_dir / f"label_corrected_v{int(revision)}.nii.gz"


def _resolve_active_gt(region: str, case_id: str, case_dir: Path) -> Path:
    """Return the file path of the currently-active GT for a case."""
    full = f"{region}/{case_id}"
    active = _get_gt_store().get_active(region, full)
    if active is not None:
        # `path` is cohort-relative; anchor under COHORT_ROOT and verify
        # it still lives under the case dir (defence-in-depth for any
        # future row that was written before this resolver shipped).
        candidate = (COHORT_ROOT / active.path).resolve()
        case_root = case_dir.resolve()
        if str(candidate).startswith(str(case_root)) and candidate.is_file():
            return candidate
    return case_dir / "label_groundtruth.nii.gz"


def _is_uploaded_case(case_dir: Path) -> bool:
    """True when a case was uploaded via the QA viewer (per its source.json)."""
    src = case_dir / "source.json"
    if not src.is_file():
        return False
    try:
        info = json.loads(src.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return info.get("kind") == "upload" or info.get("source_dataset") == "uploaded"


def _patch_manifest_case(full_case_id: str, **fields) -> None:
    """Atomically patch one case's fields in manifest.json. No-op if absent."""
    manifest_path = COHORT_ROOT / "manifest.json"
    if not manifest_path.is_file():
        return
    try:
        data = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    changed = False
    for c in data.get("cases", []):
        if c.get("case_id") == full_case_id:
            c.update(fields)
            changed = True
            break
    if not changed:
        return
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(manifest_path)


def _case_info_from_manifest(full_case_id: str) -> CaseInfo | None:
    """Build a CaseInfo from the manifest entry for `full_case_id`."""
    manifest_path = COHORT_ROOT / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        data = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for c in data.get("cases", []):
        if c.get("case_id") == full_case_id:
            return CaseInfo(**c)
    return None


def _load_dataset_label_map(case_dir: Path) -> dict[str, int]:
    """Read label_map from the cohort case's parent dataset.json.

    The cohort writes `source.json` with the source dataset name; we
    look up the parent `datasets/Dataset###_*/dataset.json` and parse
    its labels. Falls back to a minimal `{"background": 0}` map when
    metadata is missing so the GT-edit toolbar can still render.
    """
    src = case_dir / "source.json"
    if not src.is_file():
        return {"background": 0}
    try:
        info = json.loads(src.read_text())
    except (OSError, json.JSONDecodeError):
        return {"background": 0}
    ds_name = info.get("source_dataset")
    if not ds_name:
        return {"background": 0}
    datasets_root = Path(
        os.environ.get("QA_DATASETS_ROOT", FACTORY_ROOT / "datasets")
    )
    ds_json = datasets_root / ds_name / "dataset.json"
    if not ds_json.is_file():
        return {"background": 0}
    try:
        raw = json.loads(ds_json.read_text())
    except (OSError, json.JSONDecodeError):
        return {"background": 0}
    labels = raw.get("labels", {})
    out: dict[str, int] = {}
    for name, v in labels.items():
        out[name] = int(v[0]) if isinstance(v, list) else int(v)
    return out


def _compute_model_status(config_dir: Path, available_folds: list[int]) -> "ModelStatus":
    """Heuristic over fold directories.

    Distinguishes four states the catalog UI cares about:

    - `training` — any fold has a `training_log_*.txt` touched in the
      last hour (live training in flight, with or without a checkpoint).
    - `stopped` — at least one fold has `checkpoint_best.pth` (so the
      model is usable for inference), but the run is not at completion
      AND not currently active. Covers: deliberately-paused multi-fold
      runs (3 of 4 folds done, 1 never finished); single-fold runs
      that hit early-stopping mid-schedule.
    - `failed` — no fold has `checkpoint_best.pth` AND no fold has
      recent log activity. Nothing here is usable; the run died before
      producing artifacts.
    - `done` — every fold dir found has `checkpoint_best.pth` AND the
      latest epoch is at or near `total_epochs`. The model is fully
      trained.

    `available_folds` is the list of folds that already have
    `checkpoint_best.pth` (per `_discover_trained_models`). We probe
    one slot beyond `max(available_folds)+1` to catch a just-started
    fold whose checkpoint hasn't landed yet.
    """
    now = time.time()
    has_active = False
    has_checkpointed = bool(available_folds)
    has_unfinished_fold = False

    if not available_folds:
        probe = [0]
    else:
        probe = list(range(max(available_folds) + 2))

    for fold in probe:
        fold_dir = config_dir / f"fold_{fold}"
        if not fold_dir.is_dir():
            continue
        # Either checkpoint is "usable": best (live/normal) or final (kept
        # after a production transfer deletes best).
        has_usable_ckpt = (
            (fold_dir / "checkpoint_best.pth").is_file()
            or (fold_dir / "checkpoint_final.pth").is_file()
        )
        recent = False
        has_logs = False
        for log in fold_dir.glob("training_log_*.txt"):
            has_logs = True
            try:
                if now - log.stat().st_mtime < 3600:
                    recent = True
                    break
            except OSError:
                continue
        if recent:
            has_active = True
        if not has_usable_ckpt and (has_logs or any(fold_dir.glob("checkpoint_*.pth"))):
            # Fold dir has some training history but no usable checkpoint.
            has_unfinished_fold = True

    if has_active:
        return "training"
    if has_checkpointed:
        # At least one fold is usable. Distinguish "stopped midway"
        # from "fully done" via epoch completion. We can't fully
        # judge done-vs-stopped here without epoch info; rely on
        # has_unfinished_fold and a coarse epoch check.
        latest_epoch = _latest_epoch_across_folds(config_dir, available_folds)
        total = _trainer_total_epochs(config_dir)
        if has_unfinished_fold:
            return "stopped"
        if latest_epoch is not None and total is not None and latest_epoch < total * 0.9:
            return "stopped"
        return "done"
    # No checkpoint anywhere — pure failure (unless we missed an active
    # window above, in which case we'd already have returned training).
    return "failed"


@functools.lru_cache(maxsize=64)
def _load_plans_cache(model_dir: str) -> dict:
    """Cached plans.json read. Keyed on str so the cache is hashable."""
    p = Path(model_dir) / "plans.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _postprocessing_info_for(
    model_dir: Path,
    configuration: str,
    pkl_path: Path | None,
) -> PostprocessingInfo | None:
    """Build a PostprocessingInfo for a single inference.

    Reads the model's plans.json (cached) and the case's preprocessor
    pickle (when available) to surface what was actually applied. The
    runtime flags come from the centralized PREDICTOR_FLAGS dict in
    modelfactory.inference.run — the import is lazy so this module
    stays torch-free.
    """
    try:
        from modelfactory.inference.run import PREDICTOR_FLAGS
    except ImportError:
        return None
    plans = _load_plans_cache(str(model_dir))
    configs = plans.get("configurations", {}) if plans else {}
    cfg = configs.get(configuration, {})
    network_spacing = [float(x) for x in cfg.get("spacing", [])]
    resampling = cfg.get("resampling_fn_seg_kwargs", {})
    resampling_order = int(resampling.get("order", 1)) if resampling else 1

    original_spacing: list[float] = []
    if pkl_path is not None and pkl_path.is_file():
        try:
            import pickle

            with pkl_path.open("rb") as f:
                props = pickle.load(f)
            spacing = props.get("spacing_after_transp")
            if spacing is None:
                spacing = props.get("spacing")
            if spacing is not None:
                original_spacing = [float(s) for s in spacing]
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError, KeyError):
            original_spacing = []

    pkl_postproc = model_dir / "postprocessing.pkl"
    has_pp = pkl_postproc.is_file()
    keep_largest: dict[str, bool] | None = None
    region_class_order: list[list[int]] | None = None
    if has_pp:
        try:
            import pickle

            with pkl_postproc.open("rb") as f:
                pp = pickle.load(f)
            if isinstance(pp, dict):
                rco = pp.get("region_class_order")
                if isinstance(rco, list):
                    region_class_order = [list(map(int, r)) for r in rco]
                klc = pp.get("keep_largest")
                if isinstance(klc, dict):
                    keep_largest = {str(k): bool(v) for k, v in klc.items()}
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError):
            keep_largest = None
            region_class_order = None

    return PostprocessingInfo(
        test_time_augmentation=PREDICTOR_FLAGS["use_mirroring"],
        gaussian_tile_blending=PREDICTOR_FLAGS["use_gaussian"],
        tile_step_size=PREDICTOR_FLAGS["tile_step_size"],
        perform_everything_on_device=PREDICTOR_FLAGS["perform_everything_on_device"],
        network_spacing=network_spacing,
        original_spacing=original_spacing,
        resampling_order_seg=resampling_order,
        pipeline=[
            "softmax",
            "resample_to_original_spacing",
            "crop_to_valid_region",
            "argmax",
        ],
        region_class_order=region_class_order,
        keep_largest_component=keep_largest,
        has_postprocessing_pkl=has_pp,
    )


async def _recompute_metrics_for_case(region: str, case_id: str) -> None:
    """Recompute metrics for every prediction whose case matches.

    Triggered after a GT edit/activate. Writes per-revision metrics
    files (`metrics_v{rev}.json`) so older revisions stay queryable.
    `status.json.active_gt_revision` is bumped so subsequent reads of
    `/api/predictions/{id}/status` report against the new GT.

    Serialized per-case so a flurry of saves on one case doesn't
    interleave; predictions on other cases proceed unimpeded.
    """
    key = f"{region}/{case_id}"
    lock = _recompute_locks.setdefault(key, asyncio.Lock())
    async with lock:
        case_dir = COHORT_ROOT / region / case_id
        if not case_dir.is_dir():
            return
        active_rev = _get_gt_store().get_active(region, key)
        rev_num = active_rev.revision if active_rev else 0
        gt_path = _resolve_active_gt(region, case_id, case_dir)
        if not gt_path.is_file():
            return

        if not PREDICTIONS_ROOT.is_dir():
            return
        predictions = []
        for child in PREDICTIONS_ROOT.iterdir():
            status_path = child / "status.json"
            if not status_path.is_file():
                continue
            try:
                raw = json.loads(status_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if raw.get("case_id") != key:
                continue
            if raw.get("status") not in {"seg_ready", "done"}:
                continue
            predictions.append((child, raw))

        total = len(predictions)
        _recompute_status[key] = {
            "pending": total,
            "completed": 0,
            "total": total,
            "error": None,
        }
        if total == 0:
            return

        completed = 0
        for pred_dir, status_raw in predictions:
            seg_path = pred_dir / "seg.nii.gz"
            if not seg_path.is_file():
                completed += 1
                _recompute_status[key].update(
                    pending=total - completed, completed=completed,
                )
                continue
            label_map = status_raw.get("label_map") or {}
            # Preserve each prediction's HD95 policy: cross-val Dice-only
            # children stored compute_hd95=False at seg_ready, so a GT edit
            # must not silently re-add the (expensive) HD95 pass to them.
            # Older predictions lack the key and default to True (unchanged).
            want_hd95 = bool(status_raw.get("compute_hd95", True))
            try:
                metrics_payload = await asyncio.to_thread(
                    _compute_metrics, seg_path, gt_path, label_map, want_hd95,
                )
                dump = [m.model_dump() for m in metrics_payload]
                (pred_dir / f"metrics_v{rev_num}.json").write_text(
                    json.dumps(dump, indent=2)
                )
                (pred_dir / "metrics.json").write_text(json.dumps(dump, indent=2))
                _update_status(
                    pred_dir,
                    metrics=dump,
                    metrics_error=None,
                    active_gt_revision=rev_num if rev_num > 0 else None,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "metric recompute failed for prediction=%s case=%s",
                    pred_dir.name, key,
                )
                _update_status(
                    pred_dir,
                    metrics_error=f"{type(exc).__name__}: {exc}",
                    active_gt_revision=rev_num if rev_num > 0 else None,
                )
            completed += 1
            _recompute_status[key].update(
                pending=total - completed, completed=completed,
            )


def _compute_metrics(
    seg_path: Path,
    gt_path: Path,
    label_map: dict[str, int],
    compute_hd95: bool = True,
) -> list[LabelMetricOut]:
    import nibabel as nib
    import numpy as np

    from modelfactory.inference.metrics import (
        dice_per_label,
        hd95_per_label,
    )

    pred_img = nib.load(str(seg_path))
    gt_img = nib.load(str(gt_path))
    pred = np.asarray(pred_img.dataobj).astype(np.int16)
    gt = np.asarray(gt_img.dataobj).astype(np.int16)

    # NIfTI spacing in (x, y, z); our metrics use (z, y, x).
    sx, sy, sz = pred_img.header.get_zooms()[:3]
    dice = dice_per_label(pred, gt, label_map)
    # HD95 is the expensive metric (~1 min/label brute-force surface NN).
    # Cross-validation fold sub-runs pass compute_hd95=False so the 5-fold
    # spread stays Dice-only; only the OOF fold + ensemble pay for HD95.
    hd95 = hd95_per_label(pred, gt, label_map, (sz, sy, sx)) if compute_hd95 else {}

    out: list[LabelMetricOut] = []
    for d in dice:
        out.append(LabelMetricOut(
            label=d.label,
            label_name=d.label_name,
            dice=None if (d.dice != d.dice) else float(d.dice),  # NaN -> None
            hd95_mm=hd95.get(d.label),
            n_voxels_gt=d.n_voxels_gt,
            n_voxels_pred=d.n_voxels_pred,
        ))
    return out


# ── cross-validation ───────────────────────────────────────────────────────
#
# A cross-validation run for one (model, case) produces N single-fold
# predictions + 1 ensemble, each a NORMAL prediction_id under PREDICTIONS_ROOT
# (so seg/mesh/metrics routes, the result cache, and the GT-recompute scan all
# work unchanged), plus an aggregate. The cv.json manifest under CROSSVAL_ROOT
# ties them together and records which fold is the unbiased out-of-fold (OOF)
# fold for the case (from splits_final.json). Lock discipline (review R3): the
# orchestrator holds only the `_crossval_inflight` guard, never gpu_lock —
# gpu_lock is acquired one fold at a time inside `_execute_one_prediction`, so
# other reviewers' single predictions interleave fairly between folds.


def _write_cv(cv_dir: Path, payload: dict) -> None:
    """Atomically write a cv.json manifest (stamps updated_at)."""
    payload["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    tmp = cv_dir / "cv.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(cv_dir / "cv.json")


def _cohort_case_stem(case_id: str) -> str | None:
    """The nnUNet case stem (== splits_final.json val entry) for a cohort case.

    Read from the cohort manifest's `source_case_stem` — NOT the cohort
    `case_id` ("region/case_NNN"), which is a separate id (review R1).
    """
    manifest_path = COHORT_ROOT / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        raw = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for c in raw.get("cases", []):
        if c.get("case_id") == case_id:
            return c.get("source_case_stem")
    return None


def _init_child_status(
    out_dir: Path, prediction_id: str, model_id: str,
    case_id: str, folds: tuple[int, ...],
) -> None:
    """Write the initial `queued` status.json for a CV child prediction.

    Mirrors what POST /api/predict writes so the child resolves through the
    existing /api/predictions/{id}/* routes. `_execute_one_prediction` advances
    it to running → seg_ready → done.
    """
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    _write_status(out_dir, {
        "prediction_id": prediction_id,
        "status": "queued",
        "model_id": model_id,
        "case_id": case_id,
        "folds": list(folds),
        "started_at": now,
        "updated_at": now,
    })


def _read_metrics_json(pred_dir: Path) -> list[dict] | None:
    path = pred_dir / "metrics.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _annotate_cv_staleness(cv: dict) -> None:
    """Mark a completed cv.json `stale` when the case's active GT has moved on.

    Read-time staleness (review R6): a GT edit re-scores the child predictions
    but the cv.json aggregate was computed against the GT revision recorded at
    run time, so flag the mismatch instead of writing back into the recompute
    loop.
    """
    try:
        region = cv.get("region") or cv.get("case_id", "").split("/", 1)[0]
        active = _get_gt_store().get_active(region, cv.get("case_id", ""))
        current_rev = active.revision if active else None
        cv["stale"] = (
            cv.get("status") == "done" and cv.get("gt_revision") != current_rev
        )
    except Exception:  # noqa: BLE001 — staleness is advisory, never fatal
        cv["stale"] = False


async def _run_crossval_background(
    *,
    cv_run_id: str,
    cv_dir: Path,
    model_id: str,
    case_id: str,
    model_dir: Path,
    case_dir: Path,
    plan_hash: str,
    reviewer: str,
    compute_hd95: str,
) -> None:
    """Run every fold individually + the ensemble for one case, then aggregate."""
    from modelfactory.qa import crossval

    inflight_key = f"{model_id}::{case_id}"
    region = case_id.split("/", 1)[0]
    dataset_name = model_dir.parent.name
    try:
        available = list(_resolve_folds("all", model_dir))
        # Keep all single-fold + ensemble predictors resident for this run
        # (review: avoids mid-CV LRU thrash; dynamic so a 10-fold model widens
        # the cache instead of re-thrashing).
        cache = _get_cache()
        cache.max_size = max(cache.max_size, len(available) + 1)

        source_case_stem = _cohort_case_stem(case_id)
        splits_path = PREPROCESSED_ROOT / dataset_name / "splits_final.json"
        if source_case_stem:
            oof = crossval.oof_fold_for_case(splits_path, source_case_stem, available)
        else:
            oof = crossval.OofResolution(None, False, "external")

        active_rev = _get_gt_store().get_active(region, case_id)
        gt_revision = active_rev.revision if active_rev else None
        label_map = _load_label_map(model_dir)

        # Build the entry list: one per single fold, then the ensemble row.
        entries: list[dict] = [
            {
                "kind": "fold", "fold": k, "is_oof": (k == oof.oof_fold),
                "state": "queued", "prediction_id": None, "mean_fg_dice": None,
                "metrics": None, "elapsed_s": None, "error": None,
            }
            for k in available
        ]
        entries.append({
            "kind": "ensemble", "fold": None, "is_oof": False,
            "state": "queued", "prediction_id": None, "mean_fg_dice": None,
            "metrics": None, "elapsed_s": None, "error": None,
        })

        now = dt.datetime.now(dt.timezone.utc).isoformat()
        cv = {
            "cv_run_id": cv_run_id, "model_id": model_id, "case_id": case_id,
            "source_case_stem": source_case_stem, "region": region,
            "dataset_name": dataset_name, "reviewer": reviewer,
            "status": "running", "compute_hd95": compute_hd95,
            "oof_fold": oof.oof_fold, "oof_resolvable": oof.resolvable,
            "oof_reason": oof.reason, "available_folds": available,
            "folds_total": len(entries), "folds_done": 0, "current_fold": None,
            "gt_revision": gt_revision, "label_map": label_map,
            "started_at": now, "entries": entries, "aggregate": None,
            "error": None,
        }
        _write_cv(cv_dir, cv)

        def _want_hd95(is_oof: bool, is_ensemble: bool) -> bool:
            if compute_hd95 == "all":
                return True
            if compute_hd95 == "oof_and_ensemble":
                return is_oof or is_ensemble
            return False  # "none"

        for idx, entry in enumerate(entries):
            is_ensemble = entry["kind"] == "ensemble"
            is_oof = bool(entry["is_oof"])
            folds_tuple = tuple(available) if is_ensemble else (entry["fold"],)
            cv["current_fold"] = "ensemble" if is_ensemble else entry["fold"]
            entry["state"] = "running"
            _write_cv(cv_dir, cv)

            cached_id = _cached_prediction_id(plan_hash, model_id, folds_tuple, case_id)
            if cached_id is not None:
                metrics = _read_metrics_json(PREDICTIONS_ROOT / cached_id)
                entry.update(
                    prediction_id=cached_id, state="done", metrics=metrics,
                    mean_fg_dice=_mean_fg_dice(metrics), elapsed_s=None, error=None,
                )
            else:
                child_id = uuid.uuid4().hex[:12]
                child_dir = PREDICTIONS_ROOT / child_id
                child_dir.mkdir(parents=True, exist_ok=True)
                _init_child_status(child_dir, child_id, model_id, case_id, folds_tuple)
                await predict_queue.submit(
                    prediction_id=child_id, model_id=model_id,
                    case_id=case_id, reviewer=f"CV {reviewer}".strip(),
                )
                res = await _execute_one_prediction(
                    prediction_id=child_id, out_dir=child_dir,
                    model_id=model_id, case_id=case_id, model_dir=model_dir,
                    folds=folds_tuple, case_dir=case_dir, plan_hash=plan_hash,
                    compute_hd95=_want_hd95(is_oof, is_ensemble),
                    do_meshes=(is_oof or is_ensemble),
                )
                entry.update(
                    prediction_id=res["prediction_id"], state=res["status"],
                    metrics=res["metrics"], mean_fg_dice=res["mean_fg_dice"],
                    elapsed_s=res["elapsed_s"], error=res["error"],
                )
            cv["folds_done"] = idx + 1
            _write_cv(cv_dir, cv)

        cv["aggregate"] = crossval.aggregate_cv(entries, oof.oof_fold)
        cv["status"] = "done"
        cv["current_fold"] = None
        _write_cv(cv_dir, cv)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "crossval run failed cv_run_id=%s model=%s case=%s",
            cv_run_id, model_id, case_id,
        )
        try:
            path = cv_dir / "cv.json"
            cv = json.loads(path.read_text()) if path.is_file() else {
                "cv_run_id": cv_run_id, "model_id": model_id, "case_id": case_id,
            }
        except (OSError, json.JSONDecodeError):
            cv = {"cv_run_id": cv_run_id, "model_id": model_id, "case_id": case_id}
        cv["status"] = "error"
        cv["error"] = f"{type(exc).__name__}: {exc}"
        cv["current_fold"] = None
        _write_cv(cv_dir, cv)
    finally:
        _crossval_inflight.pop(inflight_key, None)


@app.post("/api/crossval", status_code=202)
async def start_crossval(req: CrossvalRequest):
    """Kick off a cross-validation run for one (model, case); return a cv_run_id.

    Runs each available fold individually + the ensemble (sequentially, through
    the shared GPU queue), computing per-fold metrics and flagging the unbiased
    out-of-fold fold. The client polls /api/crossval/{cv_run_id}/status.
    """
    case_dir = _resolve_case_dir(*req.case_id.split("/", 1))
    model_dir = _resolve_model_dir(req.model_id)
    available = _resolve_folds("all", model_dir)
    if not available:
        raise HTTPException(400, f"no folds available for {req.model_id}")
    if len(available) < 2:
        raise HTTPException(
            400, "cross-validation needs at least 2 trained folds; "
                 f"{req.model_id} has {len(available)}",
        )

    region = _model_region(model_dir.parent.name)
    case_region = req.case_id.split("/", 1)[0]
    if region is not None and region != case_region:
        raise HTTPException(
            409, f"region mismatch: model is {region}, case is {case_region}",
        )

    inflight_key = f"{req.model_id}::{req.case_id}"
    if inflight_key in _crossval_inflight:
        raise HTTPException(
            409, "a cross-validation run for this model + case is already in flight",
        )

    CROSSVAL_ROOT.mkdir(parents=True, exist_ok=True)
    _ensure_predictions_root()
    cv_run_id = uuid.uuid4().hex[:12]
    cv_dir = CROSSVAL_ROOT / cv_run_id
    cv_dir.mkdir(parents=True, exist_ok=True)
    plan_hash = result_cache.plan_hash_for_model(model_dir)
    _crossval_inflight[inflight_key] = cv_run_id

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    _write_cv(cv_dir, {
        "cv_run_id": cv_run_id, "model_id": req.model_id, "case_id": req.case_id,
        "status": "queued", "compute_hd95": req.compute_hd95,
        "available_folds": list(available), "folds_total": len(available) + 1,
        "folds_done": 0, "current_fold": None, "started_at": now,
        "entries": [], "aggregate": None, "error": None,
    })

    asyncio.create_task(_run_crossval_background(
        cv_run_id=cv_run_id, cv_dir=cv_dir, model_id=req.model_id,
        case_id=req.case_id, model_dir=model_dir, case_dir=case_dir,
        plan_hash=plan_hash, reviewer=req.reviewer or "",
        compute_hd95=req.compute_hd95,
    ))
    return {
        "cv_run_id": cv_run_id,
        "status": "queued",
        "status_url": f"/api/crossval/{cv_run_id}/status",
    }


def _load_cv(cv_run_id: str) -> dict:
    if not _SAFE_SEGMENT.match(cv_run_id):
        raise HTTPException(400, "bad cross-validation id")
    path = CROSSVAL_ROOT / cv_run_id / "cv.json"
    if not path.is_file():
        raise HTTPException(404, f"no cross-validation run {cv_run_id}")
    return json.loads(path.read_text())


def _load_cv_runs_for_model(model_id: str) -> list[dict]:
    """Scan CROSSVAL_ROOT for cv.json manifests belonging to `model_id`."""
    out: list[dict] = []
    if not CROSSVAL_ROOT.is_dir():
        return out
    for d in CROSSVAL_ROOT.iterdir():
        f = d / "cv.json"
        if not f.is_file():
            continue
        try:
            cv = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if cv.get("model_id") == model_id:
            out.append(cv)
    return out


def _compatible_case_ids_for_model(model_id: str) -> list[str]:
    """Cohort case_ids whose compatible_models include this model."""
    manifest_path = COHORT_ROOT / "manifest.json"
    if not manifest_path.is_file():
        return []
    try:
        raw = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return sorted(
        str(c["case_id"]) for c in raw.get("cases", [])
        if model_id in (c.get("compatible_models") or [])
    )


def _build_rollup(model_id: str) -> dict:
    """Read-only model-level cross-validation rollup. Never triggers runs."""
    from modelfactory.qa import crossval

    model_dir = _resolve_model_dir(model_id)  # 404 if the model is unknown
    dataset_name = model_dir.parent.name
    runs = _load_cv_runs_for_model(model_id)
    compatible = _compatible_case_ids_for_model(model_id)
    return crossval.build_model_report(model_id, dataset_name, runs, compatible)


def _csv_filename(stem: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    return f"{safe}.csv"


# Rollup routes are declared BEFORE the /{cv_run_id} routes so the literal
# "rollup" / "rollup.html" / "rollup.csv" segments win over the path param.
@app.get("/api/crossval/rollup")
def get_crossval_rollup(model_id: str):
    return JSONResponse(_build_rollup(model_id))


@app.get("/api/crossval/rollup.html")
def get_crossval_rollup_html(model_id: str):
    from modelfactory.qa import report
    return HTMLResponse(report.render_rollup_html(_build_rollup(model_id)))


@app.get("/api/crossval/rollup.csv")
def get_crossval_rollup_csv(model_id: str):
    from modelfactory.qa import report
    body = report.render_rollup_csv(_build_rollup(model_id))
    return Response(
        body, media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{_csv_filename("crossval_rollup_" + model_id)}"'},
    )


@app.get("/api/crossval/{cv_run_id}/status")
def get_crossval_status(cv_run_id: str):
    """Lightweight progress poll. Carries the full run payload once `done`."""
    cv = _load_cv(cv_run_id)
    _annotate_cv_staleness(cv)
    entries_lite = [
        {
            "kind": e.get("kind"), "fold": e.get("fold"),
            "is_oof": e.get("is_oof"), "state": e.get("state"),
            "mean_fg_dice": e.get("mean_fg_dice"),
            "prediction_id": e.get("prediction_id"),
        }
        for e in cv.get("entries", [])
    ]
    return JSONResponse({
        "cv_run_id": cv["cv_run_id"],
        "status": cv["status"],
        "model_id": cv["model_id"],
        "case_id": cv["case_id"],
        "folds_total": cv.get("folds_total"),
        "folds_done": cv.get("folds_done", 0),
        "current_fold": cv.get("current_fold"),
        "entries": entries_lite,
        "error": cv.get("error"),
        "run": cv if cv.get("status") == "done" else None,
    })


@app.get("/api/crossval/{cv_run_id}")
def get_crossval(cv_run_id: str):
    """Full cross-validation run manifest (folds + ensemble + aggregate)."""
    cv = _load_cv(cv_run_id)
    _annotate_cv_staleness(cv)
    return JSONResponse(cv)


@app.get("/api/crossval/{cv_run_id}/report.html")
def get_crossval_report_html(cv_run_id: str):
    """Self-contained per-case cross-validation report (prints to PDF)."""
    from modelfactory.qa import report
    cv = _load_cv(cv_run_id)
    _annotate_cv_staleness(cv)
    return HTMLResponse(report.render_case_html(cv))


@app.get("/api/crossval/{cv_run_id}/report.csv")
def get_crossval_report_csv(cv_run_id: str):
    """Per-label × per-fold long-format CSV for one cross-validation run."""
    from modelfactory.qa import report
    cv = _load_cv(cv_run_id)
    body = report.render_case_csv(cv)
    return Response(
        body, media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="crossval_{cv_run_id}.csv"'},
    )


# ── static frontend ──────────────────────────────────────────────────────
#
# Mounted LAST so the /api/* routes above take precedence. `html=True` makes
# StaticFiles fall through to index.html for unknown paths — the SPA router
# (or the lack of one in our case) renders the page client-side either way.
# When the web bundle has not been built (local dev), the mount is skipped
# so the importable module stays usable.

if WEB_STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_STATIC_DIR), html=True), name="web")
else:
    logger.info("web static dir %s not present — serving API only", WEB_STATIC_DIR)
